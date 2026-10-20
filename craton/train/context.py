"""The context builder: which tokens go into a window, and how each one is written.

A window is cut around an **anchor**: a moment, and a place. Its budget is the config's
``context_tokens`` (1,024 in mini), filled by one documented policy (``ContextPolicy``):

    tides    the hourly solid-Earth tide at the region's centre, the ``tides`` hours nearest
             the anchor. A next-day window takes the 12 hours before midnight and the
             ``tides_ahead`` hours after it: the forcing of the day being forecast is known in
             advance, from the Sun and the Moon.
    GNSS     the ``gnss_stations`` stations nearest the anchor within ``gnss_km`` (stations
             the window asks about come first), each with its days within ``gnss_days`` of
             the anchor day, at most ``gnss`` days. A next-day window takes the
             ``next_day_gnss_days`` days before midnight.
    quakes   everything left, nearest the anchor in time (**recency**). A next-day window
             takes the latest quakes before midnight and keeps the ``big_quakes`` largest of
             the past year at or above ``big_magnitude`` even when they are older
             (**magnitude**: big quakes trigger for longer).

Stations are picked by **distance**. A model only gets the instruments its config reads; the
budget a missing instrument would have taken goes to the others.

How a GNSS day is written. A station's position is millions of millimetres from an arbitrary
origin, and it moves a few centimetres a year, so the raw number means nothing to a network.
Every station gets a **line**: its long-term velocity and its position in 2015, fitted on the
training split. A day enters as its distance from the line. Then each station's visible days
in the window are centred on their own mean (the **level**, given as its own channel): the
same trick as reversible instance normalisation in time-series transformers (Kim et al. 2022),
and what keeps a station that jumped (an earthquake, a new antenna) readable. A hidden day is
answered as a distance from line + level, in millimetres, on CB-01's 1 mm bins.

A station nobody has seen (CB-04's held-out stations) has no position and no velocity of its
own. It borrows a line from its neighbours (their velocities, weighted by 1/distance², as the
IDW baseline does) and gets no level; the GNSS head's ramp is the model's own correction to
the borrowed velocity. Training practises this on visible stations, hidden whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from craton.bench.data import DAY, Events, Gnss, Tides
from craton.bench.splits import MAX_JUMP_M
from craton.config import ContextPolicy
from craton.model.craton import Tokens
from craton.model.encoders import (
    PAD,
    QUERY,
    VALUE_WIDTH,
    instrument_id,
    magnitude_type_id,
)
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

YEAR = 365.25 * DAY
#: Where every station's line is pinned: its position on 1 January 2015.
LINE_EPOCH = to_seconds("2015-01-01T00:00:00Z")
#: A station fits its own line only with this many days; otherwise it borrows one.
MIN_LINE_DAYS = 365
#: Borrowed lines: the nearest stations with a line of their own, weighted by 1/d².
BORROW_NEIGHBOURS = 6
BORROW_KM = 100.0
#: A GNSS day's bins: CB-01 scores east, north and up on 1 mm bins.
GNSS_BIN_M = 0.001
#: Where the tide sample a window reads sits, for every window: the region's centre.
TIDE_SCALE_M = 0.1

QUAKE = instrument_id("seismic_catalog")
GNSS = instrument_id("gnss_daily")
TIDE = instrument_id("solid_earth_tide")


def _km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def slog(x: Any) -> Any:
    """Signed log: keeps a 2 mm and a 2 km offset both on a scale a network can read."""
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------- the quakes


@dataclass(frozen=True)
class Catalog:
    """A catalogue as arrays the window builders index into, sorted by time."""

    time: FloatArray
    values: FloatArray  # (n, 4) lat, lon, depth, magnitude
    mag_type: IntArray

    @staticmethod
    def of(ev: Events) -> Catalog:
        values = np.stack([ev.lat, ev.lon, ev.depth_km, ev.mag], axis=1)
        types = np.array([magnitude_type_id(str(t)) for t in ev.mag_type], dtype=np.int64)
        return Catalog(ev.time.astype(np.float64), values.astype(np.float64), types)

    def __len__(self) -> int:
        return len(self.time)

    def select(self, rows: IntArray | BoolArray) -> Catalog:
        return Catalog(self.time[rows], self.values[rows], self.mag_type[rows])


# ---------------------------------------------------------------- station lines


@dataclass(frozen=True)
class Lines:
    """Every station's long-term line: position on ``LINE_EPOCH`` (m, NaN where nothing
    anchors it) and velocity (m/yr), plus the velocity it would borrow from its neighbours."""

    codes: tuple[str, ...]
    site: FloatArray  # (S, 2) median lat, lon
    offset: FloatArray  # (S, 3)
    velocity: FloatArray  # (S, 3) its own where it has one, else borrowed
    own: BoolArray  # (S,) the velocity is its own
    borrowed: FloatArray  # (S, 3) the neighbours' velocity, never its own

    def index(self, code: str) -> int:
        return self.codes.index(code)


def fit_lines(train: Gnss, sites: dict[str, tuple[float, float]]) -> Lines:
    """Lines for every station in ``sites``, fitted on the ``train`` days only.

    A station with ``MIN_LINE_DAYS`` training days and no jump over ``MAX_JUMP_M`` fits its
    own velocity by least squares. Every station also gets a borrowed velocity from its
    nearest such neighbours (never itself). Its offset is the mean of its training days around
    whichever velocity it uses; a station with no training day has none.
    """
    codes = tuple(sorted(sites))
    site = np.array([sites[c] for c in codes], dtype=np.float64).reshape(-1, 2)
    n = len(codes)
    offset = np.full((n, 3), np.nan)
    own_v = np.full((n, 3), np.nan)
    rows_of: dict[str, IntArray] = {}
    order = np.argsort(train.station, kind="stable")
    st = train.station[order]
    for c in codes:
        lo, hi = np.searchsorted(st, c, "left"), np.searchsorted(st, c, "right")
        rows_of[c] = order[lo:hi]
    for i, c in enumerate(codes):
        rows = rows_of[c]
        if len(rows) < MIN_LINE_DAYS:
            continue
        enu = train.enu[rows]
        t = train.time[rows]
        srt = np.argsort(t)
        if float(np.abs(np.diff(enu[srt], axis=0)).max()) > MAX_JUMP_M:
            continue
        dt = (t - LINE_EPOCH) / YEAR
        design = np.stack([np.ones_like(dt), dt], axis=1)
        coef, *_ = np.linalg.lstsq(design, enu, rcond=None)
        own_v[i] = coef[1]
    have = ~np.isnan(own_v[:, 0])
    borrowed = np.zeros((n, 3))
    fallback = np.median(own_v[have], axis=0) if have.any() else np.zeros(3)
    for i in range(n):
        d = _km(site[i, 0], site[i, 1], site[:, 0], site[:, 1])
        cand = np.nonzero(have & (d <= BORROW_KM) & (np.arange(n) != i))[0]
        cand = cand[np.argsort(d[cand])][:BORROW_NEIGHBOURS]
        if not len(cand):
            borrowed[i] = fallback
            continue
        w = 1.0 / np.maximum(d[cand], 1.0) ** 2
        borrowed[i] = (w[:, None] * own_v[cand]).sum(0) / w.sum()
    velocity = np.where(have[:, None], own_v, borrowed)
    for i, c in enumerate(codes):
        rows = rows_of[c]
        if len(rows):
            dt = (train.time[rows] - LINE_EPOCH) / YEAR
            offset[i] = (train.enu[rows] - np.outer(dt, velocity[i])).mean(0)
    return Lines(codes, site, offset, velocity, have, borrowed)


@dataclass(frozen=True)
class GnssDays:
    """GNSS days a window can read, sorted by time, stations as indices into ``lines``."""

    time: FloatArray
    station: IntArray
    enu: FloatArray  # (n, 3) metres
    lines: Lines

    @staticmethod
    def of(g: Gnss, lines: Lines) -> GnssDays:
        known = np.isin(g.station, lines.codes)
        g = g.select(known)
        idx = {c: i for i, c in enumerate(lines.codes)}
        station = np.array([idx[s] for s in g.station.tolist()], dtype=np.int64)
        order = np.argsort(g.time, kind="stable")
        return GnssDays(g.time[order], station[order], g.enu[order], lines)

    def __len__(self) -> int:
        return len(self.time)

    def between(self, t0: float, t1: float) -> IntArray:
        lo, hi = np.searchsorted(self.time, [t0, t1], "left")
        return np.arange(lo, hi, dtype=np.int64)


# ---------------------------------------------------------------- sources and windows


@dataclass(frozen=True)
class Sources:
    """Everything windows are cut from, limited to what the model reads."""

    cat: Catalog
    gnss: GnssDays | None
    tides: Tides | None
    policy: ContextPolicy
    length: int
    #: False when the model doesn't read quakes: windows then carry none (an ablation).
    quakes: bool = True

    @staticmethod
    def build(
        inputs: tuple[str, ...],
        policy: ContextPolicy,
        length: int,
        events: Events,
        gnss: GnssDays | None = None,
        tides: Tides | None = None,
    ) -> Sources:
        return Sources(
            Catalog.of(events),
            gnss if "gnss_daily" in inputs else None,
            tides if "solid_earth_tide" in inputs else None,
            policy,
            length,
            "seismic_catalog" in inputs,
        )


@dataclass
class Window:
    """One window's choice of tokens, before they are written out."""

    t_ref: float
    quakes: IntArray = field(default_factory=lambda: np.zeros(0, np.int64))
    quake_hidden: BoolArray = field(default_factory=lambda: np.zeros(0, bool))
    gnss: IntArray = field(default_factory=lambda: np.zeros(0, np.int64))
    gnss_hidden: BoolArray = field(default_factory=lambda: np.zeros(0, bool))
    #: Stations treated as never seen: borrowed line, no level, every day hidden.
    unseen: frozenset[int] = frozenset()
    tides: IntArray = field(default_factory=lambda: np.zeros(0, np.int64))
    query_at: float | None = None
    #: Quakes to write instead of ``src.cat`` rows (CB-17 reads a region's own catalogue).
    cat: Catalog | None = None

    @property
    def size(self) -> int:
        return len(self.quakes) + len(self.gnss) + len(self.tides) + (self.query_at is not None)


def tide_rows(src: Sources, t0: float, t1: float, limit: int) -> IntArray:
    """Hourly tide samples in [t0, t1), at most ``limit``, nearest the middle first."""
    if src.tides is None or limit <= 0:
        return np.zeros(0, np.int64)
    lo, hi = np.searchsorted(src.tides.time, [t0, t1], "left")
    rows = np.arange(lo, hi, dtype=np.int64)
    if len(rows) > limit:
        mid = 0.5 * (t0 + t1)
        rows = np.sort(rows[np.argsort(np.abs(src.tides.time[rows] - mid))[:limit]])
    return rows


def gnss_rows(
    src: Sources,
    lat: float,
    lon: float,
    t0: float,
    t1: float,
    first: tuple[int, ...] = (),
    budget: int | None = None,
    exclude: BoolArray | None = None,
) -> IntArray:
    """GNSS days in [t0, t1) at the stations nearest (lat, lon); ``first`` stations lead.

    ``exclude`` (per GNSS day) removes days the window may not read at all.
    """
    p = src.policy
    budget = p.gnss if budget is None else budget
    if src.gnss is None or budget <= 0:
        return np.zeros(0, np.int64)
    rows = src.gnss.between(t0, t1)
    if exclude is not None:
        rows = rows[~exclude[rows]]
    if not len(rows):
        return rows
    stations = np.unique(src.gnss.station[rows])
    site = src.gnss.lines.site[stations]
    dist = _km(lat, lon, site[:, 0], site[:, 1])
    lead = np.isin(stations, first)
    keep = lead | (dist <= p.gnss_km)
    order = np.lexsort((dist, ~lead))
    chosen = [s for s in stations[order] if keep[stations == s][0]][: p.gnss_stations]
    out: list[IntArray] = []
    left = budget
    for s in chosen:
        mine = rows[src.gnss.station[rows] == s]
        if len(mine) > left:
            mid = 0.5 * (t0 + t1)
            mine = np.sort(mine[np.argsort(np.abs(src.gnss.time[mine] - mid))[:left]])
        out.append(mine)
        left -= len(mine)
        if left <= 0:
            break
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def quakes_around(cat: Catalog, t: float, n: int) -> IntArray:
    """The ``n`` consecutive quakes centred on moment ``t`` (clamped to the catalogue)."""
    n = min(n, len(cat))
    centre = int(np.searchsorted(cat.time, t))
    start = int(np.clip(centre - n // 2, 0, len(cat) - n))
    return np.arange(start, start + n, dtype=np.int64)


def where_now(cat: Catalog, rows: IntArray, hidden: BoolArray | None = None) -> tuple[float, float]:
    """Median place of the visible quakes among ``rows``: a window's anchor place."""
    if hidden is not None:
        rows = rows[~hidden]
    if not len(rows):
        return float(np.median(cat.values[:, 0])), float(np.median(cat.values[:, 1]))
    return float(np.median(cat.values[rows, 0])), float(np.median(cat.values[rows, 1]))


# ---------------------------------------------------------------- writing a batch


@dataclass
class Batch:
    """Windows written as tensors, with the answers to everything hidden in them.

    Hidden quakes and hidden GNSS days are listed window by window, in token order: the same
    order as ``tokens.of(...) & tokens.hidden[..., 0]`` picks them out of the trunk's output.
    """

    tokens: Tokens
    quake_pos: torch.Tensor  # (Nq,) where each hidden quake sits: window * length + token
    gnss_pos: torch.Tensor  # (Ng,) the same for each hidden GNSS day
    #: Each hidden token's share of its window: 1 / (hidden tokens of that instrument in the
    #: window). A window that hides 950 quakes then counts as one example, like one that
    #: hides 140, instead of drowning the rest of the batch.
    quake_weight: torch.Tensor  # (Nq,)
    gnss_weight: torch.Tensor  # (Ng,)
    quake_targets: torch.Tensor  # (Nq, 4) lat, lon, depth, magnitude
    gnss_lo: torch.Tensor  # (Ng, 3) the true 1 mm bin, mm from the baseline
    gnss_hi: torch.Tensor
    gnss_years: torch.Tensor  # (Ng,) years since 2015
    gnss_base: FloatArray  # (Ng, 3) the baseline itself, metres (for answers)
    gnss_rows: IntArray  # (Ng,) the GNSS day each answer is for
    quake_rows: IntArray  # (Nq,) the catalogue row each answer is for

    def pin_memory(self) -> Batch:
        """Page-locked copies, so the copy to the GPU runs alongside the GPU's work."""
        out = Batch(**self.__dict__)
        out.tokens = self.tokens.pin_memory()
        for name in (
            "quake_pos",
            "gnss_pos",
            "quake_weight",
            "gnss_weight",
            "quake_targets",
            "gnss_lo",
            "gnss_hi",
            "gnss_years",
        ):
            setattr(out, name, getattr(self, name).pin_memory())
        return out


def _gnss_bins(enu: FloatArray) -> tuple[FloatArray, FloatArray]:
    """CB-01's 1 mm bins, computed exactly as the benchmark does (``Bin.edges_of``)."""
    k = np.floor(enu / GNSS_BIN_M + 1e-9)
    lo = k * GNSS_BIN_M
    return lo, lo + GNSS_BIN_M


def write(src: Sources, windows: list[Window]) -> Batch:
    """Turn windows into one padded batch of tokens, and collect what they hide."""
    length = max(src.length, max(w.size for w in windows))
    b = len(windows)
    inst = np.full((b, length), PAD, dtype=np.int64)
    values = np.zeros((b, length, VALUE_WIDTH), dtype=np.float32)
    hidden = np.zeros((b, length, VALUE_WIDTH), dtype=bool)
    label = np.zeros((b, length), dtype=np.int64)
    coords = np.zeros((b, length, 4), dtype=np.float64)
    known = np.zeros((b, length, 4), dtype=bool)
    clock = np.zeros((b, length, 2), dtype=np.float64)
    q_targets: list[FloatArray] = []
    q_rows: list[IntArray] = []
    q_pos: list[IntArray] = []
    g_pos: list[IntArray] = []
    q_w: list[FloatArray] = []
    g_w: list[FloatArray] = []
    g_lo: list[FloatArray] = []
    g_hi: list[FloatArray] = []
    g_years: list[FloatArray] = []
    g_base: list[FloatArray] = []
    g_rows: list[IntArray] = []

    for i, w in enumerate(windows):
        cat = w.cat if w.cat is not None else src.cat
        times: list[FloatArray] = []
        pos = 0
        # quakes
        n = len(w.quakes)
        if n:
            sl = slice(pos, pos + n)
            hid = w.quake_hidden
            vals = np.where(hid[:, None], 0.0, cat.values[w.quakes])
            inst[i, sl] = QUAKE
            values[i, sl, :4] = vals
            hidden[i, sl, :4] = hid[:, None]
            label[i, sl] = cat.mag_type[w.quakes]
            coords[i, sl, 0] = cat.time[w.quakes] - w.t_ref
            coords[i, sl, 1:] = vals[:, :3]
            known[i, sl, 0] = True
            known[i, sl, 1:] = ~hid[:, None]
            times.append(cat.time[w.quakes])
            q_targets.append(cat.values[w.quakes][hid])
            q_rows.append(w.quakes[hid])
            q_pos.append(i * length + pos + np.nonzero(hid)[0])
            q_w.append(np.full(int(hid.sum()), 1.0 / max(int(hid.sum()), 1)))
            pos += n
        # GNSS
        n = len(w.gnss)
        if n:
            assert src.gnss is not None
            g = src.gnss
            sl = slice(pos, pos + n)
            st = g.station[w.gnss]
            t = g.time[w.gnss]
            years = (t - LINE_EPOCH) / YEAR
            unseen = np.isin(st, list(w.unseen)) if w.unseen else np.zeros(n, bool)
            hid = w.gnss_hidden | unseen
            lines = g.lines
            vel = np.where(unseen[:, None], lines.borrowed[st], lines.velocity[st])
            off = np.nan_to_num(lines.offset[st], nan=0.0)
            line = off + years[:, None] * vel
            resid = (g.enu[w.gnss] - line) * 1000.0  # mm
            level = np.zeros((n, 3))
            anchored = np.zeros(n, bool)
            for s in np.unique(st):
                mine = st == s
                vis = mine & ~hid
                if vis.any():
                    level[mine] = resid[vis].mean(0)
                    anchored[mine] = True
            vals = np.zeros((n, 12))
            vals[:, 0:3] = np.where(hid[:, None], 0.0, (resid - level) / 10.0)
            vals[:, 3:6] = slog(level) / 4.0
            vals[:, 6] = anchored
            vals[:, 7] = lines.own[st] & ~unseen
            vals[:, 8:11] = vel * 100.0  # cm/yr
            vals[:, 11] = years / 10.0
            inst[i, sl] = GNSS
            values[i, sl, :12] = vals
            hidden[i, sl, :3] = hid[:, None]
            site = np.round(lines.site[st], 2)
            coords[i, sl, 0] = t - w.t_ref
            coords[i, sl, 1:3] = site
            known[i, sl, :3] = True
            times.append(t)
            if hid.any():
                base = line[hid] + level[hid] / 1000.0
                lo, hi = _gnss_bins(g.enu[w.gnss][hid])
                g_lo.append((lo - base) * 1000.0)
                g_hi.append((hi - base) * 1000.0)
                g_years.append(years[hid])
                g_base.append(base)
                g_rows.append(w.gnss[hid])
                g_pos.append(i * length + pos + np.nonzero(hid)[0])
                g_w.append(np.full(int(hid.sum()), 1.0 / int(hid.sum())))
            pos += n
        # tides
        n = len(w.tides)
        if n:
            assert src.tides is not None
            tide = src.tides
            sl = slice(pos, pos + n)
            inst[i, sl] = TIDE
            values[i, sl, :3] = tide.enu[w.tides] / TIDE_SCALE_M
            coords[i, sl, 0] = tide.time[w.tides] - w.t_ref
            coords[i, sl, 1] = tide.lat[w.tides]
            coords[i, sl, 2] = tide.lon[w.tides]
            known[i, sl, :3] = True
            times.append(tide.time[w.tides])
            pos += n
        # the query, last
        if w.query_at is not None:
            inst[i, pos] = QUERY
            hidden[i, pos, :] = True
            coords[i, pos, 0] = w.query_at - w.t_ref
            known[i, pos, 0] = True
            times.append(np.array([w.query_at]))
            pos += 1
        # the clock: age, and the gap since the last quake before
        every = np.concatenate(times) if times else np.zeros(0)
        qt = np.sort(cat.time[w.quakes]) if len(w.quakes) else np.zeros(0)
        # A masked window counts forward from its first quake (before and after stay apart,
        # and a week of GNSS days in the window doesn't squash every quake's age together);
        # a next-day window counts back from midnight.
        start = w.t_ref
        if w.query_at is None and len(w.quakes):
            start = float(cat.time[w.quakes].min())
        clock[i, :pos, 0] = np.abs(every - start)
        clock[i, :pos, 1] = _gaps(every, qt)

    def cat_or_empty(parts: list[FloatArray], width: int) -> FloatArray:
        return np.concatenate(parts) if parts else np.zeros((0, width))

    tokens = Tokens(
        instrument=torch.from_numpy(inst),
        values=torch.from_numpy(values),
        hidden=torch.from_numpy(hidden),
        label=torch.from_numpy(label),
        coords=torch.from_numpy(coords),
        known=torch.from_numpy(known),
        clock=torch.from_numpy(np.log10(1.0 + np.maximum(clock, 0.0)).astype(np.float32)),
        padded=bool((inst == PAD).any()),
    )

    def weights(parts: list[FloatArray]) -> torch.Tensor:
        return torch.from_numpy(
            (np.concatenate(parts) if parts else np.zeros(0)).astype(np.float32)
        )

    def positions(parts: list[IntArray]) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(parts) if parts else np.zeros(0, np.int64))

    return Batch(
        tokens=tokens,
        quake_pos=positions(q_pos),
        gnss_pos=positions(g_pos),
        quake_weight=weights(q_w),
        gnss_weight=weights(g_w),
        quake_targets=torch.from_numpy(cat_or_empty(q_targets, 4).astype(np.float32)),
        gnss_lo=torch.from_numpy(cat_or_empty(g_lo, 3).astype(np.float32)),
        gnss_hi=torch.from_numpy(cat_or_empty(g_hi, 3).astype(np.float32)),
        gnss_years=torch.from_numpy(
            (np.concatenate(g_years) if g_years else np.zeros(0)).astype(np.float32)
        ),
        gnss_base=cat_or_empty(g_base, 3),
        gnss_rows=np.concatenate(g_rows) if g_rows else np.zeros(0, np.int64),
        quake_rows=np.concatenate(q_rows) if q_rows else np.zeros(0, np.int64),
    )


def _gaps(times: FloatArray, quake_times: FloatArray) -> FloatArray:
    """Seconds since the latest quake in the window strictly before each time. Where none is
    earlier, the window's median gap between quakes stands in."""
    if not len(quake_times):
        return np.zeros(len(times))
    k = np.searchsorted(quake_times, times, "left")
    gaps = times - quake_times[np.maximum(k - 1, 0)]
    typical = float(np.median(np.diff(quake_times))) if len(quake_times) > 1 else DAY
    return np.where(k > 0, gaps, typical)
