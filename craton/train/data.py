"""Training examples: masked windows and next-day questions, cut by the context builder.

Masked windows come in three kinds, mixed in every batch:

    quake     anchored on a quake: the quakes around it, 15% with place and size hidden (their
              time and magnitude type stay visible, as in CB-01), plus the GNSS days and tide
              samples the policy adds, 15% of the GNSS days hidden too
    gnss      anchored on a GNSS day, in one of four modes:
                random      15% of the GNSS days and quakes hidden (CB-01's game)
                station     one station hidden whole, its line borrowed from its neighbours
                            (CB-04's game: rebuild a station nobody has seen)
                all_gnss    every GNSS day hidden: predict the ground's motion from quakes and
                            tides alone (cross-modal masking)
                all_quakes  every quake hidden: predict them from the ground and the tides
                            (the same game, the other way round)
    region    a box of 0.6° nobody has shown the model before this window: its quakes from a
              year earlier (the first 0, 7, 30, 90 or 365 days of that year, at most
              ``region_context`` of them), then its next quakes, all hidden, filling the window
              (CB-17's game: adapt from context alone)

Next-day questions stand at a midnight with the fused context before it and a query token,
and ask for the next 24 hours: how many, where, how big (CB-02's question).

A share of 30-day blocks, picked by a salted hash, is kept for validation: the loop watches
its loss to know when to stop. Nothing here reads the benchmark answers; every example comes
from one training split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.special import gammaln

from craton.bench import csep
from craton.bench.data import DAY
from craton.model.heads import RATE_SCALES
from craton.train.context import (
    Batch,
    Catalog,
    Sources,
    Window,
    gnss_rows,
    quakes_around,
    tide_rows,
    where_now,
    write,
)

IntArray = np.ndarray[Any, np.dtype[np.int64]]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

__all__ = [
    "Catalog",
    "Masked",
    "NextDay",
    "NextDayBatch",
    "in_validation",
    "recent_rates",
    "region_context",
]

BLOCK_DAYS = 30
VAL_SHARE = 0.1

#: How often each kind of masked window is drawn (renormalised over what the model reads).
KINDS = {"quake": 0.45, "gnss": 0.40, "region": 0.15}
#: The modes of a GNSS-anchored window.
GNSS_MODES = {"random": 0.5, "station": 0.3, "all_gnss": 0.1, "all_quakes": 0.1}
#: Context lengths a region window shows, in days: CB-17's grid.
REGION_DAYS = (0, 7, 30, 90, 365)
REGION_HALF_DEG = 0.3
#: A region window's context holds at most this share of the window; its hidden quakes fill the
#: rest, so a window is always full (no padding while training: the fast kernel stays on).
REGION_CONTEXT_SHARE = 0.75


def region_context(length: int) -> int:
    """How many context quakes a region window keeps at most (768 of 1,024)."""
    return int(REGION_CONTEXT_SHARE * length)


def in_validation(t: FloatArray, salt: str = "craton-val") -> BoolArray:
    """Whether each moment falls in a validation block (30-day blocks, 10% by salted hash)."""
    block = np.floor(t / (BLOCK_DAYS * DAY)).astype(np.int64)
    out = np.zeros(len(block), dtype=bool)
    for b in np.unique(block):
        h = int(hashlib.sha256(f"{salt}:{b}".encode()).hexdigest()[:8], 16) / 16**8
        out[block == b] = h < VAL_SHARE
    return out


def _tides_around(src: Sources, t: float) -> IntArray:
    half = 0.5 * src.policy.tides * 3600.0
    return tide_rows(src, t - half, t + half, src.policy.tides)


def _fill_quakes(src: Sources, w: Window, t: float, share: float, rng: np.random.Generator) -> None:
    """Fill the rest of the window with the quakes nearest ``t``, ``share`` of them hidden."""
    if not src.quakes:
        return
    n = src.length - w.size
    if n <= 0:
        return
    w.quakes = quakes_around(src.cat, t, n)
    w.quake_hidden = rng.random(len(w.quakes)) < share


# ---------------------------------------------------------------- masked windows


class Masked:
    """Masked windows of every kind, a fresh mask every time."""

    def __init__(
        self, src: Sources, mask_share: float, kinds: dict[str, float] | None = None
    ) -> None:
        if src.quakes and len(src.cat) < src.length:
            raise ValueError(f"{len(src.cat)} quakes is fewer than one window of {src.length}")
        self.src, self.share = src, mask_share
        kinds = {k: w for k, w in (kinds or KINDS).items() if w > 0}
        if src.gnss is None:
            kinds.pop("gnss", None)
        if not src.quakes:
            kinds.pop("quake", None)
            kinds.pop("region", None)
        total = sum(kinds.values())
        self.kinds = {k: v / total for k, v in kinds.items()}
        self.quake_rows = np.arange(len(src.cat)) if src.quakes else np.zeros(0, np.int64)
        q_val = in_validation(src.cat.time[self.quake_rows])
        self.quake_train, self.quake_val = self.quake_rows[~q_val], self.quake_rows[q_val]
        if src.gnss is not None:
            rows = np.arange(len(src.gnss))
            g_val = in_validation(src.gnss.time)
            self.gnss_train, self.gnss_val = rows[~g_val], rows[g_val]

    # one window of each kind

    def quake_window(self, row: int, rng: np.random.Generator) -> Window:
        src, cat = self.src, self.src.cat
        t = float(cat.time[row])
        w = Window(t_ref=t, tides=_tides_around(src, t))
        # The quakes come first so the GNSS stations can be picked near the visible ones.
        budget = src.length - w.size - (src.policy.gnss if src.gnss is not None else 0)
        rows = quakes_around(cat, t, budget)
        hid = rng.random(len(rows)) < self.share
        lat, lon = where_now(cat, rows, hid)
        if src.gnss is not None:
            day = np.floor(t / DAY) * DAY
            span = src.policy.gnss_days * DAY
            w.gnss = gnss_rows(src, lat, lon, day - span, day + span + DAY)
            w.gnss_hidden = rng.random(len(w.gnss)) < self.share
        more = quakes_around(cat, t, src.length - w.size)
        extra = ~np.isin(more, rows)
        hidden = np.zeros(len(more), bool)
        hidden[~extra] = hid[np.searchsorted(rows, more[~extra])]
        hidden[extra] = rng.random(int(extra.sum())) < self.share
        w.quakes, w.quake_hidden = more, hidden
        return w

    def gnss_window(self, row: int, mode: str, rng: np.random.Generator) -> Window:
        src = self.src
        assert src.gnss is not None
        g = src.gnss
        t = float(g.time[row])
        s = int(g.station[row])
        lat, lon = g.lines.site[s]
        w = Window(t_ref=t, tides=_tides_around(src, t))
        day = np.floor(t / DAY) * DAY
        span = src.policy.gnss_days * DAY
        w.gnss = gnss_rows(src, lat, lon, day - span, day + span + DAY, first=(s,))
        if mode == "station":
            w.unseen = frozenset({s})
            w.gnss_hidden = np.zeros(len(w.gnss), bool)
        elif mode == "all_gnss":
            w.gnss_hidden = np.ones(len(w.gnss), bool)
        else:
            w.gnss_hidden = rng.random(len(w.gnss)) < self.share
            w.gnss_hidden[w.gnss == row] = True
        _fill_quakes(src, w, t, 1.0 if mode == "all_quakes" else self.share, rng)
        return w

    def region_window(self, row: int, rng: np.random.Generator) -> Window | None:
        cat = self.src.cat
        lat, lon = cat.values[row, :2] + rng.uniform(-REGION_HALF_DEG, REGION_HALF_DEG, 2)
        inside = np.nonzero(
            (np.abs(cat.values[:, 0] - lat) <= REGION_HALF_DEG)
            & (np.abs(cat.values[:, 1] - lon) <= REGION_HALF_DEG)
        )[0]
        t0 = float(cat.time[row])
        days = int(rng.choice(REGION_DAYS))
        start = t0 - 365 * DAY
        before = inside[(cat.time[inside] >= start) & (cat.time[inside] < start + days * DAY)]
        before = before[len(before) - min(len(before), region_context(self.src.length)) :]
        later = inside[(cat.time[inside] >= t0) & (cat.time[inside] < t0 + 365 * DAY)]
        need = self.src.length - len(before)
        if len(later) < need:
            return None
        queries = later[:need]
        rows = np.concatenate([before, queries])
        hidden = np.concatenate([np.zeros(len(before), bool), np.ones(len(queries), bool)])
        return Window(t_ref=t0, quakes=rows, quake_hidden=hidden)

    def window(self, kind: str, rng: np.random.Generator, val: bool = False) -> Window:
        if kind == "gnss":
            rows = self.gnss_val if val else self.gnss_train
            mode = str(rng.choice(list(GNSS_MODES), p=list(GNSS_MODES.values())))
            if not self.src.quakes and mode == "all_quakes":
                mode = "random"
            return self.gnss_window(int(rng.choice(rows)), mode, rng)
        rows = self.quake_val if val else self.quake_train
        if kind == "region":
            for _ in range(50):
                w = self.region_window(int(rng.choice(rows)), rng)
                if w is not None:
                    return w
        return self.quake_window(int(rng.choice(rows)), rng)

    def sample(self, size: int, rng: np.random.Generator, val: bool = False) -> Batch:
        kinds = rng.choice(list(self.kinds), size=size, p=list(self.kinds.values()))
        return write(self.src, [self.window(str(k), rng, val) for k in kinds])

    def validation(self, size: int, n_batches: int, seed: int = 12345) -> list[Batch]:
        """A fixed set of validation windows and masks, the same at every check."""
        rng = np.random.default_rng(seed)
        if self.src.quakes and not len(self.quake_val):
            return []
        return [self.sample(size, rng, val=True) for _ in range(n_batches)]


# ---------------------------------------------------------------- next-day questions


@dataclass
class NextDayBatch:
    batch: Batch
    counts: torch.Tensor  # (B,) quakes in each next day (in the grid, M ≥ grid minimum)
    cells: torch.Tensor  # (N,) cell of every one of those quakes
    mags: torch.Tensor  # (N,) magnitude bin
    day_of: torch.Tensor  # (N,) which row of the batch it belongs to
    recent: torch.Tensor  # (B, scales) log quakes per day just before each midnight
    log_factorials: float  # sum of log n! over every (day, cell, bin) count

    @property
    def tokens(self) -> Any:
        return self.batch.tokens

    def pin_memory(self) -> NextDayBatch:
        return NextDayBatch(
            batch=self.batch.pin_memory(),
            counts=self.counts.pin_memory(),
            cells=self.cells.pin_memory(),
            mags=self.mags.pin_memory(),
            day_of=self.day_of.pin_memory(),
            recent=self.recent.pin_memory(),
            log_factorials=self.log_factorials,
        )


def recent_rates(cat: Catalog, day_starts: FloatArray) -> torch.Tensor:
    """Natural-log rate per day just before each midnight, for every (k, magnitude) in
    ``RATE_SCALES``: k / (age of the k-th last quake at or above that magnitude). Where fewer
    than k such quakes exist, the age runs from the catalogue's first quake. Reads only quakes
    before the midnight."""
    out = np.zeros((len(day_starts), len(RATE_SCALES)))
    for j, (k, m_min) in enumerate(RATE_SCALES):
        t = cat.time[cat.values[:, 3] >= m_min - 1e-9]
        first = np.searchsorted(t, day_starts, side="left")
        oldest = np.where(first >= k, t[np.maximum(first - k, 0)], cat.time[0])
        age = np.maximum(day_starts - oldest, 1.0) / DAY
        out[:, j] = np.log(k / age)
    return torch.from_numpy(out.astype(np.float32))


def midnight_window(src: Sources, day: float) -> Window:
    """The fused context before a midnight, and the query at it. Reads nothing after it
    except the tide, which the Sun and the Moon fix in advance."""
    p, cat = src.policy, src.cat
    w = Window(t_ref=day, query_at=day)
    w.tides = tide_rows(src, day - 0.5 * p.tides * 3600.0, day + p.tides_ahead * 3600.0, 10**6)
    first = int(np.searchsorted(cat.time, day, "left"))
    lat, lon = where_now(cat, np.arange(max(first - 100, 0), first))
    if src.gnss is not None:
        w.gnss = gnss_rows(src, lat, lon, day - p.next_day_gnss_days * DAY, day)
        w.gnss_hidden = np.zeros(len(w.gnss), bool)
    n = src.length - w.size
    if n > first:
        raise ValueError("a midnight has fewer quakes before it than the context needs")
    recent = np.arange(first - (n - p.big_quakes), first, dtype=np.int64)
    year = np.arange(int(np.searchsorted(cat.time, day - 365 * DAY)), recent[0], dtype=np.int64)
    big = year[cat.values[year, 3] >= p.big_magnitude - 1e-9]
    big = np.sort(big[np.argsort(-cat.values[big, 3], kind="stable")][: p.big_quakes])
    recent = np.arange(first - (n - len(big)), first, dtype=np.int64)
    w.quakes = np.concatenate([big, recent])
    w.quake_hidden = np.zeros(len(w.quakes), bool)
    return w


class NextDay:
    """Every midnight with enough history and a whole next day inside ``[start, end)``."""

    def __init__(self, src: Sources, grid: csep.Grid, start: float, end: float) -> None:
        self.src, self.grid = src, grid
        cat = src.cat
        need = max(src.length, max(k for k, _ in RATE_SCALES))  # quakes needed behind midnight
        t_min = cat.time[need] if len(cat) > need else np.inf
        first_day = max(np.ceil(start / DAY), np.ceil(t_min / DAY)) * DAY
        days = np.arange(first_day, end - DAY + 1e-6, DAY)
        if not len(days):
            raise ValueError("no whole day with enough history in the training window")
        cell = grid.cell_of(cat.values[:, 0], cat.values[:, 1])
        mbin = grid.mag_bin_of(cat.values[:, 3])
        keep = (cell >= 0) & (mbin >= 0)
        self.t, self.cell, self.mbin = cat.time[keep], cell[keep], mbin[keep]
        val = in_validation(days)
        self.train_days, self.val_days = days[~val], days[val]

    def batch(self, days: FloatArray) -> NextDayBatch:
        batch = write(self.src, [midnight_window(self.src, float(d)) for d in days])
        lo = np.searchsorted(self.t, days, side="left")
        hi = np.searchsorted(self.t, days + DAY, side="left")
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True)])
        idx = idx.astype(np.int64)
        day_of = np.repeat(np.arange(len(days)), hi - lo)
        key = (day_of * self.grid.n_cells + self.cell[idx]) * self.grid.n_mags + self.mbin[idx]
        _, per_bin = np.unique(key, return_counts=True)
        return NextDayBatch(
            batch=batch,
            counts=torch.from_numpy((hi - lo).astype(np.int64)),
            cells=torch.from_numpy(self.cell[idx].astype(np.int64)),
            mags=torch.from_numpy(self.mbin[idx].astype(np.int64)),
            day_of=torch.from_numpy(day_of.astype(np.int64)),
            recent=recent_rates(self.src.cat, days),
            log_factorials=float(gammaln(per_bin + 1.0).sum()),
        )

    def sample(self, size: int, rng: np.random.Generator) -> NextDayBatch:
        return self.batch(rng.choice(self.train_days, size))

    def validation(self, size: int) -> list[NextDayBatch]:
        days = self.val_days
        return [self.batch(days[i : i + size]) for i in range(0, len(days), size)]
