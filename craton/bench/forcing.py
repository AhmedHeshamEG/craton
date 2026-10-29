"""Forcing: what people did to the crust, as one number per forecast cell and day.

Quakes in Oklahoma followed the wastewater pumped into the ground; quakes in Groningen
followed the gas pumped out of it. Both are a **push**: an amount of something done to the
rock over a period (cubic metres injected in a month; megapascals of pressure lost in a
year, summed over the reservoir's area). Every forcing instrument has a reader that turns its
tokens into pushes, and everything after that is the same for any instrument:

1. **Pushes** (:class:`Pushes`): one per completed period and cell, with when the period
   ended. A forecast made at midnight ``t`` may use a push only if its period ended by ``t``:
   a month's volume is known once the month is over, a year's pressure drop once the next
   model date has passed. That is the no-leak rule for forcing.
2. **Daily rate**: each push spread evenly over its own period, a (days, cells) series.
3. **Features at a midnight** (:func:`features`): the rate over the 90 and 365 days up to the
   last moment known, the total since the series began, each in the cell and blurred over
   its neighbours (1 and 3 cells), divided by the box's own scale before the cut; plus how
   stale the newest known period is. Logarithms throughout, so a doubling reads the same
   anywhere.

A new forcing instrument (CO₂ storage, geothermal circulation, a mine's extracted volume) is
one reader registered in ``PUSH_READERS``; nothing else changes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter

from craton.bench import csep
from craton.bench.data import DAY, read_table

if TYPE_CHECKING:
    from craton.bench.tasks import BenchContext

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: Windows of the rate features, in days.
WINDOWS = (90, 365)
#: Neighbourhood blur in cells, as the Omori memory's (``craton.field.memory``).
SPREADS = (0.0, 1.0, 3.0)
EPS = 1e-3
YEAR = 365.25 * DAY
COLUMNS = ("time", "lat", "lon", "value", "qualifiers", "temporal_s", "spatial_km")


@dataclass(frozen=True)
class ForcingSpec:
    """Which forcing a case reads: one instrument, and for injection which well classes."""

    instrument: str
    #: ``fluid_injection``: keep wells whose class starts with one of these (empty: all).
    well_classes: tuple[str, ...] = ()
    #: ``fluid_injection``: keep wells whose formation names one of these (any case; empty:
    #: all). Oklahoma's quakes follow what goes into the Arbuckle, on top of the basement.
    formations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "instrument": self.instrument,
            "well_classes": list(self.well_classes),
        }
        if self.formations:
            out["formations"] = list(self.formations)
        return out


@dataclass(frozen=True)
class Pushes:
    """Completed periods of forcing: when each ended, how long it lasted, which cell, how
    much. Sorted by end time."""

    t_end: FloatArray
    length: FloatArray  # seconds
    cell: IntArray
    amount: FloatArray

    def __len__(self) -> int:
        return len(self.t_end)

    def select(self, mask: Any) -> Pushes:
        return Pushes(self.t_end[mask], self.length[mask], self.cell[mask], self.amount[mask])

    def known_until(self, t: FloatArray) -> FloatArray:
        """The end of the newest period completed by each moment (nan before the first)."""
        ends = np.unique(self.t_end)
        i = np.searchsorted(ends, t, side="right") - 1
        return np.where(i >= 0, ends[np.maximum(i, 0)], np.nan)


Reader = Callable[[Any, ForcingSpec, csep.Grid], Pushes]


def _table(ctx: BenchContext, instrument: str) -> Any:
    files = ctx._snapshot_files(instrument)
    if not files:
        raise ValueError(f"{ctx.region}'s snapshot has no {instrument} tokens")
    return read_table(files, COLUMNS)


def _sorted(p: Pushes) -> Pushes:
    return p.select(np.argsort(p.t_end, kind="stable"))


def injection_pushes(table: Any, spec: ForcingSpec, grid: csep.Grid) -> Pushes:
    """``fluid_injection``: each token is a volume over its own period, at its well."""
    t = np.asarray(table.column("time").to_numpy(), dtype=np.float64)
    length = np.asarray(table.column("temporal_s").to_numpy(), dtype=np.float64)
    lat = np.asarray(table.column("lat").to_numpy(), dtype=np.float64)
    lon = np.asarray(table.column("lon").to_numpy(), dtype=np.float64)
    vol = np.array([v[0] for v in table.column("value").to_pylist()], dtype=np.float64)
    keep = np.ones(len(t), dtype=bool)
    if spec.well_classes or spec.formations:
        quals = [json.loads(q) for q in table.column("qualifiers").to_pylist()]
        if spec.well_classes:
            keep &= np.array(
                [q.get("well_class", "").startswith(spec.well_classes) for q in quals], dtype=bool
            )
        if spec.formations:
            want = tuple(f.upper() for f in spec.formations)
            keep &= np.array(
                [any(w in q.get("formation", "").upper() for w in want) for q in quals], dtype=bool
            )
    cell = grid.cell_of(lat, lon)
    keep &= cell >= 0
    return _sorted(Pushes(t[keep] + length[keep], length[keep], cell[keep], vol[keep]))


def pressure_pushes(table: Any, spec: ForcingSpec, grid: csep.Grid) -> Pushes:
    """``reservoir_pressure``: between two dates of one grid point, the pressure change
    (either way) times the point's area: MPa·km², the size of what was done to the rock.
    The push ends on the later date, when it becomes known."""
    t = np.asarray(table.column("time").to_numpy(), dtype=np.float64)
    lat = np.asarray(table.column("lat").to_numpy(), dtype=np.float64)
    lon = np.asarray(table.column("lon").to_numpy(), dtype=np.float64)
    side = np.asarray(table.column("spatial_km").to_numpy(), dtype=np.float64)
    p = np.array([v[0] for v in table.column("value").to_pylist()], dtype=np.float64)
    order = np.lexsort((t, lon, lat))
    t, lat, lon, side, p = t[order], lat[order], lon[order], side[order], p[order]
    same = (lat[1:] == lat[:-1]) & (lon[1:] == lon[:-1])
    cell = grid.cell_of(lat[1:], lon[1:])
    keep = same & (cell >= 0)
    amount = np.abs(p[1:] - p[:-1]) * side[1:] ** 2
    length = t[1:] - t[:-1]
    return _sorted(Pushes(t[1:][keep], length[keep], cell[keep], amount[keep]))


#: The extension point: one reader per forcing instrument.
PUSH_READERS: dict[str, Callable[[Any, ForcingSpec, csep.Grid], Pushes]] = {
    "fluid_injection": injection_pushes,
    "reservoir_pressure": pressure_pushes,
}


def pushes(ctx: BenchContext, spec: ForcingSpec, grid: csep.Grid) -> Pushes:
    """Every push of ``spec``'s instrument inside ``grid``, from ``ctx``'s snapshot."""
    if spec.instrument not in PUSH_READERS:
        raise ValueError(f"no push reader for {spec.instrument}; known: {sorted(PUSH_READERS)}")
    key = f"pushes:{json.dumps(spec.as_dict())}:{grid.lat_min}:{grid.lon_min}:{grid.cell_deg}"
    if key not in ctx._cache:
        ctx._cache[key] = PUSH_READERS[spec.instrument](_table(ctx, spec.instrument), spec, grid)
    out: Pushes = ctx._cache[key]
    return out


# ---------------------------------------------------------------- from pushes to features


@dataclass(frozen=True)
class Daily:
    """The pushes spread over their days: ``cum[d]`` is the total up to day ``day0 + d``."""

    day0: float  # seconds, a midnight
    cum: np.ndarray  # (days + 1, cells) float64

    def total_until(self, t: FloatArray) -> np.ndarray:
        """(len(t), cells) the total pushed before each moment (whole days)."""
        d = np.clip(np.floor((t - self.day0) / DAY).astype(np.int64), 0, len(self.cum) - 1)
        return np.asarray(self.cum[d])


def daily(p: Pushes, grid: csep.Grid) -> Daily:
    """Each push spread evenly over the days of its own period."""
    if not len(p):
        raise ValueError("no pushes to spread")
    start = p.t_end - p.length
    day0 = float(np.floor(start.min() / DAY) * DAY)
    a = np.floor((start - day0) / DAY).astype(np.int64)
    b = np.maximum(np.round((p.t_end - day0) / DAY).astype(np.int64), a + 1)
    n = int(b.max())
    rate = np.zeros((n + 1, grid.n_cells))
    per_day = p.amount / (b - a)
    np.add.at(rate, (a, p.cell), per_day)
    np.add.at(rate, (b, p.cell), -per_day)
    per = np.cumsum(rate, axis=0)[:n]  # the rate on each day
    cum = np.zeros((n + 1, grid.n_cells))
    cum[1:] = np.cumsum(per, axis=0)
    return Daily(day0, cum)


def _blur(x: np.ndarray, grid: csep.Grid, sigma: float) -> np.ndarray:
    if sigma == 0:
        return x
    shape = (*x.shape[:-1], grid.n_lat, grid.n_lon)
    out = gaussian_filter(x.reshape(shape), sigma=(0,) * (x.ndim - 1) + (sigma, sigma))
    return np.asarray(out.reshape(x.shape))


@dataclass(frozen=True)
class Raw:
    """What a midnight knows of the forcing, before any scaling: per window, the rate over the
    window ending when the newest period known ended (amount per day), and the total."""

    rates: dict[int, np.ndarray]  # window → (midnights, cells)
    total: np.ndarray  # (midnights, cells)
    stale_days: FloatArray  # (midnights,) how old the newest known period is


def raw(p: Pushes, grid: csep.Grid, midnights: FloatArray) -> Raw:
    d = daily(p, grid)
    known = p.known_until(midnights)
    none = np.isnan(known)
    k = np.where(none, d.day0, known)
    total = np.where(none[:, None], 0.0, d.total_until(k))
    rates = {}
    for w in WINDOWS:
        before = np.where(none[:, None], 0.0, d.total_until(k - w * DAY))
        rates[w] = (total - before) / w
    stale = np.where(none, 10 * 365.25, (midnights - k) / DAY)
    return Raw(rates, total, stale)


@dataclass(frozen=True)
class Scales:
    """A box's own sizes, from its training days only: what "a usual push here" means."""

    rate: float
    total: float

    @classmethod
    def of(cls, r: Raw) -> Scales:
        rate = r.rates[365]
        pos_r = rate[rate > 0]
        pos_t = r.total[r.total > 0]
        return cls(
            float(pos_r.mean()) if len(pos_r) else 1.0,
            float(pos_t.mean()) if len(pos_t) else 1.0,
        )


#: Names of the features, in order (``features``' last axis).
FEATURES = tuple(
    [f"rate{w}_s{s:g}" for w in WINDOWS for s in SPREADS]
    + [f"total_s{s:g}" for s in SPREADS]
    + ["stale_years"]
)


def features(r: Raw, grid: csep.Grid, scales: Scales) -> np.ndarray:
    """(midnights, cells, len(FEATURES)) float16, every one a log of the box's own scale."""
    parts = []
    for w in WINDOWS:
        x = np.maximum(r.rates[w], 0.0) / scales.rate
        parts += [np.log(EPS + _blur(x, grid, s)) for s in SPREADS]
    x = np.maximum(r.total, 0.0) / scales.total
    parts += [np.log(EPS + _blur(x, grid, s)) for s in SPREADS]
    stale = np.broadcast_to((r.stale_days / 365.25)[:, None], r.total.shape)
    parts.append(np.minimum(stale, 10.0))
    return np.stack(parts, axis=-1).astype(np.float16)


def rate_field(r: Raw, grid: csep.Grid, window: int, spread: float) -> np.ndarray:
    """(midnights, cells) one raw rate, blurred: what the physics floor makes rates from."""
    return np.asarray(_blur(np.maximum(r.rates[window], 0.0), grid, spread))
