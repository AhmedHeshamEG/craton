"""The predictability meter: how much of space and time you must watch to catch 90% of the
M4+ quakes.

A forecast that knew exactly when and where would let you watch one cell on one day per quake.
A forecast that knows nothing makes you watch everything. Between those two, the meter asks:

* **space-time** (CB-11's number): rank every (day, cell) of the test windows by the forecast's
  M4+ rate; walk down the ranking until 90% of the real M4+ quakes are inside. The share of all
  cell-days you walked through is the window. Smaller is sharper.
* **space**: on each day that had an M4+ quake, the smallest set of cells holding a fixed share
  of that day's forecast; the share is chosen so the sets hold 90% of the real quakes (the
  same trick as conformal prediction: the coverage is guaranteed by construction, so only
  the size is compared). Reported as the radius of a circle of the same area, in km.
* **time**: rank the days by the forecast's total M4+ rate; the days you must be on alert to
  catch 90% of the M4+ quakes.
* **magnitude**: the width of the forecast's central 90% interval of magnitude, given an M4+
  quake happened.

Every width is computed the same way for every method, so they compare directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.tasks import ForecastSpec, GriddedForecast

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

COVERAGE = 0.9
M_METER = 4.0
KM_PER_DEG = 111.195


@dataclass(frozen=True)
class MeterPart:
    """One test window's forecast and quakes, ready for the meter."""

    rates: FloatArray  # (days, cells) expected M4+ quakes
    day: np.ndarray  # (n,) day of each M4+ quake
    cell: np.ndarray  # (n,) its cell
    mag_shares: FloatArray  # (days, cells, mags) or broadcastable
    first: int  # first magnitude bin at or above M_METER
    cell_km2: FloatArray  # (cells,) area of each cell


def cell_areas(grid: csep.Grid) -> FloatArray:
    lat, _ = grid.cell_centres()
    side = grid.cell_deg * KM_PER_DEG
    return np.asarray(side * side * np.cos(np.radians(lat)), dtype=np.float64)


def part(fc: GriddedForecast, obs: Events, spec: ForecastSpec) -> MeterPart:
    g = spec.grid
    first = int(g.mag_bin_of(np.array([M_METER]))[0])
    big = obs.above(M_METER)
    cell = g.cell_of(big.lat, big.lon)
    day = np.floor((big.time - spec.window.t0) / DAY).astype(np.int64)
    ok = (cell >= 0) & (day >= 0) & (day < fc.rates.shape[0])
    return MeterPart(fc.expected(first), day[ok], cell[ok], fc.mag_shares, first, cell_areas(g))


def _cover_share(scores: FloatArray, hits: FloatArray, coverage: float) -> float:
    """Share of items (ranked by ``scores``, highest first) needed to hold ``coverage`` of
    ``hits``. Ties are broken against the forecast: tied items count as all-or-nothing."""
    order = np.argsort(-scores, kind="stable")
    s, h = scores[order], hits[order]
    need = coverage * h.sum()
    cum = np.cumsum(h)
    i = int(np.searchsorted(cum, need - 1e-9))
    # every item tied with the one that crosses the line has to be watched too
    last = int(np.searchsorted(-s, -s[i], side="right")) - 1
    return float((last + 1) / len(s))


def space_time_share(parts: list[MeterPart], coverage: float = COVERAGE) -> float:
    scores = np.concatenate([p.rates.ravel() for p in parts])
    hits = np.concatenate(
        [
            np.bincount(p.day * p.rates.shape[1] + p.cell, minlength=p.rates.size).astype(float)
            for p in parts
        ]
    )
    return _cover_share(scores, hits, coverage)


def time_share(parts: list[MeterPart], coverage: float = COVERAGE) -> tuple[float, float]:
    """(share of days, days per year) on alert to catch ``coverage`` of the quakes."""
    scores = np.concatenate([p.rates.sum(axis=1) for p in parts])
    hits = np.concatenate(
        [np.bincount(p.day, minlength=p.rates.shape[0]).astype(float) for p in parts]
    )
    share = _cover_share(scores, hits, coverage)
    return share, share * 365.25


def where_quantile(p: MeterPart) -> FloatArray:
    """For each quake, the share of its day's rate in cells ranked above its own cell, plus
    its own cell: the smallest top-ranked set that contains it."""
    out = np.zeros(len(p.day))
    for i, (d, c) in enumerate(zip(p.day, p.cell, strict=True)):
        r = p.rates[d]
        tot = r.sum()
        out[i] = float(r[r >= r[c]].sum() / tot) if tot > 0 else 1.0
    return out


def space_radius(parts: list[MeterPart], coverage: float = COVERAGE) -> tuple[float, float]:
    """(radius km of the coverage-matched daily area, nominal coverage of the forecast's own
    90% area). The radius is averaged over quakes."""
    q = np.concatenate([where_quantile(p) for p in parts])
    if not len(q):
        return math.nan, math.nan
    level = float(np.quantile(q, coverage, method="higher"))
    nominal = float(np.mean(q <= coverage + 1e-12))
    radii = []
    for p in parts:
        for d in p.day:
            r = p.rates[d]
            order = np.argsort(-r, kind="stable")
            cum = np.cumsum(r[order]) / max(r.sum(), 1e-300)
            k = int(np.searchsorted(cum, level - 1e-12)) + 1
            area = float(p.cell_km2[order[:k]].sum())
            radii.append(math.sqrt(area / math.pi))
    return float(np.mean(radii)), nominal


def magnitude_width(parts: list[MeterPart], grid: csep.Grid, coverage: float = COVERAGE) -> float:
    """Mean width of the central ``coverage`` interval of magnitude, given M ≥ M_METER, at each
    quake's (day, cell)."""
    lo_t, hi_t = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    mags = grid.magnitudes
    widths = []
    for p in parts:
        shares = np.broadcast_to(p.mag_shares, (*p.rates.shape, p.mag_shares.shape[-1]))
        for d, c in zip(p.day, p.cell, strict=True):
            s = shares[d, c, p.first :]
            cdf = np.cumsum(s) / max(s.sum(), 1e-300)
            m = mags[p.first :]
            hi = min(int(np.searchsorted(cdf, hi_t)), len(m) - 1)
            lo = min(int(np.searchsorted(cdf, lo_t)), len(m) - 1)
            widths.append(float(m[hi] - m[lo]) + grid.m_step)
    return float(np.mean(widths)) if widths else math.nan
