"""Physics budgets: things no honest forecast may do, checked on every forecast.

Hard violations (CB-08 counts them; the target is zero):

* **Rates** must be finite and non-negative, magnitude shares too.
* **Gutenberg-Richter**: more small quakes than big ones. Each cell-day's magnitude shares must
  not grow with magnitude, and the b-value they imply at M3 must lie between 0.5 and 2.
* **Omori decay**: the *probe*. Give the forecaster the real catalogue up to the end of a
  mainshock's first day and then nothing at all. With no new quakes the forecast may only
  fade. Every day on which the box's M3+ rate rises by more than 0.5% is a violation.
* **The slip budget**: on calm days (no M5+ in the box in the 30 days before), the long-run
  seismic moment the forecast implies may not exceed what the crust can store. The budget is
  Kostrov's moment rate for the central Apennines with every number at its generous end:
  extension 4 mm/yr (geological estimates span ~2-4), a 15 km brittle layer, 400 km of belt,
  45° normal faults (×2), shear modulus 33 GPa: 1.6 × 10¹⁸ N·m/yr. A forecast that spreads
  Gutenberg-Richter all the way to M8 breaks it: the open top bins alone carry more moment
  than the plate can deliver. The largest quake on record in the region is the 1915 Avezzano
  earthquake, Mw 7.1 in CPTI15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.tasks import GriddedForecast

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Kostrov moment rate, generous end: 2 · μ · H · L · v (dip 45°), N·m per year.
SHEAR_MODULUS = 3.3e10
BRITTLE_KM = 15.0
BELT_KM = 400.0
EXTENSION_MM_YR = 4.0
MOMENT_BUDGET = 2 * SHEAR_MODULUS * BRITTLE_KM * 1e3 * BELT_KM * 1e3 * EXTENSION_MM_YR * 1e-3
#: Calm day: no quake this big in the box in the previous CALM_DAYS days.
CALM_MAGNITUDE = 5.0
CALM_DAYS = 30
OMORI_TOLERANCE = 0.005
B_RANGE = (0.5, 2.0)


def moment(mw: FloatArray) -> FloatArray:
    """Seismic moment in N·m of moment magnitude ``mw`` (Hanks and Kanamori 1979)."""
    return np.asarray(10.0 ** (1.5 * np.asarray(mw) + 9.1), dtype=np.float64)


@dataclass(frozen=True)
class Checks:
    bad_rates: int
    bad_shares: int
    gr_rising: int
    b_out_of_range: int
    moment_rate: float  # N·m/yr implied on calm days (nan if none)
    over_budget: int

    @property
    def violations(self) -> int:
        return (
            self.bad_rates
            + self.bad_shares
            + self.gr_rising
            + self.b_out_of_range
            + self.over_budget
        )


def calm_days(obs: Events, day_starts: FloatArray) -> np.ndarray:
    """Days with no M ≥ CALM_MAGNITUDE quake in the CALM_DAYS days before their midnight."""
    big = obs.above(CALM_MAGNITUDE).time
    out = np.ones(len(day_starts), dtype=bool)
    for t in big:
        out &= ~((day_starts > t) & (day_starts <= t + CALM_DAYS * DAY))
    return out


def check_forecast(
    fc: GriddedForecast, grid: csep.Grid, history: Events, day_starts: FloatArray
) -> Checks:
    """Every check except the Omori probe. ``history`` holds the quakes the days could read
    (and the days themselves: a calm day is calm by what came before it)."""
    rates, shares = fc.rates, np.broadcast_to(fc.mag_shares, (*fc.rates.shape, grid.n_mags))
    bad_rates = int(np.sum(~np.isfinite(rates) | (rates < 0)))
    bad_shares = int(np.sum(~np.isfinite(fc.mag_shares) | (fc.mag_shares < 0)))
    # Gutenberg-Richter: shares may only fall with magnitude (a hair of float noise allowed).
    # The last bin is open (everything from m_max up), so it is left out of the comparison.
    uniq = np.unique(fc.mag_shares.reshape(-1, grid.n_mags), axis=0)
    closed = uniq[:, :-1]
    gr_rising = int(np.sum(np.any(np.diff(closed, axis=1) > 1e-12 * closed.max(), axis=1)))
    i3 = int(grid.mag_bin_of(np.array([3.0]))[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.log10(uniq[:, i3] / uniq[:, i3 + 1]) / grid.m_step
    b_bad = int(np.sum(~np.isfinite(b) | (b < B_RANGE[0]) | (b > B_RANGE[1])))
    calm = calm_days(history, day_starts)
    if calm.any():
        per_quake = (shares[calm] * moment(grid.magnitudes)).sum(axis=-1)  # (calm days, cells)
        daily = (rates[calm] * per_quake).sum(axis=-1)
        rate_yr = float(daily.mean() * 365.25)
    else:
        rate_yr = math.nan
    over = int(math.isfinite(rate_yr) and rate_yr > MOMENT_BUDGET)
    return Checks(bad_rates, bad_shares, gr_rising, b_bad, rate_yr, over)


def omori_violations(fc: GriddedForecast, first_bin: int) -> tuple[int, FloatArray]:
    """Days on which the box's rate rose although no quake was added (the probe's forecast)."""
    total = fc.expected(first_bin).sum(axis=1)
    rise = total[1:] > total[:-1] * (1 + OMORI_TOLERANCE)
    return int(rise.sum()), total
