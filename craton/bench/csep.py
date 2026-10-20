"""CSEP-style gridded forecasts and the tests that grade them.

A gridded forecast says how many earthquakes to expect in every (space cell, magnitude bin) of a
time window. The Collaboratory for the Study of Earthquake Predictability (CSEP) grades such a
forecast with a joint Poisson log-likelihood and a family of consistency tests:

* **N-test**: is the total number right? (Zechar et al. 2010)
* **S-test**: is the *where* right, given the total? (spatial log-likelihood vs simulations)
* **M-test**: is the *how big* right, given the total?
* **Information gain per earthquake** between two forecasts: the difference of their joint
  log-likelihoods divided by the number of observed earthquakes (Rhoades et al. 2011).

This is a small NumPy implementation of the Poisson versions, written to match pyCSEP 0.8
(``csep.core.poisson_evaluations``). ``scripts/reference/csep_reference.py`` checks it against
pyCSEP itself; the numbers are in ``tests/fixtures/csep/reference.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

LN2 = math.log(2.0)


@dataclass(frozen=True)
class Grid:
    """Square cells in degrees over a box, and magnitude bins from ``m_min`` up.

    The last magnitude bin is open: it holds everything at or above ``m_max``. Magnitudes are
    reported to 0.1, so bin edges sit half a step below each value (2.95, 3.05, …): an event
    of magnitude 3.0 lands in the bin labelled 3.0.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    cell_deg: float = 0.1
    m_min: float = 3.0
    m_max: float = 8.0
    m_step: float = 0.1

    @property
    def n_lat(self) -> int:
        return round((self.lat_max - self.lat_min) / self.cell_deg)

    @property
    def n_lon(self) -> int:
        return round((self.lon_max - self.lon_min) / self.cell_deg)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    @property
    def magnitudes(self) -> FloatArray:
        """The label (lower value) of every magnitude bin."""
        n = round((self.m_max - self.m_min) / self.m_step) + 1
        return np.round(self.m_min + self.m_step * np.arange(n), 6)

    @property
    def n_mags(self) -> int:
        return len(self.magnitudes)

    def cell_centres(self) -> tuple[FloatArray, FloatArray]:
        """(lat, lon) of every cell centre, row-major: cell = i_lat * n_lon + i_lon."""
        lat = self.lat_min + self.cell_deg * (np.arange(self.n_lat) + 0.5)
        lon = self.lon_min + self.cell_deg * (np.arange(self.n_lon) + 0.5)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        return la.ravel(), lo.ravel()

    def cell_of(self, lat: FloatArray, lon: FloatArray) -> IntArray:
        """Cell index of each point, or -1 outside the box."""
        i = np.floor((lat - self.lat_min) / self.cell_deg + 1e-9).astype(np.int64)
        j = np.floor((lon - self.lon_min) / self.cell_deg + 1e-9).astype(np.int64)
        inside = (i >= 0) & (i < self.n_lat) & (j >= 0) & (j < self.n_lon)
        return np.where(inside, i * self.n_lon + j, -1)

    def mag_bin_of(self, mag: FloatArray) -> IntArray:
        """Magnitude bin of each value, or -1 below ``m_min``."""
        k = np.floor((mag - self.m_min) / self.m_step + 0.5 + 1e-9).astype(np.int64)
        return np.where(mag >= self.m_min - self.m_step / 2, np.minimum(k, self.n_mags - 1), -1)

    def counts(self, lat: FloatArray, lon: FloatArray, mag: FloatArray) -> FloatArray:
        """Observed counts as a (cells, mags) array. Events outside the grid are ignored."""
        c, k = self.cell_of(lat, lon), self.mag_bin_of(mag)
        ok = (c >= 0) & (k >= 0)
        flat = np.bincount(c[ok] * self.n_mags + k[ok], minlength=self.n_cells * self.n_mags)
        return flat.reshape(self.n_cells, self.n_mags).astype(np.float64)


def gutenberg_richter_bins(b: float, mc: float, grid: Grid) -> FloatArray:
    """Expected events in each magnitude bin of the grid, per event at or above ``mc``.

    Bins are [m - step/2, m + step/2); the last is open. Bins below ``mc`` extrapolate the
    Gutenberg-Richter line downwards, so the shares can sum to more than 1: a fit at M2.0
    forecasts M1.5+ as ``rate(>= 2.0) * shares``.
    """
    half = grid.m_step / 2
    lo = grid.magnitudes - half
    hi = np.append(grid.magnitudes[1:] - half, np.inf)
    surv_lo = 10.0 ** (-b * (lo - (mc - half)))
    surv_hi = np.where(np.isinf(hi), 0.0, 10.0 ** (-b * (np.minimum(hi, 99.0) - (mc - half))))
    return np.asarray(surv_lo - surv_hi, dtype=np.float64)


def b_value(mags: FloatArray, mc: float, step: float = 0.1) -> float:
    """Aki-Utsu maximum-likelihood b-value with the binning correction (Utsu 1965)."""
    m = mags[mags >= mc - 1e-9]
    if len(m) < 2:
        raise ValueError("need at least two events above the completeness magnitude")
    return math.log10(math.e) / (float(np.mean(m)) - (mc - step / 2))


def poisson_ll(rates: FloatArray, counts: FloatArray) -> float:
    """Joint Poisson log-likelihood (natural log) of observed counts under forecast rates."""
    rates = np.asarray(rates, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(rates < 0):
        raise ValueError("a forecast rate is negative")
    with np.errstate(divide="ignore"):
        log_r = np.where(counts > 0, np.log(rates), 0.0)
    if np.any(np.isneginf(log_r)):
        return -math.inf
    return float(np.sum(-rates + counts * log_r - gammaln(counts + 1.0)))


@dataclass(frozen=True)
class NTest:
    observed: int
    expected: float
    #: P(N >= observed): small means the forecast expected too few.
    delta_1: float
    #: P(N <= observed): small means the forecast expected too many.
    delta_2: float


def n_test(expected: float, observed: int) -> NTest:
    d1 = 1.0 - float(poisson.cdf(observed - 1, expected)) if observed > 0 else 1.0
    d2 = float(poisson.cdf(observed, expected))
    return NTest(observed, float(expected), d1, d2)


@dataclass(frozen=True)
class SimTest:
    """A simulation-based consistency test: where the observation sits among simulations."""

    observed_ll: float
    #: Share of simulated log-likelihoods at or below the observed one. Small = inconsistent.
    quantile: float
    simulated_mean: float


def _simulated_ll(weights: FloatArray, n: int, n_sims: int, rng: np.random.Generator) -> FloatArray:
    """Log-likelihoods of catalogues of ``n`` events drawn from a normalised forecast."""
    p = weights / weights.sum()
    rates = p * n
    cdf = np.cumsum(p)
    out = np.empty(n_sims)
    for s in range(n_sims):
        u = rng.random(n)
        idx = np.minimum(np.searchsorted(cdf, u, side="right"), len(p) - 1)
        counts = np.bincount(idx, minlength=len(p)).astype(np.float64)
        out[s] = poisson_ll(rates, counts)
    return out


def _conditional_test(
    marginal_rates: FloatArray, marginal_counts: FloatArray, n_sims: int, seed: int
) -> SimTest:
    n = round(marginal_counts.sum())
    if n == 0:
        return SimTest(0.0, 1.0, 0.0)
    scaled = marginal_rates * n / marginal_rates.sum()
    obs = poisson_ll(scaled, marginal_counts)
    sims = _simulated_ll(marginal_rates, n, n_sims, np.random.default_rng(seed))
    return SimTest(obs, float(np.mean(sims <= obs)), float(np.mean(sims)))


def s_test(rates: FloatArray, counts: FloatArray, n_sims: int = 1000, seed: int = 0) -> SimTest:
    """Spatial test: magnitudes summed out, the forecast rescaled to the observed total."""
    return _conditional_test(rates.sum(axis=1), counts.sum(axis=1), n_sims, seed)


def m_test(rates: FloatArray, counts: FloatArray, n_sims: int = 1000, seed: int = 0) -> SimTest:
    """Magnitude test: space summed out, the forecast rescaled to the observed total."""
    return _conditional_test(rates.sum(axis=0), counts.sum(axis=0), n_sims, seed)


def bits_per_earthquake(ll: float, observed: int) -> float:
    """A joint log-likelihood per observed earthquake, in bits."""
    if observed <= 0:
        raise ValueError("no observed earthquakes")
    return ll / observed / LN2
