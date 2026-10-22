"""Calibration: when the forecast says 14%, it should happen about 14% of the time.

Three views of the same question, all on the M3+ forecast:

1. **Reliability of cells.** Every (day, cell) gets the forecast's probability of at least
   one M3+ quake there. Group them by probability; in each group compare the average
   probability with how often a quake really came. The gap, averaged with the group sizes as
   weights, is the expected calibration error (ECE). For rare events the ECE is tiny even
   when a forecast is three times too low (0.005 against 0.0016 is a gap of 0.004), so CB-05's
   number is the gap as a *factor*: |log₂(observed ÷ forecast)| per group, averaged the same
   way, over cell-days forecast at least 0.1% likely. 0 is perfect, 1 means "off by 2x".
   The ECE is kept beside it, for the cell-days at or above 1% and for all.
2. **PIT of the day's total.** Where does the real count of the day fall inside the
   forecast's distribution? If the forecast is honest those positions (probability integral
   transforms, randomised because counts are whole numbers) are uniform; a Poisson forecast
   of a clustered process piles them at 0 and 1. Scored as the Kolmogorov-Smirnov distance
   from uniform.
3. **CRPS** of the day's total: a proper score of the whole distribution, in quakes.

``fit_dispersion`` is the post-hoc calibration: on the training split's validation days, the
negative binomial size that makes the day totals (and the cell counts) most likely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize

from craton.trust.predictive import POISSON_K, nb_cdf, nb_logpmf, nb_zero

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Probability bins of the reliability diagram (the first holds everything below 1%).
RELIABILITY_EDGES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0000001)
#: CB-05's number counts only cell-days forecast at least this likely.
ACTIVE_P = 0.001
#: The ECE is also reported for cell-days at least this likely (the plan's exit test).
COMMITTED_P = 0.01


@dataclass(frozen=True)
class Reliability:
    edges: tuple[float, ...]
    n: FloatArray  # cell-days per bin
    mean_p: FloatArray
    freq: FloatArray

    def ece(self, p_min: float = 0.0) -> float:
        lo = np.asarray(self.edges[:-1])
        sel = (lo >= p_min - 1e-12) & (self.n > 0)
        if not sel.any():
            return float("nan")
        w = self.n[sel] / self.n[sel].sum()
        return float(np.sum(w * np.abs(self.freq[sel] - self.mean_p[sel])))

    def log_error(self, p_min: float = ACTIVE_P) -> float:
        """Mean |log₂(observed ÷ forecast)| over groups at or above ``p_min``, weighted by
        size; the observed share is smoothed (Jeffreys: (hits + ½) / (n + 1))."""
        lo = np.asarray(self.edges[:-1])
        sel = (lo >= p_min - 1e-12) & (self.n > 0)
        if not sel.any():
            return float("nan")
        n = self.n[sel]
        observed = (self.freq[sel] * n + 0.5) / (n + 1)
        w = n / n.sum()
        return float(np.sum(w * np.abs(np.log2(observed / self.mean_p[sel]))))


def reliability(
    p: FloatArray, hit: Any, edges: tuple[float, ...] = RELIABILITY_EDGES
) -> Reliability:
    """Group probabilities ``p`` into ``edges``; how often ``hit`` came true in each."""
    p, hit = np.ravel(p), np.ravel(np.asarray(hit, dtype=np.float64))
    k = np.clip(np.searchsorted(np.asarray(edges), p, side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(k, minlength=len(edges) - 1).astype(np.float64)
    sp = np.bincount(k, weights=p, minlength=len(edges) - 1)
    sh = np.bincount(k, weights=hit, minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Reliability(
            edges, n, np.where(n > 0, sp / n, np.nan), np.where(n > 0, sh / n, np.nan)
        )


def mixture_cdf(n: Any, means: FloatArray, k: Any) -> FloatArray:
    """P(N ≤ n) under an equal mixture of negative binomials, ``means`` (members, days)."""
    return np.asarray(np.mean(nb_cdf(np.asarray(n)[None], means, k), axis=0), dtype=np.float64)


def randomized_pit(observed: Any, means: FloatArray, k: Any, seed: int) -> FloatArray:
    """One PIT value per day: F(n − 1) + v · (F(n) − F(n − 1)), v uniform."""
    n = np.asarray(observed, dtype=np.float64)
    lo, hi = mixture_cdf(n - 1, means, k), mixture_cdf(n, means, k)
    v = np.random.default_rng(seed).uniform(size=n.shape)
    return np.asarray(lo + v * (hi - lo), dtype=np.float64)


def ks_uniform(u: FloatArray) -> float:
    """Kolmogorov-Smirnov distance of a sample from the uniform distribution on [0, 1]."""
    u = np.sort(np.ravel(u))
    m = len(u)
    if not m:
        return float("nan")
    i = np.arange(1, m + 1)
    return float(max(np.max(i / m - u), np.max(u - (i - 1) / m)))


def crps_counts(observed: Any, means: FloatArray, k: Any) -> FloatArray:
    """CRPS per day of a count forecast: Σ_j (F(j) − 1[n ≤ j])², in quakes."""
    n = np.asarray(observed, dtype=np.float64)
    sizes = np.broadcast_to(np.asarray(k, dtype=np.float64), means.shape)
    mu = means.mean(axis=0)
    var = mu + (means.var(axis=0) if means.shape[0] > 1 else 0.0)
    var = var + np.mean(np.where(sizes >= POISSON_K, 0.0, means**2 / sizes), axis=0)
    top = int(np.max(np.maximum(n, mu + 12.0 * np.sqrt(var) + 20.0)))
    out = np.zeros(len(n))
    for j0 in range(0, top + 1, 256):
        j = np.arange(j0, min(j0 + 256, top + 1), dtype=np.float64)
        f = mixture_cdf(j[:, None], means[:, None, :], sizes[:, None, :])  # (j, days)
        step = (n[None, :] <= j[:, None]).astype(np.float64)
        out += np.sum((f - step) ** 2, axis=0)
    return out


def central_interval_hit(observed: Any, means: FloatArray, k: Any, level: float = 0.9) -> float:
    """Share of days whose count falls inside the forecast's central ``level`` interval."""
    n = np.asarray(observed, dtype=np.float64)
    lo_t, hi_t = (1 - level) / 2, 1 - (1 - level) / 2
    below = mixture_cdf(n - 1, means, k)  # P(N < n)
    at_or_below = mixture_cdf(n, means, k)
    return float(np.mean((at_or_below >= lo_t) & (below <= hi_t)))


def fit_dispersion(observed: FloatArray, means: FloatArray) -> float:
    """The negative binomial size ``k`` that makes ``observed`` most likely given ``means``
    (same shape); k between 0.05 and 10⁶, Poisson if that is best."""

    def nll(log_k: float) -> float:
        return -float(np.sum(nb_logpmf(observed, means, float(np.exp(log_k)))))

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(1e6)), method="bounded")
    k = float(np.exp(res.x))
    if nll(float(np.log(POISSON_K))) <= res.fun:
        return POISSON_K
    return k


def fit_day_dispersion(observed: FloatArray, means: FloatArray) -> tuple[float, float]:
    """(κ, γ) of the day-total size k = κ · mean^γ that make ``observed`` most likely, γ in
    [0, 1.5]; Poisson (κ huge, γ 0) if nothing beats it."""

    def nll(v: Any) -> float:
        k = np.exp(v[0]) * np.maximum(means, 1e-12) ** np.clip(v[1], 0.0, 1.5)
        return -float(np.sum(nb_logpmf(observed, means, k)))

    starts = [(x0, g0) for x0 in (0.0, 2.0) for g0 in (0.0, 0.7)]
    fits = [optimize.minimize(nll, list(s), method="Nelder-Mead") for s in starts]
    best = min(fits, key=lambda r: float(r.fun))
    poisson = -float(np.sum(nb_logpmf(observed, means, POISSON_K)))
    if poisson <= best.fun:
        return POISSON_K, 0.0
    return float(np.exp(best.x[0])), float(np.clip(best.x[1], 0.0, 1.5))


def fit_cell_dispersion(hits: Any, means: FloatArray) -> float:
    """The size ``k`` that makes the cells' at-least-one-quake outcomes most likely."""
    hit = np.ravel(np.asarray(hits)).astype(bool)
    means = np.ravel(means)

    def nll(log_k: float) -> float:
        p0 = np.clip(nb_zero(means, float(np.exp(log_k))), 1e-300, 1 - 1e-16)
        return -float(np.sum(np.log(1 - p0[hit])) + np.sum(np.log(p0[~hit])))

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(1e6)), method="bounded")
    k = float(np.exp(res.x))
    if nll(float(np.log(POISSON_K))) <= res.fun:
        return POISSON_K
    return k
