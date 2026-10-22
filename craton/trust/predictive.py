"""A forecast as a full probability distribution, not just a rate.

CB-02 and CB-03 score a forecast as Poisson rates. A Poisson day has variance equal to its
mean, and real earthquake days don't: quakes come in clusters, so a day forecast at 10 often
brings 0 or 60 (the counts are *overdispersed*). The trust tasks need the whole distribution,
so a forecaster can say more than its rates:

    members      each ensemble member's rates: where they disagree, the model is unsure
                 (epistemic uncertainty, the deep-ensemble kind)
    dispersion   the negative binomial size ``k`` of a day's total and of one cell's count:
                 variance = mean + mean²/k (k → ∞ is Poisson); the clustering the rates alone
                 can't show (aleatoric uncertainty)
    abstain      days the model refuses to vouch for, because its context looks like nothing
                 it was trained on (``craton.trust.ood``)

A plain ``Forecaster`` is read as Poisson, one member, never abstaining: that is how every
floor gets a score on the trust tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import special

from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

#: Above this the negative binomial is indistinguishable from Poisson.
POISSON_K = 1e8


@dataclass(frozen=True)
class Predictive:
    """A forecast with its uncertainty. ``members`` (members, days, cells) are on the same
    scale as ``forecast.rates`` and share its magnitude shares; their mean is the forecast."""

    forecast: GriddedForecast
    members: FloatArray | None = None
    #: A day total's size is ``day_k · mean^day_gamma``: γ = 0 is the plain negative binomial
    #: (spread grows with the mean), γ = 1 a constant variance-to-mean ratio.
    day_k: float = POISSON_K
    day_gamma: float = 0.0
    cell_k: float = POISSON_K
    abstain: BoolArray | None = None

    def day_size(self, means: FloatArray) -> FloatArray:
        """The negative binomial size of each day total, for these means."""
        if self.day_k >= POISSON_K:
            return np.full(np.shape(means), POISSON_K)
        return np.asarray(self.day_k * np.maximum(means, 1e-12) ** self.day_gamma)

    @property
    def member_rates(self) -> FloatArray:
        """(members, days, cells); a single member when there is no ensemble."""
        if self.members is None:
            return self.forecast.rates[None]
        return self.members

    def member_expected(self, first_bin: int) -> FloatArray:
        """(members, days, cells) expected count in magnitude bins ``first_bin`` and up."""
        fc = self.forecast
        shares = np.broadcast_to(fc.mag_shares, (*fc.rates.shape, fc.mag_shares.shape[-1]))
        frac = shares[..., first_bin:].sum(axis=-1)
        return np.asarray(self.member_rates * frac[None], dtype=np.float64)

    def occurrence(self, first_bin: int) -> FloatArray:
        """(days, cells) probability of at least one quake in bins ``first_bin`` and up."""
        lam = self.member_expected(first_bin)
        return np.asarray(np.mean(1.0 - nb_zero(lam, self.cell_k), axis=0), dtype=np.float64)

    def day_totals(self, first_bin: int) -> FloatArray:
        """(members, days) each member's expected total for the day over the whole grid."""
        return np.asarray(self.member_expected(first_bin).sum(axis=-1), dtype=np.float64)


@runtime_checkable
class ProbabilisticForecaster(Protocol):
    name: str

    def predictive(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> Predictive: ...


def predictive_of(method: Any, spec: ForecastSpec, ctx: BenchContext, seed: int) -> Predictive:
    """The method's full forecast, computed once per (method, task, seed, context)."""
    key = f"predictive:{method.name}:{spec.task_id}:{seed}"
    if key not in ctx._cache:
        if isinstance(method, ProbabilisticForecaster):
            ctx._cache[key] = method.predictive(spec, ctx, seed)
        else:
            ctx._cache[key] = Predictive(method.forecast(spec, ctx, seed))
    out: Predictive = ctx._cache[key]
    return out


# ---------------------------------------------------------------- negative binomial


def _sizes(k: Any) -> tuple[FloatArray, Any]:
    """Sizes as an array, and where they mean Poisson."""
    kk = np.asarray(k, dtype=np.float64)
    pois = kk >= POISSON_K
    return np.where(pois, 1.0, kk), pois


def nb_zero(mean: FloatArray, k: Any) -> FloatArray:
    """P(N = 0) for a negative binomial with this mean and size ``k`` (Poisson if k is huge);
    ``k`` broadcasts against ``mean``."""
    mean = np.asarray(mean, dtype=np.float64)
    kk, pois = _sizes(k)
    nb = np.exp(-kk * np.log1p(mean / kk))
    return np.asarray(np.where(pois, np.exp(-mean), nb), dtype=np.float64)


def nb_logpmf(n: Any, mean: Any, k: Any) -> FloatArray:
    """log P(N = n), broadcasting ``n``, ``mean`` and ``k``."""
    n = np.asarray(n, dtype=np.float64)
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 1e-300)
    kk, pois = _sizes(k)
    poisson = n * np.log(mean) - mean - special.gammaln(n + 1)
    nb = (
        special.gammaln(n + kk)
        - special.gammaln(kk)
        - special.gammaln(n + 1)
        + kk * np.log(kk / (kk + mean))
        + n * np.log(mean / (kk + mean))
    )
    return np.asarray(np.where(pois, poisson, nb), dtype=np.float64)


def nb_cdf(n: Any, mean: Any, k: Any) -> FloatArray:
    """P(N ≤ n); n < 0 gives 0."""
    n = np.asarray(n, dtype=np.float64)
    mean = np.maximum(np.asarray(mean, dtype=np.float64), 1e-300)
    nn = np.floor(np.maximum(n, 0))
    kk, pois = _sizes(k)
    poisson = special.gammaincc(nn + 1, mean)
    # P(N ≤ n) = I_p(k, n + 1), the regularised incomplete beta, p = k / (k + mean).
    nb = special.betainc(kk, nn + 1, kk / (kk + mean))
    return np.asarray(np.where(n < 0, 0.0, np.where(pois, poisson, nb)), dtype=np.float64)
