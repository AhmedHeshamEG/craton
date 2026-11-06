"""ETAS with a local alpha: the productivity refit on the running sequence, every day.

Lombardi & Marzocchi (2010) fitted ETAS on all of Italy for INGV's daily CSEP forecasts (α 1.3,
p 1.16, M2.5+) and found it undershoots the first part of the 2009 L'Aquila sequence. Their
reading: some parameters are local, and α there was 1.5 to 2.0 (Marzocchi & Lombardi 2009). A
quake's productivity K·e^{α(m − mc)} grows faster with magnitude inside that sequence than the
national average says, so its big shocks were followed by more quakes than ETAS expected.

This floor asks whether that alone is the Field's gain inside big sequences (D-121: most of the
gain is ETAS expecting too few quakes there). It keeps everything of the fitted ETAS (the
background, Omori's c and p, the spatial kernel) and, at every forecast midnight, refits only
the two productivity numbers (K, α) on the box's quakes of the last ``WINDOW_DAYS`` days, every
earlier quake acting as a parent. With fewer than ``MIN_TARGETS`` such quakes the day keeps
ETAS's own (K, α). The refit starts from ETAS's own fit and is bounded as that fit is.

The fit is a two-number maximum likelihood, made cheap by one pass on the GPU: every target
quake's triggering, summed by the parent's magnitude (``target_sums``), so the rate of target i
is μ·u_i + K·Σ_b S_ib·e^{α·m_b} for any (K, α), and so is the expected count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import minimize

from craton.bench import etas
from craton.bench.data import DAY
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: The running sequence: the box's quakes of the last month.
WINDOW_DAYS = 30.0
#: Fewer target quakes than this and the day keeps ETAS's own productivity.
MIN_TARGETS = 100
#: Parent magnitudes are summed in bins this wide (above mc).
MAG_BIN = 0.05


@dataclass(frozen=True)
class Sums:
    """Per target quake, its triggering summed by parent magnitude bin (density, per day·km²)."""

    t: FloatArray  # (targets,) days
    background: FloatArray  # (targets,) μ·u at the target
    by_bin: FloatArray  # (targets, bins)
    m_bins: FloatArray  # (bins,) magnitude above mc at each bin's centre


def _bins(m_rel: FloatArray) -> tuple[np.ndarray, FloatArray]:
    idx = np.clip(np.round(m_rel / MAG_BIN).astype(np.int64), 0, None)
    n = int(idx.max()) + 1 if len(idx) else 1
    return idx, np.asarray(np.arange(n) * MAG_BIN, dtype=np.float64)


def target_sums(
    fit: etas.EtasFit, cat: etas.Catalog, targets: np.ndarray, chunk: int = 256
) -> Sums:
    """Σ over earlier parents within R_MAX of Omori density × spatial density, by parent bin.

    Everything but K·e^{α m} of ETAS's triggering rate, so the rate at each target under any
    (K, α) is one dot product. On the GPU when there is one.
    """
    import torch

    pr = fit.params
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_rel = cat.m - fit.mc
    idx, m_bins = _bins(m_rel)
    nb = len(m_bins)
    t = torch.tensor(cat.t, device=dev, dtype=torch.float64)
    x = torch.tensor(cat.x, device=dev, dtype=torch.float64)
    y = torch.tensor(cat.y, device=dev, dtype=torch.float64)
    sigma = torch.tensor(pr.d * np.exp(pr.gamma * m_rel), device=dev, dtype=torch.float64)
    trunc = 1 - (1 + etas.R_MAX**2 / sigma) ** (1 - pr.q)
    onehot = torch.zeros((len(cat), nb), device=dev, dtype=torch.float64)
    onehot[torch.arange(len(cat), device=dev), torch.tensor(idx, device=dev)] = 1.0
    omori_norm = (pr.p - 1) * pr.c ** (pr.p - 1)
    out = np.zeros((len(targets), nb))
    for s in range(0, len(targets), chunk):
        tj = torch.tensor(targets[s : s + chunk], device=dev)
        hi = int(targets[s : s + chunk].max())  # parents strictly before the latest target
        dt = t[tj, None] - t[None, :hi]
        r2 = (x[tj, None] - x[None, :hi]) ** 2 + (y[tj, None] - y[None, :hi]) ** 2
        sg = sigma[None, :hi]
        g = omori_norm * (dt.clamp(min=0) + pr.c) ** (-pr.p)
        g = g * (pr.q - 1) / (math.pi * sg) * (1 + r2 / sg) ** (-pr.q) / trunc[None, :hi]
        g = torch.where((dt > 0) & (r2 <= etas.R_MAX**2), g, torch.zeros_like(g))
        out[s : s + chunk] = (g @ onehot[:hi]).cpu().numpy()
    u = fit.background.density(cat.x[targets], cat.y[targets])
    return Sums(cat.t[targets], pr.mu * u, out, m_bins)


@dataclass(frozen=True)
class DayFit:
    k: float
    a: float
    n_targets: int
    refit: bool


def fit_day(
    fit: etas.EtasFit,
    cat: etas.Catalog,
    sums: Sums,
    day: float,
    bins: np.ndarray,
    refit_k: bool = True,
) -> DayFit:
    """(K, α) by maximum likelihood on the quakes of [day − WINDOW_DAYS, day); with
    ``refit_k`` off, α alone (K stays ETAS's own)."""
    pr = fit.params
    sel = (sums.t >= day - WINDOW_DAYS) & (sums.t < day)
    n = int(sel.sum())
    if n < MIN_TARGETS:
        return DayFit(pr.k, pr.a, n, False)
    s_ib = sums.by_bin[sel]
    bg = sums.background[sel]
    mb = sums.m_bins
    # Each earlier parent's expected offspring inside the window, by bin (without K e^{αm}).
    before = cat.t < day
    tp = cat.t[before]
    h = etas.omori_integral(day - tp, pr.c, pr.p) - etas.omori_integral(
        day - WINDOW_DAYS - tp, pr.c, pr.p
    )
    i_b = np.bincount(bins[before], weights=h, minlength=len(mb))[: len(mb)]
    mu_w = pr.mu * WINDOW_DAYS

    def nll(v: FloatArray) -> tuple[float, FloatArray]:
        k, a = math.exp(v[0]), float(v[1])
        e = np.exp(a * mb)
        aa = s_ib @ e
        bb = s_ib @ (mb * e)
        lam = bg + k * aa
        ie, ime = float(i_b @ e), float(i_b @ (mb * e))
        f = -float(np.sum(np.log(lam))) + mu_w + k * ie
        g0 = -float(np.sum(k * aa / lam)) + k * ie
        g1 = -float(np.sum(k * bb / lam)) + k * ime
        return f / n, np.array([g0, g1]) / n

    res = minimize(
        nll,
        np.array([math.log(pr.k), pr.a]),
        jac=True,
        method="L-BFGS-B",
        bounds=[etas.BOUNDS[1] if refit_k else (math.log(pr.k),) * 2, etas.BOUNDS[2]],
    )
    k = math.exp(res.x[0]) if refit_k else pr.k
    return DayFit(k, float(res.x[1]), n, True)


@dataclass
class EtasLocalAlpha:
    """ETAS whose productivity (K, α) is refit every day on the last month's quakes."""

    name: str = "etas_local_alpha"
    kind: str = "baseline"
    long_periods: ClassVar[bool] = False
    #: Off: α alone is refit, K stays ETAS's own (the ablation in D-126).
    refit_k: bool = True

    def day_fits(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> list[DayFit]:
        """One (K, α) per forecast day; cached with the forecast."""
        return list(self._run(spec, ctx, seed)[1])

    def _run(
        self, spec: ForecastSpec, ctx: BenchContext, seed: int
    ) -> tuple[GriddedForecast, list[DayFit]]:
        key = f"etas-local-alpha:{self.refit_k}:{spec.task_id}:{spec.fit_end}:{seed}"
        if key in ctx._cache:
            out: tuple[GriddedForecast, list[DayFit]] = ctx._cache[key]
            return out
        from craton.bench.baselines import Etas, _shares

        f = Etas().fit(spec, ctx, seed)
        pr = f.params
        proj = etas.Projection.for_grid(spec.grid)
        hist = ctx.events.window(f.t_ref, spec.window.t1).above(f.mc)
        cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, f.t_ref)
        days = (spec.day_starts - f.t_ref) / DAY
        targets = np.nonzero((cat.t >= days[0] - WINDOW_DAYS) & (cat.t < days[-1]))[0]
        sums = target_sums(f, cat, targets)
        bins, _ = _bins(cat.m - f.mc)
        fits = [fit_day(f, cat, sums, float(d), bins, self.refit_k) for d in days]

        smat = etas.spatial_matrix(pr, cat, f.mc, spec.grid, proj)
        k = np.array([x.k for x in fits])[:, None]
        a = np.array([x.a for x in fits])[:, None]
        t0 = days[:, None]
        span = spec.period_days
        w = (
            k
            * np.exp(a * (cat.m[None, :] - f.mc))
            * (
                etas.omori_integral(t0 + span - cat.t[None, :], pr.c, pr.p)
                - etas.omori_integral(t0 - cat.t[None, :], pr.c, pr.p)
            )
        )
        w = np.where(cat.t[None, :] < t0, w, 0.0)
        triggered = np.asarray((smat.T @ w.T).T)
        background = span * pr.mu * f.background.cell_mass(spec.grid, proj)
        factor = np.array(
            [
                etas.cascade_factor(
                    etas.EtasParams(**{**pr.as_dict(), "k": x.k, "a": x.a}), f.b, f.mc, span
                )
                for x in fits
            ]
        )[:, None]
        rates = (background[None, :] + triggered) * factor
        scale, shares = _shares(f.b, f.mc, spec.grid)
        result = (GriddedForecast(rates * scale, shares), fits)
        ctx._cache[key] = result
        return result

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return self._run(spec, ctx, seed)[0]
