"""P9g: the local-alpha floor refits ETAS's productivity exactly as ETAS's own likelihood says."""

from __future__ import annotations

import math

import numpy as np

from craton.bench import csep, etas
from craton.bench.local_alpha import (
    MIN_TARGETS,
    WINDOW_DAYS,
    _bins,
    fit_day,
    target_sums,
)

GRID = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=2.0)


def _fit_and_catalogue(
    params: etas.EtasParams, days: float = 200.0, seed: int = 3
) -> tuple[etas.EtasFit, etas.Catalog]:
    proj = etas.Projection.for_grid(GRID)
    box = etas.box_km(GRID, proj)
    cat = etas.simulate(params, 1.0, 2.0, days, box, seed=seed)
    # Magnitudes on a 0.05 grid, as the sums bin them, so the check below is exact.
    cat = etas.Catalog(cat.t, cat.x, cat.y, 2.0 + np.round((cat.m - 2.0) / 0.05) * 0.05)
    bg = etas.Background(cat.x, cat.y, np.ones(len(cat)), box)
    fit = etas.EtasFit(params, bg, 2.0, 1.0, 0.0, etas.FitWindow(0, days), len(cat), 0.0, True)
    return fit, cat


def test_sums_give_etas_rate_at_every_target() -> None:
    params = etas.EtasParams(0.5, 0.3, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6)
    fit, cat = _fit_and_catalogue(params)
    targets = np.nonzero(cat.t >= 100.0)[0]
    sums = target_sums(fit, cat, targets)
    pairs = etas.find_pairs(cat, targets)
    trig = np.bincount(
        pairs.target, weights=etas._pair_rates(params, cat, pairs, 2.0), minlength=len(targets)
    )
    via_sums = sums.by_bin @ (params.k * np.exp(params.a * sums.m_bins))
    np.testing.assert_allclose(via_sums, trig, rtol=1e-8, atol=1e-14)


def test_quiet_day_keeps_etas() -> None:
    params = etas.EtasParams(0.05, 0.1, 1.0, 0.01, 1.1, 1.0, 0.8, 1.6)
    fit, cat = _fit_and_catalogue(params, days=60.0)
    targets = np.nonzero(cat.t >= 0)[0]
    sums = target_sums(fit, cat, targets)
    bins, _ = _bins(cat.m - 2.0)
    assert len(targets) < MIN_TARGETS
    out = fit_day(fit, cat, sums, 60.0, bins)
    assert not out.refit
    assert out.k == params.k
    assert out.a == params.a


def test_refit_recovers_a_more_productive_month() -> None:
    """A catalogue made with three times ETAS's K: the refit finds K well above ETAS's own."""
    true = etas.EtasParams(4.0, 0.24, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6)
    _, cat = _fit_and_catalogue(true, days=200.0, seed=5)
    fit, _ = _fit_and_catalogue(etas.EtasParams(4.0, 0.08, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6))
    fit = etas.EtasFit(
        fit.params, fit.background, 2.0, 1.0, 0.0, etas.FitWindow(0, 200), len(cat), 0.0, True
    )
    day = 200.0
    targets = np.nonzero(cat.t >= day - WINDOW_DAYS)[0]
    assert len(targets) >= MIN_TARGETS
    sums = target_sums(fit, cat, targets)
    bins, _ = _bins(cat.m - 2.0)
    out = fit_day(fit, cat, sums, day, bins)
    assert out.refit
    assert out.k * math.exp(out.a * 0.5) > 2 * 0.08 * math.exp(1.2 * 0.5)
    only_a = fit_day(fit, cat, sums, day, bins, refit_k=False)
    assert only_a.k == fit.params.k
