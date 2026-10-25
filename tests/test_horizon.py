"""Forecasts over periods longer than a day: ETAS stretched to the period, the Field's answers
counted over it, and the score reading each quake into its period."""

from __future__ import annotations

import numpy as np
import pytest

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.splits import Window
from craton.bench.tasks import ForecastSpec, GriddedForecast, score_forecast


def _planet() -> tuple[etas.EtasFit, etas.Catalog, csep.Grid, etas.Projection]:
    grid = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=2.0)
    proj = etas.Projection.for_grid(grid)
    params = etas.EtasParams(0.5, 0.3, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6)
    box = etas.box_km(grid, proj)
    cat = etas.simulate(params, 1.0, 2.0, 150.0, box, seed=3)
    bg = etas.Background(cat.x, cat.y, np.ones(len(cat)), box)
    fit = etas.EtasFit(params, bg, 2.0, 1.0, 0.0, etas.FitWindow(0, 150), len(cat), 0.0, True)
    return fit, cat, grid, proj


def test_a_long_period_is_its_days_added_before_the_cascade() -> None:
    fit, cat, grid, proj = _planet()
    pr = fit.params
    span = etas.forecast_days(fit, cat, np.array([150.0]), grid, proj, span=30).rates[0]
    daily = etas.forecast_days(fit, cat, 150.0 + np.arange(30.0), grid, proj).rates
    f30 = etas.cascade_factor(pr, fit.b, fit.mc, 30.0)
    f1 = etas.cascade_factor(pr, fit.b, fit.mc, 1.0)
    assert f30 > f1 > 1.0
    # Nothing happens after day 150 in this catalogue, so no day reads a parent the period
    # can't: the direct part of thirty days is the thirty days' direct parts summed.
    assert span / f30 == pytest.approx((daily / f1).sum(axis=0), rel=1e-9)


def test_the_gpu_start_line_matches_etas_over_a_long_period() -> None:
    import torch

    from craton.field.base import etas_daily

    fit, cat, grid, proj = _planet()
    starts = np.array([100.0, 120.0, 140.0])
    want = etas.forecast_days(fit, cat, starts, grid, proj, span=30).rates
    got = etas_daily(fit, cat, starts, grid, proj, torch.device("cpu"), span=30)
    assert got == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_span_counts_count_every_quake_in_every_span_it_falls_in() -> None:
    from craton.field.forecaster import M_SCORED, _counts

    grid = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=1.5)
    rng = np.random.default_rng(1)
    n = 400
    t = np.sort(rng.uniform(0, 60 * DAY, n))
    lat = rng.uniform(41.9, 42.7, n)  # some fall outside the grid
    lon = rng.uniform(13.0, 13.6, n)
    mag = rng.choice([1.5, 2.0, 2.9, 3.0, 4.1], n)
    ev = Events(t, lat, lon, np.zeros(n), mag, np.array(["ML"] * n), np.arange(n).astype(str))
    starts = DAY * np.array([0.0, 1.0, 2.0, 17.0, 25.0, 30.0])
    got = _counts(ev, grid, starts, 2.0, span_days=30)
    cell = grid.cell_of(lat, lon)
    for i, s in enumerate(starts):
        sel = (t >= s) & (t < s + 30 * DAY) & (mag >= 2.0) & (cell >= 0)
        assert got[i, :, 1].sum() == ((mag >= M_SCORED) & sel).sum()
        assert got[i, :, 0].sum() == ((mag < M_SCORED) & sel).sum()
        assert np.array_equal(
            got[i].sum(axis=1), np.bincount(cell[sel], minlength=grid.n_cells).astype(np.float32)
        )


def test_the_score_reads_each_quake_into_its_period() -> None:
    grid = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=1.5)
    window = Window("w", "2020-01-01T00:00:00Z", "2020-03-01T00:00:00Z")  # 60 days
    spec = ForecastSpec("CB-xx", "cb02_fit", window, grid, period_days=30)
    assert len(spec.day_starts) == 2
    assert spec.day_starts[1] - spec.day_starts[0] == 30 * DAY
    shares = csep.gutenberg_richter_bins(1.0, 1.5, grid)
    shares = shares / shares.sum()
    lat, lon = grid.cell_centres()
    # One M3 quake on day 45, in cell 0: only period 1 expects it.
    ev = Events(
        np.array([window.t0 + 45.5 * DAY]),
        lat[:1],
        lon[:1],
        np.zeros(1),
        np.array([3.0]),
        np.array(["ML"]),
        np.array(["q"]),
    )
    rates = np.full((2, grid.n_cells), 1e-3)
    right, wrong = rates.copy(), rates.copy()
    right[1, 0] = 5.0
    wrong[0, 0] = 5.0
    good = score_forecast(GriddedForecast(right, shares), ev, spec, 0).value
    bad = score_forecast(GriddedForecast(wrong, shares), ev, spec, 0).value
    assert good > bad
