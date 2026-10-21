from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from craton.bench import csep
from craton.bench.baselines import METHODS
from craton.bench.data import Events
from craton.bench.dists import Gaussian, Histogram
from craton.bench.splits import Window

if TYPE_CHECKING:
    from craton.bench.tasks import RegionQueries

from craton.bench.tasks import (
    CB01_BINS,
    TASKS,
    Bin,
    ForecastSpec,
    GriddedForecast,
    score_forecast,
    score_gapfill,
)


def test_every_task_has_at_least_one_baseline() -> None:
    baselines = [m for m in METHODS.values() if m.kind == "baseline"]
    for task in TASKS.values():
        if task.task_id == "CB-15":
            continue  # it measures a training run; its floor is the 6 GiB budget
        assert any(task.supports(m) for m in baselines), task.task_id


def test_bins_put_values_inside_their_bin() -> None:
    b = Bin("magnitude", 0.1, 0.05)
    lo, hi = b.edges_of(np.array([3.0, 3.04, 6.5]))
    assert lo == pytest.approx([2.95, 2.95, 6.45])
    assert hi == pytest.approx([3.05, 3.05, 6.55])
    lat = Bin("lat", 0.01)
    lo, hi = lat.edges_of(np.array([42.97]))
    assert lo[0] <= 42.97 < hi[0]
    assert {b.channel for b in CB01_BINS["gnss_daily"]} == {"east", "north", "up"}


def test_gaussian_mass_is_exact_far_from_zero() -> None:
    g = Gaussian(np.array([4_692_539.9239]), np.array([0.01]))
    lo = np.array([4_692_539.923])
    m = g.mass(lo, lo + 0.001)
    assert m[0] == pytest.approx(0.0399, abs=1e-3)
    tail = Gaussian(np.array([0.0]), np.array([1.0])).mass(np.array([6.0]), np.array([6.001]))
    assert 0 < tail[0] < 1e-10


def test_histogram_mass() -> None:
    h = Histogram(np.array([0.0, 1.0, 2.0]), np.array([0.25, 0.75]))
    assert h.mass(np.array([0.0, 1.0, 0.5]), np.array([1.0, 2.0, 1.5])) == pytest.approx(
        [0.25, 0.75, 0.5]
    )
    assert h.mass(np.array([5.0]), np.array([6.0]))[0] == 0.0
    per_query = Histogram(np.array([0.0, 1.0, 2.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert per_query.mass(np.array([0.0, 0.0]), np.array([1.0, 1.0])).tolist() == [1.0, 0.0]


def test_gapfill_ignores_a_constant_offset() -> None:
    station = np.array(["A"] * 4 + ["B"] * 4)
    truth = np.random.default_rng(0).normal(0, 0.01, (8, 3))
    shifted = truth + np.array([5.0, -3.0, 100.0])
    assert score_gapfill(station, truth, shifted).value == pytest.approx(0.0, abs=1e-9)
    noisy = truth.copy()
    noisy[:, 0] += np.array([0.001, -0.001] * 4)
    s = score_gapfill(station, truth, noisy)
    assert s.extras["rmse_east_mm"] == pytest.approx(1.0)
    assert s.value == pytest.approx(math.sqrt(1 / 3))


def _events(times: list[float], lats: list[float], mags: list[float]) -> Events:
    n = len(times)
    return Events(
        np.array(times),
        np.array(lats),
        np.full(n, 12.05),
        np.full(n, 10.0),
        np.array(mags),
        np.array(["ML"] * n),
        np.array([f"h{i}" for i in range(n)]),
    )


def test_forecast_score_is_the_joint_poisson_likelihood() -> None:
    grid = csep.Grid(41.0, 41.2, 12.0, 12.1, m_min=1.5, m_max=4.0)
    spec = ForecastSpec(
        "CB-99", "x", Window("w", "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z"), grid
    )
    rng = np.random.default_rng(3)
    rates = rng.uniform(0.5, 2.0, (2, grid.n_cells))
    shares = csep.gutenberg_richter_bins(1.0, 1.5, grid)
    fc = GriddedForecast(rates, shares)
    t0 = spec.window.t0
    obs = _events([t0 + 10, t0 + 20, t0 + 86_400 + 5], [41.05, 41.05, 41.15], [3.0, 3.0, 1.5])
    s = score_forecast(fc, obs, spec, seed=0)
    for thr in (3.0, 1.5):
        first = int(grid.mag_bin_of(np.array([thr]))[0])
        lam = rates[:, :, None] * shares[None, None, :]
        counts = np.zeros_like(lam)
        for ev in range(3):
            if obs.mag[ev] >= thr:
                d = int((obs.time[ev] - t0) // 86_400)
                counts[
                    d,
                    grid.cell_of(obs.lat[ev : ev + 1], obs.lon[ev : ev + 1])[0],
                    grid.mag_bin_of(obs.mag[ev : ev + 1])[0],
                ] += 1
        ll = csep.poisson_ll(lam[..., first:], counts[..., first:])
        n = int(counts.sum())
        assert s.extras[f"M{thr:g}.bits_per_eq"] == pytest.approx(ll / n / math.log(2))
    assert s.value == s.extras["M3.bits_per_eq"]
    assert s.extras["M3.observed"] == 2
    assert s.extras["M1.5.observed"] == 3


def test_a_sharper_true_forecast_scores_higher() -> None:
    grid = csep.Grid(41.0, 41.3, 12.0, 12.1, m_min=1.5, m_max=4.0)
    spec = ForecastSpec(
        "CB-99", "x", Window("w", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"), grid
    )
    shares = csep.gutenberg_richter_bins(1.0, 1.5, grid)
    obs = _events([spec.window.t0 + 1] * 3, [41.05] * 3, [3.0, 3.1, 3.2])
    flat = GriddedForecast(np.full((1, 3), 1.0), shares)
    sharp = GriddedForecast(np.array([[2.8, 0.1, 0.1]]), shares)
    assert score_forecast(sharp, obs, spec, 0).value > score_forecast(flat, obs, spec, 0).value


def test_ids_and_names_are_never_reused() -> None:
    from craton.bench.baselines import register_method
    from craton.bench.tasks import register_task

    with pytest.raises(ValueError, match="new ID"):
        register_task(TASKS["CB-04"])
    with pytest.raises(ValueError, match="already registered"):
        register_method(METHODS["etas"])


# ---------------------------------------------------------------- CB-17


def _region(n_ctx: int = 400, n_test: int = 50, seed: int = 0) -> RegionQueries:
    from craton.bench.tasks import CB17_START, RegionQueries
    from craton.spec.timeaxis import to_seconds

    rng = np.random.default_rng(seed)
    t0 = to_seconds(CB17_START)
    fault = lambda n: np.stack(  # noqa: E731 - every quake on one small fault
        [43.30 + rng.normal(0, 0.01, n), 12.30 + rng.normal(0, 0.01, n),
         rng.uniform(4, 8, n), rng.uniform(0.5, 2.0, n)], 1
    )  # fmt: skip
    return RegionQueries(
        context_time=np.sort(t0 + rng.uniform(0, 365 * 86400.0, n_ctx)),
        context_values=fault(n_ctx),
        context_type=["ML"] * n_ctx,
        time=np.sort(t0 + 366 * 86400.0 + rng.uniform(0, 300 * 86400.0, n_test)),
        magnitude_type=["ML"] * n_test,
    )


def test_region_context_never_reaches_past_its_days() -> None:
    from craton.bench.tasks import CB17_START
    from craton.spec.timeaxis import to_seconds

    q = _region()
    for days in (0, 7, 90, 365):
        t, v, types = q.context(days)
        assert len(t) == len(v) == len(types)
        assert np.all(t < to_seconds(CB17_START) + days * 86400.0)
    assert len(q.context(0)[0]) == 0


def test_counting_the_context_beats_ignoring_it_on_a_fault() -> None:
    """The context KDE learns where a region's fault is; its answer sharpens with N. A Histogram
    over the region, fed through CB-17's own scoring arithmetic."""
    from craton.bench.baselines import LAT_EDGES, ContextKde, _kernel_hist

    q = _region()
    kde = ContextKde()
    truth = np.full(50, 43.30)
    lo, hi = Bin("lat", 0.01).edges_of(truth)
    flat = Histogram(LAT_EDGES, np.full(len(LAT_EDGES) - 1, 1.0 / (len(LAT_EDGES) - 1)))
    _, values, _ = q.context(365)
    k = _kernel_hist(values[:, 0], LAT_EDGES, 0.02)
    k = k / k.sum()
    blended = Histogram(
        LAT_EDGES, (len(values) * k + kde.prior * flat.probs) / (len(values) + kde.prior)
    )
    assert np.all(blended.mass(lo, hi) > 20 * flat.mass(lo, hi))
