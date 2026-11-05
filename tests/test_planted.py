"""CB-33: the grade, the cells' state from the patches', and a floor that only reads the past."""

from __future__ import annotations

import numpy as np
import pytest

from craton.bench import planted
from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, forecast_grid
from craton.sim.rate_state import Patches, StressRecord


def test_grade_is_rank_correlation_on_off_steady_cells_only() -> None:
    rng = np.random.default_rng(0)
    truth = np.zeros((4, 50))
    truth[:, :20] = rng.normal(0, 1, (4, 20))
    truth[:, :20] += np.sign(truth[:, :20]) * 0.2  # all off steady
    perfect = planted.grade(truth, np.exp(truth))  # any monotone reading
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["cells_per_day"] == 20
    # what an answer says about the steady cells doesn't count
    noisy = np.exp(truth)
    noisy[:, 20:] = rng.normal(0, 100, (4, 30))
    assert planted.grade(truth, noisy)["spearman"] == pytest.approx(1.0)
    assert planted.grade(truth, -truth)["spearman"] == pytest.approx(-1.0)
    assert planted.grade(truth, np.ones_like(truth))["spearman"] == 0.0


def test_cell_state_weights_patches_as_the_cells_quakes_see_them() -> None:
    g = forecast_grid()
    lat0, lon0 = g.lat_min + 0.05, g.lon_min + 0.05
    p = Patches(
        lat=np.array([lat0, lat0, lat0 + 0.1]),
        lon=np.array([lon0, lon0, lon0]),
        strike=np.zeros(3),
        r=np.array([3.0, 1.0, 1.0]),
        tau_dot=np.ones(3),
        a_sigma=np.ones(3),
        m_max=np.full(3, 6.0),
        on_fault=np.ones(3, bool),
    )
    rec = StressRecord(at=np.zeros(1), patches=p)
    s = np.array([0.0, 2.0, -1.0])
    rec.gamma = np.exp(-s)[None]  # state = ln(1 / (gamma tau_dot)) = s
    out = planted.cell_state(rec)
    c0 = g.cell_of(np.array([lat0]), np.array([lon0]))[0]
    c1 = g.cell_of(np.array([lat0 + 0.1]), np.array([lon0]))[0]
    assert out[0, c0] == pytest.approx(np.log((3 + np.exp(2.0)) / 4))
    assert out[0, c1] == pytest.approx(-1.0)
    assert np.count_nonzero(out) == 2


def test_quake_density_reads_only_quakes_before_each_midnight() -> None:
    g = forecast_grid()
    t_mid = 1.5e9
    n = 40
    lat = np.full(n, g.lat_min + 0.05)
    lon = np.full(n, g.lon_min + 0.05)
    times = np.concatenate([t_mid - np.arange(1, 21) * DAY, t_mid + np.arange(20) * 3600.0])
    ev = Events(
        time=np.sort(times),
        lat=lat,
        lon=lon,
        depth_km=np.full(n, 8.0),
        mag=np.full(n, 2.0),
        mag_type=np.full(n, "sim"),
        token_hash=np.array([f"q{i}" for i in range(n)]),
    )
    q = planted.PlantedQuery(ev, np.array([t_mid]))
    before = planted.QuakeDensity().planted_state(q, 0)
    q_cut = planted.PlantedQuery(ev.select(ev.time < t_mid), np.array([t_mid]))
    assert np.array_equal(before, planted.QuakeDensity().planted_state(q_cut, 0))


def test_cb33_is_registered_and_takes_forecasters_and_readers() -> None:
    from craton.bench.baselines import METHODS

    task = TASKS["CB-33"]
    assert task.supports(METHODS["etas"])
    assert task.supports(METHODS["quake_density"])
    assert not task.supports(METHODS["stress_regional"])
