"""The fake planets: the physics planet checked against Dieterich (1994) in closed form."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from craton.sim import SIMULATORS, SimSpec
from craton.sim import rate_state as rs

A_SIGMA = 0.05
TAU_DOT = 0.01 / rs.YEAR  # MPa/day
R = 0.02  # quakes/day


def _one(x: float) -> np.ndarray:
    return np.array([x], dtype=np.float64)


def test_steady_state_rate_is_the_background() -> None:
    gamma = _one(1 / TAU_DOT)
    n = rs.expected_count(gamma, _one(TAU_DOT), 10.0, _one(A_SIGMA), _one(R / TAU_DOT))
    assert n[0] == pytest.approx(R * 10.0, rel=1e-9)
    after = rs.state_after(gamma, _one(TAU_DOT), 10.0, _one(A_SIGMA))
    assert after[0] == pytest.approx(1 / TAU_DOT, rel=1e-12)


@pytest.mark.parametrize("step", [0.2, 0.5, -0.1])
def test_stress_step_matches_dieterich_closed_form(step: float) -> None:
    """After a step, the exact integrator counts what eq. 12 of Dieterich (1994) integrates
    to, in every window from minutes to decades, for steps up and down (a stress shadow)."""
    gamma = 1 / TAU_DOT * np.exp(-step / A_SIGMA)
    edges = [0.0, 1 / 1440, 1 / 24, 1.0, 30.0, 365.0, 3650.0, 36500.0]
    g = _one(gamma)
    for t0, t1 in itertools.pairwise(edges):
        got = rs.expected_count(g, _one(TAU_DOT), t1 - t0, _one(A_SIGMA), _one(R / TAU_DOT))[0]
        want = quad(rs.dieterich_rate, t0, t1, args=(R, step, A_SIGMA, TAU_DOT), limit=200)[0]
        assert got == pytest.approx(want, rel=1e-6)
        g = rs.state_after(g, _one(TAU_DOT), t1 - t0, _one(A_SIGMA))


def test_aftershocks_decay_as_omori_with_p_near_one() -> None:
    step = 0.6
    ta = A_SIGMA / TAU_DOT
    t = np.logspace(-1, np.log10(ta / 30), 20)
    rate = rs.dieterich_rate(t, R, step, A_SIGMA, TAU_DOT)
    p = -np.polyfit(np.log(t), np.log(rate), 1)[0]
    assert 0.95 < p < 1.05


def test_shadow_rate_falls_below_background_and_recovers() -> None:
    rate = rs.dieterich_rate(np.array([1.0, 365.0, 20 * 3652.5]), R, -0.1, A_SIGMA, TAU_DOT)
    assert rate[0] < 0.2 * R
    assert rate[0] < rate[1] < rate[2]
    assert rate[2] == pytest.approx(R, rel=0.02)


def test_negative_and_zero_stressing_rates_stay_finite() -> None:
    gamma = _one(1 / TAU_DOT)
    for rate in (-5 * TAU_DOT, 0.0, 1e-30):
        n = rs.expected_count(gamma, _one(rate), 1.0, _one(A_SIGMA), _one(R / TAU_DOT))
        assert np.isfinite(n).all()
        assert n[0] > 0


def test_time_of_count_inverts_the_cumulative_count() -> None:
    gamma = np.full(5, 1 / TAU_DOT * np.exp(-0.4 / A_SIGMA))
    rate = np.array([TAU_DOT, -TAU_DOT, 0.0, 30 * TAU_DOT, TAU_DOT])
    a = np.full(5, A_SIGMA)
    t = np.array([0.01, 0.3, 1.0, 0.04, 2.0])
    integral = rs._cumulative(gamma, rate, t, a)
    assert rs.time_of_count(gamma, rate, a, integral) == pytest.approx(t, rel=1e-6)


def test_wells_coppersmith_round_trip() -> None:
    m = np.array([4.0, 5.5, 6.5, 7.0])
    assert rs.max_magnitude(rs.rupture_length_km(m)) == pytest.approx(m)
    assert rs.rupture_length_km(np.array([6.5]))[0] == pytest.approx(16.7, abs=0.2)


def test_stress_lobes_shadows_and_drop() -> None:
    """An M6 on a north-south fault: stress up off its ends, down beside it and on it."""
    planet = rs.RateStatePlanet(heterogeneity=0.0)
    lat = np.array([42.0, 42.15, 42.0, 42.0])
    lon = np.array([13.0, 13.0, 13.2, 13.0])
    p = rs.Patches(
        lat=lat,
        lon=lon,
        strike=np.zeros(4),
        r=np.ones(4),
        tau_dot=np.ones(4),
        a_sigma=np.ones(4),
        m_max=np.full(4, 7.0),
        on_fault=np.ones(4, bool),
    )
    ds = planet.stress_steps(
        p, _one(42.0), _one(13.0), _one(6.0), _one(0.0), np.random.default_rng(0)
    )
    # patch 0 is the rupture itself, 1 is 17 km along strike, 2 is 16 km across it
    assert ds[0] < 0
    assert ds[1] > 0
    assert ds[2] < 0
    assert ds[1] <= planet.max_step


def test_fluid_front_moves_out_like_root_t() -> None:
    planet = rs.RateStatePlanet()
    lat = 42.0 + np.array([0.0, 5.0, 10.0, 20.0]) / rs.KM_PER_DEG
    p = rs.Patches(
        lat=lat,
        lon=np.full(4, 13.0),
        strike=np.zeros(4),
        r=np.ones(4),
        tau_dot=np.ones(4),
        a_sigma=np.ones(4),
        m_max=np.full(4, 7.0),
        on_fault=np.ones(4, bool),
    )
    src = [rs.FluidSource(42.0, 13.0, t_on=0.0, t_off=1e9, p0=1.0)]
    early, late = planet.pressure(p, src, 30.0), planet.pressure(p, src, 3000.0)
    assert np.all(np.diff(early) < 0)
    assert np.all(late > early)
    off = [rs.FluidSource(42.0, 13.0, t_on=0.0, t_off=100.0, p0=1.0)]
    assert planet.pressure(p, off, 50_000.0).max() < 0.1


def test_physics_planet_is_deterministic_and_marked_synthetic() -> None:
    sim = SIMULATORS["rate_state"]
    spec = SimSpec(box=(41.0, 44.0, 11.5, 15.0), days=200.0)
    a, b = sim.simulate(spec, seed=3), sim.simulate(spec, seed=3)
    assert len(a) > 100
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.mag, b.mag)
    assert set(a.mag_type.tolist()) == {"sim"}
    assert all(h.startswith("sim-rate_state-3-") for h in a.token_hash[:5])
    assert np.all(np.diff(a.time) >= 0)
    assert a.mag.min() >= spec.m_min - 0.05
