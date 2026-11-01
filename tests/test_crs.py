"""The Coulomb rate-and-state floor: geometry, stress and Dieterich's clock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from craton.bench.crs import (
    Orientation,
    Plane,
    StateBank,
    dcff_isotropic,
    dcff_rectangle,
    load_config,
    moment_nm,
    rectangle,
    rupture_km,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT)
NORMAL = Orientation(strike=155.0, dip=45.0, rake=-90.0)


def test_plane_vectors_are_orthonormal_and_a_normal_fault_drops_its_hanging_wall() -> None:
    p = Plane.of(NORMAL)
    for v in (p.strike, p.down_dip, p.normal, p.slip):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert p.strike @ p.down_dip == pytest.approx(0.0, abs=1e-12)
    assert p.normal @ p.strike == pytest.approx(0.0, abs=1e-12)
    assert p.normal @ p.down_dip == pytest.approx(0.0, abs=1e-12)
    assert p.slip @ p.normal == pytest.approx(0.0, abs=1e-12)  # slip lies in the plane
    assert p.slip[2] < 0  # hanging wall goes down
    assert p.normal[2] > 0  # the normal points up, into the hanging wall
    assert p.down_dip[2] < 0


def test_rupture_scaling_and_moment() -> None:
    length, width = rupture_km(6.5)
    assert 15 < length < 30
    assert 10 < width < 20
    assert moment_nm(6.0) / moment_nm(5.0) == pytest.approx(10**1.5)


def test_the_rectangle_slips_its_hanging_wall_the_right_way() -> None:
    import cutde.halfspace as hs

    p = Plane.of(NORMAL)
    tris, slips = rectangle(0.0, 0.0, 10.0, 6.0, p, CFG)
    centre = tris[0].mean(axis=0)
    pts = np.array([centre + 0.01 * p.normal, centre - 0.01 * p.normal])
    d = np.einsum("oktd,td->ok", hs.disp_matrix(pts, tris, CFG.poisson), slips)
    jump = d[0] - d[1]  # hanging wall minus footwall
    assert jump / np.linalg.norm(jump) == pytest.approx(p.slip, abs=1e-3)


def test_coulomb_lobes_of_a_normal_fault() -> None:
    """Off the ends of a normal fault, along strike, same-type faults are loaded; well off its
    side, in the hanging wall at depth, they are unloaded (the stress shadow)."""
    p = Plane.of(NORMAL)
    tris, slips = rectangle(0.0, 0.0, 8.0, 6.0, p, CFG)
    length, _ = rupture_km(6.0)
    tip = 0.5 * length + 3.0
    horiz = np.array([p.down_dip[0], p.down_dip[1], 0.0])
    horiz /= np.linalg.norm(horiz)
    pts = np.array(
        [
            [tip * p.strike[0], tip * p.strike[1], -8.0],
            [-tip * p.strike[0], -tip * p.strike[1], -8.0],
            [8.0 * horiz[0], 8.0 * horiz[1], -8.0],
        ]
    )
    dc = dcff_rectangle(pts, tris, slips, p, CFG)
    assert dc[0] > 0
    assert dc[1] > 0
    assert dc[0] == pytest.approx(dc[1], rel=1e-6)
    assert dc[2] < 0


def test_isotropic_source_falls_as_r_cubed_and_stops_at_its_radius() -> None:
    pts = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -11.0], [0.0, 0.0, -12.0], [0, 0, -10.0001]])
    dc = dcff_isotropic(pts, 0.0, 0.0, 10.0, 4.0, CFG)
    assert dc[1] / dc[2] == pytest.approx(8.0)
    assert dc[0] == dc[3]  # inside the source radius: capped
    assert 0 < dc[0] < CFG.stress_drop_mpa


def bank(a: float = 0.04, ta_years: float = 10.0) -> StateBank:
    return StateBank(np.array([a]), np.array([ta_years * 365.25]), np.zeros((1, 1)), 0.0)


def test_steady_state_stays_at_the_background() -> None:
    b = bank()
    b.decay_to(100.0)
    assert float(b.mean_rate(1.0)[0, 0]) == pytest.approx(1.0)


def test_a_step_up_bursts_then_fades_and_a_step_down_goes_quiet() -> None:
    b = bank()
    b.step(np.array([0.2]))  # 5 A·sigma up
    first = float(b.mean_rate(1.0)[0, 0])
    b.decay_to(30.0)
    later = float(b.mean_rate(1.0)[0, 0])
    assert first > later > 1.0
    assert first < np.exp(5.0)
    d = bank()
    d.step(np.array([-0.2]))
    assert float(d.mean_rate(1.0)[0, 0]) < 0.01


def test_the_day_average_matches_a_numerical_integral() -> None:
    b = bank(0.04, 1.0)
    b.step(np.array([0.3]))
    b.decay_to(0.5)
    g0 = np.exp(float(b.lg[0, 0]))
    ta = 365.25
    t = np.linspace(0.0, 1.0, 200_001)
    g = 1.0 + (g0 - 1.0) * np.exp(-t / ta)
    assert float(b.mean_rate(1.0)[0, 0]) == pytest.approx(np.trapezoid(1.0 / g, t), rel=1e-6)
