"""P9f: the stress meter starts as ETAS, obeys Dieterich; the planet's record moves no quake."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from craton.bench.csep import Grid
from craton.field.config import FieldMeter, field_hash, load_field_config
from craton.field.meter import MeterFieldNet, log_expm1, meter_log_rate, scan
from craton.sim import SIMULATORS, SimSpec
from craton.sim import rate_state as rs

GRID = Grid(42.0, 43.0, 13.0, 14.0)
R, A_SIGMA, TAU_DOT = 0.05, 0.05, 0.01 / 365.25  # per day, MPa, MPa per day


def test_log_expm1_is_stable_both_ends() -> None:
    a = torch.tensor([1e-30, 1e-6, 0.5, 1.0, 3.0, 50.0, 1e4], dtype=torch.float64)
    want = torch.tensor([np.log(np.expm1(x)) if x < 700 else x for x in a.tolist()])
    assert torch.allclose(log_expm1(a), want.double(), rtol=1e-9, atol=1e-9)


def test_zero_stress_and_no_deficit_is_etas() -> None:
    log_etas = torch.linspace(-30.0, 6.0, 200, dtype=torch.float64)
    log_u = torch.full_like(log_etas, -1.5)
    zero = torch.zeros_like(log_etas)
    got = meter_log_rate(log_etas, zero, zero, log_u)
    assert torch.allclose(got, log_etas, atol=1e-9)


def _scan_counts(step_sigma: float, days: int, t_a: float) -> np.ndarray:
    """Daily counts after a stress step at t = 0 on a population at steady state: ETAS here is
    the steady background, the stress anomaly the step, held from then on."""
    cfg = load_field_config("mini")
    meter = FieldMeter(stress=True, t_a_years=t_a / 365.25, t_a_factor=2.0, max_stress=10.0)
    net = MeterFieldNet(4, GRID, cfg.model, meter, np.full(GRID.n_cells, np.log(R)))
    log_total = torch.full((days, GRID.n_cells), float(np.log(R)))
    delta = torch.full((days, GRID.n_cells), step_sigma)
    deficit, _ = scan(net, delta, log_total, torch.zeros(GRID.n_cells))
    with torch.no_grad():
        lam = meter_log_rate(log_total, delta, deficit, net.log_u()).exp()
    return np.asarray(lam[:, 0].double().numpy())


@pytest.mark.parametrize("step", [0.1, 0.3, -0.1])
def test_stress_step_matches_dieterich_closed_form(step: float) -> None:
    """The meter's day-by-day walk counts what eq. 12 of Dieterich (1994) integrates to, for a
    step up (aftershocks, Omori decay) and down (a stress shadow, then recovery)."""
    t_a = A_SIGMA / TAU_DOT
    got = _scan_counts(step / A_SIGMA, 3000, t_a)
    for d in (0, 1, 10, 100, 1000, 2999):
        want = quad(rs.dieterich_rate, d, d + 1, args=(R, step, A_SIGMA, TAU_DOT))[0]
        assert got[d] == pytest.approx(want, rel=2e-3), d


def test_quakes_beyond_etas_use_up_the_faults() -> None:
    """A stress pulse that ends leaves the rate below ETAS's, recovering over about t_a."""
    cfg = load_field_config("mini")
    meter = FieldMeter(stress=True, t_a_years=1.0)
    net = MeterFieldNet(4, GRID, cfg.model, meter, np.full(GRID.n_cells, np.log(R)))
    days = 2000
    log_total = torch.full((days, GRID.n_cells), float(np.log(R)))
    delta = torch.zeros(days, GRID.n_cells)
    delta[:30] = 2.0
    deficit, _ = scan(net, delta, log_total, torch.zeros(GRID.n_cells))
    with torch.no_grad():
        lam = meter_log_rate(log_total, delta, deficit, net.log_u()).exp()[:, 0].numpy()
    assert lam[0] > 5 * R
    assert lam[40] < 0.8 * R  # the shadow after the pulse
    assert lam[40] < lam[400] < lam[1999] <= R * (1 + 1e-4)
    assert lam[1999] == pytest.approx(R, rel=0.02)


def test_untrained_meter_field_is_etas_and_moves_both_bands_alike() -> None:
    cfg = load_field_config("mini")
    torch.manual_seed(0)
    log_r = np.random.default_rng(0).normal(-6.0, 1.0, GRID.n_cells)
    net = MeterFieldNet(10, GRID, cfg.model, FieldMeter(stress=True), log_r).eval()
    mem = torch.randn(3, GRID.n_cells, 10)
    lt = torch.randn(3, GRID.n_cells) - 3.0
    with torch.no_grad():
        zero = net(mem, lt - 1.0, lt, torch.zeros(3, GRID.n_cells))
        assert torch.allclose(zero, torch.zeros_like(zero), atol=1e-5)
        torch.nn.init.normal_(net.stress.weight)
        moved = net(mem, lt - 1.0, lt, torch.rand(3, GRID.n_cells))
    assert moved.abs().max() > 1e-2
    assert torch.equal(moved[..., 0], moved[..., 1])


def test_meter_config_is_new_and_mini_keeps_its_hash() -> None:
    mini, meter = load_field_config("mini"), load_field_config("meter")
    assert field_hash(mini)[:8] == "b99e49ae"
    assert meter.meter is not None
    assert meter.meter.stress
    assert meter.model_dump(exclude={"name", "description", "meter"}) == mini.model_dump(
        exclude={"name", "description", "meter"}
    )


def test_recording_stress_changes_no_quake() -> None:
    sim = SIMULATORS["rate_state"]
    spec = SimSpec(box=(41.0, 44.0, 11.5, 15.0), days=200.0)
    rec = rs.StressRecord(at=np.array([0.0, 50.0, 199.0]))
    a, b = sim.simulate(spec, seed=3), sim.simulate(spec, seed=3, record=rec)
    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.lat, b.lat)
    np.testing.assert_array_equal(a.mag, b.mag)
    assert rec.state.shape == rec.coulomb.shape == (3, len(rec.patches))  # type: ignore[arg-type]
    assert np.isfinite(rec.state).all()
    assert not np.array_equal(rec.state[0], rec.state[-1])  # quakes moved it


def test_without_depletion_the_deficit_is_always_zero() -> None:
    cfg = load_field_config("mini")
    meter = FieldMeter(stress=True, deplete=False)
    net = MeterFieldNet(4, GRID, cfg.model, meter, np.full(GRID.n_cells, np.log(R)))
    log_total = torch.full((50, GRID.n_cells), float(np.log(R)))
    delta = torch.full((50, GRID.n_cells), 2.0)
    deficit, _ = scan(net, delta, log_total, torch.zeros(GRID.n_cells))
    assert torch.equal(deficit, torch.zeros_like(deficit))
    with torch.no_grad():
        c = net.correction(delta, log_total, torch.ones_like(delta) * 5.0)
    assert torch.allclose(c[..., 0], c[0, 0, 0].expand_as(c[..., 0]))  # the deficit is ignored
    assert float(c[0, 0, 0]) > 1.0
