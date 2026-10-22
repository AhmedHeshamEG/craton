"""The bit split, the ceiling oracles, anisotropic ETAS and the interpolation floor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from craton.bench import csep, etas
from craton.bench.baselines import _neighbour_line
from craton.bench.ceiling import oracle_gains
from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, GriddedForecast, score_forecast


def _toy(seed: int = 0, n: int = 300) -> tuple[GriddedForecast, Events, object]:
    spec = TASKS["CB-02"].spec  # type: ignore[attr-defined]
    g = spec.grid
    nd = len(spec.day_starts)
    rng = np.random.default_rng(seed)
    rates = rng.gamma(0.5, 0.01, (nd, g.n_cells))
    shares = csep.gutenberg_richter_bins(1.0, 1.5, g)
    lat, lon = g.cell_centres()
    ci = rng.integers(0, g.n_cells, n)
    t = np.sort(spec.window.t0 + rng.uniform(0, nd * DAY, n))
    m = rng.choice([1.5, 2.0, 3.0, 3.4, 4.0, 5.2], n)
    ev = Events(t, lat[ci], lon[ci], np.zeros(n), m, np.array(["ML"] * n), np.arange(n).astype(str))
    return GriddedForecast(rates, shares / shares.sum()), ev, spec


def test_bit_split_adds_up_to_the_score() -> None:
    fc, ev, spec = _toy()
    s = score_forecast(fc, ev, spec, 0)  # type: ignore[arg-type]
    for tag in ("M3", "M1.5"):
        parts = sum(
            s.extras[f"{tag}.{k}_bits"] for k in ("how_many", "where", "how_big", "shuffle")
        )
        assert parts == pytest.approx(s.extras[f"{tag}.bits_per_eq"], abs=1e-9)
    assert "M3.bits_without_big_days" in s.extras


def test_shuffle_bits_do_not_depend_on_the_forecast() -> None:
    fc, ev, spec = _toy()
    a = score_forecast(fc, ev, spec, 0).extras  # type: ignore[arg-type]
    other = GriddedForecast(fc.rates * 3.0 + 1e-3, fc.mag_shares)
    b = score_forecast(other, ev, spec, 0).extras  # type: ignore[arg-type]
    assert a["M3.shuffle_bits"] == pytest.approx(b["M3.shuffle_bits"])
    assert a["M3.where_bits"] != pytest.approx(b["M3.where_bits"])


def test_oracles_only_add_and_the_roof_is_the_sum() -> None:
    fc, ev, spec = _toy()
    gains = oracle_gains(fc, ev, spec)  # type: ignore[arg-type]
    assert gains["count"] >= 0
    assert gains["where_cell"] > gains["where_10km"] > 0
    assert gains["roof"] == pytest.approx(gains["count"] + gains["where_cell"])


def test_neighbour_line_interpolates_and_reports_the_gap() -> None:
    vt = np.array([0.0, 2.0, 5.0]) * DAY
    venu = np.array([[0.0, 0, 0], [2.0, 0, 0], [5.0, 0, 0]])
    enu, gap = _neighbour_line(np.array([1.0, 3.0, 7.0]) * DAY, vt, venu)
    assert enu[:, 0] == pytest.approx([1.0, 3.0, 5.0])
    assert gap == pytest.approx([2.0, 3.0, 2.0])


def test_learned_shape_follows_a_line_of_aftershocks() -> None:
    rng = np.random.default_rng(1)
    n = 60
    along = rng.uniform(-10, 10, n)
    angle = math.radians(30)
    x = np.concatenate([[0.0], along * math.cos(angle) + rng.normal(0, 0.5, n)])
    y = np.concatenate([[0.0], along * math.sin(angle) + rng.normal(0, 0.5, n)])
    t = np.concatenate([[0.0], np.sort(rng.uniform(0.001, 0.2, n))])
    m = np.concatenate([[6.0], np.full(n, 2.5)])
    shapes = etas.learn_shapes(etas.Catalog(t, x, y, m), mc=2.0)
    assert list(shapes.index) == [0]
    assert abs(math.cos(shapes.angle[0] - angle)) > 0.99
    assert shapes.rho[0] > 3
    assert shapes.ready[0] == pytest.approx(etas.SHAPE_HOURS / 24)


def test_an_ellipse_keeps_its_mass() -> None:
    grid = csep.Grid(42.0, 43.0, 13.0, 14.0, cell_deg=0.1, m_min=2.0)
    proj = etas.Projection.for_grid(grid)
    params = etas.EtasParams(1.0, 0.2, 1.5, 0.01, 1.1, 1.0, 1.0, 1.5)
    cat = etas.Catalog(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([4.0]))
    circle = etas.spatial_matrix(params, cat, 2.0, grid, proj)
    shapes = etas.Shapes(
        np.array([0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.5]),
        np.array([4.0]),
        np.array([0.0]),
    )
    ellipse = etas.spatial_matrix(params, cat, 2.0, grid, proj, shapes=shapes)
    assert ellipse.sum() == pytest.approx(circle.sum(), rel=0.05)
    assert (ellipse.toarray() > 0).sum() != (circle.toarray() > 0).sum()


def test_field_base_is_etas_exactly() -> None:
    import torch

    from craton.field.base import etas_daily

    grid = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=2.0)
    proj = etas.Projection.for_grid(grid)
    params = etas.EtasParams(0.5, 0.3, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6)
    box = etas.box_km(grid, proj)
    cat = etas.simulate(params, 1.0, 2.0, 200.0, box, seed=3)
    bg = etas.Background(cat.x, cat.y, np.ones(len(cat)), box)
    fit = etas.EtasFit(params, bg, 2.0, 1.0, 0.0, etas.FitWindow(0, 150), len(cat), 0.0, True)
    days = np.arange(150.0, 200.0)
    want = etas.forecast_days(fit, cat, days, grid, proj).rates
    got = etas_daily(fit, cat, days, grid, proj, torch.device("cpu"))
    assert got == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_blind_time_hides_small_quakes_when_the_rate_is_high() -> None:
    lo = np.array([2.0, 3.0, 4.0])
    hi = np.array([3.0, 4.0, np.inf])
    quiet = np.full((1, 1440), 5.0)
    busy = np.full((1, 1440), 50_000.0)
    tb = 60.0 / 86_400
    assert etas.detection_factors(quiet, 1.0, 2.0, lo, hi, tb) == pytest.approx(1.0, abs=0.01)
    f = etas.detection_factors(busy, 1.0, 2.0, lo, hi, tb)[0]
    assert f[0] < f[1] < f[2] <= 1.0  # small ones are missed most
    seen_total = (f * busy[0, 0] * np.array([0.9, 0.09, 0.01])).sum()
    assert seen_total <= 1.0 / tb + 1e-6  # never more than one quake per blind time
    assert np.all(etas.detection_factors(busy, 1.0, 2.0, lo, hi, 0.0) == 1.0)


def test_untrained_field_is_etas_and_the_nudge_is_bounded() -> None:
    import torch

    from craton.field.config import load_field_config
    from craton.field.model import FieldNet

    cfg = load_field_config("mini")
    grid = TASKS["CB-02"].spec.grid  # type: ignore[attr-defined]
    net = FieldNet(5, grid, cfg.model).eval()
    mem = torch.randn(2, grid.n_cells, 5)
    log_etas = torch.randn(2, grid.n_cells)
    assert torch.all(net(mem, log_etas) == 0)
    with torch.no_grad():
        net.out.bias.fill_(100.0)
    assert torch.all(net(mem, log_etas) <= math.log(cfg.model.max_nudge) + 1e-6)


def test_memory_fades_and_never_reads_the_future() -> None:
    from craton.field.memory import LIFETIMES_DAYS, MemorySpec, memory

    grid = csep.Grid(42.0, 42.3, 13.0, 13.3, cell_deg=0.1, m_min=1.5)
    t0 = 1_000_000_000.0
    ev = Events(
        np.array([t0 + 0.5 * DAY, t0 + 3.5 * DAY]),
        np.array([42.15, 42.15]),
        np.array([13.15, 13.15]),
        np.zeros(2),
        np.array([2.0, 2.0]),
        np.array(["ML", "ML"]),
        np.array(["a", "b"]),
    )
    mids = t0 + DAY * np.arange(1, 4)  # days 1, 2, 3: the second quake is after all of them
    out = memory(ev, grid, mids, MemorySpec(start=t0)).astype(np.float64)
    cell = int(grid.cell_of(np.array([42.15]), np.array([13.15]))[0])
    day_lifetime = LIFETIMES_DAYS.index(1.0)
    own = np.exp(out[:, cell, day_lifetime * 2])  # count weight, own cell, 1-day counter
    assert own[1] / own[0] == pytest.approx(math.exp(-1.0), rel=1e-2)
    assert own[2] / own[1] == pytest.approx(math.exp(-1.0), rel=1e-2)
