"""P9h: dimensionless inputs start as ETAS, can't see a region's units, and CB-32's folds."""

from __future__ import annotations

import math

import numpy as np
import torch

from craton.bench.csep import Grid
from craton.bench.data import Events
from craton.bench.tasks import TASKS
from craton.bench.unseen import FOLDS, items
from craton.field.config import FieldDimless, field_hash, load_field_config
from craton.field.dimless import (
    DimlessFieldNet,
    footprint,
    omori_shift,
    rupture_km,
    rupture_memory,
)
from craton.field.memory import EPS, LIFETIMES_DAYS, MemorySpec, memory

GRID = Grid(42.0, 43.0, 13.0, 14.0)
DAY = 86_400.0
SPEC = MemorySpec(m_min=1.5, start=9e8, m_ref=2.0)
K = len(LIFETIMES_DAYS)


def _events(lat: list[float], lon: list[float], mag: list[float], t0: float = 1e9) -> Events:
    n = len(lat)
    return Events(
        time=t0 + np.arange(n, dtype=np.float64),
        lat=np.asarray(lat, np.float64),
        lon=np.asarray(lon, np.float64),
        depth_km=np.full(n, 10.0),
        mag=np.asarray(mag, np.float64),
        mag_type=np.full(n, "ML"),
        token_hash=np.array([f"h{i}" for i in range(n)]),
    )


def _net(dimless: FieldDimless, spreads: int = 4) -> DimlessFieldNet:
    shape = load_field_config("world_dimless").model
    torch.manual_seed(0)
    return DimlessFieldNet(spreads * K * 2, GRID, shape, dimless, LIFETIMES_DAYS).eval()


def _inputs(days: int = 2, spreads: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    counters = torch.rand(days, GRID.n_cells, spreads * K * 2, generator=g) * 5
    counters[:, ::3] = 0.0  # empty cells too
    mem = torch.log(EPS + counters)
    log_etas = torch.randn(days, GRID.n_cells, 2, generator=g) - 4
    return mem, log_etas


def test_the_untrained_dimensionless_field_is_etas() -> None:
    mem, le = _inputs()
    out = _net(FieldDimless())(mem, le, torch.tensor([0.3, -0.7]))
    assert out.shape == (2, GRID.n_cells, 2)
    assert torch.count_nonzero(out) == 0


def test_ratios_cannot_see_a_region_ten_times_busier() -> None:
    """Every rate and every counter ten times bigger: the inputs don't move."""
    net = _net(FieldDimless())
    mem, le = _inputs()
    s = torch.zeros(2)
    busier_mem = torch.log(EPS + 10.0 * (mem.exp() - EPS))
    a = net.features(mem, le, s)
    b = net.features(busier_mem, le + math.log(10.0), s)
    assert torch.allclose(a, b, atol=2e-3)
    raw = _net(FieldDimless(ratios=False))
    assert not torch.allclose(raw.features(mem, le, s), raw.features(busier_mem, le + 2.3, s))


def test_omori_time_slides_by_c() -> None:
    cfg = FieldDimless()
    assert omori_shift(cfg.c_ref_days, cfg, LIFETIMES_DAYS) == 0.0
    spacing = math.exp(np.mean(np.diff(np.log(LIFETIMES_DAYS))))
    assert math.isclose(omori_shift(cfg.c_ref_days * spacing, cfg, LIFETIMES_DAYS), 1.0)
    assert omori_shift(cfg.c_ref_days * 3, FieldDimless(omori_time=False), LIFETIMES_DAYS) == 0
    # A slide of one step reads every counter one lifetime longer, normalised by that lifetime.
    net = _net(FieldDimless(rupture=False), spreads=3)
    mem, le = _inputs(spreads=3)
    f0 = net.features(mem, le, torch.zeros(2))
    f1 = net.features(mem, le, torch.ones(2))
    bank0 = f0[..., :-1].reshape(2, GRID.n_cells, 3, K, 2)
    bank1 = f1[..., :-1].reshape(2, GRID.n_cells, 3, K, 2)
    held = (mem.exp() - EPS).clamp(min=0).reshape(2, GRID.n_cells, 3, K, 2)
    rate = le.logsumexp(-1).exp()[..., None, None, None]
    tau = torch.tensor(LIFETIMES_DAYS)[None, None, None, :, None]
    want = torch.log(1e-3 + held[..., 1:, :] / (rate * tau[..., 1:, :]))
    assert torch.allclose(bank1[..., :-1, :], want[...], atol=1e-4)
    assert not torch.allclose(bank0, bank1)


def test_a_small_quake_stays_in_its_cell_and_a_big_one_reaches_its_rupture() -> None:
    assert 40 < rupture_km(np.array([7.0]))[0] < 60  # Wells & Coppersmith: ~49 km
    small = _events([42.55], [13.55], [3.0])
    nights = np.array([1e9 + DAY])
    rup = rupture_memory(small, GRID, nights, SPEC)
    own = memory(small, GRID, nights, SPEC)[..., : K * 2]
    assert np.array_equal(rup, own)
    big = _events([42.55], [13.55], [6.8])
    r = np.exp(rupture_memory(big, GRID, nights, SPEC).astype(np.float64)) - EPS
    centre = 5 * GRID.n_lon + 5
    sigma = 0.5 * rupture_km(np.array([6.8]))[0]
    f = footprint(GRID, 42.55, 13.55, sigma)
    count = r[0, :, 3 * 2]  # the 4-day counter, every quake weighs 1
    assert math.isclose(count[centre], count.max(), rel_tol=1e-3)
    assert np.allclose(count / count[centre], f / f[centre], atol=2e-3)
    assert (count > 0.5 * count[centre]).sum() > 9  # the rupture covers many cells


def test_cb32_leaves_out_every_training_region_once() -> None:
    world = load_field_config("world")
    assert world.world is not None
    assert set(FOLDS) == set(world.world.regions)
    its = items()
    assert {it.region for it in its} == set(FOLDS)
    assert all(sum(it.region == r for it in its) == 2 for r in FOLDS)
    task = TASKS["CB-32"]

    class Model:
        kind = "model"

        def forecast(self, *_: object) -> None: ...

    assert not task.supports(Model())  # a model must say which region it never read


def test_the_dimless_config_keeps_every_old_hash() -> None:
    assert load_field_config("world").dimless is None
    assert field_hash(load_field_config("world"))[:8] == "aab7bfce"
    d = load_field_config("world_dimless").dimless
    assert d is not None
    assert d.pool_of == "world"
