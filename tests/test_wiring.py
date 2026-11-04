"""P9d: Omori attention starts as ETAS, and its shares are ETAS's own sum."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from craton.bench import csep, etas
from craton.field.base import etas_daily
from craton.field.config import FieldWiring, field_hash, load_field_config
from craton.field.model import FieldNet
from craton.field.wiring import BAND_MAG, WiredFieldNet, band_kernels, sources

GRID = csep.Grid(42.0, 42.6, 13.0, 13.6, cell_deg=0.1, m_min=2.0)
CPU = torch.device("cpu")


def _fit_and_catalogue() -> tuple[etas.EtasFit, etas.Catalog, etas.Projection]:
    proj = etas.Projection.for_grid(GRID)
    params = etas.EtasParams(0.5, 0.3, 1.2, 0.01, 1.1, 1.0, 0.8, 1.6)
    box = etas.box_km(GRID, proj)
    cat = etas.simulate(params, 1.0, 2.0, 200.0, box, seed=3)
    bg = etas.Background(cat.x, cat.y, np.ones(len(cat)), box)
    fit = etas.EtasFit(params, bg, 2.0, 1.0, 0.0, etas.FitWindow(0, 150), len(cat), 0.0, True)
    return fit, cat, proj


def _at_centres(cat: etas.Catalog, proj: etas.Projection, fit: etas.EtasFit) -> etas.Catalog:
    """The catalogue with every quake moved to its cell's centre and its band's magnitude:
    where the grouped shares are exact."""
    from craton.field.wiring import band_of, proj_inverse

    lat, lon = proj_inverse(proj, cat.x, cat.y)
    cell = GRID.cell_of(lat, lon)
    keep = cell >= 0
    clat, clon = GRID.cell_centres()
    x, y = proj.to_km(clat[cell[keep]], clon[cell[keep]])
    m = fit.mc + np.asarray(BAND_MAG)[band_of(cat.m[keep] - fit.mc)]
    return etas.Catalog(cat.t[keep], x, y, m)


def test_shares_add_up_to_etas_where_the_grouping_is_exact() -> None:
    fit, cat, proj = _fit_and_catalogue()
    cat = _at_centres(cat, proj, fit)
    days = np.arange(150.0, 200.0)
    src, out, bg = sources(fit, cat, days, GRID, proj, device=CPU)
    kern = band_kernels(fit, GRID, proj).astype(np.float64)
    total = np.einsum("djb,bjc->dc", src.astype(np.float64), kern) + out + bg[None]
    factor = etas.cascade_factor(fit.params, fit.b, fit.mc, 1.0)
    want = etas_daily(fit, cat, days, GRID, proj, CPU) / factor
    assert total == pytest.approx(want, rel=1e-4)


def test_off_grid_parents_are_exact() -> None:
    fit, _, proj = _fit_and_catalogue()
    far = etas.Catalog(np.array([100.0]), np.array([-60.0]), np.array([0.0]), np.array([4.0]))
    days = np.arange(101.0, 110.0)
    src, out, bg = sources(fit, far, days, GRID, proj, device=CPU)
    assert src.sum() == 0
    factor = etas.cascade_factor(fit.params, fit.b, fit.mc, 1.0)
    want = etas_daily(fit, far, days, GRID, proj, CPU) / factor
    assert out.max() > 1e-3 * bg.max()  # the far quake does reach the box
    assert out + bg[None] == pytest.approx(want, rel=1e-6)


def _nets(seed: int = 0) -> tuple[WiredFieldNet, FieldNet]:
    cfg = load_field_config("mini")
    torch.manual_seed(seed)
    wired = WiredFieldNet(54, GRID, cfg.model, FieldWiring(attention=True)).eval()
    plain = FieldNet(54, GRID, cfg.model).eval()
    plain.load_state_dict(wired.field.state_dict())
    return wired, plain


def _inputs(days: int = 2) -> tuple[torch.Tensor, ...]:
    n = GRID.n_cells
    torch.manual_seed(1)
    src = torch.rand(days, n, 5) * (torch.rand(days, n, 5) > 0.7)
    return (
        torch.randn(days, n, 54),
        torch.randn(days, n),
        src,
        torch.rand(days, n) * 1e-3,
        torch.rand(n),
    )


def test_untrained_wiring_is_the_p6_field_exactly() -> None:
    wired, plain = _nets()
    wired.kern = torch.rand(5, GRID.n_cells, GRID.n_cells)
    mem, log_e, src, out, bg = _inputs()
    with torch.no_grad():
        assert torch.equal(wired(mem, log_e, src, out, bg), plain(mem, log_e))


def test_pair_matrix_is_declustering() -> None:
    wired, _ = _nets()
    wired.kern = torch.rand(5, GRID.n_cells, GRID.n_cells)
    mem, log_e, src, out, bg = _inputs()
    with torch.no_grad():
        h = wired.field.encode(mem, log_e)
        p = wired.pair_matrix(h, src, out, bg)
        assert p.sum(-1) == pytest.approx(torch.ones(p.shape[:-1]), abs=1e-5)
        # untrained: the sink holds exactly the background's share of ETAS's sum
        plain = torch.einsum("djb,bjc->dc", src, wired.kern) + out + bg[None]
        assert torch.allclose(p[..., -1], bg[None] / plain, rtol=1e-4)


def _trained(wired: WiredFieldNet) -> WiredFieldNet:
    wired.kern = torch.rand(5, GRID.n_cells, GRID.n_cells) * (
        torch.rand(5, GRID.n_cells, GRID.n_cells) > 0.5
    )
    with torch.no_grad():
        wired.query.weight.normal_(0, 3.0)
        wired.query.bias.normal_(0, 1.0)
    return wired


def test_every_source_sends_etas_s_number_of_aftershocks() -> None:
    wired = _trained(_nets()[0])
    mem, log_e, src, _, _ = _inputs()
    with torch.no_grad():
        g = wired.bias(wired.field.encode(mem, log_e))
        assert g.abs().max() > 0.1
        moved = wired._sent(src, g)
        plain = wired._sent(src, None)
    for a, b in zip(moved, plain, strict=True):
        assert a.sum(-1) == pytest.approx(b.sum(-1), rel=1e-4, abs=1e-6)
        assert not torch.allclose(a, b)


def test_learned_wiring_moves_rates_within_the_bound() -> None:
    wired = _trained(_nets()[0])
    mem, log_e, src, out, bg = _inputs()
    with torch.no_grad():
        r = wired.ratio(wired.field.encode(mem, log_e), src, out, bg)
    bound = float(np.log(4.0)) * 2  # a target's share can move by at most max_bias², both ways
    assert r.abs().max() > 0.05
    assert r.abs().max() <= bound + 1e-5


def test_mini_keeps_its_hash() -> None:
    cfg = load_field_config("mini")
    assert cfg.wiring is None
    assert field_hash(cfg).startswith("b99e49ae")
    wired = load_field_config("wiring")
    assert wired.wiring is not None
    assert wired.wiring.attention
    assert wired.model == cfg.model
    assert wired.train == cfg.train
    assert wired.memory == cfg.memory


def test_kernel_cache_changes_nothing() -> None:
    from craton.field.base import KernelCache

    fit, cat, proj = _fit_and_catalogue()
    days = np.arange(150.0, 160.0)
    cache = KernelCache(fit, GRID, proj)
    first = cat.select(cat.t < 120.0)
    etas_daily(fit, first, days, GRID, proj, CPU, cache=cache)  # warms the cache on a prefix
    want = etas_daily(fit, cat, days, GRID, proj, CPU)
    got = etas_daily(fit, cat, days, GRID, proj, CPU, cache=cache)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)
