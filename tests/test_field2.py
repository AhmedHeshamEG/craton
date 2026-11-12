"""Field 2's heart (P11): its kernel pieces are ETAS's kernel, and at step 0 it *is* ETAS."""

from __future__ import annotations

import numpy as np
import torch

from craton.bench import etas
from craton.bench.csep import Grid
from craton.field2.config import load_field2_config
from craton.field2.heart import Heart, bin_coordinates, make_box, rates, ring_entries, to_device

GRID = Grid(42.0, 42.6, 13.0, 13.7, cell_deg=0.1, m_min=2.0, m_max=8.0)
PARAMS = etas.EtasParams(mu=0.3, k=0.4, a=1.2, c=0.01, p=1.1, d=0.8, gamma=0.9, q=1.6)


def _catalogue(n: int = 120, seed: int = 0) -> etas.Catalog:
    rng = np.random.default_rng(seed)
    proj = etas.Projection.for_grid(GRID)
    lat = rng.uniform(GRID.lat_min - 0.2, GRID.lat_max + 0.2, n)
    lon = rng.uniform(GRID.lon_min - 0.2, GRID.lon_max + 0.2, n)
    mag = 2.0 + rng.exponential(1 / (np.log(10) * 1.0), n)
    mag[5] = 6.3  # one large parent, a rupture length of ~20 km
    t = np.sort(rng.uniform(0, 400, n)) * etas.DAY
    return etas.to_catalog(t, lat, lon, mag, proj, 0.0)


def _fit(cat: etas.Catalog) -> etas.EtasFit:
    proj = etas.Projection.for_grid(GRID)
    bg = etas.Background(cat.x, cat.y, np.ones(len(cat)), etas.box_km(GRID, proj))
    return etas.EtasFit(PARAMS, bg, 2.0, 1.0, 0.0, etas.FitWindow(0.0, 400.0), len(cat), 0.0, True)


def test_rings_add_up_to_the_etas_kernel() -> None:
    cat = _catalogue()
    proj = etas.Projection.for_grid(GRID)
    bins = load_field2_config("heart").bins
    ent = ring_entries(PARAMS, cat, 2.0, GRID, proj, bins.ruler)
    full = etas.spatial_matrix(PARAMS, cat, 2.0, GRID, proj).toarray()
    mine = np.zeros_like(full)
    np.add.at(mine, (ent.parent, ent.cell), ent.share)
    assert np.allclose(mine, full, rtol=1e-12, atol=1e-15)
    # The large parent's kernel spans several rings of its own ruler.
    big = ent.ring[ent.parent == 5]
    assert big.min() <= 3
    assert len(np.unique(big)) > 2


def test_born_as_etas() -> None:
    """Untrained, the heart's daily rates are ``forecast_days``'s to float32 precision."""
    cat = _catalogue()
    fit = _fit(cat)
    proj = etas.Projection.for_grid(GRID)
    cfg = load_field2_config("heart")
    days = np.arange(200.0, 260.0)
    want = etas.forecast_days(fit, cat, days, GRID, proj).rates
    box = make_box(fit, cat, GRID, proj, cfg.bins)
    db = to_device(box, cfg.bins, torch.device("cpu"))
    net = Heart(cfg.bins, cfg.net)
    with torch.no_grad():
        log_w, log_w0 = net(db.context)
    assert float(log_w.abs().max()) == 0.0
    assert float(log_w0.abs()) == 0.0
    got = rates(db, torch.as_tensor(days), log_w, log_w0, cfg.bins).double().numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-12)
    # In bits per quake-day the difference is far under the 0.01 the plan asks for.
    assert float(np.abs(np.log2(got.sum(1) / want.sum(1))).max()) < 1e-4


def test_a_weight_moves_only_its_bin() -> None:
    cat = _catalogue()
    fit = _fit(cat)
    proj = etas.Projection.for_grid(GRID)
    cfg = load_field2_config("heart")
    box = make_box(fit, cat, GRID, proj, cfg.bins)
    db = to_device(box, cfg.bins, torch.device("cpu"))
    n = len(bin_coordinates(cfg.bins))
    days = torch.as_tensor(np.arange(300.0, 305.0))
    base = rates(db, days, torch.zeros(n), torch.zeros(()), cfg.bins)
    up = rates(db, days, torch.full((n,), float(np.log(2.0))), torch.zeros(()), cfg.bins)
    bg = box.cascade * box.background
    # Doubling every bin doubles the triggered part and leaves the background alone.
    bg_t = torch.as_tensor(bg)[None, :].float()
    assert torch.allclose(up - bg_t, 2 * (base - bg_t), rtol=1e-4)


def test_free_rings_keep_their_quakes() -> None:
    """Each ring of the free kernel hands out one unit, minus what falls outside the grid."""
    from craton.field2.free import uniform_entries

    cat = _catalogue()
    proj = etas.Projection.for_grid(GRID)
    ruler = load_field2_config("heart").bins.ruler
    ent = uniform_entries(cat, GRID, proj, ruler)
    total = np.zeros((len(cat), len(ruler)))
    np.add.at(total, (ent.parent, ent.ring), ent.share)
    assert float(total.max()) <= 1.0 + 1e-9
    # a quake in the middle of the grid keeps its inner rings whole
    mid = int(np.argmin((cat.x - cat.x.mean()) ** 2 + (cat.y - cat.y.mean()) ** 2))
    inner = total[mid][total[mid] > 0][:2]
    assert np.allclose(inner, 1.0)


def test_free_head_is_born_a_whisper() -> None:
    """With the free head on, the untrained heart is ETAS within the plan's 0.01 bit."""
    from craton.field2.free import free_to_device, make_free
    from craton.field2.world import heart_rates

    cat = _catalogue()
    fit = _fit(cat)
    proj = etas.Projection.for_grid(GRID)
    cfg = load_field2_config("heart_free")
    days = np.arange(200.0, 230.0)
    want = etas.forecast_days(fit, cat, days, GRID, proj).rates
    box = make_box(fit, cat, GRID, proj, cfg.bins)
    fb = make_free(cat, 2.0, GRID, proj, cfg.bins.ruler, cfg.net.free_magnitude)
    cpu = torch.device("cpu")
    db = to_device(box, cfg.bins, cpu)
    fd = free_to_device(fb, len(cfg.bins.ruler), box.n_cells, cpu)
    net = Heart(cfg.bins, cfg.net)
    with torch.no_grad():
        got = heart_rates(net, db, fd, torch.as_tensor(days), cfg).double().numpy()
    assert float(np.abs(np.log2(got.sum(1) / want.sum(1))).max()) < 0.01
