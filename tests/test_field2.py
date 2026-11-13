"""Field 2's heart (P11): its kernel pieces are ETAS's kernel, and at step 0 it *is* ETAS."""

from __future__ import annotations

import itertools

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


def test_rebuilt_heart_is_born_as_etas() -> None:
    """(P11, redrawn) With a state, neighbours, rulers and the smooth clock, the untrained heart
    is ETAS exactly; with the additive head on, within the plan's 0.01 bit."""
    from craton.field2.place import PlaceHeart, quakes_to_device
    from craton.field2.tested import StateQuakes

    cat = _catalogue()
    fit = _fit(cat)
    proj = etas.Projection.for_grid(GRID)
    cfg = load_field2_config("place")
    days = np.arange(200.0, 230.0)
    want = etas.forecast_days(fit, cat, days, GRID, proj).rates
    box = make_box(fit, cat, GRID, proj, cfg.bins)
    cpu = torch.device("cpu")
    db = to_device(box, cfg.bins, cpu)
    rng = np.random.default_rng(1)
    q = StateQuakes(
        cat.t * etas.DAY, rng.integers(0, GRID.n_cells, len(cat)), (cat.m - 2.0).astype("f4")
    )
    dq = quakes_to_device(q, 0.0, 0.5, cpu)
    shape = (GRID.n_lat, GRID.n_lon)
    for additive, tol in ((False, 1e-5), (True, 0.01)):
        net = PlaceHeart(cfg.bins, cfg.net.model_copy(update={"additive": additive}))
        with torch.no_grad():
            got = net(db, dq, torch.as_tensor(days), shape).double().numpy()
        bits = float(np.abs(np.log2(got.sum(1) / want.sum(1))).max())
        assert bits < tol, (additive, bits)
        if not additive:
            assert np.allclose(got, want, rtol=1e-5, atol=1e-12)


def test_the_state_is_read_from_the_past_only() -> None:
    """A quake writes a place's state only from the midnight after it."""
    from craton.field2.place import PlaceHeart, quakes_to_device
    from craton.field2.tested import StateQuakes

    cfg = load_field2_config("place")
    net = PlaceHeart(cfg.bins, cfg.net)
    last = net.readout[2]
    assert isinstance(last, torch.nn.Linear)
    torch.nn.init.normal_(last.weight)
    q = StateQuakes(np.array([10.5 * etas.DAY]), np.array([3]), np.array([1.0], "f4"))
    dq = quakes_to_device(q, 0.0, 0.5, torch.device("cpu"))
    bg = torch.full((GRID.n_cells,), 0.01, dtype=torch.float32)
    with torch.no_grad():
        s = net.state(
            dq,
            torch.tensor([10.0, 11.0], dtype=torch.float64),
            bg,
            0.01,
            (GRID.n_lat, GRID.n_lon),
            None,
        )
        empty = net.state(
            dq, torch.tensor([10.0], dtype=torch.float64), bg, 0.01, (GRID.n_lat, GRID.n_lon), None
        )
    assert torch.allclose(s[0], empty[0])
    assert not torch.allclose(s[1], s[0])


def test_origins_cover_blocks_to_the_cut() -> None:
    from craton.field2.tested import YEAR, origins_of

    o = origins_of(0.0, 0.5 * YEAR, 10 * YEAR, 2.0)
    assert o[0] >= 1.0 * YEAR - etas.DAY
    assert all(b - a > 1.9 * YEAR for a, b in itertools.pairwise(o))
    assert o[-1] < 10 * YEAR


def _scheme_box(cfg_name: str = "scheme") -> tuple[object, object, object, np.ndarray, object]:
    from craton.field2.scheme import TankQuakes, scheme_box

    cat = _catalogue()
    fit = _fit(cat)
    proj = etas.Projection.for_grid(GRID)
    cfg = load_field2_config(cfg_name)
    assert cfg.scheme is not None
    box = make_box(fit, cat, GRID, proj, cfg.bins)
    rng = np.random.default_rng(1)
    q = TankQuakes(
        cat.t.astype("f8"),
        rng.integers(0, GRID.n_cells, len(cat)),
        (cat.m - 2.0).astype("f4"),
        rng.uniform(2, 20, len(cat)).astype("f4"),
    )
    sb = scheme_box(box, q, GRID, cfg.bins, cfg.scheme, 15.0, torch.device("cpu"))
    return cfg, fit, cat, np.arange(200.0, 230.0), sb


def test_scheme_is_born_as_etas() -> None:
    """(P11, the converged scheme) Untrained, tanks + attention + g, N, b give ETAS exactly."""
    from craton.field2.scheme import SchemeModel

    cfg, fit, cat, days, sb = _scheme_box()
    want = etas.forecast_days(fit, cat, days, GRID, etas.Projection.for_grid(GRID)).rates
    net = SchemeModel(cfg.scheme).eval()
    with torch.no_grad():
        got = net(sb, torch.as_tensor(days)).double().numpy()
    assert np.allclose(got, want, rtol=1e-5, atol=1e-12)


def test_scheme_places_read_the_past_only() -> None:
    """A quake fills a place's tanks only from the midnight after it."""
    from craton.field2.scheme import SchemeModel

    cfg, _, _, _, sb = _scheme_box()
    net = SchemeModel(cfg.scheme)
    t_q = float(sb.q_t[len(sb.q_t) // 2])
    before = torch.tensor([np.floor(t_q) + 1.0], dtype=torch.float64)
    with torch.no_grad():
        x0, _ = net.places(sb, before)
        sb.q_t[len(sb.q_t) // 2] = before[0] + 0.5  # the same quake moved after the midnight
        x1, _ = net.places(sb, before)
    k = cfg.scheme.tanks
    assert float((x0[..., : 2 * k] - x1[..., : 2 * k]).abs().max()) > 0
    sb.q_t[len(sb.q_t) // 2] = before[0] + 2.0
    with torch.no_grad():
        x2, _ = net.places(sb, before)
    assert torch.allclose(x1, x2)
