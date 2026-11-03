"""P9c: the hidden-state blocks start as the physics, and the angle is physical."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from craton.bench.csep import Grid
from craton.bench.data import Events
from craton.field.config import FieldHidden, field_hash, load_field_config
from craton.field.hidden import HiddenFieldNet, K, S, W, make_net, shift_bank
from craton.field.memory import MemorySpec, memory
from craton.field.model import FieldNet
from craton.field.steer import azimuth_of, quadrupole

GRID = Grid(42.0, 43.0, 13.0, 14.0)
DAY = 86_400.0


def _events(lat: np.ndarray, lon: np.ndarray, t0: float = 1e9) -> Events:
    n = len(lat)
    return Events(
        time=t0 + np.arange(n, dtype=np.float64),
        lat=np.asarray(lat, np.float64),
        lon=np.asarray(lon, np.float64),
        depth_km=np.full(n, 10.0),
        mag=np.full(n, 2.5),
        mag_type=np.full(n, "ML"),
        token_hash=np.array([f"h{i}" for i in range(n)]),
    )


def _line(azimuth_deg: float, n: int = 41, half_km: float = 25.0) -> Events:
    """Quakes on a line through the centre of cell (5, 5), along an azimuth east of north."""
    d = np.linspace(-half_km, half_km, n)
    a = math.radians(azimuth_deg)
    lat = 42.55 + d * math.cos(a) / 111.195
    lon = 13.55 + d * math.sin(a) / (111.195 * math.cos(math.radians(42.5)))
    return _events(lat, lon)


SPEC = MemorySpec(m_min=1.5, start=9e8)


@pytest.mark.parametrize("azimuth", [0.0, 30.0, 90.0, 135.0])
def test_quadrupole_points_along_a_line_of_quakes(azimuth: float) -> None:
    ev = _line(azimuth)
    q = quadrupole(ev, GRID, np.array([1e9 + DAY]), SPEC, 15.0, torch.device("cpu"))
    centre = 5 * GRID.n_lon + 5
    got = float(azimuth_of(q[0, centre, 8].astype(np.float64)))
    err = abs((got - azimuth + 90) % 180 - 90)
    assert err < 8.0, (azimuth, got)
    assert np.all(np.abs(q) <= 1.0)


def test_shift_zero_is_the_bank_and_one_is_the_next_lifetime() -> None:
    x = torch.randn(3, K, W)
    assert torch.equal(shift_bank(x, torch.zeros(3)), x)
    one = shift_bank(x, torch.ones(3))
    assert torch.allclose(one[:, :-1], x[:, 1:])
    assert torch.allclose(one[:, -1], x[:, -1])  # held at the end


def _cfg(clock: bool, steer: bool) -> tuple[object, FieldHidden]:
    cfg = load_field_config("mini")
    return cfg, FieldHidden(clock=clock, steer=steer)


def test_untrained_hidden_field_reads_the_p6_memory_exactly() -> None:
    cfg, hidden = _cfg(True, True)
    torch.manual_seed(0)
    net = HiddenFieldNet(S * K * W, GRID, cfg.model, hidden).eval()  # type: ignore[attr-defined]
    days, cells = 2, GRID.n_cells
    mem = torch.randn(days, cells, S * K * W)
    quad = torch.rand(days, cells, K * W, 2) * 0.7
    with torch.no_grad():
        out, s, w = net.blocks(mem, torch.randn(days, cells), quad)
    assert torch.equal(out, mem)
    assert s is not None
    assert w is not None
    assert float(s.abs().max()) == 0.0
    assert float(w.abs().max()) == 0.0
    plain = FieldNet(S * K * W, GRID, cfg.model).eval()  # type: ignore[attr-defined]
    with torch.no_grad():
        assert float(net(mem, torch.zeros(days, cells), quad).abs().max()) == 0.0
        assert float(plain(mem, torch.zeros(days, cells)).abs().max()) == 0.0


def test_the_angle_turns_with_the_catalogue() -> None:
    """Rotate every quake by alpha: q turns by 2 alpha, |q| and the counters stay; theta turns
    by alpha, exactly, whatever the weights."""
    cfg, hidden = _cfg(False, True)
    torch.manual_seed(1)
    net = HiddenFieldNet(S * K * W, GRID, cfg.model, hidden)  # type: ignore[attr-defined]
    for p in net.steer.parameters():
        torch.nn.init.normal_(p, std=0.5)
    mem = torch.randn(4, 7, S * K * W)
    quad = torch.rand(4, 7, K * W, 2) * 0.6 - 0.3
    alpha = 0.37
    c, s = math.cos(2 * alpha), math.sin(2 * alpha)
    turned = torch.stack(
        [c * quad[..., 0] - s * quad[..., 1], s * quad[..., 0] + c * quad[..., 1]], -1
    )
    w0 = net.stretch(mem, quad).detach().double().numpy()
    w1 = net.stretch(mem, turned).detach().double().numpy()
    d = (azimuth_of(w1) - azimuth_of(w0) - math.degrees(alpha) + 90) % 180 - 90
    assert np.abs(d).max() < 1e-3
    assert np.allclose(np.linalg.norm(w0, axis=-1), np.linalg.norm(w1, axis=-1), atol=1e-5)
    assert np.linalg.norm(w0, axis=-1).max() < hidden.max_stretch


def test_stretched_memory_is_the_stretched_kernel() -> None:
    """log I + log(1 + Re(w̄ q)) is the log of the sum under g · (1 + e cos 2(φ − θ))."""
    ev = _line(40.0)
    spec = MemorySpec(m_min=1.5, start=9e8, spreads=(0.0,))
    t = np.array([1e9 + DAY])
    q = quadrupole(ev, GRID, t, spec, 15.0, torch.device("cpu"))[0, 5 * GRID.n_lon + 5, 8]
    e, theta = 0.5, math.radians(10.0)
    lat0 = 42.55
    y = (ev.lat - lat0) * 111.195
    x = (ev.lon - 13.55) * 111.195 * math.cos(math.radians(42.5))
    r2 = x**2 + y**2
    g = np.exp(-0.5 * r2 / 15.0**2)
    phi = np.arctan2(x, y)
    # cell centres, not the quakes' exact places, are where the kernel measures distance from
    cells = GRID.cell_of(ev.lat, ev.lon)
    clat, clon = GRID.cell_centres()
    yc = (clat[cells] - lat0) * 111.195
    xc = (clon[cells] - 13.55) * 111.195 * math.cos(math.radians(42.5))
    rc2 = xc**2 + yc**2
    gc = np.exp(-0.5 * rc2 / 15.0**2)
    pc = np.arctan2(xc, yc)
    ok = rc2 > 0
    want = 1 + e * (gc[ok] * np.cos(2 * (pc[ok] - theta))).sum() / gc.sum()
    got = 1 + e * (math.cos(2 * theta) * q[0] + math.sin(2 * theta) * q[1])
    assert abs(float(got) - want) < 0.03
    assert g.sum() > 0
    assert phi.shape == r2.shape


def test_mini_keeps_its_hash_and_its_net() -> None:
    cfg = load_field_config("mini")
    assert cfg.hidden is None
    assert field_hash(cfg)[:8] == "b99e49ae"  # the 6-member P7 checkpoints keep their name
    assert isinstance(make_net(S * K * W, GRID, cfg.model, None), FieldNet)


def test_memory_and_counters_agree() -> None:
    ev = _line(0.0)
    m = memory(ev, GRID, np.array([1e9 + DAY, 1e9 + 2 * DAY]), SPEC)
    assert m.shape == (2, GRID.n_cells, SPEC.n_features)
    assert np.all(m[1] <= m[0])
    assert np.any(m[1] < m[0])


def test_stress_map_reads_the_steer_head_anywhere() -> None:
    """A steer head that just follows the moments answers a line of quakes with its azimuth,
    in a box far from central Italy; a cell with no quakes near it gets no answer."""
    from craton.bench.stress import StressQuery
    from craton.field.stressmap import fine_grid, stretch_map

    cfg = load_field_config("hidden_steer")
    assert cfg.hidden is not None
    net = HiddenFieldNet(S * K * W, GRID, cfg.model, cfg.hidden)
    last = net.steer[-1]
    torch.nn.init.zeros_(last.weight)
    with torch.no_grad():
        last.bias.zero_()
        last.bias[0::2] = 1.0  # a = 1, b = 0 on every counter: w along the moments
    state = net.state_dict()
    lat0, lon0 = 36.25, 140.25
    a = math.radians(60.0)
    d = np.linspace(-25, 25, 51)
    ev = _events(
        lat0 + d * math.cos(a) / 111.195,
        lon0 + d * math.sin(a) / (111.195 * math.cos(math.radians(lat0))),
    )
    q = StressQuery(
        region="test",
        half="seen",
        lat=np.array([lat0, lat0 + 3.0]),
        lon=np.array([lon0, lon0]),
        cut="2001-11-01T00:00:00Z",
        mc=2.0,
        history=ev,
    )
    grid = fine_grid(q.lat, q.lon)
    members, own = stretch_map([state], cfg, q, grid)
    w = members.sum(axis=0)
    clat, clon = grid.cell_centres()
    near = (np.abs(clat - lat0) < 0.25) & (np.abs(clon - lon0) < 0.25)
    assert own[near].sum() > 40  # the twelve-year counter holds the line's quakes
    got = float(azimuth_of((w[near] * own[near, None]).sum(axis=0)))
    assert abs((got - 60.0 + 90) % 180 - 90) < 10.0
    far = (np.abs(clat - lat0 - 3.0) < 0.25) & (np.abs(clon - lon0) < 0.25)
    assert np.abs(w[far]).max() == 0.0
