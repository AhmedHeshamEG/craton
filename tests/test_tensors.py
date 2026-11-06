"""P9i: the tensor field's frames, rotations and neighbour rules."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from craton.bench.mechanisms import Mechanisms, shmax_of
from craton.field.tensors import (
    Pool,
    TensorConfig,
    TensorFieldNet,
    _sym,
    _vec,
    gather,
    load_tensor_config,
    offsets_km,
    rotate,
    rtp_of,
    une_of,
    unit_deviatoric,
)

CPU = torch.device("cpu")


def test_frames_round_trip_and_keep_shmax() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(20, 6))
    assert np.allclose(rtp_of(une_of(m)), m)
    une = une_of(m)
    # Read SHmax straight from (u, n, e): the horizontal block is nn, ee, ne.
    nn, ee, ne = une[:, 1], une[:, 2], une[:, 5]
    direct = np.degrees(np.arctan2(-ne, -(nn - ee) / 2) / 2) % 180
    assert np.allclose(shmax_of(m), direct)


def test_rotation_turns_shmax_and_offsets_together() -> None:
    m = une_of(np.array([[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))  # squeeze east-west: SHmax 90°
    t = _sym(torch.tensor(m))[None]  # (1, 1, 3, 3)
    xy = torch.tensor([[[10.0, 0.0]]], dtype=torch.float64)  # 10 km east
    m2, xy2 = rotate(t, xy, torch.tensor([math.radians(30.0)], dtype=torch.float64))
    got = shmax_of(rtp_of(_vec(m2[0]).numpy()))
    assert got[0] == pytest.approx(120.0)
    # 30° clockwise from east (azimuth 90°) is azimuth 120°: east 10 cos 30, north −10 sin 30.
    assert xy2[0, 0, 0].item() == pytest.approx(10 * math.cos(math.radians(30)))
    assert xy2[0, 0, 1].item() == pytest.approx(-10 * math.sin(math.radians(30)))


def test_unit_deviatoric_is_trace_free_and_unit() -> None:
    u = unit_deviatoric(_sym(torch.randn(5, 6, dtype=torch.float64)))
    assert torch.allclose(
        u.diagonal(dim1=-2, dim2=-1).sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-12
    )
    assert torch.allclose(u.flatten(-2).norm(dim=-1), torch.ones(5, dtype=torch.float64))


def test_offsets_cross_the_dateline() -> None:
    xy = offsets_km(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([179.9], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([-179.9], dtype=torch.float64),
    )
    assert xy[0, 0, 0].item() == pytest.approx(0.2 * 111.195, rel=1e-6)


def _pool(n: int = 60) -> Mechanisms:
    rng = np.random.default_rng(1)
    return Mechanisms(
        42.0 + rng.uniform(-1, 1, n),
        13.0 + rng.uniform(-1, 1, n),
        rng.uniform(2, 20, n),
        np.sort(rng.uniform(0, 1e8, n)),
        rng.normal(size=(n, 6)) * 1e17,
        np.full(n, 1e17),
    )


def test_gather_hides_the_target_and_its_twins() -> None:
    cfg = TensorConfig(neighbours=8, twin_km=50.0, twin_days=1e4)
    mech = _pool()
    pool = Pool.of(mech, CPU)
    idx = torch.arange(len(pool))
    b = gather(cfg, pool, pool.lat, pool.lon, pool.depth, pool.time, hide=idx)
    d_km = b.feats[..., 0] * 100.0
    dt_days = (torch.expm1(b.feats[..., 3].abs()) * 365.25).float()
    twins = (d_km < 50.0) & (dt_days < 1e4) & ~b.mask
    assert not twins.any()  # no neighbour that is both near and soon
    assert b.feats.shape == (len(pool), 8, 4)


def test_net_answers_unit_tensors_and_config_loads() -> None:
    cfg = load_tensor_config("mini")
    assert cfg.exclude_regions == ("turkey", "chile", "alaska")
    small = TensorConfig(neighbours=8)
    pool = Pool.of(_pool(), CPU)
    b = gather(small, pool, pool.lat[:4], pool.lon[:4], pool.depth[:4], None)
    out = TensorFieldNet(small)(b, b.shapes, b.xy)
    assert out.shape == (4, 3, 3)
    assert torch.allclose(out.flatten(-2).norm(dim=-1), torch.ones(4), atol=1e-5)


def test_a_pool_smaller_than_the_slots_is_padded_not_refused() -> None:
    small = TensorConfig(neighbours=8)
    pool = Pool.of(_pool(5), CPU)
    b = gather(small, pool, pool.lat[:2], pool.lon[:2], pool.depth[:2], None)
    assert b.mask.shape == (2, 8)
    assert int((~b.mask).sum(dim=1).min()) >= 1
    assert bool(b.mask[:, 5:].all())  # five mechanisms, three padding slots
