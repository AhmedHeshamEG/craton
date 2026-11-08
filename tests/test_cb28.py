"""CB-28: hide a station, rebuild its dv/v from the other pairs."""

from __future__ import annotations

import numpy as np
import pytest

from craton.bench.velocity import (
    VelocityGapFiller,
    VelocityNeighbours,
    VelocityQuery,
    VelocitySeries,
    VelocityZero,
    blank,
    folds,
)

DAY = 86_400.0


def network(n_days: int = 200, shared: float = -0.001, seed: int = 0) -> VelocitySeries:
    """Five stations on a line, every pair, a drop shared by all pairs after day 150."""
    rng = np.random.default_rng(seed)
    st = [f"XX.S{i}" for i in range(5)]
    pairs = [(a, b) for i, a in enumerate(st) for b in st[i + 1 :]]
    lon = {s: 13.0 + 0.1 * i for i, s in enumerate(st)}
    days = np.arange(n_days) * DAY
    signal = np.where(np.arange(n_days) >= 150, shared, 0.0)
    dvv = signal[None, :] + rng.normal(0, 1e-4, (len(pairs), n_days))
    return VelocitySeries(
        region="test",
        station_a=[a for a, _ in pairs],
        station_b=[b for _, b in pairs],
        lat=np.full(len(pairs), 42.0),
        lon=np.array([(lon[a] + lon[b]) / 2 for a, b in pairs]),
        days=days,
        dvv=dvv,
        sigma=np.full_like(dvv, 1e-4),
        reference_end=100 * DAY,
    )


def test_the_table_round_trips() -> None:
    s = network()
    back = VelocitySeries.from_table("test", s.table())
    assert back.station_a == s.station_a
    assert back.station_b == s.station_b
    assert np.array_equal(back.dvv, s.dvv)
    assert back.reference_end == s.reference_end
    assert back.lon == pytest.approx(s.lon)


def test_hiding_blanks_only_the_test_days_of_the_hidden_pairs() -> None:
    s = network()
    hid = s.involves("XX.S0")
    b = blank(s, hid)
    assert hid.sum() == 4
    assert np.isnan(b.dvv[hid][:, 100:]).all()
    assert np.isfinite(b.dvv[hid][:, :100]).all()
    assert np.isfinite(b.dvv[~hid]).all()


def test_every_station_with_enough_left_is_a_fold() -> None:
    s = network()
    assert folds(s) == [f"XX.S{i}" for i in range(5)]
    short = network(n_days=105)  # five test days x four pairs: too few
    assert folds(short) == []


def test_neighbours_rebuild_a_shared_drop_and_zero_does_not() -> None:
    s = network()
    hid = s.involves("XX.S2")
    q = VelocityQuery("test", "XX.S2", blank(s, hid), hid)
    for m in (VelocityZero(), VelocityNeighbours()):
        assert isinstance(m, VelocityGapFiller)
    near = VelocityNeighbours().fill_velocity(q, 0)
    zero = VelocityZero().fill_velocity(q, 0)
    assert near.shape == zero.shape == (4, 200)
    assert near[:, 160:].mean() == pytest.approx(-0.001, abs=1e-4)
    truth = s.dvv[hid][:, 100:]
    rmse = lambda x: float(np.sqrt(np.mean((x[:, 100:] - truth) ** 2)))  # noqa: E731
    assert rmse(near) < rmse(zero) / 3
