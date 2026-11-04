"""P9d: the camera reads a network's eyes off its own catalogue and counts the world."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from craton.bench.data import Events
from craton.field.camera import WINDOW, band_seen, completeness_before, world_weights

CPU = torch.device("cpu")
B = 1.0


def _gr(n: int, m_min: float, seed: int = 0, t0: float = 0.0, span: float = 1e6) -> Events:
    rng = np.random.default_rng(seed)
    mag = np.round(m_min - np.log10(rng.random(n)) / B, 1)
    time = np.sort(t0 + rng.random(n) * span)
    return Events(
        time=time,
        lat=np.full(n, 42.5),
        lon=np.full(n, 13.5),
        depth_km=np.full(n, 10.0),
        mag=mag,
        mag_type=np.full(n, "ML"),
        token_hash=np.array([f"h{i}" for i in range(n)]),
    )


def _blind(ev: Events, t0: float, t1: float, below: float) -> Events:
    """The catalogue with every quake under ``below`` in [t0, t1) missed."""
    hide = (ev.time >= t0) & (ev.time < t1) & (ev.mag < below)
    return ev.select(~hide)


def test_completeness_is_read_off_the_magnitudes() -> None:
    ev = _gr(5000, 1.0)
    mc = completeness_before(ev.mag, ev.time, np.array([5e5, 9e5]), CPU)
    assert mc == pytest.approx([1.3, 1.3], abs=0.2)  # the 1.0 bin is half-width: mode 1.1
    blind = _blind(ev, 4e5, 6e5, 2.0)
    mc = completeness_before(blind.mag, blind.time, np.array([5.9e5, 9e5]), CPU)
    assert mc[0] >= 2.0
    assert mc[1] == pytest.approx(1.3, abs=0.2)


def test_completeness_needs_a_full_window_and_never_looks_ahead() -> None:
    ev = _gr(1000, 1.0)
    at = ev.time[[WINDOW - 2, WINDOW + 50]]
    mc = completeness_before(ev.mag, ev.time, at, CPU)
    assert np.isnan(mc[0])
    later = _blind(ev, float(at[1]) + 1, 2e6, 3.0)  # blinding after the moment changes nothing
    assert completeness_before(later.mag, later.time, at[1:], CPU) == pytest.approx(mc[1:])


def test_a_blind_and_a_sharp_network_count_the_same_world() -> None:
    sharp = _gr(20000, 1.0, seed=1)
    blind = _blind(sharp, 3e5, 7e5, 1.9)
    window = (sharp.time >= 3.2e5) & (sharp.time < 7e5)
    raw_sharp = int(window.sum())
    w_blind = world_weights(blind, B, 1.0, CPU)
    in_blind = (blind.time >= 3.2e5) & (blind.time < 7e5)
    raw_blind = int(in_blind.sum())
    world_blind = float(w_blind[in_blind].sum())
    world_sharp = float(world_weights(sharp, B, 1.0, CPU)[window].sum())
    assert raw_blind < 0.3 * raw_sharp  # the blind network missed most of them
    assert world_blind == pytest.approx(world_sharp, rel=0.25)  # the camera finds them again
    assert world_sharp == pytest.approx(raw_sharp, rel=0.25)


def test_band_seen() -> None:
    mc = np.array([np.nan, 1.5, 2.0, 2.5, 3.0, 4.0])
    s = band_seen(mc, B, 2.0, 3.0)
    assert s[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert 0.01 < s[3] < 1.0
    assert s[4:] == pytest.approx([0.01, 0.01])


def test_camera_keeps_mini_and_has_its_own_config() -> None:
    from craton.field.config import field_hash, load_field_config

    assert field_hash(load_field_config("mini")).startswith("b99e49ae")
    cam = load_field_config("camera")
    assert cam.camera is not None
    assert cam.camera.world
    assert cam.model == load_field_config("mini").model
