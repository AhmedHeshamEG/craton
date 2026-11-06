"""P9i: the map referee passes a planted local signal and rejects what only looks right."""

from __future__ import annotations

import numpy as np

from craton.trust.maps import MapClaim, MapPanel, halves, judge, lock_effect, shifted


def _grid(n: int = 14, cell: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return (40.0 + (i.ravel() + 0.5) * cell), (10.0 + (j.ravel() + 0.5) * cell)


def _truth(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # Directions that turn across the region: a local pattern on top of a regional mean.
    return (60.0 + 50.0 * np.sin(lat * 1.3) * np.cos(lon * 1.1)) % 180.0


def _claim(answer: np.ndarray, floor: np.ndarray, kind: str = "axial") -> MapClaim:
    lat, lon = _grid()
    truth = _truth(lat, lon) if kind == "axial" else np.sin(lat * 1.3) * np.cos(lon * 1.1)
    return MapClaim("test", kind, 0.5, [MapPanel(lat, lon, truth, answer, floor)])


def test_a_map_that_knows_the_local_pattern_passes() -> None:
    lat, lon = _grid()
    rng = np.random.default_rng(1)
    answer = (_truth(lat, lon) + rng.normal(0, 8, len(lat))) % 180.0
    floor = np.full(len(lat), 60.0)
    v = judge(_claim(answer, floor))
    assert v.passed, v.tests


def test_the_regional_average_fails_the_shifted_maps() -> None:
    lat, _ = _grid()
    regional = np.full(len(lat), 60.0)
    worse = np.full(len(lat), 150.0)
    v = judge(_claim(regional, worse))
    assert v.tests["beats_floor"]
    assert not v.tests["beats_shifted_maps"]  # a constant map is the same slid anywhere
    assert not v.passed


def test_noise_fails() -> None:
    lat, _ = _grid()
    rng = np.random.default_rng(2)
    v = judge(_claim(rng.uniform(0, 180, len(lat)), np.full(len(lat), 60.0)))
    assert not v.passed


def test_rank_maps_and_the_lock() -> None:
    lat, lon = _grid()
    rng = np.random.default_rng(3)
    good = np.sin(lat * 1.3) * np.cos(lon * 1.1) + rng.normal(0, 0.2, len(lat))
    v = judge(_claim(good, rng.normal(0, 1, len(lat)), kind="rank"))
    assert v.passed, v.tests
    assert v.lock_hash == lock_effect("test", v.locked_lead)


def test_shift_keeps_values_and_halves_split_the_cells() -> None:
    lat, lon = _grid()
    p = MapPanel(lat, lon, lat, lat.copy(), lat)
    s = shifted(p, 0.5, np.random.default_rng(0))
    assert set(np.round(s, 6)) <= set(np.round(lat, 6))
    h = halves(p, 0.5)
    assert 0.45 < h.mean() < 0.55
