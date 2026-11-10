"""P10c: the noise instruments as Field inputs read only whole past days, near cells only."""

from __future__ import annotations

import numpy as np

from craton.bench import csep
from craton.bench.data import DAY
from craton.field.config import FieldListen, field_hash, load_field_config
from craton.field.listen import (
    N_FEATURES,
    N_PCS,
    Embeddings,
    Instruments,
    Series,
    _trailing_median,
    features,
    seislm_series,
)

GRID = csep.Grid(42.0, 43.0, 13.0, 14.0, cell_deg=0.1, m_min=1.5, m_max=8.0)
SPEC = FieldListen()
T0 = 1_500_000_000.0 - (1_500_000_000.0 % DAY)


def _series(
    n_days: int, value: float, channels: int, lat: float = 42.5, lon: float = 13.5
) -> Series:
    v = np.full((1, n_days, channels), value)
    return Series(
        np.array([lat]),
        np.array([lon]),
        T0 + DAY * np.arange(n_days, dtype=np.float64),
        v,
        np.full_like(v, 0.5),
    )


def test_a_midnight_reads_only_the_day_before() -> None:
    s = _series(10, 0.0, 3)
    s.value[0, 5] = 10.0  # loud on day 5 only
    inst = Instruments([s], [])
    starts = T0 + DAY * np.array([5, 6, 7])
    f = features(starts, GRID, SPEC, cut=np.inf, instruments=inst).astype(np.float64)
    cell = int(GRID.cell_of(np.array([42.5]), np.array([13.5]))[0])
    assert f[0, cell, 0] == 0.0  # midnight starting day 5 can't hear day 5
    assert abs(f[1, cell, 0] - 1.0) < 1e-3  # the next midnight does (10 dB = 1)
    assert f[2, cell, 0] == 0.0  # and nothing is carried forward
    assert f[1, cell, 3] > 0
    assert f.shape[-1] == N_FEATURES


def test_far_cells_and_days_without_data_hear_nothing() -> None:
    inst = Instruments([_series(10, 5.0, 3, lat=42.05, lon=13.05)], [])
    starts = T0 + DAY * np.array([-20, 3])
    f = features(starts, GRID, SPEC, cut=np.inf, instruments=inst).astype(np.float64)
    assert np.all(f[0] == 0)
    far = int(GRID.cell_of(np.array([42.95]), np.array([13.95]))[0])
    assert np.all(f[1, far] == 0)


def test_velocity_with_a_reference_past_the_cut_is_not_read() -> None:
    v = _series(10, 0.002, 2)
    inst = Instruments([], [(v, T0 + 5 * DAY)])
    starts = T0 + DAY * np.array([8])
    cell = int(GRID.cell_of(np.array([42.5]), np.array([13.5]))[0])
    read = features(starts, GRID, SPEC, cut=T0 + 6 * DAY, instruments=inst).astype(np.float64)
    assert abs(read[0, cell, 4] - 1.0) < 1e-3  # 0.2% = 1
    blind = features(starts, GRID, SPEC, cut=T0 + 4 * DAY, instruments=inst)
    assert np.all(blind == 0)


def test_trailing_median_leaves_the_day_itself_out() -> None:
    x = np.arange(10, dtype=np.float64).reshape(1, 10, 1)
    m = _trailing_median(x, 4)
    assert np.isnan(m[0, 0, 0])
    assert m[0, 5, 0] == np.median([1, 2, 3, 4])


def test_mini_keeps_its_hash_and_listen_differs() -> None:
    mini = load_field_config("mini")
    assert mini.listen is None
    assert field_hash(mini) != field_hash(load_field_config("listen"))
    assert field_hash(mini) == field_hash(mini.model_copy(update={"listen": None}))


def _embeddings(n_days: int, rng: np.random.Generator) -> Embeddings:
    v = rng.normal(size=(2, n_days, 240))
    return Embeddings(
        np.array([42.5, 42.6]),
        np.array([13.5, 13.6]),
        T0 + DAY * np.arange(n_days, dtype=np.float64),
        v,
        np.full_like(v, 0.1),
    )


def test_seislm_basis_sees_only_days_before_the_cut() -> None:
    rng = np.random.default_rng(0)
    e = _embeddings(200, rng)
    cut = T0 + 120 * DAY
    a = seislm_series([e], cut, 30)[0]
    e.value[:, 120:] += 50 * rng.normal(size=(2, 80, 240))  # the future changes wildly
    b = seislm_series([e], cut, 30)[0]
    assert a.value.shape == (2, 200, N_PCS)
    np.testing.assert_allclose(a.value[:, :120], b.value[:, :120])


def test_seislm_ear_fills_the_loudness_columns() -> None:
    e = _embeddings(100, np.random.default_rng(1))
    inst = Instruments([], [], [e])
    starts = T0 + DAY * np.array([80.0])
    f = features(starts, GRID, SPEC, cut=T0 + 70 * DAY, instruments=inst).astype(np.float64)
    cell = int(GRID.cell_of(np.array([42.5]), np.array([13.5]))[0])
    assert np.any(f[0, cell, :3] != 0)
    assert f[0, cell, 3] > 0
    assert np.all(f[0, :, 4:] == 0)


def test_the_band_ear_keeps_p10c_hash() -> None:
    listen = load_field_config("listen")
    assert listen.listen is not None
    assert listen.listen.ear == "bands"
    seislm = load_field_config("listen_seislm")
    assert field_hash(seislm) != field_hash(listen)
