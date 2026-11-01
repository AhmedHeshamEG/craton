"""P9a: the World Stress Map as a truth source and CB-23."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from craton.adapters.cards import rendered
from craton.bench.baselines import METHODS
from craton.bench.data import Events
from craton.bench.stress import (
    CASES,
    MIN_RECORDS,
    NO_ANSWER_DEG,
    QuakeLineation,
    StressCells,
    StressInterpolation,
    StressQuery,
    StressRegional,
    axial_error,
    axial_mean,
    axial_spread,
    cells_of,
)
from craton.bench.tasks import TASKS
from craton.bench.truth import TRUTH_SOURCES, WSM, parse_wsm

HEAD = "ID,LAT,LON,AZI,TYPE,DEPTH,QUALITY,REGIME\n"


def _csv(rows: list[str]) -> bytes:
    return ("﻿" + HEAD + "\n".join(rows) + "\n").encode("utf-8")


def _events(lat: np.ndarray, lon: np.ndarray, mag: float = 3.0) -> Events:
    n = len(lat)
    return Events(
        np.arange(n, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.full(n, 10.0),
        np.full(n, mag),
        np.full(n, "ML"),
        np.array([f"h{i}" for i in range(n)]),
    )


def _query(**kw: object) -> StressQuery:
    base: dict[str, object] = {
        "region": "x",
        "half": "seen",
        "lat": np.array([42.25]),
        "lon": np.array([13.25]),
        "cut": "2007-01-01T00:00:00Z",
        "mc": 2.0,
        "history": _events(np.zeros(0), np.zeros(0)),
    }
    base.update(kw)
    return StressQuery(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- axial angles


def test_axial_mean_treats_0_and_180_as_one_direction() -> None:
    azi, r = axial_mean(np.array([175.0, 5.0]))
    assert min(azi, 180 - azi) == pytest.approx(0.0, abs=1e-9)
    assert r == pytest.approx(math.cos(math.radians(10.0)))  # 10 degrees apart, doubled
    _, r = axial_mean(np.array([0.0, 90.0]))
    assert r == pytest.approx(0.0, abs=1e-12)


def test_axial_mean_weights() -> None:
    azi, _ = axial_mean(np.array([10.0, 50.0]), np.array([3.0, 0.0]))
    assert azi == pytest.approx(10.0)


def test_axial_error_range_and_no_answer() -> None:
    err = axial_error(np.array([0.0, 170.0, 90.0, math.nan]), np.array([180.0, 10.0, 0.0, 30.0]))
    assert err.tolist() == pytest.approx([0.0, 20.0, 90.0, NO_ANSWER_DEG])


def test_axial_spread() -> None:
    assert axial_spread(1.0) == 0.0
    assert axial_spread(0.0) == 90.0
    assert 10 < axial_spread(0.8) < 25


# ---------------------------------------------------------------- the truth


def test_parse_keeps_only_good_records() -> None:
    rec = parse_wsm(
        _csv(
            [
                "a,42.1,13.1,150,FMS,10,C,NF",
                "b,42.2,13.2,999,FMS,10,C,NF",  # no azimuth
                "c,42.3,13.3,140,FMS,10,D,NF",  # quality D
                "d,42.3,13.3,140,FMS,10,Xmi,NF",  # judged unreliable
                "e,42.4,13.4,160,BO,,A,",  # no depth, no regime
            ]
        )
    )
    assert len(rec) == 2
    assert rec.quality.tolist() == ["C", "A"]
    assert math.isnan(rec.depth_km[1])
    assert rec.regime.tolist() == ["NF", "U"]


def test_cells_need_enough_shallow_records() -> None:
    rows = [f"r{i},42.1{i},13.1{i},{140 + i},FMS,10,C,NF" for i in range(MIN_RECORDS)]
    rows += [f"s{i},43.1{i},13.1{i},40,FMS,10,C,TF" for i in range(MIN_RECORDS - 1)]
    rows += ["deep,43.2,13.2,40,FMS,90,A,TF"]  # below 40 km: never counted
    rows += ["far,10.0,10.0,40,FMS,5,A,TF"]  # outside the box
    cells = cells_of(parse_wsm(_csv(rows)), (41.0, 44.0, 11.5, 15.0))
    assert len(cells) == 1
    assert cells.lat[0] == 42.25
    assert cells.lon[0] == 13.25
    assert cells.azimuth[0] == pytest.approx(141.0, abs=0.01)
    assert cells.n[0] == MIN_RECORDS
    assert cells.regime == ["NF"]
    back = StressCells.from_table(cells.table())
    assert back.azimuth.tolist() == cells.azimuth.tolist()


def test_quality_weights_the_cell_mean() -> None:
    rows = ["a,42.1,13.1,100,FMS,10,A,NF", "b,42.1,13.1,100,FMS,10,A,NF"]
    rows += ["c,42.1,13.1,130,FMS,10,C,NF"]
    cells = cells_of(parse_wsm(_csv(rows)), (41.0, 44.0, 11.5, 15.0))
    # A weighs (25/15)^2 = 2.8x a C: the mean sits much nearer 100 than the plain 110.
    assert 100 < cells.azimuth[0] < 106


# ---------------------------------------------------------------- the floors


def test_regional_floor_is_the_mean_of_the_others() -> None:
    q = _query(
        truth_lat=np.array([42.0, 43.0]),
        truth_lon=np.array([13.0, 13.0]),
        truth_azimuth=np.array([170.0, 10.0]),
    )
    got = StressRegional().stress_azimuths(q, 0)
    assert min(got[0], 180 - got[0]) == pytest.approx(0.0, abs=1e-6)


def test_interpolation_prefers_the_near_cell() -> None:
    q = _query(
        truth_lat=np.array([42.25, 44.25]),
        truth_lon=np.array([13.75, 13.25]),
        truth_azimuth=np.array([30.0, 120.0]),
    )
    assert StressInterpolation().stress_azimuths(q, 0)[0] == pytest.approx(30.0, abs=1.0)


def test_lineation_finds_the_line_and_abstains_when_empty() -> None:
    rng = np.random.default_rng(0)
    t = rng.uniform(-0.3, 0.3, 400)
    # a line running north-west / south-east (azimuth ~135 degrees east of north)
    lat = 42.25 - t + rng.normal(0, 0.01, 400)
    lon = 13.25 + t / math.cos(math.radians(42.25)) + rng.normal(0, 0.01, 400)
    got = QuakeLineation().stress_azimuths(_query(history=_events(lat, lon)), 0)
    assert axial_error(got, np.array([135.0]))[0] < 3.0
    none = QuakeLineation().stress_azimuths(_query(), 0)
    assert math.isnan(none[0])


def test_lineation_ignores_quakes_below_completeness() -> None:
    lat = np.full(30, 42.25)
    q = _query(history=_events(lat, np.full(30, 13.25), mag=1.0), mc=2.0)
    assert math.isnan(QuakeLineation().stress_azimuths(q, 0)[0])


# ---------------------------------------------------------------- the task


def test_cb23_is_registered_with_its_floors() -> None:
    task = TASKS["CB-23"]
    assert task.direction == "lower"
    for name in ("stress_regional", "stress_interpolation", "quake_lineation"):
        assert task.supports(METHODS[name])
    assert not task.supports(METHODS["etas"])
    assert sum(c.half == "zero_shot" for c in CASES) == 3
    assert len({c.region for c in CASES}) == len(CASES)


def test_truth_floors_never_see_the_cell_they_answer() -> None:
    task = TASKS["CB-23"]

    class Spy:
        name = "spy"
        reads_truth_elsewhere = True

        def __init__(self) -> None:
            self.seen: list[int] = []

        def stress_azimuths(self, q: StressQuery, seed: int) -> np.ndarray:
            here = (np.abs(q.truth_lat - q.lat[0]) < 1e-9) & (np.abs(q.truth_lon - q.lon[0]) < 1e-9)
            assert not here.any()
            self.seen.append(len(q.truth_lat))
            return np.zeros(1)

    class Ctx:
        def __init__(self) -> None:
            self._cache: dict[str, object] = {}

    cells = StressCells(
        np.array([42.25, 42.75, 43.25]),
        np.array([13.25, 13.25, 13.25]),
        np.array([10.0, 20.0, 30.0]),
        np.array([3, 3, 3]),
        np.zeros(3),
        ["NF"] * 3,
    )

    class View:
        snapshot_id = ""
        events = _events(np.zeros(0), np.zeros(0))

    task.cells = lambda ctx, case: cells  # type: ignore[method-assign]
    task.view = lambda ctx, case, snap: View()  # type: ignore[method-assign]
    task.frozen_snapshots = lambda ctx: {}  # type: ignore[method-assign]
    try:
        spy = Spy()
        got = task.answers(spy, Ctx(), CASES[0], 0)  # type: ignore[attr-defined]
    finally:
        for name in ("cells", "view", "frozen_snapshots"):
            del task.__dict__[name]
    assert got.shape == (3,)
    assert spy.seen == [2, 2, 2]


def test_wsm_is_a_truth_source_with_a_card() -> None:
    assert TRUTH_SOURCES["world_stress_map"] is WSM
    assert WSM.card().license == "CC-BY-4.0"
    docs = rendered()
    assert "world_stress_map.md" in docs
    assert "truth only" in docs["world_stress_map.md"]
    assert "## Truth sources" in docs["README.md"]


def test_wsm_refuses_a_file_that_is_not_the_release(tmp_path: Path) -> None:
    class Fake:
        def get(self, url: str) -> bytes:
            return b"not the release"

    with pytest.raises(ValueError, match="hashes to"):
        WSM.raw_bytes(tmp_path, Fake())
    with pytest.raises(FileNotFoundError):
        WSM.raw_bytes(tmp_path / "empty", fetch=False)
