import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from craton.bench import splits
from craton.bench.data import Gnss, load_events, load_gnss
from craton.bench.tasks import TASKS, load_frozen
from craton.spec import validate_token
from craton.spec.examples import example
from craton.spec.timeaxis import to_iso, to_seconds
from craton.store.parquet import write_tokens


def _token(name: str, when: str, station: str | None = None, mag: float | None = None) -> object:
    d = example(name).to_dict()
    d["time"] = to_seconds(when)
    d["time_iso"] = to_iso(d["time"])
    if station is not None:
        d["qualifiers"]["station"] = station
    if mag is not None:
        d["value"] = [mag]
        d["qualifiers"]["event_id"] = f"{when}-{mag}"
    return validate_token(d)


@pytest.fixture
def token_root(tmp_path: Path) -> Path:
    root = tmp_path / "tokens"
    toks = [
        _token("norcia_2016", "2008-06-01T00:00:00Z", mag=3.1),
        _token("norcia_2016", "2009-04-06T01:32:00Z", mag=6.1),  # era B
        _token("norcia_2016", "2012-05-20T00:00:00Z", mag=2.4),
        _token("norcia_2016", "2016-10-30T06:40:00Z", mag=6.5),  # era A
        _token("norcia_2016", "2020-01-01T00:00:00Z", mag=2.0),
        _token("arquata_gnss", "2015-01-01T12:00:00Z", station="ARQT"),
        _token("arquata_gnss", "2015-01-01T12:00:00Z", station="HELD"),
        _token("arquata_gnss", "2019-01-01T12:00:00Z", station="ARQT"),
    ]
    write_tokens(toks, root)  # type: ignore[arg-type]
    return root


def _files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def test_pretrain_leaves_out_eras_and_heldout_stations(token_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pretrain"
    m = splits.materialise(splits.SPLITS["pretrain"], _files(token_root), token_root, out, ["HELD"])
    assert m["tokens"] == 5
    assert m["left_out"] == 3
    ev = load_events(out)
    assert sorted(ev.mag.tolist()) == [2.0, 2.4, 3.1]
    assert set(load_gnss(out).station.tolist()) == {"ARQT"}
    assert splits.leak_problems(out, ["HELD"]) == []


def test_fit_splits_stop_before_their_test_window(token_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "cb03"
    splits.materialise(splits.SPLITS["cb03_fit"], _files(token_root), token_root, out, ["HELD"])
    assert load_events(out).mag.tolist() == [3.1]
    out2 = tmp_path / "cb02"
    splits.materialise(splits.SPLITS["cb02_fit"], _files(token_root), token_root, out2, ["HELD"])
    assert sorted(load_events(out2).mag.tolist()) == [2.4, 3.1, 6.1]


def test_materialise_is_reproducible(token_root: Path, tmp_path: Path) -> None:
    a = splits.materialise(
        splits.SPLITS["pretrain"], _files(token_root), token_root, tmp_path / "a", ["HELD"]
    )
    b = splits.materialise(
        splits.SPLITS["pretrain"], _files(token_root), token_root, tmp_path / "b", ["HELD"]
    )
    assert a["id"] == b["id"]
    assert a["shards"] == b["shards"]


def test_leak_check_catches_a_smuggled_token(token_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pretrain"
    splits.materialise(splits.SPLITS["pretrain"], _files(token_root), token_root, out, ["HELD"])
    # Swap a clean shard for the unfiltered file, keeping the manifest: era tokens get in.
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    raw = sorted((token_root / "instrument=seismic_catalog").rglob("*.parquet"))
    era_file = next(p for p in raw if "year=2016" in p.as_posix())
    shard = next(s for s in manifest["shards"] if "seismic_catalog" in s["path"])
    target = out / shard["path"]
    shutil.copy(era_file, target)
    problems = splits.leak_problems(out, ["HELD"])
    assert any("does not match its name" in p for p in problems)
    # Rename it to its true hash and fix the manifest: now the fingerprint passes, the leak shows.
    true_name = target.parent / era_file.name
    target.rename(true_name)
    shard["path"] = true_name.relative_to(out).as_posix()
    manifest["id"] = splits.manifest_id(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = splits.leak_problems(out, ["HELD"])
    assert any("held-out token" in p for p in problems)


def test_leak_check_catches_benchmark_answers(token_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "pretrain"
    splits.materialise(splits.SPLITS["pretrain"], _files(token_root), token_root, out, ["HELD"])
    answer = load_events(out).token_hash[0]
    assert any("benchmark answer" in p for p in splits.leak_problems(out, ["HELD"], {answer}))
    assert any("different held-out" in p for p in splits.leak_problems(out, ["OTHER"]))


def _gnss(stations: dict[str, tuple[float, float, int, float]]) -> Gnss:
    rows = []
    for code, (lat, lon, days, jump) in stations.items():
        for d in range(days):
            rows.append((d * 86400.0, code, lat, lon, jump if d == days // 2 else 0.0))
    t, st, la, lo, j = zip(*rows, strict=True)
    enu = np.zeros((len(t), 3))
    enu[:, 0] = np.cumsum(j)  # a jump persists, like a monument that moved
    return Gnss(
        np.array(t),
        np.array(st),
        np.array(la),
        np.array(lo),
        enu,
        np.full((len(t), 3), 0.001),
        np.array([f"h{i}" for i in range(len(t))]),
    )


def test_stable_stations_and_heldout_choice() -> None:
    sites = {
        f"S{i:03d}": (42.0 + 0.01 * (i % 7), 13.0 + 0.01 * (i // 7), 800, 0.0) for i in range(40)
    }
    sites["MOVE"] = (42.02, 13.02, 800, 3.0)
    sites["FEWD"] = (42.03, 13.03, 100, 0.0)
    sites["FAR1"] = (44.5, 16.0, 800, 0.0)
    g = _gnss(sites)
    stable = splits.stable_stations(g)
    assert "MOVE" not in stable
    assert "S000" in stable
    chosen = splits.choose_heldout(splits.station_sites(g), stable)
    assert chosen == splits.choose_heldout(splits.station_sites(g), stable)
    assert len(chosen) == round(0.1 * len(sites))
    assert not {"MOVE", "FEWD", "FAR1"} & set(chosen)


def test_frozen_record_matches_the_code() -> None:
    frozen = load_frozen()
    assert frozen is not None
    assert splits.definition_problems(frozen) == []
    assert set(frozen["tasks"]) == set(TASKS)
    for task_id, task in TASKS.items():
        f = frozen["tasks"][task_id]
        assert (f["metric"], f["direction"]) == (task.metric, task.direction)
        # The fingerprint is recomputable from the code and the frozen file hashes alone.
        assert task.fingerprint(f["files"]) == f["fingerprint"]


def test_eras_and_forecast_windows() -> None:
    assert splits.ERA_A.t0 < TASKS["CB-02"].window.t0 < splits.ERA_A.t1  # type: ignore[attr-defined]
    assert TASKS["CB-03"].window.t0 == splits.ERA_B.t0  # type: ignore[attr-defined]
    frozen = load_frozen()
    assert frozen is not None
    assert len(frozen["heldout_stations"]) == 29


def test_written_tables_are_parquet(token_root: Path) -> None:
    for p in _files(token_root):
        assert pq.read_metadata(p).num_rows > 0
