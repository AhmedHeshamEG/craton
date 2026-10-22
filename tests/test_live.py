"""Locked forecasts: seals, the ledger, grades, the page, the live catalogue parser."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from craton.bench.data import Events
from craton.bench.tasks import TASKS
from craton.trust import lock

SPEC = TASKS["CB-02"].spec  # type: ignore[attr-defined]


def _events(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, mag: np.ndarray) -> Events:
    n = len(t)
    return Events(
        np.asarray(t, float),
        np.asarray(lat, float),
        np.asarray(lon, float),
        np.full(n, 10.0),
        np.asarray(mag, float),
        np.asarray(["ML"] * n, dtype=np.str_),
        np.asarray([f"q{i}" for i in range(n)], dtype=np.str_),
    )


def _body(day: str = "2026-10-23") -> dict[str, object]:
    g = SPEC.grid
    return {
        "schema": lock.SCHEMA,
        "region": "central_italy",
        "day": day,
        "issued": "2026-10-22T22:30:00Z",
        "magnitude": 3.0,
        "craton": {"rates": [0.01] * g.n_cells, "day_k": 2.0, "abstain": False},
        "etas": {"rates": [0.02] * g.n_cells},
    }


def test_seal_detects_any_edit(tmp_path: Path) -> None:
    body = _body()
    sealed = lock.seal(body)
    assert lock.intact(sealed)
    edited = json.loads(json.dumps(sealed))
    edited["craton"]["rates"][3] = 0.011
    assert not lock.intact(edited)
    path = lock.write_locked(tmp_path, "central_italy", body)
    assert lock.read_locked(path)["sha256"] == sealed["sha256"]
    with pytest.raises(FileExistsError):
        lock.write_locked(tmp_path, "central_italy", body)
    ledger = (tmp_path / "forecasts" / "central_italy" / "ledger.jsonl").read_text("utf-8")
    assert sealed["sha256"] in ledger


def test_grade_and_page(tmp_path: Path) -> None:
    g = SPEC.grid
    body = lock.seal(_body())
    lat, lon = g.cell_centres()
    from craton.spec.timeaxis import to_seconds

    t0 = to_seconds("2026-10-23T00:00:00Z")
    ev = _events(np.array([t0 + 3600]), lat[[10]], lon[[10]], np.array([3.4]))
    gr = lock.grade(body, ev, g)
    assert gr["quakes"] == 1
    # one quake, Craton expects half as many everywhere: Craton's total is closer to 1
    want = (math.log(0.01) - 0.01 * g.n_cells) - (math.log(0.02) - 0.02 * g.n_cells)
    assert gr["gain_bits"] == pytest.approx(want / math.log(2))
    lock.write_locked(tmp_path, "central_italy", _body())
    gp = lock.grade_path(tmp_path, "central_italy", "2026-10-23")
    gp.parent.mkdir(parents=True)
    gp.write_text(json.dumps(gr), "utf-8")
    page = lock.page(tmp_path, "central_italy").read_text("utf-8")
    assert "1 days locked · 1 graded" in page
    assert "Never a prediction" in page


def test_sig_rounds_to_significant_digits() -> None:
    assert lock.sig([0.000123456789, 12345.6789, 0.0]) == [0.000123457, 12345.7, 0.0]


def test_live_parser_reads_the_ingv_text_like_the_adapter() -> None:
    from craton.adapters.ingv import _body_lines
    from craton.adapters.region import load_region
    from craton.live import _month_starts, parse_lines

    raw = (Path(__file__).parent / "fixtures" / "ingv" / "norcia.txt").read_bytes()
    ev = parse_lines(_body_lines(raw), load_region("central_italy"))
    assert len(ev) == 12
    assert ev.mag.max() == pytest.approx(6.5)
    assert np.all(np.diff(ev.time) >= 0)
    assert ev.token_hash[0].startswith("ingv-")
    from craton.spec.timeaxis import to_seconds

    months = _month_starts(to_seconds("2026-08-15T00:00:00Z"), to_seconds("2026-10-22T00:00:00Z"))
    assert len(months) == 3
