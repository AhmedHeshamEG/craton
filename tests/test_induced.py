"""P8e: forcing instruments, their adapters, the forcing reader and CB-30."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from craton.adapters.base import BuildContext
from craton.adapters.crs import to_wgs84
from craton.adapters.fdsn import magnitude_type
from craton.adapters.gridded import PressureGrid, date_columns
from craton.adapters.region import Region, load_region
from craton.adapters.uic import BARREL_M3, UicWorkbook
from craton.adapters.xlsx import column_index, records
from craton.bench import csep
from craton.bench.data import DAY
from craton.bench.forcing import (
    FEATURES,
    Pushes,
    Scales,
    daily,
    features,
    pressure_pushes,
    raw,
)
from craton.bench.induced import CASES, N_PERIODS, PERIOD_DAYS, case_spec, period_counts
from craton.bench.results import RunResults, TaskResult
from craton.bench.tasks import TASKS
from craton.spec.timeaxis import to_seconds
from craton.spec.token import TokenError, validate_token
from craton.store.raw import RawStore

# ---------------------------------------------------------------- a workbook, by hand


def _xlsx(rows: list[list[Any]]) -> bytes:
    """The smallest .xlsx Excel writes: one sheet, inline strings and numbers."""

    def cell(ref: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
        return f'<c r="{ref}"><v>{v}</v></c>'

    def col(i: int) -> str:
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    body = "".join(
        f'<row r="{r + 1}">'
        + "".join(cell(f"{col(c)}{r + 1}", v) for c, v in enumerate(row))
        + "</row>"
        for r, row in enumerate(rows)
    )
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            f'<workbook {ns} {rel}><sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/></Relationships>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>"
        )
    return out.getvalue()


MONTHS = [
    f"{m} Vol"
    for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
]
HEADER = [
    "API",
    "LAT",
    "LON",
    "InjTopDepth",
    "InjBotDepth",
    "TotalDepth",
    "WellType",
    "FluidType",
    "FormationName",
    "MeasurementType",
    *MONTHS,
]


def test_column_letters_and_a_workbook_by_hand() -> None:
    assert [column_index(c) for c in ("A", "Z", "AA", "BD")] == [0, 25, 26, 55]
    rows = list(records(_xlsx([["a", "b", "c"], [1.5, None, "x"], [None, None, None]])))
    assert rows == [{"a": 1.5, "b": None, "c": "x"}]


# ---------------------------------------------------------------- the injection workbook


def _oklahoma() -> Region:
    base = load_region("oklahoma")
    return base.model_copy(update={"start": "2015-01-01T00:00:00Z"})


def test_uic_workbook_months_wells_and_doubles(tmp_path: Path) -> None:
    region = _oklahoma()
    well = [3500300026, 36.9, -98.2, 4900, 5000, 5100, "2DNC", "Saltwater", "ARBUCKLE", "METERED"]
    rows = [
        HEADER,
        well + [100.0] + [0.0] * 10 + [50.0],
        well + [100.0] + [0.0] * 10 + [50.0],  # the same row twice: a double report
        [3500300027, 36.0, -97.0, None, None, 3000, "2RIn", "Saltwater", "RED FORK", "ESTIMATED"]
        + [10.0] * 12,
        [3500300028, 0.0, 0.0, 1, 2, 3, "2DNC", "x", "y", "METERED"] + [1.0] * 12,  # no place
    ]
    raw = RawStore(tmp_path)
    raw.put("uic_workbook", "2015", "u", _xlsx(rows), "2026-10-29", ".xlsx")
    region = region.model_copy(
        update={
            "sources": {
                **region.sources,
                "uic_workbook": {**region.options("uic_workbook"), "years": [2015, 2015]},
            }
        }
    )
    ctx = BuildContext(region=region, raw=raw)
    toks = [t for batch in UicWorkbook().tokens(ctx) for t in batch]
    assert len(toks) == 2 + 12
    first = [t for t in toks if t.qualifiers["well"] == "3500300026"]
    assert [round(t.value[0], 6) for t in first] == [
        round(100 * BARREL_M3, 6),
        round(50 * BARREL_M3, 6),
    ]
    assert first[0].time == to_seconds("2015-01-01T00:00:00Z")
    assert first[0].resolution.temporal_s == 31 * DAY
    assert first[1].resolution.temporal_s == 31 * DAY  # December
    assert first[0].depth_km == pytest.approx(4950 * 0.0003048)
    assert first[0].qualifiers["well_class"] == "2DNC"
    assert first[0].spec == "1.2.0"
    assert first[0].kind == "forcing"
    other = [t for t in toks if t.qualifiers["well"] == "3500300027"]
    assert other[0].depth_km == pytest.approx(3000 * 0.0003048)  # no interval: total depth
    assert other[0].uncertainty[0] == pytest.approx(0.3 * other[0].value[0])
    assert ctx.rejects["duplicate_row"] == 1
    assert ctx.rejects["invalid:no_well_or_location"] == 1


def test_a_forcing_token_needs_the_spec_that_brought_its_instrument() -> None:
    tok = {
        "spec": "1.2.0",
        "kind": "forcing",
        "instrument": "fluid_injection",
        "lat": 36.0,
        "lon": -97.0,
        "depth_km": 2.0,
        "time": 0.0,
        "value": [10.0],
        "uncertainty": [1.0],
        "uncertainty_basis": "assumed",
        "qualifiers": {"well": "35003"},
        "provenance": {
            "source": "s",
            "dataset_version": "v",
            "license": "CC0-1.0",
            "adapter_version": "1.0.0",
            "content_hash": "sha256:" + "0" * 64,
            "synthetic": False,
        },
        "resolution": {"spatial_km": 0.1, "temporal_s": 2.6e6},
    }
    assert validate_token(tok).spec == "1.2.0"
    with pytest.raises(TokenError) as err:
        validate_token({**tok, "spec": "1.1.0"})
    assert "spec_too_old" in err.value.codes
    with pytest.raises(TokenError):
        validate_token({**tok, "value": [-1.0]})  # nothing is injected backwards


# ---------------------------------------------------------------- the pressure grid


def test_rd_new_lands_on_amersfoort_and_groningen() -> None:
    lat, lon = to_wgs84(
        "EPSG:28992", np.array([155_000.0, 233_883.0]), np.array([463_000.0, 582_065.0])
    )
    assert (lat[0], lon[0]) == pytest.approx((52.15517, 5.38721), abs=1e-5)
    assert (lat[1], lon[1]) == pytest.approx((53.2193, 6.5682), abs=3e-4)  # the Martinitoren
    with pytest.raises(ValueError, match="no conversion"):
        to_wgs84("EPSG:3035", lat, lon)


def test_pressure_grid_tokens_inside_box_and_window(tmp_path: Path) -> None:
    region = load_region("groningen").model_copy(update={"start": "1990-01-01T00:00:00Z"})
    csv = (
        "X,Y,P1989_10_01,P1990_10_01,P1991_10_01\n"
        "245000,590000,300.0,290.0,281.5\n"
        "100000,400000,300.0,300.0,300.0\n"  # far outside the box
    )
    opts = region.options("pressure_grid")
    raw = RawStore(tmp_path)
    raw.put("pressure_grid", f"{opts['record']}/{opts['file']}", "u", csv.encode(), "d", ".csv")
    ctx = BuildContext(region=region, raw=raw)
    toks = [t for batch in PressureGrid().tokens(ctx) for t in batch]
    assert [t.value[0] for t in toks] == pytest.approx([29.0, 28.15])
    assert toks[0].qualifiers["model"].startswith("NAM")
    assert toks[0].resolution.spatial_km == 0.5
    assert ctx.rejects["outside_window"] == 2
    assert date_columns(["X", "P2001_10_01", "junk"], "P%Y_%m_%d") == {
        "P2001_10_01": to_seconds("2001-10-01T00:00:00Z")
    }


def test_knmi_writes_its_ml_as_mln() -> None:
    assert magnitude_type("MLn") is None
    assert magnitude_type("MLn", {"MLn": "ML"}) == "ML"
    assert load_region("groningen").options("fdsn_catalog")["magnitude_aliases"]["MLnq"] == "ML"


# ---------------------------------------------------------------- forcing, without leaks


GRID = csep.Grid(0.0, 0.2, 0.0, 0.2, cell_deg=0.1, m_min=1.0, m_max=5.0)


def _monthly(n: int, cell: int = 0, amount: float = 30.0) -> Pushes:
    ends = to_seconds("2020-01-01T00:00:00Z") + DAY * 30 * np.arange(1, n + 1)
    return Pushes(ends, np.full(n, 30 * DAY), np.full(n, cell), np.full(n, amount))


def test_a_push_is_spread_over_its_own_days() -> None:
    d = daily(_monthly(2), GRID)
    t = d.day0 + DAY * np.array([0.0, 15.0, 30.0, 60.0, 90.0])
    assert d.total_until(t)[:, 0] == pytest.approx([0.0, 15.0, 30.0, 60.0, 60.0])
    assert d.total_until(t)[:, 1:].sum() == 0


def test_a_midnight_never_reads_a_period_that_has_not_ended() -> None:
    known, more = _monthly(3), _monthly(6)
    midnights = known.t_end[-1] + DAY * np.array([0.0, 10.0, 29.0])
    a, b = raw(known, GRID, midnights), raw(more, GRID, midnights)
    for w in a.rates:
        assert np.array_equal(a.rates[w], b.rates[w])
    assert np.array_equal(a.total, b.total)
    assert b.stale_days.tolist() == [0.0, 10.0, 29.0]
    # One day later the fourth month is known, and it shows.
    after = raw(more, GRID, known.t_end[-1:] + 30 * DAY)
    assert after.total[0, 0] == pytest.approx(4 * 30.0)


def test_before_anything_is_known_the_forcing_reads_zero() -> None:
    p = _monthly(2)
    r = raw(p, GRID, p.t_end[:1] - DAY)
    assert r.total.sum() == 0
    assert all(v.sum() == 0 for v in r.rates.values())
    f = features(r, GRID, Scales(1.0, 1.0))
    assert f.shape == (1, GRID.n_cells, len(FEATURES))
    assert np.isfinite(f.astype(np.float64)).all()


def test_pressure_pushes_are_changes_times_area() -> None:
    import pyarrow as pa

    t0, t1, t2 = (to_seconds(f"{y}-10-01T00:00:00Z") for y in (2000, 2001, 2002))
    table = pa.table(
        {
            "time": [t0, t1, t2, t0, t1],
            "lat": [0.05, 0.05, 0.05, 0.15, 0.15],
            "lon": [0.05, 0.05, 0.05, 0.15, 0.15],
            "value": [[30.0], [29.0], [29.5], [30.0], [30.0]],
            "spatial_km": [0.5] * 5,
        }
    )
    p = pressure_pushes(table, None, GRID)  # type: ignore[arg-type]
    assert sorted(p.amount.tolist()) == pytest.approx([0.0, 0.125, 0.25])  # |ΔP| · 0.25 km²
    assert set(p.t_end.tolist()) == {t1, t2}


# ---------------------------------------------------------------- CB-30


def test_cb30_is_registered_with_two_cases_of_whole_periods() -> None:
    assert "CB-30" in TASKS
    assert [c.key for c in CASES] == ["groningen", "oklahoma"]
    for case in CASES:
        spec = case_spec(case)
        assert len(spec.day_starts) == N_PERIODS
        assert spec.window.t1 - spec.window.t0 == N_PERIODS * PERIOD_DAYS * DAY
        assert spec.scored > spec.mc
        assert spec.fit_end == "2017-01-01T00:00:00Z"
        region = load_region(case.region)
        lat0, lat1, lon0, lon1 = case.box
        assert region.box.contains(lat0, lon0)
        assert region.box.contains(lat1, lon1)
        assert spec.grid.n_cells <= 1050


def test_period_counts_put_each_quake_in_its_period() -> None:
    from craton.bench.data import Events

    starts = np.array([0.0, 30 * DAY])
    ev = Events(
        time=np.array([DAY, 31 * DAY, 61 * DAY, 2 * DAY]),
        lat=np.array([0.05, 0.15, 0.05, 0.05]),
        lon=np.array([0.05, 0.15, 0.05, 0.05]),
        depth_km=np.zeros(4),
        mag=np.array([2.0, 2.0, 2.0, 0.5]),
        mag_type=np.array(["ML"] * 4),
        token_hash=np.array(["a", "b", "c", "d"]),
    )
    n = period_counts(ev, GRID, starts, 30)
    assert n.sum() == 2
    assert n[0, 0] == 1
    assert n[1, 3] == 1


# ---------------------------------------------------------------- carrying a run's results


def _result(task: str, fp: str) -> TaskResult:
    return TaskResult(
        task_id=task,
        metric="m",
        direction="higher",
        fingerprint=fp,
        seeds=(0, 1, 2),
        values=(1.0, 1.0, 1.0),
    )


def test_carry_copies_only_what_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime

    from craton.bench import harness

    frozen = {"tasks": {"CB-02": {"fingerprint": "a" * 16}, "CB-30": {"fingerprint": "b" * 16}}}
    monkeypatch.setattr(harness, "load_frozen", lambda region=None: frozen)
    runs = tmp_path / "results" / "runs"
    runs.mkdir(parents=True)

    def run(rid: str, results: tuple[TaskResult, ...], cfg: str = "baseline") -> RunResults:
        return RunResults(
            run_id=rid,
            model="etas",
            kind="baseline",
            config_hash=cfg,
            git_commit="x",
            created=datetime(2026, 10, 29, tzinfo=UTC),
            results=results,
        )

    old = run("old", (_result("CB-02", "a" * 16),))
    (runs / "old.json").write_text(old.model_dump_json(), "utf-8")
    ctx = type("Ctx", (), {"region": "central_italy"})()
    new = run("new", (_result("CB-30", "b" * 16),))
    merged = harness.carry(tmp_path, ctx, new)  # type: ignore[arg-type]
    assert [(r.task_id, r.carried_from) for r in merged.results] == [
        ("CB-02", "old"),
        ("CB-30", None),
    ]
    with pytest.raises(RuntimeError, match="a changed method"):
        harness.carry(tmp_path, ctx, run("new", new.results, cfg="other"))  # type: ignore[arg-type]
    frozen["tasks"]["CB-02"]["fingerprint"] = "c" * 16
    with pytest.raises(RuntimeError, match="frozen data changed"):
        harness.carry(tmp_path, ctx, new)  # type: ignore[arg-type]


def test_the_physics_floor_finds_a_planted_push() -> None:
    """On a synthetic box whose background rate is proportional to the push, the floor's
    fitted push share dominates ETAS's own smoothed background."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    push = np.zeros((60, GRID.n_cells))
    push[:, 0] = np.linspace(1, 10, 60)
    lam = 0.5 * push / push.mean() + 0.01
    n = rng.poisson(lam)
    x, y = push / push.mean(), np.full_like(push, 0.25)

    def nll(theta: np.ndarray) -> float:
        a, b = np.exp(theta)
        m = a * x + b * y
        return float(-(n * np.log(m) - m).sum())

    a, b = np.exp(minimize(nll, np.zeros(2), method="Nelder-Mead").x)
    assert a == pytest.approx(0.5, rel=0.2)
    assert b < 0.1


def test_injection_pushes_keep_the_classes_and_formations_asked_for() -> None:
    import json

    import pyarrow as pa

    from craton.bench.forcing import ForcingSpec, injection_pushes

    def q(cls: str, fm: str) -> str:
        return json.dumps({"well": "1", "well_class": cls, "formation": fm})

    table = pa.table(
        {
            "time": [0.0] * 4,
            "lat": [0.05] * 4,
            "lon": [0.05] * 4,
            "value": [[1.0], [2.0], [4.0], [8.0]],
            "temporal_s": [30 * DAY] * 4,
            "qualifiers": [
                q("2DNC", "ARBUCKLE"),
                q("2DCm", "Arbuckle Group"),
                q("2RIn", "ARBUCKLE"),
                q("2DNC", "RED FORK"),
            ],
        }
    )
    every = injection_pushes(table, ForcingSpec("fluid_injection"), GRID)
    disposal = injection_pushes(table, ForcingSpec("fluid_injection", ("2D",)), GRID)
    arbuckle = injection_pushes(table, ForcingSpec("fluid_injection", ("2D",), ("arbuckle",)), GRID)
    assert [every.amount.sum(), disposal.amount.sum(), arbuckle.amount.sum()] == [15.0, 11.0, 3.0]
    assert (arbuckle.t_end == 30 * DAY).all()
