"""P9g: CB-31's truth, how a method is read, and the rule that a graded model never reads it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from craton.bench import csep
from craton.bench.data import DAY
from craton.bench.forcing import ForcingSpec, pushes
from craton.bench.induced import EtasForced, case_spec
from craton.bench.planted import QuakeDensity
from craton.bench.pressure import (
    GAS_YEARS,
    INSTRUMENT,
    GroningenPressure,
    HiddenInstrumentError,
    Truth,
    blind_view,
    build_truth,
    cell_points,
    gas_year,
    grade,
    year_rates,
)
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast

ROOT = Path(__file__).resolve().parents[1]
TASK = GroningenPressure()


def test_blind_view_refuses_the_pressure_by_source() -> None:
    spec = case_spec(TASK.case)
    view = blind_view(ROOT, TASK.case, spec, "")
    with pytest.raises(HiddenInstrumentError):
        view._snapshot_files(INSTRUMENT)
    with pytest.raises(HiddenInstrumentError):
        pushes(view, ForcingSpec(INSTRUMENT), spec.grid)


class _Peeker:
    """A forecaster that tries to read the pressure while it forecasts."""

    name = "peeker"
    kind = "baseline"
    long_periods = True

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        pushes(ctx, ForcingSpec(INSTRUMENT), spec.grid)
        raise AssertionError("read the pressure")


def test_a_graded_method_cannot_read_the_pressure() -> None:
    ctx = BenchContext(ROOT)
    assert TASK.supports(_Peeker())
    with pytest.raises(HiddenInstrumentError):
        TASK.answer(_Peeker(), ctx, 0)


class _Twin:
    name = "with-twin"

    def forecast_forced(self, *a: Any) -> None: ...

    def quake_only_forecaster(self) -> Any: ...


def test_who_is_graded() -> None:
    assert not TASK.supports(EtasForced())  # reads the push, has no quake-only twin
    assert TASK.supports(_Twin())  # answers through its twin
    assert TASK.supports(QuakeDensity())


def _table(dates: list[float], points: list[tuple[float, float, list[float]]]) -> pa.Table:
    rows: dict[str, list[Any]] = {"time": [], "lat": [], "lon": [], "value": []}
    for lat, lon, ps in points:
        for t, p in zip(dates, ps, strict=True):
            rows["time"].append(t)
            rows["lat"].append(lat)
            rows["lon"].append(lon)
            rows["value"].append([p])
    return pa.table(rows)


def test_truth_is_the_yearly_drop_on_covered_cells() -> None:
    grid = csep.Grid(53.0, 53.1, 6.4, 6.5, cell_deg=0.05, m_min=1.0)
    full = int(np.ceil(0.5 * cell_points(grid).max())) + 1
    years = [1957, *GAS_YEARS, GAS_YEARS[-1] + 1]
    dates = [gas_year(y)[0] for y in years]
    pts = []
    for i in range(full):  # cell 0 well covered; pressure falls 0.3 MPa a year after 2017
        lat, lon = 53.001 + 0.0004 * i, 6.401
        pts.append((lat, lon, [35.0, *[20.0 - 0.3 * k for k in range(len(years) - 1)]]))
    pts.append((53.06, 6.46, [35.0] * len(years)))  # one point only: not graded
    tab = _table(dates, pts)
    truth = build_truth(tab, grid)
    assert list(truth.cells) == [0]
    np.testing.assert_allclose(truth.loading[:, 0], 0.3, atol=1e-9)
    assert truth.static[0, 0] == pytest.approx(35.0 - (20.0 - 0.15))


def test_year_rates_average_the_periods_of_each_gas_year() -> None:
    starts = np.array([gas_year(y)[0] + k * 30 * DAY for y in GAS_YEARS for k in range(12)])
    rates = np.repeat(np.arange(len(GAS_YEARS), dtype=float), 12)[:, None] * np.ones((1, 3))
    out = year_rates(rates, starts)
    np.testing.assert_allclose(out[:, 0], np.arange(len(GAS_YEARS)))


def test_grade_ranks() -> None:
    n = 20
    load = np.tile(np.linspace(0, 1, n), (len(GAS_YEARS), 1))
    truth = Truth(np.arange(n), load, -load)
    assert grade(truth, load)["spearman"] == pytest.approx(1.0)
    assert grade(truth, load)["static"] == pytest.approx(-1.0)
    assert grade(truth, np.zeros_like(load))["spearman"] == 0.0
