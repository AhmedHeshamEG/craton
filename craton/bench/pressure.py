"""CB-31, Groningen pressure: can a forecaster that reads only quakes see the reservoir drain?

Under Groningen, NAM's reservoir model gives the gas pressure on 6,871 points of 500 m, one map
a year (each 1 October, the start of a gas year). Craton reads it as CB-30's push (P8e). This
task asks the opposite question: a model that **never** reads it, only the quakes, forecasts
the field's next years, and its map of where quakes will come is graded against where the
pressure is falling.

**The truth.** Per CB-30 Groningen cell (0.05°) at least half covered by the reservoir model,
the mean pressure drop over one gas year, in MPa: the reservoir's loading that year. Six held-out
gas years, October 2017 to October 2023 (the model's later dates are projections). The drop is the
part that moves: production fell from 2014 and stopped in 2023, the centre of the field (the most
depleted part) levelled off and began to refill from the edges while the flanks still drained,
so the map of yearly drop turns against the map of total drop over these years (rank correlation
−0.02 in 2017-18, −0.66 in 2022-23, 86 cells). The total drop, which barely changes after 2017 (0.99
between years), is reported beside it as ``static``: a map anyone who saw the quakes before
2017 already knows.

**Reading a forecaster.** The expected count per cell over CB-30's 30-day periods that start in
the gas year (each reading only the quakes before its start), summed over magnitudes. Rate-and-
state ties the rate of quakes to the rate of loading (Dieterich 1994; Candela et al. 2019 for
Groningen), so a forecaster that has read the loading off the quakes ranks the cells as the drop
does. A method may answer directly instead (``PressureReader``).

**The rule (D-117).** A model graded here never reads the pressure. Every method answers
through a ``BlindView`` that refuses the reservoir-pressure tokens by source. A method built to
read the forcing (``forecast_forced``) answers only through its quake-only twin
(``quake_only_forecaster``); one that has no such twin, like ``etas_forced``, is not graded.

Score: Spearman's rank correlation of answer and yearly drop over the graded cells, averaged over
the six years. Higher is better; 0 is a map that knows nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from craton.bench.data import DAY, Events
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    ForecastSpec,
    Score,
    Task,
    _write_table,
    register_task,
)
from craton.bench.world import RegionView
from craton.spec.timeaxis import to_iso, to_seconds

if TYPE_CHECKING:
    from craton.bench.csep import Grid
    from craton.bench.induced import Case

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

CASE = "groningen"
INSTRUMENT = "reservoir_pressure"
#: Gas years graded: each from 1 October of that year to the next.
GAS_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
GAS_YEAR_START = "10-01"
#: A cell is graded when the reservoir model covers at least this share of it.
MIN_COVER = 0.5
#: Area of one model point (km²): the grid is 500 m.
POINT_KM2 = 0.25
#: The floor's look back.
DENSITY_DAYS = 365


def gas_year(y: int) -> tuple[float, float]:
    return to_seconds(f"{y}-{GAS_YEAR_START}T00:00:00Z"), to_seconds(
        f"{y + 1}-{GAS_YEAR_START}T00:00:00Z"
    )


class HiddenInstrumentError(PermissionError):
    """A graded model asked for an instrument CB-31 keeps from it."""


@dataclass
class BlindView(RegionView):
    """A region view that refuses some instruments' tokens: the no-pressure rule, by source."""

    hidden: frozenset[str] = field(default_factory=frozenset)

    def _snapshot_files(self, instrument: str) -> list[Path]:
        if instrument in self.hidden:
            raise HiddenInstrumentError(
                f"{instrument} is the truth of CB-31; a graded model never reads it"
            )
        return super()._snapshot_files(instrument)


def blind_view(root: Path, case: Case, spec: ForecastSpec, snapshot_id: str) -> BlindView:
    g = spec.grid
    return BlindView(
        root=root,
        region=case.region,
        snapshot_id=snapshot_id,
        box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
        cut=spec.fit_end,
        hidden=frozenset({INSTRUMENT}),
    )


# ---------------------------------------------------------------- the truth


def cell_points(grid: Grid) -> FloatArray:
    """(cells,) how many 500 m points fill each cell completely."""
    lat_c = grid.lat_min + grid.cell_deg * (np.arange(grid.n_cells) // grid.n_lon + 0.5)
    km2 = (grid.cell_deg * 111.195) ** 2 * np.cos(np.radians(lat_c))
    return np.asarray(km2 / POINT_KM2, dtype=np.float64)


def pressure_maps(table: Any, grid: Grid) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(dates,), (dates, cells) mean pressure in MPa over each cell's points, (cells,) points."""
    t = np.asarray(table.column("time").to_numpy(), dtype=np.float64)
    lat = np.asarray(table.column("lat").to_numpy(), dtype=np.float64)
    lon = np.asarray(table.column("lon").to_numpy(), dtype=np.float64)
    p = np.array([v[0] for v in table.column("value").to_pylist()], dtype=np.float64)
    cell = grid.cell_of(lat, lon)
    ok = cell >= 0
    dates = np.unique(t)
    i = np.searchsorted(dates, t[ok])
    s = np.zeros((len(dates), grid.n_cells))
    n = np.zeros((len(dates), grid.n_cells))
    np.add.at(s, (i, cell[ok]), p[ok])
    np.add.at(n, (i, cell[ok]), 1.0)
    pts = n.max(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return dates, mean, pts


@dataclass(frozen=True)
class Truth:
    cells: np.ndarray  # (graded,) cell indices
    loading: FloatArray  # (years, graded) MPa lost over the gas year
    static: FloatArray  # (years, graded) MPa lost since the first date, mid gas year


def build_truth(table: Any, grid: Grid) -> Truth:
    dates, mean, pts = pressure_maps(table, grid)
    graded = np.nonzero(pts >= MIN_COVER * cell_points(grid))[0]
    p0 = mean[0, graded]
    loading, static = [], []
    for y in GAS_YEARS:
        a, b = gas_year(y)
        ia, ib = np.searchsorted(dates, a), np.searchsorted(dates, b)
        if ia >= len(dates) or ib >= len(dates) or dates[ia] != a or dates[ib] != b:
            raise ValueError(f"the pressure model has no map on {to_iso(a)} or {to_iso(b)}")
        pa_, pb = mean[ia, graded], mean[ib, graded]
        loading.append(pa_ - pb)
        static.append(p0 - (pa_ + pb) / 2)
    if np.isnan(np.array(loading)).any():
        raise ValueError("a graded cell lost its points between dates")
    return Truth(graded, np.array(loading), np.array(static))


# ---------------------------------------------------------------- reading a method


class PressureQuery:
    """What a ``PressureReader`` gets: the box's quakes (read only before each gas year), the
    gas years and the grid."""

    def __init__(self, events: Events, grid: Grid, mc: float) -> None:
        self.events = events
        self.grid = grid
        self.mc = mc
        self.starts = np.array([gas_year(y)[0] for y in GAS_YEARS])


@runtime_checkable
class PressureReader(Protocol):
    name: str

    def pressure_map(self, query: PressureQuery, seed: int) -> FloatArray:
        """(gas years, cells) any monotone reading of each cell's loading that year."""
        ...


def year_rates(rates: FloatArray, starts: FloatArray) -> FloatArray:
    """(gas years, cells) mean of the 30-day periods' total rate whose start is in each year."""
    out = []
    for y in GAS_YEARS:
        a, b = gas_year(y)
        m = (starts >= a) & (starts < b)
        if not m.any():
            raise ValueError(f"no CB-30 period starts in gas year {y}")
        out.append(rates[m].mean(axis=0))
    return np.array(out)


def recent_quakes(query: PressureQuery) -> FloatArray:
    """CB-31's floor: each cell's quakes of the year before the gas year, M ≥ mc."""
    ev = query.events.above(query.mc - 1e-9)
    cell = query.grid.cell_of(ev.lat, ev.lon)
    out = np.zeros((len(query.starts), query.grid.n_cells))
    for i, t in enumerate(query.starts):
        m = (ev.time < t) & (ev.time >= t - DENSITY_DAYS * DAY) & (cell >= 0)
        out[i] = np.log(np.bincount(cell[m], minlength=query.grid.n_cells) + 0.5)
    return out


def grade(truth: Truth, answer: FloatArray) -> dict[str, float]:
    def rho(a: FloatArray, b: FloatArray) -> float:
        if np.ptp(a) == 0:
            return 0.0  # a flat answer ranks nothing
        r = float(spearmanr(a, b)[0])
        return 0.0 if math.isnan(r) else r

    a = answer[:, truth.cells]
    load = [rho(x, t) for x, t in zip(a, truth.loading, strict=True)]
    stat = [rho(x, t) for x, t in zip(a, truth.static, strict=True)]
    out = {"static": float(np.mean(stat)), "cells": float(len(truth.cells))}
    out |= {f"loading.{y}": v for y, v in zip(GAS_YEARS, load, strict=True)}
    return {"spearman": float(np.mean(load)), **out}


class GroningenPressure(Task):
    task_id = "CB-31"
    name = "Groningen pressure"
    metric = "rank correlation of a quake-only forecast map and the yearly reservoir pressure drop"
    direction = "higher"

    @property
    def case(self) -> Case:
        # imported on use: CB-30's module may still be loading when this one registers
        from craton.bench.induced import CASE_BY_KEY

        return CASE_BY_KEY[CASE]

    @property
    def spec(self) -> ForecastSpec:
        from craton.bench.induced import case_spec

        return case_spec(self.case)

    def definition(self) -> dict[str, Any]:
        return {
            "case": CASE,
            "grid": self.spec.grid.__dict__,
            "truth": "mean reservoir pressure drop (MPa) over each gas year, per 0.05 deg cell, "
            f"NAM reservoir model V1 2023 ({INSTRUMENT} tokens)",
            "gas_years": [f"{y}-{GAS_YEAR_START} to {y + 1}-{GAS_YEAR_START}" for y in GAS_YEARS],
            "graded_cells": f"at least {MIN_COVER:g} of the cell covered by 500 m model points",
            "answer": "expected count per cell, mean of the CB-30 periods starting in the gas "
            "year, all magnitudes; or a PressureReader's map",
            "metric": "Spearman per gas year over graded cells, averaged",
            "also": "static: the same against the total drop since 1957 (mid gas year)",
            "rule": "a graded method never reads the reservoir_pressure tokens (D-117); a "
            "forcing reader answers through its quake-only twin",
        }

    def _snapshot(self, ctx: BenchContext) -> str:
        from craton.bench.induced import InducedQuakes

        return InducedQuakes().frozen_snapshots(ctx).get(self.case.region, "")

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.forcing import _table
        from craton.bench.world import view_for

        spec = self.spec
        view = view_for(ctx.root, self.case.region, spec, self._snapshot(ctx))
        truth = build_truth(_table(view, INSTRUMENT), spec.grid)
        cols = {"cell": truth.cells.astype(np.int64)}
        for i, y in enumerate(GAS_YEARS):
            cols[f"loading_{y}"] = truth.loading[i]
            cols[f"static_{y}"] = truth.static[i]
        sha = _write_table(ctx.task_dir(self.task_id) / "truth.parquet", pa.table(cols))
        r = [float(spearmanr(truth.loading[i], truth.static[i])[0]) for i in range(len(GAS_YEARS))]
        return {
            "files": {"truth.parquet": sha},
            "summary": {
                "snapshot": view.snapshot_id,
                "graded_cells": len(truth.cells),
                "loading_mpa": {
                    str(y): [
                        round(float(v), 3) for v in np.percentile(truth.loading[i], [5, 50, 95])
                    ]
                    for i, y in enumerate(GAS_YEARS)
                },
                "loading_vs_static": {
                    str(y): round(v, 3) for y, v in zip(GAS_YEARS, r, strict=True)
                },
            },
        }

    def truth(self, ctx: BenchContext) -> Truth:
        t = pq.read_table(ctx.task_dir(self.task_id) / "truth.parquet")

        def col(n: str) -> FloatArray:
            return np.asarray(t.column(n).to_numpy(), dtype=np.float64)

        return Truth(
            np.asarray(t.column("cell").to_numpy(), dtype=np.int64),
            np.stack([col(f"loading_{y}") for y in GAS_YEARS]),
            np.stack([col(f"static_{y}") for y in GAS_YEARS]),
        )

    def supports(self, method: Any) -> bool:
        if isinstance(method, PressureReader):
            return True
        if callable(getattr(method, "quake_only_forecaster", None)):
            return True
        if hasattr(method, "forecast_forced"):
            return False  # built to read the forcing, with no quake-only twin
        return (
            isinstance(method, Forecaster)
            and getattr(method, "long_periods", False)
            and not getattr(method, "regional_only", False)
        )

    def answer(self, method: Any, ctx: BenchContext, seed: int) -> FloatArray:
        case = self.case
        spec = self.spec
        view = blind_view(ctx.root, case, spec, self._snapshot(ctx))
        if isinstance(method, PressureReader):
            out = np.asarray(
                method.pressure_map(PressureQuery(view.events, spec.grid, spec.mc), seed)
            )
        else:
            twin = getattr(method, "quake_only_forecaster", None)
            if callable(twin):
                fc = twin().forecast_forced(spec, view, case.forcing, seed)
            else:
                fc = method.forecast(spec, view, seed)
            out = year_rates(np.asarray(fc.rates, dtype=np.float64), spec.day_starts)
        if out.shape != (len(GAS_YEARS), spec.grid.n_cells):
            raise ValueError(f"{method.name}: CB-31 answer has shape {out.shape}")
        return out

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        g = grade(self.truth(ctx), self.answer(method, ctx, seed))
        return Score(g.pop("spearman"), g)


register_task(GroningenPressure())
