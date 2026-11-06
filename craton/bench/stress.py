"""CB-23, hidden-state truth: which way is the crust being squeezed?

Nobody measures the stress inside a fault zone directly. What can be measured, here and there,
is its direction: the azimuth of the largest horizontal squeeze (SHmax), read from how quakes
slipped (focal mechanisms), from how boreholes break, from hydraulic fractures. The World
Stress Map collects every such reading on Earth (``craton.bench.truth``).

A model that has really inferred the crust's hidden state from quakes alone should know that
direction without ever being shown a single mechanism. CB-23 asks for it: for every 0.5° cell
of seven regions that holds at least three good WSM readings (quality A-C, crust above 40 km),
the method gives one SHmax azimuth, reading nothing but the region's quake catalogue before a
cut (where, when and how big; never how they slipped). The score is the angle between its
azimuth and the cell's WSM azimuth, 0° to 90° (axial: 10° and 190° are the same direction),
averaged over a region's cells and then over the regions. Lower is better; a coin gives 45°.

The regions are central Italy and CB-22's six. Central Italy and the three seen regions are the
ones Craton has trained on; Turkey, Chile and Alaska it never saw.

Floors:

- ``stress_regional``: one azimuth per region, the WSM mean of every **other** cell. The
  "regional-average stress" a seismologist quotes from memory. It reads the answers (of the
  other cells), so beating it from quakes alone means knowing something local.
- ``stress_interpolation``: the WSM azimuths of the other cells, weighted by distance (Gaussian,
  100 km). The strongest simple method, and the one that's hardest to beat.
- ``quake_lineation``: quakes only, as Craton: the long axis of the epicentre cloud around each
  cell, taken as SHmax. Right where faults strike along SHmax (normal faults), wrong by 90°
  where they strike across it (thrusts); the floor doesn't know which, and neither does
  anything that reads positions alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench.data import Events
from craton.bench.tasks import BenchContext, Score, Task, _write_table, register_task
from craton.bench.truth import QUALITY_SIGMA, WSM, StressRecords
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Cell size (degrees): the finest of WSM's own gridded products.
CELL_DEG = 0.5
#: A cell is scored if it holds at least this many WSM records.
MIN_RECORDS = 3
#: Records deeper than this (km) are left out: the brittle crust only, no slabs.
MAX_DEPTH_KM = 40.0
#: The angle a method that gives no answer (NaN) is charged: a coin's expected error.
NO_ANSWER_DEG = 45.0
SEEN, ZERO_SHOT = "seen", "zero_shot"
EARTH_KM = 6371.0


@dataclass(frozen=True)
class StressCase:
    """A region of CB-23 and the quakes a method may read: its box, before ``cut``."""

    region: str
    half: str
    cut: str
    mc: float  # history magnitudes from here (the region's completeness)


CASES = (
    # central Italy: the history CB-02's model may read (up to the sequence).
    StressCase("central_italy", SEEN, "2016-08-24T00:00:00Z", 2.0),
    # CB-22's regions, at its cut: the world Field trained on the seen ones to 2007.
    StressCase("california", SEEN, "2007-01-01T00:00:00Z", 2.5),
    StressCase("japan", SEEN, "2007-01-01T00:00:00Z", 3.0),
    StressCase("new_zealand", SEEN, "2007-01-01T00:00:00Z", 3.0),
    StressCase("turkey", ZERO_SHOT, "2007-01-01T00:00:00Z", 3.0),
    StressCase("chile", ZERO_SHOT, "2007-01-01T00:00:00Z", 4.5),
    StressCase("alaska", ZERO_SHOT, "2007-01-01T00:00:00Z", 3.0),
)


# ---------------------------------------------------------------- axial angles


def axial_mean(azimuth: FloatArray, weight: FloatArray | None = None) -> tuple[float, float]:
    """(mean direction in [0, 180), resultant length R in [0, 1]) of axial angles in degrees.

    Doubling the angles makes 0° and 180° one point on the circle; R is 1 when all agree and 0
    when they cancel.
    """
    a = np.radians(2.0 * np.asarray(azimuth, dtype=np.float64))
    w = np.ones_like(a) if weight is None else np.asarray(weight, dtype=np.float64)
    if len(a) == 0 or w.sum() <= 0:
        return math.nan, 0.0
    c, s = float((w * np.cos(a)).sum() / w.sum()), float((w * np.sin(a)).sum() / w.sum())
    return float(np.degrees(math.atan2(s, c)) / 2.0) % 180.0, math.hypot(c, s)


def axial_spread(r: float) -> float:
    """Circular standard deviation (degrees) of axial angles with resultant length ``r``."""
    if r <= 0:
        return 90.0
    return float(np.degrees(math.sqrt(-2.0 * math.log(min(r, 1.0))))) / 2.0


def axial_error(a: FloatArray, b: FloatArray) -> FloatArray:
    """The angle between two axial directions, 0° to 90°; NaN answers cost NO_ANSWER_DEG."""
    d = np.abs((np.asarray(a, dtype=np.float64) - b + 90.0) % 180.0 - 90.0)
    return np.where(np.isfinite(d), d, NO_ANSWER_DEG)


def km_between(lat0: float, lon0: float, lat: FloatArray, lon: FloatArray) -> FloatArray:
    p0, p = math.radians(lat0), np.radians(lat)
    dp, dl = p - p0, np.radians(lon - lon0)
    h = np.sin(dp / 2) ** 2 + math.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return np.asarray(2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1))), dtype=np.float64)


# ---------------------------------------------------------------- the truth, cell by cell


@dataclass(frozen=True)
class StressCells:
    """One region's scored cells: centre, WSM azimuth, how many records and how they agree."""

    lat: FloatArray
    lon: FloatArray
    azimuth: FloatArray
    n: np.ndarray[Any, np.dtype[np.int64]]
    spread_deg: FloatArray
    regime: list[str]

    def __len__(self) -> int:
        return len(self.lat)

    def table(self) -> pa.Table:
        return pa.table(
            {
                "lat": self.lat,
                "lon": self.lon,
                "azimuth": self.azimuth,
                "n": self.n,
                "spread_deg": self.spread_deg,
                "regime": self.regime,
            }
        )

    @staticmethod
    def from_table(t: pa.Table) -> StressCells:
        return StressCells(
            np.asarray(t.column("lat").to_numpy(), dtype=np.float64),
            np.asarray(t.column("lon").to_numpy(), dtype=np.float64),
            np.asarray(t.column("azimuth").to_numpy(), dtype=np.float64),
            np.asarray(t.column("n").to_numpy(), dtype=np.int64),
            np.asarray(t.column("spread_deg").to_numpy(), dtype=np.float64),
            [str(x) for x in t.column("regime").to_pylist()],
        )


def cells_of(
    rec: StressRecords, box: tuple[float, float, float, float], min_records: int = MIN_RECORDS
) -> StressCells:
    """Bin a box's records into CELL_DEG cells; keep cells with ``min_records`` or more.

    A cell's azimuth is the axial mean of its records weighted by 1/σ² of their quality.
    """
    lat0, lat1, lon0, lon1 = box
    keep = (
        (rec.lat >= lat0)
        & (rec.lat < lat1)
        & (rec.lon >= lon0)
        & (rec.lon < lon1)
        & ~(rec.depth_km > MAX_DEPTH_KM)
    )
    r = rec.select(keep)
    i = np.floor(r.lat / CELL_DEG + 1e-9).astype(np.int64)
    j = np.floor(r.lon / CELL_DEG + 1e-9).astype(np.int64)
    w = np.array([1.0 / QUALITY_SIGMA[q] ** 2 for q in r.quality], dtype=np.float64)
    out: dict[str, list[Any]] = {k: [] for k in ("lat", "lon", "azi", "n", "sd", "reg")}
    for key in sorted(set(zip(i.tolist(), j.tolist(), strict=True))):
        m = (i == key[0]) & (j == key[1])
        if m.sum() < min_records:
            continue
        azi, rr = axial_mean(r.azimuth[m], w[m])
        regimes, counts = np.unique(r.regime[m], return_counts=True)
        out["lat"].append((key[0] + 0.5) * CELL_DEG)
        out["lon"].append((key[1] + 0.5) * CELL_DEG)
        out["azi"].append(round(azi, 3))
        out["n"].append(int(m.sum()))
        out["sd"].append(round(axial_spread(rr), 3))
        out["reg"].append(str(regimes[np.argmax(counts)]))
    return StressCells(
        np.asarray(out["lat"], dtype=np.float64),
        np.asarray(out["lon"], dtype=np.float64),
        np.asarray(out["azi"], dtype=np.float64),
        np.asarray(out["n"], dtype=np.int64),
        np.asarray(out["sd"], dtype=np.float64),
        out["reg"],
    )


# ---------------------------------------------------------------- the protocol


@dataclass(frozen=True)
class StressQuery:
    """What a method gets: cells to answer and the quakes it may read.

    ``truth_*`` is filled only for methods that declare ``reads_truth_elsewhere`` (the WSM
    floors), and then never holds the cell being answered.
    """

    region: str
    half: str
    lat: FloatArray
    lon: FloatArray
    cut: str
    mc: float
    history: Events
    truth_lat: FloatArray = field(default_factory=lambda: np.zeros(0))
    truth_lon: FloatArray = field(default_factory=lambda: np.zeros(0))
    truth_azimuth: FloatArray = field(default_factory=lambda: np.zeros(0))


@runtime_checkable
class StressMapper(Protocol):
    name: str

    def stress_azimuths(self, query: StressQuery, seed: int) -> FloatArray:
        """SHmax azimuth (degrees east of north, axial) for every query cell; NaN = no idea."""
        ...


# ---------------------------------------------------------------- the floors


class StressRegional:
    """One direction per region: the WSM mean of every other cell."""

    name = "stress_regional"
    kind = "baseline"
    reads_truth_elsewhere = True

    def stress_azimuths(self, query: StressQuery, seed: int) -> FloatArray:
        azi, _ = axial_mean(query.truth_azimuth)
        return np.full(len(query.lat), azi)


class StressInterpolation:
    """The other cells' WSM azimuths, Gaussian-weighted by distance."""

    name = "stress_interpolation"
    kind = "baseline"
    reads_truth_elsewhere = True
    sigma_km = 100.0

    def stress_azimuths(self, query: StressQuery, seed: int) -> FloatArray:
        out = np.empty(len(query.lat))
        for k, (la, lo) in enumerate(zip(query.lat, query.lon, strict=True)):
            d = km_between(float(la), float(lo), query.truth_lat, query.truth_lon)
            w = np.exp(-0.5 * (d / self.sigma_km) ** 2)
            if w.sum() < 1e-6:  # nothing within ~500 km: the regional mean
                w = np.ones_like(d)
            out[k], _ = axial_mean(query.truth_azimuth, w)
        return out


class QuakeLineation:
    """Quakes only: the long axis of the epicentres within 50 km, taken as SHmax."""

    name = "quake_lineation"
    kind = "baseline"
    radius_km = 50.0
    min_quakes = 20

    def stress_azimuths(self, query: StressQuery, seed: int) -> FloatArray:
        ev = query.history.above(query.mc)
        out = np.full(len(query.lat), math.nan)
        for k, (la, lo) in enumerate(zip(query.lat, query.lon, strict=True)):
            d = km_between(float(la), float(lo), ev.lat, ev.lon)
            m = d <= self.radius_km
            if m.sum() < self.min_quakes:
                continue
            x = (ev.lon[m] - lo) * 111.195 * math.cos(math.radians(float(la)))
            y = (ev.lat[m] - la) * 111.195
            cov = np.cov(np.vstack([x, y]))
            vals, vecs = np.linalg.eigh(cov)
            ex, ey = vecs[:, int(np.argmax(vals))]
            out[k] = math.degrees(math.atan2(ex, ey)) % 180.0
        return out


# ---------------------------------------------------------------- the task


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


class StressTruth(Task):
    task_id = "CB-23"
    name = "Hidden-state truth"
    metric = "degrees between inferred and measured SHmax, mean over cells, then over regions"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "truth": f"{WSM.name} {WSM.version} sha256 {WSM.sha256}",
            "records": "quality A-C, azimuth 0-180, depth <= 40 km or not given",
            "cells": f"{CELL_DEG} deg, at least {MIN_RECORDS} records; azimuth = axial mean "
            "weighted by 1/sigma^2 (A 15, B 20, C 25 deg)",
            "cases": [c.__dict__ for c in CASES],
            "inputs": "the region's quakes inside its box before the cut, from its magnitude "
            "of completeness: positions, times, magnitudes; never mechanisms",
            "error": "axial angle 0-90 deg; no answer (NaN) counts 45",
            "primary": "mean over the regions of the mean error over the region's cells",
            "truth_floors": "methods reading other cells' truth are called once per cell, "
            "with that cell left out",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.adapters.region import load_region

        rec = WSM.records(ctx.root)
        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}, "wsm_records": len(rec)}
        for case in CASES:
            region = load_region(case.region, ctx.root)
            b = region.box
            view = self.view(ctx, case, "")
            summary["snapshots"][case.region] = view.snapshot_id
            cells = cells_of(rec, (b.lat_min, b.lat_max, b.lon_min, b.lon_max))
            files[f"{case.region}.parquet"] = _write_table(
                folder / f"{case.region}.parquet", cells.table()
            )
            summary[case.region] = {
                "cells": len(cells),
                "records": int(cells.n.sum()),
                "median_spread_deg": round(float(np.median(cells.spread_deg)), 1),
                "history_quakes": len(self.history(view, case)),
            }
        return {"files": files, "summary": summary}

    def supports(self, method: Any) -> bool:
        return isinstance(method, StressMapper)

    def frozen_snapshots(self, ctx: BenchContext) -> dict[str, str]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        out: dict[str, str] = task.get("summary", {}).get("snapshots", {})
        return out

    def view(self, ctx: BenchContext, case: StressCase, snapshot_id: str) -> Any:
        from craton.adapters.region import load_region
        from craton.bench.world import RegionView

        key = f"cb23-view:{case.region}"
        if key not in ctx._cache:
            b = load_region(case.region, ctx.root).box
            ctx._cache[key] = RegionView(
                root=ctx.root,
                region=case.region,
                snapshot_id=snapshot_id,
                box=(b.lat_min, b.lat_max, b.lon_min, b.lon_max),
                cut=case.cut,
            )
        return ctx._cache[key]

    @staticmethod
    def history(view: Any, case: StressCase) -> Events:
        ev: Events = view.events
        return ev.select(ev.time < to_seconds(case.cut)).above(case.mc)

    def cells(self, ctx: BenchContext, case: StressCase) -> StressCells:
        return StressCells.from_table(
            pq.read_table(ctx.task_dir(self.task_id) / f"{case.region}.parquet")
        )

    def answers(self, method: Any, ctx: BenchContext, case: StressCase, seed: int) -> FloatArray:
        """The method's azimuths for every scored cell of a case."""
        cells = self.cells(ctx, case)
        snap = self.frozen_snapshots(ctx).get(case.region, "")
        view = self.view(ctx, case, snap)
        if snap and view.snapshot_id != snap:
            raise ValueError(f"CB-23 was frozen on {case.region} snapshot {snap}")
        hist = self.history(view, case)
        q = StressQuery(
            region=case.region,
            half=case.half,
            lat=cells.lat,
            lon=cells.lon,
            cut=case.cut,
            mc=case.mc,
            history=hist,
        )
        if not getattr(method, "reads_truth_elsewhere", False):
            got = np.asarray(method.stress_azimuths(q, seed), dtype=np.float64)
        else:
            got = np.empty(len(cells))
            for k in range(len(cells)):
                others = np.arange(len(cells)) != k
                one = replace(
                    q,
                    lat=cells.lat[k : k + 1],
                    lon=cells.lon[k : k + 1],
                    truth_lat=cells.lat[others],
                    truth_lon=cells.lon[others],
                    truth_azimuth=cells.azimuth[others],
                )
                got[k] = np.asarray(method.stress_azimuths(one, seed), dtype=np.float64)[0]
        if got.shape != (len(cells),):
            raise ValueError(f"{method.name} gave {got.shape} azimuths for {len(cells)} cells")
        return got

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        extras: dict[str, float] = {}
        per: dict[str, list[float]] = {SEEN: [], ZERO_SHOT: []}
        for case in CASES:
            cells = self.cells(ctx, case)
            err = axial_error(self.answers(method, ctx, case, seed), cells.azimuth)
            mean = float(err.mean())
            per[case.half].append(mean)
            extras[f"{case.region}.deg"] = mean
            extras[f"{case.region}.within_20"] = float((err <= 20.0).mean())
            extras[f"{case.region}.cells"] = float(len(cells))
        extras[f"{SEEN}.deg"] = _mean(per[SEEN])
        extras[f"{ZERO_SHOT}.deg"] = _mean(per[ZERO_SHOT])
        return Score(_mean(per[SEEN] + per[ZERO_SHOT]), extras)


register_task(StressTruth())
