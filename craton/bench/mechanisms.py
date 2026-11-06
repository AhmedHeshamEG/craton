"""CB-34, stress from mechanisms: how quakes slipped, graded on stress nobody read from a quake.

CB-23 asks for the direction of the crust's largest horizontal squeeze (SHmax) from where,
when and how big quakes were. Positions alone can't tell a thrust from a normal fault (D-107),
and every quake-only method so far sits near the lineation floor's 45°. How a quake slipped
does tell: a focal mechanism's pressure axis points roughly along the squeeze. So CB-34 gives a
method the moment tensors too (the Global CMT catalogue, ``craton.adapters.gcmt``, every
solution before the cut inside the region's box), and grades it only on World Stress Map
records that were **not** read from quakes: borehole breakouts, drilling-induced fractures,
hydraulic fracturing, overcoring, fault-slip data and volcanic alignments. The answer never
reads its own input (D-116): WSM's quake-based records are GCMT's and the regional networks'
mechanisms, and they are left out of the truth entirely.

Cells are CB-23's 0.5° cells, the same seven regions and cuts, with **one** non-quake record
enough (they are rare: 5 cells in Alaska, 77 in California; D-132). The score is the same
axial angle, 0-90°, mean over a region's cells and then over the regions.

Floors:

- ``stress_regional`` and ``stress_interpolation`` (CB-23's, reading the other cells'
  non-quake answers, one cell left out at a time);
- ``quake_lineation`` (CB-23's, positions only);
- ``mechanism_kostrov``: the textbook answer from mechanisms. Sum the tensors near the cell,
  each scaled to unit moment (Kostrov 1974, so one big quake doesn't speak for all), and take
  the horizontal direction in which the summed tensor squeezes most (Lund & Townend 2007's
  SHmax of a full tensor, with stress read as minus the moment tensor). No learning, no truth.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench.data import read_table
from craton.bench.stress import (
    CASES,
    CELL_DEG,
    MAX_DEPTH_KM,
    SEEN,
    ZERO_SHOT,
    StressCase,
    StressCells,
    StressMapper,
    StressQuery,
    StressTruth,
    axial_error,
    cells_of,
    km_between,
)
from craton.bench.tasks import BenchContext, Score, Task, _write_table, register_task
from craton.bench.truth import WSM, StressRecords
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: The region whose snapshot holds the moment tensors (``configs/regions/gcmt_world.yaml``).
MECHANISM_REGION = "gcmt_world"
#: WSM record types read from quakes: single mechanisms, averages and formal inversions.
QUAKE_KINDS = ("FMS", "FMA", "FMF")
#: One non-quake record makes a cell (D-132).
MIN_RECORDS = 1
TENSOR_COLUMNS = ("time", "lat", "lon", "depth_km", "value", "qualifiers")


# ---------------------------------------------------------------- the tensors


@dataclass(frozen=True)
class Mechanisms:
    """Moment tensors: centroid, time, the six elements (rr, tt, pp, rt, rp, tp) and M0, N·m."""

    lat: FloatArray
    lon: FloatArray
    depth_km: FloatArray
    time: FloatArray
    m: FloatArray  # (n, 6)
    m0: FloatArray
    event_id: np.ndarray[Any, np.dtype[np.str_]] = field(
        default_factory=lambda: np.zeros(0, dtype=np.str_)
    )

    def __len__(self) -> int:
        return len(self.lat)

    def select(self, mask: np.ndarray[Any, Any]) -> Mechanisms:
        ids = self.event_id[mask] if len(self.event_id) == len(self.lat) else self.event_id
        return Mechanisms(
            self.lat[mask],
            self.lon[mask],
            self.depth_km[mask],
            self.time[mask],
            self.m[mask],
            self.m0[mask],
            ids,
        )

    def inside(self, box: tuple[float, float, float, float], before: float) -> Mechanisms:
        lat0, lat1, lon0, lon1 = box
        return self.select(
            (self.lat >= lat0)
            & (self.lat < lat1)
            & (self.lon >= lon0)
            & (self.lon < lon1)
            & (self.time < before)
        )

    @property
    def unit(self) -> FloatArray:
        """Each tensor divided by its scalar moment: the shape of the slip, not its size."""
        return np.asarray(self.m / self.m0[:, None], dtype=np.float64)


def mechanisms_from_table(t: pa.Table) -> Mechanisms:
    v = np.asarray(t.column("value").to_pylist(), dtype=np.float64).reshape(-1, 7)
    ids = [json.loads(q)["event_id"] for q in t.column("qualifiers").to_pylist()]
    mech = Mechanisms(
        np.asarray(t.column("lat").to_numpy(), dtype=np.float64),
        np.asarray(t.column("lon").to_numpy(), dtype=np.float64),
        np.asarray(t.column("depth_km").to_numpy(), dtype=np.float64),
        np.asarray(t.column("time").to_numpy(), dtype=np.float64),
        v[:, :6],
        v[:, 6],
        np.asarray(ids, dtype=np.str_),
    )
    return mech.select(np.argsort(mech.time, kind="stable"))


def load_mechanisms(root: Path, snapshot_id: str = "") -> tuple[Mechanisms, str]:
    """Every moment tensor of the mechanism snapshot (the frozen one, or the latest)."""
    ctx = BenchContext(root=root, region=MECHANISM_REGION, snapshot_id=snapshot_id)
    files = ctx._snapshot_files("moment_tensor")
    return mechanisms_from_table(read_table(files, TENSOR_COLUMNS)), ctx.snapshot_id


def horizontal(m: FloatArray) -> FloatArray:
    """The horizontal 2×2 block (north, east) of tensors in GCMT's (r, t, p) frame, as
    (..., 3): NN = tt, EE = pp, NE = −tp (t points south)."""
    m = np.asarray(m, dtype=np.float64)
    return np.stack([m[..., 1], m[..., 2], -m[..., 5]], axis=-1)


def shmax_of(m: FloatArray) -> FloatArray:
    """SHmax azimuth (degrees east of north, 0-180) of moment tensors (..., 6).

    Stress is read as minus the moment tensor (a squeeze shortens the rock: negative moment
    along the pressure axis). Lund & Townend (2007) show SHmax of a full tensor is the
    horizontal direction of the largest horizontal normal stress, which for −M is the
    direction where n·M·n is smallest: the angle α with tan 2α = 2 NE / (NN − EE) at the
    minimum. NaN where the horizontal block is isotropic (no direction).
    """
    h = horizontal(m)
    nn, ee, ne = h[..., 0], h[..., 1], h[..., 2]
    # n·M·n = (NN+EE)/2 + (NN−EE)/2 cos 2α + NE sin 2α, smallest at 2α = atan2(−NE, −(NN−EE)/2).
    two_a = np.arctan2(-ne, -(nn - ee) / 2.0)
    out = np.degrees(two_a / 2.0) % 180.0
    flat = np.hypot((nn - ee) / 2.0, ne) <= 1e-12 * (np.abs(nn) + np.abs(ee) + 1e-30)
    return np.asarray(np.where(flat, math.nan, out), dtype=np.float64)


# ---------------------------------------------------------------- the protocol


@dataclass(frozen=True)
class MechanismQuery(StressQuery):
    """CB-23's query plus the moment tensors before the cut inside the region's box."""

    mechanisms: Mechanisms = field(
        default_factory=lambda: Mechanisms(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 6)), np.zeros(0)
        )
    )


@runtime_checkable
class MechanismStressMapper(Protocol):
    name: str

    def mechanism_stress_azimuths(self, query: MechanismQuery, seed: int) -> FloatArray:
        """SHmax azimuth for every query cell, reading the mechanisms; NaN = no idea."""
        ...


class MechanismKostrov:
    """Unit-moment tensors summed with a Gaussian of distance; SHmax of the sum."""

    name = "mechanism_kostrov"
    kind = "baseline"
    sigma_km = 50.0
    #: Mechanisms farther than this many σ are left out; a cell with none answers NaN.
    reach = 3.0
    max_depth_km = MAX_DEPTH_KM

    def mechanism_stress_azimuths(self, query: MechanismQuery, seed: int) -> FloatArray:
        mech = query.mechanisms
        mech = mech.select(~(mech.depth_km > self.max_depth_km))
        unit = mech.unit
        out = np.full(len(query.lat), math.nan)
        for k, (la, lo) in enumerate(zip(query.lat, query.lon, strict=True)):
            d = km_between(float(la), float(lo), mech.lat, mech.lon)
            near = d <= self.reach * self.sigma_km
            if not near.any():
                continue
            w = np.exp(-0.5 * (d[near] / self.sigma_km) ** 2)
            out[k] = float(shmax_of((w[:, None] * unit[near]).sum(axis=0)))
        return out


# ---------------------------------------------------------------- the truth


def non_quake(rec: StressRecords) -> StressRecords:
    """The records that were not read from earthquakes."""
    return rec.select(~np.isin(rec.kind, QUAKE_KINDS))


def non_quake_cells(rec: StressRecords, box: tuple[float, float, float, float]) -> StressCells:
    return cells_of(non_quake(rec), box, MIN_RECORDS)


# ---------------------------------------------------------------- the task


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


def _box(region: str, root: Path) -> tuple[float, float, float, float]:
    from craton.adapters.region import load_region

    b = load_region(region, root).box
    return (b.lat_min, b.lat_max, b.lon_min, b.lon_max)


class StressFromMechanisms(Task):
    task_id = "CB-34"
    name = "Stress from mechanisms"
    metric = (
        "degrees between inferred SHmax and non-quake World Stress Map records, mean over "
        "cells, then over regions"
    )
    direction = "lower"

    #: CB-23's views, so the quake history a method reads is the same.
    _views = StressTruth()

    def definition(self) -> dict[str, Any]:
        return {
            "truth": f"{WSM.name} {WSM.version} sha256 {WSM.sha256}",
            "records": "quality A-C, azimuth 0-180, depth <= 40 km or not given, and not "
            f"read from quakes (types other than {', '.join(QUAKE_KINDS)})",
            "cells": f"{CELL_DEG} deg, at least {MIN_RECORDS} record; azimuth = axial mean "
            "weighted by 1/sigma^2 (A 15, B 20, C 25 deg)",
            "cases": [c.__dict__ for c in CASES],
            "inputs": "CB-23's quake history, plus every Global CMT moment tensor inside the "
            "region's box before the cut (snapshot of gcmt_world)",
            "error": "axial angle 0-90 deg; no answer (NaN) counts 45",
            "primary": "mean over the regions of the mean error over the region's cells",
            "truth_floors": "methods reading other cells' truth are called once per cell, "
            "with that cell left out",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        rec = WSM.records(ctx.root)
        mech, mech_snap = load_mechanisms(ctx.root, self.mechanism_snapshot(ctx))
        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {
            "snapshots": {},
            "mechanism_snapshot": mech_snap,
            "non_quake_records": len(non_quake(rec)),
        }
        for case in CASES:
            box = _box(case.region, ctx.root)
            view = self._views.view(ctx, case, "")
            summary["snapshots"][case.region] = view.snapshot_id
            cells = non_quake_cells(rec, box)
            files[f"{case.region}.parquet"] = _write_table(
                folder / f"{case.region}.parquet", cells.table()
            )
            m = mech.inside(box, to_seconds(case.cut))
            summary[case.region] = {
                "cells": len(cells),
                "records": int(cells.n.sum()),
                "mechanisms": len(m),
                "crustal_mechanisms": int((~(m.depth_km > MAX_DEPTH_KM)).sum()),
                "history_quakes": len(self._views.history(view, case)),
            }
        return {"files": files, "summary": summary}

    def supports(self, method: Any) -> bool:
        return isinstance(method, (MechanismStressMapper, StressMapper))

    def _frozen_summary(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        out: dict[str, Any] = frozen.get("tasks", {}).get(self.task_id, {}).get("summary", {})
        return out

    def mechanism_snapshot(self, ctx: BenchContext) -> str:
        return str(self._frozen_summary(ctx).get("mechanism_snapshot", ""))

    def mechanisms(self, ctx: BenchContext) -> Mechanisms:
        key = "cb34-mechanisms"
        if key not in ctx._cache:
            snap = self.mechanism_snapshot(ctx)
            mech, got = load_mechanisms(ctx.root, snap)
            if snap and got != snap:
                raise ValueError(f"CB-34 was frozen on gcmt_world snapshot {snap}")
            ctx._cache[key] = mech
        out: Mechanisms = ctx._cache[key]
        return out

    def cells(self, ctx: BenchContext, case: StressCase) -> StressCells:
        return StressCells.from_table(
            pq.read_table(ctx.task_dir(self.task_id) / f"{case.region}.parquet")
        )

    def query(self, ctx: BenchContext, case: StressCase) -> MechanismQuery:
        """What a method reads for a case: the cells, the quakes and the tensors before the cut."""
        cells = self.cells(ctx, case)
        snap = self._frozen_summary(ctx).get("snapshots", {}).get(case.region, "")
        view = self._views.view(ctx, case, snap)
        if snap and view.snapshot_id != snap:
            raise ValueError(f"CB-34 was frozen on {case.region} snapshot {snap}")
        box = _box(case.region, ctx.root)
        return MechanismQuery(
            region=case.region,
            half=case.half,
            lat=cells.lat,
            lon=cells.lon,
            cut=case.cut,
            mc=case.mc,
            history=self._views.history(view, case),
            mechanisms=self.mechanisms(ctx).inside(box, to_seconds(case.cut)),
        )

    def answers(self, method: Any, ctx: BenchContext, case: StressCase, seed: int) -> FloatArray:
        cells = self.cells(ctx, case)
        q = self.query(ctx, case)

        def ask(one: MechanismQuery) -> FloatArray:
            if isinstance(method, MechanismStressMapper):
                return np.asarray(method.mechanism_stress_azimuths(one, seed), dtype=np.float64)
            return np.asarray(method.stress_azimuths(one, seed), dtype=np.float64)

        if not getattr(method, "reads_truth_elsewhere", False):
            got = ask(q)
        else:
            got = np.empty(len(cells))
            for k in range(len(cells)):
                others = np.arange(len(cells)) != k
                got[k] = ask(
                    replace(
                        q,
                        lat=cells.lat[k : k + 1],
                        lon=cells.lon[k : k + 1],
                        truth_lat=cells.lat[others],
                        truth_lon=cells.lon[others],
                        truth_azimuth=cells.azimuth[others],
                    )
                )[0]
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


register_task(StressFromMechanisms())
