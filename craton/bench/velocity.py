"""CB-28, velocity gap-fill: hide a station, rebuild its dv/v.

``seismic_velocity_change`` (P10b) measures how fast the crust between two seismometers is,
day by day, relative to a quiet reference period. CB-28 hides one station at a time: every
pair it belongs to loses its values after the reference period, and a method rebuilds them
from everything else (the other pairs on every day, the hidden pairs' own reference months).

That asks the question the instrument has to answer before a forecaster may lean on it: is
the change a property of the crust, shared by neighbouring paths, or noise that belongs to one
pair? If the neighbours can't rebuild it, a model reading it learns noise.

Cases: ``central_italy_noise`` (2016-17: Amatrice, Visso, Norcia, Campotosto) and
``laquila_noise`` (2009), the 10-day stacks, corrected ``dvv``. A station is a fold when at
least one of its pairs has test days and at least two other pairs stay visible.

Score: RMSE of the rebuilt dv/v on the hidden pairs' test days, in percent, mean over the
folds of a region, then over the regions. A NaN answer counts as 0 ("nothing changed").

Floors:

- ``velocity_zero``: dv/v = 0, the reference state; what a model gets for saying nothing;
- ``velocity_neighbours``: each hidden pair's day is the mean of the visible pairs that day,
  weighted by 1/σ² and a Gaussian of the distance between the pairs' midpoints (30 km).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench.data import read_table
from craton.bench.tasks import BenchContext, Score, Task, _write_table, register_task

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

INSTRUMENT = "seismic_velocity_change"
REGIONS = ("central_italy_noise", "laquila_noise")
STACK = "10"
MIN_TEST_DAYS = 30
MIN_VISIBLE_PAIRS = 2
COLUMNS = ("lat", "lon", "time", "value", "uncertainty", "qualifiers")


@dataclass(frozen=True)
class VelocitySeries:
    """One region's dv/v: ``dvv`` and ``sigma`` (P, D), NaN where a pair had no value."""

    region: str
    station_a: list[str]
    station_b: list[str]
    lat: FloatArray
    lon: FloatArray
    days: FloatArray
    dvv: FloatArray
    sigma: FloatArray
    reference_end: float

    def table(self) -> pa.Table:
        p, d = self.dvv.shape
        return pa.table(
            {
                "station_a": np.repeat(self.station_a, d).tolist(),
                "station_b": np.repeat(self.station_b, d).tolist(),
                "lat": np.repeat(self.lat, d),
                "lon": np.repeat(self.lon, d),
                "time": np.tile(self.days, p),
                "dvv": self.dvv.ravel(),
                "sigma": self.sigma.ravel(),
                "reference_end": np.full(p * d, self.reference_end),
            }
        )

    @classmethod
    def from_table(cls, region: str, t: pa.Table) -> VelocitySeries:
        a = t.column("station_a").to_pylist()
        b = t.column("station_b").to_pylist()
        keys = list(dict.fromkeys(zip(a, b, strict=True)))
        days = np.unique(np.asarray(t.column("time").to_numpy(), dtype=np.float64))
        shape = (len(keys), len(days))
        lat = np.asarray(t.column("lat").to_numpy(), dtype=np.float64).reshape(shape)[:, 0]
        lon = np.asarray(t.column("lon").to_numpy(), dtype=np.float64).reshape(shape)[:, 0]
        return cls(
            region=region,
            station_a=[k[0] for k in keys],
            station_b=[k[1] for k in keys],
            lat=lat,
            lon=lon,
            days=days,
            dvv=np.asarray(t.column("dvv").to_numpy(), dtype=np.float64).reshape(shape),
            sigma=np.asarray(t.column("sigma").to_numpy(), dtype=np.float64).reshape(shape),
            reference_end=float(t.column("reference_end")[0].as_py()),
        )

    def stations(self) -> list[str]:
        return sorted(set(self.station_a) | set(self.station_b))

    def involves(self, station: str) -> BoolArray:
        return np.array(
            [station in (a, b) for a, b in zip(self.station_a, self.station_b, strict=True)]
        )


def series_from_tokens(region: str, t: pa.Table) -> VelocitySeries:
    """The 10-day corrected series of every pair, on one day grid."""
    rows: dict[tuple[str, str], dict[float, tuple[float, float]]] = {}
    where: dict[tuple[str, str], tuple[float, float]] = {}
    ref_end = math.nan
    for lat, lon, time, value, unc, q in zip(
        *(t.column(c).to_pylist() for c in COLUMNS), strict=True
    ):
        qual = json.loads(q)
        if qual["stack_days"] != STACK:
            continue
        key = (qual["station_a"], qual["station_b"])
        rows.setdefault(key, {})[time] = (value[0], unc[0])
        where[key] = (lat, lon)
        end = qual["reference"].split("/")[1]
        ref_end = float(np.datetime64(end + "T00:00:00", "s").astype(np.int64))
    keys = sorted(rows)
    days = np.array(sorted({d for r in rows.values() for d in r}), dtype=np.float64)
    col = {d: i for i, d in enumerate(days)}
    dvv = np.full((len(keys), len(days)), np.nan)
    sig = np.full((len(keys), len(days)), np.nan)
    for p, k in enumerate(keys):
        for d, (v, s) in rows[k].items():
            dvv[p, col[d]] = v
            sig[p, col[d]] = s
    return VelocitySeries(
        region,
        [k[0] for k in keys],
        [k[1] for k in keys],
        np.array([where[k][0] for k in keys]),
        np.array([where[k][1] for k in keys]),
        days,
        dvv,
        sig,
        ref_end,
    )


@dataclass(frozen=True)
class VelocityQuery:
    """What a method sees for one fold: every pair, the hidden ones blank after the reference."""

    region: str
    station: str
    series: VelocitySeries
    hidden: BoolArray


@runtime_checkable
class VelocityGapFiller(Protocol):
    name: str

    def fill_velocity(self, query: VelocityQuery, seed: int) -> FloatArray:
        """dv/v for the hidden pairs, (hidden pairs, days); only the test days are graded."""
        ...


def km(lat1: float, lon1: float, lat2: FloatArray, lon2: FloatArray) -> FloatArray:
    la1, lo1 = math.radians(lat1), math.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    h = np.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    out: FloatArray = 6371.0 * 2 * np.arcsin(np.sqrt(h))
    return out


class VelocityZero:
    """Nothing changed since the reference."""

    name = "velocity_zero"
    kind = "baseline"

    def fill_velocity(self, query: VelocityQuery, seed: int) -> FloatArray:
        return np.zeros((int(query.hidden.sum()), len(query.series.days)))


class VelocityNeighbours:
    """The visible pairs' dv/v that day, weighted by 1/σ² and closeness of midpoints."""

    name = "velocity_neighbours"
    kind = "baseline"
    sigma_km = 30.0

    def fill_velocity(self, query: VelocityQuery, seed: int) -> FloatArray:
        s = query.series
        vis = ~query.hidden
        v, sg = s.dvv[vis], s.sigma[vis]
        out = []
        for p in np.flatnonzero(query.hidden):
            d = km(float(s.lat[p]), float(s.lon[p]), s.lat[vis], s.lon[vis])
            w = np.exp(-0.5 * (d / self.sigma_km) ** 2)[:, None] / sg**2
            w = np.where(np.isfinite(v) & np.isfinite(w), w, 0.0)
            tot = w.sum(axis=0)
            out.append(
                np.where(tot > 0, (np.nan_to_num(v) * w).sum(axis=0) / np.maximum(tot, 1e-30), 0.0)
            )
        return np.array(out).reshape(-1, len(s.days))


def folds(s: VelocitySeries) -> list[str]:
    """Stations that can be hidden: a test day on one of their pairs, two other pairs left."""
    test = s.days >= s.reference_end
    out = []
    for st in s.stations():
        hid = s.involves(st)
        if (~hid).sum() < MIN_VISIBLE_PAIRS:
            continue
        if np.isfinite(s.dvv[hid][:, test]).sum() < MIN_TEST_DAYS:
            continue
        out.append(st)
    return out


def blank(s: VelocitySeries, hidden: BoolArray) -> VelocitySeries:
    """The series with the hidden pairs' days after the reference set to NaN."""
    test = s.days >= s.reference_end
    dvv, sig = s.dvv.copy(), s.sigma.copy()
    dvv[np.ix_(hidden, test)] = np.nan
    sig[np.ix_(hidden, test)] = np.nan
    return VelocitySeries(
        s.region, s.station_a, s.station_b, s.lat, s.lon, s.days, dvv, sig, s.reference_end
    )


class VelocityGapFill(Task):
    task_id = "CB-28"
    name = "Velocity gap-fill"
    metric = "RMSE (%) of a hidden station's dv/v, rebuilt from the other pairs"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "regions": list(REGIONS),
            "instrument": f"{INSTRUMENT}, stack_days {STACK}, channel dvv (corrected)",
            "fold": f"one station: every pair holding it is hidden after the reference; needs "
            f">= {MIN_TEST_DAYS} test values and >= {MIN_VISIBLE_PAIRS} other pairs",
            "inputs": "every other pair on every day; the hidden pairs before the reference ends",
            "error": "RMSE in percent over the hidden pairs' test days with a value; NaN = 0",
            "primary": "mean over regions of the mean over folds",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}}
        for region in REGIONS:
            view = BenchContext(root=ctx.root, region=region)
            s = series_from_tokens(region, read_table(view._snapshot_files(INSTRUMENT), COLUMNS))
            files[f"{region}.parquet"] = _write_table(folder / f"{region}.parquet", s.table())
            test = s.days >= s.reference_end
            summary["snapshots"][region] = view.snapshot_id
            summary[region] = {
                "pairs": len(s.station_a),
                "stations": len(s.stations()),
                "folds": folds(s),
                "test_values": int(np.isfinite(s.dvv[:, test]).sum()),
            }
        return {"files": files, "summary": summary}

    def series(self, ctx: BenchContext, region: str) -> VelocitySeries:
        path: Path = ctx.task_dir(self.task_id) / f"{region}.parquet"
        return VelocitySeries.from_table(region, pq.read_table(path))

    def supports(self, method: Any) -> bool:
        return isinstance(method, VelocityGapFiller)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        extras: dict[str, float] = {}
        per_region = []
        for region in REGIONS:
            s = self.series(ctx, region)
            test = s.days >= s.reference_end
            rmses = []
            for st in folds(s):
                hid = s.involves(st)
                got = np.asarray(
                    method.fill_velocity(VelocityQuery(region, st, blank(s, hid), hid), seed),
                    dtype=np.float64,
                )
                if got.shape != (int(hid.sum()), len(s.days)):
                    raise ValueError(f"CB-28: {method.name} gave {got.shape}")
                truth = s.dvv[hid][:, test]
                pred = np.nan_to_num(got[:, test])
                ok = np.isfinite(truth)
                rmse = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)) * 100)
                extras[f"{region}.{st}.pct"] = rmse
                rmses.append(rmse)
            extras[f"{region}.pct"] = float(np.mean(rmses))
            extras[f"{region}.folds"] = float(len(rmses))
            per_region.append(float(np.mean(rmses)))
        return Score(float(np.mean(per_region)), extras)


register_task(VelocityGapFill())
