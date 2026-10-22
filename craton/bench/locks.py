"""CratonBench, P7: the tasks that test whether a forecaster can be trusted.

    CB-05  Calibration            does "14%" happen 14% of the time? (ECE of committed cell-days)
    CB-06  Unseen region          does it know Calabria and Sicily are not what it learned?
    CB-07  Fake-precursor attack  does a glitch in the inputs make it raise alarms?
    CB-08  Physics sanity         rates, Gutenberg-Richter, Omori decay, the slip budget
    CB-11  Predictability meter   how much space-time must you watch to catch 90% of M4+?

CB-05, CB-07, CB-08 and CB-11 read the forecasts of CB-02 and CB-03 (the same test windows,
the same frozen answers), so every forecaster, every floor included, is scored on them with
no extra work. CB-06 has its own protocol, ``OodDetector``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench.data import DAY, EVENT_COLUMNS, Events, events_from_table, read_table
from craton.bench.splits import ERA_A, ERA_B, Window
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    Forecaster,
    ForecastSpec,
    Score,
    SequenceForecast,
    Task,
    _write_table,
    register_task,
)
from craton.paths import snapshots_dir
from craton.spec.timeaxis import to_seconds
from craton.store.parquet import file_sha256
from craton.trust import attacks, calibration, meter, physics
from craton.trust.ood import OodQueries, OodScores
from craton.trust.predictive import predictive_of

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

FORECAST_TASKS = ("CB-02", "CB-03")
M_CALIBRATION = 3.0


def _sequence(task_id: str) -> SequenceForecast:
    t = TASKS[task_id]
    assert isinstance(t, SequenceForecast)
    return t


def _first_bin(spec: ForecastSpec, m: float) -> int:
    return int(spec.grid.mag_bin_of(np.array([m]))[0])


class ReadsForecasts(Task):
    """A task scored on the forecasts of CB-02 and CB-03: its test files are theirs."""

    reads: tuple[str, ...] = FORECAST_TASKS

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        return {"files": self.files(ctx), "summary": {"reads": list(self.reads)}}

    def files(self, ctx: BenchContext) -> dict[str, str]:
        return {
            f"{tid}/{p.name}": file_sha256(p)
            for tid in self.reads
            for p in sorted(ctx.task_dir(tid).glob("*.parquet"))
        }

    def supports(self, method: Any) -> bool:
        return isinstance(method, Forecaster)


# ---------------------------------------------------------------- CB-05 calibration


class Calibration(ReadsForecasts):
    task_id = "CB-05"
    name = "Calibration"
    metric = "calibration error, |log₂(observed ÷ forecast)| over M3+ cell-days ≥ 0.1% likely"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "reads": list(self.reads),
            "magnitude": M_CALIBRATION,
            "event": "at least one M3+ quake in the 0.1° cell on the day",
            "bins": list(calibration.RELIABILITY_EDGES),
            "primary": "weighted mean of |log2(observed / forecast)| over the bins at or above "
            f"{calibration.ACTIVE_P}, observed smoothed (hits + 1/2) / (n + 1)",
            "also": f"ECE at or above {calibration.COMMITTED_P} and over all cell-days",
            "day_totals": "randomised PIT (KS distance from uniform), CRPS, 90% interval hits",
            "default": "a forecaster without a predictive distribution is read as Poisson",
        }

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        probs, hits, pits, crps, inside = [], [], [], [], []
        extras: dict[str, float] = {}
        for tid in self.reads:
            task = _sequence(tid)
            spec = task.spec
            pred = predictive_of(method, spec, ctx, seed)
            obs = task.observed(ctx).above(M_CALIBRATION)
            first = _first_bin(spec, M_CALIBRATION)
            day = np.floor((obs.time - spec.window.t0) / DAY).astype(np.int64)
            cell = spec.grid.cell_of(obs.lat, obs.lon)
            p = pred.occurrence(first)
            hit = np.zeros(p.shape, dtype=bool)
            hit[day, cell] = True
            n_day = np.bincount(day, minlength=p.shape[0]).astype(np.float64)
            means = pred.day_totals(first)
            k = pred.day_size(means)
            u = calibration.randomized_pit(n_day, means, k, seed)
            c = calibration.crps_counts(n_day, means, k)
            ins = calibration.central_interval_hit(n_day, means, k)
            rel = calibration.reliability(p, hit)
            extras[f"{tid}.log_error"] = rel.log_error()
            extras[f"{tid}.ece_committed"] = rel.ece(calibration.COMMITTED_P)
            extras[f"{tid}.pit_ks"] = calibration.ks_uniform(u)
            extras[f"{tid}.crps"] = float(np.mean(c))
            if pred.members is not None and pred.members.shape[0] > 1:
                lam = pred.member_expected(first)
                active = p >= calibration.ACTIVE_P
                spread = np.log(np.maximum(lam, 1e-12)).std(axis=0)
                extras[f"{tid}.member_spread"] = float(spread[active].mean())
            probs.append(p.ravel())
            hits.append(hit.ravel())
            pits.append(u)
            crps.append(c)
            inside.append(ins * len(n_day))
            extras[f"{tid}.day_k"] = min(pred.day_k, 1e8)
            extras[f"{tid}.day_gamma"] = pred.day_gamma
            extras[f"{tid}.cell_k"] = min(pred.cell_k, 1e8)
        rel = calibration.reliability(np.concatenate(probs), np.concatenate(hits))
        for i, (n, mp, fr) in enumerate(zip(rel.n, rel.mean_p, rel.freq, strict=True)):
            extras[f"bin{i}.cell_days"] = float(n)
            if n:
                extras[f"bin{i}.forecast"] = float(mp)
                extras[f"bin{i}.observed"] = float(fr)
        u_all = np.concatenate(pits)
        extras["ece_all"] = rel.ece()
        extras["ece_committed"] = rel.ece(calibration.COMMITTED_P)
        extras["pit_ks"] = calibration.ks_uniform(u_all)
        extras["pit_low_share"] = float(np.mean(u_all < 0.05))
        extras["pit_high_share"] = float(np.mean(u_all > 0.95))
        extras["crps"] = float(np.mean(np.concatenate(crps)))
        extras["interval90_hits"] = float(sum(inside) / len(u_all))
        return Score(rel.log_error(), extras)


# ---------------------------------------------------------------- CB-06 unseen region

#: CB-06's unseen region and its frozen snapshot (``configs/regions/southern_italy.yaml``).
UNSEEN_REGION = "southern_italy"
UNSEEN_SNAPSHOT = "6dd8efd2fbc23ce6"
#: A context is the quakes before a midnight; a midnight counts only with this many behind it.
CONTEXT_QUAKES = 1024
REFERENCE_EVERY = 10
IN_REGION_EVERY = 3
UNSEEN_EVERY = 15


@runtime_checkable
class OodDetector(Protocol):
    name: str

    def ood_scores(self, queries: OodQueries, ctx: BenchContext, seed: int) -> OodScores:
        """Higher = less like the training data; ``reference`` scored without itself."""
        ...


def region_events(ctx: BenchContext, region: str, snapshot_id: str) -> Events:
    path = snapshots_dir(ctx.root) / region / f"{snapshot_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no snapshot {snapshot_id} of {region}; `craton data build {region}`"
        )
    snap = json.loads(path.read_text("utf-8"))
    files = sorted(
        ctx.tokens_root / f["path"]
        for a in snap["adapters"].values()
        for f in a["files"]
        if f["instrument"] == "seismic_catalog"
    )
    return events_from_table(read_table(files, EVENT_COLUMNS))


def _midnights(start: str, end: str, every: int) -> FloatArray:
    return np.arange(to_seconds(start), to_seconds(end), every * DAY, dtype=np.float64)


def _enough(ev: Events, mids: FloatArray) -> FloatArray:
    first = np.searchsorted(ev.time, mids, "left")
    return np.asarray(mids[first >= CONTEXT_QUAKES], dtype=np.float64)


def reference_midnights(ctx: BenchContext) -> FloatArray:
    """The training contexts every detector compares with: a midnight every REFERENCE_EVERY
    days of the ``pretrain`` split, the eras and the 60 days after them left out."""
    ref = _midnights("1988-01-01T00:00:00Z", "2026-09-01T00:00:00Z", REFERENCE_EVERY)
    out = np.zeros(len(ref), dtype=bool)
    for w in (ERA_A, ERA_B):
        out |= (ref >= w.t0) & (ref < w.t1 + 60 * DAY)
    return _enough(ctx.train_events("pretrain"), ref[~out])


class UnseenRegion(Task):
    task_id = "CB-06"
    name = "Unseen region"
    metric = (
        "AUROC of the out-of-distribution score, Calabria + Sicily vs central Italy's test eras"
    )
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        return {
            "unseen_region": UNSEEN_REGION,
            "unseen_snapshot": UNSEEN_SNAPSHOT,
            "context_quakes": CONTEXT_QUAKES,
            "reference": f"pretrain split, a midnight every {REFERENCE_EVERY} days 1988 to "
            "2026-09, eras and the 60 days after them left out",
            "in_region": f"snapshot, a midnight every {IN_REGION_EVERY} days inside eras A and B",
            "unseen": f"Calabria + Sicily, a midnight every {UNSEEN_EVERY} days 1990 to 2026-09",
            "abstain": "past the 95th percentile of the reference contexts' own scores",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        south = region_events(ctx, UNSEEN_REGION, UNSEEN_SNAPSHOT)
        sets = {
            "reference": reference_midnights(ctx),
            "in_region": _enough(
                ctx.events,
                np.concatenate(
                    [_midnights(w.start, w.end, IN_REGION_EVERY) for w in (ERA_A, ERA_B)]
                ),
            ),
            "unseen": _enough(
                south, _midnights("1990-01-01T00:00:00Z", "2026-09-01T00:00:00Z", UNSEEN_EVERY)
            ),
        }
        mids = pa.table(
            {
                "set": [k for k, v in sets.items() for _ in v],
                "time": np.concatenate(list(sets.values())),
            }
        )
        unseen = pa.table(
            {
                "time": south.time,
                "lat": south.lat,
                "lon": south.lon,
                "depth_km": south.depth_km,
                "magnitude": south.mag,
                "magnitude_type": south.mag_type.tolist(),
                "token_hash": south.token_hash.tolist(),
            }
        )
        folder = ctx.task_dir(self.task_id)
        return {
            "files": {
                "midnights.parquet": _write_table(folder / "midnights.parquet", mids),
                "unseen.parquet": _write_table(folder / "unseen.parquet", unseen),
            },
            "summary": {k: len(v) for k, v in sets.items()} | {"unseen_quakes": len(south)},
        }

    def load_queries(self, ctx: BenchContext) -> OodQueries:
        folder = ctx.task_dir(self.task_id)
        m = pq.read_table(folder / "midnights.parquet")
        kind = np.asarray(m.column("set").to_pylist())
        t = np.asarray(m.column("time").to_numpy(), dtype=np.float64)
        u = pq.read_table(folder / "unseen.parquet")
        ev = Events(
            time=np.asarray(u.column("time").to_numpy(), float),
            lat=np.asarray(u.column("lat").to_numpy(), float),
            lon=np.asarray(u.column("lon").to_numpy(), float),
            depth_km=np.asarray(u.column("depth_km").to_numpy(), float),
            mag=np.asarray(u.column("magnitude").to_numpy(), float),
            mag_type=np.asarray(u.column("magnitude_type").to_pylist(), dtype=np.str_),
            token_hash=np.asarray(u.column("token_hash").to_pylist(), dtype=np.str_),
        )
        return OodQueries(t[kind == "reference"], t[kind == "in_region"], t[kind == "unseen"], ev)

    def supports(self, method: Any) -> bool:
        return isinstance(method, OodDetector)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        q = self.load_queries(ctx)
        s = method.ood_scores(q, ctx, seed)
        if (len(s.reference), len(s.in_region), len(s.unseen)) != (
            len(q.reference),
            len(q.in_region),
            len(q.unseen),
        ):
            raise ValueError("CB-06: one score per midnight, in order")
        summary = s.summary()
        return Score(summary.pop("auroc"), summary)


# ---------------------------------------------------------------- CB-07 fake precursors

ATTACK_TASK = "CB-03"
ALARM_MAGNITUDE = 4.0
#: An alarm: at least this many M4+ quakes expected in a cell and its 8 neighbours (~30 km)
#: that day. About ten times the usual level there (~10⁻⁴): "elevated", the level where
#: operational forecasts start telling people. (At 0.01 only a real sequence ever rings, and
#: no glitch was measured to reach it, so the test would have no teeth.)
ALARM_LEVEL = 0.001


def neighbourhood(x: FloatArray, n_lat: int, n_lon: int) -> FloatArray:
    """(days, cells) summed over each cell and its 8 neighbours."""
    g = x.reshape(len(x), n_lat, n_lon)
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)))
    out = sum(
        p[:, 1 + di : 1 + di + n_lat, 1 + dj : 1 + dj + n_lon]
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    )
    return np.asarray(out, dtype=np.float64).reshape(len(x), -1)


def false_alarms(fc: Any, obs: Events, spec: ForecastSpec) -> tuple[int, int]:
    """(false alarm cell-days, alarm cell-days) of a forecast."""
    g = spec.grid
    lam = neighbourhood(fc.expected(_first_bin(spec, ALARM_MAGNITUDE)), g.n_lat, g.n_lon)
    alarm = lam >= ALARM_LEVEL
    big = obs.above(ALARM_MAGNITUDE)
    hit = np.zeros(lam.shape)
    day = np.floor((big.time - spec.window.t0) / DAY).astype(np.int64)
    np.add.at(hit, (day, g.cell_of(big.lat, big.lon)), 1.0)
    real = neighbourhood(hit, g.n_lat, g.n_lon) > 0
    return int(np.sum(alarm & ~real)), int(np.sum(alarm))


class FakePrecursors(ReadsForecasts):
    task_id = "CB-07"
    name = "Fake-precursor attack"
    metric = "false alarms under attack ÷ clean false alarms, mean over the counted attacks"
    direction = "lower"
    reads = (ATTACK_TASK,)

    def definition(self) -> dict[str, Any]:
        return {
            "window": ATTACK_TASK,
            "attacks": {
                a.name: {"counted": a.counted, "what": a.what} for a in attacks.ATTACKS.values()
            },
            "season": [attacks.SEASON[0], attacks.SEASON[1]],
            "attack_days": attacks.ATTACK_DAYS,
            "site": f"seeded, at least {attacks.KEEP_AWAY_KM:g} km from L'Aquila",
            "alarm": f"expected M{ALARM_MAGNITUDE:g}+ in a cell and its 8 neighbours "
            f">= {ALARM_LEVEL}",
            "false_alarm": "an alarm cell-day with no real M4+ in the same 3x3 cells that day",
            "ratio": "(false alarms under attack + 1) / (clean false alarms + 1)",
        }

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        task = _sequence(ATTACK_TASK)
        spec, obs = task.spec, task.observed(ctx)
        clean_fa, clean_alarms = false_alarms(
            predictive_of(method, spec, ctx, seed).forecast, obs, spec
        )
        extras = {
            "clean_false_alarms": float(clean_fa),
            "clean_alarms": float(clean_alarms),
            "clean_false_alarm_rate": clean_fa / (len(spec.day_starts) * spec.grid.n_cells),
        }
        counted = []
        for attack in attacks.ATTACKS.values():
            actx = attacks.attacked(ctx, attack, seed)
            fa, alarms = false_alarms(predictive_of(method, spec, actx, seed).forecast, obs, spec)
            extras[f"{attack.name}.alarms"] = float(alarms)
            ratio = (fa + 1) / (clean_fa + 1)
            extras[f"{attack.name}.ratio"] = ratio
            extras[f"{attack.name}.false_alarms"] = float(fa)
            if attack.counted:
                counted.append(ratio)
        return Score(float(np.mean(counted)), extras)


# ---------------------------------------------------------------- CB-08 physics


@dataclass(frozen=True)
class Probe:
    """The Omori probe: the catalogue up to ``cut``, nothing after, then ``days`` forecasts."""

    task_id: str
    cut: str
    days: int = 30


PROBES = (Probe("CB-02", "2016-08-25T00:00:00Z"), Probe("CB-03", "2009-04-07T00:00:00Z"))


class PhysicsSanity(ReadsForecasts):
    task_id = "CB-08"
    name = "Physics sanity"
    metric = "hard physics violations (rates, Gutenberg-Richter, Omori probe, slip budget)"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "reads": list(self.reads),
            "rates": "finite and non-negative, magnitude shares too",
            "gutenberg_richter": "shares never grow with magnitude; b at M3 in "
            f"{list(physics.B_RANGE)}",
            "omori_probe": [p.__dict__ for p in PROBES],
            "omori_tolerance": physics.OMORI_TOLERANCE,
            "moment_budget_nm_per_yr": physics.MOMENT_BUDGET,
            "calm_day": f"no M{physics.CALM_MAGNITUDE:g}+ in the {physics.CALM_DAYS} days before",
        }

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        total = 0
        extras: dict[str, float] = {"moment_budget": physics.MOMENT_BUDGET}
        for tid in self.reads:
            task = _sequence(tid)
            spec = task.spec
            fc = predictive_of(method, spec, ctx, seed).forecast
            history = ctx.events.window(spec.window.t0 - physics.CALM_DAYS * DAY, spec.window.t1)
            c = physics.check_forecast(fc, spec.grid, history, spec.day_starts)
            total += c.violations
            extras[f"{tid}.bad_rates"] = float(c.bad_rates + c.bad_shares)
            extras[f"{tid}.gr_violations"] = float(c.gr_rising + c.b_out_of_range)
            extras[f"{tid}.moment_rate"] = c.moment_rate
            extras[f"{tid}.over_budget"] = float(c.over_budget)
        for probe in PROBES:
            task = _sequence(probe.task_id)
            cut = to_seconds(probe.cut)
            window = Window("omori_probe", probe.cut, _iso(cut + probe.days * DAY))
            spec = replace(task.spec, window=window)
            pctx = replace(
                ctx,
                _cache={k: v for k, v in ctx._cache.items() if k.startswith("etas-fit:")}
                | {"events": ctx.events.window(-math.inf, cut)},
            )
            fc = method.forecast(spec, pctx, seed)
            bad, series = physics.omori_violations(fc, _first_bin(spec, 3.0))
            total += bad
            extras[f"{probe.task_id}.omori_violations"] = float(bad)
            extras[f"{probe.task_id}.omori_day1"] = float(series[0])
            extras[f"{probe.task_id}.omori_day30"] = float(series[-1])
        return Score(float(total), extras)


def _iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- CB-11 the meter


class PredictabilityMeter(ReadsForecasts):
    task_id = "CB-11"
    name = "Predictability meter"
    metric = "share of space-time (%) watched to catch 90% of the M4+ quakes"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "reads": list(self.reads),
            "magnitude": meter.M_METER,
            "coverage": meter.COVERAGE,
            "primary": "rank all (day, cell) of both windows by M4+ rate; share watched",
            "space": "coverage-matched daily area as an equal-area radius, km",
            "time": "share of days on alert, ranked by the day's M4+ rate",
            "magnitude_width": "central 90% interval of magnitude given M4+",
        }

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        parts = []
        for tid in self.reads:
            task = _sequence(tid)
            spec = task.spec
            fc = predictive_of(method, spec, ctx, seed).forecast
            parts.append(meter.part(fc, task.observed(ctx), spec))
        share = meter.space_time_share(parts)
        t_share, alert = meter.time_share(parts)
        radius, nominal = meter.space_radius(parts)
        width = meter.magnitude_width(parts, _sequence("CB-02").spec.grid)
        return Score(
            100.0 * share,
            {
                "quakes": float(sum(len(p.day) for p in parts)),
                "time_share": t_share,
                "alert_days_per_year": alert,
                "space_radius_km": radius,
                "space_nominal_coverage": nominal,
                "magnitude_width": width,
            },
        )


register_task(Calibration())
register_task(UnseenRegion())
register_task(FakePrecursors())
register_task(PhysicsSanity())
register_task(PredictabilityMeter())
