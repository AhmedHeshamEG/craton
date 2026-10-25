"""CB-22, world sequences: the same daily forecast anywhere on Earth, against ETAS fitted there.

Until P8d every forecast Craton made was of central Italy. CB-22 asks the same question of
twelve sequences in five other regions, each one the year after its first big shock:

- **seen** regions (California, Japan, New Zealand): Craton trained on their past up to
  2007-01-01, never on the sequences, which all start later;
- **zero-shot** regions (Turkey, Chile, Alaska): Craton never saw a single quake from them.
  This half is the headline: can a model of the Earth's crust forecast a place it has never
  read?

Each sequence is scored like CB-02, on a 3° × 3.5° box (CB-02's size, 1,050 cells of 0.1°)
centred on its first big shock, in bits per quake from the region's completeness + 1
(``Region.completeness``, central Italy's M2 + 1 = M3). The comparison is ETAS fitted on that
box's own past at the region's completeness, up to the midnight of the first big shock: the
best simple model a seismologist there would build. The primary number is the gain over that
ETAS in bits per quake, averaged over the zero-shot regions (each region the mean of its
sequences), so ETAS itself scores 0 and anything below 0 loses to it.

A region's quakes come from its own snapshot (``craton data build <region>``); the observed
quakes of every sequence are frozen with the task, and the snapshots it was frozen from are
checked at scoring time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from craton.adapters.region import Region, load_region
from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.splits import Window
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    ForecastSpec,
    GriddedForecast,
    Score,
    Task,
    _write_table,
    register_task,
    score_forecast,
)
from craton.paths import data_dir
from craton.spec.timeaxis import to_iso, to_seconds

#: A sequence's box: CB-02's 3° × 3.5° in 0.1° cells.
BOX_LAT, BOX_LON, CELL = 3.0, 3.5, 0.1
#: Days scored after the first big shock's day.
TEST_DAYS = 365
#: ETAS fits on at most this many years before the first big shock.
FIT_YEARS = 20
#: The largest magnitude any world forecast may give probability to (the largest ever, 9.5).
MAX_MAGNITUDE = 9.5
#: Cells of the forecast grid start at the region's completeness; the open top bin from here.
GRID_M_MAX = 9.5
SEEN, ZERO_SHOT = "seen", "zero_shot"
HALVES = (SEEN, ZERO_SHOT)


@dataclass(frozen=True)
class Sequence:
    """A held-out earthquake sequence: the day of its first big shock (UTC) and where."""

    key: str
    name: str
    region: str
    day: str  # YYYY-MM-DD, UTC: the first M6+ of the sequence
    lat: float
    lon: float
    largest: float  # the sequence's largest magnitude (for the record, never used)
    half: str


SEQUENCES = (
    Sequence(
        "el_mayor", "El Mayor-Cucapah 2010", "california", "2010-04-04", 32.29, -115.30, 7.2, SEEN
    ),
    Sequence(
        "ridgecrest", "Ridgecrest 2019", "california", "2019-07-04", 35.71, -117.50, 7.1, SEEN
    ),
    Sequence("tohoku", "Tohoku 2011", "japan", "2011-03-09", 38.30, 142.37, 9.0, SEEN),
    Sequence("kumamoto", "Kumamoto 2016", "japan", "2016-04-14", 32.75, 130.76, 7.0, SEEN),
    Sequence(
        "darfield",
        "Darfield-Christchurch 2010",
        "new_zealand",
        "2010-09-03",
        -43.53,
        172.17,
        7.1,
        SEEN,
    ),
    Sequence("kaikoura", "Kaikoura 2016", "new_zealand", "2016-11-13", -42.74, 173.05, 7.8, SEEN),
    Sequence("van", "Van 2011", "turkey", "2011-10-23", 38.72, 43.51, 7.1, ZERO_SHOT),
    Sequence("elazig", "Elazig 2020", "turkey", "2020-01-24", 38.36, 39.06, 6.7, ZERO_SHOT),
    Sequence(
        "kahramanmaras", "Kahramanmaras 2023", "turkey", "2023-02-06", 37.17, 37.03, 7.8, ZERO_SHOT
    ),
    Sequence("maule", "Maule 2010", "chile", "2010-02-27", -36.12, -72.90, 8.8, ZERO_SHOT),
    Sequence("iquique", "Iquique 2014", "chile", "2014-03-16", -19.61, -70.77, 8.2, ZERO_SHOT),
    Sequence("illapel", "Illapel 2015", "chile", "2015-09-16", -31.57, -71.67, 8.3, ZERO_SHOT),
    Sequence("anchorage", "Anchorage 2018", "alaska", "2018-11-30", 61.35, -149.96, 7.1, ZERO_SHOT),
    Sequence("chignik", "Chignik 2021", "alaska", "2021-07-29", 55.36, -157.89, 8.2, ZERO_SHOT),
)


def _iso_day(day: str, plus_days: int = 0) -> str:
    d = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(days=plus_days)
    return to_iso(d.timestamp()) or ""


def _snap(v: float) -> float:
    return round(round(v / CELL) * CELL, 6)


def sequence_box(seq: Sequence, region: Region) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max): the box centred on the first big shock, moved
    inside the region's box if it would stick out."""
    b = region.box
    lat0 = min(max(_snap(seq.lat - BOX_LAT / 2), b.lat_min), b.lat_max - BOX_LAT)
    lon0 = min(max(_snap(seq.lon - BOX_LON / 2), b.lon_min), b.lon_max - BOX_LON)
    lat0, lon0 = _snap(lat0), _snap(lon0)
    return lat0, round(lat0 + BOX_LAT, 6), lon0, round(lon0 + BOX_LON, 6)


def region_mc(region: Region) -> float:
    if region.completeness is None:
        raise ValueError(f"region {region.name} states no completeness; CB-22 needs one")
    return region.completeness


def sequence_spec(seq: Sequence, region: Region) -> ForecastSpec:
    """CB-02's question about one world sequence, in the region's own terms."""
    mc = region_mc(region)
    lat0, lat1, lon0, lon1 = sequence_box(seq, region)
    grid = csep.Grid(lat0, lat1, lon0, lon1, cell_deg=CELL, m_min=mc, m_max=GRID_M_MAX)
    cut = _iso_day(seq.day)
    window = Window(f"cb22_{seq.key}", _iso_day(seq.day, 1), _iso_day(seq.day, 1 + TEST_DAYS))
    fit_from = max(to_seconds(region.start), to_seconds(cut) - FIT_YEARS * 365.25 * DAY)
    return ForecastSpec(
        task_id=f"CB-22-{seq.key}",
        fit_split="world",
        window=window,
        grid=grid,
        mc=mc,
        fit_from=_iso_day(datetime.fromtimestamp(fit_from, UTC).strftime("%Y-%m-%d")),
        cut=cut,
        thresholds=(mc + 1.0, mc),
        max_magnitude=MAX_MAGNITUDE,
    )


# ---------------------------------------------------------------- a region seen through a box


@dataclass
class RegionView(BenchContext):
    """Any region's snapshot seen through one box, with a cut-off: the calls a forecaster
    makes on central Italy (``events``, ``train_events``, ``split_dir``) answered from it.

    ``events`` holds every quake of the snapshot inside the box (forecast history and the
    answers); ``train_events`` only those before ``cut``, the only ones a method may fit on.
    """

    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    cut: str = ""
    _ready: set[str] = field(default_factory=set)

    @property
    def view_id(self) -> str:
        lat0, lat1, lon0, lon1 = self.box
        return f"{self.snapshot_id}-{lat0:g}_{lat1:g}_{lon0:g}_{lon1:g}-{self.cut[:10]}"

    @property
    def bench_dir(self) -> Path:
        return data_dir(self.root) / "bench" / "world" / self.region / self.snapshot_id

    @property
    def events(self) -> Events:
        if "view" not in self._cache:
            ev = BenchContext.events.fget(self)  # type: ignore[attr-defined]
            lat0, lat1, lon0, lon1 = self.box
            self._cache["view"] = ev.select(
                (ev.lat >= lat0) & (ev.lat < lat1) & (ev.lon >= lon0) & (ev.lon < lon1)
            )
        out: Events = self._cache["view"]
        return out

    def train_events(self, split: str) -> Events:
        return self.events.select(self.events.time < to_seconds(self.cut))

    def split_dir(self, name: str) -> Path:
        folder = self.bench_dir / "splits" / f"{name}-{self.view_id}"
        if name not in self._ready:
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / "manifest.json"
            body = f'{{"id": "{self.view_id}-{name}"}}\n'
            if not manifest.is_file() or manifest.read_text("utf-8") != body:
                manifest.write_text(body, "utf-8")
            self._ready.add(name)
        return folder

    def has_split(self, split: str) -> bool:
        return True


def view_for(root: Path, region: str, spec: ForecastSpec, snapshot_id: str = "") -> RegionView:
    g = spec.grid
    return RegionView(
        root=root,
        region=region,
        snapshot_id=snapshot_id,
        box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
        cut=spec.fit_end,
    )


def observed(view: RegionView, spec: ForecastSpec) -> Events:
    g = spec.grid
    ev = view.events.window(spec.window.t0, spec.window.t1)
    return ev.select((g.cell_of(ev.lat, ev.lon) >= 0) & (g.mag_bin_of(ev.mag) >= 0))


# ---------------------------------------------------------------- the task


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


class WorldSequences(Task):
    task_id = "CB-22"
    name = "World sequences"
    metric = (
        "bits per quake above ETAS fitted on the region's own past, mean over never-seen regions"
    )
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        return {
            "sequences": [s.__dict__ for s in SEQUENCES],
            "box": f"{BOX_LAT:g} x {BOX_LON:g} degrees of {CELL:g} degree cells, centred on the "
            "first big shock, moved inside the region's box",
            "window": f"the {TEST_DAYS} days after the first big shock's UTC day",
            "magnitudes": "from the region's completeness; scored from completeness + 1 "
            "(primary) and completeness",
            "reference": f"ETAS fitted on the box's quakes at the region's completeness, at most "
            f"{FIT_YEARS} years before the first big shock's midnight, run day by day",
            "primary": "mean over zero-shot regions of the mean gain over the reference (bits "
            "per quake from completeness + 1) of the region's sequences",
            "seen_regions_trained_until": "2007-01-01 (no sequence starts before 2010)",
            "scoring": "as CB-02 (score_forecast), max magnitude 9.5",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}}
        for seq in SEQUENCES:
            region = load_region(seq.region, ctx.root)
            spec = sequence_spec(seq, region)
            view = view_for(ctx.root, seq.region, spec)
            summary["snapshots"][seq.region] = view.snapshot_id
            from craton.bench.planets import events_table

            obs = observed(view, spec)
            files[f"{seq.key}.parquet"] = _write_table(
                folder / f"{seq.key}.parquet", events_table(obs)
            )
            hist = view.train_events("world").above(spec.mc)
            summary[seq.key] = {
                "box": list(view.box),
                "window": spec.window.as_dict(),
                "mc": spec.mc,
                f"M>={spec.scored:g}": int((obs.mag >= spec.scored - 1e-9).sum()),
                f"M>={spec.mc:g}": int((obs.mag >= spec.mc - 1e-9).sum()),
                "largest": float(obs.mag.max()) if len(obs) else None,
                "fit_quakes": int(
                    ((hist.time >= to_seconds(spec.fit_from)) & (hist.time < view_t(spec))).sum()
                ),
            }
        return {"files": files, "summary": summary}

    def supports(self, method: Any) -> bool:
        return isinstance(method, Forecaster) and not getattr(method, "regional_only", False)

    def frozen_snapshots(self, ctx: BenchContext) -> dict[str, str]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        out: dict[str, str] = task.get("summary", {}).get("snapshots", {})
        return out

    def observed_of(self, ctx: BenchContext, seq: Sequence) -> Events:
        from craton.bench.planets import read_events

        return read_events(ctx.task_dir(self.task_id) / f"{seq.key}.parquet")

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        from craton.bench.baselines import Etas

        snaps = self.frozen_snapshots(ctx)
        world = getattr(method, "world_forecaster", None)
        forecaster = world() if callable(world) else method
        extras: dict[str, float] = {}
        gains: dict[str, dict[str, list[float]]] = {h: {} for h in HALVES}
        for seq in SEQUENCES:
            region = load_region(seq.region, ctx.root)
            spec = sequence_spec(seq, region)
            key = f"cb22-view:{seq.key}"
            if key not in ctx._cache:
                ctx._cache[key] = view_for(ctx.root, seq.region, spec, snaps.get(seq.region, ""))
            view: RegionView = ctx._cache[key]
            obs = self.observed_of(ctx, seq)
            fc = forecaster.forecast(spec, view, seed)
            _check(fc, spec)
            sc = score_forecast(fc, obs, spec, seed)
            ref_key = f"cb22-etas:{seq.key}:{seed}"
            if ref_key not in ctx._cache:
                ctx._cache[ref_key] = score_forecast(
                    Etas().forecast(spec, view, seed), obs, spec, seed
                ).value
            gain = sc.value - float(ctx._cache[ref_key])
            tag = f"M{spec.scored:g}"
            extras[f"{seq.key}.bits"] = sc.value
            extras[f"{seq.key}.gain"] = gain
            extras[f"{seq.key}.observed"] = sc.extras[f"{tag}.observed"]
            extras[f"{seq.key}.expected"] = sc.extras[f"{tag}.expected"]
            for part in ("how_many_bits", "where_bits", "how_big_bits"):
                extras[f"{seq.key}.{part}"] = sc.extras.get(f"{tag}.{part}", math.nan)
            gains[seq.half].setdefault(seq.region, []).append(gain)
        for half in HALVES:
            per_seq = [g for v in gains[half].values() for g in v]
            per_region = [_mean(v) for v in gains[half].values()]
            extras[f"{half}.mean_gain"] = _mean(per_seq)
            extras[f"{half}.sequences_beaten"] = _mean([float(g > 0) for g in per_seq])
            extras[f"{half}.regions_beaten"] = _mean([float(g > 0) for g in per_region])
            for name, v in gains[half].items():
                extras[f"{half}.{name}.mean_gain"] = _mean(v)
        primary = _mean([_mean(v) for v in gains[ZERO_SHOT].values()])
        return Score(primary, extras)


def view_t(spec: ForecastSpec) -> float:
    return to_seconds(spec.fit_end)


def _check(fc: GriddedForecast, spec: ForecastSpec) -> None:
    days = len(spec.day_starts)
    if fc.rates.shape != (days, spec.grid.n_cells):
        raise ValueError(f"{spec.task_id}: forecast has shape {fc.rates.shape}")
    if np.any(fc.rates < 0) or not np.all(np.isfinite(fc.rates)):
        raise ValueError(f"{spec.task_id}: forecast rates must be finite and >= 0")


register_task(WorldSequences())
