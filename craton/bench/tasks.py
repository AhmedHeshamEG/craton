"""CratonBench: CB-01 … CB-04, CB-15, CB-17, their frozen test data, and how they are scored.

Each task has an ID, a metric, a direction (lower or higher is better) and a **fingerprint**:
the hash of its definition, its metric version and every frozen test file. If any of those
changes, it's a different task and needs a new ID; the ratchet refuses a changed fingerprint.

Methods (baselines now, Craton models later) plug in by implementing one protocol per family:

    MaskedPredictor     CB-01   a distribution for every hidden channel of a masked token
    Forecaster          CB-02/3 expected quakes per (day, cell, magnitude bin)
    GapFiller           CB-04   daily positions of a GNSS station nobody let you see
    EfficiencyReport    CB-15   what the model's training run cost on the GPU
    InContextPredictor  CB-17   a region never seen: its next year, from its first N days
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench import csep
from craton.bench.data import (
    DAY,
    Events,
    Gnss,
    Tides,
    instrument_files,
    load_events,
    load_gnss,
    load_tides,
    read_table,
)
from craton.bench.dists import ChannelDist
from craton.bench.splits import (
    ERA_A,
    ERA_B,
    ERAS,
    SPLITS,
    Window,
    choose_heldout,
    stable_stations,
    station_sites,
)
from craton.paths import data_dir, snapshots_dir, tokens_dir
from craton.spec.timeaxis import to_seconds
from craton.store.parquet import file_sha256

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

REGION = "central_italy"
FROZEN_DIR = Path(__file__).parent / "frozen"


def frozen_path(region: str = REGION) -> Path:
    return FROZEN_DIR / f"{region}.json"


def load_frozen(region: str = REGION) -> dict[str, Any] | None:
    path = frozen_path(region)
    if not path.is_file():
        return None
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _write_table(path: Path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", compression_level=9, version="2.6")
    return file_sha256(path)


def _salted_share(token_hash: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}:{token_hash}".encode()).hexdigest()[:8], 16) / 16**8


# ---------------------------------------------------------------- the context


@dataclass
class BenchContext:
    """One region's snapshot, its splits and its prepared tasks, under ``data/bench/``."""

    root: Path
    region: str = REGION
    snapshot_id: str = ""
    _cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            frozen = load_frozen(self.region)
            if frozen is not None:
                self.snapshot_id = frozen["snapshot"]
            else:
                latest = snapshots_dir(self.root) / self.region / "LATEST"
                if not latest.is_file():
                    raise FileNotFoundError(
                        f"no snapshot of {self.region}; run `craton data build {self.region}`"
                    )
                self.snapshot_id = latest.read_text(encoding="utf-8").strip()

    @property
    def snapshot_path(self) -> Path:
        return snapshots_dir(self.root) / self.region / f"{self.snapshot_id}.json"

    @property
    def has_data(self) -> bool:
        return self.snapshot_path.is_file()

    @cached_property
    def snapshot(self) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        return body

    @property
    def tokens_root(self) -> Path:
        return tokens_dir(self.root)

    @property
    def bench_dir(self) -> Path:
        return data_dir(self.root) / "bench" / self.region / self.snapshot_id

    def split_dir(self, name: str) -> Path:
        return self.bench_dir / "splits" / name

    def task_dir(self, task_id: str) -> Path:
        return self.bench_dir / "tasks" / task_id

    def _snapshot_files(self, instrument: str) -> list[Path]:
        out = []
        for adapter in self.snapshot["adapters"].values():
            out += [
                self.tokens_root / f["path"]
                for f in adapter["files"]
                if f["instrument"] == instrument
            ]
        return sorted(out)

    @property
    def events(self) -> Events:
        """Every quake in the snapshot: forecast history and test answers, never training."""
        if "events" not in self._cache:
            from craton.bench.data import EVENT_COLUMNS, events_from_table

            files = self._snapshot_files("seismic_catalog")
            self._cache["events"] = events_from_table(read_table(files, EVENT_COLUMNS))
        out: Events = self._cache["events"]
        return out

    @property
    def gnss(self) -> Gnss:
        if "gnss" not in self._cache:
            from craton.bench.data import GNSS_COLUMNS, gnss_from_table

            files = self._snapshot_files("gnss_daily")
            self._cache["gnss"] = gnss_from_table(read_table(files, GNSS_COLUMNS))
        out: Gnss = self._cache["gnss"]
        return out

    @property
    def tides(self) -> Tides:
        """The hourly tide series of the snapshot: computed forcing, readable for any time."""
        if "tides" not in self._cache:
            from craton.bench.data import TIDE_COLUMNS, tides_from_table

            files = self._snapshot_files("solid_earth_tide")
            self._cache["tides"] = tides_from_table(read_table(files, TIDE_COLUMNS))
        out: Tides = self._cache["tides"]
        return out

    def tide_links(self) -> tuple[list[str], list[str], FloatArray]:
        """(token hash, content hash, time) of every tide token: content hash = its event."""
        files = self._snapshot_files("solid_earth_tide")
        t = read_table(files, ["token_hash", "content_hash", "time"])
        return (
            t.column("token_hash").to_pylist(),
            t.column("content_hash").to_pylist(),
            np.asarray(t.column("time").to_numpy(), dtype=np.float64),
        )

    @cached_property
    def heldout(self) -> list[str]:
        frozen = load_frozen(self.region)
        if frozen is not None and frozen["snapshot"] == self.snapshot_id:
            return list(frozen["heldout_stations"])
        return choose_heldout(station_sites(self.gnss), stable_stations(self.gnss))

    def train_events(self, split: str) -> Events:
        """Quakes a method may fit on: read from the split's shards, never the snapshot."""
        return load_events(self.split_dir(split))

    def train_gnss(self, split: str) -> Gnss:
        return load_gnss(self.split_dir(split))

    def train_tides(self, split: str) -> Tides:
        return load_tides(self.split_dir(split))

    def has_split(self, split: str) -> bool:
        return (self.split_dir(split) / "manifest.json").is_file() and bool(
            instrument_files(self.split_dir(split), "seismic_catalog")
        )


# ---------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Score:
    value: float
    extras: dict[str, float]


class Task(ABC):
    task_id: str
    name: str
    metric: str
    direction: str
    #: Bumped only by a new task ID; part of the fingerprint.
    version: str = "1"

    @abstractmethod
    def definition(self) -> dict[str, Any]:
        """Everything that defines the test, as data."""

    @abstractmethod
    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        """Write the frozen test files; return {"files": {name: sha256}, "summary": {...}}."""

    @abstractmethod
    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        """Score one method with one seed."""

    @abstractmethod
    def supports(self, method: Any) -> bool:
        """Whether the method implements this task's protocol."""

    def fingerprint(self, files: dict[str, str]) -> str:
        body = {
            "task": self.task_id,
            "version": self.version,
            "metric": self.metric,
            "direction": self.direction,
            "definition": self.definition(),
            "files": files,
        }
        return hashlib.sha256(_canon(body).encode()).hexdigest()

    def files(self, ctx: BenchContext) -> dict[str, str]:
        folder = ctx.task_dir(self.task_id)
        return {p.name: file_sha256(p) for p in sorted(folder.glob("*.parquet"))}


# ---------------------------------------------------------------- CB-01 geo-perplexity


@dataclass(frozen=True)
class Bin:
    """A channel's scoring bins: edges at k·width − offset."""

    channel: str
    width: float
    offset: float = 0.0

    def edges_of(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        k = np.floor((x + self.offset) / self.width + 1e-9)
        lo = k * self.width - self.offset
        return lo, lo + self.width


#: The bins that turn continuous answers into "bits". Part of CB-01's definition.
CB01_BINS: dict[str, tuple[Bin, ...]] = {
    "seismic_catalog": (
        Bin("lat", 0.01),
        Bin("lon", 0.01),
        Bin("depth_km", 1.0),
        Bin("magnitude", 0.1, 0.05),
    ),
    "gnss_daily": (Bin("east", 0.001), Bin("north", 0.001), Bin("up", 0.001)),
}
CB01_MASK_SHARE = 0.15
CB01_MIN_STATION_DAYS = 30
#: A channel costs at most this many bits: a probability of 2^-64 or less counts as 2^-64.
MAX_BITS = 64.0


@dataclass(frozen=True)
class MaskedQueries:
    """The masked tokens of CB-01, minus everything that would give the answer away."""

    token_hash: list[str]
    instrument: list[str]
    time: FloatArray
    qualifiers: list[dict[str, str]]
    site_lat: FloatArray  # station site for fixed instruments (rounded to 0.01°), NaN for events
    site_lon: FloatArray

    def where(self, instrument: str) -> np.ndarray[Any, np.dtype[np.int64]]:
        return np.nonzero(np.asarray(self.instrument) == instrument)[0]


@runtime_checkable
class MaskedPredictor(Protocol):
    name: str

    def predict_masked(
        self, queries: MaskedQueries, ctx: BenchContext, seed: int
    ) -> dict[str, dict[str, ChannelDist]]:
        """{instrument: {channel: distribution over that instrument's queries, in order}}."""
        ...


class GeoPerplexity(Task):
    task_id = "CB-01"
    name = "Geo-perplexity"
    metric = "bits per masked token, mean over instruments"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "eras": [w.as_dict() for w in ERAS],
            "instruments": sorted(CB01_BINS),
            "bins": {i: [b.__dict__ for b in bs] for i, bs in CB01_BINS.items()},
            "mask_share": CB01_MASK_SHARE,
            "min_station_days": CB01_MIN_STATION_DAYS,
            "max_bits_per_channel": MAX_BITS,
            "revealed": "kind, instrument, qualifiers, time; station site to 0.01° for GNSS",
            "hidden_also": "the tide token computed at each masked quake",
            "train_split": "pretrain",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        heldout = set(ctx.heldout)
        ev = ctx.events
        in_era = np.zeros(len(ev), dtype=bool)
        for w in ERAS:
            in_era |= (ev.time >= w.t0) & (ev.time < w.t1)
        ev = ev.select(in_era)

        g = ctx.gnss
        g_era = np.zeros(len(g), dtype=bool)
        for w in ERAS:
            g_era |= (g.time >= w.t0) & (g.time < w.t1)
        train_days: dict[str, int] = {}
        for code in g.stations:
            train_days[code] = int(((g.station == code) & ~g_era).sum())
        sites = station_sites(g)
        known = np.array(
            [s not in heldout and train_days[s] >= CB01_MIN_STATION_DAYS for s in g.station]
        )
        g = g.select(g_era & known)

        ev_mask = np.array(
            [_salted_share(h, self.task_id) < CB01_MASK_SHARE for h in ev.token_hash]
        )
        g_mask = np.array([_salted_share(h, self.task_id) < CB01_MASK_SHARE for h in g.token_hash])
        ev, g = ev.select(ev_mask), g.select(g_mask)

        hashes = [*ev.token_hash.tolist(), *g.token_hash.tolist()]
        instruments = ["seismic_catalog"] * len(ev) + ["gnss_daily"] * len(g)
        times = np.concatenate([ev.time, g.time])
        ev_quals = [{"magnitude_type": t} for t in ev.mag_type.tolist()]
        g_quals = [{"station": s, "reference_frame": "IGS20"} for s in g.station.tolist()]
        site_lat = np.concatenate(
            [np.full(len(ev), np.nan), [round(sites[s][0], 2) for s in g.station]]
        )
        site_lon = np.concatenate(
            [np.full(len(ev), np.nan), [round(sites[s][1], 2) for s in g.station]]
        )
        order = np.lexsort((np.array(hashes), times))
        queries = pa.table(
            {
                "token_hash": [hashes[i] for i in order],
                "instrument": [instruments[i] for i in order],
                "time": times[order],
                "qualifiers": [_canon((ev_quals + g_quals)[i]) for i in order],
                "site_lat": site_lat[order],
                "site_lon": site_lon[order],
            }
        )
        ev_vals = np.stack([ev.lat, ev.lon, ev.depth_km, ev.mag], axis=1).tolist()
        values = ev_vals + g.enu.tolist()
        answers = pa.table(
            {
                "token_hash": [hashes[i] for i in order],
                "value": [values[i] for i in order],
            }
        )
        masked_events = set(ev.token_hash.tolist())
        tide_hash, tide_link, _ = ctx.tide_links()
        linked = sorted(h for h, c in zip(tide_hash, tide_link, strict=True) if c in masked_events)
        hidden = pa.table({"token_hash": sorted({*hashes, *linked})})

        folder = ctx.task_dir(self.task_id)
        files = {
            "answers.parquet": _write_table(folder / "answers.parquet", answers),
            "hidden.parquet": _write_table(folder / "hidden.parquet", hidden),
            "queries.parquet": _write_table(folder / "queries.parquet", queries),
        }
        return {
            "files": files,
            "summary": {
                "queries": len(order),
                "seismic_catalog": len(ev),
                "gnss_daily": len(g),
                "hidden_tides": len(linked),
            },
        }

    def load_queries(self, ctx: BenchContext) -> MaskedQueries:
        t = pq.read_table(ctx.task_dir(self.task_id) / "queries.parquet")
        return MaskedQueries(
            token_hash=t.column("token_hash").to_pylist(),
            instrument=t.column("instrument").to_pylist(),
            time=np.asarray(t.column("time").to_numpy(), dtype=np.float64),
            qualifiers=[json.loads(q) for q in t.column("qualifiers").to_pylist()],
            site_lat=np.asarray(t.column("site_lat").to_numpy(zero_copy_only=False), float),
            site_lon=np.asarray(t.column("site_lon").to_numpy(zero_copy_only=False), float),
        )

    def hidden(self, ctx: BenchContext) -> set[str]:
        t = pq.read_table(ctx.task_dir(self.task_id) / "hidden.parquet")
        return set(t.column("token_hash").to_pylist())

    def answers(self, ctx: BenchContext) -> list[list[float]]:
        t = pq.read_table(ctx.task_dir(self.task_id) / "answers.parquet")
        out: list[list[float]] = t.column("value").to_pylist()
        return out

    def supports(self, method: Any) -> bool:
        return isinstance(method, MaskedPredictor)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        queries = self.load_queries(ctx)
        answers = self.answers(ctx)
        preds = method.predict_masked(queries, ctx, seed)
        per_inst: dict[str, float] = {}
        extras: dict[str, float] = {}
        for inst, bins in CB01_BINS.items():
            idx = queries.where(inst)
            if not len(idx):
                continue
            truth = np.array([answers[i] for i in idx], dtype=np.float64)
            bits = np.zeros(len(idx))
            for c, b in enumerate(bins):
                lo, hi = b.edges_of(truth[:, c])
                mass = preds[inst][b.channel].mass(lo, hi)
                ch_bits = np.minimum(-np.log2(np.maximum(mass, 2.0**-MAX_BITS)), MAX_BITS)
                extras[f"{inst}.{b.channel}_bits"] = float(np.mean(ch_bits))
                bits += ch_bits
            per_inst[inst] = float(np.mean(bits))
            extras[f"{inst}_bits_per_token"] = per_inst[inst]
            extras[f"{inst}_queries"] = float(len(idx))
        return Score(float(np.mean(list(per_inst.values()))), extras)


# ---------------------------------------------------------------- CB-02 / CB-03 forecasts


@dataclass(frozen=True)
class ForecastSpec:
    """What a forecaster is asked: daily forecasts over a test window, fitted on a split."""

    task_id: str
    fit_split: str
    window: Window
    grid: csep.Grid

    @property
    def day_starts(self) -> FloatArray:
        n = round((self.window.t1 - self.window.t0) / DAY)
        return np.asarray(self.window.t0 + DAY * np.arange(n), dtype=np.float64)


@dataclass(frozen=True)
class GriddedForecast:
    """Expected quakes per (day, cell) at or above the grid's lowest bin, and how they split
    over magnitude bins: ``mag_shares`` broadcasts to (days, cells, mags) and sums to 1 over
    the magnitude axis."""

    rates: FloatArray  # (days, cells)
    mag_shares: FloatArray

    def expected(self, first_bin: int = 0) -> FloatArray:
        """(days, cells) expected count in bins ``first_bin`` and up."""
        shares = np.broadcast_to(self.mag_shares, (*self.rates.shape, self.mag_shares.shape[-1]))
        return np.asarray(self.rates * shares[..., first_bin:].sum(axis=-1), dtype=np.float64)

    def at(self, day: Any, cell: Any, mag_bin: Any) -> FloatArray:
        shares = np.broadcast_to(self.mag_shares, (*self.rates.shape, self.mag_shares.shape[-1]))
        return np.asarray(self.rates[day, cell] * shares[day, cell, mag_bin], dtype=np.float64)


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast: ...


#: Grid for the forecasts: 0.1° cells over the region, magnitude bins from 1.5 up.
def forecast_grid() -> csep.Grid:
    return csep.Grid(41.0, 44.0, 11.5, 15.0, cell_deg=0.1, m_min=1.5, m_max=8.0)


THRESHOLDS = (3.0, 1.5)
N_SIMS = 1000


class SequenceForecast(Task):
    direction = "higher"
    metric = "log-likelihood per earthquake (bits), M≥3, daily 0.1° forecasts"

    def __init__(self, task_id: str, name: str, window: Window, fit_split: str) -> None:
        self.task_id = task_id
        self.name = name
        self.window = window
        self.fit_split = fit_split

    @property
    def spec(self) -> ForecastSpec:
        return ForecastSpec(self.task_id, self.fit_split, self.window, forecast_grid())

    def definition(self) -> dict[str, Any]:
        g = forecast_grid()
        return {
            "window": self.window.as_dict(),
            "fit_split": self.fit_split,
            "grid": g.__dict__,
            "thresholds": list(THRESHOLDS),
            "primary_threshold": THRESHOLDS[0],
            "magnitudes": "catalogue preferred magnitude of any type (ML, Md, Mw)",
            "day": "UTC midnight to midnight; a forecast may read every quake before its day",
            "csep": f"Poisson; daily N-test, S and M tests on the summed forecast, {N_SIMS} sims",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        g = forecast_grid()
        obs = ctx.events.window(self.window.t0, self.window.t1)
        obs = obs.select((g.cell_of(obs.lat, obs.lon) >= 0) & (g.mag_bin_of(obs.mag) >= 0))
        table = pa.table(
            {
                "time": obs.time,
                "lat": obs.lat,
                "lon": obs.lon,
                "depth_km": obs.depth_km,
                "magnitude": obs.mag,
                "magnitude_type": obs.mag_type.tolist(),
                "token_hash": obs.token_hash.tolist(),
            }
        )
        sha = _write_table(ctx.task_dir(self.task_id) / "observed.parquet", table)
        types = dict(zip(*np.unique(obs.mag_type, return_counts=True), strict=True))
        return {
            "files": {"observed.parquet": sha},
            "summary": {
                "days": len(self.spec.day_starts),
                **{f"M>={t:g}": int((obs.mag >= t - 1e-9).sum()) for t in THRESHOLDS},
                "magnitude_types": {str(k): int(v) for k, v in types.items()},
                "largest": float(obs.mag.max()) if len(obs) else None,
            },
        }

    def observed(self, ctx: BenchContext) -> Events:
        t = pq.read_table(ctx.task_dir(self.task_id) / "observed.parquet")
        return Events(
            time=np.asarray(t.column("time").to_numpy(), float),
            lat=np.asarray(t.column("lat").to_numpy(), float),
            lon=np.asarray(t.column("lon").to_numpy(), float),
            depth_km=np.asarray(t.column("depth_km").to_numpy(), float),
            mag=np.asarray(t.column("magnitude").to_numpy(), float),
            mag_type=np.asarray(t.column("magnitude_type").to_pylist(), dtype=np.str_),
            token_hash=np.asarray(t.column("token_hash").to_pylist(), dtype=np.str_),
        )

    def supports(self, method: Any) -> bool:
        return isinstance(method, Forecaster)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        spec = self.spec
        from craton.trust.predictive import predictive_of

        # Computed once per (method, task, seed): CB-05, CB-07, CB-08 and CB-11 read it too.
        fc = predictive_of(method, spec, ctx, seed).forecast
        days = len(spec.day_starts)
        if fc.rates.shape != (days, spec.grid.n_cells):
            raise ValueError(f"{self.task_id}: forecast has shape {fc.rates.shape}")
        if np.any(fc.rates < 0) or not np.all(np.isfinite(fc.rates)):
            raise ValueError(f"{self.task_id}: forecast rates must be finite and >= 0")
        return score_forecast(fc, self.observed(ctx), spec, seed)


#: Days holding a quake this big are left out of ``bits_without_big_days``.
BIG_DAY_MAGNITUDE = 5.0


def bit_split(
    fc: GriddedForecast,
    expected: FloatArray,
    day: Any,
    cell: Any,
    mag: Any,
    counts: Any,
    first: int,
    n_obs: int,
    tag: str,
) -> dict[str, float]:
    """The score per quake, split exactly into where its bits come from.

    A Poisson forecast over (day, cell, bin) is a Poisson total per day times a multinomial
    over that day's cells and bins, so the log-likelihood is a sum of:

        how_many   N log R − R − log N! per day (the day's total, R expected, N seen)
        where      log of the share of the day's rate in the quake's cell
        how_big    log of the share of the cell's rate in the quake's magnitude bin
        shuffle    log N! − Σ log n! (how many ways the day's quakes could be arranged): the
                   same for every forecast, so it never moves a comparison

    Each in bits per quake; the four add up to ``bits_per_eq``. ``day``, ``cell``, ``mag``
    and ``counts`` list the observed (day, cell, bin) keys and their counts.
    """
    per_day = expected.sum(axis=1)
    n_day = np.bincount(day, weights=counts, minlength=len(per_day))
    with np.errstate(divide="ignore"):
        how_many = float(
            np.sum(n_day * np.log(per_day)) - per_day.sum() - sum(math.lgamma(n + 1) for n in n_day)
        )
        share_cell = expected[day, cell] / per_day[day]
        shares = np.broadcast_to(fc.mag_shares, (*fc.rates.shape, fc.mag_shares.shape[-1]))
        share_mag = shares[day, cell, mag] / shares[day, cell, first:].sum(axis=-1)
        where = float(np.sum(counts * np.log(share_cell)))
        how_big = float(np.sum(counts * np.log(share_mag)))
    shuffle = float(
        sum(math.lgamma(n + 1) for n in n_day) - sum(math.lgamma(n + 1) for n in counts)
    )
    scale = 1.0 / (n_obs * csep.LN2)
    return {
        f"{tag}.how_many_bits": how_many * scale,
        f"{tag}.where_bits": where * scale,
        f"{tag}.how_big_bits": how_big * scale,
        f"{tag}.shuffle_bits": shuffle * scale,
    }


def score_forecast(fc: GriddedForecast, obs: Events, spec: ForecastSpec, seed: int) -> Score:
    """Joint Poisson log-likelihood per quake, plus the CSEP consistency tests, as extras."""
    g = spec.grid
    day = np.floor((obs.time - spec.window.t0) / DAY).astype(np.int64)
    cell = g.cell_of(obs.lat, obs.lon)
    mbin = g.mag_bin_of(obs.mag)
    extras: dict[str, float] = {}
    primary = math.nan
    for thr in THRESHOLDS:
        first = int(g.mag_bin_of(np.array([thr]))[0])
        sel = mbin >= first
        keys = (day[sel] * g.n_cells + cell[sel]) * g.n_mags + mbin[sel]
        uniq, counts = np.unique(keys, return_counts=True)
        d, rest = np.divmod(uniq, g.n_cells * g.n_mags)
        c, m = np.divmod(rest, g.n_mags)
        lam = fc.at(d, c, m)
        expected = fc.expected(first)
        with np.errstate(divide="ignore"):
            log_terms = counts * np.log(lam) - np.array([math.lgamma(n + 1) for n in counts])
        ll = float(np.sum(log_terms) - expected.sum())
        n_obs = int(sel.sum())
        bits = csep.bits_per_earthquake(ll, n_obs) if n_obs else math.nan
        tag = f"M{thr:g}"
        extras[f"{tag}.observed"] = float(n_obs)
        extras[f"{tag}.expected"] = float(expected.sum())
        extras[f"{tag}.bits_per_eq"] = bits
        if n_obs:
            extras.update(bit_split(fc, expected, d, c, m, counts, first, n_obs, tag))
            big_days = np.unique(day[obs.mag >= BIG_DAY_MAGNITUDE - 1e-9])
            calm = ~np.isin(d, big_days)
            n_calm = int(counts[calm].sum())
            if n_calm:
                ll_calm = float(
                    np.sum(log_terms[calm])
                    - np.delete(expected, big_days[big_days < len(expected)], axis=0).sum()
                )
                extras[f"{tag}.bits_without_big_days"] = csep.bits_per_earthquake(ll_calm, n_calm)
        per_day = expected.sum(axis=1)
        obs_day = np.bincount(day[sel], minlength=len(per_day))
        tests = [csep.n_test(e, int(o)) for e, o in zip(per_day, obs_day, strict=True)]
        extras[f"{tag}.days_too_few"] = float(np.mean([t.delta_1 < 0.025 for t in tests]))
        extras[f"{tag}.days_too_many"] = float(np.mean([t.delta_2 < 0.025 for t in tests]))
        shares = np.broadcast_to(fc.mag_shares, (*fc.rates.shape, g.n_mags))[..., first:]
        summed = np.einsum("dc,dcm->cm", fc.rates, shares)
        counts_cm = g.counts(obs.lat[sel], obs.lon[sel], obs.mag[sel])[:, first:]
        s = csep.s_test(summed, counts_cm, N_SIMS, seed)
        mt = csep.m_test(summed, counts_cm, N_SIMS, seed)
        extras[f"{tag}.s_test_quantile"] = s.quantile
        extras[f"{tag}.m_test_quantile"] = mt.quantile
        if thr == THRESHOLDS[0]:
            primary = bits
    return Score(primary, extras)


# ---------------------------------------------------------------- CB-04 GPS gap-fill


@dataclass(frozen=True)
class GapQueries:
    """Days of hidden GNSS stations: the site is known, the positions are not."""

    station: list[str]
    time: FloatArray
    site_lat: FloatArray  # median site, rounded to 0.001°
    site_lon: FloatArray


@runtime_checkable
class GapFiller(Protocol):
    name: str

    def fill(self, queries: GapQueries, ctx: BenchContext, seed: int) -> FloatArray:
        """(n, 3) east, north, up in metres; a constant offset per station doesn't matter."""
        ...


class GpsGapFill(Task):
    task_id = "CB-04"
    name = "GPS gap-fill"
    metric = "RMSE (mm), hidden stations, offset per station and component removed"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "stations": "the frozen held-out list",
            "days": "every day each held-out station has inside the region window",
            "revealed": "station code, day, site to 0.001°",
            "inputs": "every GNSS token of every other station",
            "stations_are": "stable sites only (see the split definition)",
            "offset": "each station's mean residual per component is removed before the RMSE",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        g = ctx.gnss
        sites = station_sites(g)
        sel = np.isin(g.station, ctx.heldout)
        h = g.select(sel)
        queries = pa.table(
            {
                "station": h.station.tolist(),
                "time": h.time,
                "site_lat": [round(sites[s][0], 3) for s in h.station],
                "site_lon": [round(sites[s][1], 3) for s in h.station],
                "token_hash": h.token_hash.tolist(),
            }
        )
        answers = pa.table({"token_hash": h.token_hash.tolist(), "enu": h.enu.tolist()})
        folder = ctx.task_dir(self.task_id)
        return {
            "files": {
                "answers.parquet": _write_table(folder / "answers.parquet", answers),
                "queries.parquet": _write_table(folder / "queries.parquet", queries),
            },
            "summary": {"stations": len(set(h.station.tolist())), "station_days": len(h)},
        }

    def load_queries(self, ctx: BenchContext) -> GapQueries:
        t = pq.read_table(ctx.task_dir(self.task_id) / "queries.parquet")
        return GapQueries(
            station=t.column("station").to_pylist(),
            time=np.asarray(t.column("time").to_numpy(), float),
            site_lat=np.asarray(t.column("site_lat").to_numpy(), float),
            site_lon=np.asarray(t.column("site_lon").to_numpy(), float),
        )

    def answer_hashes(self, ctx: BenchContext) -> set[str]:
        t = pq.read_table(ctx.task_dir(self.task_id) / "answers.parquet", columns=["token_hash"])
        return set(t.column("token_hash").to_pylist())

    def supports(self, method: Any) -> bool:
        return isinstance(method, GapFiller)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        q = self.load_queries(ctx)
        t = pq.read_table(ctx.task_dir(self.task_id) / "answers.parquet")
        truth = np.asarray(t.column("enu").to_pylist(), dtype=np.float64)
        pred = np.asarray(method.fill(q, ctx, seed), dtype=np.float64)
        if pred.shape != truth.shape or not np.all(np.isfinite(pred)):
            raise ValueError(f"CB-04: need finite (n, 3) positions, got {pred.shape}")
        return score_gapfill(np.asarray(q.station), truth, pred)


def score_gapfill(station: Any, truth: FloatArray, pred: FloatArray) -> Score:
    resid = (truth - pred) * 1000.0  # mm
    for s in np.unique(station):
        sel = station == s
        resid[sel] -= resid[sel].mean(axis=0)
    rmse = float(np.sqrt(np.mean(resid**2)))
    names = ("east", "north", "up")
    extras = {
        f"rmse_{n}_mm": float(np.sqrt(np.mean(resid[:, i] ** 2))) for i, n in enumerate(names)
    }
    extras["station_days"] = float(len(resid))
    extras["stations"] = float(len(np.unique(station)))
    return Score(rmse, extras)


# ---------------------------------------------------------------- CB-15 efficiency


#: The laptop's memory. A run that needs more can't be scored on CB-15 at all.
VRAM_BUDGET_BYTES = 6 * 2**30


@runtime_checkable
class EfficiencyReport(Protocol):
    name: str

    def efficiency(self, ctx: BenchContext, seed: int) -> dict[str, Any]:
        """Measurements of the reference training run for this seed: ``device`` (name),
        ``tokens_per_s``, ``flops_per_token``, ``peak_vram_bytes``, ``train_seconds``."""
        ...


class Efficiency(Task):
    """CB-15: how much of the GPU the training run used (MFU), inside the 6 GiB budget.

    Has no test files: its fingerprint is its definition, peak table included.
    """

    task_id = "CB-15"
    name = "Efficiency"
    metric = "model FLOPs utilisation (%) of the pretraining run, inside 6 GiB of VRAM"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        from craton.train.flops import PEAK_FLOPS

        return {
            "reference_run": "the training run of the CB-01 model (split pretrain), per seed",
            "flops_per_token": "6N + 12·L·H·Q·T (PaLM appendix B), N = trunk parameters",
            "peak_flops": PEAK_FLOPS,
            "vram_budget_bytes": VRAM_BUDGET_BYTES,
            "rule": "a run whose peak VRAM exceeds the budget cannot be scored",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        return {"files": {}, "summary": {"budget_gib": VRAM_BUDGET_BYTES / 2**30}}

    def supports(self, method: Any) -> bool:
        return isinstance(method, EfficiencyReport)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        from craton.train.flops import PEAK_FLOPS

        r = method.efficiency(ctx, seed)
        if r["device"] not in PEAK_FLOPS:
            raise ValueError(f"CB-15: no peak FLOP/s on record for {r['device']!r}")
        if r["peak_vram_bytes"] > VRAM_BUDGET_BYTES:
            raise ValueError(
                f"CB-15: peak VRAM {r['peak_vram_bytes'] / 2**30:.2f} GiB is over the 6 GiB budget"
            )
        mfu = r["tokens_per_s"] * r["flops_per_token"] / PEAK_FLOPS[r["device"]]
        return Score(
            100.0 * mfu,
            {
                "tokens_per_s": float(r["tokens_per_s"]),
                "peak_vram_mib": r["peak_vram_bytes"] / 2**20,
                "train_minutes": r["train_seconds"] / 60.0,
            },
        )


# ---------------------------------------------------------------- CB-17 in-context adaptation


#: CB-17: the region's history "begins" here; the context is its first N days from this moment.
CB17_START = "2022-01-01T00:00:00Z"
#: The test year: every quake inside the region, place and size hidden.
CB17_TEST = Window("cb17_test", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z")
CB17_DAYS = (0, 7, 30, 90, 365)


@dataclass(frozen=True)
class RegionQueries:
    """CB-17's question: a region's first year of quakes, and its next year's, hidden."""

    context_time: FloatArray
    context_values: FloatArray  # (n, 4) lat, lon, depth, magnitude
    context_type: list[str]
    time: FloatArray
    magnitude_type: list[str]

    def context(self, days: int) -> tuple[FloatArray, FloatArray, list[str]]:
        """The quakes of the first ``days`` days, and nothing after."""
        sel = self.context_time < to_seconds(CB17_START) + days * DAY
        types = [t for t, k in zip(self.context_type, sel, strict=True) if k]
        return self.context_time[sel], self.context_values[sel], types


@runtime_checkable
class InContextPredictor(Protocol):
    name: str

    def predict_in_context(
        self, queries: RegionQueries, days: int, ctx: BenchContext, seed: int
    ) -> dict[str, ChannelDist]:
        """{channel: distribution over every test quake}, reading only ``queries.context(days)``
        of the region (and the training split, which never saw it)."""
        ...


class InContextAdaptation(Task):
    """CB-17: does a model learn a place it has never seen from context alone?

    The model is trained on ``cb17_fit``, which holds nothing inside the region at any time.
    It is then shown the region's first N days of quakes (N = 0, 7, 30, 90, 365), with no
    retraining, and asked for every quake of the following year: where and how big, in bits
    on CB-01's bins. The score is the bits at N = 365; the whole curve is in the extras.
    """

    task_id = "CB-17"
    name = "In-context adaptation"
    metric = "bits per quake in a region never trained on, with its first 365 days in context"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        from craton.bench.splits import CB17_REGION

        return {
            "region": CB17_REGION.as_dict(),
            "context_start": CB17_START,
            "context_days": list(CB17_DAYS),
            "primary_days": CB17_DAYS[-1],
            "test": CB17_TEST.as_dict(),
            "bins": [b.__dict__ for b in CB01_BINS["seismic_catalog"]],
            "max_bits_per_channel": MAX_BITS,
            "revealed": "time and magnitude type of every test quake",
            "train_split": "cb17_fit",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.splits import CB17_REGION

        ev = ctx.events
        ev = ev.select(CB17_REGION.contains(ev.lat, ev.lon))
        t0 = to_seconds(CB17_START)
        before = ev.window(t0, t0 + CB17_DAYS[-1] * DAY)
        test = ev.window(CB17_TEST.t0, CB17_TEST.t1)
        context = pa.table(
            {
                "time": before.time,
                "lat": before.lat,
                "lon": before.lon,
                "depth_km": before.depth_km,
                "magnitude": before.mag,
                "magnitude_type": before.mag_type.tolist(),
            }
        )
        queries = pa.table(
            {
                "token_hash": test.token_hash.tolist(),
                "time": test.time,
                "magnitude_type": test.mag_type.tolist(),
            }
        )
        answers = pa.table(
            {
                "token_hash": test.token_hash.tolist(),
                "value": np.stack([test.lat, test.lon, test.depth_km, test.mag], 1).tolist(),
            }
        )
        folder = ctx.task_dir(self.task_id)
        return {
            "files": {
                "answers.parquet": _write_table(folder / "answers.parquet", answers),
                "context.parquet": _write_table(folder / "context.parquet", context),
                "queries.parquet": _write_table(folder / "queries.parquet", queries),
            },
            "summary": {
                "context_quakes": {
                    str(d): int((before.time < t0 + d * DAY).sum()) for d in CB17_DAYS
                },
                "test_quakes": len(test),
                "largest_test": float(test.mag.max()) if len(test) else None,
            },
        }

    def load_queries(self, ctx: BenchContext) -> RegionQueries:
        folder = ctx.task_dir(self.task_id)
        c = pq.read_table(folder / "context.parquet")
        q = pq.read_table(folder / "queries.parquet")
        values = np.stack(
            [np.asarray(c.column(k).to_numpy(), float) for k in ("lat", "lon", "depth_km")]
            + [np.asarray(c.column("magnitude").to_numpy(), float)],
            axis=1,
        )
        return RegionQueries(
            context_time=np.asarray(c.column("time").to_numpy(), float),
            context_values=values.reshape(-1, 4),
            context_type=c.column("magnitude_type").to_pylist(),
            time=np.asarray(q.column("time").to_numpy(), float),
            magnitude_type=q.column("magnitude_type").to_pylist(),
        )

    def answer_hashes(self, ctx: BenchContext) -> set[str]:
        t = pq.read_table(ctx.task_dir(self.task_id) / "answers.parquet", columns=["token_hash"])
        return set(t.column("token_hash").to_pylist())

    def supports(self, method: Any) -> bool:
        return isinstance(method, InContextPredictor)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        queries = self.load_queries(ctx)
        t = pq.read_table(ctx.task_dir(self.task_id) / "answers.parquet")
        truth = np.asarray(t.column("value").to_pylist(), dtype=np.float64).reshape(-1, 4)
        extras: dict[str, float] = {"test_quakes": float(len(truth))}
        curve = {}
        for days in CB17_DAYS:
            dists = method.predict_in_context(queries, days, ctx, seed)
            bits = np.zeros(len(truth))
            for c, b in enumerate(CB01_BINS["seismic_catalog"]):
                lo, hi = b.edges_of(truth[:, c])
                mass = dists[b.channel].mass(lo, hi)
                ch = np.minimum(-np.log2(np.maximum(mass, 2.0**-MAX_BITS)), MAX_BITS)
                if days == CB17_DAYS[-1]:
                    extras[f"{b.channel}_bits"] = float(np.mean(ch))
                bits += ch
            curve[days] = float(np.mean(bits))
            extras[f"bits_{days}d"] = curve[days]
        extras["gain_bits"] = curve[CB17_DAYS[0]] - curve[CB17_DAYS[-1]]
        return Score(curve[CB17_DAYS[-1]], extras)


# ---------------------------------------------------------------- the registry


TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Add a task to CratonBench. An ID is given once and never reused or redefined."""
    if task.task_id in TASKS:
        raise ValueError(f"{task.task_id} is taken; a changed task needs a new ID")
    TASKS[task.task_id] = task
    return task


register_task(GeoPerplexity())
register_task(
    SequenceForecast(
        "CB-02",
        "Apennines 2016-17 forecast",
        Window("cb02_test", "2016-08-25T00:00:00Z", ERA_A.end),
        "cb02_fit",
    )
)
register_task(
    SequenceForecast(
        "CB-03",
        "L'Aquila 2009 era holdout",
        Window("cb03_test", ERA_B.start, ERA_B.end),
        "cb03_fit",
    )
)
register_task(GpsGapFill())
register_task(Efficiency())
register_task(InContextAdaptation())

assert all(t.fit_split in SPLITS for t in TASKS.values() if isinstance(t, SequenceForecast))
assert to_seconds(ERA_A.start) < to_seconds("2016-08-25T00:00:00Z")

# The P7 trust tasks register themselves through the same registry (CB-05 … CB-11).
from craton.bench import locks as _locks  # noqa: E402, F401
