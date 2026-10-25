"""CB-09, two planets: tie ETAS where ETAS is the truth, beat it where physics is.

A forecaster scored only on the real Earth can't tell its skill from its luck: nobody knows
what the best possible forecast of 2016 was. On a fake planet somebody does. Two kinds:

- **The ETAS planet** (``etas_italy``): central Italy exactly as the ETAS floor sees it, its
  own fitted parameters as the whole truth. Nothing is left for a learned correction to
  find, so a forecaster that beats fitted ETAS here by more than noise is fitting noise.
  The test is a tie.
- **The physics planet** (``rate_state``, ``craton.sim.rate_state``): rate-and-state faults,
  Coulomb stress transfer with stress shadows, a fluid source nobody is told about. ETAS is
  the wrong model here in ways that are known, so there is room above it. The test is a win.

Each planet copies CB-02's calendar: quakes from 1985, a forecaster fits on everything
before 2016-08-24 (the ``cb02_fit`` cut-off) and forecasts every day from 2016-08-25 to the
end of 2017 on the same 0.1° grid, scored the same way (``score_forecast``). The planets are
the first three seeds of each simulator whose test window holds an M5+ and at least
``MIN_TEST_QUAKES`` M3+, the way CB-02's holds a sequence; they're written once and frozen.
Benchmark seed ``k`` scores planet ``k`` of each kind with method seed ``k``.

The primary number is bits per M3+ quake on the physics planet. The extras hold the ETAS
planet's score, what ETAS with the planet's true parameters scores there (the line nobody
can beat on average), the bit split on both, and the **sim-to-real gap**: how far each
simulator's catalogues are from the real Apennines on a few plain statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.splits import ERA_A, SPLITS, Window
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    ForecastSpec,
    GriddedForecast,
    Score,
    Task,
    _write_table,
    forecast_grid,
    register_task,
    score_forecast,
)
from craton.paths import data_dir
from craton.sim import ITALY_AS_ETAS, SIMULATORS, SimSpec
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

PLANET_START = "1985-01-01T00:00:00Z"
FIT_SPLIT = "cb02_fit"
WINDOW = Window("cb09_test", "2016-08-25T00:00:00Z", ERA_A.end)
#: (simulator, smallest magnitude it makes) per kind of planet.
KINDS = {"etas_planet": ("etas_italy", 2.0), "physics_planet": ("rate_state", 1.5)}
PRIMARY = "physics_planet"
PLANETS_PER_KIND = 3
MIN_TEST_QUAKES = 100
MIN_TEST_LARGEST = 5.0
MAX_SEEDS = 200
#: The real catalogue the planets are compared with: CB-02's split, from the ETAS fit start.
REAL_FROM = "2005-01-01T00:00:00Z"
STAT_MC = 2.0
REPORT_STEP = 0.1


def planet_file(kind: str, k: int) -> str:
    return f"{kind}-{k}.parquet"


def events_table(ev: Events) -> pa.Table:
    return pa.table(
        {
            "time": ev.time,
            "lat": ev.lat,
            "lon": ev.lon,
            "depth_km": ev.depth_km,
            "magnitude": ev.mag,
            "magnitude_type": ev.mag_type.tolist(),
            "token_hash": ev.token_hash.tolist(),
        }
    )


def read_events(path: Path) -> Events:
    t = pq.read_table(path)
    return Events(
        time=np.asarray(t.column("time").to_numpy(), float),
        lat=np.asarray(t.column("lat").to_numpy(), float),
        lon=np.asarray(t.column("lon").to_numpy(), float),
        depth_km=np.asarray(t.column("depth_km").to_numpy(), float),
        mag=np.asarray(t.column("magnitude").to_numpy(), float),
        mag_type=np.asarray(t.column("magnitude_type").to_pylist(), dtype=np.str_),
        token_hash=np.asarray(t.column("token_hash").to_pylist(), dtype=np.str_),
    )


def in_grid(ev: Events) -> Events:
    g = forecast_grid()
    return ev.select((g.cell_of(ev.lat, ev.lon) >= 0) & (g.mag_bin_of(ev.mag) >= 0))


def simulate_planet(kind: str, seed: int) -> Events:
    name, m_min = KINDS[kind]
    g = forecast_grid()
    t0, t1 = to_seconds(PLANET_START), WINDOW.t1
    # A catalogue reports to 0.1, so "M2.0" is everything from 1.95 (the grid's bins and
    # every fit read magnitudes that way): simulate from half a step below, then round.
    low = m_min - REPORT_STEP / 2
    spec = SimSpec((g.lat_min, g.lat_max, g.lon_min, g.lon_max), (t1 - t0) / DAY, t0, low)
    return SIMULATORS[name].simulate(spec, seed)


def qualifies(ev: Events) -> bool:
    test = in_grid(ev.window(WINDOW.t0, WINDOW.t1))
    return bool(
        (test.mag >= 3.0 - 1e-9).sum() >= MIN_TEST_QUAKES
        and len(test)
        and test.mag.max() >= MIN_TEST_LARGEST - 1e-9
    )


# ---------------------------------------------------------------- sim-to-real gap


def catalog_stats(ev: Events, t0: float, t1: float) -> dict[str, float]:
    """A handful of plain numbers that any catalogue has, M ≥ ``STAT_MC``, in [t0, t1)."""
    ev = in_grid(ev.window(t0, t1).above(STAT_MC))
    g = forecast_grid()
    days = round((t1 - t0) / DAY)
    per_day = np.bincount(((ev.time - t0) // DAY).astype(np.int64), minlength=days)
    busiest = np.sort(per_day)[::-1][: max(days // 100, 1)]
    cells = np.sort(np.bincount(g.cell_of(ev.lat, ev.lon), minlength=g.n_cells))[::-1]
    return {
        "quakes_per_day": len(ev) / days,
        "b_value": csep.b_value(ev.mag, STAT_MC),
        "busiest_1pct_days_share": float(busiest.sum() / max(len(ev), 1)),
        "busiest_5pct_cells_share": float(cells[: g.n_cells // 20].sum() / max(len(ev), 1)),
        "daily_count_cv": float(per_day.std() / max(per_day.mean(), 1e-9)),
        "largest": float(ev.mag.max()) if len(ev) else math.nan,
    }


#: The statistics the gap averages (the largest quake is a single draw, shown but not scored).
GAP_STATS = (
    "quakes_per_day",
    "b_value",
    "busiest_1pct_days_share",
    "busiest_5pct_cells_share",
    "daily_count_cv",
)


def sim_to_real_gap(planet: dict[str, float], real: dict[str, float]) -> float:
    """Mean |log₂(planet ÷ real)| over ``GAP_STATS``: 1 = off by a factor of two on average."""
    return float(np.mean([abs(math.log2(planet[k] / real[k])) for k in GAP_STATS]))


# ---------------------------------------------------------------- a planet as a context


@dataclass
class PlanetContext(BenchContext):
    """A fake planet dressed as a region: the same calls a forecaster makes on central Italy
    (``events``, ``train_events``, ``split_dir``) answered from one synthetic catalogue."""

    planet: Events | None = None
    planet_id: str = ""
    _splits_ready: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.region = f"planet-{self.planet_id}"
        self.snapshot_id = self.planet_id

    @property
    def has_data(self) -> bool:
        return self.planet is not None

    @property
    def bench_dir(self) -> Path:
        return data_dir(self.root) / "bench" / "planets" / self.planet_id

    @property
    def events(self) -> Events:
        assert self.planet is not None
        return self.planet

    def train_events(self, split: str) -> Events:
        before = SPLITS[split].before
        if before is None:
            raise ValueError(f"{split} has no cut-off; a planet only has its past")
        return self.events.select(self.events.time < to_seconds(before))

    def split_dir(self, name: str) -> Path:
        folder = self.bench_dir / "splits" / name
        if name not in self._splits_ready:
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / "manifest.json"
            body = f'{{"id": "{self.planet_id}{name}"}}\n'
            if not manifest.is_file() or manifest.read_text("utf-8") != body:
                manifest.write_text(body, "utf-8")
            self._splits_ready.add(name)
        return folder


def truth_forecast(planet: PlanetContext, spec: ForecastSpec) -> GriddedForecast:
    """ETAS with the ETAS planet's own parameters and its uniform background."""
    from craton.bench.baselines import _shares

    _, mc = KINDS["etas_planet"]
    b = 1.16
    g = spec.grid
    proj = etas.Projection.for_grid(g)
    box = etas.box_km(g, proj)
    t_ref = to_seconds(PLANET_START)
    bg = etas.Background(np.zeros(1), np.zeros(1), np.ones(1), box, floor=1.0)
    fit = etas.EtasFit(ITALY_AS_ETAS, bg, mc, b, t_ref, etas.FitWindow(0.0, 1.0), 0, 0.0, True)
    hist = planet.events.window(t_ref, spec.window.t1).above(mc)
    cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, t_ref)
    daily = etas.forecast_days(fit, cat, (spec.day_starts - t_ref) / DAY, g, proj)
    scale, shares = _shares(b, mc, g)
    return GriddedForecast(daily.rates * scale, shares)


# ---------------------------------------------------------------- the task


class TwoPlanets(Task):
    task_id = "CB-09"
    name = "Two planets"
    metric = "bits per M3+ quake on the physics planet, daily 0.1° forecasts"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        return {
            "kinds": {k: {"simulator": s, "m_min": m} for k, (s, m) in KINDS.items()},
            "primary": PRIMARY,
            "calendar": {"start": PLANET_START, "fit_split": FIT_SPLIT, **WINDOW.as_dict()},
            "grid": forecast_grid().__dict__,
            "planets": f"per kind, the first {PLANETS_PER_KIND} simulator seeds whose test "
            f"window holds an M{MIN_TEST_LARGEST:g}+ and at least {MIN_TEST_QUAKES} M3+",
            "magnitudes": f"simulated from m_min - {REPORT_STEP / 2:g}, reported to "
            f"{REPORT_STEP:g}",
            "seeds": "benchmark seed k scores planet k of each kind with method seed k",
            "scoring": "as CB-02 (score_forecast)",
            "gap": f"mean |log2(planet / real)| over {', '.join(GAP_STATS)}, M>={STAT_MC:g}, "
            f"vs the real cb02_fit split from {REAL_FROM[:10]}",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        t_real, t_cut = to_seconds(REAL_FROM), to_seconds(SPLITS[FIT_SPLIT].before or "")
        real = catalog_stats(ctx.train_events(FIT_SPLIT), t_real, t_cut)
        summary: dict[str, Any] = {"real": real}
        for kind in KINDS:
            chosen: list[int] = []
            stats = []
            for seed in range(MAX_SEEDS):
                ev = simulate_planet(kind, seed)
                if not qualifies(ev):
                    continue
                k = len(chosen)
                files[planet_file(kind, k)] = _write_table(
                    folder / planet_file(kind, k), events_table(ev)
                )
                chosen.append(seed)
                st = catalog_stats(ev, t_real, t_cut)
                test = in_grid(ev.window(WINDOW.t0, WINDOW.t1))
                stats.append(
                    {
                        **st,
                        "gap": sim_to_real_gap(st, real),
                        "test_m3": int((test.mag >= 3.0 - 1e-9).sum()),
                        "test_largest": float(test.mag.max()),
                        "quakes": len(ev),
                    }
                )
                if len(chosen) == PLANETS_PER_KIND:
                    break
            if len(chosen) < PLANETS_PER_KIND:
                raise RuntimeError(f"{kind}: only {len(chosen)} planets in {MAX_SEEDS} seeds")
            summary[kind] = {
                "seeds": chosen,
                "planets": stats,
                "gap": float(np.mean([s["gap"] for s in stats])),
            }
        return {"files": files, "summary": summary}

    def planet(self, ctx: BenchContext, kind: str, k: int) -> PlanetContext:
        key = f"cb09:{kind}:{k}"
        if key not in ctx._cache:
            path = ctx.task_dir(self.task_id) / planet_file(kind, k)
            from craton.store.parquet import file_sha256

            ctx._cache[key] = PlanetContext(
                root=ctx.root,
                planet=read_events(path),
                planet_id=f"{kind}-{file_sha256(path)[:12]}",
            )
        out: PlanetContext = ctx._cache[key]
        return out

    def spec(self, kind: str, k: int) -> ForecastSpec:
        return ForecastSpec(f"{self.task_id}-{kind}-{k}", FIT_SPLIT, WINDOW, forecast_grid())

    def supports(self, method: Any) -> bool:
        return isinstance(method, Forecaster)

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        k = seed % PLANETS_PER_KIND
        extras: dict[str, float] = {"planet": float(k)}
        primary = math.nan
        for kind in KINDS:
            pctx = self.planet(ctx, kind, k)
            spec = self.spec(kind, k)
            obs = in_grid(pctx.events.window(WINDOW.t0, WINDOW.t1))
            fc = method.forecast(spec, pctx, seed)
            days = len(spec.day_starts)
            if fc.rates.shape != (days, spec.grid.n_cells):
                raise ValueError(f"{spec.task_id}: forecast has shape {fc.rates.shape}")
            sc = score_forecast(fc, obs, spec, seed)
            for name in (
                "bits_per_eq",
                "how_many_bits",
                "where_bits",
                "how_big_bits",
                "observed",
                "expected",
            ):
                extras[f"{kind}.{name}"] = sc.extras[f"M3.{name}"]
            if kind == "etas_planet":
                truth = score_forecast(truth_forecast(pctx, spec), obs, spec, seed)
                extras["etas_planet.truth_bits"] = truth.value
            if kind == PRIMARY:
                primary = sc.value
        return Score(primary, extras)


register_task(TwoPlanets())
