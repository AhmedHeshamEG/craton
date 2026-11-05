"""CB-33, planted stress: does a forecaster's picture of the crust match the physics planet's?

The physics planet (P8a, ``craton.sim.rate_state``) knows something no real catalogue does:
the state of every fault patch, at every moment. In rate-and-state friction that state is a
stress (Dieterich 1994): ln(R / r), how far a patch's quake rate sits above (+) or below (−)
its own background, every quake's Coulomb push and the fluid front since the planet began
folded into one number. This task asks whether what a forecaster believes about the planet,
read through the same law, matches it.

**The truth.** CB-09's three physics planets, simulated again with ``StressRecord`` on (it
draws no random numbers, and ``prepare`` checks every quake comes out the same as the frozen
catalogue). Per 0.1° cell, the state of its patches averaged the way the cell's quakes see
it: ln(Σ r_i e^{s_i} / Σ r_i), the cell's true rate over its background. Every seventh test
midnight from the seventh, the same calendar as CB-09 (2016-08-25 to the end of 2017).

**What is graded.** Most of a planet sits at steady loading (half its cells within ±0.002 of
zero): nothing there to infer. The questions are the cells at least ``OFF_STEADY`` (10% in
rate) off steady that day, ~90 per day per planet, the stress shadows among them. The score
is Spearman's rank correlation between truth and answer over those cells, averaged over the
graded days. Ranks, so any monotone reading of the answer scores the same.

**Reading a forecaster.** A forecast is turned into a state by the same law the truth obeys:
ln(rate / the cell's long-run rate), the long-run rate being its M1.5+ quakes from 1985 to
the cut (the forecaster's own training catalogue), plus a half. A method may instead answer
directly (``PlantedReader``); the floor ``quake_density`` does: the cell's last year of
M1.5+ quakes over its long-run rate.

Benchmark seed ``k`` scores physics planet ``k`` with method seed ``k``, as CB-09 does, so a
Field trained there for CB-09 answers here without training again.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from craton.bench import planets
from craton.bench.data import DAY, Events
from craton.bench.splits import SPLITS
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    Score,
    Task,
    _write_table,
    forecast_grid,
    load_frozen,
    register_task,
)
from craton.sim import SIMULATORS, SimSpec
from craton.sim.rate_state import StressRecord
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

KIND = "physics_planet"
GRADE_EVERY = 7
OFF_STEADY = 0.1
SHADOW = -1.0
LONG_RUN_M = 1.5
DENSITY_DAYS = 365


def truth_file(k: int) -> str:
    return f"state-{k}.parquet"


def graded_days() -> np.ndarray[Any, np.dtype[np.int64]]:
    """Indices into the test window's days that are graded."""
    n = round((planets.WINDOW.t1 - planets.WINDOW.t0) / DAY)
    return np.arange(GRADE_EVERY - 1, n, GRADE_EVERY)


def cell_state(record: StressRecord) -> FloatArray:
    """(times, cells) the cells' state from the patches': ln(Σ r e^s / Σ r)."""
    g = forecast_grid()
    p = record.patches
    assert p is not None
    cell = g.cell_of(p.lat, p.lon)
    ok = cell >= 0
    w = np.zeros(g.n_cells)
    np.add.at(w, cell[ok], p.r[ok])
    num = np.zeros((len(record.at), g.n_cells))
    np.add.at(num.T, cell[ok], (np.exp(record.state[:, ok]) * p.r[ok]).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(num / w)
    return np.asarray(np.where(w > 0, out, 0.0), dtype=np.float64)


def long_run_rate(events: Events) -> FloatArray:
    """(cells,) M1.5+ quakes per day from the planet's start to the CB-09 cut, plus a half."""
    g = forecast_grid()
    t0, cut = to_seconds(planets.PLANET_START), to_seconds(SPLITS[planets.FIT_SPLIT].before or "")
    ev = events.window(t0, cut).above(LONG_RUN_M - 1e-9)
    c = g.cell_of(ev.lat, ev.lon)
    n = np.bincount(c[c >= 0], minlength=g.n_cells).astype(np.float64)
    return np.asarray((n + 0.5) / ((cut - t0) / DAY), dtype=np.float64)


class PlantedQuery:
    """What a ``PlantedReader`` gets: the planet's catalogue (read only before each midnight),
    the graded midnights and the grid."""

    def __init__(self, events: Events, midnights: FloatArray) -> None:
        self.events = events
        self.midnights = midnights
        self.grid = forecast_grid()


@runtime_checkable
class PlantedReader(Protocol):
    name: str

    def planted_state(self, query: PlantedQuery, seed: int) -> FloatArray:
        """(graded midnights, cells) any monotone reading of each cell's state."""
        ...


class QuakeDensity:
    """CB-33's floor: the cell's last year of M1.5+ quakes over its long-run rate."""

    name = "quake_density"
    kind = "baseline"

    def planted_state(self, query: PlantedQuery, seed: int) -> FloatArray:
        g = query.grid
        ev = query.events.above(LONG_RUN_M - 1e-9)
        cell = g.cell_of(ev.lat, ev.lon)
        base = long_run_rate(query.events)
        out = np.zeros((len(query.midnights), g.n_cells))
        for i, t in enumerate(query.midnights):
            m = (ev.time < t) & (ev.time >= t - DENSITY_DAYS * DAY) & (cell >= 0)
            n = np.bincount(cell[m], minlength=g.n_cells)
            out[i] = np.log((n + 0.5) / DENSITY_DAYS / base)
        return out


def grade(truth: FloatArray, answer: FloatArray) -> dict[str, float]:
    """Spearman over the off-steady cells of each graded day, and how the shadows fare."""
    rhos, below, shadows, cells = [], 0.0, 0.0, 0.0
    for t, a in zip(truth, answer, strict=True):
        m = np.abs(t) >= OFF_STEADY
        if m.sum() < 3:
            continue
        cells += float(m.sum())
        flat = np.ptp(a[m]) == 0  # a flat answer ranks nothing
        r = 0.0 if flat else float(spearmanr(a[m], t[m])[0])
        rhos.append(0.0 if math.isnan(r) else r)
        s = t[m] <= SHADOW
        below += float((a[m][s] < np.median(a[m])).sum())
        shadows += float(s.sum())
    return {
        "spearman": float(np.mean(rhos)) if rhos else math.nan,
        "graded_days": float(len(rhos)),
        "cells_per_day": cells / max(len(rhos), 1),
        "shadow_cells_per_day": shadows / max(len(rhos), 1),
        "shadows_below_median": below / shadows if shadows else math.nan,
    }


class PlantedStress(Task):
    task_id = "CB-33"
    name = "Planted stress"
    metric = "rank correlation of inferred and true state, off-steady cells of physics planets"
    direction = "higher"

    def __init__(self) -> None:
        self.cb09 = planets.TwoPlanets()

    def definition(self) -> dict[str, Any]:
        return {
            "planets": f"CB-09's {planets.PLANETS_PER_KIND} {KIND}s, simulated again with the "
            "stress record on; every quake checked against CB-09's frozen catalogue",
            "truth": "per 0.1 deg cell, ln(sum r_i exp(s_i) / sum r_i) over its patches, "
            "s = ln(1 / (gamma * tau_dot)) (Dieterich 1994), at test midnights",
            "calendar": planets.WINDOW.as_dict(),
            "graded": f"every {GRADE_EVERY}th midnight from the {GRADE_EVERY}th; cells with "
            f"|state| >= {OFF_STEADY:g}",
            "metric": "Spearman rank correlation per graded day, averaged",
            "forecaster": f"ln(daily rate / (M{LONG_RUN_M:g}+ quakes per day, planet start to "
            "the cb02_fit cut, + 0.5 over the span))",
            "seeds": "benchmark seed k scores physics planet k with method seed k",
            "grid": forecast_grid().__dict__,
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        frozen = (load_frozen(ctx.region) or {}).get("tasks", {}).get("CB-09")
        if frozen is None:
            raise RuntimeError("CB-33 is built on CB-09's frozen planets: freeze CB-09 first")
        seeds = frozen["summary"][KIND]["seeds"]
        folder = ctx.task_dir(self.task_id)
        g = forecast_grid()
        t0 = to_seconds(planets.PLANET_START)
        days = graded_days()
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"seeds": seeds, "planets": []}
        for k, seed in enumerate(seeds):
            name, m_min = planets.KINDS[KIND]
            low = m_min - planets.REPORT_STEP / 2
            spec = SimSpec(
                (g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                (planets.WINDOW.t1 - t0) / DAY,
                t0,
                low,
            )
            rec = StressRecord(at=(planets.WINDOW.t0 - t0) / DAY + days)
            ev = SIMULATORS[name].simulate(spec, seed, rec)
            frozen_ev = self.cb09.planet(ctx, KIND, k).events
            if not (
                len(ev) == len(frozen_ev)
                and np.array_equal(ev.time, frozen_ev.time)
                and np.array_equal(ev.mag, frozen_ev.mag)
                and np.array_equal(ev.lat, frozen_ev.lat)
            ):
                raise RuntimeError(f"physics planet {k} no longer matches CB-09's frozen one")
            state = cell_state(rec)
            files[truth_file(k)] = _write_table(
                folder / truth_file(k),
                pa.table({f"d{d}": state[i].astype(np.float32) for i, d in enumerate(days)}),
            )
            off = np.abs(state) >= OFF_STEADY
            summary["planets"].append(
                {
                    "seed": int(seed),
                    "cells_per_day": float(off.sum(axis=1).mean()),
                    "shadow_cells_per_day": float((state <= SHADOW).sum(axis=1).mean()),
                    "largest_state": float(state.max()),
                    "deepest_shadow": float(state.min()),
                }
            )
        return {"files": files, "summary": summary}

    def truth(self, ctx: BenchContext, k: int) -> FloatArray:
        t = pq.read_table(ctx.task_dir(self.task_id) / truth_file(k))
        return np.stack([np.asarray(t.column(f"d{d}").to_numpy(), float) for d in graded_days()])

    def supports(self, method: Any) -> bool:
        return isinstance(method, PlantedReader | Forecaster)

    def answer(self, method: Any, ctx: BenchContext, k: int, seed: int) -> FloatArray:
        pctx = self.cb09.planet(ctx, KIND, k)
        spec = self.cb09.spec(KIND, k)
        days = graded_days()
        if isinstance(method, PlantedReader):
            query = PlantedQuery(pctx.events, spec.day_starts[days])
            out = np.asarray(method.planted_state(query, seed), dtype=np.float64)
        else:
            fc = method.forecast(spec, pctx, seed)
            rates = np.asarray(fc.rates, dtype=np.float64)[days]
            out = np.log(np.maximum(rates, 1e-30) / long_run_rate(pctx.events)[None])
        if out.shape != (len(days), spec.grid.n_cells):
            raise ValueError(f"{method.name}: CB-33 answer has shape {out.shape}")
        return out

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        k = seed % planets.PLANETS_PER_KIND
        g = grade(self.truth(ctx, k), self.answer(method, ctx, k, seed))
        return Score(g.pop("spearman"), {"planet": float(k), **g})


register_task(PlantedStress())
