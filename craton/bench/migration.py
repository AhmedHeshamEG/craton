"""CB-24, migration: after the first big shock, where does the sequence light up next?

A sequence doesn't stay where it starts. Central Italy 2016 began at Amatrice (24 August),
jumped north to Visso (26 October), filled the gap at Norcia (30 October) and then went
south to Campotosto (18 January 2017). If a model has learned which patches of crust are
wired to which, from decades of quakes before the sequence, it should know where the
sequence goes next better than a ruler does.

CB-24 asks exactly that, in cells of 0.1°. A cell **lights up** at its first quake of the
scored magnitude (the region's completeness + 1; M3 in central Italy) on or after the day of
the first big shock. The first cell to light is given. Then, before every next lighting, the
method sees the cells lit so far (where, when, how big their first quake was) and the
region's quakes before the sequence, and ranks every cell that hasn't lit yet. It never sees
the sequence's other quakes: the question is the wiring, not the aftershock rate of the
last few hours, which CB-02 already asks.

The score is how often the cell that really lit next is in the method's top 3, over the
first ``MAX_LIGHTS`` lightings of each sequence, averaged over the sequences. Ties are
broken at random, in expectation, so a method that ranks nothing scores a coin's 3 in
~1,000.

The sequences are central Italy 2016-17 (the one the plan names), L'Aquila 2009, and
CB-22's fourteen around the world, in the same 3° × 3.5° boxes, so the test runs anywhere a
region file exists. The cut is the midnight before each first big shock: a method may read
nothing after it except the lightings it is shown.

Floors:

- ``distance_only``: the cells closest to the last one that lit. A sequence creeps.
- ``quake_history``: the cells with the most quakes of the scored size before the
  sequence. Busy places stay busy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.tasks import BenchContext, Score, Task, _write_table, forecast_grid, register_task
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: Lightings scored per sequence after the first: the early ones are the migration; after
#: a few dozen, a big sequence (Tohoku lit 675 cells in a year) is lighting its whole box.
MAX_LIGHTS = 50
#: How many of the method's best cells count as a hit.
TOP = 3
#: Days after the first big shock's midnight in which a cell may light.
WINDOW_DAYS = 365
SEEN, ZERO_SHOT = "seen", "zero_shot"
HALVES = (SEEN, ZERO_SHOT)
EARTH_KM = 6371.0


@dataclass(frozen=True)
class MigrationCase:
    """One sequence: its region, the cut (the midnight before its first big shock), its grid
    and the magnitude at which a cell lights."""

    key: str
    name: str
    region: str
    half: str
    cut: str
    grid: csep.Grid
    mc: float
    threshold: float


def cases(root: Any) -> list[MigrationCase]:
    """Central Italy's two sequences, then CB-22's, each in its own region's terms."""
    from craton.adapters.region import load_region
    from craton.bench.world import SEQUENCES, sequence_spec

    g = forecast_grid()
    italy = [
        MigrationCase(
            "amatrice_norcia",
            "Amatrice-Visso-Norcia 2016",
            "central_italy",
            SEEN,
            "2016-08-24T00:00:00Z",
            g,
            2.0,
            3.0,
        ),
        MigrationCase(
            "laquila", "L'Aquila 2009", "central_italy", SEEN, "2009-04-06T00:00:00Z", g, 2.0, 3.0
        ),
    ]
    world = []
    for s in SEQUENCES:
        spec = sequence_spec(s, load_region(s.region, root))
        world.append(
            MigrationCase(
                s.key, s.name, s.region, s.half, spec.fit_end, spec.grid, spec.mc, spec.scored
            )
        )
    return italy + world


def km_matrix(lat0: FloatArray, lon0: FloatArray, lat: FloatArray, lon: FloatArray) -> FloatArray:
    """Great-circle km between every (lat0, lon0) and every (lat, lon): shape (len0, len)."""
    p0, p = np.radians(lat0)[:, None], np.radians(lat)[None, :]
    dl = np.radians(lon[None, :] - lon0[:, None])
    h = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return np.asarray(2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1))), dtype=np.float64)


# ---------------------------------------------------------------- the lightings


@dataclass(frozen=True)
class Lightings:
    """The cells of one sequence in the order they lit, with the quake that lit each."""

    cell: IntArray
    quakes: Events  # one per lit cell, same order

    def __len__(self) -> int:
        return len(self.cell)


def lightings_of(ev: Events, case: MigrationCase) -> Lightings:
    """Every cell's first quake at or above the threshold, from the cut, in time order."""
    t0 = to_seconds(case.cut)
    seq = ev.window(t0, t0 + WINDOW_DAYS * DAY).above(case.threshold)
    cell = case.grid.cell_of(seq.lat, seq.lon)
    seq, cell = seq.select(cell >= 0), cell[cell >= 0]
    _, first = np.unique(cell, return_index=True)
    order = np.sort(first)
    return Lightings(cell[order].astype(np.int64), seq.select(order))


# ---------------------------------------------------------------- the protocol


@dataclass(frozen=True)
class MigrationQuery:
    """What a method sees before one lighting: the grid, the history before the cut (from the
    region's completeness), and the cells lit so far with the quakes that lit them."""

    case: str
    region: str
    half: str
    grid: csep.Grid
    cut: str
    mc: float
    threshold: float
    history: Events
    lit_cell: IntArray
    lit: Events

    @property
    def unlit(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        out = np.ones(self.grid.n_cells, dtype=bool)
        out[self.lit_cell] = False
        return out


@runtime_checkable
class MigrationRanker(Protocol):
    name: str

    def next_cell_scores(self, query: MigrationQuery, seed: int) -> FloatArray:
        """One score per grid cell, higher = more likely to light next; lit cells are
        ignored; NaN = no idea (ranked last)."""
        ...


def top_hit(scores: FloatArray, unlit: np.ndarray[Any, Any], true: int) -> tuple[float, float]:
    """(chance the true cell is in the top ``TOP``, expected reciprocal rank), ties broken at
    random among the unlit cells."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    v = s[true]
    above = int(np.sum(unlit & (s > v)))
    ties = int(np.sum(unlit & (s == v))) - 1  # the others it ties with
    hit = min(max((TOP - above) / (ties + 1), 0.0), 1.0)
    rr = float(np.mean(1.0 / (above + 1 + np.arange(ties + 1))))
    return hit, rr


# ---------------------------------------------------------------- the floors


class DistanceOnly:
    """The cells nearest the one that lit last."""

    name = "distance_only"
    kind = "baseline"

    def next_cell_scores(self, query: MigrationQuery, seed: int) -> FloatArray:
        la, lo = query.grid.cell_centres()
        last = query.lit_cell[-1:]
        return np.asarray(-km_matrix(la[last], lo[last], la, lo)[0], dtype=np.float64)


class QuakeHistory:
    """The cells with the most quakes of the scored size before the sequence."""

    name = "quake_history"
    kind = "baseline"

    def next_cell_scores(self, query: MigrationQuery, seed: int) -> FloatArray:
        ev = query.history.above(query.threshold)
        c = query.grid.cell_of(ev.lat, ev.lon)
        return np.bincount(c[c >= 0], minlength=query.grid.n_cells).astype(np.float64)


# ---------------------------------------------------------------- the task


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


class Migration(Task):
    task_id = "CB-24"
    name = "Migration"
    metric = "share of next lit cells in the method's top 3, mean over sequences"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        from craton.bench.world import SEQUENCES

        return {
            "sequences": ["amatrice_norcia 2016-08-24", "laquila 2009-04-06"]
            + [f"{s.key} {s.day}" for s in SEQUENCES],
            "cells": "0.1 deg; central Italy CB-02's grid, the others CB-22's boxes",
            "lights": f"a cell's first quake at or above completeness + 1 within {WINDOW_DAYS} "
            "days of the first big shock's UTC midnight; the first lit cell is given",
            "scored": f"the next {MAX_LIGHTS} lightings of each sequence (all, if fewer)",
            "inputs": "the box's quakes before the cut from completeness, and the cells lit so "
            "far with the quake that lit each; nothing else after the cut",
            "score": f"chance the true next cell is in the top {TOP} of the unlit cells, ties "
            "broken at random in expectation; mean over a sequence's lightings, then over "
            "sequences",
        }

    def view(self, ctx: BenchContext, case: MigrationCase, snapshot_id: str = "") -> Any:
        if case.region == ctx.region:
            return ctx
        from craton.bench.world import RegionView

        key = f"cb24-view:{case.key}"
        if key not in ctx._cache:
            g = case.grid
            ctx._cache[key] = RegionView(
                root=ctx.root,
                region=case.region,
                snapshot_id=snapshot_id,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=case.cut,
            )
        return ctx._cache[key]

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.planets import events_table

        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}}
        for case in cases(ctx.root):
            view = self.view(ctx, case)
            summary["snapshots"][case.region] = view.snapshot_id
            lit = lightings_of(view.events, case)
            files[f"{case.key}.parquet"] = _write_table(
                folder / f"{case.key}.parquet", events_table(lit.quakes)
            )
            days = (lit.quakes.time - to_seconds(case.cut)) / DAY
            n = min(len(lit) - 1, MAX_LIGHTS)
            summary[case.key] = {
                "lit": len(lit),
                "scored": n,
                "threshold": case.threshold,
                "days_to_last_scored": round(float(days[n]), 2) if n > 0 else None,
            }
        return {"files": files, "summary": summary}

    def supports(self, method: Any) -> bool:
        return isinstance(method, MigrationRanker)

    def frozen_snapshots(self, ctx: BenchContext) -> dict[str, str]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        out: dict[str, str] = task.get("summary", {}).get("snapshots", {})
        return out

    def lightings(self, ctx: BenchContext, case: MigrationCase) -> Lightings:
        from craton.bench.planets import read_events

        q = read_events(ctx.task_dir(self.task_id) / f"{case.key}.parquet")
        return Lightings(case.grid.cell_of(q.lat, q.lon).astype(np.int64), q)

    def score_case(
        self, method: Any, ctx: BenchContext, case: MigrationCase, seed: int
    ) -> tuple[float, float, float, int]:
        """(top-3 share, mean reciprocal rank, a coin's top-3 share, lightings scored)."""
        snap = self.frozen_snapshots(ctx).get(case.region, "")
        view = self.view(ctx, case, snap)
        if snap and view.snapshot_id != snap:
            raise ValueError(f"CB-24 was frozen on {case.region} snapshot {snap}")
        ev: Events = view.events
        history = ev.select(ev.time < to_seconds(case.cut)).above(case.mc)
        lit = self.lightings(ctx, case)
        n = min(len(lit) - 1, MAX_LIGHTS)
        hits, rrs, coin = [], [], []
        for k in range(1, n + 1):
            q = MigrationQuery(
                case=case.key,
                region=case.region,
                half=case.half,
                grid=case.grid,
                cut=case.cut,
                mc=case.mc,
                threshold=case.threshold,
                history=history,
                lit_cell=lit.cell[:k],
                lit=lit.quakes.select(np.arange(k)),
            )
            s = np.asarray(method.next_cell_scores(q, seed), dtype=np.float64)
            if s.shape != (case.grid.n_cells,):
                raise ValueError(f"{method.name} gave {s.shape} scores for {case.grid.n_cells}")
            unlit = q.unlit
            hit, rr = top_hit(s, unlit, int(lit.cell[k]))
            hits.append(hit)
            rrs.append(rr)
            coin.append(min(TOP / int(unlit.sum()), 1.0))
        return _mean(hits), _mean(rrs), _mean(coin), n

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        extras: dict[str, float] = {}
        per: dict[str, list[float]] = {h: [] for h in HALVES}
        for case in cases(ctx.root):
            hit, rr, coin, n = self.score_case(method, ctx, case, seed)
            extras[f"{case.key}.top3"] = hit
            extras[f"{case.key}.mrr"] = rr
            extras[f"{case.key}.coin"] = coin
            extras[f"{case.key}.scored"] = float(n)
            if n:
                per[case.half].append(hit)
        for half in HALVES:
            extras[f"{half}.top3"] = _mean(per[half])
        return Score(_mean(per[SEEN] + per[ZERO_SHOT]), extras)


register_task(Migration())
