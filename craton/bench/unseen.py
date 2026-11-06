"""CB-32, unseen regions: each training region forecast by a model that never read it.

CB-22's zero-shot half asks a world model about Turkey, Chile and Alaska, which it never
read. Three regions, all on the global network's terms, is a small sample and nothing to
compare against: the model also never had a chance to *fail* on a region it knows. CB-32
turns every region the world Field trains on into a test, one at a time (leave-one-region-
out, the cross-validation of transfer):

- **folds**: central Italy, California, Japan, New Zealand. For each, the method's model is
  trained again on the other three only, with the same cut (2007-01-01), and asked about the
  region it left out;
- **sequences**: central Italy's two (CB-02's 2016-17 window and CB-03's 2009 one, each on
  CB-02's grid and split), and CB-22's six seen-region sequences (two per region, the same
  boxes, windows and frozen quakes, copied here so this task stands on its own);
- **reference**: ETAS fitted on the box's own past (CB-02's floor in Italy, CB-22's reference
  elsewhere), so ETAS scores 0 and anything below 0 loses to the best simple model a
  seismologist there would build;
- **primary**: the mean over the four folds of each fold's mean gain, in bits per quake from
  completeness + 1 (central Italy: M3).

A method that does not train on regions (ETAS, Poisson, the other floors) answers every fold
with the same forecast it always gives. A model must answer through ``unseen_forecaster``
(the region it may not have read), or it is not scored: a model trained on every region would
be answering about places it has already seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from craton.adapters.region import load_region
from craton.bench.data import Events
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    ForecastSpec,
    Score,
    Task,
    _write_table,
    register_task,
    score_forecast,
)
from craton.bench.world import (
    MIN_FIT_QUAKES,
    SEEN,
    SEQUENCES,
    RegionView,
    observed,
    sequence_spec,
    view_for,
    view_t,
)
from craton.spec.timeaxis import to_seconds

#: The regions the world Field trains on, each left out in turn.
FOLDS = ("central_italy", "california", "japan", "new_zealand")
#: Central Italy's sequences are CB-02's and CB-03's windows on CB-02's grid.
ITALY_TASKS = ("CB-02", "CB-03")


@dataclass(frozen=True)
class Item:
    """One scored sequence: its key, its fold, and the CB-22 sequence or Italian task it is."""

    key: str
    region: str
    source: str  # "CB-22:<key>" or the Italian task's ID


def items() -> tuple[Item, ...]:
    out = [Item(t.lower().replace("-", ""), "central_italy", t) for t in ITALY_TASKS]
    out += [
        Item(s.key, s.region, f"CB-22:{s.key}")
        for s in SEQUENCES
        if s.half == SEEN and s.region in FOLDS
    ]
    return tuple(out)


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


class UnseenRegions(Task):
    task_id = "CB-32"
    name = "Unseen-region forecast"
    metric = "bits per quake above ETAS fitted there, by a model that never read the region"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        return {
            "folds": list(FOLDS),
            "sequences": [it.__dict__ for it in items()],
            "protocol": "leave one region out: the model is trained on the other folds' "
            "regions only (same cut as the world Field, 2007-01-01) and forecasts every "
            "sequence of the region it left out; methods that do not train on regions answer "
            "as they always do",
            "reference": "ETAS fitted on the box's own past: CB-02's etas floor for central "
            "Italy (its own fit split per window), CB-22's reference elsewhere",
            "scoring": "as CB-02 and CB-22 (score_forecast); bits per quake from the region's "
            "completeness + 1 (central Italy M3)",
            "primary": "mean over folds of the fold's mean gain over the reference",
            "qualifies": f"a CB-22 sequence counts only if its reference ETAS fits on at least "
            f"{MIN_FIT_QUAKES} quakes (CB-22's rule)",
        }

    # ------------------------------------------------------------------ the questions

    def _italy(self, ctx: BenchContext, task_id: str) -> tuple[ForecastSpec, BenchContext]:
        from craton.bench.tasks import TASKS

        task: Any = TASKS[task_id]
        return task.spec, ctx

    def _world(
        self, ctx: BenchContext, key: str, snapshot: str = ""
    ) -> tuple[ForecastSpec, RegionView]:
        seq = next(s for s in SEQUENCES if s.key == key)
        spec = sequence_spec(seq, load_region(seq.region, ctx.root))
        cache = f"cb32-view:{key}"
        if cache not in ctx._cache:
            ctx._cache[cache] = view_for(ctx.root, seq.region, spec, snapshot)
        view: RegionView = ctx._cache[cache]
        return spec, view

    def question(
        self, ctx: BenchContext, it: Item, snapshots: dict[str, str] | None = None
    ) -> tuple[ForecastSpec, BenchContext]:
        if it.source in ITALY_TASKS:
            return self._italy(ctx, it.source)
        return self._world(ctx, it.key, (snapshots or {}).get(it.region, ""))

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.planets import events_table

        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}}
        for it in items():
            spec, view = self.question(ctx, it)
            if isinstance(view, RegionView):
                summary["snapshots"][it.region] = view.snapshot_id
                obs = observed(view, spec)
                hist = view.train_events("world").above(spec.mc)
                n_fit = int(
                    ((hist.time >= to_seconds(spec.fit_from)) & (hist.time < view_t(spec))).sum()
                )
            else:
                g = spec.grid
                obs = view.events.window(spec.window.t0, spec.window.t1)
                obs = obs.select((g.cell_of(obs.lat, obs.lon) >= 0) & (g.mag_bin_of(obs.mag) >= 0))
                n_fit = len(view.train_events(spec.fit_split).above(spec.mc))
            files[f"{it.key}.parquet"] = _write_table(
                folder / f"{it.key}.parquet", events_table(obs)
            )
            summary[it.key] = {
                "region": it.region,
                "source": it.source,
                "window": spec.window.as_dict(),
                "mc": spec.mc,
                f"M>={spec.scored:g}": int((obs.mag >= spec.scored - 1e-9).sum()),
                "largest": float(obs.mag.max()) if len(obs) else None,
                "fit_quakes": n_fit,
            }
        return {"files": files, "summary": summary}

    # ------------------------------------------------------------------ scoring

    def supports(self, method: Any) -> bool:
        if not isinstance(method, Forecaster) or getattr(method, "regional_only", False):
            return False
        if getattr(method, "kind", "") == "model":
            return callable(getattr(method, "unseen_forecaster", None))
        return True

    def frozen_snapshots(self, ctx: BenchContext) -> dict[str, str]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        out: dict[str, str] = task.get("summary", {}).get("snapshots", {})
        return out

    def observed_of(self, ctx: BenchContext, it: Item) -> Events:
        from craton.bench.planets import read_events

        return read_events(ctx.task_dir(self.task_id) / f"{it.key}.parquet")

    def forecaster_for(self, method: Any, region: str) -> Any:
        unseen = getattr(method, "unseen_forecaster", None)
        return unseen(region) if callable(unseen) else method

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        from craton.bench.baselines import Etas
        from craton.bench.world import _check

        snaps = self.frozen_snapshots(ctx)
        extras: dict[str, float] = {}
        gains: dict[str, list[float]] = {}
        for it in items():
            spec, view = self.question(ctx, it, snaps)
            obs = self.observed_of(ctx, it)
            fc = self.forecaster_for(method, it.region).forecast(spec, view, seed)
            _check(fc, spec)
            sc = score_forecast(fc, obs, spec, seed)
            ref_key = f"cb32-etas:{it.key}:{seed}"
            if ref_key not in ctx._cache:
                ctx._cache[ref_key] = score_forecast(
                    Etas().forecast(spec, view, seed), obs, spec, seed
                ).value
            gain = sc.value - float(ctx._cache[ref_key])
            tag = f"M{spec.scored:g}"
            extras[f"{it.key}.bits"] = sc.value
            extras[f"{it.key}.gain"] = gain
            extras[f"{it.key}.observed"] = sc.extras[f"{tag}.observed"]
            extras[f"{it.key}.expected"] = sc.extras[f"{tag}.expected"]
            for part in ("how_many_bits", "where_bits", "how_big_bits"):
                extras[f"{it.key}.{part}"] = sc.extras.get(f"{tag}.{part}", math.nan)
            if it.source in ITALY_TASKS:
                ok = True
            else:
                ok = Etas().fit(spec, view, seed).n_targets >= MIN_FIT_QUAKES
            extras[f"{it.key}.qualified"] = float(ok)
            if ok:
                gains.setdefault(it.region, []).append(gain)
        per_fold = {r: _mean(gains.get(r, [])) for r in FOLDS}
        for r, g in per_fold.items():
            extras[f"{r}.mean_gain"] = g
        extras["folds_beaten"] = _mean([float(g > 0) for g in per_fold.values()])
        every = [g for v in gains.values() for g in v]
        extras["sequences_beaten"] = _mean([float(g > 0) for g in every])
        extras["mean_gain_per_sequence"] = _mean(every)
        return Score(_mean(list(per_fold.values())), extras)


register_task(UnseenRegions())
