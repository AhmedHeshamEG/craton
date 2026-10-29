"""CB-20 and CB-21: the Italian forecasts again, with a machine-learning catalogue as input.

A machine-learning catalogue (a neural network picking phases on every station) finds many
more quakes than the routine one, above all in the first hours after a big shock, when the
routine network is swamped. It is also built with hindsight: located with the whole sequence
in hand. So it gets tasks of its own, and CB-02 and CB-03 never see it.

What changes is only the **input**. Each task scores the same quakes, on the same days, as
the task it extends (CB-20 extends CB-02, CB-21 extends CB-03): INGV's, frozen. Every
forecast reads every quake before its day from the **union** of the two catalogues (D-094):

1. every INGV quake, as INGV has it;
2. plus each machine-learning quake INGV has no match for (none within ``MATCH_S`` seconds
   and ``MATCH_KM`` km), its magnitude moved onto INGV's scale.

The magnitude map is a quantile map fitted on the matched pairs whose INGV magnitude is below
``CALIBRATE_BELOW``, so the M3+ quakes CB-20 and CB-21 score never shape it. Every method sees
the same union (ETAS too), so CB-20 minus CB-02 for one method is what the better catalogue
is worth to it, and Craton minus ETAS on CB-20 is the same race as CB-02 on better data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from craton.bench.data import Events, FloatArray
from craton.bench.splits import Window
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    ForecastSpec,
    Score,
    SequenceForecast,
    forecast_grid,
    load_frozen,
    register_task,
)

#: Two quakes are the same quake if this close in time (s) and space (km).
MATCH_S = 3.0
MATCH_KM = 10.0
#: The magnitude map is fitted only on pairs below this INGV magnitude (the scored ones are not).
CALIBRATE_BELOW = 3.0
#: Quantiles of the map (0.5% to 99.5%).
MAP_POINTS = 100
KM_PER_DEG = 111.195


@dataclass(frozen=True)
class MagnitudeMap:
    """A monotone map from one catalogue's magnitudes onto another's, by quantiles;
    straight lines of slope 1 beyond the fitted range."""

    source: FloatArray
    target: FloatArray

    def __call__(self, m: FloatArray) -> FloatArray:
        x, y = self.source, self.target
        out = np.interp(m, x, y)
        out = np.where(m > x[-1], y[-1] + (m - x[-1]), out)
        return np.asarray(np.where(m < x[0], y[0] + (m - x[0]), out), dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        """The map at a few magnitudes, for the summary."""
        at = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return {f"{m:g}": round(float(v), 3) for m, v in zip(at, self(at), strict=True)}


def match(
    base: Events, extra: Events
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.bool_]]]:
    """For each ``extra`` quake, the index of its closest-in-time ``base`` match within
    ``MATCH_S`` and ``MATCH_KM`` (-1 when none). Both are sorted by time."""
    lo = np.searchsorted(base.time, extra.time - MATCH_S)
    hi = np.searchsorted(base.time, extra.time + MATCH_S, side="right")
    idx = np.full(len(extra), -1, dtype=np.int64)
    width = int((hi - lo).max()) if len(extra) else 0
    best = np.full(len(extra), np.inf)
    coslat = np.cos(np.radians(extra.lat))
    for k in range(width):
        j = lo + k
        ok = j < hi
        jj = np.where(ok, j, 0)
        km = KM_PER_DEG * np.hypot(base.lat[jj] - extra.lat, (base.lon[jj] - extra.lon) * coslat)
        dt = np.abs(base.time[jj] - extra.time)
        better = ok & (km < MATCH_KM) & (dt < best)
        idx = np.where(better, jj, idx)
        best = np.where(better, dt, best)
    return idx, idx >= 0


def fit_map(base: Events, extra: Events, idx: np.ndarray[Any, np.dtype[np.int64]]) -> MagnitudeMap:
    """Quantile map of ``extra``'s magnitudes onto ``base``'s, from pairs below
    ``CALIBRATE_BELOW`` on the base scale."""
    hit = idx >= 0
    src, dst = extra.mag[hit], base.mag[idx[hit]]
    low = dst < CALIBRATE_BELOW
    if low.sum() < MAP_POINTS:
        raise ValueError(f"only {int(low.sum())} matched quakes to map magnitudes on")
    q = np.linspace(0.005, 0.995, MAP_POINTS)
    return MagnitudeMap(np.quantile(src[low], q), np.quantile(dst[low], q))


@dataclass(frozen=True)
class Union:
    """The merged catalogue and how it was made."""

    events: Events
    added: Events
    map: MagnitudeMap
    matched: int
    overlay: int


def union(base: Events, extra: Events) -> Union:
    """Every ``base`` quake plus each ``extra`` quake it has no match for, magnitudes mapped."""
    idx, hit = match(base, extra)
    mmap = fit_map(base, extra, idx)
    new = extra.select(~hit)
    added = Events(
        new.time,
        new.lat,
        new.lon,
        new.depth_km,
        mmap(new.mag),
        new.mag_type,
        new.token_hash,
    )
    return Union(Events.concat([base, added]), added, mmap, int(hit.sum()), len(extra))


@dataclass
class EnhancedContext(BenchContext):
    """Central Italy's context, whose ``events`` (the forecast history) are the union with a
    machine-learning catalogue. Splits, training data and the scored answers are untouched."""

    overlay: str = ""
    overlay_snapshot: str = ""

    @cached_property
    def built(self) -> Union:
        ml = BenchContext(self.root, region=self.overlay, snapshot_id=self.overlay_snapshot)
        base = BenchContext.events.fget(self)  # type: ignore[attr-defined]
        return union(base, ml.events)

    @property
    def events(self) -> Events:
        return self.built.events


class EnhancedForecast(SequenceForecast):
    """A daily Italian forecast task scored like ``base`` but with a richer input catalogue."""

    def __init__(
        self,
        task_id: str,
        name: str,
        base: str,
        window: Window,
        fit_split: str,
        overlay: str,
    ) -> None:
        super().__init__(task_id, name, window, fit_split)
        self.base = base
        self.overlay = overlay
        self.metric = (
            "log-likelihood per earthquake (bits), M≥3, daily 0.1° forecasts, "
            "INGV + machine-learning catalogue as input"
        )

    @property
    def spec(self) -> ForecastSpec:
        # Training reads only the split, which the overlay never touches: the base task's
        # checkpoints are this task's too.
        return ForecastSpec(
            self.task_id, self.fit_split, self.window, forecast_grid(), train_task=self.base
        )

    def definition(self) -> dict[str, Any]:
        return {
            **super().definition(),
            "extends": self.base,
            "input": f"INGV + {self.overlay} (union, D-094)",
            "match": {"seconds": MATCH_S, "km": MATCH_KM},
            "magnitude_map": f"quantiles of matched pairs below INGV M{CALIBRATE_BELOW:g}",
        }

    def overlay_snapshot(self, ctx: BenchContext) -> str:
        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        return str(task.get("summary", {}).get("overlay_snapshot", ""))

    def context(self, ctx: BenchContext) -> EnhancedContext:
        key = f"enhanced:{self.overlay}"
        if key not in ctx._cache:
            ctx._cache[key] = EnhancedContext(
                ctx.root,
                region=ctx.region,
                snapshot_id=ctx.snapshot_id,
                overlay=self.overlay,
                overlay_snapshot=self.overlay_snapshot(ctx),
            )
        out: EnhancedContext = ctx._cache[key]
        return out

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        info = super().prepare(ctx)
        view = self.context(ctx)
        u = view.built
        w = u.added.window(self.window.t0, self.window.t1)
        info["summary"].update(
            {
                "overlay": self.overlay,
                "overlay_snapshot": view.overlay_snapshot
                or BenchContext(ctx.root, region=self.overlay).snapshot_id,
                "overlay_quakes": u.overlay,
                "matched": u.matched,
                "added": len(u.added),
                "added_M>=2_in_window": int((w.mag >= 2.0 - 1e-9).sum()),
                "magnitude_map": u.map.as_dict(),
            }
        )
        return info

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        return super().score(method, self.context(ctx), seed)


def _window(task_id: str) -> Window:
    task = TASKS[task_id]
    assert isinstance(task, SequenceForecast)
    return task.window


register_task(
    EnhancedForecast(
        "CB-20",
        "Apennines, enhanced inputs",
        "CB-02",
        _window("CB-02"),
        "cb02_fit",
        "central_italy_ml_2016",
    )
)
register_task(
    EnhancedForecast(
        "CB-21",
        "L'Aquila, enhanced inputs",
        "CB-03",
        _window("CB-03"),
        "cb03_fit",
        "laquila_ml_2009",
    )
)
