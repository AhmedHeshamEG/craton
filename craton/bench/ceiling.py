"""The building: how far above ETAS a forecast could possibly climb on CB-02 and CB-03.

Each floor is an oracle: ETAS's own forecast with one thing replaced by the truth, read
from the very quakes it is scored on. No method can use them; they measure the room.

    count        each day's total is the number that really happened (ETAS's where and size)
    where_10km   each day's map is where that day's quakes really were, blurred to ~10 km
                 (a Gaussian of one 0.1° cell), with 10% left on ETAS's map
    where_cell   the same, to the exact cell, with 1% left on ETAS's map
    roof         count and where_cell together: knowing how many and exactly where

Every number is bits per M3+ quake above ETAS, so floor k of the building is a forecast that
gives the real quakes 2^k times ETAS's probability. ``craton bench ceiling`` writes
``results/ceiling.json``; the scoreboard draws it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, BenchContext, ForecastSpec, GriddedForecast, Score

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
LN2 = math.log(2.0)
THRESHOLD = 3.0
ORACLES = ("count", "where_10km", "where_cell", "roof")


def _observed_cells(obs: Events, spec: ForecastSpec) -> tuple[Any, Any, Any]:
    g = spec.grid
    first = int(g.mag_bin_of(np.array([THRESHOLD]))[0])
    sel = g.mag_bin_of(obs.mag) >= first
    day = np.floor((obs.time[sel] - spec.window.t0) / DAY).astype(np.int64)
    cell = g.cell_of(obs.lat[sel], obs.lon[sel])
    return day, cell, first


def _truth_map(day: Any, cell: Any, n_days: int, spec: ForecastSpec, sigma: float) -> FloatArray:
    """(days, cells) share of each day's quakes per cell, blurred by ``sigma`` cells."""
    g = spec.grid
    img = np.zeros((n_days, g.n_lat, g.n_lon))
    np.add.at(img, (day, cell // g.n_lon, cell % g.n_lon), 1.0)
    if sigma > 0:
        img = np.stack([gaussian_filter(x, sigma, mode="constant") for x in img])
    flat = img.reshape(n_days, -1)
    total = flat.sum(axis=1, keepdims=True)
    return np.asarray(flat / np.where(total > 0, total, 1.0), dtype=np.float64)


def oracle_gains(fc: GriddedForecast, obs: Events, spec: ForecastSpec) -> dict[str, float]:
    """Bits per M3+ quake each oracle adds on top of the forecast ``fc``."""
    day, cell, first = _observed_cells(obs, spec)
    n = len(day)
    expected = fc.expected(first)
    n_days = expected.shape[0]
    rate = expected.sum(axis=1)
    seen = np.bincount(day, minlength=n_days).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        truth_ll = np.where(seen > 0, seen * np.log(seen), 0.0) - seen
        count = float(np.sum(truth_ll - (seen * np.log(rate) - rate)))
    share = expected / rate[:, None]
    out = {"count": count / n / LN2}
    for name, sigma, keep in (("where_10km", 1.0, 0.10), ("where_cell", 0.0, 0.01)):
        truth = _truth_map(day, cell, n_days, spec, sigma)
        mixed = (1 - keep) * truth + keep * share
        out[name] = float(np.mean(np.log(mixed[day, cell] / share[day, cell]))) / LN2
    out["roof"] = out["count"] + out["where_cell"]
    return out


def building(ctx: BenchContext, etas_scores: dict[str, Score], seed: int = 0) -> dict[str, Any]:
    """The ceiling record for CB-02 and CB-03, over ETAS's forecast with ``seed``."""
    from craton.bench.baselines import Etas

    out: dict[str, Any] = {}
    for task_id, score in etas_scores.items():
        task = TASKS[task_id]
        spec = task.spec  # type: ignore[attr-defined]
        fc = Etas().forecast(spec, ctx, seed)
        gains = oracle_gains(fc, task.observed(ctx), spec)  # type: ignore[attr-defined]
        out[task_id] = {
            "etas": score.value,
            "above_etas_bits": gains,
            "absolute": {k: score.value + v for k, v in gains.items()},
        }
    return out


def write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8", newline="\n")


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    out: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return out
