"""Does GPS pay at 30 days? CB-19's Field with three GNSS features added to its memory (D-085).

For every station and midnight: how far it moved over the last 30 days beyond its own one-year
trend, from 5-day medians at each end so one bad day can't fake it. Per forecast cell, a Gaussian
(25 km, cut at 75 km) average over the stations near it gives three features: log1p of the
horizontal departure (mm), the vertical departure (÷10, clipped at ±50 mm) and log1p of the
station weight (how well the cell is covered). CB-04's held-out stations are left out.

Everything else is the Field CB-19 trains (`configs/field/mini.yaml`); the checkpoints go next to
its own with a `-gnss` suffix. About 90 minutes per seed on the RTX 2060; two seeds in parallel fit.

    uv run --extra cu126 python scripts/ablations/gps_thirty_days.py 0
    uv run --extra cu126 python scripts/ablations/gps_thirty_days.py CB-02 --no-jumps 0

P8f: any daily or 30-day task (CB-02, CB-03, CB-19), and ``--no-jumps``: the same features
from positions whose day-to-day steps above ``JUMP_MM`` are removed (coseismic offsets).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import craton.field.forecaster as field
from craton.bench.csep import Grid
from craton.bench.data import DAY, Events, Gnss
from craton.bench.tasks import TASKS, BenchContext, ForecastSpec
from craton.field.memory import MemorySpec

ROOT = Path(__file__).resolve().parents[2]
SHORT, LONG, SMOOTH, SIGMA_KM = 30, 365, 5, 25.0
#: P8f, jumps removed: a day-to-day step of the smoothed position bigger than this (mm, any
#: component) is taken as a quake's coseismic offset (or an antenna change) and zeroed, so what
#: is left is slow motion only: trend departures, afterslip, transients.
JUMP_MM = 4.0


def without_jumps(pos: np.ndarray) -> np.ndarray:
    """(days, stations, 3) positions with every step above ``JUMP_MM`` taken out."""
    step = np.diff(pos, axis=0)
    big = np.abs(step) > JUMP_MM
    step = np.where(big | ~np.isfinite(step), 0.0, step)
    out = np.concatenate([np.zeros_like(pos[:1]), np.cumsum(step, axis=0)], axis=0)
    first = np.where(np.isfinite(pos), pos, np.nan)
    start = np.nanmedian(first[: max(1, LONG)], axis=0)  # keeps the level; only steps matter
    return np.where(np.isfinite(pos), out + np.nan_to_num(start), np.nan)


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """The median of each day and the w − 1 before it, ignoring missing days."""
    pad = np.full((w - 1, x.shape[1]), np.nan)
    v = sliding_window_view(np.concatenate([pad, x]), w, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-missing windows stay NaN
        return np.nanmedian(v, axis=-1)


def gnss_features(gnss: Gnss, grid: Grid, midnights: np.ndarray, jumps: bool = True) -> np.ndarray:
    """(midnights, cells, 3) float16: the 30-day departure per cell, as the docstring says."""
    first = float(midnights.min()) - (LONG + SMOOTH + 2) * DAY
    n_days = round((float(midnights.max()) - first) / DAY) + 1
    stations = gnss.stations
    lat, lon = np.zeros(len(stations)), np.zeros(len(stations))
    pos = np.full((n_days, len(stations), 3), np.nan)
    for j, s in enumerate(stations):
        sel = gnss.station == s
        d = np.floor((gnss.time[sel] - first) / DAY).astype(np.int64)
        ok = (d >= 0) & (d < n_days)
        pos[d[ok], j] = gnss.enu[sel][ok] * 1000.0
        lat[j], lon[j] = np.median(gnss.lat[sel]), np.median(gnss.lon[sel])
    med = np.stack([_rolling_median(pos[..., k], SMOOTH) for k in range(3)], axis=-1)
    if not jumps:
        med = without_jumps(med)
    # The last whole day before each midnight.
    i = np.round((midnights - first) / DAY).astype(np.int64) - 1
    now, short, long_ = med[i], med[i - SHORT], med[i - LONG]
    dep = (now - short) - (now - long_) * SHORT / LONG
    h = np.log1p(np.hypot(dep[..., 0], dep[..., 1]))
    up = np.clip(dep[..., 2], -50, 50) / 10.0
    have = np.isfinite(h) & np.isfinite(up)
    clat, clon = grid.cell_centres()
    kx = 111.2 * np.cos(np.radians(np.mean(clat)))
    d2 = ((clat[:, None] - lat[None]) * 111.2) ** 2 + ((clon[:, None] - lon[None]) * kx) ** 2
    w = np.exp(-0.5 * d2 / SIGMA_KM**2)  # (cells, stations)
    w[d2 > (3 * SIGMA_KM) ** 2] = 0.0
    out = np.zeros((len(midnights), grid.n_cells, 3), np.float32)
    for t in range(len(midnights)):
        ww = w * have[t][None]
        tot = ww.sum(axis=1)
        safe = np.maximum(tot, 1e-9)
        out[t, :, 0] = (ww @ np.nan_to_num(h[t])) / safe
        out[t, :, 1] = (ww @ np.nan_to_num(up[t])) / safe
        out[t, :, 2] = np.log1p(tot)
    return out.astype(np.float16)


class GnssField(field.FieldForecaster):
    """The Field, its checkpoints kept apart from the real ones."""

    suffix: str = "-gnss"

    def folder(
        self,
        ctx: BenchContext,
        spec: ForecastSpec,
        seed: int,
        config: field.FieldConfig | None = None,
    ) -> Path:
        p = super().folder(ctx, spec, seed, config)
        return p.with_name(p.name + self.suffix)


def main(task_id: str, seed: int, jumps: bool) -> None:
    ctx = BenchContext(ROOT)
    gnss = ctx.gnss
    gnss = gnss.select(~np.isin(gnss.station, list(ctx.heldout)))
    plain = field.memory

    def memory(events: Events, grid: Grid, starts: np.ndarray, spec: MemorySpec) -> np.ndarray:
        return np.concatenate(
            [plain(events, grid, starts, spec), gnss_features(gnss, grid, starts, jumps)], -1
        )

    field.memory = memory
    ff = GnssField("mini", log=lambda s: print(s, flush=True))
    ff.suffix = "-gnss" if jumps else "-gnss-nojumps"

    class Method:
        name, long_periods = "craton-mini-gnss", True

        def forecast(self, spec: ForecastSpec, c: BenchContext, sd: int) -> field.GriddedForecast:
            return ff.forecast(spec, c, sd)

    t0 = time.time()
    score = TASKS[task_id].score(Method(), ctx, seed)
    e = score.extras
    kind = "GPS" if jumps else "GPS without jumps"
    print(
        f"{task_id} seed {seed} with {kind}: {score.value:.3f} bits per M3+ quake "
        f"(how many {e['M3.how_many_bits']:.3f}, where {e['M3.where_bits']:.3f}), "
        f"{(time.time() - t0) / 60:.0f} min"
    )
    path = ROOT / "results" / "ablation" / "gps_thirty_days.json"
    out = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    out.setdefault(task_id, {}).setdefault("with_jumps" if jumps else "without_jumps", {})[
        str(seed)
    ] = {
        "bits": score.value,
        "how_many": e["M3.how_many_bits"],
        "where": e["M3.where_bits"],
    }
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    jumps = "--no-jumps" not in args
    args = [a for a in args if a != "--no-jumps"]
    task = args[0] if args[0].startswith("CB-") else "CB-19"
    main(task, int(args[-1]), jumps)
