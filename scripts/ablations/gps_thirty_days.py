"""Does GPS pay at 30 days? CB-19's Field with three GNSS features added to its memory (D-085).

For every station and midnight: how far it moved over the last 30 days beyond its own one-year
trend, from 5-day medians at each end so one bad day can't fake it. Per forecast cell, a Gaussian
(25 km, cut at 75 km) average over the stations near it gives three features: log1p of the
horizontal departure (mm), the vertical departure (÷10, clipped at ±50 mm) and log1p of the
station weight (how well the cell is covered). CB-04's held-out stations are left out.

Everything else is the Field CB-19 trains (`configs/field/mini.yaml`); the checkpoints go next to
its own with a `-gnss` suffix. About 90 minutes per seed on the RTX 2060; two seeds in parallel fit.

    uv run --extra cu126 python scripts/ablations/gps_thirty_days.py 0
"""

from __future__ import annotations

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


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    """The median of each day and the w − 1 before it, ignoring missing days."""
    pad = np.full((w - 1, x.shape[1]), np.nan)
    v = sliding_window_view(np.concatenate([pad, x]), w, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-missing windows stay NaN
        return np.nanmedian(v, axis=-1)


def gnss_features(gnss: Gnss, grid: Grid, midnights: np.ndarray) -> np.ndarray:
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

    def folder(
        self,
        ctx: BenchContext,
        spec: ForecastSpec,
        seed: int,
        config: field.FieldConfig | None = None,
    ) -> Path:
        p = super().folder(ctx, spec, seed, config)
        return p.with_name(p.name + "-gnss")


def main() -> None:
    seed = int(sys.argv[1])
    ctx = BenchContext(ROOT)
    gnss = ctx.gnss
    gnss = gnss.select(~np.isin(gnss.station, list(ctx.heldout)))
    plain = field.memory

    def memory(events: Events, grid: Grid, starts: np.ndarray, spec: MemorySpec) -> np.ndarray:
        return np.concatenate(
            [plain(events, grid, starts, spec), gnss_features(gnss, grid, starts)], -1
        )

    field.memory = memory
    ff = GnssField("mini", log=lambda s: print(s, flush=True))

    class Method:
        name, long_periods = "craton-mini-gnss", True

        def forecast(self, spec: ForecastSpec, c: BenchContext, sd: int) -> field.GriddedForecast:
            return ff.forecast(spec, c, sd)

    t0 = time.time()
    score = TASKS["CB-19"].score(Method(), ctx, seed)
    e = score.extras
    print(
        f"CB-19 seed {seed} with GPS: {score.value:.3f} bits per M3+ quake "
        f"(how many {e['M3.how_many_bits']:.3f}, where {e['M3.where_bits']:.3f}), "
        f"{(time.time() - t0) / 60:.0f} min"
    )


if __name__ == "__main__":
    main()
