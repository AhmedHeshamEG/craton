"""Listening in the Field (P10c): the noise instruments as inputs to every forecast cell.

Two instruments report every day, quake or no quake (P10a, P10b):

- ``seismic_noise_power``: each station's daily band power in three bands (dB). The Field
  reads how loud a station was on its last whole day **over its own median of the
  ``baseline_days`` before**, so a quiet station and a noisy one speak the same unit, and a
  channel's fixed gain offset cancels.
- ``seismic_velocity_change``: dv/v of each station pair, the 10- and 30-day stacks.

At each midnight a cell reads the stations (and pair midpoints) near it, through a Gaussian of
``sigma_km`` cut at ``cut_km``, weighted by how well each value is known. Seven numbers come
out per cell: three loudness changes and how many stations it hears, two dv/v and how many
pairs it hears. They are appended to the Omori memory, so the Field's own input layer is the
encoder: nothing in the trunk changes.

What a midnight may read: a token's time is the midnight starting the (last) day it covers, so
at midnight ``M`` only tokens stamped ``M − 1 day`` or earlier are whole. Exactly one day is
read (``M − 1 day``); a station that was silent that day is missing, never carried forward.
Before the instruments exist (central Italy before mid-2015) every cell reads zeros and
"hears nothing", which the Field learns to ignore.

The dv/v of a region is read only if its reference period ends at or before the training cut:
its values are stretches against a reference waveform, and a reference that runs into the test
era carries test-era noise. ``laquila_noise``'s runs to April 2009, inside CB-03's test year,
so CB-03 hears loudness only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from craton.bench.csep import Grid
from craton.bench.data import DAY, read_table
from craton.field.config import FieldListen

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

NOISE = "seismic_noise_power"
VELOCITY = "seismic_velocity_change"
COLUMNS = ("lat", "lon", "time", "value", "uncertainty", "qualifiers")
STACKS = ("10", "30")
#: A station's median needs this many of its ``baseline_days`` before it is read.
MIN_BASELINE_SHARE = 1 / 3
#: Feature scales: loudness change in units of 10 dB, dv/v (a fraction) in units of 0.2%.
DB_SCALE, DB_CLIP = 10.0, 3.0
DVV_SCALE, DVV_CLIP = 0.002, 3.0
N_FEATURES = 7
NAMES = (
    "noise_low", "noise_mid", "noise_high", "noise_heard",
    "dvv_10", "dvv_30", "dvv_heard",
)  # fmt: skip


@dataclass(frozen=True)
class Series:
    """Values of one instrument on one day grid: (sources, days, channels), NaN where missing."""

    lat: FloatArray
    lon: FloatArray
    days: FloatArray  # midnights, seconds since 1970
    value: FloatArray
    sigma: FloatArray


def _day_grid(times: list[float]) -> FloatArray:
    t = np.asarray(times, dtype=np.float64)
    return np.arange(t.min(), t.max() + DAY / 2, DAY)


def noise_series(table: Any, baseline_days: int) -> Series:
    """Per station (components averaged), the three band powers on its last day over the median
    of its ``baseline_days`` before it (dB), and their uncertainty."""
    cols = {c: table.column(c).to_pylist() for c in COLUMNS}
    stations: dict[str, tuple[float, float]] = {}
    days = _day_grid(cols["time"])
    rows: list[tuple[str, int, list[float], list[float]]] = []
    for lat, lon, t, v, u, q in zip(*(cols[c] for c in COLUMNS), strict=True):
        sta = json.loads(q)["station"]
        stations.setdefault(sta, (lat, lon))
        rows.append((sta, round((t - days[0]) / DAY), v, u))
    names = sorted(stations)
    at = {s: i for i, s in enumerate(names)}
    total = np.zeros((len(names), len(days), 3))
    var = np.zeros_like(total)
    n = np.zeros((len(names), len(days), 1))
    for sta, d, v, u in rows:
        total[at[sta], d] += v
        var[at[sta], d] += np.square(u)
        n[at[sta], d] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.where(n > 0, total / n, np.nan)
        sigma = np.where(n > 0, np.sqrt(var) / n, np.nan)
    base = _trailing_median(level, baseline_days)
    lat = np.array([stations[s][0] for s in names])
    lon = np.array([stations[s][1] for s in names])
    return Series(lat, lon, days, level - base, sigma)


def _trailing_median(x: FloatArray, w: int) -> FloatArray:
    """(S, D, C): each day's median of the ``w`` days before it (not itself), NaN when fewer
    than ``MIN_BASELINE_SHARE`` of them hold a value."""
    from numpy.lib.stride_tricks import sliding_window_view

    s, d, c = x.shape
    pad = np.concatenate([np.full((s, w, c), np.nan), x[:, :-1]], axis=1)  # day i sees i-w..i-1
    win = sliding_window_view(pad, w, axis=1)[:, :d]  # (S, D, C, w)
    enough = np.isfinite(win).sum(-1) >= max(1, math.ceil(w * MIN_BASELINE_SHARE))
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=-1)
    out: FloatArray = np.where(enough, med, np.nan)
    return out


def velocity_series(table: Any) -> tuple[Series, float]:
    """Per pair, the corrected dv/v of the 10- and 30-day stacks (a fraction), and the latest end
    of any pair's reference period (seconds)."""
    cols = {c: table.column(c).to_pylist() for c in COLUMNS}
    days = _day_grid(cols["time"])
    where: dict[tuple[str, str], tuple[float, float]] = {}
    rows: list[tuple[tuple[str, str], int, int, float, float]] = []
    ref_end = -math.inf
    for lat, lon, t, v, u, q in zip(*(cols[c] for c in COLUMNS), strict=True):
        qual = json.loads(q)
        if qual["stack_days"] not in STACKS:
            continue
        key = (qual["station_a"], qual["station_b"])
        where.setdefault(key, (lat, lon))
        rows.append((key, round((t - days[0]) / DAY), STACKS.index(qual["stack_days"]), v[0], u[0]))
        end = qual["reference"].split("/")[1]
        ref_end = max(ref_end, float(np.datetime64(end + "T00:00:00", "s").astype(np.int64)))
    keys = sorted(where)
    at = {k: i for i, k in enumerate(keys)}
    value = np.full((len(keys), len(days), len(STACKS)), np.nan)
    sigma = np.full_like(value, np.nan)
    for key, d, c, v, u in rows:
        value[at[key], d, c] = v
        sigma[at[key], d, c] = u
    lat = np.array([where[k][0] for k in keys])
    lon = np.array([where[k][1] for k in keys])
    return Series(lat, lon, days, value, sigma), ref_end


def _cell_weights(series: Series, grid: Grid, spec: FieldListen) -> FloatArray:
    """(cells, sources) Gaussian closeness, zero beyond ``cut_km``."""
    from craton.bench.velocity import km

    lat, lon = grid.cell_centres()
    w = np.zeros((grid.n_cells, len(series.lat)))
    for j in range(len(series.lat)):
        d = km(float(series.lat[j]), float(series.lon[j]), np.asarray(lat), np.asarray(lon))
        w[:, j] = np.where(d <= spec.cut_km, np.exp(-0.5 * (d / spec.sigma_km) ** 2), 0.0)
    return w


def _spread(
    series: Series, starts: FloatArray, grid: Grid, spec: FieldListen, scale: float, clip: float
) -> FloatArray:
    """(days, cells, channels + 1): each midnight's last whole day, averaged over the sources
    near each cell (Gaussian × 1/σ²), and log1p of the Gaussian weight of the sources heard."""
    n_ch = series.value.shape[-1]
    out = np.zeros((len(starts), grid.n_cells, n_ch + 1), np.float32)
    idx = np.round((starts - DAY - series.days[0]) / DAY).astype(np.int64)
    ok = (idx >= 0) & (idx < len(series.days))
    if not ok.any():
        return out
    w = _cell_weights(series, grid, spec)  # (cells, S)
    v = series.value[:, idx[ok]]  # (S, days, C)
    s = series.sigma[:, idx[ok]]
    good = np.isfinite(v) & np.isfinite(s) & (s > 0)
    inv = np.where(good, 1.0 / np.square(np.where(good, s, 1.0)), 0.0)
    num = np.einsum("cs,sdk->dck", w, np.where(good, v, 0.0) * inv)
    den = np.einsum("cs,sdk->dck", w, inv)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, 0.0)
    heard = np.einsum("cs,sd->dc", w, good.any(-1).astype(np.float64))
    out[ok, :, :n_ch] = np.clip(mean / scale, -clip, clip)
    out[ok, :, n_ch] = np.log1p(heard)
    return out


@dataclass
class Instruments:
    """The noise instruments of every listened region, loaded once per process."""

    noise: list[Series]
    velocity: list[tuple[Series, float]]


_LOADED: dict[tuple[str, tuple[str, ...], int], Instruments] = {}


def load(spec: FieldListen, root: Path | None = None) -> Instruments:
    from craton.bench.tasks import BenchContext
    from craton.paths import find_root

    base = root or find_root()
    key = (str(base), spec.regions, spec.baseline_days)
    if key not in _LOADED:
        noise, velocity = [], []
        for region in spec.regions:
            view = BenchContext(root=base, region=region)
            if spec.noise_power:
                noise.append(
                    noise_series(
                        read_table(view._snapshot_files(NOISE), COLUMNS), spec.baseline_days
                    )
                )
            if spec.velocity:
                velocity.append(
                    velocity_series(read_table(view._snapshot_files(VELOCITY), COLUMNS))
                )
        _LOADED[key] = Instruments(noise, velocity)
    return _LOADED[key]


def features(
    starts: FloatArray,
    grid: Grid,
    spec: FieldListen,
    cut: float,
    instruments: Instruments | None = None,
) -> np.ndarray:
    """(days, cells, ``N_FEATURES``) float16 for the midnights ``starts``; ``cut`` is the
    training cut-off (seconds), which decides whose dv/v may be read."""
    inst = instruments or load(spec)
    out = np.zeros((len(starts), grid.n_cells, N_FEATURES), np.float32)
    for s in inst.noise:
        part = _spread(s, starts, grid, spec, DB_SCALE, DB_CLIP)
        # Regions' windows do not overlap; where two regions both cover a day, the sum of
        # their averages would double, so each fills only days the others left empty.
        empty = out[..., 3:4] == 0
        out[..., :4] = np.where(empty, part, out[..., :4])
    for s, ref_end in inst.velocity:
        if ref_end > cut:
            continue
        part = _spread(s, starts, grid, spec, DVV_SCALE, DVV_CLIP)
        empty = out[..., 6:7] == 0
        out[..., 4:] = np.where(empty, part, out[..., 4:])
    return out.astype(np.float16)
