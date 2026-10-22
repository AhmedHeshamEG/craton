"""Omori memory: what every forecast cell remembers about the quakes that happened in it.

ETAS says an earthquake's aftershocks fade as a power law in time, (t + c)^-p. A power law is
a weighted sum of exponentials with log-spaced lifetimes (a power law has no preferred scale,
so it needs one exponential per decade or so). So instead of remembering every quake, each
cell keeps a few **fading counters**: a counter with lifetime τ adds a quake's weight when it
happens and forgets e^(-Δt/τ) of it as time passes. With lifetimes from an hour to a dozen
years, any Omori-like decay is a fixed recipe over the counters, and ETAS's triggering is one
such recipe. A learned model can find others.

This is a diagonal state-space layer with fixed decays (S4 / Mamba's long memory without the
learning): constant cost per day, however long the history.

Each counter comes in two weights (every quake counts 1; or 10^(m − 2), how many aftershocks
a quake of magnitude m makes relative to an M2) and three spreads (the cell itself, and
Gaussians of 1 and 3 cells, ~10 and ~30 km), so a cell also remembers its neighbourhood.
Features are log(ε + counter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from craton.bench.csep import Grid
from craton.bench.data import DAY, Events

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Counter lifetimes in days: 1 hour, 4 hours, 1 day, 4, 16, 64, 256 days, 3 and 12 years.
LIFETIMES_DAYS = (1 / 24, 4 / 24, 1.0, 4.0, 16.0, 64.0, 256.0, 3 * 365.25, 12 * 365.25)
#: Neighbourhood blur in cells (0 = the cell alone).
SPREADS = (0.0, 1.0, 3.0)
#: Productivity weight 10^(PRODUCTIVITY · (m − M_REF)).
PRODUCTIVITY = 1.0
M_REF = 2.0
EPS = 1e-4


@dataclass(frozen=True)
class MemorySpec:
    """What the memory counts: quakes at or above ``m_min``, since ``start`` (seconds)."""

    m_min: float = 1.5
    start: float = 631_152_000.0  # 1990-01-01
    lifetimes: tuple[float, ...] = LIFETIMES_DAYS
    spreads: tuple[float, ...] = SPREADS

    @property
    def n_features(self) -> int:
        return len(self.lifetimes) * 2 * len(self.spreads)


def memory(events: Events, grid: Grid, midnights: FloatArray, spec: MemorySpec) -> np.ndarray:
    """(days, cells, features) float16: the log counters at each midnight, from quakes before it.

    ``midnights`` must be increasing. A quake at exactly midnight belongs to the day it starts.
    """
    if np.any(np.diff(midnights) <= 0):
        raise ValueError("midnights must be increasing")
    ev = events.above(spec.m_min).window(spec.start, float(midnights[-1]))
    cell = grid.cell_of(ev.lat, ev.lon)
    keep = cell >= 0
    t, c, m = ev.time[keep], cell[keep], ev.mag[keep]
    weights = np.stack([np.ones_like(m), 10.0 ** (PRODUCTIVITY * (m - M_REF))], axis=1)
    taus = np.asarray(spec.lifetimes)
    k, w = len(taus), weights.shape[1]
    state = np.zeros((grid.n_cells, k, w))
    out = np.zeros((len(midnights), grid.n_cells, spec.n_features), np.float16)
    t_prev = spec.start
    i0 = 0
    for d, now in enumerate(midnights):
        i1 = int(np.searchsorted(t, now))
        state *= np.exp(-(now - t_prev) / DAY / taus)[None, :, None]
        if i1 > i0:
            fade = np.exp(-((now - t[i0:i1]) / DAY)[:, None] / taus[None, :])
            np.add.at(state, c[i0:i1], fade[:, :, None] * weights[i0:i1, None, :])
        t_prev, i0 = float(now), i1
        img = state.reshape(grid.n_lat, grid.n_lon, k * w)
        parts = []
        for s in spec.spreads:
            if s == 0:
                parts.append(img)
            else:
                parts.append(
                    np.stack(
                        [gaussian_filter(img[..., j], s, mode="constant") for j in range(k * w)],
                        axis=-1,
                    )
                )
        out[d] = np.log(EPS + np.concatenate(parts, axis=-1).reshape(grid.n_cells, -1))
    return out
