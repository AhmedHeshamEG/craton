"""The steerable kernel's input: which way the quakes around each cell are lined up.

The Field's neighbourhood memory (``craton.field.memory``) blurs past quakes with a round
Gaussian: a quake 20 km north of a cell counts as much as one 20 km east. Real aftershocks
don't spread round. They run along the fault that slipped and gather in the lobes where the
slip pushed the rock closer to failure (Coulomb stress, King, Stein & Lin 1994), and both
patterns have a direction.

A kernel stretched along an angle θ, ``g(r) · (1 + e·cos 2(φ − θ))`` with φ the azimuth from
the cell to the quake, sums over past quakes into two parts:

    Σ g(r) · (1 + e cos 2(φ − θ)) = I + e · Re(e^{−2iθ} · Q),   Q = Σ g(r) · e^{2iφ}

``I`` is the round sum and ``Q`` the **quadrupole moment**: one complex number per cell
whose angle is twice the direction the quakes line up along and whose size says how strongly.
So any stretched kernel, at any angle, is a recipe over ``I`` and ``Q`` (a steerable filter,
Freeman & Adelson 1991): compute both once, steer later. Rotate the catalogue by α and Q
turns by 2α; that is what lets the angle mean something physical.

This module computes ``q = Q / I`` for every counter of the memory (so ``|q| ≤ 1``), with
distances in kilometres, never in degrees: a 0.1° cell is 11 km tall and ~8 km wide in
central Italy, and an angle computed on the grid would lean east-west on its own. The
convolutions run on the GPU.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from craton.bench.csep import Grid
from craton.bench.data import Events
from craton.field.memory import MemorySpec, counter_states

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

KM_PER_DEG = 111.195
#: The kernel reaches this many widths.
REACH = 3.0
#: Below this round sum (in quakes) a cell has no direction: q = 0.
MIN_SUM = 1e-3
DAY_BLOCK = 512


def kernels(grid: Grid, sigma_km: float) -> torch.Tensor:
    """(3, 1, h, w) the round Gaussian and the two quadrupole kernels on the grid's cells,
    azimuth φ east of north: cos 2φ = (y² − x²)/r², sin 2φ = 2xy/r² (zero at the centre)."""
    lat0 = 0.5 * (grid.lat_min + grid.lat_max)
    dy = grid.cell_deg * KM_PER_DEG
    dx = dy * math.cos(math.radians(lat0))
    ny = max(1, math.ceil(REACH * sigma_km / dy))
    nx = max(1, math.ceil(REACH * sigma_km / dx))
    y = torch.arange(-ny, ny + 1, dtype=torch.float64)[:, None] * dy
    x = torch.arange(-nx, nx + 1, dtype=torch.float64)[None, :] * dx
    r2 = x**2 + y**2
    g = torch.exp(-0.5 * r2 / sigma_km**2)
    safe = torch.where(r2 > 0, r2, torch.ones_like(r2))
    q1 = torch.where(r2 > 0, g * (y**2 - x**2) / safe, torch.zeros_like(g))
    q2 = torch.where(r2 > 0, g * 2 * x * y / safe, torch.zeros_like(g))
    # Image rows run south to north (cell = i_lat * n_lon + i_lon); conv2d correlates, so
    # the kernel at row +k reads the cell k rows north: y grows with the row, as here.
    return torch.stack([g, q1, q2])[:, None].float()


def quadrupole(
    events: Events,
    grid: Grid,
    midnights: FloatArray,
    spec: MemorySpec,
    sigma_km: float,
    device: torch.device | None = None,
) -> np.ndarray:
    """(days, cells, lifetimes · weights, 2) float16: ``q = Q / I`` of every memory counter at
    each midnight, from quakes before it, as (Re, Im). Counters in the memory's order."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ker = kernels(grid, sigma_km).to(dev)
    pad = (ker.shape[-1] // 2, ker.shape[-1] // 2, ker.shape[-2] // 2, ker.shape[-2] // 2)
    kw = len(spec.lifetimes) * 2
    out = np.zeros((len(midnights), grid.n_cells, kw, 2), np.float16)
    block: list[np.ndarray] = []

    def flush(end: int) -> None:
        imgs = torch.tensor(np.stack(block), dtype=torch.float32, device=dev)  # (b, cells, k, w)
        b = imgs.shape[0]
        x = imgs.reshape(b, grid.n_lat, grid.n_lon, kw).permute(0, 3, 1, 2)
        x = functional.pad(x.reshape(b * kw, 1, grid.n_lat, grid.n_lon), pad)
        y = functional.conv2d(x, ker).reshape(b, kw, 3, grid.n_lat, grid.n_lon)
        i, q = y[:, :, 0], y[:, :, 1:]
        q = torch.where(i[:, :, None] > MIN_SUM, q / i[:, :, None].clamp(min=MIN_SUM), 0.0)
        q = q.permute(0, 3, 4, 1, 2).reshape(b, grid.n_cells, kw, 2)
        out[end - b : end] = q.clamp(-1, 1).half().cpu().numpy()
        block.clear()

    for d, state in enumerate(counter_states(events, grid, midnights, spec)):
        block.append(state.astype(np.float32))
        if len(block) == DAY_BLOCK:
            flush(d + 1)
    if block:
        flush(len(midnights))
    return out


def azimuth_of(v: FloatArray) -> FloatArray:
    """Axial azimuth in degrees east of north, [0, 180), of spin-2 vectors (..., 2)."""
    return np.asarray(np.degrees(0.5 * np.arctan2(v[..., 1], v[..., 0])) % 180.0)
