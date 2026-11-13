"""Field 2's second counting head: a kernel of its own, born at zero (``net.free_head``).

The first head is ETAS's kernel with learned weights on it: it can bend ETAS, never reach
where ETAS has no mass. This head starts from nothing. Every quake, down to half a magnitude
below what the network fully hears (``free_magnitude``), sends each place a learned amount per
day, read in the same bins (age on the log clock, size, distance in its own rupture lengths),
spread evenly over the ring it falls in:

    λ_free(cell, day) = e^{s} · Σ_e  e^{g(bin_e, ctx)} · u_e

``u_e`` is the share of a ring's area (its quadrature points) that lands in the cell, so each
ring of each quake carries e^{s + g} quakes a day in total, split by area. ``s`` starts at
``FREE_START``, so the head is born as a whisper (about 0.2% of a tile's rate) and the heart is
ETAS within 0.01 bit; ``g`` starts at zero.

Quakes below mc matter because the network hears some of them: a cell where many small ones
are piling up is a cell ETAS (which only counts from mc) cannot see warming. The Field's memory
reads them too, from mc − 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from craton.bench import etas
from craton.bench.csep import Grid
from craton.field2.heart import CONTEXT, Entries, _mlp, ring_of

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: The head's log scale at birth and how fast it moves: s = FREE_START + FREE_SPEED * u, u born at
#: 0. e^-18 per ring per quake per day adds about 0.2% to a tile's rate (a tile has ~10^4 quakes
#: and 13 rings): born within 0.01 bit of ETAS, the plan's bar, yet loud enough for its shape to
#: learn. At e^-25 it was exactly silent and never woke: a uniform whisper is, on average, neither
#: good nor bad, so its gradient's sign flipped every step and Adam's momentum stayed at zero.
FREE_START = -18.0
FREE_SPEED = 20.0


def uniform_entries(
    cat: etas.Catalog, grid: Grid, proj: etas.Projection, edges: tuple[float, ...], chunk: int = 256
) -> Entries:
    """For every quake, each ring of its ruler (within ``etas.R_MAX``) spread evenly over its
    quadrature points: rows (quake, cell, ring, share of the ring in the cell)."""
    xs, ys, cells, _ = etas._fine_points(grid, proj)
    dxs, dys = xs[1] - xs[0], ys[1] - ys[0]
    rx, ry = math.ceil(etas.R_MAX / dxs) + 1, math.ceil(etas.R_MAX / dys) + 1
    grid_y, grid_x = np.meshgrid(np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij")
    oy, ox = grid_y.ravel(), grid_x.ravel()
    n_rings = len(edges)
    keys, vals = [], []
    for s in range(0, len(cat), chunk):
        sl = slice(s, s + chunk)
        ix = np.rint((cat.x[sl] - xs[0]) / dxs).astype(np.int64)
        iy = np.rint((cat.y[sl] - ys[0]) / dys).astype(np.int64)
        gx = ix[:, None] + ox[None, :]
        gy = iy[:, None] + oy[None, :]
        valid = (gx >= 0) & (gx < len(xs)) & (gy >= 0) & (gy < len(ys))
        gxc, gyc = np.clip(gx, 0, len(xs) - 1), np.clip(gy, 0, len(ys) - 1)
        dx, dy = xs[gxc] - cat.x[sl, None], ys[gyc] - cat.y[sl, None]
        r = np.sqrt(dx**2 + dy**2)
        disc = r <= etas.R_MAX
        ring = ring_of(r, np.broadcast_to(cat.m[sl, None], r.shape), edges)
        # Points per (quake, ring) over the whole disc, inside the grid or not: a ring half
        # outside the box puts only half its quakes in it, as ETAS's kernel does.
        rows = np.broadcast_to(np.arange(r.shape[0])[:, None], r.shape)
        per = np.zeros((r.shape[0], n_rings))
        np.add.at(per, (rows[disc], ring[disc]), 1.0)
        cell = np.where(valid & disc, cells[gyc, gxc], -1)
        keep = cell >= 0
        q, pidx = np.nonzero(keep)
        rg = ring[q, pidx]
        keys.append(((q + s) * grid.n_cells + cell[q, pidx]) * n_rings + rg)
        vals.append(1.0 / per[q, rg])
    if not keys:
        empty = np.zeros(0, np.int64)
        return Entries(empty, empty, empty, np.zeros(0))
    key, inv = np.unique(np.concatenate(keys), return_inverse=True)
    share = np.bincount(inv, weights=np.concatenate(vals))
    pc = key // n_rings
    return Entries(pc // grid.n_cells, pc % grid.n_cells, key % n_rings, np.asarray(share))


@dataclass(frozen=True)
class FreeBox:
    """The free head's quakes (down to mc − below) and their ring pieces; days since t_ref."""

    t: FloatArray
    mag_bin: IntArray
    entries: Entries


def make_free(
    history: etas.Catalog,
    mc: float,
    grid: Grid,
    proj: etas.Projection,
    ruler: tuple[float, ...],
    magnitude: tuple[float, ...],
) -> FreeBox:
    quakes = history.select(history.m >= mc + magnitude[0] - 1e-9)
    m_rel = quakes.m - mc
    mag_bin = np.searchsorted(np.asarray(magnitude), m_rel + 1e-9, side="right") - 1
    return FreeBox(
        quakes.t,
        np.clip(mag_bin, 0, None).astype(np.int64),
        uniform_entries(quakes, grid, proj, ruler),
    )


def free_coordinates(
    age_days: tuple[float, ...], magnitude: tuple[float, ...], ruler: tuple[float, ...]
) -> np.ndarray:
    from craton.field2.config import HeartBins
    from craton.field2.heart import bin_coordinates

    return bin_coordinates(HeartBins(age_days=age_days, magnitude=magnitude, ruler=ruler))


class FreeHead(nn.Module):
    """g(bin, ctx) per free bin, and the head's log scale s."""

    def __init__(self, coords: np.ndarray, hidden: int, layers: int, cap: float) -> None:
        super().__init__()
        self.cap = cap
        self.register_buffer("bin_x", torch.from_numpy(coords))
        self.kernel = _mlp(coords.shape[1] + len(CONTEXT), hidden, layers)
        self.wake = nn.Parameter(torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Log quakes per day per ring for every free bin (s + g), from the standardised
        context ``z``."""
        bin_x: torch.Tensor = self.bin_x  # type: ignore[assignment]
        x = torch.cat([bin_x, z.expand(bin_x.shape[0], -1)], dim=1)
        g = self.kernel(x).squeeze(-1)
        return FREE_START + FREE_SPEED * self.wake + self.cap * torch.tanh(g / self.cap)


@dataclass
class DeviceFree:
    t: torch.Tensor
    e_parent: torch.Tensor
    e_cell: torch.Tensor
    e_static: torch.Tensor
    e_share: torch.Tensor
    n_cells: int


def free_to_device(fb: FreeBox, n_rings: int, n_cells: int, dev: torch.device) -> DeviceFree:
    e = fb.entries
    return DeviceFree(
        torch.as_tensor(fb.t, dtype=torch.float64, device=dev),
        torch.as_tensor(e.parent, dtype=torch.int64, device=dev),
        torch.as_tensor(e.cell, dtype=torch.int64, device=dev),
        torch.as_tensor(fb.mag_bin[e.parent] * n_rings + e.ring, dtype=torch.int64, device=dev),
        torch.as_tensor(e.share, dtype=torch.float32, device=dev),
        n_cells,
    )


def free_rates(
    fd: DeviceFree,
    days: torch.Tensor,
    log_rate: torch.Tensor,
    age_days: tuple[float, ...],
    n_static: int,
    span: float,
) -> torch.Tensor:
    """(days, cells) the free head's quakes from each midnight, ``span`` days long."""
    age = days.double()[:, None] - fd.t[None, :]
    seen = age > 0
    edges = torch.as_tensor(age_days, dtype=torch.float64, device=age.device)
    a_bin = (torch.searchsorted(edges, age.contiguous(), right=True) - 1).clamp(min=0)
    idx = a_bin[:, fd.e_parent] * n_static + fd.e_static[None, :]
    on = seen[:, fd.e_parent].float()
    piece = on * fd.e_share[None, :] * log_rate.exp()[idx]
    out = torch.zeros(len(days), fd.n_cells, device=age.device, dtype=piece.dtype)
    out.index_add_(1, fd.e_cell, piece)
    return out * span
