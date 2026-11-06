"""ETAS as the Field's starting line: daily ETAS rates for every day the Field learns from.

The same arithmetic as ``craton.bench.etas.forecast_days`` (which the tests hold it to), run
on the GPU: the Field needs ETAS's forecast for thousands of training days, and the CPU's
sparse product takes about an hour for central Italy where the GPU takes a minute. The
kernel matrix (parents × cells) is cut into blocks of parents so it fits in 6 GB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from craton.bench import etas
from craton.bench.csep import Grid

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

PARENT_BLOCK = 16_384
DAY_BLOCK = 256


class KernelCache:
    """ETAS's spatial kernel rows, computed once per quake for one fit and grid: a caller
    that asks again and again about the same history plus a few new quakes (CB-24's
    rankers) pays for the new ones only."""

    def __init__(self, fit: etas.EtasFit, grid: Grid, proj: etas.Projection) -> None:
        self.fit, self.grid, self.proj = fit, grid, proj
        self.index: dict[tuple[float, float, float], int] = {}
        self.mat = sp.csr_matrix((0, grid.n_cells))
        self.bands: Any = None

    def rows(self, cat: etas.Catalog) -> sp.csr_matrix:
        keys = list(zip(cat.x.tolist(), cat.y.tolist(), cat.m.tolist(), strict=True))
        new = [i for i, k in enumerate(keys) if k not in self.index]
        if new:
            more = etas.spatial_matrix(
                self.fit.params, cat.select(np.asarray(new)), self.fit.mc, self.grid, self.proj
            )
            for j, i in enumerate(new):
                self.index.setdefault(keys[i], self.mat.shape[0] + j)
            self.mat = sp.vstack([self.mat, more], format="csr")
        return self.mat[np.asarray([self.index[k] for k in keys], dtype=np.int64)]


def etas_daily(
    fit: etas.EtasFit,
    history: etas.Catalog,
    day_starts: FloatArray,
    grid: Grid,
    proj: etas.Projection,
    device: torch.device | None = None,
    span: float = 1.0,
    cache: KernelCache | None = None,
) -> FloatArray:
    """(days, cells) ETAS's expected quakes at or above ``fit.mc`` in the ``span`` days from
    each start (one day unless asked), reading only quakes before it. ``day_starts`` in days
    since ``fit.t_ref``."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr = fit.params
    parents = history.select(history.m >= fit.mc - 1e-9)
    smat = cache.rows(parents) if cache else etas.spatial_matrix(pr, parents, fit.mc, grid, proj)
    prod = pr.k * np.exp(pr.a * (parents.m - fit.mc))
    out = torch.zeros((len(day_starts), grid.n_cells), dtype=torch.float64, device=dev)
    starts = torch.tensor(day_starts, dtype=torch.float64, device=dev)
    for p0 in range(0, len(parents), PARENT_BLOCK):
        p1 = min(p0 + PARENT_BLOCK, len(parents))
        kern = torch.tensor(smat[p0:p1].toarray(), dtype=torch.float64, device=dev)
        t_par = torch.tensor(parents.t[p0:p1], dtype=torch.float64, device=dev)
        w_par = torch.tensor(prod[p0:p1], dtype=torch.float64, device=dev)
        for d0 in range(0, len(day_starts), DAY_BLOCK):
            t0 = starts[d0 : d0 + DAY_BLOCK, None]
            w = _omori_integral(t0 + span - t_par[None], pr.c, pr.p) - _omori_integral(
                t0 - t_par[None], pr.c, pr.p
            )
            w = torch.where(t_par[None] < t0, w, torch.zeros_like(w)) * w_par[None]
            out[d0 : d0 + DAY_BLOCK] += w @ kern
    background = span * pr.mu * fit.background.cell_mass(grid, proj)
    factor = etas.cascade_factor(pr, fit.b, fit.mc, span)
    rates = (torch.tensor(background, device=dev)[None] + out) * factor
    return np.asarray(rates.cpu().numpy(), dtype=np.float64)


def _omori_integral(t: torch.Tensor, c: float, p: float) -> torch.Tensor:
    tt = torch.clamp(t, min=0.0)
    return 1.0 - (c / (tt + c)) ** (p - 1)
