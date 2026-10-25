"""ETAS as the Field's starting line: daily ETAS rates for every day the Field learns from.

The same arithmetic as ``craton.bench.etas.forecast_days`` (which the tests hold it to), run
on the GPU: the Field needs ETAS's forecast for thousands of training days, and the CPU's
sparse product takes about an hour for central Italy where the GPU takes a minute. The
kernel matrix (parents × cells) is cut into blocks of parents so it fits in 6 GB.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from craton.bench import etas
from craton.bench.csep import Grid

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

PARENT_BLOCK = 16_384
DAY_BLOCK = 256


def etas_daily(
    fit: etas.EtasFit,
    history: etas.Catalog,
    day_starts: FloatArray,
    grid: Grid,
    proj: etas.Projection,
    device: torch.device | None = None,
    span: float = 1.0,
) -> FloatArray:
    """(days, cells) ETAS's expected quakes at or above ``fit.mc`` in the ``span`` days from
    each start (one day unless asked), reading only quakes before it. ``day_starts`` in days
    since ``fit.t_ref``."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr = fit.params
    parents = history.select(history.m >= fit.mc - 1e-9)
    smat = etas.spatial_matrix(pr, parents, fit.mc, grid, proj)
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
