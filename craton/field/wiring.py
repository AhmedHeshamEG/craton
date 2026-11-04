"""The wiring (P9d): Omori attention, which cells excite which, starting as ETAS.

ETAS already says who triggers whom. A cell's expected quakes tomorrow are a sum: its
background, plus one share from every past quake, ``productivity × Omori(time) ×
kernel(distance)``. Normalise those shares and you have Zhuang's stochastic declustering:
the probability that a quake in this cell would be the child of that parent, or of nobody
(the background).

That sum is attention. Each source (the place a past quake sits in) sends its aftershocks
out over the target cells with weights given by ETAS's kernel: a power law in distance, the
way ALiBi biases attention by distance, only here the bias is ETAS's own kernel. The block
adds a learned score on top:

    sent(j → c) ∝ kernel(j, c) · exp(G(j, c)),   G = q_j · k_c  (bounded)

normalised over the targets, so **each source sends exactly as many aftershocks as ETAS
says it does**: its productivity is ETAS's, the wiring only decides where they land. That
is the law the block obeys (offspring are conserved), and it is also what keeps it from
doing the Field's job: a first version normalised over the sources, could raise or lower a
cell's total, and its output and the Field's correction ran to opposite limits and cancelled.
How many is the Field's; where is the wiring's. The background (the attention sink of
StreamingLLM, here ETAS's μ, a physical quantity) and quakes off the grid stay as ETAS has
them.

``q`` and ``k`` come from the Field's own cell features, so the wiring between two cells is
read off what both have been doing, not a free number per pair (that can't move to a new
region). ``G = 0`` sends every aftershock where ETAS sends it: the untrained block is ETAS,
bit for bit. The block's output is the ratio of the rewired sum to ETAS's own sum, so ETAS's
rate stays exact underneath. The normalised shares, rewired, are the learned "who triggered
whom" (``pair_matrix``).

To keep it on a 6 GB GPU the parents are grouped by the 0.1° cell they sit in and a
magnitude band (``BANDS``): the share of band ``b`` in source cell ``j`` is the summed
productivity × Omori of its quakes over the day, times one kernel from the cell's centre
with the band's magnitude. That grouping is only used to *rewire* ETAS's sources; ETAS's
rate itself stays exact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from craton.bench import etas
from craton.bench.csep import Grid
from craton.field.base import DAY_BLOCK, PARENT_BLOCK, KernelCache, _omori_integral
from craton.field.config import FieldShape, FieldWiring
from craton.field.model import FieldNet

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Magnitude bands above ETAS's mc, and the magnitude each band's kernel is drawn with.
BANDS = (0.0, 1.0, 2.0, 3.0, 4.0)
BAND_MAG = (0.5, 1.5, 2.5, 3.5, 4.5)
TINY = 1e-30


def band_of(m_rel: FloatArray) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(BANDS), m_rel, side="right") - 1, 0, len(BANDS) - 1)


def band_kernels(fit: etas.EtasFit, grid: Grid, proj: etas.Projection) -> np.ndarray:
    """(bands, source cells, target cells) float32: ETAS's spatial kernel of a quake at each
    cell's centre with each band's magnitude, integrated over every target cell."""
    lat, lon = grid.cell_centres()
    x, y = proj.to_km(lat, lon)
    out = np.zeros((len(BANDS), grid.n_cells, grid.n_cells), np.float32)
    for b, dm in enumerate(BAND_MAG):
        cat = etas.Catalog(np.zeros(grid.n_cells), x, y, np.full(grid.n_cells, fit.mc + dm))
        out[b] = etas.spatial_matrix(fit.params, cat, fit.mc, grid, proj).toarray()
    return out


def sources(
    fit: etas.EtasFit,
    history: etas.Catalog,
    day_starts: FloatArray,
    grid: Grid,
    proj: etas.Projection,
    span: float = 1.0,
    device: torch.device | None = None,
    cache: KernelCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What ETAS's sum is made of, for each day (``day_starts`` in days since ``fit.t_ref``):

    - (days, cells, bands) float32: each source cell's summed productivity × Omori integral,
      by magnitude band (in-grid parents);
    - (days, cells) float32: the exact triggered rate from parents outside the grid;
    - (cells,) float32: the background term.

    All before ETAS's cascade factor, the same arithmetic as ``base.etas_daily``."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pr = fit.params
    parents = history.select(history.m >= fit.mc - 1e-9)
    lat, lon = proj_inverse(proj, parents.x, parents.y)
    cell = grid.cell_of(lat, lon)
    slot = np.where(cell >= 0, cell * len(BANDS) + band_of(parents.m - fit.mc), -1)
    outside = np.flatnonzero(cell < 0)
    off = parents.select(outside)
    smat = None
    if len(outside):
        smat = cache.rows(off) if cache else etas.spatial_matrix(pr, off, fit.mc, grid, proj)
    prod = pr.k * np.exp(pr.a * (parents.m - fit.mc))
    n_d = len(day_starts)
    src = torch.zeros((n_d, grid.n_cells * len(BANDS)), dtype=torch.float64, device=dev)
    out = torch.zeros((n_d, grid.n_cells), dtype=torch.float64, device=dev)
    starts = torch.tensor(day_starts, dtype=torch.float64, device=dev)
    pos_out = np.full(len(parents), -1)
    pos_out[outside] = np.arange(len(outside))
    for p0 in range(0, len(parents), PARENT_BLOCK):
        p1 = min(p0 + PARENT_BLOCK, len(parents))
        t_par = torch.tensor(parents.t[p0:p1], dtype=torch.float64, device=dev)
        w_par = torch.tensor(prod[p0:p1], dtype=torch.float64, device=dev)
        sl = torch.tensor(slot[p0:p1], device=dev)
        inside = sl >= 0
        rows = pos_out[p0:p1]
        kern = None
        if smat is not None and (rows >= 0).any():
            kern = torch.tensor(smat[rows[rows >= 0]].toarray(), dtype=torch.float64, device=dev)
            out_mask = torch.tensor(rows >= 0, device=dev)
        for d0 in range(0, n_d, DAY_BLOCK):
            t0 = starts[d0 : d0 + DAY_BLOCK, None]
            w = _omori_integral(t0 + span - t_par[None], pr.c, pr.p) - _omori_integral(
                t0 - t_par[None], pr.c, pr.p
            )
            w = torch.where(t_par[None] < t0, w, torch.zeros_like(w)) * w_par[None]
            src[d0 : d0 + DAY_BLOCK].index_add_(1, sl[inside], w[:, inside])
            if kern is not None:
                out[d0 : d0 + DAY_BLOCK] += w[:, out_mask] @ kern
    background = span * pr.mu * fit.background.cell_mass(grid, proj)
    s = src.reshape(n_d, grid.n_cells, len(BANDS)).float().cpu().numpy()
    return s, out.float().cpu().numpy(), np.asarray(background, np.float32)


def proj_inverse(proj: etas.Projection, x: FloatArray, y: FloatArray) -> tuple[Any, Any]:
    """(lat, lon) of km coordinates: the inverse of ``proj.to_km`` (equirectangular)."""
    return proj.lat0 + np.asarray(y) / proj.ky, proj.lon0 + np.asarray(x) / proj.kx


class WiredFieldNet(nn.Module):
    """The P6 Field plus Omori attention: the Field's correction, plus the log ratio of the
    rewired ETAS sum to ETAS's own. ``kern`` is ``band_kernels``' output (a buffer that is
    not saved: it belongs to the grid the net runs on)."""

    def __init__(
        self, n_memory: int, grid: Grid, shape: FieldShape, wiring: FieldWiring, kern: Any = None
    ) -> None:
        super().__init__()
        self.field = FieldNet(n_memory, grid, shape)
        d, k = shape.d_model, wiring.d_key
        self.max_log = math.log(wiring.max_bias)
        self.scale = 1.0 / math.sqrt(k)
        self.query = nn.Linear(d, k)  # the source asks: where do my aftershocks go?
        self.key = nn.Linear(d, k)  # the target answers
        nn.init.zeros_(self.query.weight)  # G = 0: the untrained block is ETAS
        nn.init.zeros_(self.query.bias)
        n = grid.n_cells
        k0 = torch.zeros(len(BANDS), n, n) if kern is None else torch.as_tensor(kern)
        self.register_buffer("kern", k0.float(), persistent=False)

    def _kern(self) -> torch.Tensor:
        kern = self.kern
        assert isinstance(kern, torch.Tensor)
        return kern

    def bias(self, h: torch.Tensor) -> torch.Tensor:
        """(days, sources, targets) G, bounded to ± log(max_bias)."""
        h = h.float()
        with torch.autocast(h.device.type, enabled=False):
            g = torch.einsum("djk,dck->djc", self.query(h), self.key(h)) * self.scale
            return self.max_log * torch.tanh(g / self.max_log)

    def _sent(self, src: torch.Tensor, g: torch.Tensor | None) -> list[torch.Tensor]:
        """Per band, (days, sources, targets): the aftershocks each source sends to each
        target, ETAS's (``g`` None) or rewired; every source's row sums to ETAS's."""
        kern = self._kern()
        out = []
        for b in range(kern.shape[0]):
            kb = kern[b][None].expand(src.shape[0], -1, -1)
            if g is None:
                out.append(src[..., b, None] * kb)
                continue
            e = kb * g.exp()
            z = e.sum(-1) / kb.sum(-1).clamp(min=TINY)
            out.append((src[..., b] / z.clamp(min=TINY))[..., None] * e)
        return out

    def ratio(
        self, h: torch.Tensor, src: torch.Tensor, out: torch.Tensor, bg: torch.Tensor
    ) -> torch.Tensor:
        """(days, cells) log of the rewired ETAS sum over ETAS's own: zero when G is zero."""
        g = self.bias(h)
        with torch.autocast(src.device.type, enabled=False):  # shares of 1e-12 die in fp16
            src, rest = src.float(), out.float() + bg.float()[None]
            wired = sum(s.sum(1) for s in self._sent(src, g))
            plain = sum(s.sum(1) for s in self._sent(src, torch.zeros_like(g)))
            assert isinstance(wired, torch.Tensor)
            assert isinstance(plain, torch.Tensor)
            return torch.log((rest + wired).clamp(min=TINY)) - torch.log(
                (rest + plain).clamp(min=TINY)
            )

    def pair_matrix(
        self, h: torch.Tensor, src: torch.Tensor, out: torch.Tensor, bg: torch.Tensor
    ) -> torch.Tensor:
        """(days, cells, cells + 2) the learned declustering: for each target cell, the
        probability that its next quake is the child of each source cell, of a quake off the
        grid, or of nobody (the background)."""
        with torch.autocast(src.device.type, enabled=False):
            sent = sum(self._sent(src.float(), self.bias(h)))
            assert isinstance(sent, torch.Tensor)
            full = torch.cat(
                [
                    sent.transpose(1, 2),
                    out.float()[..., None],
                    bg.float()[None, :, None].expand(src.shape[0], -1, 1),
                ],
                dim=-1,
            )
            return full / full.sum(-1, keepdim=True).clamp(min=TINY)

    def forward(
        self,
        memory: torch.Tensor,
        log_etas: torch.Tensor,
        src: torch.Tensor,
        out: torch.Tensor,
        bg: torch.Tensor,
    ) -> torch.Tensor:
        h = self.field.encode(memory, log_etas)
        corr = self.field.head(h)
        return corr + self.ratio(h, src, out, bg)[..., None]
