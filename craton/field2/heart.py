"""Field 2's heart: attention that counts, born as ETAS.

A transformer's attention gives each query a *share* of the keys: the weights are a softmax, so
they sum to one, and twice as many keys don't make twice the answer. Earthquakes add up. Ten
aftershock sequences overlapping in one place make roughly ten times the quakes. So the heart's
attention is a **sum**: every past quake (a key) sends a message to every place (a query), and
the place's rate is the total of what it receives. ETAS is exactly this, with one fixed rule
for the weights. The heart keeps ETAS's rule and learns how far to bend it.

What a quake's message to a place depends on, read in bins:

- **age**, on a log clock: 0-30 min, 30 min-2 h, 2-6 h, 6 h-1 day, 1-3, 3-10, 10-30, 30-100
  days, … , 3,000 days and older. On this clock Omori's law is a straight line;
- **size**, m − mc: how far above what the network hears the quake was;
- **distance in the quake's own ruler**: rupture lengths (Wells and Coppersmith 1994), so an M7
  10 km away and an M5 1 km away are "as close" as each other. Self-similarity, if the Earth
  has it, becomes one shared weight.

No bin knows where on Earth it is. The weights come from a small network that reads the bin's
coordinates and the **context**: the region's own ETAS numbers (productivity, Omori's c and p,
the kernel's size, b, the completeness, the background per km²). That is how "born as ETAS"
and "no addresses" live together: the region's ETAS is told, never its name or position.

The rate of a cell on a day:

    λ = cascade · ( e^{h₀(ctx)} · background_cell + Σ_e  w_e · e^{h(bin_e, ctx)} )

where the sum runs over (parent, cell, ring) pieces of every parent's ETAS kernel, ``w_e`` is
ETAS's own weight of that piece (productivity × the Omori integral over the day × the share of
the kernel in that cell and ring), and ``h`` is the network's log weight, capped at ±cap. The
network's last layers start at exactly zero, so at step 0 every e^h is 1 and λ is ETAS's
forecast, number for number (``tests/test_field2.py`` holds it to 10⁻⁶).

Every forecast is a sum, so every forecast has a receipt: how much of it came from each bin.
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
from craton.field2.config import HeartBins, HeartNet

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: The context: the region's ETAS numbers, in this order.
CONTEXT = (
    "log10_mu_per_km2",
    "log10_k",
    "a",
    "log10_c",
    "p",
    "log10_d",
    "gamma",
    "q",
    "b",
    "mc",
    "branching",
)


# ---------------------------------------------------------------- the pieces of a kernel


@dataclass(frozen=True)
class Entries:
    """ETAS's kernel of every parent, cut by cell and ring: one row per (parent, cell, ring)
    with the share of the parent's kernel that lands there. Rows are sorted by parent."""

    parent: IntArray
    cell: IntArray
    ring: IntArray
    share: FloatArray

    def __len__(self) -> int:
        return len(self.parent)


def ring_of(r_km: FloatArray, m: FloatArray, edges: tuple[float, ...]) -> IntArray:
    """Which ring of the ruler (rupture lengths) a distance falls in."""
    ruler = r_km / etas.rupture_length_km(m)
    return np.asarray(np.searchsorted(np.asarray(edges), ruler, side="right") - 1, np.int64)


def ring_entries(
    params: etas.EtasParams,
    cat: etas.Catalog,
    mc: float,
    grid: Grid,
    proj: etas.Projection,
    edges: tuple[float, ...],
    chunk: int = 256,
) -> Entries:
    """``etas.spatial_matrix`` (isotropic), with each parent's kernel also cut into rings of
    its own rupture length. Summed over rings, the shares are that matrix exactly."""
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
        r2 = dx**2 + dy**2
        inside = valid & (r2 <= etas.R_MAX**2)
        m_abs = np.broadcast_to(cat.m[sl, None], r2.shape)
        w = np.where(inside, etas._spatial(params, r2, m_abs - mc), 0.0)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        cell = np.where(inside, cells[gyc, gxc], -1)
        keep = cell >= 0
        r, cidx = np.nonzero(keep)
        ring = ring_of(np.sqrt(r2[r, cidx]), m_abs[r, cidx], edges)
        keys.append(((r + s) * grid.n_cells + cell[r, cidx]) * n_rings + ring)
        vals.append(w[r, cidx])
    if not keys:
        empty = np.zeros(0, np.int64)
        return Entries(empty, empty, empty, np.zeros(0))
    key, inv = np.unique(np.concatenate(keys), return_inverse=True)
    share = np.bincount(inv, weights=np.concatenate(vals))
    ring = key % n_rings
    pc = key // n_rings
    return Entries(pc // grid.n_cells, pc % grid.n_cells, ring, np.asarray(share, np.float64))


# ---------------------------------------------------------------- one box, ready to forecast


@dataclass(frozen=True)
class Box:
    """Everything the heart needs to forecast one box: its parents, their kernel pieces, ETAS's
    background and cascade, and the context. Times are days since the ETAS fit's ``t_ref``."""

    t: FloatArray  # (P,) parent times
    mag_bin: IntArray  # (P,)
    prod: FloatArray  # (P,) k·e^{a(m − mc)}
    entries: Entries
    background: FloatArray  # (cells,) background quakes per day at or above mc
    cascade: float
    c: float
    p: float
    span: float
    context: FloatArray  # (len(CONTEXT),)
    n_cells: int


def context_of(fit: etas.EtasFit, grid: Grid, proj: etas.Projection) -> FloatArray:
    """The region's ETAS numbers as the heart reads them (``CONTEXT``)."""
    pr = fit.params
    x0, x1, y0, y1 = etas.box_km(grid, proj)
    area = abs((x1 - x0) * (y1 - y0))
    return np.array(
        [
            math.log10(pr.mu / area),
            math.log10(pr.k),
            pr.a,
            math.log10(pr.c),
            pr.p,
            math.log10(pr.d),
            pr.gamma,
            pr.q,
            fit.b,
            fit.mc,
            min(fit.branching_ratio, 2.0),
        ]
    )


def make_box(
    fit: etas.EtasFit,
    history: etas.Catalog,
    grid: Grid,
    proj: etas.Projection,
    bins: HeartBins,
    span: float = 1.0,
) -> Box:
    """A box from its ETAS fit and every quake it knows (model units, as ``forecast_days``)."""
    pr = fit.params
    parents = history.select(history.m >= fit.mc - 1e-9)
    ent = ring_entries(pr, parents, fit.mc, grid, proj, bins.ruler)
    m_rel = parents.m - fit.mc
    mag_bin = np.searchsorted(np.asarray(bins.magnitude), m_rel, side="right") - 1
    return Box(
        t=parents.t,
        mag_bin=np.clip(mag_bin, 0, None).astype(np.int64),
        prod=pr.k * np.exp(pr.a * m_rel),
        entries=ent,
        background=pr.mu * fit.background.cell_mass(grid, proj),
        cascade=etas.cascade_factor(pr, fit.b, fit.mc, span),
        c=pr.c,
        p=pr.p,
        span=span,
        context=context_of(fit, grid, proj),
        n_cells=grid.n_cells,
    )


# ---------------------------------------------------------------- the network


def bin_coordinates(bins: HeartBins) -> np.ndarray:
    """(bins, 3) each bin's centre on the log clock, in size and on the log ruler, scaled to
    about ±1; flat index = (age · sizes + size) · rings + ring."""

    def centres(edges: tuple[float, ...], log: bool, last: float) -> np.ndarray:
        e = np.asarray(edges, np.float64)
        hi = np.append(e[1:], e[-1] * last if log else e[-1] + last)
        if log:
            lo = np.where(e > 0, e, hi / 2)
            return np.asarray(np.log10(np.sqrt(lo * hi)))
        return np.asarray((e + hi) / 2)

    age = centres(bins.age_days, True, 3.0) / 2.0
    size = centres(bins.magnitude, False, 1.0) / 2.0
    ruler = centres(bins.ruler, True, 3.0) / 1.5
    a, s, r = np.meshgrid(age, size, ruler, indexing="ij")
    return np.stack([a.ravel(), s.ravel(), r.ravel()], axis=1).astype(np.float32)


def _mlp(n_in: int, hidden: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    width = n_in
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    last = nn.Linear(width, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


class Heart(nn.Module):
    """Log weights for every bin and for the background, from the bins and the context.

    The context is standardised with the training tiles' mean and spread (buffers, saved with
    the weights). Both last layers start at zero: born as ETAS."""

    def __init__(self, bins: HeartBins, net: HeartNet) -> None:
        super().__init__()
        n_ctx = len(CONTEXT)
        self.cap = net.cap
        self.reads_context = net.context
        self.register_buffer("bin_x", torch.from_numpy(bin_coordinates(bins)))
        self.register_buffer("ctx_mean", torch.zeros(n_ctx))
        self.register_buffer("ctx_std", torch.ones(n_ctx))
        self.kernel = _mlp(3 + n_ctx, net.hidden, net.layers)
        self.background = _mlp(n_ctx, net.hidden, 1)
        self.free: nn.Module | None = None
        if net.free_head:
            from craton.field2.free import FreeHead, free_coordinates

            coords = free_coordinates(bins.age_days, net.free_magnitude, bins.ruler)
            self.free = FreeHead(coords, net.hidden, net.layers, net.cap)

    def _z(self, context: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.ctx_mean  # type: ignore[assignment]
        std: torch.Tensor = self.ctx_std  # type: ignore[assignment]
        z = ((context - mean) / std).float()
        # (P11) Without context, every region gets the same weights: one rule for the Earth.
        return z if self.reads_context else torch.zeros_like(z)

    def free_log_rate(self, context: torch.Tensor) -> torch.Tensor | None:
        """(P11, the free head) log quakes per day per ring for each free bin, or None."""
        if self.free is None:
            return None
        out: torch.Tensor = self.free(self._z(context))
        return out

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(log weight per bin, log background weight) for one box's raw context."""
        z = self._z(context)
        bin_x: torch.Tensor = self.bin_x  # type: ignore[assignment]
        x = torch.cat([bin_x, z.expand(bin_x.shape[0], -1)], dim=1)
        h = self.kernel(x).squeeze(-1)
        h0 = self.background(z[None, :]).squeeze()
        return self.cap * torch.tanh(h / self.cap), self.cap * torch.tanh(h0 / self.cap)


# ---------------------------------------------------------------- the sum


@dataclass
class DeviceBox:
    """A box's arrays on the device, ready for ``rates``."""

    t: torch.Tensor  # float64
    prod: torch.Tensor  # float64
    e_parent: torch.Tensor  # int64
    e_cell: torch.Tensor  # int64
    e_static: torch.Tensor  # int64: size · rings + ring
    e_share: torch.Tensor  # float32
    background: torch.Tensor  # float32
    context: torch.Tensor  # float32
    box: Box


def to_device(box: Box, bins: HeartBins, dev: torch.device) -> DeviceBox:
    e = box.entries
    n_rings = len(bins.ruler)
    static = box.mag_bin[e.parent] * n_rings + e.ring
    return DeviceBox(
        t=torch.as_tensor(box.t, dtype=torch.float64, device=dev),
        prod=torch.as_tensor(box.prod, dtype=torch.float64, device=dev),
        e_parent=torch.as_tensor(e.parent, dtype=torch.int64, device=dev),
        e_cell=torch.as_tensor(e.cell, dtype=torch.int64, device=dev),
        e_static=torch.as_tensor(static, dtype=torch.int64, device=dev),
        e_share=torch.as_tensor(e.share, dtype=torch.float32, device=dev),
        background=torch.as_tensor(box.background, dtype=torch.float32, device=dev),
        context=torch.as_tensor(box.context, dtype=torch.float32, device=dev),
        box=box,
    )


def rates(
    db: DeviceBox,
    days: torch.Tensor,
    log_w: torch.Tensor,
    log_w0: torch.Tensor,
    bins: HeartBins,
) -> torch.Tensor:
    """(days, cells) expected quakes at or above mc from each midnight in ``days`` (days since
    ``t_ref``), with log weights ``log_w`` per bin and ``log_w0`` on the background. All-zero
    weights give ETAS's ``forecast_days`` exactly."""
    box = db.box
    n_static = len(bins.magnitude) * len(bins.ruler)
    age = days.double()[:, None] - db.t[None, :]  # (D, P)
    seen = age > 0

    def omori(x: torch.Tensor) -> torch.Tensor:
        return 1.0 - (box.c / (x.clamp(min=0.0) + box.c)) ** (box.p - 1)

    w = torch.where(seen, db.prod[None, :] * (omori(age + box.span) - omori(age)), 0.0).float()
    edges = torch.as_tensor(bins.age_days, dtype=torch.float64, device=age.device)
    a_bin = (torch.searchsorted(edges, age.contiguous(), right=True) - 1).clamp(min=0)
    idx = a_bin[:, db.e_parent] * n_static + db.e_static[None, :]
    piece = w[:, db.e_parent] * db.e_share[None, :] * log_w.exp()[idx]
    trig = torch.zeros(len(days), box.n_cells, device=age.device, dtype=piece.dtype)
    trig.index_add_(1, db.e_cell, piece)
    bg = log_w0.exp() * db.background * box.span
    return box.cascade * (bg[None, :] + trig)


def receipt(
    db: DeviceBox, day: float, cell: int, log_w: torch.Tensor, bins: HeartBins
) -> dict[str, float]:
    """(P11, the receipt) What one cell's forecast on one day is made of: its expected count
    split by the age of the quakes that sent it (and the background). Sums to the forecast."""
    names = [f"age≥{a:g}d" for a in bins.age_days]
    sel = db.e_cell == cell
    sub = DeviceBox(
        db.t,
        db.prod,
        db.e_parent[sel],
        db.e_cell[sel],
        db.e_static[sel],
        db.e_share[sel],
        db.background,
        db.context,
        db.box,
    )
    out: dict[str, float] = {}
    days = torch.tensor([day], dtype=torch.float64, device=db.t.device)
    age = days[:, None] - db.t[None, :]
    edges = torch.as_tensor(bins.age_days, dtype=torch.float64, device=age.device)
    a_bin = (torch.searchsorted(edges, age.contiguous(), right=True) - 1).clamp(min=0)[0]
    for k, name in enumerate(names):
        keep = a_bin[sub.e_parent] == k
        part = DeviceBox(
            sub.t,
            sub.prod,
            sub.e_parent[keep],
            sub.e_cell[keep],
            sub.e_static[keep],
            sub.e_share[keep],
            torch.zeros_like(sub.background),
            sub.context,
            sub.box,
        )
        out[name] = float(rates(part, days, log_w, torch.zeros(()), bins)[0, cell])
    bg_only = float(db.box.cascade * db.background[cell] * db.box.span)
    out["background"] = bg_only
    return out
