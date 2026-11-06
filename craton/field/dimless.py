"""Dimensionless inputs (P9h): a world Field that reads only ratios.

The world Field learned four regions at once and gained on Turkey, Chile and Alaska about
as much as the simplest fix to ETAS does (CB-22, P8d). One reason a network transfers badly
is that its inputs carry the units of the place it learned: how many quakes a day a cell
has, how fast its aftershocks die, how far a quake reaches. A region with ten times the
quakes of another sends the network numbers it has never seen, even if its physics is the
same. Physicists solve this with dimensionless groups (Reynolds number: a pipe and a river
behave alike at the same ratio). Here, four:

1. **Magnitude above completeness, m − mc** (since P8d): the memory counts from mc − 0.5, a
   quake at mc weighs 1, and the score starts at mc + 1, in every region.
2. **Counts over ETAS** (``ratios``): each fading counter divided by what ETAS's own rate
   for the cell would put in it, ``r · τ``. ETAS reaches the network only as each cell's
   share of the box. Multiply a region's rates by any factor and these inputs don't move.
   (``anchor``, v2) The box's whole ETAS rate over its own long-run rate (its 12-year
   counter): how deep into a sequence the box is. v1 left this out and lost Tohoku on how
   many (D-130): the raw Field reads the box's absolute rate, which carries the same signal
   in the region's units.
3. **Time in Omori units** (``omori_time``): aftershock rates fall as (t + c)^-p, and c is
   the box's own clock (mostly how long the network stays blind after a big shock). The
   bank's lifetimes are re-read in units of c: a box with twice central Italy's c reads each
   counter at twice the lifetime, by sliding along the log-spaced bank (P9c's
   ``shift_bank``, here a fixed slide per box, not a learned one).
4. **Distance in rupture lengths** (``rupture``): one more spread of counters, where each
   quake counts at full weight over a Gaussian as wide as half its rupture length
   (Wells & Coppersmith 1994, all slip types, log10 L = −2.44 + 0.59 M, km). A cell 30 km
   from a M7 (rupture ~50 km) is inside it; 30 km from a M5 (~3 km), far outside.

Everything else is the world Field: same trunk, same ETAS base, same zero-init output, so
the untrained dimensionless Field is ETAS exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from craton.bench.csep import Grid
from craton.bench.data import DAY, Events
from craton.field.config import FieldDimless, FieldShape
from craton.field.hidden import shift_bank
from craton.field.memory import EPS, PRODUCTIVITY, MemorySpec
from craton.field.model import OUTPUTS, place_code

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Wells & Coppersmith (1994), subsurface rupture length, all slip types: log10 L = A + B·M.
WC_A, WC_B = -2.44, 0.59
#: The rupture Gaussian's sigma, as a share of the rupture length.
SIGMA_SHARE = 0.5
#: Quakes whose sigma is under this many cells count in their own cell only.
MIN_SIGMA_CELLS = 0.5
KM_PER_DEG = 111.195
#: A counter holding this share of what ETAS expects reads as empty.
RATIO_EPS = 1e-3
#: (anchor) A box with no long-run history reads as this share of its current rate.
ANCHOR_EPS = 1e-3


def rupture_km(mag: FloatArray) -> FloatArray:
    return np.asarray(10.0 ** (WC_A + WC_B * np.asarray(mag, dtype=np.float64)))


def footprint(grid: Grid, lat: float, lon: float, sigma_km: float) -> FloatArray:
    """(cells,) a Gaussian of ``sigma_km`` around (lat, lon), 1 at its centre."""
    clat, clon = grid.cell_centres()
    dy = (np.asarray(clat) - lat) * KM_PER_DEG
    dx = (np.asarray(clon) - lon) * KM_PER_DEG * math.cos(math.radians(lat))
    return np.asarray(np.exp(-0.5 * (dx * dx + dy * dy) / sigma_km**2))


def rupture_memory(
    events: Events, grid: Grid, midnights: FloatArray, spec: MemorySpec
) -> np.ndarray:
    """(days, cells, lifetimes · 2) float16: the memory's two weights at every lifetime, each
    quake counted over its own rupture's Gaussian (a small one in its own cell), log(ε + ·).
    Reads only quakes before each midnight, like ``craton.field.memory.memory``."""
    if np.any(np.diff(midnights) <= 0):
        raise ValueError("midnights must be increasing")
    pick = (
        (events.mag >= spec.m_min - 1e-9)
        & (events.time >= spec.start)
        & (events.time < float(midnights[-1]))
    )
    ev = events.select(pick)
    cell = grid.cell_of(ev.lat, ev.lon)
    keep = cell >= 0
    t, c, m = ev.time[keep], cell[keep], ev.mag[keep]
    lat, lon = ev.lat[keep], ev.lon[keep]
    weights = np.stack([np.ones_like(m), 10.0 ** (PRODUCTIVITY * (m - spec.m_ref))], axis=1)
    sigma_km = SIGMA_SHARE * rupture_km(m)
    wide = sigma_km >= MIN_SIGMA_CELLS * grid.cell_deg * KM_PER_DEG
    taus = np.asarray(spec.lifetimes)
    state = np.zeros((grid.n_cells, len(taus), 2))
    out = np.zeros((len(midnights), grid.n_cells, len(taus) * 2), np.float16)
    t_prev, i0 = spec.start, 0
    for d, now in enumerate(midnights):
        i1 = int(np.searchsorted(t, now))
        state *= np.exp(-(now - t_prev) / DAY / taus)[None, :, None]
        if i1 > i0:
            fade = np.exp(-((now - t[i0:i1]) / DAY)[:, None] / taus[None, :])
            add = fade[:, :, None] * weights[i0:i1, None, :]
            small = ~wide[i0:i1]
            np.add.at(state, c[i0:i1][small], add[small])
            for j in np.flatnonzero(wide[i0:i1]):
                k = i0 + j
                f = footprint(grid, float(lat[k]), float(lon[k]), float(sigma_km[k]))
                state += f[:, None, None] * add[j][None]
        t_prev, i0 = float(now), i1
        out[d] = np.log(EPS + state.reshape(grid.n_cells, -1))
    return out


def omori_shift(c_days: float, cfg: FieldDimless, lifetimes: tuple[float, ...]) -> float:
    """Steps along the bank that turn central Italy's lifetimes into this box's Omori time:
    a box with c = r · c_ref reads lifetime τ at index k + log(r) / log(spacing)."""
    if not cfg.omori_time:
        return 0.0
    spacing = math.exp(np.mean(np.diff(np.log(np.asarray(lifetimes)))))
    return math.log(c_days / cfg.c_ref_days) / math.log(spacing)


class DimlessFieldNet(nn.Module):
    """The world Field's trunk reading the dimensionless inputs.

    ``memory`` (days, cells, spreads · K · 2) is the standard memory, with the rupture spread
    appended when it is on; ``log_etas`` (days, cells, 2) ETAS's log rate in both bins;
    ``shift`` (days,) each day's box in steps along the bank."""

    def __init__(
        self,
        n_memory: int,
        grid: Grid,
        shape: FieldShape,
        dimless: FieldDimless,
        lifetimes: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.dimless = dimless
        self.k = len(lifetimes)
        if n_memory % (self.k * 2):
            raise ValueError(f"{n_memory} memory features are not spreads × {self.k} × 2")
        self.spreads = n_memory // (self.k * 2)
        self.register_buffer("log_tau", torch.log(torch.tensor(lifetimes)).float())
        d = shape.d_model
        self.max_log = math.log(shape.max_nudge)
        n_etas = (2 if dimless.anchor else 1) if dimless.ratios else 2
        self.inp = nn.Sequential(nn.Linear(n_memory + n_etas, d), nn.GELU(), nn.Linear(d, d))
        code = place_code(grid)
        self.register_buffer("place", code)
        self.place_proj = nn.Linear(code.shape[1], d)
        layer = nn.TransformerEncoderLayer(
            d, shape.n_heads, 4 * d, shape.dropout, batch_first=True, norm_first=True
        )
        self.cells = (
            nn.TransformerEncoder(layer, shape.n_layers, enable_nested_tensor=False)
            if shape.n_layers
            else None
        )
        self.out = nn.Linear(d, len(OUTPUTS))
        nn.init.zeros_(self.out.weight)  # the untrained Field is ETAS exactly
        nn.init.zeros_(self.out.bias)

    def tau_log(self) -> torch.Tensor:
        """(K,) the bank's log lifetimes, in days."""
        out = self.log_tau
        assert isinstance(out, torch.Tensor)
        return out

    def features(
        self, memory: torch.Tensor, log_etas: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """(days, cells, n_memory + 1 or 2): what the trunk reads."""
        mem = memory.float()
        le = log_etas.float()
        # (anchor) the box's long-run rate: its longest counter, every quake weighing 1, before
        # any slide; read off the raw memory so Omori time can't move it.
        long_run = (mem[..., (self.k - 1) * 2].exp() - EPS).clamp(min=0.0).sum(-1)
        long_run = long_run / self.tau_log()[-1].exp()
        bank = mem.reshape(*mem.shape[:-1], self.spreads, self.k, 2)
        log_tau = self.tau_log().expand(*mem.shape[:-1], self.k)
        if self.dimless.omori_time:
            s = shift.float()[:, None].expand(*mem.shape[:-1])
            bank = shift_bank(bank, s[..., None].expand(*s.shape, self.spreads))
            log_tau = shift_bank(log_tau[..., None], s)[..., 0]
        if not self.dimless.ratios:
            total = le[..., 1].logsumexp(-1, keepdim=True).expand_as(le[..., 1])
            return torch.cat([bank.flatten(-3), le[..., 1:], total[..., None]], dim=-1)
        held = (bank.exp() - EPS).clamp(min=0.0)
        log_rate = le.logsumexp(-1)  # ETAS's daily rate at or above mc, per cell
        expect = (log_rate + 0.0)[..., None] + log_tau  # (days, cells, K): log r·τ
        ratio = torch.log(RATIO_EPS + held / expect.exp()[..., None, :, None])
        share = le[..., 1] - le[..., 1].logsumexp(-1, keepdim=True)
        parts = [ratio.flatten(-3), share[..., None]]
        if self.dimless.anchor:
            now = log_rate.logsumexp(-1)  # (days,) ETAS's rate for the whole box
            lr = torch.log(long_run + ANCHOR_EPS * now.exp())
            parts.append((now - lr)[:, None, None].expand(*share.shape, 1))
        return torch.cat(parts, dim=-1)

    def forward(
        self, memory: torch.Tensor, log_etas: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """(days, cells, 2) bounded log corrections for the two bins, as the Field's."""
        x = self.features(memory, log_etas, shift)
        h = self.inp(x) + self.place_proj(self.place)[None]
        if self.cells is not None:
            h = self.cells(h)
        c = self.out(h).float()
        return self.max_log * torch.tanh(c / self.max_log)
