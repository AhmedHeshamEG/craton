"""Field 2's heart, rebuilt (P11): a query with content.

The first heart (``craton.field2.heart``) gave each past quake a weight by its age, size and
distance, and added. Its *query was empty*: a place had nothing of its own to say, so the heart
could not tell "this place is running hot" from "this place is calm", and ETAS's miss on the
test sequences is all of that kind (how many, never where). The rebuild gives each place a
voice, and every part has its own switch (``HeartNet``):

- **A state** (``state`` channels). Every quake from mc − 0.5 (``state_below_mc``) writes into
  the place it falls in, a learned amount per channel that depends on its age and size; the
  writing fades on the log clock as the network learns it to. Divided by what the place's
  background would have written, so the state is a dimensionless "how busy, over usual". The
  state is learned from the forecast loss alone: nothing is handed to it but the quakes.
- **Neighbours** (``neighbours`` rounds). Each round, every place reads the average state of
  the places around it (3 × 3 and 9 × 9 cells) and mixes it into its own: relative geometry
  only, no address. Quakes infect their neighbours; so do states.
- **The query bends the message.** A quake's message to a place is ETAS's weight times
  ``e^{h(bin) + Σ_r u_r(bin) · v_r(state)}``: ``h`` the first heart's bin weight, and ``rank``
  directions in which the receiving place's state turns it up or down (a busy place may hear
  yesterday's quakes louder and last year's softer).
- **Rulers** (``rulers``). Age is read in the region's own Omori c (log10 age / c), the state in
  the place's own background, size above completeness, distance in rupture lengths. The
  region's ETAS numbers scale the inputs and are never an input themselves.
- **A smooth clock** (``smooth_clock``). A parent's weight is blended between the two age bins
  around it on the log clock, so nothing jumps when a quake turns one day old.
- **A plus beside the times** (``additive``). Quakes the place's state adds by itself, as a
  multiple of its background: where ETAS has no mass, the state can still speak.
- **How excited ETAS is** (``excitement``, off by default). Two more state inputs: ETAS's rate
  over the background, in the cell and in the box.

Born as ETAS: the readout's last layer and the kernel's last layer start at zero, so with the
additive head off an untrained heart is ETAS exactly; with it on, its whisper (``ADD_START``,
a thousandth of the background) keeps it within the plan's 0.01 bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from craton.field2.config import HeartBins, HeartNet
from craton.field2.heart import CONTEXT, DeviceBox, _mlp, rates
from craton.field2.tested import StateQuakes

#: The additive head's log scale at birth: a thousandth of the background.
ADD_START = math.log(1e-3)
#: Quakes older than this many days before a forecast no longer write a state.
STATE_MAX_AGE = 3650.0
#: Neighbourhoods each round reads (cells per side).
NEIGHBOURHOODS = (3, 9)


def age_centres(edges: tuple[float, ...]) -> np.ndarray:
    """log10 of each age bin's centre in days (the first bin's lower edge is taken as half its
    upper one, the last bin's upper edge as three times its lower one)."""
    e = np.asarray(edges, np.float64)
    hi = np.append(e[1:], e[-1] * 3.0)
    lo = np.where(e > 0, e, hi / 2)
    return np.asarray(np.log10(np.sqrt(lo * hi)))


def static_coordinates(bins: HeartBins) -> np.ndarray:
    """(sizes · rings, 2) size and log ruler of each (size, ring) pair, about ±1."""
    from craton.field2.heart import bin_coordinates

    x = bin_coordinates(bins)
    n_static = len(bins.magnitude) * len(bins.ruler)
    return np.asarray(x[:n_static, 1:], np.float32)


def age_coordinate(log10_days: torch.Tensor, c: float, rulers: bool) -> torch.Tensor:
    """The clock the network reads: log10 age in Omori c with rulers, in days without."""
    if rulers:
        return (log10_days - math.log10(c) - 3.0) / 3.0
    return log10_days / 2.0


@dataclass
class DeviceQuakes:
    """The state's quakes on the device: days since the box's t_ref, cell, m − mc."""

    t: torch.Tensor  # float64
    cell: torch.Tensor  # int64
    m_rel: torch.Tensor  # float32


def quakes_to_device(q: StateQuakes, t_ref: float, below: float, dev: torch.device) -> DeviceQuakes:
    keep = q.m_rel >= -below - 1e-6
    return DeviceQuakes(
        torch.as_tensor((q.time[keep] - t_ref) / 86_400.0, dtype=torch.float64, device=dev),
        torch.as_tensor(q.cell[keep], dtype=torch.int64, device=dev),
        torch.as_tensor(q.m_rel[keep], dtype=torch.float32, device=dev),
    )


class PlaceHeart(nn.Module):
    """The rebuilt heart. ``forward`` gives the (day, cell) rate at or above mc."""

    def __init__(self, bins: HeartBins, net: HeartNet) -> None:
        super().__init__()
        self.bins = bins
        self.net = net
        self.cap = net.cap
        n_ctx = len(CONTEXT) if net.context else 0
        self.register_buffer("ctx_mean", torch.zeros(len(CONTEXT)))
        self.register_buffer("ctx_std", torch.ones(len(CONTEXT)))
        self.register_buffer("age_c", torch.from_numpy(age_centres(bins.age_days)))
        self.register_buffer("static_x", torch.from_numpy(static_coordinates(bins)))
        k = net.state
        self.rank = net.rank if k else 0
        # the kernel: one log weight per bin and the rank directions the state turns it
        self.kernel = _mlp(3 + n_ctx, net.hidden, net.layers)
        if self.rank:
            self.turns = nn.Sequential(
                nn.Linear(3 + n_ctx, net.hidden), nn.GELU(), nn.Linear(net.hidden, self.rank)
            )
        self.background = _mlp(n_ctx, net.hidden, 1) if n_ctx else None
        self.bg0 = nn.Parameter(torch.zeros(()))
        if k:
            self.writer = nn.Sequential(
                nn.Linear(2, net.hidden), nn.GELU(), nn.Linear(net.hidden, k)
            )
            n_in = k + (2 if net.excitement else 0)
            self.mixers = nn.ModuleList(
                [nn.Linear(n_in * (1 + len(NEIGHBOURHOODS)), n_in) for _ in range(net.neighbours)]
            )
            n_out = self.rank + 1 + (1 if net.additive else 0)
            last = nn.Linear(net.hidden, n_out)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.readout = nn.Sequential(nn.Linear(n_in, net.hidden), nn.GELU(), last)
        self.add0 = nn.Parameter(torch.tensor(ADD_START)) if net.additive else None

    # ------------------------------------------------------------ pieces

    def _z(self, context: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.ctx_mean  # type: ignore[assignment]
        std: torch.Tensor = self.ctx_std  # type: ignore[assignment]
        return ((context - mean) / std).float()

    def table(self, context: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(ages, sizes · rings) log weights and (ages, sizes · rings, rank) turns."""
        age_c: torch.Tensor = self.age_c  # type: ignore[assignment]
        static_x: torch.Tensor = self.static_x  # type: ignore[assignment]
        a = age_coordinate(age_c, c, self.net.rulers).float()
        n_a, n_s = len(a), len(static_x)
        x = torch.cat(
            [a[:, None, None].expand(n_a, n_s, 1), static_x[None].expand(n_a, n_s, 2)], -1
        )
        if self.net.context:
            z = self._z(context)
            x = torch.cat([x, z.expand(n_a, n_s, -1)], -1)
        h = self.kernel(x).squeeze(-1)
        u = self.turns(x) if self.rank else None
        return h, u

    def state(
        self,
        dq: DeviceQuakes,
        days: torch.Tensor,
        background: torch.Tensor,
        c: float,
        shape: tuple[int, int],
        excitement: torch.Tensor | None,
    ) -> torch.Tensor:
        """(days, cells, channels) each place's state at each midnight."""
        n_cells = shape[0] * shape[1]
        age = days[:, None] - dq.t[None, :]  # (D, Q)
        on = (age > 0) & (age < STATE_MAX_AGE)
        d_i, q_i = torch.nonzero(on, as_tuple=True)
        la = torch.log10(age[d_i, q_i].clamp(min=1e-4))
        x = torch.stack([age_coordinate(la, c, self.net.rulers).float(), dq.m_rel[q_i] / 2.0], -1)
        w = torch.exp(self.writer(x).clamp(-20.0, 20.0))  # (pairs, K)
        s = torch.zeros(len(days) * n_cells, w.shape[1], device=w.device, dtype=w.dtype)
        s.index_add_(0, d_i * n_cells + dq.cell[q_i], w)
        s = s.view(len(days), n_cells, -1) / background.clamp(min=1e-12)[None, :, None]
        z = torch.log1p(s)
        if excitement is not None:
            z = torch.cat([z, excitement], -1)
        for mix in self.mixers:
            g = z.permute(0, 2, 1).reshape(len(days), -1, *shape)
            reads = [z]
            for k in NEIGHBOURHOODS:
                avg = F.avg_pool2d(g, k, stride=1, padding=k // 2, count_include_pad=False)
                reads.append(avg.reshape(len(days), g.shape[1], -1).permute(0, 2, 1))
            z = z + F.gelu(mix(torch.cat(reads, -1)))
        out: torch.Tensor = self.readout(z)
        return out

    # ------------------------------------------------------------ the rate

    def forward(
        self,
        db: DeviceBox,
        dq: DeviceQuakes | None,
        days: torch.Tensor,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """(days, cells) expected quakes at or above mc from each midnight in ``days``."""
        box = db.box
        bins = self.bins
        n_static = len(bins.magnitude) * len(bins.ruler)
        h, u = self.table(db.context, box.c)
        v = None
        if self.net.state and dq is not None:
            exc = None
            if self.net.excitement:
                with torch.no_grad():
                    zero = torch.zeros(len(bins.age_days) * n_static, device=days.device)
                    base = rates(db, days, zero, torch.zeros((), device=days.device), bins)
                    bg = (box.cascade * db.background * box.span)[None, :]
                    cell = torch.log(base / bg.clamp(min=1e-30))
                    tot = torch.log(base.sum(1, keepdim=True) / bg.sum(1, keepdim=True))
                    exc = torch.stack([cell, tot.expand_as(cell)], -1).float()
            v = self.state(dq, days.double(), db.background, box.c, shape, exc)

        age = days.double()[:, None] - db.t[None, :]  # (D, P)
        seen = age > 0

        def omori(x: torch.Tensor) -> torch.Tensor:
            return 1.0 - (box.c / (x.clamp(min=0.0) + box.c)) ** (box.p - 1)

        w = torch.where(seen, db.prod[None, :] * (omori(age + box.span) - omori(age)), 0.0)
        w = w.float()
        age_c: torch.Tensor = self.age_c  # type: ignore[assignment]
        la = torch.log10(age.clamp(min=1e-6))
        if self.net.smooth_clock:
            lo = (torch.searchsorted(age_c, la.contiguous(), right=True) - 1).clamp(
                0, len(age_c) - 2
            )
            f = ((la - age_c[lo]) / (age_c[lo + 1] - age_c[lo])).clamp(0.0, 1.0).float()
            pairs: list[tuple[torch.Tensor, torch.Tensor | None]] = [(lo, 1.0 - f), (lo + 1, f)]
        else:
            edges = torch.as_tensor(bins.age_days, dtype=torch.float64, device=age.device)
            a_bin = (torch.searchsorted(edges, age.contiguous(), right=True) - 1).clamp(min=0)
            pairs = [(a_bin, None)]
        # Gathers go through index_select: its backward is an atomic add, where indexing a
        # small table with millions of positions backs up through a sort (four times slower).
        h_flat = h.reshape(-1)
        u_cols = (
            [u.reshape(-1, self.rank)[:, r].contiguous() for r in range(self.rank)]
            if (u is not None)
            else []
        )
        v_cells = (
            [v[:, :, r].index_select(1, db.e_cell) for r in range(self.rank)]
            if v is not None and u is not None
            else []
        )

        def take(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return table.index_select(0, idx.reshape(-1)).view(idx.shape)

        log_w = torch.zeros(len(days), len(db.e_parent), device=w.device)
        for a_idx, frac in pairs:
            idx = a_idx.index_select(1, db.e_parent) * n_static + db.e_static[None, :]
            part = take(h_flat, idx)
            for col, vc in zip(u_cols, v_cells, strict=False):
                part = part + take(col, idx) * vc
            log_w = log_w + (part if frac is None else frac.index_select(1, db.e_parent) * part)
        log_w = self.cap * torch.tanh(log_w / self.cap)
        piece = w.index_select(1, db.e_parent) * db.e_share[None, :] * log_w.exp()
        trig = torch.zeros(len(days), box.n_cells, device=w.device, dtype=piece.dtype)
        trig.index_add_(1, db.e_cell, piece)

        b = self.bg0.expand(1, 1)
        if self.background is not None:
            b = b + self.background(self._z(db.context)[None, :])
        if v is not None:
            b = b + v[:, :, self.rank]
        b = self.cap * torch.tanh(b / self.cap)
        bg = db.background[None, :] * box.span * b.exp()
        lam = box.cascade * (bg + trig)
        if self.add0 is not None and v is not None:
            a = self.add0 + self.cap * 2 * torch.tanh(v[:, :, self.rank + 1] / (self.cap * 2))
            lam = lam + box.cascade * db.background[None, :] * box.span * a.exp()
        return lam


def n_parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def describe(net: HeartNet) -> dict[str, Any]:
    """The switches of a rebuilt heart, for its card."""
    return {
        "state": net.state,
        "hears_from": f"mc - {net.state_below_mc:g}",
        "neighbours": net.neighbours,
        "rank": net.rank,
        "rulers": net.rulers,
        "smooth_clock": net.smooth_clock,
        "additive": net.additive,
        "excitement": net.excitement,
        "context": net.context,
    }
