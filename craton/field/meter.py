"""The stress meter (P9f): the Field's only output is a stress, and rates come from it through
rate-and-state.

Dieterich (1994) turned seismicity into a stress meter: a population of faults under steady
loading makes quakes at its background rate r, and a Coulomb stress history S(t) moves that
rate in one exact way. Written per day, with the stress held through each day (Heimisson
and Segall 2018 give the continuous form):

    count_d = u · ln(1 + exp(x_d − L_d) · Δ / t_a),   L_{d+1} = L_d + count_d / u,

where x = S / Aσ is the stress in units of the faults' direct effect, t_a = Aσ / τ̇ the
aftershock duration, u = r · t_a the quakes the background makes in one t_a, and L the
population already used up. Two things follow that a free correction never has to obey:

- **One stress, every magnitude.** Rate-and-state moves the rate of all sizes alike, so the
  M2-3 and M3+ rates of a cell move by the same factor (the P6 Field moves them apart).
- **Quakes use up their faults.** Every quake the model expects beyond ETAS's stays in L: a
  cell that was pushed hard keeps a deficit, and its rate sits below ETAS's afterwards unless
  the stress keeps rising. The deficit fades over about t_a as ETAS's own quakes reload it.

**ETAS is the starting line.** ETAS's own rates are a stress history too (invert the law
above). Measured from that history, the Field outputs a stress anomaly δ per cell per day,
and with D the model's quakes beyond ETAS's so far (its deficit),

    count = u · ln(1 + (e^{λ_E / u} − 1) · e^{δ − D / u}),

which is λ_E exactly at δ = 0 and D = 0, for any t_a. The last layer starts at zero, so the
untrained meter Field *is* ETAS, the Field's rule since P6. Each cell's t_a is learned,
starting at ``t_a_years``, within a factor ``t_a_factor`` either way.

D needs every earlier day's output, which a random batch of days doesn't have. So D is read
off a whole pass over the training days (``scan``), redone after every epoch, with no
gradient through it: the network learns a stress for each day given the deficit its last
pass left there. A checkpoint carries the deficit at the end of its training days into the
test window (``carry``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from craton.bench.csep import Grid
from craton.field.config import FieldMeter, FieldShape
from craton.field.model import FieldNet

YEAR_DAYS = 365.25


def log_expm1(a: torch.Tensor) -> torch.Tensor:
    """ln(e^a − 1) for a > 0, stable from 1e-30 to 1e4."""
    big = a > 1.0
    safe_big = torch.where(big, a, torch.ones_like(a))
    safe_small = torch.where(big, torch.ones_like(a), a)
    return torch.where(
        big, safe_big + torch.log1p(-torch.exp(-safe_big)), torch.log(torch.expm1(safe_small))
    )


def meter_log_rate(
    log_etas: torch.Tensor, delta: torch.Tensor, deficit: torch.Tensor, log_u: torch.Tensor
) -> torch.Tensor:
    """log of the day's expected count: ETAS's (``log_etas``, every magnitude at or above mc)
    through rate-and-state with stress anomaly ``delta`` (units of Aσ) and ``deficit`` quakes
    beyond ETAS's so far; ``log_u`` = ln(r · t_a). Exactly ``log_etas`` at delta = deficit = 0."""
    u = log_u.exp()
    a = (log_etas - log_u).exp().clamp(min=1e-30)
    v = log_expm1(a) + delta - deficit / u
    return log_u + torch.log(nn.functional.softplus(v).clamp(min=1e-38))


class MeterFieldNet(nn.Module):
    """The P6 Field's cells and attention, one stress out of each cell, rates through the meter.
    ``log_r`` is ETAS's background, every magnitude at or above mc, per cell per day."""

    def __init__(
        self, n_memory: int, grid: Grid, shape: FieldShape, meter: FieldMeter, log_r: np.ndarray
    ) -> None:
        super().__init__()
        self.meter = meter
        self.field = FieldNet(n_memory, grid, shape)
        self.stress = nn.Linear(shape.d_model, 1)
        nn.init.zeros_(self.stress.weight)  # the untrained meter is ETAS exactly
        nn.init.zeros_(self.stress.bias)
        self.log_ta_shift = nn.Parameter(torch.zeros(grid.n_cells))
        self.register_buffer("log_r", torch.as_tensor(log_r, dtype=torch.float32))
        #: The deficit at the end of the training days, carried into the test window.
        self.register_buffer("carry", torch.zeros(grid.n_cells))

    def log_t_a(self) -> torch.Tensor:
        """(cells,) ln t_a in days."""
        f = math.log(self.meter.t_a_factor)
        base = math.log(self.meter.t_a_years * YEAR_DAYS)
        return base + f * torch.tanh(self.log_ta_shift / f)

    def log_u(self) -> torch.Tensor:
        log_r: torch.Tensor = self.log_r
        return log_r + self.log_t_a()

    def delta(self, memory: torch.Tensor, log_etas3: torch.Tensor) -> torch.Tensor:
        """(days, cells) the stress anomaly over ETAS's, in units of Aσ, bounded."""
        h = self.field.encode(memory, log_etas3)
        c = self.stress(h)[..., 0].float()
        m = self.meter.max_stress
        return m * torch.tanh(c / m)

    def correction(
        self, delta: torch.Tensor, log_total: torch.Tensor, deficit: torch.Tensor
    ) -> torch.Tensor:
        """(days, cells, 2) the log factor on ETAS's M2-3 and M3+ rates: one, for both."""
        if not self.meter.deplete:
            deficit = torch.zeros_like(deficit)
        lam = meter_log_rate(log_total.float(), delta, deficit.float(), self.log_u())
        c = lam - log_total.float()
        return torch.stack([c, c], dim=-1)

    def forward(
        self,
        memory: torch.Tensor,
        log_etas3: torch.Tensor,
        log_total: torch.Tensor,
        deficit: torch.Tensor,
    ) -> torch.Tensor:
        return self.correction(self.delta(memory, log_etas3), log_total, deficit)


@torch.no_grad()
def scan(
    net: MeterFieldNet, delta: torch.Tensor, log_total: torch.Tensor, start: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk consecutive days in order: (days, cells) the deficit each day starts with, and the
    deficit after the last day. ``delta`` and ``log_total`` are (days, cells)."""
    log_u = net.log_u().to(delta.device)
    d = start.to(delta.device, torch.float32).clone()
    if not net.meter.deplete:
        return torch.zeros_like(delta, dtype=torch.float32), d
    out = torch.empty_like(delta, dtype=torch.float32)
    lt = log_total.float()
    for i in range(delta.shape[0]):
        out[i] = d
        lam = meter_log_rate(lt[i], delta[i], d, log_u).exp()
        d = d + (lam - lt[i].exp())
    return out, d


def deltas(
    net: MeterFieldNet,
    memory: np.ndarray,
    log_etas3: np.ndarray,
    dev: torch.device,
    batch: int = 16,
) -> torch.Tensor:
    """(days, cells) the net's stress anomaly for every day, in eval mode, on ``dev``."""
    was = net.training
    net.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(memory), batch):
            mem = torch.from_numpy(memory[i : i + batch]).to(dev).float()
            lb = torch.from_numpy(log_etas3[i : i + batch]).to(dev)
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                parts.append(net.delta(mem, lb).float())
    net.train(was)
    return torch.cat(parts)
