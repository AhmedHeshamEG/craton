"""The hidden-state layer (P9c): two blocks in front of the Field, each starting as the physics.

**Clock-shift memory (selective Omori).** The memory's counters fade with fixed lifetimes,
from an hour to twelve years, four times apart. In rate-and-state friction (Dieterich 1994) an
aftershock sequence lasts ``t_a = Aσ / τ̇``: where the crust is loaded faster, the same
sequence plays out faster, as if time ran faster there. Speeding a clock up by 4^s turns a
counter of lifetime τ into one of lifetime τ / 4^s, and on a bank spaced four times apart
that is a slide of ``s`` steps along the bank (SITHCon, Jacques et al. 2021: rescaling time
is an index shift). So one learned number per cell per day, ``s``, read off that cell's own
memory, slides its counters (linear in between, the ends held). ``s = 0`` is the fixed P6
memory; ``s`` is a stressing-rate meter, in steps of 4x. It is Mamba's trick, decay that
depends on the input, with the decay constrained to mean one physical thing.

**Steerable kernel.** The neighbourhood counters blur past quakes with a round kernel. Here
each cell gets a complex number ``w`` (``|w| < e_max``) that stretches its kernel to
``1 + |w| cos 2(φ − θ)``, θ = arg(w) / 2, through the quadrupole moments of
``craton.field.steer``: ``log(I + Re(w̄ Q)) = log I + log(1 + Re(w̄ q))``. ``w = 0`` is the round
kernel. ``w`` comes from an **equivariant** head: complex weights, chosen by inputs that don't
change when the catalogue is rotated (the cell's own counters and the sizes |q|), multiply
the moments q, which turn with it. Rotate every quake by α about a cell and its θ turns by α,
exactly. Nothing in the head knows where north is, so θ can only be read off the quakes, and
that is what CB-23 grades as SHmax, with no head trained for it.

Both blocks' last layers start at zero: the untrained hidden Field computes the same thing
as the untrained P6 Field (``tests/test_hidden.py``).
"""

from __future__ import annotations

import torch
from torch import nn

from craton.bench.csep import Grid
from craton.field.config import FieldHidden, FieldShape
from craton.field.memory import LIFETIMES_DAYS, SPREADS
from craton.field.model import FieldNet

K = len(LIFETIMES_DAYS)
W = 2
S = len(SPREADS)
HEAD = 32


def _zero_last(seq: nn.Sequential) -> nn.Sequential:
    last = seq[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return seq


def shift_bank(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """x (..., K, W) along the lifetime axis, read at index k + s (s: (...)), linear in
    between, held at the ends. ``s = 0`` returns x exactly."""
    k = torch.arange(K, device=x.device, dtype=s.dtype)
    pos = (k + s[..., None]).clamp(0, K - 1)  # (..., K)
    lo = pos.floor().clamp(max=K - 2)
    frac = (pos - lo)[..., None].to(x.dtype)
    lo_i = lo.long()[..., None].expand(*lo.shape, x.shape[-1])
    a = torch.gather(x, -2, lo_i)
    b = torch.gather(x, -2, lo_i + 1)
    return a * (1 - frac) + b * frac  # exact at both ends: s = 0 is the bank, bit for bit


class HiddenFieldNet(nn.Module):
    """The P6 Field with the hidden-state blocks in front. ``quad`` is
    ``craton.field.steer.quadrupole``'s output, needed when the kernel steers."""

    def __init__(self, n_memory: int, grid: Grid, shape: FieldShape, hidden: FieldHidden) -> None:
        super().__init__()
        if n_memory != S * K * W:
            raise ValueError(f"the hidden blocks read the standard memory ({S * K * W} features)")
        self.hidden = hidden
        self.field = FieldNet(n_memory, grid, shape)
        if hidden.clock:
            self.clock = _zero_last(
                nn.Sequential(nn.Linear(n_memory + 1, HEAD), nn.GELU(), nn.Linear(HEAD, 1))
            )
        if hidden.steer:
            # Invariants: the cell's own counters and |q|; out: a complex weight per counter.
            self.steer = _zero_last(
                nn.Sequential(nn.Linear(2 * K * W, HEAD), nn.GELU(), nn.Linear(HEAD, 2 * K * W))
            )

    def shift(self, memory: torch.Tensor, log_etas: torch.Tensor) -> torch.Tensor:
        """(days, cells) the clock shift, in steps of the lifetime bank."""
        m = self.hidden.max_shift
        x = torch.cat([memory, log_etas[..., None]], dim=-1)
        return m * torch.tanh(self.clock(x)[..., 0].float() / m)

    def stretch(self, memory: torch.Tensor, quad: torch.Tensor) -> torch.Tensor:
        """(days, cells, 2) w as (Re, Im): the kernel's stretch, |w| < max_stretch."""
        own = memory[..., : K * W]
        q = quad.float()
        size = q.norm(dim=-1)
        c = self.steer(torch.cat([own, size], dim=-1)).float()
        c = c.reshape(*c.shape[:-1], K * W, 2)
        a, b = c[..., 0], c[..., 1]
        q1, q2 = q[..., 0], q[..., 1]
        v = torch.stack([(a * q1 - b * q2).sum(-1), (a * q2 + b * q1).sum(-1)], dim=-1)
        w: torch.Tensor = self.hidden.max_stretch * v / (1.0 + v.norm(dim=-1, keepdim=True))
        return w

    def blocks(
        self, memory: torch.Tensor, log_etas: torch.Tensor, quad: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The memory the Field reads, the clock shift and the stretch (None if off)."""
        mem = memory.float()
        s = w = None
        if self.hidden.steer:
            if quad is None:
                raise ValueError("the steerable kernel needs the quadrupole moments")
            w = self.stretch(mem, quad)
        bank = mem.reshape(*mem.shape[:-1], S, K, W)
        if self.hidden.clock:
            s = self.shift(mem, log_etas.float())
            bank = shift_bank(bank, s[..., None].expand(*s.shape, S))
        if w is not None:
            assert quad is not None
            q = quad.float().reshape(*quad.shape[:-2], K, W, 2)
            if s is not None:
                q = torch.stack([shift_bank(q[..., 0], s), shift_bank(q[..., 1], s)], dim=-1)
            dot = w[..., None, None, 0] * q[..., 0] + w[..., None, None, 1] * q[..., 1]
            bump = torch.log1p(dot.clamp(min=-0.99))  # |w| < 1 and |q| ≤ 1 keep it > -1
            bank = torch.cat([bank[..., :1, :, :], bank[..., 1:, :, :] + bump[..., None, :, :]], -3)
        return bank.flatten(-3), s, w

    def forward(
        self, memory: torch.Tensor, log_etas: torch.Tensor, quad: torch.Tensor | None = None
    ) -> torch.Tensor:
        mem, _, _ = self.blocks(memory, log_etas, quad)
        out: torch.Tensor = self.field(mem, log_etas)
        return out


def make_net(n_memory: int, grid: Grid, shape: FieldShape, hidden: FieldHidden | None) -> nn.Module:
    """The Field a config asks for: P6's, or P6's with the hidden-state blocks."""
    if hidden is None or not (hidden.clock or hidden.steer):
        return FieldNet(n_memory, grid, shape)
    return HiddenFieldNet(n_memory, grid, shape, hidden)
