"""Where and when a token is: positional encodings over four axes.

A language model only needs one axis, the position in the sentence. A quake has four:
latitude, longitude, depth and time. Rotary encoding (RoPE, Su et al. 2021) handles them by
turning pairs of features in each attention head by an angle proportional to the coordinate.
When a query and a key are compared, only the *difference* of their angles survives, so
attention sees how far apart two quakes are in space and time, never where the map's origin is.

Each head's features are split into four blocks, one per axis, and each block holds a few
pairs turned at different wavelengths: a minute up to four years for time, 2 km up to 200 km
for the map, 3 km and 30 km for depth. This is the "axial" form of multi-dimensional RoPE
(Heo et al. 2024, arXiv 2403.13298).

A coordinate can be unknown: CB-01 hides where a masked quake was. Its block is then zeroed
instead of turned, so attention between it and another token uses time and content only.

The sinusoidal fallback adds the same waves to the token vector once, before the trunk, the
way the original transformer did with word positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

#: The four axes, in the order of the ``coords`` and ``known`` tensors.
AXES = ("time", "lat", "lon", "depth")


@dataclass(frozen=True)
class AxisWaves:
    """``pairs`` feature pairs for one axis, wavelengths log-spaced from ``shortest`` to
    ``longest`` (in the axis's own unit: seconds, degrees or km)."""

    axis: str
    pairs: int
    shortest: float
    longest: float

    def frequencies(self) -> torch.Tensor:
        """Angular frequency of each pair, radians per unit, float64."""
        if self.pairs == 1:
            waves = torch.tensor([self.shortest], dtype=torch.float64)
        else:
            waves = torch.logspace(
                math.log10(self.shortest),
                math.log10(self.longest),
                self.pairs,
                dtype=torch.float64,
            )
        return 2 * math.pi / waves


def default_waves(head_dim: int) -> tuple[AxisWaves, ...]:
    """The split used by Craton-mini: of every 32 head features, 12 for time, 8 each for
    latitude and longitude, 4 for depth. Wider heads scale every block."""
    if head_dim % 16:
        raise ValueError(f"4D rotary encoding needs a head width divisible by 16, got {head_dim}")
    u = head_dim // 16  # pairs per 1/16 of the head
    return (
        AxisWaves("time", 3 * u, 60.0, 4 * 365.25 * 86_400.0),
        AxisWaves("lat", 2 * u, 0.02, 2.0),
        AxisWaves("lon", 2 * u, 0.02, 2.0),
        AxisWaves("depth", u, 3.0, 30.0),
    )


class Rope4D(nn.Module):
    """Rotary encoding over (time, lat, lon, depth), applied to queries and keys in every layer.

    ``angles`` are computed once per batch in float64, because a minute-long wave over a
    decade-long window turns millions of radians and float32 would lose the phase.
    """

    freqs: torch.Tensor
    axis_of_pair: torch.Tensor

    def __init__(self, head_dim: int, waves: tuple[AxisWaves, ...] | None = None) -> None:
        super().__init__()
        self.waves = waves or default_waves(head_dim)
        if 2 * sum(w.pairs for w in self.waves) != head_dim:
            raise ValueError("the axis blocks must fill the head exactly")
        if tuple(w.axis for w in self.waves) != AXES:
            raise ValueError(f"axes must be {AXES}")
        freqs = torch.cat([w.frequencies() for w in self.waves])
        self.register_buffer("freqs", freqs, persistent=False)
        axis_of_pair = torch.cat(
            [torch.full((w.pairs,), i, dtype=torch.long) for i, w in enumerate(self.waves)]
        )
        self.register_buffer("axis_of_pair", axis_of_pair, persistent=False)

    def tables(
        self, coords: torch.Tensor, known: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos and sin, (batch, 1, tokens, head_dim / 2), zero where the axis is unknown.

        ``coords`` is (batch, tokens, 4) float64 in seconds, degrees, degrees, km; ``known``
        is (batch, tokens, 4) bool.
        """
        pos = coords.to(torch.float64)[..., self.axis_of_pair]  # (B, T, P)
        angle = pos * self.freqs
        gate = known[..., self.axis_of_pair].to(torch.float64)
        cos = (torch.cos(angle) * gate).to(dtype).unsqueeze(1)
        sin = (torch.sin(angle) * gate).to(dtype).unsqueeze(1)
        return cos, sin


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Turn consecutive feature pairs of ``x`` (batch, heads, tokens, head_dim)."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Sinusoidal4D(nn.Module):
    """The fallback: sin and cos of every coordinate at the same wavelengths, projected to the
    model width and added to the token vector once. Unknown axes contribute nothing."""

    freqs: torch.Tensor
    axis_of_pair: torch.Tensor

    def __init__(self, d_model: int, head_dim: int) -> None:
        super().__init__()
        self.waves = default_waves(head_dim)
        freqs = torch.cat([w.frequencies() for w in self.waves])
        self.register_buffer("freqs", freqs, persistent=False)
        axis_of_pair = torch.cat(
            [torch.full((w.pairs,), i, dtype=torch.long) for i, w in enumerate(self.waves)]
        )
        self.register_buffer("axis_of_pair", axis_of_pair, persistent=False)
        self.proj = nn.Linear(2 * len(freqs), d_model, bias=False)

    def forward(self, coords: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        pos = coords.to(torch.float64)[..., self.axis_of_pair]
        angle = pos * self.freqs
        gate = known[..., self.axis_of_pair].to(torch.float64)
        feats = torch.cat([torch.sin(angle) * gate, torch.cos(angle) * gate], dim=-1)
        out: torch.Tensor = self.proj(feats.to(self.proj.weight.dtype))
        return out


POSITIONALS = ("rope4d", "sinusoidal")
