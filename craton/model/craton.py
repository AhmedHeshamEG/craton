"""Craton-mini v0: encoder → 4D positions → trunk → heads, built from a config.

The model reads a window of tokens sorted by time. A token is a quake (encoded from its
numbers) or a query (a learned vector standing at a moment, asking "what happens next?").
Where a quake's channels are hidden, the encoder is told; where its place is hidden, the
positional encoding is told. Then the trunk mixes everything and the heads answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from craton.config import CratonConfig
from craton.model.encoders import ENCODERS
from craton.model.heads import MaskedHead, NextDayHead
from craton.model.positional import Rope4D, Sinusoidal4D, default_waves
from craton.model.trunk import Trunk


@dataclass
class Tokens:
    """A batch of token windows, every tensor (batch, tokens, ...).

    values    (B, T, 4) float32  lat, lon, depth, magnitude (anything where hidden)
    hidden    (B, T, 4) bool     which of those the model may not see
    mag_type  (B, T)    long     magnitude type id
    coords    (B, T, 4) float64  time (s, relative to the window), lat, lon, depth
    known     (B, T, 4) bool     which coordinates the positional encoding may use
    query     (B, T)    bool     query tokens (no measurement, just a moment)
    clock     (B, T, 2) float32  log10 of seconds: age before the window's reference moment,
                                 and gap since the previous quake (how busy the fault is)
    """

    values: torch.Tensor
    hidden: torch.Tensor
    mag_type: torch.Tensor
    coords: torch.Tensor
    known: torch.Tensor
    query: torch.Tensor
    clock: torch.Tensor

    def to(self, device: torch.device | str) -> Tokens:
        return Tokens(
            *(getattr(self, f).to(device, non_blocking=True) for f in self.__dataclass_fields__)
        )

    @property
    def shape(self) -> tuple[int, int]:
        b, t = self.query.shape
        return int(b), int(t)


class CratonModel(nn.Module):
    def __init__(self, config: CratonConfig) -> None:
        super().__init__()
        m = config.model
        self.config = config
        self.encoder = ENCODERS["seismic_catalog"].build(m.d_model)
        self.query_vec = nn.Parameter(torch.randn(m.d_model) * 0.02)
        self.clock = nn.Sequential(
            nn.Linear(2, m.d_model), nn.GELU(), nn.Linear(m.d_model, m.d_model)
        )
        head_dim = m.d_model // m.n_heads
        self.rope: Rope4D | None = None
        self.sinusoidal: Sinusoidal4D | None = None
        if m.positional == "rope4d":
            self.rope = Rope4D(head_dim, default_waves(head_dim))
        else:
            self.sinusoidal = Sinusoidal4D(m.d_model, head_dim)
        self.drop = nn.Dropout(m.dropout)
        self.trunk = Trunk(m.d_model, m.n_layers, m.n_heads, m.ff_width, m.dropout)
        self.masked = MaskedHead(m.d_model)
        self.next_day = NextDayHead(m.d_model)
        self.apply(_init)
        self.next_day._start_from_persistence()

    def embed(self, tok: Tokens) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        x = self.encoder(tok.values, tok.hidden, tok.mag_type)
        x = torch.where(tok.query.unsqueeze(-1), self.query_vec.to(x.dtype), x)
        x = x + self.clock(tok.clock)
        if self.sinusoidal is not None:
            x = x + self.sinusoidal(tok.coords, tok.known)
        rope = None
        if self.rope is not None:
            rope = self.rope.tables(tok.coords, tok.known, torch.float32)
        return self.drop(x), rope

    def forward(self, tok: Tokens) -> torch.Tensor:
        """Hidden states (B, T, d) after the trunk."""
        x, rope = self.embed(tok)
        out: torch.Tensor = self.trunk(x, rope)
        return out

    def attention(self, tok: Tokens, layer: int) -> torch.Tensor:
        x, rope = self.embed(tok)
        return self.trunk.attention(x, rope, layer)

    def count(self) -> dict[str, int]:
        """Parameters per part: the config's trunk claim, plus encoders and heads."""

        def n(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters())

        return {
            "trunk": n(self.trunk),
            "encoder": n(self.encoder)
            + n(self.clock)
            + self.query_vec.numel()
            + (n(self.sinusoidal) if self.sinusoidal is not None else 0),
            "masked_head": n(self.masked),
            "next_day_head": n(self.next_day),
            "total": n(self),
        }


def _init(mod: nn.Module) -> None:
    if isinstance(mod, nn.Linear):
        nn.init.normal_(mod.weight, std=0.02)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
