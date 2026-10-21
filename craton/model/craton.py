"""Craton-mini: encoders → 4D positions → trunk → heads, built from a config.

The model reads a window of tokens. A token is a measurement (a quake, a GNSS day, a tide
sample, each encoded by its own instrument's encoder) or a query (a learned vector standing at
a moment, asking "what happens next?"). Where a measurement's channels are hidden, its encoder
is told; where its place is hidden, the positional encoding is told. Then the trunk mixes
everything and the heads answer.

The config lists which instruments the model reads (``model.inputs``). A token of any other
instrument is refused, so an ablation that leaves GNSS out can never see a GNSS day by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from craton.config import CratonConfig
from craton.model.encoders import ENCODERS, PAD, QUERY, instrument_id
from craton.model.heads import GnssHead, MaskedHead, NextDayHead
from craton.model.positional import Rope4D, Sinusoidal4D, default_waves
from craton.model.trunk import Trunk


@dataclass
class Tokens:
    """A batch of token windows, every tensor (batch, tokens, ...).

    instrument (B, T)    long     registry id of the token's instrument; 0 = a query, -1 = pad
    values     (B, T, V) float32  the instrument's channels, first ones first (0 where hidden)
    hidden     (B, T, V) bool     which of those the model may not see
    label      (B, T)    long     a per-instrument label (a quake's magnitude type), else 0
    coords     (B, T, 4) float64  time (s, relative to the window), lat, lon, depth
    known      (B, T, 4) bool     which coordinates the positional encoding may use
    clock      (B, T, 2) float32  log10 of seconds: age before the window's reference moment,
                                  and gap since the previous quake (how busy the fault is)
    padded     bool               whether any window has padding (known on the CPU, so the
                                  model never has to ask the GPU)
    """

    instrument: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    label: torch.Tensor
    coords: torch.Tensor
    known: torch.Tensor
    clock: torch.Tensor
    padded: bool = False

    def to(self, device: torch.device | str) -> Tokens:
        moved = {
            f: getattr(self, f).to(device, non_blocking=True)
            for f in self.__dataclass_fields__
            if f != "padded"
        }
        return Tokens(**moved, padded=self.padded)

    def pin_memory(self) -> Tokens:
        pinned = {
            f: getattr(self, f).pin_memory() for f in self.__dataclass_fields__ if f != "padded"
        }
        return Tokens(**pinned, padded=self.padded)

    @property
    def query(self) -> torch.Tensor:
        return self.instrument == QUERY

    @property
    def keep(self) -> torch.Tensor | None:
        """(B, T) bool of real tokens, or None when no window has padding."""
        return self.instrument != PAD if self.padded else None

    def of(self, name: str) -> torch.Tensor:
        """(B, T) bool: the tokens of one instrument."""
        return self.instrument == instrument_id(name)

    @property
    def shape(self) -> tuple[int, int]:
        b, t = self.instrument.shape
        return int(b), int(t)


class CratonModel(nn.Module):
    def __init__(self, config: CratonConfig) -> None:
        super().__init__()
        m = config.model
        self.config = config
        unknown = [name for name in m.inputs if name not in ENCODERS]
        if unknown:
            raise ValueError(f"no encoder registered for {unknown}")
        self.encoders = nn.ModuleDict({name: ENCODERS[name].build(m.d_model) for name in m.inputs})
        # One learned vector per instrument: which kind of token this is (modality embedding).
        self.modality = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(m.d_model) * 0.02) for name in m.inputs}
        )
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
        self.gnss = GnssHead(m.d_model)
        self.next_day = NextDayHead(m.d_model)
        self.apply(_init)
        self.next_day._start_from_persistence()
        self.gnss._start_small()

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.encoders.keys())

    def embed(self, tok: Tokens) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Every encoder runs over every token and each token keeps its own instrument's
        answer: a few percent more arithmetic in small networks, and no step where the CPU
        has to wait for the GPU to say which tokens are which."""
        b, t = tok.shape
        d = self.query_vec.shape[0]
        x = torch.zeros(b, t, d, device=tok.values.device, dtype=torch.float32)
        if not self.training:  # a check that needs the GPU's answer: outside training only
            seen = tok.query | (tok.instrument == PAD)
            for name in self.encoders:
                seen = seen | tok.of(name)
            if not bool(seen.all()):
                raise ValueError("a token's instrument is not one this model reads")
        for name, enc in self.encoders.items():
            k = ENCODERS[name].width
            out = enc(tok.values[..., :k], tok.hidden[..., :k], tok.label) + self.modality[name]
            x = torch.where(tok.of(name).unsqueeze(-1), out.to(x.dtype), x)
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
        out: torch.Tensor = self.trunk(x, rope, tok.keep)
        return out

    def attention(self, tok: Tokens, layer: int) -> torch.Tensor:
        x, rope = self.embed(tok)
        return self.trunk.attention(x, rope, layer, tok.keep)

    def count(self) -> dict[str, int]:
        """Parameters per part: the config's trunk claim, plus encoders and heads."""

        def n(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters())

        return {
            "trunk": n(self.trunk),
            "encoder": n(self.encoders)
            + n(self.modality)
            + n(self.clock)
            + self.query_vec.numel()
            + (n(self.sinusoidal) if self.sinusoidal is not None else 0),
            "masked_head": n(self.masked),
            "gnss_head": n(self.gnss),
            "next_day_head": n(self.next_day),
            "total": n(self),
        }


def _init(mod: nn.Module) -> None:
    if isinstance(mod, nn.Linear):
        nn.init.normal_(mod.weight, std=0.02)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)
