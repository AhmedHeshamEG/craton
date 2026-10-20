"""Encoders: one per instrument, each turning that instrument's numbers into a token vector.

This is the geology version of a word embedding. A word is looked up in a table; a quake is a
handful of continuous numbers (where, how deep, how big) plus a few labels (the magnitude
type), so a small network maps them into the model's d-dimensional space instead.

Every encoder is registered under the instrument it reads. P5 adds GNSS and tides by
registering two more; nothing here changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

#: Magnitude types the catalogue encoder knows. Anything else shares the last slot.
MAGNITUDE_TYPES = ("ML", "Md", "Mw", "Mb", "Ms", "Mwp", "other")
#: How the four event channels are scaled before the network sees them.
EVENT_CENTRE = (42.5, 13.25, 10.0, 2.0)  # lat °, lon °, depth km, magnitude
EVENT_SCALE = (1.5, 1.75, 10.0, 1.0)
EVENT_CHANNELS = ("lat", "lon", "depth_km", "magnitude")


def magnitude_type_id(name: str) -> int:
    return MAGNITUDE_TYPES.index(name) if name in MAGNITUDE_TYPES else len(MAGNITUDE_TYPES) - 1


@dataclass(frozen=True)
class EncoderSpec:
    instrument: str
    channels: tuple[str, ...]
    build: Callable[[int], nn.Module]


class EventEncoder(nn.Module):
    """``seismic_catalog`` tokens: (lat, lon, depth, magnitude) and the magnitude type.

    A hidden channel is replaced by zero, and a learned "this was hidden" vector per channel
    is added instead, so the network can tell a hidden depth from a depth of 10 km.
    """

    centre: torch.Tensor
    scale: torch.Tensor

    def __init__(self, d_model: int) -> None:
        super().__init__()
        n = len(EVENT_CHANNELS)
        self.register_buffer("centre", torch.tensor(EVENT_CENTRE), persistent=False)
        self.register_buffer("scale", torch.tensor(EVENT_SCALE), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(n, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.hidden = nn.Parameter(torch.zeros(n, d_model))
        self.mag_type = nn.Embedding(len(MAGNITUDE_TYPES), d_model)
        nn.init.normal_(self.hidden, std=0.02)
        nn.init.normal_(self.mag_type.weight, std=0.02)

    def forward(
        self, values: torch.Tensor, hidden: torch.Tensor, mag_type: torch.Tensor
    ) -> torch.Tensor:
        """values (B, T, 4) float, hidden (B, T, 4) bool, mag_type (B, T) long → (B, T, d)."""
        x = (values - self.centre) / self.scale
        x = torch.where(hidden, torch.zeros_like(x), x).clamp(-10.0, 10.0)
        out: torch.Tensor = self.mlp(x) + hidden.to(x.dtype) @ self.hidden
        emb: torch.Tensor = self.mag_type(mag_type)
        return out + emb


ENCODERS: dict[str, EncoderSpec] = {}


def register_encoder(spec: EncoderSpec) -> EncoderSpec:
    """Teach the model a new instrument. Names are unique and never redefined."""
    if spec.instrument in ENCODERS:
        raise ValueError(f"an encoder for {spec.instrument!r} is already registered")
    ENCODERS[spec.instrument] = spec
    return spec


register_encoder(EncoderSpec("seismic_catalog", EVENT_CHANNELS, EventEncoder))
