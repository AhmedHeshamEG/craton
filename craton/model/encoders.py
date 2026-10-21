"""Encoders: one per instrument, each turning that instrument's numbers into a token vector.

This is the geology version of a word embedding. A word is looked up in a table; a quake is a
handful of continuous numbers (where, how deep, how big) plus a few labels (the magnitude
type), so a small network maps them into the model's d-dimensional space instead.

Every encoder is registered under the instrument it reads, and the registry only grows: an
instrument's id is its place in the registry, so it never changes once given. A model reads
the instruments its config lists and nothing else. Every token carries up to
``VALUE_WIDTH`` numbers; each encoder reads the first ``len(channels)`` of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

#: Numbers a token can carry into the model. Each encoder reads its own first few.
VALUE_WIDTH = 12
#: Instrument id 0 is not an instrument: it marks a query token (a moment, asking a question).
QUERY = 0
#: Id -1 is padding: a window shorter than the context is filled with it, and nothing attends to it.
PAD = -1

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

    @property
    def width(self) -> int:
        return len(self.channels)


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
        self, values: torch.Tensor, hidden: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """values (..., 4) float, hidden (..., 4) bool, label (...) magnitude type → (..., d)."""
        x = (values - self.centre) / self.scale
        x = torch.where(hidden, torch.zeros_like(x), x).clamp(-10.0, 10.0)
        out: torch.Tensor = self.mlp(x) + hidden.to(x.dtype) @ self.hidden
        emb: torch.Tensor = self.mag_type(label)
        return out + emb


class ScalarEncoder(nn.Module):
    """Any instrument whose channels arrive already scaled to about ±1: a small network over
    the visible channels, plus a learned "this was hidden" vector per hidden channel."""

    def __init__(self, d_model: int, width: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.hidden = nn.Parameter(torch.zeros(width, d_model))
        nn.init.normal_(self.hidden, std=0.02)

    def forward(
        self, values: torch.Tensor, hidden: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        x = torch.where(hidden, torch.zeros_like(values), values).clamp(-10.0, 10.0)
        out: torch.Tensor = self.mlp(x) + hidden.to(x.dtype) @ self.hidden
        return out


#: A GNSS day as the model reads it (``craton.train.context`` computes these, all scaled):
#: the day's distance from the station's own long-term line in cm (hidden when the day is
#: masked), how far the station's visible days in this window sit from that line (signed
#: log of mm), whether it has any visible day here, whether its line came from its own data
#: or its neighbours', the line's velocity (cm/yr) and the years from 2015 to the day.
GNSS_CHANNELS = (
    "east",
    "north",
    "up",
    "level_east",
    "level_north",
    "level_up",
    "anchored",
    "own_line",
    "v_east",
    "v_north",
    "v_up",
    "years",
)
#: A solid-Earth tide sample: east, north, up displacement in decimetres.
TIDE_CHANNELS = ("east", "north", "up")


ENCODERS: dict[str, EncoderSpec] = {}


def register_encoder(spec: EncoderSpec) -> EncoderSpec:
    """Teach the model a new instrument. Names are unique and never redefined."""
    if spec.instrument in ENCODERS:
        raise ValueError(f"an encoder for {spec.instrument!r} is already registered")
    if spec.width > VALUE_WIDTH:
        raise ValueError(f"{spec.instrument}: {spec.width} channels, at most {VALUE_WIDTH}")
    ENCODERS[spec.instrument] = spec
    return spec


def instrument_id(name: str) -> int:
    """The id a token of this instrument carries: its place in the registry, from 1."""
    return list(ENCODERS).index(name) + 1


register_encoder(EncoderSpec("seismic_catalog", EVENT_CHANNELS, EventEncoder))
register_encoder(
    EncoderSpec("gnss_daily", GNSS_CHANNELS, lambda d: ScalarEncoder(d, len(GNSS_CHANNELS)))
)
register_encoder(
    EncoderSpec("solid_earth_tide", TIDE_CHANNELS, lambda d: ScalarEncoder(d, len(TIDE_CHANNELS)))
)
