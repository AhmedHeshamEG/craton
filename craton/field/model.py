"""The Field: the 1,050 forecast cells as tokens, starting from ETAS.

At each midnight every 0.1° cell becomes a token carrying its Omori memory
(``craton.field.memory``), ETAS's log rate for it, the whole box's ETAS log rate and its place
(a sinusoidal code of latitude and longitude). Attention between cells lets a cell read its
neighbours' memory, the way a vision transformer reads patches. Out of each cell come two
numbers: how much to move ETAS's rate of M2-3 quakes there, and of M3+ quakes.

**ETAS is the starting line.** The output layer starts at zero, so the untrained Field *is*
ETAS. The correction is bounded smoothly, ``ln(max_nudge) · tanh(c / ln(max_nudge))``: the
Field can move ETAS at most ``max_nudge`` times either way in any cell on any day. This is
residual learning on a strong base, the way fine-tuning starts from a model that already
works.

Small quakes in, big quakes scored: the memory counts every quake down to M1.5 and the loss
learns from M2+, but only the M3+ output reaches CB-02 and CB-03.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from craton.bench.csep import Grid
from craton.field.config import FieldShape

#: The two outputs per cell: M2-3 and M3+ (the scored one).
OUTPUTS = ("m2_3", "m3_up")
POSITION_FREQUENCIES = 8


def place_code(grid: Grid) -> torch.Tensor:
    """(cells, 4·POSITION_FREQUENCIES) sines and cosines of each cell centre's place."""
    lat, lon = grid.cell_centres()
    freq = 2.0 ** torch.arange(POSITION_FREQUENCIES).float()
    parts = []
    for v, lo, span in (
        (lat, grid.lat_min, grid.lat_max - grid.lat_min),
        (lon, grid.lon_min, grid.lon_max - grid.lon_min),
    ):
        x = torch.tensor((v - lo) / span).float()[:, None] * math.pi * freq[None]
        parts += [torch.sin(x), torch.cos(x)]
    return torch.cat(parts, dim=1)


class FieldNet(nn.Module):
    def __init__(self, n_memory: int, grid: Grid, shape: FieldShape) -> None:
        super().__init__()
        d = shape.d_model
        self.max_log = math.log(shape.max_nudge)
        self.inp = nn.Sequential(nn.Linear(n_memory + 2, d), nn.GELU(), nn.Linear(d, d))
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

    def forward(self, memory: torch.Tensor, log_etas: torch.Tensor) -> torch.Tensor:
        """memory (days, cells, F), log_etas (days, cells) ETAS's log M3+ rate →
        (days, cells, 2) bounded log corrections for M2-3 and M3+."""
        total = log_etas.logsumexp(-1, keepdim=True).expand_as(log_etas)
        x = torch.cat([memory, log_etas[..., None], total[..., None]], dim=-1)
        h = self.inp(x) + self.place_proj(self.place)[None]
        if self.cells is not None:
            h = self.cells(h)
        c = self.out(h).float()
        return self.max_log * torch.tanh(c / self.max_log)
