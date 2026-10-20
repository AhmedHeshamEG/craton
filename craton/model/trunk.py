"""The trunk: a stack of pre-norm transformer layers, the same block GPT-2 uses.

Each layer lets every token look at every other token (attention), then thinks about what it
saw on its own (a two-layer feed-forward network). Layer norm comes first in each half and a
residual connection adds the result back, so a deep stack stays trainable. Attention runs on
PyTorch's fused ``scaled_dot_product_attention``, which picks a memory-efficient kernel on
the RTX 2060.

The trunk knows nothing about earthquakes. Encoders bring tokens in, heads take answers out.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from craton.model.positional import rotate


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _qkv(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = rotate(q, *rope), rotate(k, *rope)
        return q, k, v

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self._qkv(x, rope)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out: torch.Tensor = self.out(y.transpose(1, 2).reshape(b, t, d))
        return out

    def weights(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        """Attention probabilities (batch, heads, tokens, tokens), for figures and probes."""
        q, k, _ = self._qkv(x, rope)
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        return torch.softmax(scores, dim=-1)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), rope))
        out: torch.Tensor = x + self.drop(self.ff(self.norm2(x)))
        return out


class Trunk(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, rope)
        out: torch.Tensor = self.norm(x)
        return out

    def attention(
        self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, layer: int
    ) -> torch.Tensor:
        """Run up to ``layer`` and return that layer's attention probabilities."""
        for block in self.blocks[:layer]:
            x = block(x, rope)
        blk = self.blocks[layer]
        assert isinstance(blk, Block)
        return blk.attn.weights(blk.norm1(x), rope)
