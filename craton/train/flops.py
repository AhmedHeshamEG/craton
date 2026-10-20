"""How much arithmetic a training token costs, and how much of the GPU a run used.

Model FLOPs utilisation (MFU, from the PaLM paper, Chowdhery et al. 2022, appendix B) is the
LLM world's efficiency number: the FLOPs a run *needed* per second, divided by the most the
chip can do. A token costs 6N FLOPs for the weights (2N forward, 4N backward, N = trunk
parameters) plus 12·L·H·Q·T for attention (L layers, H heads, Q head width, T context).
Recomputed activations don't count; only the work the model itself requires.

Unlike tokens per second or peak memory, MFU doesn't get worse just because a model gets
bigger, so the ratchet can guard it without freezing the model's size.
"""

from __future__ import annotations

from craton.config import CratonConfig, trunk_params

#: Dense fp16 tensor-core peak with fp32 accumulation (what autocast matmuls use), in FLOP/s.
#: RTX 2060: 240 tensor cores x 64 FMA x 2 FLOP x 1.68 GHz boost = 51.6 T with fp16
#: accumulation; GeForce Turing runs fp32 accumulation at half rate, so 25.8 T.
PEAK_FLOPS = {"NVIDIA GeForce RTX 2060": 25.8e12}


def flops_per_token(config: CratonConfig) -> float:
    m = config.model
    head = m.d_model // m.n_heads
    return float(6 * trunk_params(m) + 12 * m.n_layers * m.n_heads * head * m.context_tokens)


def mfu(tokens_per_s: float, config: CratonConfig, device: str) -> float:
    """Share of the device's peak, 0 to 1. Raises for a device with no published peak."""
    if device not in PEAK_FLOPS:
        raise ValueError(f"no peak FLOP/s on record for {device!r}; add it to PEAK_FLOPS")
    return tokens_per_s * flops_per_token(config) / PEAK_FLOPS[device]
