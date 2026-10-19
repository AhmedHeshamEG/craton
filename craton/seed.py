"""Seeds and determinism.

Every benchmark number in Craton is a mean over three seeds, and every run has to be
repeatable by anyone with one command. That only works if one call fixes every source of
randomness: Python's, NumPy's and PyTorch's, on the CPU and the GPU.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

#: The three seeds every CratonBench score is averaged over.
BENCH_SEEDS: tuple[int, int, int] = (0, 1, 2)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (if installed), and optionally force deterministic kernels.

    Deterministic mode makes cuDNN and cuBLAS pick reproducible algorithms. It can cost some
    speed; benchmarks always use it.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this workspace setting to be deterministic (CUDA >= 10.2).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base: int, *names: str | int) -> int:
    """A stable child seed from a base seed and a path of names.

    ``derive_seed(0, "ensemble", 3)`` is the same number on every machine and every Python
    version (unlike ``hash()``), so ensemble members, data workers and synthetic planets each
    get their own stream without colliding.
    """
    text = "/".join([str(base), *map(str, names)])
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def rng(seed: int, *names: str | int) -> np.random.Generator:
    """A NumPy generator for one named purpose. Prefer this over the global NumPy state."""
    return np.random.default_rng(derive_seed(seed, *names) if names else seed)
