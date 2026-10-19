import random

import numpy as np
import pytest

from craton.seed import BENCH_SEEDS, derive_seed, rng, seed_everything


def draw() -> tuple[float, float, list[float]]:
    out: list[float] = []
    try:
        import torch

        out = torch.rand(3).tolist()
    except ImportError:
        pass
    return random.random(), float(np.random.rand()), out


def test_same_seed_same_numbers() -> None:
    seed_everything(7)
    a = draw()
    seed_everything(7)
    assert draw() == a


def test_different_seed_different_numbers() -> None:
    seed_everything(1)
    a = draw()
    seed_everything(2)
    assert draw() != a


def test_derive_seed_is_stable_and_separates_names() -> None:
    # Fixed value: must never change across machines or Python versions.
    assert derive_seed(0, "ensemble", 3) == derive_seed(0, "ensemble", 3)
    assert derive_seed(0, "ensemble", 3) != derive_seed(0, "ensemble", 4)
    assert derive_seed(0, "a") != derive_seed(1, "a")
    assert derive_seed(0, "ensemble", 3) == 1330601906


def test_named_generators_are_independent_and_repeatable() -> None:
    assert rng(0, "data").random() == rng(0, "data").random()
    assert rng(0, "data").random() != rng(0, "planet").random()


def test_bench_uses_three_seeds() -> None:
    assert len(BENCH_SEEDS) == 3


def test_seed_range_checked() -> None:
    with pytest.raises(ValueError, match="seed must be"):
        seed_everything(-1)
