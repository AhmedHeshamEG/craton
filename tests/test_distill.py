"""P9e: CB-26's pieces that need no Julia (the search itself runs only in the benchmark)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from craton.bench.catalog import BY_ID
from craton.bench.tasks import TASKS
from craton.field.distill import Law, count_terms, input_names, poisson_gain
from craton.field.memory import MemorySpec


def test_inputs_are_named_in_the_memory_order() -> None:
    names = input_names()
    assert len(names) == MemorySpec().n_features + 2
    assert len(set(names)) == len(names)
    assert names[:4] == ["n_1h_c", "p_1h_c", "n_4h_c", "p_4h_c"]
    assert names[-4:] == ["n_12y_k30", "p_12y_k30", "etas", "box"]


def test_terms_are_counted_as_written() -> None:
    assert count_terms("a + b*c - d") == 3
    assert count_terms("x*(a + b)") == 1
    assert count_terms("0.4") == 1


def test_poisson_gain_is_zero_for_etas_and_rewards_the_truth() -> None:
    rng = np.random.default_rng(0)
    base = np.full((200, 50), math.log(0.05))
    truth = rng.normal(0.0, 0.8, size=base.shape)
    counts = rng.poisson(np.exp(base + truth)).astype(np.float64)
    assert poisson_gain(base, np.zeros_like(base), counts) == 0.0
    assert poisson_gain(base, truth, counts) > 0.1
    assert poisson_gain(base, -truth, counts) < 0.0


def test_a_law_reads_inputs_by_name_and_keeps_the_fields_bound() -> None:
    names = input_names()
    x = np.zeros((3, 4, len(names)))
    x[..., names.index("p_16d_k10")] = np.array([0.0, 1.0, 100.0, -100.0])
    x[..., names.index("etas")] = -2.0
    law = Law("0.5*p_16d_k10 - 0.1*etas", 2, 7, math.log(4.0), {})
    c = law.correction(x)
    assert c.shape == (3, 4)
    b = math.log(4.0)
    np.testing.assert_allclose(c[0, :2], b * np.tanh(np.array([0.2, 0.7]) / b))
    assert np.all(np.abs(c) <= math.log(4.0) + 1e-12)


def test_cb26_is_registered_for_field_methods_only() -> None:
    task = TASKS["CB-26"]
    assert task.direction == "higher"
    assert BY_ID["CB-26"].phase == "P9"
    assert task.supports(SimpleNamespace(field_config="mini"))
    assert not task.supports(SimpleNamespace(name="etas"))
    assert task.prepare(SimpleNamespace())["files"] == {}  # type: ignore[arg-type]
