"""P8f: the curriculum (planets → regions → central Italy), blind twins and pumped planets."""

from __future__ import annotations

import numpy as np
import torch

from craton.bench.data import DAY, Events
from craton.bench.planets import KINDS
from craton.field.config import field_hash, load_field_config
from craton.field.planets import (
    PLANET_MC,
    PUSH_PERIOD_DAYS,
    blind,
    planet_grid,
    pumped_pushes,
)


def _events(time_days: list[float], mags: list[float]) -> Events:
    n = len(mags)
    return Events(
        time=np.asarray(time_days, float) * DAY,
        lat=np.full(n, 42.5),
        lon=np.full(n, 13.0),
        depth_km=np.full(n, 10.0),
        mag=np.asarray(mags, float),
        mag_type=np.asarray(["sim"] * n, dtype=np.str_),
        token_hash=np.asarray([f"q{i}" for i in range(n)], dtype=np.str_),
    )


def test_new_sections_leave_old_field_checkpoints_alone() -> None:
    # The hashes every checkpoint on the board was named with, before P8f.
    assert field_hash(load_field_config("mini")).startswith("b99e49ae")
    assert field_hash(load_field_config("world")).startswith("aab7bfce")
    assert field_hash(load_field_config("forced")).startswith("a6ba6957")


def test_the_curriculum_chain_ends_at_planets() -> None:
    chain, name = [], "curriculum"
    while name:
        cfg = load_field_config(name)
        chain.append(name)
        name = cfg.curriculum.start_from if cfg.curriculum else ""
    assert chain[-1] == "planets"
    assert load_field_config("planets").planets is not None


def test_no_training_planet_is_one_cb09_scores() -> None:
    # CB-09's planets are the first qualifying seeds under 200 (craton.bench.planets.MAX_SEEDS).
    for name in ("planets", "pumped"):
        p = load_field_config(name).planets
        assert p is not None
        assert p.first_seed >= 200
        assert p.kind in KINDS


def test_a_blind_network_misses_small_quakes_right_after_a_big_one() -> None:
    # An M6 at day 0: an hour later mc is 6 - 4.5 - 0.75 log10(1/24) = 2.53.
    ev = _events([0.0, 1 / 24, 1 / 24 + 1e-3, 30.0], [6.0, 2.0, 3.0, 2.0])
    kept = blind(ev)
    assert list(kept.mag) == [6.0, 3.0, 2.0]


def test_blindness_fades_and_small_parents_hide_nothing() -> None:
    ev = _events([0.0, 3.0, 10.0, 10.01], [6.0, 2.0, 3.5, 2.0])
    # Three days after an M6, mc is 6 - 4.5 - 0.36 = 1.14: nothing above M2 is missed; an
    # M3.5 blinds nothing (below BLINDING_MAGNITUDE).
    assert len(blind(ev)) == 4


def test_pumped_pushes_switch_on_and_off_in_one_cell_each() -> None:
    g = planet_grid()
    p = pumped_pushes("physics_planet", 2000, g)
    assert len(p) > 0
    assert np.all(np.diff(p.t_end) >= 0)
    assert np.all(p.length == PUSH_PERIOD_DAYS * DAY)
    assert np.all(p.amount > 0)
    for cell in np.unique(p.cell):
        ends = p.t_end[p.cell == cell]
        # One run of consecutive months per source: on, then off for good.
        gaps = np.diff(np.unique(ends)) / DAY
        assert np.all(gaps <= PUSH_PERIOD_DAYS + 1e-6) or len(np.unique(p.cell)) < 2


def test_a_member_starts_from_the_stage_before_without_its_place() -> None:
    from craton.field.forecaster import train_member
    from craton.field.model import FieldNet

    g = planet_grid()
    cfg = load_field_config("curriculum")
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"max_epochs": 1, "patience": 1})}
    )
    n_days, n_mem = 64, 4
    rng = np.random.default_rng(0)

    class D:
        starts = np.arange(n_days) * DAY
        memory = rng.normal(size=(n_days, g.n_cells, n_mem)).astype(np.float16)
        log_etas = np.full((n_days, g.n_cells, 2), -6.0, np.float32)
        counts = np.zeros((n_days, g.n_cells, 2), np.float32)
        span_days = 1

    start = FieldNet(n_mem, g, cfg.model)
    torch.nn.init.constant_(start.out.bias, 0.3)
    init = {k: v for k, v in start.state_dict().items() if k != "place"}
    # A learning rate of zero: the member must come back as exactly where it started.
    frozen = cfg.model_copy(update={"train": cfg.train.model_copy(update={"lr": 1e-12})})
    state, _ = train_member(D(), g, frozen, 0, init=init)  # type: ignore[arg-type]
    assert torch.allclose(state["out.bias"], init["out.bias"], atol=1e-6)
    assert PLANET_MC == 2.0
