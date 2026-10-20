import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from craton.bench import csep, harness
from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, VRAM_BUDGET_BYTES, Efficiency
from craton.config import TrainSettings
from craton.model.heads import RATE_SCALES
from craton.train import checkpoint
from craton.train.data import (
    Catalog,
    MaskedWindows,
    NextDay,
    context_rows,
    in_validation,
    query_tokens,
    recent_rates,
    window_tokens,
)
from craton.train.flops import PEAK_FLOPS, flops_per_token, mfu
from craton.train.loop import lr_at, train
from tests.test_model import tiny_config

T0 = 1_000_000_000.0


def synthetic(n: int = 3000, seed: int = 0) -> Catalog:
    rng = np.random.default_rng(seed)
    t = T0 + np.cumsum(rng.exponential(3600.0, n))
    ev = Events(
        time=t,
        lat=42 + rng.random(n),
        lon=13 + rng.random(n),
        depth_km=rng.uniform(0, 20, n),
        mag=np.round(rng.uniform(1.0, 4.0, n), 1),
        mag_type=np.array(["ML"] * n),
        token_hash=np.array([f"h{i}" for i in range(n)]),
    )
    return Catalog.of(ev)


# ---------------------------------------------------------------- examples


def test_window_hides_place_and_size_but_not_time() -> None:
    cat = synthetic(100)
    rows = np.arange(10)[None, :]
    hidden = np.zeros((1, 10), dtype=bool)
    hidden[0, 3] = True
    tok = window_tokens(cat, rows, hidden, cat.time[rows[:, 0]])
    assert torch.all(tok.values[0, 3] == 0)
    assert tok.hidden[0, 3].all()
    assert not tok.hidden[0, 2].any()
    assert tok.known[0, 3].tolist() == [True, False, False, False]
    assert torch.all(tok.coords[0, 3, 1:] == 0)
    assert tok.coords[0, 3, 0].item() == pytest.approx(cat.time[3] - cat.time[0])
    assert tok.values[0, 2, 0].item() == pytest.approx(cat.values[2, 0], abs=1e-5)


def test_query_reads_only_the_past() -> None:
    cat = synthetic(200)
    day = np.floor(cat.time[150] / DAY) * DAY + DAY
    tok = query_tokens(cat, np.array([day]), 16)
    assert tok.query[0].tolist() == [False] * 15 + [True]
    assert torch.all(tok.coords[0, :15, 0] < 0)  # every context quake is before midnight
    assert tok.coords[0, 15, 0].item() == 0.0
    assert tok.known[0, 15].tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="fewer quakes"):
        context_rows(cat, np.array([cat.time[3]]), 16)


def test_recent_rates_measure_the_past() -> None:
    cat = synthetic(2000)
    day = np.ceil(cat.time[1500] / DAY) * DAY
    r = recent_rates(cat, np.array([day]))
    assert r.shape == (1, len(RATE_SCALES))
    # quakes every hour on average: about 24 a day, at every scale
    assert np.exp(r[0, 2:4].numpy()) == pytest.approx(24, rel=0.3)
    # M3+ are a third of these synthetic quakes (uniform 1.0-4.0): about 8 a day
    assert np.exp(r[0, 5].numpy()) == pytest.approx(8, rel=0.5)


def test_masked_windows_hide_about_the_share() -> None:
    cat = synthetic(3000)
    mw = MaskedWindows(cat, 64, 0.15)
    b = mw.sample(32, np.random.default_rng(0))
    share = b.tokens.hidden[..., 0].float().mean().item()
    assert 0.1 < share < 0.2
    assert b.targets.shape == (int(b.tokens.hidden[..., 0].sum()), 4)
    assert len(mw.train_starts) > len(mw.val_starts) > 0


def test_next_day_counts_the_next_24_hours() -> None:
    cat = synthetic(3000)
    grid = csep.Grid(41.0, 44.0, 11.5, 15.0, cell_deg=0.1, m_min=1.5, m_max=8.0)
    nd = NextDay(cat, grid, 32, cat.time[0], cat.time[-1])
    days = nd.train_days[:5]
    b = nd.batch(days)
    for i, d in enumerate(days):
        sel = (cat.time >= d) & (cat.time < d + DAY) & (cat.values[:, 3] >= 1.45)
        assert b.counts[i].item() == sel.sum()
    assert len(b.cells) == b.counts.sum().item() == len(b.day_of)


def test_validation_blocks_are_stable_and_about_a_tenth() -> None:
    t = T0 + np.arange(0, 20 * 365) * DAY
    v = in_validation(t)
    assert np.array_equal(v, in_validation(t))
    assert 0.03 < v.mean() < 0.2


# ---------------------------------------------------------------- the loop


def test_schedule_warms_up_then_decays() -> None:
    cfg = tiny_config()
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"warmup_steps": 10, "max_steps": 100})}
    )
    lrs = [lr_at(s, cfg) for s in range(100)]
    assert lrs[0] < lrs[9] == pytest.approx(cfg.train.lr)
    assert lrs[-1] == pytest.approx(0.1 * cfg.train.lr, rel=0.05)
    assert all(a >= b for a, b in itertools.pairwise(lrs[10:]))


def test_a_tiny_model_trains_on_the_cpu() -> None:
    cfg = tiny_config()
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={"max_steps": 30, "eval_every": 10, "warmup_steps": 3, "patience": 10}
            )
        }
    )
    cat = synthetic(3000)
    res = train(cfg, cat, 0, (cat.time[0], cat.time[-1]), device=torch.device("cpu"))
    first, last = res.history[0], res.history[-1]
    assert last["val_masked"] < first["val_masked"] + 1.0
    assert res.stats["steps"] == 30
    assert res.stats["tokens_seen"] > 0
    assert res.stats["device"] == "cpu"
    assert res.stats["peak_vram_bytes"] == 0
    assert set(res.state) == set(type_check_state(cfg))


def type_check_state(cfg: Any) -> dict[str, torch.Tensor]:
    from craton.model.craton import CratonModel

    return CratonModel(cfg).state_dict()


def test_objectives_are_validated() -> None:
    base = {"micro_batch": 1, "lr": 1e-3, "seeds": (0, 1, 2)}
    with pytest.raises(ValidationError, match="unknown objectives"):
        TrainSettings(**base, objectives={"magic": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="positive weight"):
        TrainSettings(**base, objectives={"masked": 0.0})  # type: ignore[arg-type]


# ---------------------------------------------------------------- CB-15


class FakeRun:
    name = "fake"

    def __init__(self, **kw: Any) -> None:
        self.stats = {
            "device": "NVIDIA GeForce RTX 2060",
            "tokens_per_s": 100_000.0,
            "flops_per_token": 47e6,
            "peak_vram_bytes": 2 * 2**30,
            "train_seconds": 600.0,
            **kw,
        }

    def efficiency(self, ctx: Any, seed: int) -> dict[str, Any]:
        return self.stats


def test_efficiency_is_mfu_inside_the_budget() -> None:
    task = TASKS["CB-15"]
    assert isinstance(task, Efficiency)
    assert task.supports(FakeRun())
    s = task.score(FakeRun(), None, 0)  # type: ignore[arg-type]
    assert s.value == pytest.approx(100 * 100_000 * 47e6 / 25.8e12)
    assert s.extras["peak_vram_mib"] == 2048
    with pytest.raises(ValueError, match="over the 6 GiB budget"):
        task.score(FakeRun(peak_vram_bytes=VRAM_BUDGET_BYTES + 1), None, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no peak"):
        task.score(FakeRun(device="cpu"), None, 0)  # type: ignore[arg-type]


def test_flops_follow_the_palm_formula() -> None:
    cfg = tiny_config()
    m = cfg.model
    assert flops_per_token(cfg) == 6 * cfg.params + 12 * m.n_layers * m.d_model * 16
    assert mfu(1e6, cfg, "NVIDIA GeForce RTX 2060") == pytest.approx(
        1e6 * flops_per_token(cfg) / PEAK_FLOPS["NVIDIA GeForce RTX 2060"]
    )


def test_freezing_adds_new_tasks_and_never_changes_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    frozen = {"snapshot": "a", "tasks": {"CB-01": {"fingerprint": "x"}}}
    body = {"snapshot": "a", "tasks": {"CB-01": {"fingerprint": "x"}, "CB-15": {"f": "y"}}}
    assert harness._add_tasks(frozen, body, path, lambda m: None) == []
    assert json.loads(path.read_text())["tasks"]["CB-15"] == {"f": "y"}
    changed = {"snapshot": "a", "tasks": {"CB-01": {"fingerprint": "z"}, "CB-15": {}}}
    problems = harness._add_tasks(frozen, changed, tmp_path / "g.json", lambda m: None)
    assert problems
    assert not (tmp_path / "g.json").exists()


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_and_card(tmp_path: Path) -> None:
    cfg = tiny_config()
    key = checkpoint.CheckpointKey("cb02_fit", "abcdef0123456789", cfg, 1)
    state = {"w": torch.ones(2)}
    stats = {
        "params": {"total": 10, "trunk": 6, "encoder": 2, "masked_head": 1, "next_day_head": 1},
        "objectives": {"masked": 1.0, "next_day": 1.0},
        "steps": 5,
        "best_step": 5,
        "tokens_seen": 100,
        "train_seconds": 60.0,
        "device": "cpu",
        "precision": "fp32",
        "peak_vram_bytes": 0,
        "tokens_per_s": 10.0,
        "best_val": 1.5,
    }
    data = {"snapshot": "s", "quakes": 3, "first": "2000-01-01", "last": "2001-01-01"}
    folder = checkpoint.save(tmp_path, key, state, [], stats, data)
    loaded = checkpoint.load(tmp_path, key)
    assert loaded is not None
    assert torch.equal(loaded[0]["w"], state["w"])
    card = (folder / "card.md").read_text(encoding="utf-8")
    assert "does not predict earthquakes" in card
    assert "cb02_fit" in card
    assert key.name.endswith("-s1")
    assert checkpoint.TRAIN_VERSION in key.name
    other = checkpoint.CheckpointKey("cb02_fit", "abcdef0123456789", cfg, 2)
    assert checkpoint.load(tmp_path, other) is None
