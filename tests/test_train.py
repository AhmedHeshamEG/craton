import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from craton.bench import csep, harness
from craton.bench.data import DAY, Events, Gnss, Tides
from craton.bench.splits import grown_from
from craton.bench.tasks import TASKS, VRAM_BUDGET_BYTES, Bin, Efficiency
from craton.config import ContextPolicy, TrainSettings
from craton.model.encoders import instrument_id
from craton.model.heads import RATE_SCALES
from craton.train import checkpoint
from craton.train.context import (
    LINE_EPOCH,
    YEAR,
    Catalog,
    GnssDays,
    Sources,
    Window,
    fit_lines,
    write,
)
from craton.train.data import (
    Masked,
    NextDay,
    in_validation,
    midnight_window,
    recent_rates,
)
from craton.train.flops import PEAK_FLOPS, flops_per_token, mfu
from craton.train.loop import Stream, batches, lr_at, train
from tests.test_model import ALL_INPUTS, tiny_config

T0 = 1_000_000_000.0
#: Small budgets so a 64-token window holds every kind of token.
POLICY = ContextPolicy(
    tides=6, tides_ahead=4, gnss=16, gnss_days=2, gnss_stations=4, next_day_gnss_days=2,
    big_quakes=4,
)  # fmt: skip
LENGTH = 64


def synthetic_events(n: int = 3000, seed: int = 0) -> Events:
    rng = np.random.default_rng(seed)
    t = T0 + np.cumsum(rng.exponential(3600.0, n))
    return Events(
        time=t,
        lat=42 + rng.random(n),
        lon=13 + rng.random(n),
        depth_km=rng.uniform(0, 20, n),
        mag=np.round(rng.uniform(1.0, 4.0, n), 1),
        mag_type=np.array(["ML"] * n),
        token_hash=np.array([f"h{i}" for i in range(n)]),
    )


def synthetic(n: int = 3000, seed: int = 0) -> Catalog:
    return Catalog.of(synthetic_events(n, seed))


#: Six stations: (code, lat, lon, east velocity m/yr). Their positions are far from zero, as
#: real ones are, and every station's series starts at its own arbitrary origin.
STATIONS = (
    ("AAAA", 42.2, 13.2, 0.020),
    ("BBBB", 42.4, 13.4, 0.022),
    ("CCCC", 42.6, 13.1, 0.024),
    ("DDDD", 42.8, 13.5, 0.021),
    ("EEEE", 42.5, 13.7, 0.023),
    ("FFFF", 42.3, 13.9, 0.025),
)


def synthetic_gnss(days: int = 140, seed: int = 0) -> Gnss:
    rng = np.random.default_rng(seed)
    first = np.floor(T0 / DAY) * DAY + 0.5 * DAY
    rows = []
    for k, (code, lat, lon, ve) in enumerate(STATIONS):
        t = first + DAY * np.arange(days)
        dt = (t - LINE_EPOCH) / YEAR
        enu = np.stack([1000.0 * k + ve * dt, 4_700_000.0 + 0.015 * dt, 300.0 + 0 * dt], 1)
        enu += rng.normal(0, 0.002, enu.shape)
        rows.append((t, np.full(days, code), np.full(days, lat), np.full(days, lon), enu))
    t, st, la, lo, enu = (np.concatenate([r[i] for r in rows]) for i in range(5))
    return Gnss(
        t, st.astype(np.str_), la, lo, enu, np.full_like(enu, 0.002),
        np.array([f"g{i}" for i in range(len(t))]),
    )  # fmt: skip


def synthetic_tides(t0: float, t1: float) -> Tides:
    t = np.arange(np.ceil(t0 / 3600) * 3600, t1, 3600.0)
    enu = 0.1 * np.stack([np.sin(t / 44_712), np.cos(t / 44_712), np.sin(t / 43_082)], 1)
    return Tides(t, np.full(len(t), 42.5), np.full(len(t), 13.25), enu)


def synthetic_sources(inputs: tuple[str, ...] = ALL_INPUTS, n: int = 3000) -> Sources:
    ev = synthetic_events(n)
    g = synthetic_gnss()
    sites = {c: (la, lo) for c, la, lo, _ in STATIONS}
    days = GnssDays.of(g, fit_lines(g, sites))
    tides = synthetic_tides(ev.time[0] - DAY, ev.time[-1] + 2 * DAY)
    return Sources.build(inputs, POLICY, LENGTH, ev, days, tides)


# ---------------------------------------------------------------- windows


def test_window_hides_place_and_size_but_not_time() -> None:
    src = synthetic_sources(("seismic_catalog",))
    cat = src.cat
    hidden = np.zeros(10, dtype=bool)
    hidden[3] = True
    w = Window(t_ref=float(cat.time[0]), quakes=np.arange(10), quake_hidden=hidden)
    b = write(src, [w])
    tok = b.tokens
    assert torch.all(tok.values[0, 3] == 0)
    assert tok.hidden[0, 3, :4].all()
    assert not tok.hidden[0, 2].any()
    assert tok.known[0, 3].tolist() == [True, False, False, False]
    assert torch.all(tok.coords[0, 3, 1:] == 0)
    assert tok.coords[0, 3, 0].item() == pytest.approx(cat.time[3] - cat.time[0])
    assert tok.values[0, 2, 0].item() == pytest.approx(cat.values[2, 0], abs=1e-5)
    assert b.quake_rows.tolist() == [3]
    assert torch.allclose(b.quake_targets[0].double(), torch.from_numpy(cat.values[3]), atol=1e-4)
    assert tok.instrument[0, 10:].eq(-1).all()
    assert tok.padded


def test_hidden_positions_point_at_hidden_tokens() -> None:
    src = synthetic_sources()
    b = Masked(src, 0.15).sample(8, np.random.default_rng(1))
    tok = b.tokens
    flat_inst = tok.instrument.flatten()
    flat_hid = tok.hidden[..., 0].flatten()
    assert torch.all(flat_inst[b.quake_pos] == instrument_id("seismic_catalog"))
    assert torch.all(flat_inst[b.gnss_pos] == instrument_id("gnss_daily"))
    assert torch.all(flat_hid[b.quake_pos])
    assert torch.all(flat_hid[b.gnss_pos])
    quake_hidden = (flat_hid & (flat_inst == instrument_id("seismic_catalog"))).sum()
    assert int(quake_hidden) == len(b.quake_pos) == len(b.quake_targets)


def test_masked_windows_hide_about_the_share() -> None:
    src = synthetic_sources(("seismic_catalog",))
    masked = Masked(src, 0.15)
    assert set(masked.kinds) == {"quake", "region"}
    rng = np.random.default_rng(0)
    ws = [masked.quake_window(int(r), rng) for r in rng.choice(masked.quake_train, 32)]
    share = np.mean(np.concatenate([w.quake_hidden for w in ws]))
    assert 0.1 < share < 0.2
    assert all(w.size == LENGTH for w in ws)
    assert len(masked.quake_train) > len(masked.quake_val) > 0


def test_fused_windows_hold_every_instrument_under_the_budget() -> None:
    src = synthetic_sources()
    masked = Masked(src, 0.15)
    rng = np.random.default_rng(2)
    row = int(np.searchsorted(src.cat.time, src.gnss.time[300]))  # type: ignore[union-attr]
    w = masked.quake_window(row, rng)
    assert w.size == LENGTH
    assert 0 < len(w.tides) <= POLICY.tides
    assert 0 < len(w.gnss) <= POLICY.gnss
    assert len(np.unique(src.gnss.station[w.gnss])) <= POLICY.gnss_stations  # type: ignore[union-attr]


def test_a_station_hidden_whole_borrows_its_line() -> None:
    src = synthetic_sources()
    g = src.gnss
    assert g is not None
    masked = Masked(src, 0.15)
    row = int(np.nonzero(g.station == 2)[0][60])
    w = masked.gnss_window(row, "station", np.random.default_rng(0))
    b = write(src, [w])
    gi = instrument_id("gnss_daily")
    tok = b.tokens
    mine = (tok.instrument[0] == gi).numpy().copy()
    mine[mine] = g.station[w.gnss] == 2
    assert mine.any()
    assert tok.hidden[0, mine, :3].all()
    assert torch.all(tok.values[0, mine, 6] == 0)  # no level: nothing of it is visible
    assert torch.all(tok.values[0, mine, 7] == 0)  # its line is borrowed
    others = (tok.instrument[0] == gi).numpy() & ~mine
    assert not tok.hidden[0, others, 0].any()


def test_gnss_answers_sit_on_cb01_bins() -> None:
    src = synthetic_sources()
    g = src.gnss
    assert g is not None
    b = Masked(src, 0.15).sample(8, np.random.default_rng(3))
    assert len(b.gnss_rows)
    truth = g.enu[b.gnss_rows]
    lo = b.gnss_base + b.gnss_lo.double().numpy() / 1000.0
    for c, ch in enumerate(("east", "north", "up")):
        edges, _ = Bin(ch, 0.001).edges_of(truth[:, c])
        assert np.allclose(lo[:, c], edges, atol=2e-6)
    assert np.all(np.abs(b.gnss_lo.numpy()) < 50)  # a few mm from line + level, not metres


def test_lines_recover_velocities_and_never_borrow_from_themselves() -> None:
    g = synthetic_gnss(days=400)
    sites = {c: (la, lo) for c, la, lo, _ in STATIONS}
    lines = fit_lines(g, sites)
    assert lines.own.all()
    for i, (_, _, _, ve) in enumerate(STATIONS):
        assert lines.velocity[i, 0] == pytest.approx(ve, abs=0.003)
    alone = fit_lines(g, {**sites, "ZZZZ": (42.45, 13.45)})
    z = alone.index("ZZZZ")
    assert not alone.own[z]
    assert np.isnan(alone.offset[z, 0])
    assert 0.019 < alone.velocity[z, 0] < 0.026  # borrowed from the six around it
    assert not np.allclose(alone.borrowed[0], alone.velocity[0])


def test_region_windows_fill_with_a_new_place() -> None:
    src = synthetic_sources(("seismic_catalog",), n=12_000)
    masked = Masked(src, 0.15)
    rng = np.random.default_rng(4)
    w = None
    for _ in range(50):
        w = masked.region_window(int(rng.choice(masked.quake_train[3000:])), rng)
        if w is not None and (~w.quake_hidden).any():
            break
    assert w is not None
    assert w.size == LENGTH
    ctx = w.quakes[~w.quake_hidden]
    qs = w.quakes[w.quake_hidden]
    t0 = src.cat.time[qs].min()
    assert src.cat.time[ctx].max() < t0  # the context is the place's past...
    assert src.cat.time[ctx].min() >= t0 - 365 * DAY - 1  # ...from the year before
    box = src.cat.values[w.quakes, :2]
    assert np.ptp(box[:, 0]) <= 0.6 + 1e-9
    assert np.ptp(box[:, 1]) <= 0.6 + 1e-9


def test_query_reads_only_the_past_except_the_tide() -> None:
    src = synthetic_sources()
    day = float(np.floor(src.cat.time[1500] / DAY) * DAY + DAY)
    w = midnight_window(src, day)
    b = write(src, [w])
    tok = b.tokens
    assert tok.query[0, -1]
    assert w.size == LENGTH
    t = tok.coords[0, :, 0]
    quake = tok.instrument[0] == instrument_id("seismic_catalog")
    gnss = tok.instrument[0] == instrument_id("gnss_daily")
    tide = tok.instrument[0] == instrument_id("solid_earth_tide")
    assert torch.all(t[quake] < 0)
    assert torch.all(t[gnss] < 0)
    assert t[tide].max().item() > 0  # the forcing of the day ahead is known in advance
    assert tok.known[0, -1].tolist() == [True, False, False, False]
    big = src.cat.values[w.quakes, 3] >= POLICY.big_magnitude - 1e-9
    assert big.sum() >= 1


def test_recent_rates_measure_the_past() -> None:
    cat = synthetic(2000)
    day = np.ceil(cat.time[1500] / DAY) * DAY
    r = recent_rates(cat, np.array([day]))
    assert r.shape == (1, len(RATE_SCALES))
    # quakes every hour on average: about 24 a day, at every scale
    assert np.exp(r[0, 2:4].numpy()) == pytest.approx(24, rel=0.3)
    # M3+ are a third of these synthetic quakes (uniform 1.0-4.0): about 8 a day
    assert np.exp(r[0, 5].numpy()) == pytest.approx(8, rel=0.5)


def test_next_day_counts_the_next_24_hours() -> None:
    src = synthetic_sources()
    cat = src.cat
    grid = csep.Grid(41.0, 44.0, 11.5, 15.0, cell_deg=0.1, m_min=1.5, m_max=8.0)
    nd = NextDay(src, grid, cat.time[0], cat.time[-1])
    days = nd.train_days[:5]
    b = nd.batch(days)
    for i, d in enumerate(days):
        sel = (cat.time >= d) & (cat.time < d + DAY) & (cat.values[:, 3] >= 1.45)
        assert b.counts[i].item() == sel.sum()
    assert len(b.cells) == b.counts.sum().item() == len(b.day_of)
    assert b.log_factorials >= 0.0


def test_validation_blocks_are_stable_and_about_a_tenth() -> None:
    t = T0 + np.arange(0, 20 * 365) * DAY
    v = in_validation(t)
    assert np.array_equal(v, in_validation(t))
    assert 0.03 < v.mean() < 0.2


def test_the_stream_is_fixed_by_its_seed() -> None:
    src = synthetic_sources()
    masked = Masked(src, 0.15)
    a = next(batches(Stream(masked, None, 4, 7), 0))[0]
    b = next(batches(Stream(masked, None, 4, 7), 0))[0]
    c = next(batches(Stream(masked, None, 4, 8), 0))[0]
    assert a is not None
    assert b is not None
    assert c is not None
    assert torch.equal(a.tokens.values, b.tokens.values)
    assert not torch.equal(a.tokens.values, c.tokens.values)


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


@pytest.mark.parametrize(
    "inputs",
    [ALL_INPUTS, ("seismic_catalog",), ("gnss_daily", "solid_earth_tide")],
)
def test_a_tiny_model_trains_on_the_cpu(inputs: tuple[str, ...]) -> None:
    cfg = tiny_config(inputs=inputs, length=LENGTH)
    objectives = {"masked": 1.0, "gnss": 1.0, "next_day": 1.0}
    cfg = cfg.model_copy(
        update={
            "context": POLICY,
            "train": cfg.train.model_copy(
                update={
                    "max_steps": 12,
                    "eval_every": 6,
                    "warmup_steps": 3,
                    "patience": 10,
                    "objectives": objectives,
                }
            ),
        }
    )
    src = synthetic_sources(inputs)
    window = (float(src.cat.time[0]), float(src.cat.time[-1]))
    res = train(cfg, src, 0, window, device=torch.device("cpu"))
    last = res.history[-1]
    assert ("val_masked" in last) == ("seismic_catalog" in inputs)
    assert ("val_gnss" in last) == ("gnss_daily" in inputs)
    assert ("val_next_day" in last) == ("seismic_catalog" in inputs)
    assert all(np.isfinite(v) for v in last.values())
    assert res.stats["steps"] == 12
    assert res.stats["tokens_seen"] > 0
    assert res.stats["device"] == "cpu"
    assert res.stats["peak_vram_bytes"] == 0
    assert res.stats["inputs"] == list(inputs)
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
    TrainSettings(**base, objectives={"gnss": 1.0})  # type: ignore[arg-type]


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


def _record(splits: dict[str, Any], tasks: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot": "a",
        "definition": {"version": "1", "splits": splits},
        "splits": {name: {"id": f"id-{name}"} for name in splits},
        "tasks": tasks,
    }


def test_freezing_adds_new_tasks_and_splits_and_never_changes_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    frozen = _record({"pretrain": {"x": 1}}, {"CB-01": {"fingerprint": "x"}})
    body = _record(
        {"pretrain": {"x": 1}, "cb17_fit": {"y": 2}},
        {"CB-01": {"fingerprint": "x"}, "CB-17": {"f": "y"}},
    )
    assert harness._add_tasks(frozen, body, path, lambda m: None) == []
    out = json.loads(path.read_text())
    assert out["tasks"]["CB-17"] == {"f": "y"}
    assert out["splits"]["cb17_fit"] == {"id": "id-cb17_fit"}
    assert out["definition"]["splits"]["cb17_fit"] == {"y": 2}
    changed = _record({"pretrain": {"x": 9}}, {"CB-01": {"fingerprint": "x"}})
    problems = harness._add_tasks(frozen, changed, tmp_path / "g.json", lambda m: None)
    assert problems
    assert not (tmp_path / "g.json").exists()
    task_changed = _record({"pretrain": {"x": 1}}, {"CB-01": {"fingerprint": "z"}})
    assert harness._add_tasks(frozen, task_changed, tmp_path / "h.json", lambda m: None)
    assert grown_from(frozen["definition"], body["definition"])
    assert not grown_from(body["definition"], frozen["definition"])


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_and_card(tmp_path: Path) -> None:
    cfg = tiny_config()
    key = checkpoint.CheckpointKey("cb02_fit", "abcdef0123456789", cfg, 1)
    state = {"w": torch.ones(2)}
    stats = {
        "params": {
            "total": 11,
            "trunk": 6,
            "encoder": 2,
            "masked_head": 1,
            "gnss_head": 1,
            "next_day_head": 1,
        },
        "objectives": {"masked": 1.0, "gnss": 1.0, "next_day": 1.0},
        "inputs": list(ALL_INPUTS),
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
