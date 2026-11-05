"""P9e: the leak audit. Quakes after the cut must change nothing a Field trains on, and the
controls (an earlier cut, answers dealt to the wrong days) must be what they say."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, SequenceForecast
from craton.field.config import field_hash, load_field_config
from craton.field.forecaster import FieldForecaster, build_days, shuffled_answers
from craton.spec.timeaxis import to_seconds

GRID = csep.Grid(42.0, 43.0, 13.0, 14.0, cell_deg=0.1, m_min=1.5, m_max=8.0)
T0 = to_seconds("1995-01-01T00:00:00Z")
CUT = to_seconds("1996-01-01T00:00:00Z")


def _events(n: int, t0: float, t1: float, seed: int, m_max: float = 4.5) -> Events:
    rng = np.random.default_rng(seed)
    return Events(
        time=np.sort(rng.uniform(t0, t1, n)),
        lat=rng.uniform(42.05, 42.95, n),
        lon=rng.uniform(13.05, 13.95, n),
        depth_km=np.full(n, 10.0),
        mag=np.round(rng.uniform(1.5, m_max, n), 1),
        mag_type=np.full(n, "ML"),
        token_hash=np.array([f"{seed}-{i}" for i in range(n)]),
    )


def _concat(a: Events, b: Events) -> Events:
    both = Events(
        **{k: np.concatenate([getattr(a, k), getattr(b, k)]) for k in a.__dataclass_fields__}
    )
    return both.select(np.argsort(both.time, kind="stable"))


def _fit() -> etas.EtasFit:
    proj = etas.Projection.for_grid(GRID)
    box = etas.box_km(GRID, proj)
    bg = etas.Background(
        x=np.array([(box[0] + box[1]) / 2]),
        y=np.array([(box[2] + box[3]) / 2]),
        w=np.array([1.0]),
        box_km=box,
    )
    params = etas.EtasParams(mu=0.5, k=0.02, a=1.8, c=0.01, p=1.1, d=2.0, gamma=0.8, q=1.5)
    return etas.EtasFit(
        params,
        bg,
        mc=2.0,
        b=1.0,
        t_ref=T0,
        window=etas.FitWindow(0.0, 365.0),
        n_targets=100,
        log_likelihood=0.0,
        converged=True,
    )


def test_canary_quakes_after_the_cut_change_no_training_day() -> None:
    """The audit's canaries: a swarm and an M6.5 at and after the cut. Every array a Field
    trains on (memory, ETAS's rates, the answers) must come out bit for bit the same."""
    cfg = load_field_config("mini")
    past = _events(400, T0, CUT, seed=1)
    canaries = _concat(
        _events(300, CUT, CUT + 60 * DAY, seed=2, m_max=6.5), _events(1, CUT, CUT, 3)
    )
    starts = np.arange(T0, CUT - DAY + 1, DAY)
    clean = build_days(_fit(), past, starts, GRID, cfg, True)
    dirty = build_days(_fit(), _concat(past, canaries), starts, GRID, cfg, True)
    assert clean.counts is not None
    assert dirty.counts is not None
    assert clean.counts.sum() > 0
    np.testing.assert_array_equal(clean.memory, dirty.memory)
    np.testing.assert_array_equal(clean.log_etas, dirty.log_etas)
    np.testing.assert_array_equal(clean.counts, dirty.counts)


def test_a_canary_before_the_cut_is_seen() -> None:
    """The same check is not blind: one quake a day before the cut does change the last day."""
    cfg = load_field_config("mini")
    past = _events(400, T0, CUT - 3 * DAY, seed=1)
    starts = np.arange(T0, CUT - DAY + 1, DAY)
    one = _events(1, CUT - 2 * DAY + 3600, CUT - 2 * DAY + 3600, seed=4)
    clean = build_days(_fit(), past, starts, GRID, cfg, True)
    dirty = build_days(_fit(), _concat(past, one), starts, GRID, cfg, True)
    assert not np.array_equal(clean.memory[-1], dirty.memory[-1])
    assert np.array_equal(clean.memory[:-1], dirty.memory[:-1])


def test_shuffled_answers_deal_real_days_to_the_wrong_days() -> None:
    cfg = load_field_config("mini")
    days = build_days(
        _fit(), _events(400, T0, CUT, seed=1), np.arange(T0, CUT - DAY + 1, DAY), GRID, cfg, True
    )
    assert days.counts is not None
    out = shuffled_answers(days, 7)
    assert out.counts is not None
    totals = days.counts.sum(axis=(1, 2))
    assert sorted(out.counts.sum(axis=(1, 2)).tolist()) == sorted(totals.tolist())
    assert not np.array_equal(out.counts, days.counts)
    np.testing.assert_array_equal(out.memory, days.memory)  # the inputs stay where they were
    np.testing.assert_array_equal(shuffled_answers(days, 7).counts, out.counts)  # fixed per seed


def test_audit_configs_leave_mini_alone_and_cut_earlier_only() -> None:
    assert field_hash(load_field_config("mini"))[:8] == "b99e49ae"
    cut = FieldForecaster("audit_cut")
    shuffled = FieldForecaster("audit_shuffled")
    assert len({field_hash(f.config) for f in (cut, shuffled)}) == 2
    task = TASKS["CB-02"]
    assert isinstance(task, SequenceForecast)
    moved = cut.audited(task.spec)
    assert to_seconds(moved.fit_end) < to_seconds(task.spec.fit_end)
    assert moved.window == task.spec.window
    assert shuffled.audited(task.spec) == task.spec
    audit = cut.config.audit
    assert audit is not None
    late = cut.config.model_copy(
        update={"audit": audit.model_copy(update={"cut": "2017-01-01T00:00:00Z"})}
    )
    with pytest.raises(ValueError, match="must come before"):
        FieldForecaster.audited(SimpleNamespace(config=late), task.spec)  # type: ignore[arg-type]
