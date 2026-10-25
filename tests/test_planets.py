"""CB-09: a fake planet answers the same calls a region does, and the gap meter is honest."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from craton.bench import planets
from craton.bench.baselines import PoissonClimatology
from craton.bench.data import DAY
from craton.bench.splits import SPLITS
from craton.bench.tasks import TASKS
from craton.spec.timeaxis import to_seconds


@pytest.fixture(scope="module")
def etas_planet() -> planets.Events:
    return planets.simulate_planet("etas_planet", 5)  # planet 0 of the frozen record


def test_cb09_is_registered_after_the_locks() -> None:
    task = TASKS["CB-09"]
    assert isinstance(task, planets.TwoPlanets)
    assert task.direction == "higher"
    d = task.definition()
    assert d["primary"] == "physics_planet"
    assert set(d["kinds"]) == {"etas_planet", "physics_planet"}


def test_planet_context_cuts_the_past_like_a_split(
    tmp_path: Path, etas_planet: planets.Events
) -> None:
    ctx = planets.PlanetContext(root=tmp_path, planet=etas_planet, planet_id="p-test")
    cut = to_seconds(SPLITS["cb02_fit"].before or "")
    past = ctx.train_events("cb02_fit")
    assert len(past)
    assert past.time.max() < cut
    assert len(ctx.events) > len(past)
    manifest = json.loads((ctx.split_dir("cb02_fit") / "manifest.json").read_text("utf-8"))
    assert manifest["id"] == "p-testcb02_fit"
    assert "planets" in ctx.bench_dir.parts
    with pytest.raises(ValueError, match="no cut-off"):
        ctx.train_events("pretrain")


def test_planets_copy_cb02s_calendar(etas_planet: planets.Events) -> None:
    assert etas_planet.time.min() >= to_seconds(planets.PLANET_START)
    assert etas_planet.time.max() < planets.WINDOW.t1
    assert planets.qualifies(etas_planet)
    assert not planets.qualifies(etas_planet.select(etas_planet.mag < 4.9))


def test_gap_is_zero_against_itself_and_one_for_a_doubling(
    etas_planet: planets.Events,
) -> None:
    t0, t1 = to_seconds("2005-01-01T00:00:00Z"), to_seconds("2016-08-24T00:00:00Z")
    st = planets.catalog_stats(etas_planet, t0, t1)
    assert planets.sim_to_real_gap(st, st) == 0.0
    doubled = {k: 2 * v for k, v in st.items()}
    assert planets.sim_to_real_gap(doubled, st) == pytest.approx(1.0)
    assert 1.0 < st["b_value"] < 1.4
    assert 0 < st["busiest_1pct_days_share"] < 1


def test_a_floor_runs_on_a_planet_and_the_truth_beats_it(
    tmp_path: Path, etas_planet: planets.Events
) -> None:
    """A baseline forecasts a planet through the same calls it makes on Italy, and ETAS with
    the planet's true parameters is far above a map of the past there."""
    ctx = planets.PlanetContext(root=tmp_path, planet=etas_planet, planet_id="p-test")
    task = TASKS["CB-09"]
    assert isinstance(task, planets.TwoPlanets)
    spec = task.spec("etas_planet", 0)
    obs = planets.in_grid(ctx.events.window(planets.WINDOW.t0, planets.WINDOW.t1))
    floor = planets.score_forecast(PoissonClimatology().forecast(spec, ctx, 0), obs, spec, 0)
    truth = planets.score_forecast(planets.truth_forecast(ctx, spec), obs, spec, 0)
    assert math.isfinite(floor.value)
    assert truth.value > floor.value + 1.0
    assert truth.extras["M3.observed"] == floor.extras["M3.observed"] >= 100
    days = len(spec.day_starts)
    assert days == round((planets.WINDOW.t1 - planets.WINDOW.t0) / DAY)


def test_events_round_trip_through_parquet(tmp_path: Path, etas_planet: planets.Events) -> None:
    path = tmp_path / "p.parquet"
    planets._write_table(path, planets.events_table(etas_planet))
    back = planets.read_events(path)
    np.testing.assert_array_equal(back.time, etas_planet.time)
    np.testing.assert_array_equal(back.token_hash, etas_planet.token_hash)
