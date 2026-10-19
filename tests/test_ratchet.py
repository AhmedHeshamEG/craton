from datetime import UTC, datetime

import pytest

from craton.bench.ratchet import accept, judge
from craton.bench.results import RunResults, Scoreboard, TaskResult

FP = "a" * 64


def task(
    task_id: str = "CB-01",
    values: tuple[float, ...] = (3.0, 3.1, 2.9),
    *,
    direction: str = "lower",
    fingerprint: str = FP,
    seeds: tuple[int, ...] | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        metric="bits/token",
        direction=direction,  # type: ignore[arg-type]
        fingerprint=fingerprint,
        seeds=seeds if seeds is not None else tuple(range(len(values))),
        values=values,
    )


def run(*results: TaskResult, run_id: str = "r1") -> RunResults:
    return RunResults(
        run_id=run_id,
        model="mini",
        config_hash="abc",
        git_commit="deadbeef",
        created=datetime(2026, 10, 19, tzinfo=UTC),
        results=results,
    )


def board_with(*results: TaskResult) -> Scoreboard:
    first = run(*results, run_id="champion")
    return accept(Scoreboard(), first, judge(Scoreboard(), first))


def status(board: Scoreboard, r: RunResults, task_id: str = "CB-01") -> str:
    return next(v.status for v in judge(board, r).verdicts if v.task_id == task_id)


def test_empty_board_and_empty_run_pass() -> None:
    report = judge(Scoreboard(), run())
    assert report.passed
    assert report.summary == "nothing to judge"


def test_first_score_is_new() -> None:
    assert status(Scoreboard(), run(task())) == "new"


def test_clear_improvement_passes() -> None:
    board = board_with(task(values=(3.0, 3.1, 2.9)))
    assert status(board, run(task(values=(2.0, 2.1, 1.9)))) == "improved"


def test_change_within_noise_is_same_and_keeps_champion() -> None:
    board = board_with(task(values=(3.0, 3.1, 2.9)))
    slightly_worse = run(task(values=(3.05, 3.15, 2.95)), run_id="r2")
    report = judge(board, slightly_worse)
    assert report.passed
    assert report.verdicts[0].status == "same"
    after = accept(board, slightly_worse, report)
    assert after.entries["CB-01"].run_id == "champion"
    assert len(after.history) == 2


def test_regression_beyond_noise_fails() -> None:
    board = board_with(task(values=(3.0, 3.1, 2.9)))
    report = judge(board, run(task(values=(4.0, 4.1, 3.9))))
    assert not report.passed
    assert report.verdicts[0].status == "regressed"


def test_higher_is_better_direction() -> None:
    board = board_with(task(values=(0.5, 0.51, 0.49), direction="higher"))
    assert status(board, run(task(values=(0.3, 0.31, 0.29), direction="higher"))) == "regressed"
    assert status(board, run(task(values=(0.7, 0.71, 0.69), direction="higher"))) == "improved"


def test_missing_task_fails() -> None:
    board = board_with(task("CB-01"), task("CB-04"))
    report = judge(board, run(task("CB-01")))
    assert not report.passed
    assert {v.task_id: v.status for v in report.verdicts}["CB-04"] == "missing"


def test_fewer_than_three_seeds_is_invalid() -> None:
    assert status(Scoreboard(), run(task(values=(3.0, 3.1)))) == "invalid"


def test_repeated_seeds_are_invalid() -> None:
    assert status(Scoreboard(), run(task(seeds=(0, 0, 1)))) == "invalid"


def test_nan_is_invalid() -> None:
    assert status(Scoreboard(), run(task(values=(3.0, float("nan"), 2.9)))) == "invalid"


def test_changed_fingerprint_is_invalid() -> None:
    board = board_with(task())
    assert status(board, run(task(fingerprint="b" * 64))) == "invalid"


def test_zero_noise_still_catches_tiny_regressions() -> None:
    board = board_with(task(values=(3.0, 3.0, 3.0)))
    assert status(board, run(task(values=(3.001, 3.001, 3.001)))) == "regressed"


def test_accept_refuses_failing_run() -> None:
    board = board_with(task(values=(3.0, 3.1, 2.9)))
    bad = run(task(values=(5.0, 5.1, 4.9)), run_id="bad")
    with pytest.raises(ValueError, match="failed the ratchet"):
        accept(board, bad, judge(board, bad))


def test_accept_refuses_same_run_twice() -> None:
    board = board_with(task())
    again = run(task(values=(1.0, 1.1, 0.9)), run_id="champion")
    with pytest.raises(ValueError, match="already on the scoreboard"):
        accept(board, again, judge(board, again))


def test_duplicate_task_in_one_run_rejected() -> None:
    with pytest.raises(ValueError, match="only once"):
        run(task(), task())
