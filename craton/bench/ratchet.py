"""The ratchet: Craton is only allowed to get better.

A ratchet turns one way. A new run is compared with the champion score of every task on the
scoreboard, and it is accepted only if nothing got worse beyond noise. Noise is two standard
deviations across seeds, taken from whichever side is noisier.

Comparing against the champion, never against the previous run, is what stops slow decay:
a run that is a hair worse but "within noise" keeps the old champion in place, so a hundred
small losses can't stack up into a big one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

from craton.bench.results import HistoryItem, RunResults, Scoreboard, ScoreEntry, TaskResult, now

#: Every score is a mean over at least this many seeds.
MIN_SEEDS = 3
#: How many standard deviations count as noise.
NOISE_SIGMAS = 2.0

Status = Literal["new", "improved", "same", "regressed", "missing", "invalid"]
FAILING: frozenset[Status] = frozenset({"regressed", "missing", "invalid"})


@dataclass(frozen=True)
class Verdict:
    task_id: str
    status: Status
    detail: str
    champion: float | None = None
    candidate: float | None = None
    noise: float | None = None


@dataclass(frozen=True)
class Report:
    run_id: str
    verdicts: tuple[Verdict, ...]

    @property
    def passed(self) -> bool:
        return not any(v.status in FAILING for v in self.verdicts)

    def count(self, status: Status) -> int:
        return sum(v.status == status for v in self.verdicts)

    @property
    def summary(self) -> str:
        if not self.verdicts:
            return "nothing to judge"
        order: tuple[Status, ...] = ("improved", "new", "same", "regressed", "missing", "invalid")
        return ", ".join(f"{self.count(s)} {s}" for s in order if self.count(s))


def improvement(direction: str, champion: float, candidate: float) -> float:
    """Signed change, positive when the candidate is better."""
    return champion - candidate if direction == "lower" else candidate - champion


def _validity_problem(result: TaskResult) -> str | None:
    if len(result.values) < MIN_SEEDS:
        return f"only {len(result.values)} seed(s); every score needs at least {MIN_SEEDS}"
    if len(set(result.seeds)) != len(result.seeds):
        return "repeated seeds"
    if not result.finite:
        return "a value is NaN or infinite"
    return None


def judge(board: Scoreboard, run: RunResults) -> Report:
    """Compare a run with the scoreboard, task by task."""
    verdicts: list[Verdict] = []
    candidates = {r.task_id: r for r in run.results}
    lane = board.lane(run)

    for task_id in sorted(set(lane) | set(candidates)):
        champ = lane.get(task_id)
        cand = candidates.get(task_id)

        if cand is None:
            assert champ is not None
            verdicts.append(
                Verdict(task_id, "missing", "a benchmark can never be dropped", champion=champ.mean)
            )
            continue

        problem = _validity_problem(cand)
        if problem:
            verdicts.append(Verdict(task_id, "invalid", problem))
            continue

        if champ is None:
            verdicts.append(
                Verdict(task_id, "new", "first score for this task", candidate=cand.mean)
            )
            continue

        if cand.fingerprint != champ.fingerprint:
            verdicts.append(
                Verdict(
                    task_id,
                    "invalid",
                    "the task's test data or metric changed; a changed task needs a new ID",
                )
            )
            continue
        if (cand.metric, cand.direction) != (champ.metric, champ.direction):
            verdicts.append(
                Verdict(task_id, "invalid", "metric or direction differs from the board")
            )
            continue

        noise = NOISE_SIGMAS * max(champ.std, cand.std)
        gain = improvement(champ.direction, champ.mean, cand.mean)
        if gain < -noise:
            status: Status = "regressed"
            detail = f"worse by {-gain:.4g}, beyond noise {noise:.4g}"
        elif gain > noise:
            status = "improved"
            detail = f"better by {gain:.4g}, beyond noise {noise:.4g}"
        else:
            status = "same"
            detail = f"within noise ({gain:+.4g} vs ±{noise:.4g}); the champion stays"
        verdicts.append(Verdict(task_id, status, detail, champ.mean, cand.mean, noise))

    return Report(run.run_id, tuple(verdicts))


def accept(board: Scoreboard, run: RunResults, report: Report) -> Scoreboard:
    """Fold a passing run into the scoreboard: new and improved tasks get a new champion."""
    if not report.passed:
        raise ValueError(f"run {run.run_id} failed the ratchet: {report.summary}")
    if any(h.run_id == run.run_id for h in board.history):
        raise ValueError(f"run {run.run_id} is already on the scoreboard")

    when = now()
    entries = dict(board.entries)
    baselines = {t: dict(by) for t, by in board.baselines.items()}
    by_id = {r.task_id: r for r in run.results}
    for v in report.verdicts:
        if v.status in ("new", "improved"):
            r = by_id[v.task_id]
            entry = ScoreEntry(
                task_id=r.task_id,
                metric=r.metric,
                direction=r.direction,
                fingerprint=r.fingerprint,
                mean=r.mean,
                std=r.std,
                n_seeds=len(r.values),
                run_id=run.run_id,
                model=run.model,
                accepted=when,
                extras={k: statistics.fmean(vs) for k, vs in sorted(r.extras.items())},
            )
            if run.kind == "model":
                entries[v.task_id] = entry
            else:
                baselines.setdefault(v.task_id, {})[run.model] = entry
    item = HistoryItem(
        run_id=run.run_id, model=run.model, kind=run.kind, accepted=when, summary=report.summary
    )
    return Scoreboard(
        schema_version=board.schema_version,
        entries=entries,
        baselines=baselines,
        history=(*board.history, item),
    )
