"""What a benchmark run writes, and what the scoreboard keeps.

A run produces one ``RunResults`` JSON: per task, the score of every seed, plus a fingerprint
of the task's test data and metric. The scoreboard keeps the champion score per task and the
full history of accepted runs.
"""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from craton.bench.catalog import Direction

TASK_ID_PATTERN = r"^CB-\d{2}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskResult(_Strict):
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    metric: str
    direction: Direction
    #: sha256 of the task's frozen test data and metric code. If it changes, it's a new task.
    fingerprint: str = Field(min_length=16)
    seeds: tuple[int, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def _one_value_per_seed(self) -> TaskResult:
        if len(self.seeds) != len(self.values):
            raise ValueError(
                f"{self.task_id}: {len(self.seeds)} seeds but {len(self.values)} values"
            )
        return self

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def std(self) -> float:
        """Sample standard deviation across seeds (0 for a single seed)."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def finite(self) -> bool:
        return all(math.isfinite(v) for v in self.values)


class RunResults(_Strict):
    run_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    model: str
    config_hash: str
    git_commit: str
    created: datetime
    results: tuple[TaskResult, ...]

    @field_validator("results")
    @classmethod
    def _unique_tasks(cls, v: tuple[TaskResult, ...]) -> tuple[TaskResult, ...]:
        ids = [r.task_id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("a run may report each task only once")
        return v

    @classmethod
    def load(cls, path: Path) -> RunResults:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class ScoreEntry(_Strict):
    task_id: str = Field(pattern=TASK_ID_PATTERN)
    metric: str
    direction: Direction
    fingerprint: str
    mean: float
    std: float
    n_seeds: int
    run_id: str
    model: str
    accepted: datetime


class HistoryItem(_Strict):
    run_id: str
    model: str
    accepted: datetime
    summary: str


class Scoreboard(_Strict):
    schema_version: int = 1
    entries: dict[str, ScoreEntry] = Field(default_factory=dict)
    history: tuple[HistoryItem, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Scoreboard:
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")


def now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)
