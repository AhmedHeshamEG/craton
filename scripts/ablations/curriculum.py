"""The curriculum on central Italy: does starting from the world Field help CB-02 and CB-03?

Two Fields per task and seed, the same data, the same seeds, the same members: `mini` (P6,
starts from ETAS) and a curriculum config (starts every member from a member of an earlier
stage's Field). Prints both scores, the paired gain and its split into how many / where.

    uv run --extra cu126 python scripts/ablations/curriculum.py curriculum CB-02,CB-03 0 1 2

Run seeds one after another, not in parallel: each run rewrites the results file it read at start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.tasks import TASKS, BenchContext, ForecastSpec, GriddedForecast
from craton.field.forecaster import FieldForecaster

ROOT = Path(__file__).resolve().parents[2]


class Method:
    """A Field config as a benchmark method, so every task scores it its own way (CB-20 reads
    the union catalogue through its own context)."""

    long_periods = True

    def __init__(self, config: str, log: object = None) -> None:
        self.name = config
        self.field = FieldForecaster(config, log=log or (lambda _: None))  # type: ignore[arg-type]

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return self.field.forecast(spec, ctx, seed)


def main(config: str, tasks: list[str], seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    log = lambda m: print(m, flush=True)  # noqa: E731
    path = ROOT / "results" / "ablation" / f"{config}_on_italy.json"
    out: dict[str, dict[str, dict[str, float]]] = (
        json.loads(path.read_text("utf-8")) if path.is_file() else {}
    )
    for task_id in tasks:
        task = TASKS[task_id]
        for seed in seeds:
            new = task.score(Method(config, log), ctx, seed)
            old = task.score(Method("mini"), ctx, seed)
            many = new.extras["M3.how_many_bits"] - old.extras["M3.how_many_bits"]
            where = new.extras["M3.where_bits"] - old.extras["M3.where_bits"]
            out.setdefault(task_id, {})[str(seed)] = {
                "new": new.value,
                "mini": old.value,
                "gain": new.value - old.value,
                "how_many": many,
                "where": where,
            }
            print(
                f"{task_id} seed {seed}: {config} {new.value:.3f}  mini {old.value:.3f}  "
                f"gain {new.value - old.value:+.3f}  (how many {many:+.3f}, where {where:+.3f})",
                flush=True,
            )
            path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3:]] or [0, 1, 2])
