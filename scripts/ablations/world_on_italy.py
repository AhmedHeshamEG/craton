"""The world Field on central Italy: CB-02 and CB-03 from one Field for every region (P8d).

The Italy Field trains on central Italy up to each task's own cut-off (CB-02: 2016-08-24,
CB-03: 2009-01-01). The world Field trained on four regions, central Italy among them, but
only up to 2007-01-01 everywhere, so it has seen less of Italy than either. This asks what
being one model for the Earth costs (or pays) at home. Not on the scoreboard: the Italy
checkpoints stay the ratchet's floor until the world beats them (PHASES, P8d).

    uv run --extra cu126 python scripts/ablations/world_on_italy.py 0 1 2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.baselines import Etas
from craton.bench.tasks import TASKS, BenchContext, score_forecast
from craton.field.world import WorldField

ROOT = Path(__file__).resolve().parents[2]


def main(seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    world = WorldField(log=lambda m: print(m, flush=True))
    for task_id in ("CB-02", "CB-03"):
        task = TASKS[task_id]
        spec = task.spec  # type: ignore[attr-defined]
        obs = task.observed(ctx)  # type: ignore[attr-defined]
        for seed in seeds:
            w = score_forecast(world.forecast(spec, ctx, seed), obs, spec, seed)
            e = score_forecast(Etas().forecast(spec, ctx, seed), obs, spec, seed)
            many = w.extras["M3.how_many_bits"] - e.extras["M3.how_many_bits"]
            where = w.extras["M3.where_bits"] - e.extras["M3.where_bits"]
            print(
                f"{task_id} seed {seed}: world {w.value:.3f}  etas {e.value:.3f}  "
                f"gain {w.value - e.value:+.3f}  (how many {many:+.3f}, where {where:+.3f})",
                flush=True,
            )


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
