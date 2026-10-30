"""CB-22 through a curriculum stage: does the earth Field (planets first, Sumatra added) transfer
better than the world Field to regions it never saw?

Scores CB-22 for one world-type Field config per seed, exactly as the benchmark does, and prints
the never-seen half per region next to the board's world Field and ETAS-incomplete numbers.

    uv run --extra cu126 python scripts/ablations/earth_on_world.py earth 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.tasks import TASKS, BenchContext
from craton.field.world import field_stage

ROOT = Path(__file__).resolve().parents[2]


def main(config: str, seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    stage = field_stage(config, lambda m: print(m, flush=True))

    class Method:
        name = config

        def world_forecaster(self) -> object:
            return stage

    path = ROOT / "results" / "ablation" / f"{config}_on_world.json"
    out: dict[str, dict[str, float]] = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    for seed in seeds:
        sc = TASKS["CB-22"].score(Method(), ctx, seed)
        keep = {
            k: v for k, v in sc.extras.items() if k.endswith("mean_gain") or k.endswith(".gain")
        }
        out[str(seed)] = {"primary": sc.value, **keep}
        regions = {k.split(".")[1]: v for k, v in keep.items() if k.count(".") == 2}
        print(
            f"CB-22 seed {seed}: {config} {sc.value:+.3f} (zero-shot); "
            + ", ".join(f"{r} {v:+.3f}" for r, v in regions.items()),
            flush=True,
        )
        path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2:]] or [0, 1, 2])
