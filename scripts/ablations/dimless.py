"""Do dimensionless inputs transfer? The world Field against itself reading only ratios.

Both Fields share everything (regions, cut, pool of days, network, training); the only
difference is what each cell reads (``craton.field.dimless``). Scored per seed, paired, on
the two tests of working where the model never trained:

- CB-32 (leave one region out): each of central Italy, California, Japan and New Zealand
  forecast by a fold that never read it;
- CB-22 (zero-shot half): Turkey, Chile and Alaska, never read by either Field; and the seen
  half for the record.

Folds and Fields must be trained first (``scripts/ablations/unseen_regions.py``).

    uv run --no-sync python scripts/ablations/dimless.py 0 1 2
        → results/ablation/dimless.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from craton.bench.tasks import TASKS, BenchContext
from craton.train.method import CratonMini

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"world": "world", "dimless": "world_dimless"}
TASK_IDS = ("CB-32", "CB-22")


def main(seeds: list[int]) -> None:
    out = ROOT / "results" / "ablation" / "dimless.json"
    ctx = BenchContext(ROOT)
    rows: list[dict[str, Any]] = []
    for name, cfg in CONFIGS.items():
        method = CratonMini(world_config=cfg, log=print)
        for tid in TASK_IDS:
            task: Any = TASKS[tid]
            for seed in seeds:
                sc = task.score(method, ctx, seed)
                row = {"field": name, "task": tid, "seed": seed, "value": sc.value, **sc.extras}
                rows.append(row)
                print(f"{name:8s} {tid} seed {seed}: {sc.value:+.4f}", flush=True)
    paired = []
    for tid in TASK_IDS:
        for seed in seeds:
            a = next(
                r for r in rows if r["field"] == "world" and r["task"] == tid and r["seed"] == seed
            )
            b = next(
                r
                for r in rows
                if r["field"] == "dimless" and r["task"] == tid and r["seed"] == seed
            )
            keys = [k for k in a if k.endswith("mean_gain") or k.endswith(".gain")]
            paired.append(
                {
                    "task": tid,
                    "seed": seed,
                    "delta": b["value"] - a["value"],
                    **{f"delta.{k}": b[k] - a[k] for k in keys if k in b},
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "paired": paired}, indent=1) + "\n", "utf-8")
    for p in paired:
        print(f"{p['task']} seed {p['seed']}: dimless - world {p['delta']:+.4f}")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
