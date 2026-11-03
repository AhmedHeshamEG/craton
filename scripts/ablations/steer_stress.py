"""CB-23 from the steerable kernel: is the angle the forecaster learned the crust's squeeze?

Reads the members a steering Field config trained for CB-02 (run the forecast ablation
first) and scores their angle on CB-23 per seed, next to the floors.

    uv run --extra cu126 python scripts/ablations/steer_stress.py hidden_steer 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.tasks import TASKS, BenchContext
from craton.field.stressmap import FieldStress

ROOT = Path(__file__).resolve().parents[2]


def main(config: str, seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    path = ROOT / "results" / "ablation" / f"{config}_on_cb23.json"
    out: dict[str, dict[str, float]] = {}
    for seed in seeds:
        s = TASKS["CB-23"].score(FieldStress(config), ctx, seed)
        out[str(seed)] = {"deg": s.value, **s.extras}
        regions = ", ".join(
            f"{k.split('.')[0]} {v:.1f}" for k, v in s.extras.items() if k.endswith(".deg")
        )
        print(f"CB-23 seed {seed}: {s.value:.2f} deg ({regions})", flush=True)
        path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2:]] or [0, 1, 2])
