"""CB-24 through the Field: does the learned wiring know where a sequence goes next?

Three rankers, each the next day's forecast from the moment of the last lighting: ETAS alone,
the P6 Field (`mini`), and the Field with Omori attention (`wiring`). None is trained for
CB-24 (`craton/field/migrate.py`). Floors: distance only 9.3%, quake history 1.4%.

    uv run --extra cu126 python scripts/ablations/wiring_cb24.py etas,mini,wiring 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.tasks import TASKS, BenchContext
from craton.field.migrate import FieldRanker

ROOT = Path(__file__).resolve().parents[2]


def main(rankers: list[str], seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    path = ROOT / "results" / "ablation" / "wiring_on_cb24.json"
    out: dict[str, dict[str, dict[str, float]]] = (
        json.loads(path.read_text("utf-8")) if path.is_file() else {}
    )
    for name in rankers:
        for seed in seeds:
            ranker = FieldRanker(None if name == "etas" else name, ROOT)
            s = TASKS["CB-24"].score(ranker, ctx, seed)
            out.setdefault(name, {})[str(seed)] = {"top3": s.value, **s.extras}
            print(
                f"CB-24 {name} seed {seed}: {s.value:.4f} (seen {s.extras['seen.top3']:.4f}, "
                f"zero-shot {s.extras['zero_shot.top3']:.4f})",
                flush=True,
            )
            path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1].split(","), [int(s) for s in sys.argv[2:]] or [0, 1, 2])
