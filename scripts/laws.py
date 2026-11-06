"""Copy the distilled laws (CB-26) out of the gitignored checkpoints into the public record.

Each Field checkpoint that CB-26 has scored holds a ``law.json``: the formula, its size, and
the list of formulas PySR offered at every size with each one's gain on the validation days.
This writes them, one file per config, to ``results/laws/<config>.json``.

    uv run python scripts/laws.py [config]      (default: mini)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from craton.bench.distilled import SOURCES
from craton.bench.tasks import TASKS, BenchContext, SequenceForecast
from craton.field.forecaster import FieldForecaster

ROOT = Path(__file__).resolve().parents[1]


def main(config: str) -> None:
    ff = FieldForecaster(config)
    ctx = BenchContext(root=ROOT)
    out: dict[str, dict[str, object]] = {}
    for tid in SOURCES:
        task = TASKS[tid]
        assert isinstance(task, SequenceForecast)
        for seed in ff.config.train.seeds:
            path = ff.folder(ctx, task.spec, seed) / "law.json"
            if path.is_file():
                out.setdefault(tid, {})[str(seed)] = json.loads(path.read_text("utf-8"))
    dest = ROOT / "results" / "laws" / f"{config}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {dest.relative_to(ROOT)}: {sum(len(v) for v in out.values())} laws")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mini")
