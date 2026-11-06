"""Where a looser cap wins and loses (P9i): the big day against every other day.

`cap16` is the Field with a ×16 bound instead of ×4 (`configs/field/cap16.yaml`, trained and scored
by `scripts/ablations/curriculum.py cap16 CB-02,CB-03 0`). This splits its difference from `mini`
into the mainshock day (Norcia, 2016-10-30; L'Aquila, 2009-04-06) and all the other test days, in
bits per quake of the whole task, seed 0.

    uv run --extra cu126 python scripts/ablations/cap16_days.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cap_and_mainshock import BIG_DAY, day_ll

from craton.bench.baselines import Etas
from craton.bench.tasks import TASKS, BenchContext
from craton.field.forecaster import FieldForecaster
from craton.spec.timeaxis import to_seconds

ROOT = Path(__file__).resolve().parents[2]


def main(seed: int = 0) -> None:
    ctx = BenchContext(ROOT)
    out: dict[str, dict[str, float]] = {}
    for tid in ("CB-02", "CB-03"):
        task = TASKS[tid]
        spec = task.spec  # type: ignore[attr-defined]
        obs = task.observed(ctx)  # type: ignore[attr-defined]
        d = int((to_seconds(f"{BIG_DAY[tid][0]}T00:00:00Z") - spec.window.t0) // spec.period)
        e, n = day_ll(Etas().forecast(spec, ctx, seed), obs, spec)
        m, _ = day_ll(FieldForecaster("mini").forecast(spec, ctx, seed), obs, spec)
        c, _ = day_ll(FieldForecaster("cap16").forecast(spec, ctx, seed), obs, spec)
        per = n.sum() * math.log(2)
        rest = np.ones(len(n), dtype=bool)
        rest[d] = False
        out[tid] = {
            "seed": seed,
            "big_day": BIG_DAY[tid][0],
            "big_day_cap16_minus_mini": float((c[d] - m[d]) / per),
            "other_days_cap16_minus_mini": float((c[rest] - m[rest]).sum() / per),
            "big_day_mini_over_etas": float((m[d] - e[d]) / per),
            "big_day_cap16_over_etas": float((c[d] - e[d]) / per),
        }
        print(tid, json.dumps(out[tid]), flush=True)
    path = ROOT / "results" / "ablation" / "cap16_days.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
