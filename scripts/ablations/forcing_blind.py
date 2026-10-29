"""Is it the pump? CB-30's Field trained without the forcing, next to the one with it (D-099).

The forced Field (`craton.field.forced`) reads each cell's Omori memory plus ten forcing
features. This trains the same Field on the same days with the memory alone
(`with_forcing=False`), scores both on CB-30's two cases, and writes per seed and case the bits
per quake of each, ETAS's, and the difference: the part of Craton's gain the measured push
pays for. About 10 minutes per seed for each Field on the RTX 2060.

    uv run --extra cu126 python scripts/ablations/forcing_blind.py 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.baselines import Etas
from craton.bench.induced import CASES, InducedQuakes, case_spec, forecast_of
from craton.bench.tasks import BenchContext, score_forecast
from craton.field.forced import ForcedField

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ablation" / "forcing_blind.json"


def main(seeds: list[int]) -> None:
    ctx = BenchContext(ROOT)
    task = InducedQuakes()
    fields = {
        "forcing": ForcedField("forced", log=print, with_forcing=True),
        "blind": ForcedField("forced", log=print, with_forcing=False),
    }
    rows = []
    for seed in seeds:
        for case in CASES:
            spec = case_spec(case)
            view = task.view(ctx, case, spec)
            obs = task.observed_of(ctx, case)
            row: dict[str, float | int | str] = {"seed": seed, "case": case.key}
            etas = score_forecast(forecast_of(Etas(), spec, view, case, seed), obs, spec, seed)
            row["etas"] = etas.value
            for name, f in fields.items():
                fc = f.forecast_forced(spec, view, case.forcing, seed)
                sc = score_forecast(fc, obs, spec, seed)
                row[name] = sc.value
                row[f"{name}_expected"] = sc.extras[f"M{spec.scored:g}.expected"]
            row["observed"] = etas.extras[f"M{spec.scored:g}.observed"]
            row["forcing_pays"] = float(row["forcing"]) - float(row["blind"])
            print(json.dumps(row), flush=True)
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=1) + "\n", "utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
