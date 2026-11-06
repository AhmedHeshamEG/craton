"""The leak audit on CB-02 (P9e): does the Field's gain over ETAS survive the two controls?

1. **The cut moved back a year.** `audit_cut` trains (and fits its ETAS) as if the split ended
   on 2015-08-24 instead of 2016-08-24, and is scored against that earlier ETAS on the same
   test window. A gain that needs the last year before Amatrice would shrink to nothing here.
2. **Answers dealt to the wrong days.** `audit_shuffled` trains on every day's memory with
   another day's quakes. What it keeps over ETAS is what holds on every day (a fixed map of
   where ETAS is off); what it loses is the part of the gain that reads time.

    uv run --extra cu126 python scripts/ablations/leak_audit.py 0 1 2

Seeds one after another (each trains six members); writes results/ablation/leak_audit.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.baselines import Etas
from craton.bench.tasks import TASKS, BenchContext, ForecastSpec, GriddedForecast, Score
from craton.field.forecaster import FieldForecaster

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ablation" / "leak_audit.json"


class FieldMethod:
    def __init__(self, config: str, log: Any) -> None:
        self.name = config
        self.field = FieldForecaster(config, log=log)

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return self.field.forecast(spec, ctx, seed)


class EtasAt:
    """ETAS fitted with the same cut a Field config trains on."""

    def __init__(self, cut: str | None) -> None:
        self.name, self.cut = f"etas@{cut}" if cut else "etas", cut  # names key the cache

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return Etas().forecast(replace(spec, cut=self.cut) if self.cut else spec, ctx, seed)


def row(new: Score, base: Score) -> dict[str, float]:
    return {
        "field": new.value,
        "etas": base.value,
        "gain": new.value - base.value,
        "how_many": new.extras["M3.how_many_bits"] - base.extras["M3.how_many_bits"],
        "where": new.extras["M3.where_bits"] - base.extras["M3.where_bits"],
    }


def main(seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    task = TASKS["CB-02"]
    log = lambda m: print(m, flush=True)  # noqa: E731
    out: dict[str, dict[str, dict[str, float]]] = (
        json.loads(OUT.read_text("utf-8")) if OUT.is_file() else {}
    )
    cut = FieldForecaster("audit_cut").config.audit
    assert cut is not None
    assert cut.cut
    runs = (("mini", None), ("audit_cut", cut.cut), ("audit_shuffled", None))
    for seed in seeds:
        etas_scores: dict[str | None, Score] = {}
        for config, at in runs:
            if at not in etas_scores:
                etas_scores[at] = task.score(EtasAt(at), ctx, seed)
            r = row(task.score(FieldMethod(config, log), ctx, seed), etas_scores[at])
            out.setdefault(config, {})[str(seed)] = r
            print(
                f"seed {seed} {config}: {r['field']:.3f} vs ETAS {r['etas']:.3f}  gain "
                f"{r['gain']:+.3f} (how many {r['how_many']:+.3f}, where {r['where']:+.3f})",
                flush=True,
            )
            OUT.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
