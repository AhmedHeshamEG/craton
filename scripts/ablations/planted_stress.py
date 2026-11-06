"""CB-33 (and CB-09) for a Field config next to `mini`: does its picture of the physics planet's
hidden state beat the plain Field's, and what does it cost the forecast there?

A config's Fields for CB-09's physics planets are trained the first time (about an hour per
planet); `mini`'s are CB-09's own checkpoints.

    uv run --extra cu126 python scripts/ablations/planted_stress.py meter 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.planets import KINDS, WINDOW, TwoPlanets, in_grid
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    ForecastSpec,
    GriddedForecast,
    Score,
    score_forecast,
)
from craton.field.forecaster import FieldForecaster

ROOT = Path(__file__).resolve().parents[2]


class Method:
    def __init__(self, config: str) -> None:
        self.name = config
        self.field = FieldForecaster(config, log=lambda m: print(m, flush=True))

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        return self.field.forecast(spec, ctx, seed)


def physics_planet(method: Method, ctx: BenchContext, seed: int) -> Score:
    """CB-09's primary half alone (bits per M3+ quake on physics planet ``seed``), so the ETAS
    planet is not trained on for nothing."""
    cb09 = TwoPlanets()
    k = seed % 3
    pctx, spec = cb09.planet(ctx, "physics_planet", k), cb09.spec("physics_planet", k)
    assert "physics_planet" in KINDS
    obs = in_grid(pctx.events.window(WINDOW.t0, WINDOW.t1))
    return score_forecast(method.forecast(spec, pctx, seed), obs, spec, seed)


def main(config: str, seeds: list[int]) -> None:
    ctx = BenchContext(root=ROOT)
    path = ROOT / "results" / "ablation" / f"{config}_on_planets.json"
    out: dict[str, dict[str, dict[str, float]]] = (
        json.loads(path.read_text("utf-8")) if path.is_file() else {}
    )
    for seed in seeds:
        for task_id in ("CB-33", "CB-09"):
            if task_id == "CB-09":
                new = physics_planet(Method(config), ctx, seed)
                old = physics_planet(Method("mini"), ctx, seed)
            else:
                new = TASKS[task_id].score(Method(config), ctx, seed)
                old = TASKS[task_id].score(Method("mini"), ctx, seed)
            row = {"new": new.value, "mini": old.value, "gain": new.value - old.value}
            for k in ("shadows_below_median", "M3.where_bits", "M3.how_many_bits"):
                if k in new.extras:
                    row[f"{k}.new"], row[f"{k}.mini"] = new.extras[k], old.extras[k]
            out.setdefault(task_id, {})[str(seed)] = row
            print(
                f"{task_id} seed {seed}: {config} {new.value:.3f}  mini {old.value:.3f}  "
                f"gain {new.value - old.value:+.3f}",
                flush=True,
            )
            path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2:]] or [0, 1, 2])
