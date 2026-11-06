"""Is the Field's gain a local alpha? ETAS with its productivity refit daily, beside `mini`.

Lombardi & Marzocchi (2010) found Italy-wide ETAS undershoots the start of the 2009 L'Aquila
sequence and blamed a locally larger alpha. `etas_local_alpha` refits ETAS's productivity on
the box's last 30 days of quakes every day (`craton.bench.local_alpha`). This scores, per seed
and task, ETAS, that floor, the same floor refitting alpha alone, and `mini` (the Field), and
compares them day by day: how much of each day's gain over ETAS in *how many* the floor shares
with the Field.

    uv run --extra cu126 python scripts/ablations/local_alpha.py 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

from craton.bench.baselines import Etas
from craton.bench.local_alpha import EtasLocalAlpha
from craton.bench.tasks import TASKS, BenchContext, GriddedForecast, score_forecast
from craton.field.forecaster import FieldForecaster

ROOT = Path(__file__).resolve().parents[2]
TASK_IDS = ("CB-02", "CB-03")


def day_counts(fc: GriddedForecast, obs_day: np.ndarray, first: int) -> np.ndarray:
    """Per day, log Poisson probability of the day's count at the scored magnitude (nats)."""
    r = fc.expected(first).sum(axis=1)
    return np.asarray(obs_day * np.log(r) - r)


def main(seeds: list[int]) -> None:
    out = ROOT / "results" / "ablation" / "local_alpha.json"
    ctx = BenchContext(ROOT)
    methods: dict[str, Any] = {
        "etas": Etas(),
        "local": EtasLocalAlpha(),
        "alpha_only": EtasLocalAlpha(refit_k=False),
        "mini": FieldForecaster("mini", log=print),
    }
    rows = []
    for tid in TASK_IDS:
        task: Any = TASKS[tid]
        spec = task.spec
        obs = task.observed(ctx)
        first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
        sc_obs = obs.above(spec.scored - 1e-9)
        day = np.floor((sc_obs.time - spec.day_starts[0]) / 86400).astype(int)
        n_day = np.bincount(day[day >= 0], minlength=len(spec.day_starts))[: len(spec.day_starts)]
        busy = n_day >= np.percentile(n_day, 90)
        for seed in seeds:
            row: dict[str, object] = {"task": tid, "seed": seed}
            per_day = {}
            tag = f"M{spec.scored:g}"
            for name, m in methods.items():
                fc = m.forecast(spec, ctx, seed)
                sc = score_forecast(fc, obs, spec, seed)
                row[name] = sc.value
                row[f"{name}.how_many"] = sc.extras[f"{tag}.how_many_bits"]
                row[f"{name}.where"] = sc.extras[f"{tag}.where_bits"]
                row[f"{name}.expected"] = sc.extras[f"{tag}.expected"]
                per_day[name] = day_counts(fc, n_day, first)
            row["observed"] = sc.extras[f"{tag}.observed"]
            for name in ("local", "alpha_only"):
                f = methods[name]
                assert isinstance(f, EtasLocalAlpha)
                fits = f.day_fits(spec, ctx, seed)
                refit = np.array([x.refit for x in fits])
                a = np.array([x.a for x in fits])
                k = np.array([x.k for x in fits])
                row[f"{name}.refit_days"] = int(refit.sum())
                if refit.any():
                    row[f"{name}.alpha_median"] = float(np.median(a[refit]))
                    row[f"{name}.alpha_busy_median"] = float(np.median(a[refit & busy]))
                    row[f"{name}.k_median"] = float(np.median(k[refit]))
            base = per_day["etas"]
            g_mini = per_day["mini"] - base
            g_loc = per_day["local"] - base
            row["day_gain_corr"] = float(np.corrcoef(g_mini, g_loc)[0, 1])
            row["busy_days"] = int(busy.sum())
            q = n_day.sum()
            for name, g in (("mini", g_mini), ("local", g_loc)):
                row[f"{name}.count_gain_busy"] = float(g[busy].sum() / q / np.log(2))
                row[f"{name}.count_gain_quiet"] = float(g[~busy].sum() / q / np.log(2))
            f0 = Etas().fit(spec, ctx, seed)
            row["etas.alpha"] = f0.params.a
            row["etas.k"] = f0.params.k
            row["etas.p"] = f0.params.p
            print(json.dumps(row), flush=True)
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1) + "\n", "utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
