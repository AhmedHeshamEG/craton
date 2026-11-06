"""Two checks on the Field (P9i): does it hit its own cap, and how much of CB-03 is one day?

1. **The cap.** Each Field member moves ETAS's rate in a cell by e^c with c bounded smoothly at
   ln 4 (`max_nudge`, `craton.field.model`). On the L'Aquila night the published gains over the
   long-term rate were 5-25× (INGV's operational forecast), and the Field gave about 3×: is it
   the cap that held it back? This counts, on the CB-02 and CB-03 test days, the cell-days where
   a member's |c| is within 10% of the bound, weighted three ways (plain, by the Field's
   expected M3+ count, and by the M3+ quakes that happened there).
2. **One day.** CB-03 scored day by day: the gain over ETAS of `mini` and `etas_local_alpha`,
   with and without the mainshock day (2009-04-06); and CB-02 without Norcia's (2016-10-30).

    uv run --extra cu126 python scripts/ablations/cap_and_mainshock.py 0 1 2
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from craton.bench.baselines import Etas
from craton.bench.local_alpha import EtasLocalAlpha
from craton.bench.tasks import TASKS, BenchContext, GriddedForecast
from craton.field.forecaster import FieldForecaster, build_days, predict_members
from craton.spec.timeaxis import to_seconds

ROOT = Path(__file__).resolve().parents[2]
#: The day each task is also scored without, and the mainshock's epicentre.
BIG_DAY = {
    "CB-02": ("2016-10-30", 42.83, 13.11),  # Norcia Mw 6.5
    "CB-03": ("2009-04-06", 42.34, 13.38),  # L'Aquila Mw 6.3
}
#: "At the cap": |c| at least this share of ln(max_nudge).
NEAR = 0.9


def day_ll(fc: GriddedForecast, obs: Any, spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per day: Poisson log-likelihood of the scored quakes (nats) and how many there were."""
    g = spec.grid
    first = int(g.mag_bin_of(np.array([spec.scored]))[0])
    day = np.floor((obs.time - spec.window.t0) / spec.period).astype(np.int64)
    cell, mbin = g.cell_of(obs.lat, obs.lon), g.mag_bin_of(obs.mag)
    sel = mbin >= first
    keys = (day[sel] * g.n_cells + cell[sel]) * g.n_mags + mbin[sel]
    uniq, counts = np.unique(keys, return_counts=True)
    d, rest = np.divmod(uniq, g.n_cells * g.n_mags)
    c, m = np.divmod(rest, g.n_mags)
    lam = fc.at(d, c, m)
    terms = counts * np.log(lam) - np.array([math.lgamma(n + 1) for n in counts])
    n_days = len(spec.day_starts)
    ll = np.bincount(d, weights=terms, minlength=n_days) - fc.expected(first).sum(axis=1)
    return ll, np.bincount(d, weights=counts, minlength=n_days)


def bits(ll: np.ndarray, n: np.ndarray, keep: np.ndarray) -> float:
    return float(ll[keep].sum() / n[keep].sum() / math.log(2))


def cap_rows(ff: FieldForecaster, task: Any, ctx: BenchContext, seed: int) -> dict[str, Any]:
    spec = ff.audited(task.spec)
    fit = Etas().fit(spec, ctx, seed)
    states, _ = ff.states(spec, ctx, seed, fit)
    days = build_days(fit, ff.inputs(spec, ctx), spec.day_starts, spec.grid, ff.config, False)
    rates = predict_members(states, days, spec.grid, ff.config)  # (members, days, cells) M3+
    log_etas = days.log_etas[..., 1].astype(np.float64)
    c = np.log(rates) - log_etas[None]
    bound = math.log(ff.config.model.max_nudge)
    at = np.abs(c) >= NEAR * bound
    up = c >= NEAR * bound
    weight = rates.mean(axis=0)
    obs = task.observed(ctx).above(spec.scored - 1e-9)
    day = np.floor((obs.time - spec.window.t0) / spec.period).astype(np.int64)
    cell = spec.grid.cell_of(obs.lat, obs.lon)
    ok = (day >= 0) & (day < len(spec.day_starts)) & (cell >= 0)
    q_at = at[:, day[ok], cell[ok]]
    big, lat, lon = BIG_DAY[task.task_id]
    d_big = int((to_seconds(f"{big}T00:00:00Z") - spec.window.t0) // spec.period)
    c_big = c[:, d_big]
    epi = int(spec.grid.cell_of(np.array([lat]), np.array([lon]))[0])
    mean_c = np.log(weight) - log_etas
    return {
        "bound_ln": bound,
        "cell_days_at_cap": float(at.mean()),
        "cell_days_at_cap_up": float(up.mean()),
        "expected_weighted_at_cap": float((at * weight[None]).sum() / (weight.sum() * len(rates))),
        "quakes_in_a_capped_cell_day": float(q_at.mean()),
        "quakes_where_every_member_is_capped": float(q_at.all(axis=0).mean()),
        "big_day": big,
        "big_day_max_factor": float(np.exp(c_big.max())),
        "big_day_epicentre_factor_members": [float(x) for x in np.exp(c_big[:, epi])],
        "big_day_epicentre_factor_mean": float(np.exp(mean_c[d_big, epi])),
        "big_day_cells_at_cap": int(at[:, d_big].any(axis=0).sum()),
        "max_factor_any_day": float(np.exp(c.max())),
    }


def main(seeds: list[int]) -> None:
    out = ROOT / "results" / "ablation" / "cap_mainshock.json"
    ctx = BenchContext(ROOT)
    ff = FieldForecaster("mini", log=print)
    methods: dict[str, Any] = {"etas": Etas(), "local": EtasLocalAlpha(), "mini": ff}
    rows = []
    for tid in ("CB-02", "CB-03"):
        task: Any = TASKS[tid]
        spec = task.spec
        obs = task.observed(ctx)
        big, _, _ = BIG_DAY[tid]
        d_big = int((to_seconds(f"{big}T00:00:00Z") - spec.window.t0) // spec.period)
        for seed in seeds:
            row: dict[str, Any] = {"task": tid, "seed": seed}
            per = {
                name: day_ll(m.forecast(spec, ctx, seed), obs, spec) for name, m in methods.items()
            }
            n = per["etas"][1]
            every = np.ones(len(n), dtype=bool)
            calm = every.copy()
            calm[d_big] = False
            row["quakes"] = int(n.sum())
            row["quakes_big_day"] = int(n[d_big])
            base_all, base_calm = bits(per["etas"][0], n, every), bits(per["etas"][0], n, calm)
            for name in ("local", "mini"):
                ll = per[name][0]
                row[f"{name}.gain"] = bits(ll, n, every) - base_all
                row[f"{name}.gain_without_big_day"] = bits(ll, n, calm) - base_calm
                row[f"{name}.big_day_share_of_gain_bits"] = float(
                    (ll[d_big] - per["etas"][0][d_big]) / max(1e-9, (ll - per["etas"][0]).sum())
                )
            row["mini_over_local"] = row["mini.gain"] - row["local.gain"]
            row["mini_over_local_without_big_day"] = (
                row["mini.gain_without_big_day"] - row["local.gain_without_big_day"]
            )
            row["cap"] = cap_rows(ff, task, ctx, seed)
            print(json.dumps(row), flush=True)
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"near": NEAR, "rows": rows}, indent=1) + "\n", "utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
