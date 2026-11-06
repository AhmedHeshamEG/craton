"""A linear probe of the Field's inside (P9i): does its hidden state track the fault state?

Hydrology LSTMs trained only on rain and river flow turned out to hold snow and soil moisture
in their cells, readable by a straight line (Kratzert et al. 2019; Lees et al. 2022). The
physics planets of CB-33 know each cell's true rate-and-state state on every graded day; the
Field trained on that planet (CB-09's) never sees it. Here a ridge regression reads it off:

- from each member's 64 hidden features per cell (after attention between cells, before the
  head), and, as the floor, from the Field's own inputs (the fading counters, log1p, plus ETAS's
  log rate and the box's), the same straight line on what goes in;
- fit on the first half of the graded days, scored on the second half the way CB-33 is
  (Spearman over the cells at least 10% off steady, averaged over days).

A hidden state that beats its inputs holds the state in a form the inputs only hide.

    uv run --extra cu126 python scripts/ablations/linear_probe.py 0 1 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from craton.bench.baselines import Etas
from craton.bench.planted import KIND, OFF_STEADY, grade, graded_days, long_run_rate
from craton.bench.tasks import TASKS, BenchContext
from craton.field.forecaster import FieldForecaster, build_days, make_field, predict_members

ROOT = Path(__file__).resolve().parents[2]
RIDGE = 1.0


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = RIDGE) -> tuple[np.ndarray, ...]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return mu, sd, w, np.array([y.mean()])


def ridge_apply(p: tuple[np.ndarray, ...], x: np.ndarray) -> np.ndarray:
    mu, sd, w, b = p
    return np.asarray(((x - mu) / sd) @ w + b[0])


def probe(feats: np.ndarray, truth: np.ndarray, half: int) -> dict[str, float]:
    """feats (days, cells, F); fit on days [:half] off-steady cells, grade days [half:]."""
    off = np.abs(truth[:half]) >= OFF_STEADY
    p = ridge_fit(feats[:half][off], truth[:half][off])
    ans = ridge_apply(p, feats[half:].reshape(-1, feats.shape[-1])).reshape(truth[half:].shape)
    g = grade(truth[half:], ans)
    return {"spearman": g["spearman"]}


def main(seeds: list[int]) -> None:
    out = ROOT / "results" / "ablation" / "linear_probe.json"
    ctx = BenchContext(ROOT)
    task: Any = TASKS["CB-33"]
    ff = FieldForecaster("mini", log=print)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    gdays = graded_days()
    half = len(gdays) // 2
    for k in seeds:
        seed = k
        pctx = task.cb09.planet(ctx, KIND, k)
        spec = ff.audited(task.cb09.spec(KIND, k))
        fit = Etas().fit(spec, pctx, seed)
        states, _ = ff.states(spec, pctx, seed, fit)
        days = build_days(
            fit, ff.inputs(spec, pctx), spec.day_starts[gdays], spec.grid, ff.config, False
        )
        truth = task.truth(ctx, k)
        le = days.log_etas[..., 1].astype(np.float32)
        total = np.broadcast_to(np.log(np.exp(le.astype(np.float64)).sum(1))[:, None], le.shape)
        inputs = np.concatenate(
            [
                np.log1p(np.maximum(days.memory, 0)),
                le[..., None],
                total[..., None].astype(np.float32),
            ],
            -1,
        )
        row: dict[str, Any] = {"planet": k, "graded_days": len(gdays), "fit_days": half}
        row["inputs"] = probe(inputs, truth, half)
        hidden = []
        for state in states:
            net = make_field(days, spec.grid, ff.config)
            net.load_state_dict(state)
            net.to(dev).eval()
            with torch.no_grad():
                h = net.encode(
                    torch.from_numpy(days.memory).to(dev).float(), torch.from_numpy(le).to(dev)
                )
            hidden.append(probe(h.float().cpu().numpy(), truth, half)["spearman"])
        row["hidden_members"] = hidden
        row["hidden"] = {"spearman": float(np.mean(hidden))}
        both = []
        for state in states:
            net = make_field(days, spec.grid, ff.config)
            net.load_state_dict(state)
            net.to(dev).eval()
            with torch.no_grad():
                h = net.encode(
                    torch.from_numpy(days.memory).to(dev).float(), torch.from_numpy(le).to(dev)
                )
            both.append(
                probe(np.concatenate([inputs, h.float().cpu().numpy()], -1), truth, half)[
                    "spearman"
                ]
            )
        row["inputs_and_hidden"] = {"spearman": float(np.mean(both))}
        # The Field's own answer (CB-33's reading of its forecast), on the same held-out days.
        rates = predict_members(states, days, spec.grid, ff.config).mean(0)
        ans = np.log(np.maximum(rates, 1e-30) / long_run_rate(pctx.events)[None])
        row["field_answer"] = {"spearman": grade(truth[half:], ans[half:])["spearman"]}
        print(json.dumps(row), flush=True)
        rows.append(row)
    summary = {
        key: float(np.mean([r[key]["spearman"] for r in rows]))
        for key in ("inputs", "hidden", "inputs_and_hidden", "field_answer")
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"ridge": RIDGE, "rows": rows, "mean": summary}, indent=1) + "\n", "utf-8"
    )
    print(json.dumps(summary))
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
