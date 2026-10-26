"""The data behind the project page (GitHub Pages): every score, from the files that hold them.

Writes site/data.json from results/scoreboard.json, results/ceiling.json and results/runs/.
The page (site/index.html) draws it; the Pages workflow runs this on every push to main, so
the page can never disagree with SCOREBOARD.md.

    uv run python scripts/site.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from craton.bench.catalog import CATALOG
from craton.bench.world import SEQUENCES, ZERO_SHOT

ROOT = Path(__file__).resolve().parents[1]

RUNS = ROOT / "results" / "runs"
NAMES = {t.task_id: t for t in CATALOG}
OUT = ROOT / "site" / "data.json"
FORECASTS = (
    ("CB-02", "Apennines 2016-17, daily"),
    ("CB-03", "L'Aquila 2009, daily"),
    ("CB-09", "Physics planet"),
    ("CB-19", "Apennines, 30 days"),
    ("CB-22 seen", "World, regions trained on"),
    ("CB-22", "World, never-seen regions"),
    ("CB-29", "EarthquakeNPP, in bits"),
)
FLOORS = ("etas_incomplete", "etas_aniso", "poisson")
#: The phase each accepted craton-mini run shipped with; later runs fall back to their date.
PHASES = {
    "20261020-105609": ("P4", "First model"),
    "20261021-122053": ("P5", "Fusion"),
    "20261022-014206": ("P6", "The Field"),
    "20261022-123525": ("P7", "Locks"),
    "20261025-132227": ("P8a", "Two planets"),
    "20261025-195238": ("P8b", "Thirty days"),
    "20261026-102634": ("P8d", "The whole Earth"),
}
ORACLES = (
    ("roof", "Roof: count and exact cell"),
    ("where_cell", "Knowing the exact cell"),
    ("count", "Knowing each day's count"),
    ("where_10km", "Knowing where to ~10 km"),
)


def _results(path: Path) -> dict[str, dict[str, Any]]:
    return {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["results"]}


def _latest(model: str, task: str) -> dict[str, Any]:
    for path in sorted(RUNS.glob(f"*-{model}.json"), reverse=True):
        body = _results(path)
        if task in body:
            return body[task]
    raise KeyError((model, task))


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs)


def vs_etas(entries: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    """Bits per quake above ETAS, per forecasting task and method."""
    rows: list[dict[str, Any]] = []
    for task, label in FORECASTS:
        gains: dict[str, float] = {}
        if task == "CB-22 seen":
            for model, method in (
                ("craton-mini", "craton"),
                *((m.replace("_", "-"), m) for m in FLOORS),
            ):
                gains[method] = _mean(_latest(model, "CB-22")["extras"]["seen.mean_gain"])
        elif task == "CB-22":
            gains["craton"] = entries[task]["mean"]
            gains |= {m: base[task][m]["mean"] for m in FLOORS if m in base[task]}
        elif task == "CB-29":
            etas = base[task]["etas"]["mean"]
            gains["craton"] = (etas - entries[task]["mean"]) / math.log(2)
        else:
            etas = base[task]["etas"]["mean"]
            gains["craton"] = entries[task]["mean"] - etas
            gains |= {m: base[task][m]["mean"] - etas for m in FLOORS if m in base[task]}
        rows += [
            {"task": task, "label": label, "method": m, "gain": round(g, 4)}
            for m, g in gains.items()
        ]
    return rows


def climb(board: dict[str, Any]) -> list[dict[str, Any]]:
    accepted = {h["run_id"] for h in board["history"]}
    rows = []
    for path in sorted(RUNS.glob("*-craton-mini.json")):
        if path.stem not in accepted:
            continue
        run_id = path.stem.removesuffix("-craton-mini")
        phase, name = PHASES.get(run_id, (run_id[4:8], run_id[:8]))
        body = _results(path)
        for task in ("CB-02", "CB-03"):
            rows.append(
                {
                    "phase": phase,
                    "name": name,
                    "task": task,
                    "craton": round(_mean(body[task]["values"]), 4),
                    "etas": round(board["baselines"][task]["etas"]["mean"], 4),
                    "etas_incomplete": round(
                        board["baselines"][task]["etas_incomplete"]["mean"], 4
                    ),
                }
            )
    return rows


def sequences() -> list[dict[str, Any]]:
    craton = _latest("craton-mini", "CB-22")["extras"]
    floor = _latest("etas-incomplete", "CB-22")["extras"]
    return [
        {
            "name": s.name,
            "region": s.region.replace("_", " ").title(),
            "largest": s.largest,
            "seen": s.half != ZERO_SHOT,
            "counted": min(craton[f"{s.key}.qualified"]) >= 1,
            "gain": round(_mean(craton[f"{s.key}.gain"]), 4),
            "low": round(min(craton[f"{s.key}.gain"]), 4),
            "high": round(max(craton[f"{s.key}.gain"]), 4),
            "floor": round(_mean(floor[f"{s.key}.gain"]), 4),
            "quakes": int(craton[f"{s.key}.observed"][0]),
            "fit_quakes": int(craton[f"{s.key}.fit_quakes"][0]),
        }
        for s in SEQUENCES
    ]


def building(entries: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    ceiling = json.loads((ROOT / "results" / "ceiling.json").read_text("utf-8"))
    rows = [
        {
            "rung": label,
            "kind": "oracle",
            **{t: round(ceiling[t]["above_etas_bits"][k], 3) for t in ceiling},
        }
        for k, label in ORACLES
    ]
    rows.append(
        {"rung": "Craton", "kind": "craton"}
        | {t: round(entries[t]["mean"] - base[t]["etas"]["mean"], 3) for t in ceiling}
    )
    rows.append(
        {"rung": "ETAS-incomplete", "kind": "floor"}
        | {
            t: round(base[t]["etas_incomplete"]["mean"] - base[t]["etas"]["mean"], 3)
            for t in ceiling
        }
    )
    rows.append({"rung": "ETAS", "kind": "floor"} | {t: 0.0 for t in ceiling})
    return rows


def scoreboard(entries: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    """Each scored task against its best floor; a gap under twice the seed spread is a tie."""
    rows = []
    for task, e in sorted(entries.items(), key=lambda kv: int(kv[0][3:])):
        if task not in base:
            continue
        lower = e["direction"] == "lower"
        best = (min if lower else max)(base[task].items(), key=lambda kv: kv[1]["mean"])
        gap = (e["mean"] - best[1]["mean"]) * (-1 if lower else 1)
        verdict = "tie" if abs(gap) <= 2 * e["std"] else "win" if gap > 0 else "loss"
        rows.append(
            {
                "id": task,
                "name": NAMES[task].name,
                "measures": NAMES[task].measures,
                "metric": e["metric"],
                "lower_is_better": lower,
                "craton": e["mean"],
                "std": e["std"],
                "floor": best[1]["mean"],
                "floor_method": best[0],
                "verdict": verdict,
            }
        )
    return rows


def main() -> None:
    board = json.loads((ROOT / "results" / "scoreboard.json").read_text("utf-8"))
    entries, base = board["entries"], board["baselines"]
    data = {
        "updated": max(h["accepted"] for h in board["history"]),
        "vs_etas": vs_etas(entries, base),
        "climb": climb(board),
        "sequences": sequences(),
        "building": building(entries, base),
        "scoreboard": scoreboard(entries, base),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
