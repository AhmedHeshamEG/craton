"""P11, the viable test: Field 2's heart trained on the world and scored on CB-32 and CB-29.

Three questions, three answers, all in ``results/ablation/field2_heart.json``:

1. **Born equal to ETAS?** ``born``: the network at step 0 on CB-32 and CB-29; its gain over
   ETAS must be 0 within 0.01 bits per quake.
2. **Above ETAS on every seed?** ``score``: a heart config on CB-32 (four folds, each heart
   trained without its region) and CB-29 (zero-shot), three seeds.
3. **How far from the Field?** Paired with the accepted runs of ``craton-mini`` (the world
   Field) and ``etas_local_alpha`` (the best floor) on the same tasks and seeds.

    uv run --no-sync python scripts/ablations/field2_heart.py train heart 0 1 2
    uv run --no-sync python scripts/ablations/field2_heart.py score heart 0 1 2
    uv run --no-sync python scripts/ablations/field2_heart.py born heart 0

``train`` fits every fold (and the all-region heart CB-29 reads); cached folds are skipped, so
the same line resumes after a stop. ``score`` runs the config through the bench harness and
merges its result into the file; ``--tasks`` narrows it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from craton.bench import harness
from craton.bench.tasks import BenchContext
from craton.bench.unseen import FOLDS
from craton.field2.method import Field2Method
from craton.field2.world import Field2World
from craton.train import method as _mini  # noqa: F401  registers craton-mini

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("CB-32", "CB-29")
OUT = ROOT / "results" / "ablation" / "field2_heart.json"
KEEP = ("mean_gain", "gain_over_etas", "folds_beaten", "sequences_beaten", ".gain")


def log(msg: str) -> None:
    print(msg, flush=True)


def train(config: str, seeds: list[int], regions: list[str]) -> None:
    for seed in seeds:
        for region in regions:
            w = Field2World(config, log=log, holdout=region)
            name = f"without {region}" if region else "all regions"
            if (w.folder(ROOT, seed) / "model.pt").is_file():
                log(f"{config} {name}, seed {seed}: cached")
                continue
            t0 = time.perf_counter()
            log(f"{config} {name}, seed {seed}: training")
            w.states(ROOT, seed)
            log(f"{config} {name}, seed {seed}: {(time.perf_counter() - t0) / 60:.1f} min")


def accepted(model: str, task: str) -> dict[str, Any] | None:
    for path in sorted((ROOT / "results" / "runs").glob("*.json"), reverse=True):
        body = json.loads(path.read_text("utf-8"))
        if body.get("model") != model:
            continue
        for r in body["results"]:
            if r["task_id"] == task:
                return dict(r)
    return None


def _load() -> dict[str, Any]:
    if OUT.is_file():
        out: dict[str, Any] = json.loads(OUT.read_text("utf-8"))
        return out
    return {"runs": {}, "paired": {}}


def _save(out: dict[str, Any]) -> None:
    for model in ("craton-mini", "etas_local_alpha", "etas_incomplete"):
        for task in TASKS:
            r = accepted(model, task)
            if r is not None:
                out["paired"].setdefault(model, {})[task] = {
                    "seeds": r["seeds"],
                    "values": r["values"],
                    "extras": {k: v for k, v in r["extras"].items() if k.endswith(KEEP)},
                }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", "utf-8")


def score(config: str, seeds: list[int], tasks: list[str], born: bool = False) -> None:
    ctx = BenchContext(root=ROOT, region="central_italy")
    name = f"{config}-born" if born else config
    m = Field2Method(name=f"field2-{name}", config_name=config, born=born, log=log)
    # Field 2 is not on the board until it wins the world gate (P12): an ablation record only.
    res = harness.run(m, ctx, seeds, tasks, log=log, config_hash=name)
    out = _load()
    for r in res.results:
        out["runs"].setdefault(name, {})[r.task_id] = {
            "seeds": list(r.seeds),
            "values": list(r.values),
            "extras": {k: list(v) for k, v in r.extras.items()},
        }
        log(f"{name} {r.task_id}: {[round(v, 4) for v in r.values]}")
    _save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("train", "score", "born"))
    ap.add_argument("config", help="a Field 2 config (configs/field2/)")
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--regions", nargs="*", default=["all", *FOLDS])
    args = ap.parse_args()
    if args.step == "train":
        train(args.config, args.seeds, ["" if r == "all" else r for r in args.regions])
    else:
        score(args.config, args.seeds, args.tasks, born=args.step == "born")


if __name__ == "__main__":
    main()
