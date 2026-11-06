"""CB-32's folds: the world Field trained again with one region left out, one seed at a time.

Each fold is a full world Field (every member) that never read a single quake of the region
it leaves out, cached in ``results/checkpoints/world-<config>-...-without-<region>-s<seed>``.
CB-32 (``craton.bench.unseen``) then asks each fold about its own missing region.

    uv run --no-sync python scripts/ablations/unseen_regions.py world 0 1 2
    uv run --no-sync python scripts/ablations/unseen_regions.py world_dimless 0 --regions japan
    uv run --no-sync python scripts/ablations/unseen_regions.py world_dimless 0 --regions all

``all`` is no fold: the Field trained on every region (what CB-22 and CB-24 read).

Folds already trained are skipped, so the same line resumes after a stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from craton.bench.unseen import FOLDS
from craton.field.world import WorldField

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="a world Field config (configs/field/)")
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--regions", nargs="*", default=list(FOLDS), help="folds to train")
    args = ap.parse_args()
    for seed in args.seeds:
        for region in args.regions:
            wf = WorldField(args.config, log=print, holdout="" if region == "all" else region)
            if (wf.folder(ROOT, seed) / "model.pt").is_file():
                print(f"{args.config} without {region}, seed {seed}: cached", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"{args.config} without {region}, seed {seed}: training", flush=True)
            wf.states(ROOT, seed)
            print(
                f"{args.config} without {region}, seed {seed}: done in "
                f"{(time.perf_counter() - t0) / 60:.1f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()
