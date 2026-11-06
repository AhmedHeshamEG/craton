"""Every map the README shows, through the map referee (``craton.trust.maps``).

A map in the README is a claim: "this inferred map matches that truth". This builds each claim
from the benchmark's own answers and truth and writes the verdicts to
``results/referee/maps.json``. A claim that fails is said to fail where its figure is shown.

    uv run --extra cu126 python scripts/referee_maps.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from craton.bench.baselines import METHODS
from craton.bench.planted import KIND, OFF_STEADY
from craton.bench.stress import CASES, CELL_DEG
from craton.bench.tasks import TASKS, BenchContext
from craton.train import method as _craton_mini  # noqa: F401  registers craton-mini
from craton.trust.maps import MapClaim, MapPanel, judge

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)


def stress_claim(tid: str, method: str, floor: str, seed: int, statement: str) -> MapClaim:
    task: Any = TASKS[tid]
    ctx = BenchContext(ROOT)
    panels = []
    for case in CASES:
        cells = task.cells(ctx, case)
        panels.append(
            MapPanel(
                cells.lat,
                cells.lon,
                cells.azimuth,
                task.answers(METHODS[method], ctx, case, seed),
                task.answers(METHODS[floor], ctx, case, seed),
            )
        )
    return MapClaim(
        f"{tid} {method} over {floor} (seed {seed})", "axial", CELL_DEG, panels, statement
    )


def planted_claim(seed: int) -> MapClaim:
    task: Any = TASKS["CB-33"]
    ctx = BenchContext(ROOT)
    k = seed
    truth = task.truth(ctx, k)
    ans = task.answer(METHODS["craton-mini"], ctx, k, seed)
    flo = task.answer(METHODS["quake_density"], ctx, k, seed)
    lat, lon = task.cb09.spec(KIND, k).grid.cell_centres()
    panels = [
        MapPanel(lat, lon, t, a, f, np.abs(t) >= OFF_STEADY)
        for t, a, f in zip(truth, ans, flo, strict=True)
    ]
    return MapClaim(
        f"CB-33 craton-mini over quake_density (planet {k})",
        "rank",
        task.cb09.spec(KIND, k).grid.cell_deg,
        panels,
        "craton-mini's forecast, read as a stress, ranks the physics planet's hidden state "
        "better than the last year of quakes does",
    )


def pressure_claim(seed: int) -> MapClaim:
    task: Any = TASKS["CB-31"]
    ctx = BenchContext(ROOT)
    truth = task.truth(ctx)
    ans = task.answer(METHODS["craton-mini"], ctx, seed)
    flo = task.answer(METHODS["quake_density"], ctx, seed)
    grid = task.spec.grid
    lat, lon = grid.cell_centres()
    c = truth.cells
    panels = [
        MapPanel(lat[c], lon[c], t, a[c], f[c])
        for t, a, f in zip(truth.loading, ans, flo, strict=True)
    ]
    return MapClaim(
        f"CB-31 craton-mini over quake_density (seed {seed})",
        "rank",
        grid.cell_deg,
        panels,
        "craton-mini's quake-only map ranks Groningen's cells by the year's pressure drop "
        "(the README says it does not)",
    )


def main(which: list[str]) -> None:
    claims: list[MapClaim] = []
    if "CB-23" in which:
        claims.append(
            stress_claim(
                "CB-23",
                "stress_interpolation",
                "stress_regional",
                0,
                "copying the neighbouring cells' squeeze knows more than one direction per region",
            )
        )
    if "CB-34" in which:
        for s in SEEDS:
            for floor in ("mechanism_kostrov", "stress_interpolation"):
                claims.append(
                    stress_claim(
                        "CB-34",
                        "craton-mini",
                        floor,
                        s,
                        f"the tensor field's squeeze matches stress no quake measured better "
                        f"than {floor}",
                    )
                )
    if "CB-33" in which:
        claims += [planted_claim(s) for s in SEEDS]
    if "CB-31" in which:
        claims += [pressure_claim(s) for s in SEEDS]
    out = ROOT / "results" / "referee" / "maps.json"
    old = json.loads(out.read_text("utf-8")) if out.is_file() else {"verdicts": []}
    keep = [v for v in old["verdicts"] if v["name"].split(" ")[0] not in which]
    verdicts = []
    for c in claims:
        v = judge(c)
        print(
            f"{'PASS' if v.passed else 'fail'}  {v.name}: skill {v.skill:.3f} floor "
            f"{v.floor_skill:.3f} shifted q {v.shifted_quantile:.2f} lock {v.locked_lead:+.3f}/"
            f"{v.held_out_lead:+.3f}  {v.tests}",
            flush=True,
        )
        verdicts.append(v.as_dict())
    out.parent.mkdir(parents=True, exist_ok=True)
    body = {"verdicts": sorted(keep + verdicts, key=lambda v: v["name"])}
    out.write_text(json.dumps(body, indent=1) + "\n", "utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["CB-23", "CB-31", "CB-33", "CB-34"])
