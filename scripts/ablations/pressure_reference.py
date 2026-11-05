"""CB-31's reference: the quakes that really came, graded as if they were a forecast (D-128).

No forecast can know where the year's quakes will be better than the quakes themselves. This
grades each gas year's real quake counts per cell (M1.0+ and M1.5+) against the yearly pressure
drop and the total drop, exactly as CB-31 grades a method. If even these rank the cells against
the yearly drop, the quakes do not carry it, and no model that reads them can.

    uv run python scripts/ablations/pressure_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import craton.bench.baselines  # noqa: F401  (registers every task)
from craton.bench.pressure import GAS_YEARS, blind_view, gas_year, grade
from craton.bench.tasks import TASKS, BenchContext

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ctx = BenchContext(ROOT)
    task: Any = TASKS["CB-31"]
    truth = task.truth(ctx)
    spec = task.spec
    events = blind_view(ROOT, task.case, spec, task._snapshot(ctx)).events
    rows = []
    for mag in (1.0, 1.5):
        ev = events.above(mag - 1e-9)
        cell = spec.grid.cell_of(ev.lat, ev.lon)
        counts = []
        for y in GAS_YEARS:
            a, b = gas_year(y)
            m = (ev.time >= a) & (ev.time < b) & (cell >= 0)
            counts.append(np.bincount(cell[m], minlength=spec.grid.n_cells).astype(np.float64))
        c = np.array(counts)
        row = {"magnitude": mag, **grade(truth, c)}
        row["quakes_per_year"] = c[:, truth.cells].sum(axis=1).astype(int).tolist()
        print(json.dumps(row), flush=True)
        rows.append(row)
    out = ROOT / "results" / "ablation" / "pressure_reference.json"
    out.write_text(json.dumps({"rows": rows}, indent=1) + "\n", "utf-8")
    print(f"wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
