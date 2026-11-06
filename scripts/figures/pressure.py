"""CB-31 (P9g): what a quake-only forecast sees of Groningen's reservoir, and what it misses.

Three maps of the 86 graded 0.05° cells, coloured by rank: the total pressure drop since 1957
(gas year 2022-23), the drop over that year alone (CB-31's truth), and craton-mini's map of where
quakes will come that year, from the quakes alone. Under them, CB-31 and its `static` twin for
every method, and the quakes that really came, graded the same way.

    uv run --extra cu126 --group figures python scripts/figures/pressure.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
import craton.train.method  # noqa: F401  (registers craton-mini)
from craton.bench.baselines import METHODS
from craton.bench.pressure import GAS_YEARS
from craton.bench.tasks import TASKS, BenchContext

YEAR = len(GAS_YEARS) - 1  # 2022-23
SHOWN = (
    ("craton-mini", "craton-mini"),
    ("quake_density", "last year's quakes (floor)"),
    ("etas", "ETAS"),
    ("poisson", "Poisson"),
)


def board() -> dict[str, tuple[float, float]]:
    out = {}
    for name, _ in SHOWN:
        run = sorted((ROOT / "results" / "runs").glob(f"*-{name.replace('_', '-')}.json"))[-1]
        body = json.loads(run.read_text("utf-8"))
        r = next(r for r in body["results"] if r["task_id"] == "CB-31")
        out[name] = (float(np.mean(r["values"])), float(np.mean(r["extras"]["static"])))
    ref = json.loads((ROOT / "results" / "ablation" / "pressure_reference.json").read_text())
    row = ref["rows"][0]
    out["observed"] = (float(row["spearman"]), float(row["static"]))
    return out


def data() -> dict[str, Any]:
    ctx = BenchContext(ROOT)
    task: Any = TASKS["CB-31"]
    truth = task.truth(ctx)
    mini = task.answer(METHODS["craton-mini"], ctx, 0)
    return {"truth": truth, "mini": mini[YEAR], "grid": task.spec.grid, "board": board()}


def ranked(values: np.ndarray, cells: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    r = rankdata(values)
    out[cells] = 2 * (r - 1) / max(len(r) - 1, 1) - 1
    return out


def draw(theme: Theme, d: dict[str, Any]) -> None:
    font, bold = inter(400), inter(600)
    g, truth = d["grid"], d["truth"]
    cells = truth.cells
    rows, cols = np.divmod(cells, g.n_lon)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    cmap = LinearSegmentedColormap.from_list("p", [theme.faint, theme.mid, ACCENT])
    cmap.set_bad(theme.ground)
    fig = plt.figure(figsize=(12, 7.4), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.93,
        "The quakes know where the gas field drained, not where it drains now",
        color=theme.ink,
        fontproperties=bold,
        fontsize=19,
    )
    fig.text(
        0.04,
        0.865,
        "Groningen, gas year October 2022 to October 2023. Each square is a 0.05° cell over the "
        "reservoir, coloured by rank (orange = most).\nNo map here reads the pressure: "
        "craton-mini's is a forecast made from earlier quakes alone.",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )
    yr = f"{GAS_YEARS[YEAR]}-{(GAS_YEARS[YEAR] + 1) % 100:02d}"
    panels = (
        ("Total pressure drop since 1957", truth.static[YEAR], "decades of production"),
        (f"Pressure drop over {yr}", truth.loading[YEAR], "CB-31's truth: this year's loading"),
        ("craton-mini's map", d["mini"][cells], "expected quakes per cell that year"),
    )
    for k, (title, v, note) in enumerate(panels):
        ax = fig.add_axes((0.04 + k * 0.32, 0.36, 0.28, 0.42), facecolor=theme.ground)
        img = ranked(np.asarray(v), cells, g.n_cells).reshape(g.n_lat, g.n_lon)[r0:r1, c0:c1]
        ax.imshow(img, origin="lower", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_aspect(1 / np.cos(np.radians(53.3)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        colour = ACCENT if k == 1 else theme.ink
        ax.set_title(title, color=colour, fontproperties=bold, fontsize=12, loc="left")
        ax.text(
            0,
            -0.05,
            note,
            transform=ax.transAxes,
            color=theme.muted,
            fontproperties=font,
            fontsize=9,
            va="top",
        )
    b = d["board"]
    fig.text(
        0.04,
        0.26,
        "Rank correlation, mean of six gas years (2017-2023)",
        color=theme.ink,
        fontproperties=bold,
        fontsize=12,
    )
    fig.text(
        0.40,
        0.222,
        "with this year's drop",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
        ha="right",
    )
    fig.text(
        0.55,
        0.222,
        "with the total drop",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
        ha="right",
    )
    y = 0.19
    for name, label in (*SHOWN, ("observed", "the quakes that really came")):
        load, stat = b[name]
        colour = ACCENT if name == "craton-mini" else theme.ink
        weight = bold if name in ("craton-mini", "observed") else font
        fig.text(0.04, y, label, color=colour, fontproperties=weight, fontsize=10)
        fig.text(
            0.40, y, f"{load:+.2f}", color=colour, fontproperties=bold, fontsize=10.5, ha="right"
        )
        fig.text(
            0.55,
            y,
            f"{stat:+.2f}",
            color=theme.muted,
            fontproperties=font,
            fontsize=10.5,
            ha="right",
        )
        y -= 0.036
    fig.text(
        0.60,
        0.205,
        "Every map, the real quakes included, ranks the cells\nagainst this year's drop. Quakes "
        "follow the total\ndrop, years of loading, the long memory rate-and-\nstate gives faults. "
        "This year's drop is not in them.",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
        va="top",
    )
    out = ASSETS / f"pressure-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
