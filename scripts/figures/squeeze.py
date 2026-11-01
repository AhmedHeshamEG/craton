"""The answer key of CB-23 (P9a): which way the crust is squeezed, cell by cell.

Each orange bar is one 0.5° cell's direction of largest horizontal squeeze (SHmax) from the
World Stress Map, the length of the bar fixed, its thickness the number of readings behind it.
The grey dots are what a method may read instead: the region's quakes before the cut, a
magnitude above its completeness. Nothing else. On the right, how close the simple methods get,
in degrees (a coin: 45°).

Reads CB-23's frozen cells, the region snapshots and the latest run of each floor in
`results/runs/`.

    uv run --group figures python scripts/figures/squeeze.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.adapters.region import load_region
from craton.bench.stress import CASES, StressCells, StressTruth
from craton.bench.tasks import BenchContext

TITLES = {
    "central_italy": "Central Italy",
    "california": "California",
    "japan": "Japan",
    "new_zealand": "New Zealand",
    "turkey": "Turkey",
    "chile": "Chile",
    "alaska": "Alaska",
}
TOP = ("central_italy", "california", "japan", "new_zealand")
BOTTOM = ("turkey", "chile", "alaska")
FLOORS = (
    ("stress_interpolation", "the neighbours' answers, within ~100 km"),
    ("stress_regional", "one direction for the whole region"),
    ("quake_lineation", "the long axis of the quake cloud"),
)


def data() -> tuple[dict[str, StressCells], dict[str, tuple[np.ndarray, np.ndarray]]]:
    ctx = BenchContext(ROOT)
    task = StressTruth()
    cells, quakes = {}, {}
    snaps = task.frozen_snapshots(ctx)
    for case in CASES:
        cells[case.region] = task.cells(ctx, case)
        ev = task.history(task.view(ctx, case, snaps.get(case.region, "")), case).above(
            case.mc + 1.0
        )
        quakes[case.region] = (ev.lon, ev.lat)
    return cells, quakes


def floors() -> dict[str, float]:
    out: dict[str, float] = {}
    for name, _ in FLOORS:
        for path in sorted(
            (ROOT / "results" / "runs").glob(f"*-{name.replace('_', '-')}.json"), reverse=True
        ):
            body = {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["results"]}
            if "CB-23" in body:
                out[name] = float(np.mean(body["CB-23"]["values"]))
                break
    return out


def _panel(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    name: str,
    theme: Theme,
    cells: StressCells,
    q: tuple[np.ndarray, np.ndarray],
) -> None:
    bold = inter(600)
    box = load_region(name, ROOT).box
    ax = fig.add_axes(rect, facecolor=theme.ground)
    ax.scatter(q[0], q[1], s=0.6, c=theme.mid, alpha=0.45, linewidths=0, rasterized=True)
    mid = (box.lat_min + box.lat_max) / 2
    half = 0.21  # half a bar, in degrees of latitude
    a = np.radians(cells.azimuth)
    dx = half * np.sin(a) / np.cos(np.radians(mid))
    dy = half * np.cos(a)
    segs = np.stack(
        [
            np.column_stack([cells.lon - dx, cells.lat - dy]),
            np.column_stack([cells.lon + dx, cells.lat + dy]),
        ],
        axis=1,
    )
    widths = 0.8 + 1.4 * np.clip(np.log10(cells.n) / 2.0, 0, 1)
    ax.add_collection(LineCollection(segs, colors=ACCENT, linewidths=widths, capstyle="round"))
    ax.set_xlim(box.lon_min, box.lon_max)
    ax.set_ylim(box.lat_min, box.lat_max)
    ax.set_aspect(1 / np.cos(np.radians(mid)), adjustable="box", anchor="S")
    ax.axis("off")
    x, y, _, h = rect
    fig.text(x, y + h + 0.012, TITLES[name], color=theme.ink, fontproperties=bold, fontsize=11)
    fig.text(
        x,
        y - 0.03,
        f"{len(cells)} cells · {int(cells.n.sum()):,} readings",
        color=theme.muted,
        fontproperties=inter(400),
        fontsize=8.5,
    )


def draw(
    theme: Theme,
    cells: dict[str, StressCells],
    quakes: dict[str, tuple[np.ndarray, np.ndarray]],
    f: dict[str, float],
) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(14, 10), dpi=200, facecolor=theme.ground)
    for i, name in enumerate(TOP):
        _panel(fig, (0.03 + 0.245 * i, 0.55, 0.21, 0.28), name, theme, cells[name], quakes[name])
    for i, name in enumerate(BOTTOM):
        _panel(fig, (0.03 + 0.245 * i, 0.12, 0.21, 0.28), name, theme, cells[name], quakes[name])
    x0, y = 0.765, 0.36
    fig.text(
        x0, y, "How close simple methods get", color=theme.ink, fontproperties=bold, fontsize=12
    )
    y -= 0.05
    for name, about in FLOORS:
        if name not in f:
            continue
        fig.text(x0, y, f"{f[name]:.1f}°", color=theme.ink, fontproperties=bold, fontsize=22)
        fig.text(x0, y - 0.024, about, color=theme.muted, fontproperties=font, fontsize=9)
        y -= 0.075
    fig.text(x0, y, "45°", color=theme.muted, fontproperties=bold, fontsize=22)
    fig.text(x0, y - 0.024, "a coin", color=theme.muted, fontproperties=font, fontsize=9)
    fig.text(
        0.03,
        0.95,
        "Which way the crust is squeezed",
        color=theme.ink,
        fontproperties=bold,
        fontsize=20,
    )
    fig.text(
        0.03,
        0.918,
        "Orange: the direction of largest horizontal squeeze in every 0.5° cell with three or "
        "more readings (World Stress Map 2025).",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    fig.text(
        0.03,
        0.895,
        "Grey: the quakes a method may read instead, before the cut. CB-23 asks for the orange "
        "from the grey alone.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    out = ASSETS / f"squeeze-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    cells, quakes = data()
    f = floors()
    for theme in THEMES:
        draw(theme, cells, quakes, f)


if __name__ == "__main__":
    main()
