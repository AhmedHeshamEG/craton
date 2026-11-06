"""CB-34 (P9i): the squeeze from how quakes slipped, next to stress no quake measured.

Each orange bar is a 0.5° cell's direction of largest horizontal squeeze from World Stress Map
records that were not read from quakes: boreholes, drilling fractures, hydraulic fracturing,
overcoring, fault-slip data. The ink bar on top of it is Craton's answer at that cell, from the
tensor field (masked modelling of Global CMT mechanisms, never a stress record). Grey dots:
the crustal mechanisms it read, before the cut. Right: every method in degrees (a coin: 45°).

Reads CB-34's frozen cells, the mechanism snapshot, the tensor field (seed 0) and the latest
run of each method in `results/runs/`.

    uv run --extra cu126 --group figures python scripts/figures/mechanisms.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.adapters.region import load_region
from craton.bench.mechanisms import MAX_DEPTH_KM, StressFromMechanisms
from craton.bench.stress import CASES, StressCells
from craton.bench.tasks import BenchContext
from craton.field.tensors import TensorFieldMapper

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
METHODS = (
    ("stress_interpolation", "the neighbouring cells' own answers", False),
    ("craton-mini", "Craton: the tensor field, from mechanisms", True),
    ("mechanism_kostrov", "mechanisms summed nearby (Kostrov)", False),
    ("stress_regional", "one direction for the whole region", False),
    ("quake_lineation", "the long axis of the quake cloud", False),
)


def data() -> tuple[dict[str, Any], ...]:
    ctx = BenchContext(ROOT)
    task = StressFromMechanisms()
    field = TensorFieldMapper("mini", ROOT)
    cells, mechs, answers = {}, {}, {}
    for case in CASES:
        q = task.query(ctx, case)
        cells[case.region] = task.cells(ctx, case)
        m = q.mechanisms.select(~(q.mechanisms.depth_km > MAX_DEPTH_KM))
        mechs[case.region] = (m.lon, m.lat)
        answers[case.region] = field.mechanism_stress_azimuths(q, 0)
    return cells, mechs, answers


def scores() -> dict[str, float]:
    out: dict[str, float] = {}
    for name, _, _ in METHODS:
        for path in sorted(
            (ROOT / "results" / "runs").glob(f"*-{name.replace('_', '-')}.json"), reverse=True
        ):
            body = {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["results"]}
            if "CB-34" in body:
                out[name] = float(np.mean(body["CB-34"]["values"]))
                break
    return out


def _bars(lon: np.ndarray, lat: np.ndarray, azi: np.ndarray, mid: float, half: float) -> Any:
    a = np.radians(azi)
    dx = half * np.sin(a) / np.cos(np.radians(mid))
    dy = half * np.cos(a)
    ok = np.isfinite(a)
    return np.stack(
        [
            np.column_stack([lon[ok] - dx[ok], lat[ok] - dy[ok]]),
            np.column_stack([lon[ok] + dx[ok], lat[ok] + dy[ok]]),
        ],
        axis=1,
    )


def _panel(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    name: str,
    theme: Theme,
    cells: StressCells,
    m: tuple[np.ndarray, np.ndarray],
    answer: np.ndarray,
) -> None:
    bold = inter(600)
    box = load_region(name, ROOT).box
    ax = fig.add_axes(rect, facecolor=theme.ground)
    ax.scatter(m[0], m[1], s=1.2, c=theme.mid, alpha=0.5, linewidths=0, rasterized=True)
    mid = (box.lat_min + box.lat_max) / 2
    # Half a bar: a fixed share of the panel, so small regions don't get giant bars.
    size = 0.028 * (box.lat_max - box.lat_min)
    ax.add_collection(
        LineCollection(
            _bars(cells.lon, cells.lat, cells.azimuth, mid, size),
            colors=ACCENT,
            linewidths=3.0,
            capstyle="round",
        )
    )
    ax.add_collection(
        LineCollection(
            _bars(cells.lon, cells.lat, answer, mid, 1.35 * size),
            colors=theme.ink,
            linewidths=1.3,
            capstyle="round",
        )
    )
    ax.set_xlim(box.lon_min, box.lon_max)
    ax.set_ylim(box.lat_min, box.lat_max)
    ax.set_aspect(1 / np.cos(np.radians(mid)), adjustable="box", anchor="S")
    ax.axis("off")
    x, y, _, h = rect
    fig.text(x, y + h + 0.012, TITLES[name], color=theme.ink, fontproperties=bold, fontsize=11)
    err = np.abs((answer - cells.azimuth + 90.0) % 180.0 - 90.0)
    err = np.where(np.isfinite(err), err, 45.0)
    fig.text(
        x,
        y - 0.03,
        f"{len(cells)} cells · Craton {err.mean():.0f}° off",
        color=theme.muted,
        fontproperties=inter(400),
        fontsize=8.5,
    )


def draw(theme: Theme, d: tuple[dict[str, Any], ...], s: dict[str, float]) -> None:
    cells, mechs, answers = d
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(14, 10), dpi=200, facecolor=theme.ground)
    for i, name in enumerate(TOP):
        rect = (0.03 + 0.245 * i, 0.55, 0.21, 0.28)
        _panel(fig, rect, name, theme, cells[name], mechs[name], answers[name])
    for i, name in enumerate(BOTTOM):
        rect = (0.03 + 0.245 * i, 0.12, 0.21, 0.28)
        _panel(fig, rect, name, theme, cells[name], mechs[name], answers[name])
    x0, y = 0.765, 0.43
    fig.text(x0, y, "Degrees off, seven regions", color=theme.ink, fontproperties=bold, fontsize=12)
    y -= 0.05
    rows = [(s[n], about, theme.ink) for n, about, ours in METHODS if n in s and not ours]
    rows += [(s[n], about, ACCENT) for n, about, ours in METHODS if n in s and ours]
    rows.append((45.0, "a coin", theme.muted))
    for value, about, colour in sorted(rows, key=lambda r: r[0]):
        fig.text(x0, y, f"{value:.1f}°", color=colour, fontproperties=bold, fontsize=22)
        fig.text(x0, y - 0.024, about, color=theme.muted, fontproperties=font, fontsize=9)
        y -= 0.062
    fig.text(
        0.03,
        0.95,
        "The squeeze, from how quakes slipped",
        color=theme.ink,
        fontproperties=bold,
        fontsize=20,
    )
    fig.text(
        0.03,
        0.918,
        "Orange: the largest horizontal squeeze from boreholes, drilling and fault-slip data, "
        "never from a quake (World Stress Map 2025).",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    fig.text(
        0.03,
        0.895,
        "Ink: Craton's answer, from the Global CMT mechanisms (grey) it read before the cut. "
        "Turkey, Chile and Alaska it never trained on.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    out = ASSETS / f"mechanisms-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    s = scores()
    for theme in THEMES:
        draw(theme, d, s)


if __name__ == "__main__":
    main()
