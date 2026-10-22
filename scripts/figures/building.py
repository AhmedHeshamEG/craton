"""The building: how far above ETAS a forecast can climb, and where everything stands on it.

Two towers, CB-02 (Apennines 2016-17) and CB-03 (L'Aquila 2009). The height is bits per M3+
quake above ETAS, so floor k gives the real quakes 2^k times ETAS's probability. ETAS is the
ground floor. The roof and the floors under it are oracles (ETAS with the day's count, or its
map, replaced by the truth): they read the answers, so they measure the room. Below ground:
Craton before the Field (P5). The accent: Craton now.

Reads results/ceiling.json (`craton bench ceiling`), results/scoreboard.json and P5's run;
trains nothing.

    uv run --extra cu126 --group figures python scripts/figures/building.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.results import Scoreboard

P5_RUN = "20261021-122053-craton-mini"
TASKS = (("CB-02", "Apennines 2016-17"), ("CB-03", "L'Aquila 2009"))
ORACLES = (
    ("roof", "roof · count and exact cell"),
    ("where_cell", "knowing the exact cell"),
    ("count", "knowing each day's count"),
    ("where_10km", "knowing where to ~10 km"),
)
FLOORS = {"etas_incomplete": "ETAS-incomplete", "etas_aniso": "ETAS, stretched zones"}


def _p5(task: str) -> float:
    run = json.loads((ROOT / "results" / "runs" / f"{P5_RUN}.json").read_text("utf-8"))
    for r in run["results"]:
        if r["task_id"] == task:
            return sum(r["values"]) / len(r["values"])
    raise KeyError(task)


def draw(theme: Theme, out: str) -> None:
    ceiling = json.loads((ROOT / "results" / "ceiling.json").read_text("utf-8"))
    board = Scoreboard.load(ROOT / "results" / "scoreboard.json")
    width_px, height_px, dpi = 2400, 1500, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.93, "The building above ETAS", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.885,
        "Bits per M3+ quake above ETAS, the 1988 standard. Floor k gives the real quakes 2^k "
        "times ETAS's probability.",
        10.5,
        theme.muted,
        va="baseline",
    )
    lo_bits, hi_bits = -6.5, 6.5
    y_lo, y_hi = 0.08, 0.8

    def y_of(bits: float) -> float:
        b = max(min(bits, hi_bits), lo_bits)
        return y_lo + (b - lo_bits) / (hi_bits - lo_bits) * (y_hi - y_lo)

    for col, (task, title) in enumerate(TASKS):
        x0, x1 = 0.16 + col * 0.42, 0.32 + col * 0.42
        rec = ceiling[task]
        etas = rec["etas"]
        text((x0 + x1) / 2, 0.83, f"{task} · {title}", 12, theme.ink, 600, ha="center")
        # The tower: ground to roof, oracles as dashed floors.
        roof = rec["above_etas_bits"]["roof"]
        ax.plot(
            [x0, x0, x1, x1],
            [y_of(0), y_of(roof), y_of(roof), y_of(0)],
            color=theme.faint,
            linewidth=1.2 * k,
        )
        placed: list[float] = []
        for key, label in ORACLES:
            v = rec["above_etas_bits"][key]
            y_text = y_of(v)
            while any(abs(y_text - y) < 0.028 for y in placed):
                y_text -= 0.028  # two floors at nearly the same height: stack their labels
            placed.append(y_text)
            ax.plot(
                [x0, x1],
                [y_of(v)] * 2,
                color=theme.faint,
                linewidth=0.8 * k,
                linestyle=(0, (3, 3)) if key != "roof" else "-",
            )
            text(x1 + 0.01, y_text, f"{label}  +{v:.2f}", 8, theme.muted, va="center")
        # Ground floor: ETAS. Basement: Craton P5.
        ax.plot([x0 - 0.03, x1 + 0.03], [y_of(0)] * 2, color=theme.ink, linewidth=1.6 * k)
        text(x0 - 0.035, y_of(0), f"ETAS  {etas:.2f}", 9, theme.ink, 500, ha="right", va="center")
        p5 = _p5(task) - etas
        ax.plot([x0, x1], [y_of(p5)] * 2, color=theme.mid, linewidth=1.4 * k)
        text((x0 + x1) / 2, y_of(p5) - 0.03, f"Craton P5  {p5:+.2f}", 9, theme.mid, ha="center")
        for name in FLOORS:
            e = board.baselines.get(task, {}).get(name)
            if e is None:
                continue
            v = e.mean - etas
            ax.plot([x0 + 0.02, x1 - 0.02], [y_of(v)] * 2, color=theme.muted, linewidth=0.9 * k)
        entry = board.entries.get(task)
        if entry is not None:
            v = entry.mean - etas
            ax.add_patch(
                plt.Rectangle(
                    (x0, y_of(0)), x1 - x0, y_of(v) - y_of(0), color=ACCENT, alpha=0.9, linewidth=0
                )
            )
            text(
                (x0 + x1) / 2,
                y_of(v) + 0.015,
                f"Craton P6, the Field  {v:+.2f}",
                10,
                ACCENT,
                600,
                ha="center",
            )
    text(
        0.06,
        0.025,
        "Dashed floors are oracles: ETAS with the truth put in, so no forecast can use them. "
        "Thin grey bars: ETAS-incomplete and ETAS with stretched zones. Mean of 3 seeds.",
        8,
        theme.muted,
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    for theme in THEMES:
        draw(theme, f"building-{theme.name}.png")
        print(f"wrote docs/assets/building-{theme.name}.png")


if __name__ == "__main__":
    main()
