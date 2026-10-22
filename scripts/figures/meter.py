"""The predictability meter: how much of space and time you must watch to catch 90% of the
M4+ quakes of L'Aquila 2009 and the Apennines 2016-17.

One bar per forecast, on a log scale from "everything" down: watching everything (100%), a map
of the past (the Poisson floor), ETAS, Craton (the accent). Beside each bar, what the share
means in days on alert and in kilometres. Long-term, the project's goal of "exact time and
place" is this bar shrinking towards one cell on one day per quake.

Reads results/scoreboard.json (CB-11, `craton ratchet --accept`); computes nothing.

    uv run --extra cu126 --group figures python scripts/figures/meter.py
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.results import Scoreboard

ROWS = (
    ("everything", "Watching everything", None),
    ("poisson", "A map of the past", "poisson"),
    ("etas", "ETAS, the 1988 standard", "etas"),
    ("craton", "Craton", None),
)


def draw(theme: Theme, out: str) -> None:
    board = Scoreboard.load(ROOT / "results" / "scoreboard.json")
    task = "CB-11"
    floors = board.baselines.get(task, {})
    champ = board.entries.get(task)
    width_px, height_px, dpi = 2400, 1200, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.9, "The predictability meter", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.845,
        "How much of central Italy's space and time you had to watch to catch 90% of the M4+ "
        "quakes of 2009 and 2016-17.",
        10.5,
        theme.muted,
        va="baseline",
    )
    x0, x1 = 0.3, 0.64
    lo, hi = math.log10(0.1), math.log10(100.0)

    def x_of(pct: float) -> float:
        return x0 + (math.log10(max(pct, 0.1)) - lo) / (hi - lo) * (x1 - x0)

    for pct in (0.1, 1, 10, 100):
        ax.plot([x_of(pct)] * 2, [0.13, 0.74], color=theme.faint, linewidth=0.6 * k)
        label = f"{pct:g}%"
        text(x_of(pct), 0.1, label, 8, theme.muted, ha="center")
    y = 0.66
    for key, label, name in ROWS:
        if key == "everything":
            pct, extra = 100.0, {}
        elif key == "craton":
            if champ is None:
                continue
            pct, extra = champ.mean, champ.extras
        else:
            e = floors.get(name or "")
            if e is None:
                continue
            pct, extra = e.mean, e.extras
        color = ACCENT if key == "craton" else (theme.faint if key == "everything" else theme.mid)
        ax.add_patch(plt.Rectangle((x0, y - 0.035), x_of(pct) - x0, 0.07, color=color, linewidth=0))
        weight = 600 if key == "craton" else 400
        text(x0 - 0.015, y, label, 11, theme.ink, weight, ha="right", va="center")
        value = f"{pct:.2f}%" if pct < 99 else "100%"
        text(
            x_of(pct) + 0.012,
            y,
            value,
            11,
            color if key == "craton" else theme.ink,
            weight,
            va="center",
        )
        if extra:
            days = extra.get("alert_days_per_year")
            km = extra.get("space_radius_km")
            detail = []
            if days is not None:
                detail.append(f"{days:.0f} alert days a year")
            if km is not None:
                detail.append(f"{km:.0f} km radius")
            text(0.75, y, " · ".join(detail), 8.5, theme.muted, va="center")
        y -= 0.15
    text(
        0.06,
        0.03,
        "Rank every day and 0.1° cell by the forecast's M4+ rate, watch from the top until 90% "
        "of the real quakes are caught. Mean of 3 seeds. CB-11 on the scoreboard.",
        8,
        theme.muted,
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    for theme in THEMES:
        draw(theme, f"meter-{theme.name}.png")
        print(f"wrote docs/assets/meter-{theme.name}.png")


if __name__ == "__main__":
    main()
