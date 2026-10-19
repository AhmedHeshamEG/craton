"""How Craton works, in one picture: anything measured → one token → one model → a distribution.

uv run --group figures python scripts/figures/architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from style import ACCENT, ASSETS, THEMES, Theme, inter

plt.rcParams["svg.fonttype"] = "path"  # text as outlines: identical everywhere

W, H = 16.0, 6.4  # inches; drawn in data units 0..16 x 0..6.4

COLUMNS = [
    (
        "01",
        "Anything measured",
        [
            "seismometers",
            "GPS stations",
            "solid-Earth tides",
            "rainfall, reservoirs",
            "rock porosity, friction",
            "a monk's diary, 1349",
        ],
    ),
    ("02", "One token", []),
    (
        "03",
        "An encoder each",
        ["one small network per", "instrument, all writing", "into one shared space"],
    ),
    (
        "04",
        "Attention in 4D",
        ["every token looks at", "every other one across", "lat · lon · depth · time"],
    ),
    ("05", "A distribution", ["probabilities and windows,", "never a date"]),
]

SLOTS = [
    "kind",
    "instrument",
    "lat · lon · depth",
    "time",
    "value",
    "uncertainty",
    "provenance",
    "resolution",
]


def arrow(ax: plt.Axes, x0: float, x1: float, y: float, theme: Theme) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=theme.muted
        )
    )


def draw(theme: Theme, preview: Path | None = None) -> None:
    fig = plt.figure(figsize=(W, H), facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.set_facecolor(theme.ground)

    num, title, body = inter(600), inter(600), inter(400)
    xs = [0.7, 3.85, 7.0, 10.15, 13.3]
    top = 5.55
    for (n, head, lines), x in zip(COLUMNS, xs, strict=True):
        ax.text(x, top, n, fontproperties=num, fontsize=11, color=ACCENT, va="baseline")
        ax.text(
            x, top - 0.45, head, fontproperties=title, fontsize=17, color=theme.ink, va="baseline"
        )
        for i, line in enumerate(lines):
            ax.text(
                x,
                top - 1.05 - i * 0.36,
                line,
                fontproperties=body,
                fontsize=11.5,
                color=theme.muted,
                va="baseline",
            )

    # 02: the token, as a stack of slots.
    sx, sy, sw, sh, gap = xs[1], 4.62, 2.45, 0.34, 0.08
    for i, slot in enumerate(SLOTS):
        y = sy - i * (sh + gap)
        accent = slot in ("lat · lon · depth", "time", "uncertainty")
        ax.add_patch(
            FancyBboxPatch(
                (sx, y - sh),
                sw,
                sh,
                boxstyle="round,pad=0,rounding_size=0.07",
                lw=0.9,
                ec=ACCENT if accent else theme.faint,
                fc="none",
            )
        )
        ax.text(
            sx + 0.14,
            y - sh / 2,
            slot,
            fontproperties=body,
            fontsize=10.5,
            color=theme.ink if accent else theme.muted,
            va="center",
        )

    # 03: one encoder per instrument, all landing in one shared space.
    ex, names = xs[2], ["seismic", "gnss", "tides", "rock"]
    bar_x, bar_y0, bar_y1 = ex + 2.2, 1.15, 2.95
    for i, nm in enumerate(names):
        y = 2.8 - i * 0.52
        ax.add_patch(
            FancyBboxPatch(
                (ex, y - 0.17),
                1.05,
                0.34,
                boxstyle="round,pad=0,rounding_size=0.07",
                lw=0.9,
                ec=theme.faint,
                fc="none",
            )
        )
        ax.text(
            ex + 0.13, y, nm, fontproperties=body, fontsize=10.5, color=theme.muted, va="center"
        )
        target = bar_y1 - 0.25 - i * (bar_y1 - bar_y0 - 0.5) / 3
        ax.plot([ex + 1.05, bar_x], [y, target], color=ACCENT, lw=0.9, alpha=0.7)
    ax.add_patch(
        FancyBboxPatch(
            (bar_x, bar_y0),
            0.14,
            bar_y1 - bar_y0,
            boxstyle="round,pad=0,rounding_size=0.06",
            lw=0,
            fc=ACCENT,
        )
    )
    ax.text(
        bar_x + 0.07,
        bar_y0 - 0.3,
        "one space",
        fontproperties=body,
        fontsize=10,
        color=theme.muted,
        ha="center",
    )

    # 04: a tiny attention web over a 4D lattice.
    rng = np.random.default_rng(4)
    px = xs[3] + 0.15 + rng.random(11) * 2.3
    py = 1.15 + rng.random(11) * 1.75
    hub = 4
    for i in range(len(px)):
        if i != hub:
            w = rng.uniform(0.15, 1.0)
            ax.plot(
                [px[hub], px[i]],
                [py[hub], py[i]],
                color=ACCENT,
                lw=0.4 + 1.2 * w,
                alpha=0.25 + 0.6 * w,
                solid_capstyle="round",
            )
    ax.scatter(px, py, s=26, color=theme.ink, zorder=3, linewidths=0)
    ax.scatter([px[hub]], [py[hub]], s=70, color=ACCENT, zorder=4, linewidths=0)

    # 05: a forecast as a distribution with its 90% window shaded.
    t = np.linspace(0, 1, 300)
    dens = 0.85 * np.exp(-((t - 0.32) ** 2) / 0.018) + 0.35 * np.exp(-((t - 0.62) ** 2) / 0.03)
    dx, dy, dw, dh = xs[4], 1.2, 2.4, 1.6
    cx, cy = dx + t * dw, dy + dens / dens.max() * dh
    window = (t > 0.12) & (t < 0.78)
    ax.fill_between(cx[window], dy, cy[window], color=ACCENT, alpha=0.22, lw=0)
    ax.plot(cx, cy, color=ACCENT, lw=1.8)
    ax.plot([dx, dx + dw], [dy, dy], color=theme.faint, lw=0.9)
    ax.text(
        dx + 0.12 * dw, dy - 0.3, "90% window", fontproperties=body, fontsize=10, color=theme.muted
    )

    for x0 in xs[:-1]:
        arrow(ax, x0 + 2.72, x0 + 2.98, 2.05, theme)

    # The interpreter, underneath everything.
    ax.plot([xs[0], xs[4] + 2.45], [0.62, 0.62], color=theme.faint, lw=0.8)
    ax.text(xs[0], 0.22, "On top:", fontproperties=title, fontsize=11.5, color=theme.ink)
    ax.text(
        xs[0] + 0.85,
        0.22,
        "a translator that answers in plain words and may only quote numbers the model returned"
        "   ·   a microscope that turns its internal vectors into named, tested concepts",
        fontproperties=body,
        fontsize=11.5,
        color=theme.muted,
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"how-it-works-{theme.name}.svg"
    fig.savefig(out, facecolor=theme.ground, metadata={"Date": None})
    if preview is not None:
        fig.savefig(preview / out.with_suffix(".png").name, facecolor=theme.ground, dpi=110)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    # --preview DIR also writes PNGs there, for looking at the result.
    where = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--preview" else None
    for theme in THEMES:
        draw(theme, where)
