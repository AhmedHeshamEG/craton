"""What a bit is, in one picture: two bets on the same map, one quake, and a ruler of doublings.

uv run --group figures python scripts/figures/bits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from style import ACCENT, ASSETS, THEMES, Theme, inter

plt.rcParams["svg.fonttype"] = "path"  # text as outlines: identical everywhere

W, H = 16.0, 5.6  # inches; drawn in data units
NX, NY = 10, 6  # the toy map: 60 squares
QUAKE = (6, 3)  # the square the quake lands in
PEAK = (5.2, 2.6)  # where both bets are centred, a little off the quake
CELL = 0.44

# Real scores from the README, as marks on the ruler: (bits, label).
MARKS = [(0.30, "Apennines 2016-17"), (1.16, "L'Aquila 2009")]


def bet(share: float) -> np.ndarray:
    """A bump over the map that sums to 1 and puts exactly `share` on the quake's square."""
    xs, ys = np.meshgrid(np.arange(NX), np.arange(NY))
    d2 = (xs - PEAK[0]) ** 2 + (ys - PEAK[1]) ** 2
    lo, hi = 0.3, 50.0  # the bump's width: wider puts less on the quake's square
    for _ in range(80):
        mid = (lo + hi) / 2
        p = np.exp(-d2 / (2 * mid**2))
        p /= p.sum()
        if p[QUAKE[1], QUAKE[0]] > share:
            lo = mid
        else:
            hi = mid
    out: np.ndarray = p
    return out


def grid(
    ax: plt.Axes, x0: float, y0: float, p: np.ndarray, top: float, colour: str, theme: Theme
) -> None:
    """One bet as shaded squares; `top` is the share drawn at full colour, shared by both maps."""
    for j in range(NY):
        for i in range(NX):
            ax.add_patch(
                Rectangle(
                    (x0 + i * CELL, y0 + j * CELL),
                    CELL * 0.92,
                    CELL * 0.92,
                    fc=to_rgba(colour, 0.06 + 0.94 * float(p[j, i] / top)),
                    ec="none",
                )
            )
    qx, qy = x0 + QUAKE[0] * CELL, y0 + QUAKE[1] * CELL
    ax.add_patch(Rectangle((qx, qy), CELL * 0.92, CELL * 0.92, fc="none", ec=theme.ink, lw=1.6))


def draw(theme: Theme, preview: Path | None = None) -> None:
    fig = plt.figure(figsize=(W, H), facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.set_facecolor(theme.ground)
    num, title, body = inter(600), inter(600), inter(400)

    def head(x: float, n: str, text: str) -> None:
        ax.text(x, 4.95, n, fontproperties=num, fontsize=11, color=ACCENT, va="baseline")
        ax.text(x, 4.5, text, fontproperties=title, fontsize=17, color=theme.ink, va="baseline")

    def note(x: float, y: float, text: str, colour: str | None = None, size: float = 11.5) -> None:
        ax.text(
            x, y, text, fontproperties=body, fontsize=size, color=colour or theme.muted,
            va="baseline",
        )  # fmt: skip

    gw = NX * CELL
    gy = 1.25
    panels = [
        (0.7, "01", "ETAS's bet", bet(0.02), theme.mid, "2%"),
        (0.7 + gw + 0.75, "02", "Craton's bet", bet(0.04), ACCENT, "4%"),
    ]
    top = max(float(p.max()) for _, _, _, p, _, _ in panels)
    for x, n, text, p, colour, share in panels:
        head(x, n, text)
        grid(ax, x, gy, p, top, colour, theme)
        ax.text(x, 0.62, share, fontproperties=title, fontsize=22, color=theme.ink, va="baseline")
        note(x + 0.95, 0.74, "of the bet was on the square")
        note(x + 0.95, 0.44, "where the quake happened")
    note(
        0.7,
        4.08,
        "Each map adds up to 100%. Stronger colour = more of the bet. The outlined square shook.",
    )

    # 03: the ruler of doublings.
    rx0 = 0.7 + 2 * gw + 1.75
    rx1 = W - 0.75
    head(rx0, "03", "Count the doublings")
    note(rx0, 4.08, "4% is twice 2%: one doubling, +1 bit for this quake.")
    lo, hi, ry = -1.3, 1.45, 2.45

    def at(bits: float) -> float:
        return rx0 + (bits - lo) / (hi - lo) * (rx1 - rx0)

    ax.plot([rx0, rx1], [ry, ry], color=theme.faint, lw=1.2)
    for bits, times, word in [
        (-1.0, "×½", "half"),
        (0.0, "×1", "same as ETAS"),
        (1.0, "×2", "twice"),
    ]:
        x = at(bits)
        ax.plot([x, x], [ry - 0.12, ry + 0.12], color=theme.muted, lw=1.2)
        ax.text(
            x, ry - 0.48, f"{bits:+.0f}" if bits else "0", fontproperties=title, fontsize=15,
            color=theme.ink, ha="center", va="baseline",
        )  # fmt: skip
        ax.text(
            x, ry - 0.82, f"{times}  {word}", fontproperties=body, fontsize=10.5,
            color=theme.muted, ha="center", va="baseline",
        )  # fmt: skip
    for k, (bits, label) in enumerate(MARKS):
        x = at(bits)
        ax.plot([x, x], [ry, ry + 0.42 + 0.62 * k], color=ACCENT, lw=1.2)
        ax.plot([x], [ry], marker="o", ms=7, color=ACCENT)
        ax.text(
            x - 0.1, ry + 0.52 + 0.62 * k, f"{bits:+.2f}", fontproperties=title, fontsize=13,
            color=ACCENT, ha="right", va="baseline",
        )  # fmt: skip
        ax.text(
            x - 0.1, ry + 0.27 + 0.62 * k, f"{label}, ×{2**bits:.1f}", fontproperties=body,
            fontsize=10, color=theme.muted, ha="right", va="baseline",
        )  # fmt: skip
    ax.text(
        rx1, ry - 0.48, "bits", fontproperties=body, fontsize=10.5, color=theme.muted,
        ha="right", va="baseline",
    )  # fmt: skip
    note(rx0, 0.74, "A score is this, averaged over every quake in a test.")
    note(rx0, 0.44, "The orange marks are Craton's real scores against ETAS.")

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"bits-{theme.name}.svg"
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
