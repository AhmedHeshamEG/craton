"""What each instrument is worth: the P5 ablation, as one table.

Craton-mini is trained again with one instrument left out at a time (earthquakes, GNSS, the
solid-Earth tide) and scored on the same frozen tasks. Each cell is how much worse the model
gets without that instrument, in the task's own unit. Orange: it helps, beyond noise (two
standard deviations over three seeds, the ratchet's rule). Grey: within noise, whatever the
sign. Blank: that variant can't answer the task.

Reads results/ablation/*.json (written by `craton bench ablate`); trains nothing.

    uv run --extra cu126 --group figures python scripts/figures/ablation.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench import ablation

#: Rows drawn, as indices into ``ablation.ROWS`` (the CB-01 total is the mean of its halves).
SHOWN = (1, 2, 3, 4, 5)
LABELS = {
    1: ("A hidden earthquake", "CB-01 · where and how big, bits"),
    2: ("A hidden GNSS day", "CB-01 · east, north, up, bits"),
    3: ("Forecast: Apennines 2016-17", "CB-02 · bits per M3+ quake"),
    4: ("Forecast: L'Aquila 2009", "CB-03 · bits per M3+ quake"),
    5: ("A GNSS station nobody saw", "CB-04 · rebuilt daily, mm"),
}


def _fmt(x: float, unit: str) -> str:
    if unit == "mm":
        return f"{x:+.2f} mm"
    return f"{x:+.2f}"


def draw(runs: dict, theme: Theme, out: str) -> None:
    width_px, height_px, dpi = 2400, 1350, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(theme.ground)

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.9, "What each instrument is worth", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.845,
        "Craton-mini retrained without one instrument at a time, scored on the same frozen "
        "tests. Each cell: how much worse it gets.",
        10.5,
        theme.muted,
        va="baseline",
    )

    cols = [ablation.QUAKES, ablation.GNSS, ablation.TIDES]
    x_label, x_cols = 0.06, (0.52, 0.68, 0.84)
    y_head, y0, dy = 0.74, 0.64, 0.115
    for x, inst in zip(x_cols, cols, strict=True):
        text(
            x, y_head, f"without {ablation.NAMES[inst].lower()}", 10, theme.muted, 500, ha="center"
        )
    ax.plot([0.06, 0.94], [y_head - 0.03] * 2, color=theme.faint, linewidth=0.8 * k)

    for r, row in enumerate(SHOWN):
        y = y0 - r * dy
        title, sub = LABELS[row]
        text(x_label, y + 0.012, title, 12.5, theme.ink, 500)
        text(x_label, y - 0.035, sub, 8.5, theme.muted)
        unit = ablation.ROWS[row][3]
        for x, inst in zip(x_cols, cols, strict=True):
            s = ablation.share(runs, row, inst)
            if s is None:
                text(x, y - 0.006, "·", 14, theme.faint, ha="center")
                continue
            helps = s.beyond_noise and s.value > 0
            color = ACCENT if helps else (theme.ink if s.beyond_noise else theme.muted)
            text(
                x,
                y - 0.006,
                _fmt(s.value, unit),
                15 if helps else 12,
                color,
                600 if helps else 400,
                ha="center",
            )
            if not s.beyond_noise:
                text(x, y - 0.045, f"noise ±{s.noise:.2f}", 7.5, theme.faint, ha="center")
        if r < len(SHOWN) - 1:
            ax.plot(
                [0.06, 0.94],
                [y - dy / 2 - 0.012] * 2,
                color=theme.faint,
                linewidth=0.4 * k,
                alpha=0.6,
            )

    text(
        0.06,
        0.035,
        "Positive = the model is better with it. Orange: beyond noise (2σ over 3 seeds). "
        "Grey: within noise. ·: that variant can't answer (no GNSS, no station to rebuild; "
        "no quakes, nothing to forecast).",
        8,
        theme.muted,
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    runs = ablation.load(ROOT)
    if "all" not in runs:
        raise SystemExit("no ablation results yet: run `craton bench ablate`")
    for theme in THEMES:
        draw(runs, theme, f"ablation-{theme.name}.png")
        print(f"wrote docs/assets/ablation-{theme.name}.png")


if __name__ == "__main__":
    main()
