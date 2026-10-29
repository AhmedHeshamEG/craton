"""What the routine network missed: Norcia's first three days, hour by hour.

M2+ quakes per hour after the Mw 6.5 Norcia mainshock (30 October 2016, 06:40 UTC). Grey: the
quakes in INGV's routine catalogue. The accent stacked on top: the quakes Tan et al.'s
machine-learning catalogue found that INGV doesn't have, on INGV's magnitude scale (D-094).
That stack is what CB-20 adds to every forecast's history.

Reads the prepared CB-20 union (`craton bench prepare`); a minute.

    uv run --extra cu126 --group figures python scripts/figures/missed.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.enhanced import EnhancedForecast
from craton.bench.tasks import TASKS, BenchContext
from craton.spec.timeaxis import to_seconds

NORCIA = to_seconds("2016-10-30T06:40:18Z")
HOURS = 72
M_MIN = 2.0


def series() -> tuple[np.ndarray, np.ndarray]:
    """M2+ quakes per hour after Norcia: INGV's, and the ones the union adds."""
    ctx = BenchContext(ROOT)
    task = TASKS["CB-20"]
    assert isinstance(task, EnhancedForecast)
    built = task.context(ctx).built
    base = ctx.events

    def per_hour(t: np.ndarray, m: np.ndarray) -> np.ndarray:
        keep = (t >= NORCIA) & (t < NORCIA + HOURS * 3600) & (m >= M_MIN - 1e-9)
        h = np.floor((t[keep] - NORCIA) / 3600).astype(np.int64)
        return np.bincount(h, minlength=HOURS).astype(float)

    # Only quakes in CB-02's grid count, as in the forecasts.
    grid = task.spec.grid

    def inside(ev):  # type: ignore[no-untyped-def]
        return ev.select(grid.cell_of(ev.lat, ev.lon) >= 0)

    b, a = inside(base), inside(built.added)
    return per_hour(b.time, b.mag), per_hour(a.time, a.mag)


def draw(theme: Theme, out: str, data: tuple[np.ndarray, np.ndarray]) -> None:
    ingv, added = data
    width_px, height_px, dpi = 2400, 1200, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.9, "What the routine network missed", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.845,
        "M2+ quakes per hour in the three days after the Mw 6.5 Norcia earthquake, "
        "30 October 2016.",
        10.5,
        theme.muted,
        va="baseline",
    )
    x0, x1, y0, y1 = 0.1, 0.94, 0.17, 0.74
    top = float((ingv + added).max())
    step_y = 50 if top <= 300 else 100
    ymax = np.ceil(top / step_y) * step_y

    def y_of(n: float) -> float:
        return y0 + n / ymax * (y1 - y0)

    for v in np.arange(0, ymax + 1, step_y):
        ax.plot([x0, x1], [y_of(v)] * 2, color=theme.faint, linewidth=0.5 * k)
        text(x0 - 0.012, y_of(v), f"{v:,.0f}", 8, theme.muted, ha="right", va="center")
    step = (x1 - x0) / HOURS
    for i in range(HOURS):
        cx = x0 + step * i
        ax.add_patch(
            plt.Rectangle(
                (cx + step * 0.1, y0), step * 0.8, y_of(ingv[i]) - y0, color=theme.mid, lw=0
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (cx + step * 0.1, y_of(ingv[i])),
                step * 0.8,
                y_of(added[i]) - y0,
                color=ACCENT,
                lw=0,
            )
        )
    for h in (0, 12, 24, 36, 48, 60, 72):
        text(x0 + step * h, y0 - 0.04, f"{h} h", 8, theme.muted, ha="center")
    day1 = slice(0, 24)
    lx = 0.6
    ax.add_patch(plt.Rectangle((lx, 0.775), 0.018, 0.022, color=theme.mid, lw=0))
    text(lx + 0.026, 0.786, "in INGV's catalogue", 9, theme.muted, va="center")
    ax.add_patch(plt.Rectangle((lx + 0.19, 0.775), 0.018, 0.022, color=ACCENT, lw=0))
    text(lx + 0.216, 0.786, "found by machine learning", 9, ACCENT, 600, va="center")
    text(
        x0 + step * 26,
        y_of(top * 0.72),
        f"First day: INGV has {ingv[day1].sum():,.0f}.\n"
        f"Machine learning finds {added[day1].sum():,.0f} more.",
        11,
        theme.ink,
        600,
        va="top",
    )
    text(
        0.06,
        0.03,
        "Added quakes: Tan et al. (2021), not matched to an INGV quake within 3 s and 10 km, "
        "magnitudes mapped onto INGV's scale. CB-02's grid.",
        8,
        theme.muted,
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    data = series()
    for theme in THEMES:
        draw(theme, f"missed-{theme.name}.png", data)
        print(f"wrote docs/assets/missed-{theme.name}.png")


if __name__ == "__main__":
    main()
