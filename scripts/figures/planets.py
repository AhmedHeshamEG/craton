"""Two fake planets and the real Apennines, side by side.

Every M3+ quake from 1985 to the end of 2017, placed by how far along the chain it is (north-
west at the top, south-east at the bottom) and when it happened. The top band is INGV's real
catalogue; the other two are CB-09's first planet of each kind: the ETAS planet, where
Italy's own fitted ETAS is the whole truth, and the physics planet, where faults, stress
transfer and a hidden fluid source make the quakes. M5+ in the accent.

Reads the frozen CB-09 planets and the snapshot (`craton bench prepare`); computes nothing new.

    uv run --extra cu126 --group figures python scripts/figures/planets.py
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench import planets
from craton.bench.data import Events
from craton.bench.tasks import TASKS, BenchContext
from craton.sim.rate_state import APENNINE_AXIS, KM_PER_DEG
from craton.spec.timeaxis import to_seconds

YEAR_S = 365.25 * 86_400.0


def along_chain(ev: Events) -> np.ndarray:
    """Kilometres along the chain from its north-western end."""
    (lat0, lon0), (lat1, lon1) = APENNINE_AXIS[0], APENNINE_AXIS[-1]
    kx = KM_PER_DEG * math.cos(math.radians((lat0 + lat1) / 2))
    ax, ay = (lon1 - lon0) * kx, (lat1 - lat0) * KM_PER_DEG
    norm = math.hypot(ax, ay)
    return ((ev.lon - lon0) * kx * ax + (ev.lat - lat0) * KM_PER_DEG * ay) / norm


def bands() -> list[tuple[str, str, Events]]:
    ctx = BenchContext(ROOT)
    task = TASKS["CB-09"]
    assert isinstance(task, planets.TwoPlanets)
    t0, t1 = to_seconds(planets.PLANET_START), planets.WINDOW.t1
    real = planets.in_grid(ctx.events.window(t0, t1))
    return [
        ("The real Apennines", "INGV, 1985-2017", real),
        (
            "The ETAS planet",
            "Italy's own ETAS is the whole truth",
            task.planet(ctx, "etas_planet", 0).events,
        ),
        (
            "The physics planet",
            "faults, stress transfer, a hidden fluid source",
            task.planet(ctx, "physics_planet", 0).events,
        ),
    ]


def draw(theme: Theme, out: str, data: list[tuple[str, str, Events]]) -> None:
    width_px, height_px, dpi = 2400, 1500, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    top = fig.add_axes((0, 0, 1, 1))
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        top.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.925, "Two fake planets and the real Apennines", 22, theme.ink, 600)
    text(
        0.06,
        0.88,
        "Every M3+ quake from 1985 to 2017, placed along the chain. Craton has to tie ETAS on "
        "the middle one and beat it on the bottom one.",
        10.5,
        theme.muted,
    )
    t0 = to_seconds(planets.PLANET_START)
    h, gap, y_top = 0.215, 0.05, 0.8
    for i, (title, sub, ev) in enumerate(data):
        y0 = y_top - (i + 1) * h - i * gap
        ax = fig.add_axes((0.06, y0, 0.66, h), facecolor=theme.ground)
        ev = ev.above(3.0)
        x = 1985 + (ev.time - t0) / YEAR_S
        y = along_chain(ev)
        small = ev.mag < 5.0
        ax.scatter(
            x[small],
            y[small],
            s=(0.6 + 1.6 * (ev.mag[small] - 3.0)) * k,
            color=theme.mid,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            x[~small], y[~small], s=14 * k, color=ACCENT, linewidths=0, zorder=3, rasterized=True
        )
        ax.set_xlim(1985, 2018)
        ax.set_ylim(420, -40)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme.faint)
        ax.set_yticks([])
        ax.tick_params(colors=theme.muted, length=0, labelsize=8 * k)
        if i < len(data) - 1:
            ax.set_xticklabels([])
        for lab in ax.get_xticklabels():
            lab.set_fontproperties(inter(400))
        ty = y0 + h * 0.62
        text(0.75, ty, title, 13, theme.ink, 600)
        text(0.75, ty - 0.04, sub, 9, theme.muted)
        big = int((~small).sum())
        text(
            0.75,
            ty - 0.08,
            f"{len(ev):,} M3+ quakes · {big} of M5+ · largest M{ev.mag.max():.1f}",
            9,
            theme.muted,
        )
    text(0.75, 0.035, "north-west at the top of each band, south-east below", 8, theme.muted)
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


if __name__ == "__main__":
    data = bands()
    for th in THEMES:
        draw(th, f"planets-{th.name}.png", data)
        print(f"wrote planets-{th.name}.png")
