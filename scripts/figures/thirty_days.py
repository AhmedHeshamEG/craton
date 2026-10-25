"""Thirty days ahead: CB-19's sixteen forecasts of the 2016-17 Apennine sequence.

One column per 30-day period from 25 August 2016: the M3+ quakes that happened (grey bar, log
scale), what ETAS expected (a white or black tick) and what Craton expected (the accent).
Every forecast reads only the quakes before its period starts, so the period holding Visso
and Norcia was forecast from a sequence that was fading.

Runs both forecasts for seed 0 from the Field checkpoints `craton bench run craton-mini`
trains (it refuses to train them itself) and ETAS's fit: a few minutes.

    uv run --extra cu126 --group figures python scripts/figures/thirty_days.py
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.baselines import Etas
from craton.bench.tasks import TASKS, BenchContext, SequenceForecast
from craton.field.forecaster import FieldForecaster

SEED = 0
M_SCORED = 3.0


def series() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Period starts, observed M3+ counts, ETAS's and Craton's expected M3+ counts."""
    ctx = BenchContext(ROOT)
    task = TASKS["CB-19"]
    assert isinstance(task, SequenceForecast)
    spec = task.spec
    ff = FieldForecaster("mini")
    if not (ff.folder(ctx, spec, SEED) / "model.pt").is_file():
        raise SystemExit("no CB-19 Field checkpoint: run `craton bench run craton-mini` first")
    first = int(spec.grid.mag_bin_of(np.array([M_SCORED]))[0])
    etas_m3 = Etas().forecast(spec, ctx, SEED).expected(first).sum(axis=1)
    craton_m3 = ff.forecast(spec, ctx, SEED).expected(first).sum(axis=1)
    obs = task.observed(ctx)
    obs = obs.select(obs.mag >= M_SCORED - 1e-9)
    period = np.floor((obs.time - spec.window.t0) / spec.period).astype(np.int64)
    seen = np.bincount(period, minlength=len(spec.day_starts))
    return spec.day_starts, seen.astype(float), etas_m3, craton_m3


def draw(theme: Theme, out: str, data: tuple[np.ndarray, ...]) -> None:
    starts, seen, etas_m3, craton_m3 = data
    width_px, height_px, dpi = 2400, 1200, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.9, "Thirty days ahead", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.845,
        "Sixteen forecasts of the 2016-17 Apennine sequence, each made from only the quakes "
        "before its 30 days began. M3+ quakes per period.",
        10.5,
        theme.muted,
        va="baseline",
    )
    x0, x1, y0, y1 = 0.1, 0.94, 0.17, 0.74
    lo, hi = 0.0, math.ceil(math.log10(max(seen.max(), etas_m3.max(), craton_m3.max()) * 1.2))

    def y_of(n: float) -> float:
        return y0 + (math.log10(max(n, 1.0)) - lo) / (hi - lo) * (y1 - y0)

    for p in range(int(lo), int(hi) + 1):
        ax.plot([x0, x1], [y_of(10.0**p)] * 2, color=theme.faint, linewidth=0.5 * k)
        text(x0 - 0.012, y_of(10.0**p), f"{10**p:,}", 8, theme.muted, ha="right", va="center")
    n = len(starts)
    step = (x1 - x0) / n
    tick = ACCENT
    for i in range(n):
        cx = x0 + step * (i + 0.5)
        ax.add_patch(
            plt.Rectangle(
                (cx - step * 0.32, y0), step * 0.64, y_of(seen[i]) - y0, color=theme.faint, lw=0
            )
        )
        ax.plot(
            [cx - step * 0.4, cx + step * 0.4], [y_of(etas_m3[i])] * 2, color=theme.ink, lw=1.6 * k
        )
        ax.plot(
            [cx - step * 0.4, cx + step * 0.4], [y_of(craton_m3[i])] * 2, color=tick, lw=2.4 * k
        )
        if i % 2 == 0:
            day = datetime.fromtimestamp(float(starts[i]), UTC)
            text(cx, y0 - 0.04, day.strftime("%d %b %Y").lstrip("0"), 7.5, theme.muted, ha="center")
    # The legend, in words, top right.
    lx = 0.66
    ax.add_patch(plt.Rectangle((lx, 0.775), 0.018, 0.022, color=theme.faint, lw=0))
    text(lx + 0.026, 0.786, "what happened", 9, theme.muted, va="center")
    ax.plot([lx + 0.13, lx + 0.15], [0.786] * 2, color=theme.ink, lw=1.6 * k)
    text(lx + 0.158, 0.786, "ETAS", 9, theme.muted, va="center")
    ax.plot([lx + 0.2, lx + 0.22], [0.786] * 2, color=tick, lw=2.4 * k)
    text(lx + 0.228, 0.786, "Craton", 9, tick, 600, va="center")
    norcia = int(np.argmax(seen))
    cx = x0 + step * (norcia + 0.5)
    text(
        cx + step * 0.55,
        y_of(seen[norcia]) - 0.01,
        f"Visso and Norcia: {seen[norcia]:,.0f} quakes,\n"
        f"ETAS expected {etas_m3[norcia]:,.0f}, Craton {craton_m3[norcia]:,.0f}",
        8.5,
        theme.ink,
        va="top",
    )
    text(
        0.06,
        0.03,
        "Expected M3+ quakes in the 0.1° grid over each period, seed 0. Bits per quake for all "
        "three seeds: CB-19 on the scoreboard.",
        8,
        theme.muted,
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    data = series()
    for theme in THEMES:
        draw(theme, f"thirty-days-{theme.name}.png", data)
        print(f"wrote docs/assets/thirty-days-{theme.name}.png")


if __name__ == "__main__":
    main()
