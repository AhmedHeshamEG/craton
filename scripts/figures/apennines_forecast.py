"""The test the first model has to pass: the Apennines, 2016-17, one day at a time.

Bars: how many M3+ earthquakes INGV located in central Italy each day of CB-02's test window
(25 August 2016 to 31 December 2017). The orange line: how many the ETAS baseline expected
that morning, knowing only the quakes before midnight. The dashed line: the long-run Poisson
rate, which knows nothing about aftershocks.

Fits the ETAS baseline itself (seed 0, about three minutes) from the frozen CB-02 split.

    uv run --extra cu126 --group figures python scripts/figures/apennines_forecast.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.baselines import Etas, PoissonClimatology
from craton.bench.data import DAY
from craton.bench.tasks import TASKS, BenchContext, SequenceForecast

THRESHOLD = 3.0
SHOCKS = [  # (day, label) of the main shocks inside the window
    ("2016-10-26", "Visso Mw 5.9"),
    ("2016-10-30", "Norcia Mw 6.5"),
    ("2017-01-18", "Campotosto, four M5+"),
]


def load() -> dict:
    ctx = BenchContext(ROOT)
    task = TASKS["CB-02"]
    assert isinstance(task, SequenceForecast)
    spec = task.spec
    first = int(spec.grid.mag_bin_of(np.array([THRESHOLD]))[0])
    etas = Etas().forecast(spec, ctx, 0).expected(first).sum(axis=1)
    poisson = PoissonClimatology().forecast(spec, ctx, 0).expected(first).sum(axis=1)
    obs = task.observed(ctx).above(THRESHOLD)
    day = np.floor((obs.time - spec.window.t0) / DAY).astype(int)
    counts = np.bincount(day, minlength=len(spec.day_starts))
    runs = sorted((ROOT / "results" / "runs").glob("*.json"))
    scores = {}
    for path in runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        for r in run["results"]:
            if r["task_id"] == "CB-02":
                scores[run["model"]] = float(np.mean(r["values"]))
    days = [dt.datetime.fromtimestamp(t, dt.UTC).replace(tzinfo=None) for t in spec.day_starts]
    return {"days": days, "etas": etas, "poisson": poisson, "obs": counts, "scores": scores}


def draw(d: dict, theme: Theme, out: str) -> None:
    width_px, height_px, dpi = 2400, 1100, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0.07, 0.17, 0.89, 0.62))
    ax.set_facecolor(theme.ground)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=8 * k, length=0)
    ax.tick_params(which="minor", length=0)

    days = d["days"]
    obs = d["obs"].astype(float)
    shown = obs > 0
    ax.bar(
        np.array(days)[shown],
        obs[shown],
        width=1.0,
        bottom=0.03,
        color=theme.mid,
        alpha=0.75,
        linewidth=0,
    )
    ax.plot(days, d["etas"], color=ACCENT, linewidth=1.5 * k, solid_joinstyle="round")
    ax.plot(days, d["poisson"], color=theme.muted, linewidth=1.0 * k, linestyle=(0, (3, 3)))
    ax.set_yscale("log")
    ax.set_ylim(0.03, 400)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(days[0], days[-1] + dt.timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(9, 11, 1, 3, 5, 7, 9, 11)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(inter(400))
        lab.set_fontsize(8 * k)

    for i, (when, label) in enumerate(SHOCKS):
        x = dt.datetime.fromisoformat(when)
        left = i == 0  # Visso's label goes left of its bar, Norcia's right of its own
        ax.text(
            x + dt.timedelta(days=-4 if left else 4),
            260 if i != 1 else 150,
            label,
            color=theme.ink,
            fontsize=8 * k,
            fontproperties=inter(400),
            ha="right" if left else "left",
            va="center",
        )

    # The legend lives in the words, not in a box.
    ax.text(
        days[-1],
        d["poisson"][-1] * 0.8,
        "long-run rate (Poisson)",
        color=theme.muted,
        fontsize=8 * k,
        fontproperties=inter(400),
        ha="right",
        va="top",
    )
    peak = int(np.argmax(d["etas"]))
    ax.annotate(
        "ETAS expected",
        xy=(days[peak], d["etas"][peak]),
        xytext=(days[peak] + dt.timedelta(days=40), d["etas"][peak] * 1.6),
        color=ACCENT,
        fontsize=8.5 * k,
        fontproperties=inter(600),
        va="center",
        arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 0.6 * k},
    )

    fig.text(
        0.07,
        0.90,
        "The test Craton has to pass",
        color=theme.ink,
        fontsize=15 * k,
        fontproperties=inter(600),
        ha="left",
    )
    s = d["scores"]
    note = (
        "M3+ earthquakes per day in central Italy, 25 Aug 2016 to 31 Dec 2017 (CratonBench "
        "CB-02). Each morning's forecast sees only the past."
    )
    fig.text(0.07, 0.845, note, color=theme.muted, fontsize=9 * k, fontproperties=inter(400))
    if "etas" in s and "poisson" in s:
        gain = s["etas"] - s["poisson"]

        def minus(x: float) -> str:
            return f"{x:.2f}".replace("-", "−")

        foot = (
            f"Log-likelihood per M3+ quake: ETAS {minus(s['etas'])} bits, Poisson "
            f"{minus(s['poisson'])} bits. ETAS gains {gain:.1f} bits per quake; "
            "that is the floor."
        )
        fig.text(0.07, 0.05, foot, color=theme.muted, fontsize=8.5 * k, fontproperties=inter(400))
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)
    print("wrote", ASSETS / out)


def main() -> None:
    d = load()
    for theme in THEMES:
        draw(d, theme, f"apennines-forecast-{theme.name}.png")


if __name__ == "__main__":
    main()
