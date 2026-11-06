"""P9g: craton-mini against ETAS with a local alpha, day by day, on the two held-out sequences.

Each line is the running sum of a forecast's log-probability of the day's quakes minus ETAS's,
in bits per quake of the whole window, so its end is the gain over ETAS. Accent: craton-mini;
ink: ETAS with its productivity refit every day on the last month (Lombardi & Marzocchi 2010);
muted: the same refitting alpha alone. Seed 0.

    uv run --extra cu126 --group figures python scripts/figures/local_alpha.py
"""

from __future__ import annotations

import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.baselines import Etas
from craton.bench.data import DAY
from craton.bench.local_alpha import EtasLocalAlpha
from craton.bench.tasks import TASKS, BenchContext
from craton.field.forecaster import FieldForecaster

SEED = 0
PANELS = (("CB-02", "Apennines 2016-17"), ("CB-03", "L'Aquila 2009"))


def daily_bits(fc: Any, obs: Any, spec: Any) -> np.ndarray:
    """(days,) log2-probability of each day's quakes at or above the scored magnitude."""
    first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
    ev = obs.above(spec.scored - 1e-9)
    day = np.floor((ev.time - spec.day_starts[0]) / DAY).astype(int)
    cell = spec.grid.cell_of(ev.lat, ev.lon)
    mbin = spec.grid.mag_bin_of(ev.mag)
    ok = (day >= 0) & (day < len(spec.day_starts)) & (cell >= 0) & (mbin >= first)
    out = -fc.expected(first).sum(axis=1)
    np.add.at(out, day[ok], np.log(fc.at(day[ok], cell[ok], mbin[ok])))
    return np.asarray(out / np.log(2))


def data() -> dict[str, dict[str, Any]]:
    ctx = BenchContext(ROOT)
    methods = {
        "mini": FieldForecaster("mini"),
        "local": EtasLocalAlpha(),
        "alpha": EtasLocalAlpha(refit_k=False),
    }
    out = {}
    for tid, _ in PANELS:
        task: Any = TASKS[tid]
        spec, obs = task.spec, task.observed(ctx)
        n = len(obs.above(spec.scored - 1e-9))
        base = daily_bits(Etas().forecast(spec, ctx, SEED), obs, spec)
        lines = {
            k: np.cumsum(daily_bits(m.forecast(spec, ctx, SEED), obs, spec) - base) / n
            for k, m in methods.items()
        }
        big = obs.above(5.0 - 1e-9)
        lines["big_days"] = np.floor((big.time - spec.day_starts[0]) / DAY)
        lines["t0"] = np.array([spec.day_starts[0]])
        out[tid] = lines
    return out


def draw(theme: Theme, d: dict[str, dict[str, Any]]) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 6.2), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.92,
        "ETAS refit on the last month matches the network on 2016, half of it on 2009",
        color=theme.ink,
        fontproperties=bold,
        fontsize=19,
    )
    fig.text(
        0.04,
        0.855,
        "Gain over ETAS, added up day by day (bits per quake). ETAS refit every day on the last "
        "month's quakes,\nthe fix Lombardi & Marzocchi (2010) pointed to, gains on the same "
        "days as the network. On L'Aquila the network finds as much again.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    styles = (
        ("mini", "craton-mini", ACCENT, "-", 2.2),
        ("local", "ETAS, K and alpha refit daily", theme.ink, "-", 1.6),
        ("alpha", "ETAS, alpha alone refit daily", theme.muted, "--", 1.3),
    )
    for k, (tid, title) in enumerate(PANELS):
        lines = d[tid]
        ax = fig.add_axes((0.06 + k * 0.48, 0.13, 0.42, 0.6), facecolor=theme.ground)
        x = np.arange(len(lines["mini"]))
        for b in lines["big_days"]:
            ax.axvline(b, color=theme.faint, lw=0.8, zorder=0)
        ends = {key: float(lines[key][-1]) for key, *_ in styles}
        step = 0.06 * (max(ends.values()) - min(min(ends.values()), 0.0))
        for key, label, colour, ls, lw in styles:
            ax.plot(x, lines[key], color=colour, ls=ls, lw=lw, label=label)
            y = ends[key]
            # an end label closer than `step` below another one steps down out of its way
            if any(o != key and (ends[o], o) > (y, key) and ends[o] - y < step for o in ends):
                y -= step
            ax.text(
                x[-1] + 4,
                y,
                f"{lines[key][-1]:+.2f}",
                color=colour,
                fontproperties=bold,
                fontsize=10,
                va="center",
            )
        ax.axhline(0, color=theme.faint, lw=0.8)
        ax.set_xlim(0, x[-1] * 1.1)
        ax.set_title(
            f"{tid} · {title}", color=theme.ink, fontproperties=bold, fontsize=12, loc="left"
        )
        ax.set_xlabel("days into the test window", color=theme.muted, fontproperties=font)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(theme.faint)
        ax.tick_params(colors=theme.muted, labelsize=9)
        if k == 0:
            leg = ax.legend(frameon=False, loc="lower right", prop=font, fontsize=9)
            for t in leg.get_texts():
                t.set_color(theme.ink)
    fig.text(
        0.06,
        0.02,
        "Grey verticals: days with an M5+ quake. Seed 0; the floors score the same on every seed.",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    out = ASSETS / f"local-alpha-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
