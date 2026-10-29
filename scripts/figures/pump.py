"""The quakes followed the pump: CB-30's two human-made cases, period by period.

For Groningen and Oklahoma, one row each: in grey, how hard people pushed the rock (the
reservoir's pressure loss per year; the wastewater pumped into the Arbuckle per month), and
the quakes that came in every 30 days. From 2017, the test: ETAS's forecast (blind to the
pump) and Craton's (it reads the pump), both made with only the past.

Needs CB-30 prepared and the forced Field's seed-0 checkpoint (`craton bench run craton-mini
--tasks CB-30`); a few minutes.

    uv run --extra cu126 --group figures python scripts/figures/pump.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.baselines import Etas
from craton.bench.data import DAY
from craton.bench.forcing import daily, pushes
from craton.bench.induced import CASES, CUT, PERIOD_DAYS, InducedQuakes, case_spec, forecast_of
from craton.bench.tasks import BenchContext
from craton.field.forced import ForcedField
from craton.spec.timeaxis import to_seconds

SEED = 0
TITLES = {
    "groningen": ("Groningen", "gas out: the reservoir's pressure loss", "M1.5+"),
    "oklahoma": ("Oklahoma", "wastewater in: Arbuckle disposal", "M3+"),
}


def series(key: str) -> dict[str, np.ndarray]:
    ctx = BenchContext(ROOT)
    task = InducedQuakes()
    case = next(c for c in CASES if c.key == key)
    spec = case_spec(case)
    view = task.view(ctx, case, spec)
    g = spec.grid
    # Periods counted back from the cut, so the bars line up with the test's own periods.
    before = int((to_seconds(CUT) - to_seconds(case.train_from)) // (PERIOD_DAYS * DAY))
    first = to_seconds(CUT) - before * PERIOD_DAYS * DAY
    period = PERIOD_DAYS * DAY
    starts = first + period * np.arange(round((spec.window.t1 - first) / period))
    # Counted as the scorer counts: every quake in the scored magnitude bins.
    s_first = int(g.mag_bin_of(np.array([spec.scored]))[0])
    ev = view.events
    ev = ev.select((g.cell_of(ev.lat, ev.lon) >= 0) & (g.mag_bin_of(ev.mag) >= s_first))
    k = np.floor((ev.time - first) / (PERIOD_DAYS * DAY)).astype(np.int64)
    k = k[(k >= 0) & (k < len(starts))]
    observed = np.bincount(k, minlength=len(starts)).astype(float)
    d = daily(pushes(view, case.forcing, g), g)
    push = d.total_until(starts + PERIOD_DAYS * DAY).sum(axis=1) - d.total_until(starts).sum(axis=1)
    etas = forecast_of(Etas(), spec, view, case, SEED).expected(s_first).sum(axis=1)
    field = ForcedField("forced").forecast_forced(spec, view, case.forcing, SEED)
    craton = field.expected(s_first).sum(axis=1)
    test_from = round((to_seconds(CUT) - first) / (PERIOD_DAYS * DAY))
    return {
        "starts": starts,
        "observed": observed,
        "push": push,
        "etas": etas,
        "craton": craton,
        "test_from": np.array([test_from]),
    }


def draw(theme: Theme, out: str, data: dict[str, dict[str, np.ndarray]]) -> None:
    width_px, height_px, dpi = 2400, 1500, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def text(x: float, y: float, s: str, size: float, color: str, w: int = 400, **kw) -> None:
        ax.text(x, y, s, fontproperties=inter(w), fontsize=size * k, color=color, **kw)

    text(0.06, 0.925, "The quakes followed the pump", 22, theme.ink, 600, va="baseline")
    text(
        0.06,
        0.885,
        "Quakes every 30 days (bars) under how hard people pushed the rock (grey). "
        "From 2017, forecasts made with only the past.",
        10.5,
        theme.muted,
        va="baseline",
    )
    x0, x1 = 0.08, 0.95
    rows = (("groningen", 0.49, 0.80), ("oklahoma", 0.1, 0.41))
    t_lo = min(float(v["starts"][0]) for v in data.values())
    t_hi = max(float(v["starts"][-1]) for v in data.values()) + PERIOD_DAYS * DAY

    def x_of(t: float) -> float:
        return x0 + (t - t_lo) / (t_hi - t_lo) * (x1 - x0)

    for key, y0, y1 in rows:
        v = data[key]
        name, push_label, mag = TITLES[key]
        starts, obs = v["starts"], v["observed"]
        cut = int(v["test_from"][0])
        top = max(float(obs.max()), float(v["etas"].max()), float(v["craton"].max()))
        top = top * 1.1

        def y_of(n: float, y0: float = y0, y1: float = y1, top: float = top) -> float:
            return y0 + min(n, top) / top * (y1 - y0)

        push = v["push"] / max(float(v["push"].max()), 1e-30)
        xs = [x_of(t + PERIOD_DAYS * DAY / 2) for t in starts]
        ax.fill_between(
            xs, y0, [y0 + p * (y1 - y0) for p in push], color=theme.faint, lw=0, alpha=0.55
        )
        w = (x_of(starts[0] + PERIOD_DAYS * DAY) - x_of(starts[0])) * 0.7
        for t, n in zip(starts, obs, strict=True):
            if n:
                ax.add_patch(plt.Rectangle((x_of(t), y0), w, y_of(n) - y0, color=theme.mid, lw=0))
        # The forecasts are of the test periods only, from the cut on.
        tx = [x_of(t + PERIOD_DAYS * DAY / 2) for t in starts[cut : cut + len(v["etas"])]]
        ax.plot(tx, [y_of(n) for n in v["etas"]], color=theme.ink, lw=1.0 * k, alpha=0.6)
        ax.plot(tx, [y_of(n) for n in v["craton"]], color=ACCENT, lw=1.8 * k)
        xc = x_of(starts[cut])
        ax.plot([xc, xc], [y0, y1], color=theme.muted, lw=0.6 * k, ls=(0, (2, 2)))
        ax.plot([x0, x1], [y0, y0], color=theme.faint, lw=0.6 * k)
        text(x0, y1 + 0.02, name, 13, theme.ink, 600, va="baseline")
        text(
            x0 + 0.12,
            y1 + 0.02,
            f"{mag} quakes per 30 days · grey: {push_label}",
            9,
            theme.muted,
            va="baseline",
        )
        n_obs = obs[cut:].sum()
        text(
            x1,
            y1 + 0.02,
            f"2017-24: {n_obs:,.0f} came · ETAS expected {v['etas'].sum():,.0f} · "
            f"Craton {v['craton'].sum():,.0f}",
            9,
            theme.ink,
            va="baseline",
            ha="right",
        )
    for year in range(2006, 2025, 2):
        t = to_seconds(f"{year}-01-01T00:00:00Z")
        if t_lo <= t <= t_hi:
            text(x_of(t), 0.06, str(year), 8, theme.muted, ha="center")
    lx, ly = 0.62, 0.925
    ax.plot([lx, lx + 0.03], [ly, ly], color=theme.ink, lw=1.0 * k, alpha=0.6)
    text(lx + 0.037, ly, "ETAS (blind to the pump)", 8.5, theme.muted, va="center")
    ax.plot([lx + 0.19, lx + 0.22], [ly, ly], color=ACCENT, lw=1.8 * k)
    text(lx + 0.227, ly, "Craton (reads it)", 8.5, ACCENT, 600, va="center")
    text(
        0.06,
        0.035,
        "Seed 0. The same Field without the pump scores about as well on average (D-099). "
        "Pressure: NAM via TNO · volumes: OCC · quakes: KNMI, ComCat.",
        7.5,
        theme.muted,
        va="center",
    )
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)


def main() -> None:
    data = {c.key: series(c.key) for c in CASES}
    for theme in THEMES:
        draw(theme, f"pump-{theme.name}.png", data)
        print(f"wrote docs/assets/pump-{theme.name}.png")


if __name__ == "__main__":
    main()
