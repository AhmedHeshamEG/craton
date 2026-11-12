"""P10: how fast the hum crossed central Italy's crust, day by day, under the 2016-17 sequence.

The network's dv/v (every station pair's 10-day stack, weighted by 1/σ²) from the
``central_italy_noise`` region's ``seismic_velocity_change`` tokens, with the four large shocks
marked and the day's M3+ quakes underneath. The band is the scatter measured on the
reference period (two standard deviations). Read from the tokens: no model is run.

    uv run --group figures python scripts/figures/velocity.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

from craton.bench.data import read_table
from craton.bench.tasks import BenchContext
from craton.bench.velocity import VelocitySeries, series_from_tokens
from craton.field.listen import COLUMNS, VELOCITY

SHOCKS = (
    ("2016-08-24", "Amatrice", 6.0),
    ("2016-10-26", "Visso", 5.9),
    ("2016-10-30", "Norcia", 6.5),
    ("2017-01-18", "Campotosto", 5.5),
)
SHOWN = ("2015-09-01", "2018-06-30")


def _day(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def network(s: VelocitySeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(days, network dv/v in %, pairs heard) with 1/σ² weights."""
    w = 1.0 / s.sigma**2
    ok = np.isfinite(s.dvv) & np.isfinite(w)
    num = np.where(ok, s.dvv * w, 0.0).sum(axis=0)
    den = np.where(ok, w, 0.0).sum(axis=0)
    net = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan) * 100.0
    return s.days, net, ok.sum(axis=0)


def data() -> dict[str, np.ndarray | float]:
    view = BenchContext(root=ROOT, region="central_italy_noise")
    s = series_from_tokens(view.region, read_table(view._snapshot_files(VELOCITY), COLUMNS))
    days, net, pairs = network(s)
    ref = days < s.reference_end
    quakes = BenchContext(root=ROOT, region="central_italy").events.above(3.0)
    t0 = _day(SHOWN[0]).timestamp()
    t1 = _day(SHOWN[1]).timestamp()
    q = quakes.window(t0, t1)
    edges = np.arange(t0, t1 + 86_400, 86_400)
    counts, _ = np.histogram(q.time, edges)
    return {
        "days": days,
        "net": net,
        "pairs": pairs.astype(float),
        "scatter": float(np.nanstd(net[ref])),
        "reference_end": s.reference_end,
        "q_days": edges[:-1],
        "q_counts": counts.astype(float),
    }


def _dates(seconds: np.ndarray) -> list[datetime]:
    return [datetime.fromtimestamp(float(x), UTC) for x in seconds]


def draw(theme: Theme, d: dict[str, np.ndarray | float]) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 6.4), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.92,
        "Four large shocks, and the hum's speed barely moved",
        color=theme.ink,
        fontproperties=bold,
        fontsize=19,
    )
    fig.text(
        0.04,
        0.855,
        "How fast seismic noise crossed central Italy's crust (dv/v, every station pair's 10-day "
        "stack, the network's weighted mean).\nThe band is two standard deviations of its "
        "scatter before the sequence. Underneath: M3+ quakes per day.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    days = np.asarray(d["days"])
    net = np.asarray(d["net"])
    t0, t1 = _day(SHOWN[0]).timestamp(), _day(SHOWN[1]).timestamp()
    sel = (days >= t0) & (days <= t1)
    ax = fig.add_axes((0.07, 0.33, 0.89, 0.43), facecolor=theme.ground)
    band = 2 * float(d["scatter"])
    x = _dates(days[sel])
    ax.fill_between(
        [_day(SHOWN[0]), _day(SHOWN[1])], -band, band, color=theme.faint, alpha=0.45, lw=0
    )
    ax.plot(x, net[sel], color=ACCENT, lw=1.3)
    ax.axhline(0, color=theme.faint, lw=0.8)
    for iso, _name, _mag in SHOCKS:
        ax.axvline(_day(iso), color=theme.muted, lw=0.8, ls=(0, (2, 3)))
    ymax = max(float(np.nanmax(np.abs(net[sel]))), band) * 1.25
    ax.set_ylim(-ymax, ymax)
    # Visso and Norcia are four days apart: the first two labels sit left of their line, the
    # last two right, on alternating rows.
    for i, (iso, name, mag) in enumerate(SHOCKS):
        left = i < 2
        ax.text(
            _day(iso),
            ymax * (0.92 - 0.14 * (i % 2)),
            f"{name} M{mag:g} " if left else f" {name} M{mag:g}",
            color=theme.ink,
            fontproperties=font,
            fontsize=8.5,
            va="top",
            ha="right" if left else "left",
        )
    ax.set_ylabel("dv/v, %", color=theme.muted, fontproperties=font)
    ref = datetime.fromtimestamp(float(d["reference_end"]), UTC)
    ax.text(
        _day(SHOWN[0]),
        -ymax * 0.92,
        f"  reference (until {ref:%d %b %Y})",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    q = fig.add_axes((0.07, 0.1, 0.89, 0.17), facecolor=theme.ground, sharex=ax)
    qx = _dates(np.asarray(d["q_days"]))
    q.bar(qx, np.asarray(d["q_counts"]), width=1.0, color=theme.mid, lw=0)
    q.set_yscale("symlog", linthresh=1)
    q.set_ylabel("M3+ / day", color=theme.muted, fontproperties=font)
    for a in (ax, q):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_color(theme.faint)
        a.spines["bottom"].set_color(theme.faint)
        a.tick_params(colors=theme.muted, labelsize=9)
    ax.tick_params(labelbottom=False)
    q.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    q.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    q.set_xlim(_day(SHOWN[0]), _day(SHOWN[1]))
    fig.text(
        0.04,
        0.02,
        "Amatrice dips about two standard deviations; Norcia goes the other way. "
        "The 20 Hz recordings (four stations, 0.5-1 Hz) say the same.",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    out = ASSETS / f"velocity-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
