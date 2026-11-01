"""CB-24 (P9b): the order central Italy lit up in 2016-17, and how far a ruler gets.

Every square is a 0.1° cell at the moment of its first M3+ quake after the Amatrice shock,
brighter the earlier it lit; the grey dots are the quakes before the sequence, all a method may
read besides the lightings it is shown. The four big shocks are named. On the right, how often
the simple methods put the cell that really lit next in their top 3, over the sixteen sequences
of CB-24 (a coin: 3 in ~1,000).

Reads CB-24's frozen lightings, the central Italy snapshot and the latest run of each floor in
`results/runs/`.

    uv run --group figures python scripts/figures/migration.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.bench.data import DAY
from craton.bench.migration import Migration, cases
from craton.bench.tasks import BenchContext
from craton.spec.timeaxis import to_seconds

KEY = "amatrice_norcia"
#: The big shocks, for labels only (INGV; UTC).
SHOCKS = (
    ("Amatrice", "24 Aug 2016", 42.70, 13.23),
    ("Visso", "26 Oct", 42.91, 13.13),
    ("Norcia", "30 Oct", 42.83, 13.11),
    ("Campotosto", "18 Jan 2017", 42.53, 13.28),
)
FLOORS = (
    ("distance_only", "nearest to the cell that lit last"),
    ("quake_history", "the busiest cells before the sequence"),
)
VIEW = (41.95, 43.45, 12.45, 13.85)  # lat0, lat1, lon0, lon1


def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ctx = BenchContext(ROOT)
    task = Migration()
    case = next(c for c in cases(ROOT) if c.key == KEY)
    lit = task.lightings(ctx, case)
    days = (lit.quakes.time - to_seconds(case.cut)) / DAY
    la, lo = case.grid.cell_centres()
    ev = ctx.events
    before = ev.select(ev.time < to_seconds(case.cut)).above(case.threshold)
    return la[lit.cell], lo[lit.cell], days, before.lon, before.lat


def floors() -> dict[str, float]:
    out: dict[str, float] = {}
    for name, _ in FLOORS:
        for path in sorted(
            (ROOT / "results" / "runs").glob(f"*-{name.replace('_', '-')}.json"), reverse=True
        ):
            body = {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["results"]}
            if "CB-24" in body:
                out[name] = float(np.mean(body["CB-24"]["values"]))
                break
    return out


def draw(theme: Theme, d: tuple[np.ndarray, ...], f: dict[str, float]) -> None:
    lat, lon, days, qx, qy = d
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 9), dpi=200, facecolor=theme.ground)
    ax = fig.add_axes((0.03, 0.05, 0.56, 0.8), facecolor=theme.ground)
    lat0, lat1, lon0, lon1 = VIEW
    ax.scatter(qx, qy, s=1.2, c=theme.mid, alpha=0.5, linewidths=0, rasterized=True)
    # Earlier = brighter: the order is in the light, one accent only.
    order = np.argsort(-days)  # draw late ones first, early on top
    rank = np.argsort(np.argsort(days)) / max(len(days) - 1, 1)
    colours = [to_rgba(ACCENT, 1.0 - 0.75 * rank[i]) for i in order]
    # Each cell at its true extent, a hair inside so neighbours stay apart.
    g = 0.1 * 0.94
    for k in order:
        ax.add_patch(
            plt.Rectangle(
                (lon[k] - g / 2, lat[k] - g / 2),
                g,
                g,
                facecolor=colours[list(order).index(k)],
                linewidth=0,
                zorder=3,
            )
        )
    for name, when, la, lo in SHOCKS:
        ax.annotate(
            f"{name}\n{when}",
            (lo, la),
            xytext=(
                lo + (0.42 if name == "Amatrice" else -0.55),
                la + {"Amatrice": -0.05, "Visso": 0.12, "Norcia": 0.0}.get(name, -0.16),
            ),
            color=theme.ink,
            fontproperties=font,
            fontsize=9.5,
            arrowprops={"arrowstyle": "-", "color": theme.muted, "lw": 0.7},
            zorder=4,
        )
    mid = (lat0 + lat1) / 2
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians(mid)), adjustable="box", anchor="W")
    ax.axis("off")

    x0, y = 0.6, 0.62
    fig.text(
        x0, y, "Is the next cell in the top 3?", color=theme.ink, fontproperties=bold, fontsize=13
    )
    fig.text(
        x0,
        y - 0.03,
        "sixteen sequences, central Italy to Alaska",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )
    y -= 0.11
    for name, about in FLOORS:
        if name not in f:
            continue
        fig.text(x0, y, f"{100 * f[name]:.0f}%", color=theme.ink, fontproperties=bold, fontsize=26)
        fig.text(x0, y - 0.03, about, color=theme.muted, fontproperties=font, fontsize=9.5)
        y -= 0.11
    fig.text(x0, y, "0.3%", color=theme.muted, fontproperties=bold, fontsize=26)
    fig.text(x0, y - 0.03, "a coin", color=theme.muted, fontproperties=font, fontsize=9.5)
    fig.text(
        x0,
        0.2,
        "Craton's turn: P9d, the wiring learned\nfrom decades of quakes before the sequence.",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )
    for i, (label, a) in enumerate((("first", 1.0), ("last", 0.25))):
        fig.patches.append(
            plt.Rectangle(
                (x0 + 0.13 * i, 0.73),
                0.018,
                0.024,
                transform=fig.transFigure,
                facecolor=to_rgba(ACCENT, a),
                linewidth=0,
            )
        )
        fig.text(
            x0 + 0.025 + 0.13 * i,
            0.735,
            f"lit {label}",
            color=theme.muted,
            fontproperties=font,
            fontsize=9.5,
        )
    fig.text(
        0.03,
        0.93,
        "Where the sequence went next",
        color=theme.ink,
        fontproperties=bold,
        fontsize=20,
    )
    fig.text(
        0.03,
        0.895,
        f"The {len(days)} cells of 0.1° that lit up (a first M3+ quake) in the year after "
        "Amatrice; "
        "brighter = earlier. Grey: the quakes before it.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    out = ASSETS / f"migration-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    f = floors()
    for theme in THEMES:
        draw(theme, d, f)


if __name__ == "__main__":
    main()
