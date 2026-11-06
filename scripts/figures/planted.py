"""CB-33 (P9f): the physics planet's hidden state, next to what a forecast implies about it.

Planet 0, the first graded midnight after its largest quake (an M5.7). Left, the truth the
planet wrote down (each cell's rate over its own background, ln R / r: orange above, blue
below, the shadows). Middle, the same reading of craton-mini's forecast that CB-33 grades.
Right, the floor: the cell's last year of quakes over its long-run rate. Only the graded cells
(at least 10% off steady) are drawn, coloured by rank within the day, since ranks are what is
scored. Under the maps, CB-33 per planet for craton-mini and the floors.

    uv run --extra cu126 --group figures python scripts/figures/planted.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, spearmanr
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
import craton.train.method  # noqa: F401  (registers craton-mini)
from craton.bench import planted
from craton.bench.baselines import METHODS
from craton.bench.data import DAY
from craton.bench.planets import WINDOW
from craton.bench.tasks import TASKS, BenchContext, forecast_grid

SHADOW_BLUE = "#4f8fe6"
PLANET = 0
METHODS_SHOWN = (
    ("craton-mini", "craton-mini"),
    ("quake_density", "quake density"),
    ("etas", "ETAS"),
    ("crs", "Coulomb rate-and-state"),
)


def board() -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, _ in METHODS_SHOWN:
        run = sorted((ROOT / "results" / "runs").glob(f"*-{name.replace('_', '-')}.json"))[-1]
        body = json.loads(run.read_text("utf-8"))
        r = next(r for r in body["results"] if r["task_id"] == "CB-33")
        out[name] = [float(v) for v in r["values"]]
    return out


def data() -> dict[str, object]:
    ctx = BenchContext(root=ROOT)
    task = TASKS["CB-33"]
    assert isinstance(task, planted.PlantedStress)
    truth = task.truth(ctx, PLANET)
    mini = task.answer(METHODS["craton-mini"], ctx, PLANET, PLANET)
    floor = task.answer(METHODS["quake_density"], ctx, PLANET, PLANET)
    days = planted.graded_days()
    test = task.cb09.planet(ctx, planted.KIND, PLANET).events.window(WINDOW.t0, WINDOW.t1)
    big_day = int((test.time[int(np.argmax(test.mag))] - WINDOW.t0) // DAY)
    i = int(np.flatnonzero(days > big_day)[0])  # the first graded midnight after the largest
    return {
        "truth": truth[i],
        "mini": mini[i],
        "floor": floor[i],
        "day": int(days[i]),
        "big": float(test.mag.max()),
        "big_day": big_day,
        "rho_mini": float(
            spearmanr(mini[i][np.abs(truth[i]) >= 0.1], truth[i][np.abs(truth[i]) >= 0.1])[0]
        ),
        "rho_floor": float(
            spearmanr(floor[i][np.abs(truth[i]) >= 0.1], truth[i][np.abs(truth[i]) >= 0.1])[0]
        ),
        "board": board(),
    }


def ranked(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Values of the graded cells as ranks from −1 (lowest) to +1 (highest), NaN elsewhere."""
    out = np.full(v.shape, np.nan)
    r = rankdata(v[m])
    out[m] = 2 * (r - 1) / max(len(r) - 1, 1) - 1
    return out


def draw(theme: Theme, d: dict[str, object]) -> None:
    font, bold = inter(400), inter(600)
    g = forecast_grid()
    truth = np.asarray(d["truth"])
    m = np.abs(truth) >= planted.OFF_STEADY
    rows, cols = np.divmod(np.flatnonzero(m), g.n_lon)
    pad = 4
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, g.n_lat)
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, g.n_lon)
    cmap = LinearSegmentedColormap.from_list("planted", [SHADOW_BLUE, theme.faint, ACCENT])
    cmap.set_bad(theme.ground)

    fig = plt.figure(figsize=(12, 7.6), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.93,
        "A crust nobody can see, and what the forecasts say about it",
        color=theme.ink,
        fontproperties=bold,
        fontsize=19,
    )
    when = WINDOW.t0 + int(d["day"]) * DAY  # type: ignore[call-overload]
    fig.text(
        0.04,
        0.862,
        f"Physics planet 0, {np.datetime64(int(when), 's').astype('datetime64[D]')} on its "
        f"calendar, {int(d['day']) - int(d['big_day'])} days after its "  # type: ignore[call-overload]
        f"M{float(d['big']):.1f}. Each cell: how far its quake rate sits from its own "
        "normal.\nOnly the cells at least 10% off are graded; colour = rank that day.",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
        wrap=True,
    )
    panels = (
        ("The planted truth", truth, "the planet's own record, never shown to anyone"),
        ("craton-mini", np.asarray(d["mini"]), f"rank correlation {float(d['rho_mini']):.2f}"),
        (
            "Quake density",
            np.asarray(d["floor"]),
            f"the floor · rank correlation {float(d['rho_floor']):.2f}",
        ),
    )
    for k, (title, v, note) in enumerate(panels):
        ax = fig.add_axes((0.04 + k * 0.32, 0.36, 0.28, 0.44), facecolor=theme.ground)
        img = ranked(v, m).reshape(g.n_lat, g.n_lon)[r0:r1, c0:c1]
        ax.imshow(img, origin="lower", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(theme.faint)
        ax.set_title(
            title,
            color=ACCENT if k == 0 else theme.ink,
            fontproperties=bold,
            fontsize=12,
            loc="left",
        )
        ax.text(
            0,
            -0.06,
            note,
            transform=ax.transAxes,
            color=theme.muted,
            fontproperties=font,
            fontsize=9,
            va="top",
        )
    fig.text(
        0.04,
        0.275,
        "blue: stress shadow (below normal)   ·   orange: above normal   ·   "
        f"cells are 0.1° (about 10 km); box {c1 - c0} × {r1 - r0} cells",
        color=theme.muted,
        fontproperties=font,
        fontsize=9,
    )

    # CB-33 per planet
    b = d["board"]
    assert isinstance(b, dict)
    fig.text(
        0.04,
        0.215,
        "CB-33 on all three planets (rank correlation, higher is better)",
        color=theme.ink,
        fontproperties=bold,
        fontsize=12,
    )
    y = 0.165
    for name, label in METHODS_SHOWN:
        vals = b[name]
        colour = ACCENT if name == "craton-mini" else theme.ink
        fig.text(
            0.04,
            y,
            label,
            color=colour,
            fontproperties=bold if colour == ACCENT else font,
            fontsize=10,
        )
        fig.text(0.27, y, f"{np.mean(vals):.2f}", color=colour, fontproperties=bold, fontsize=11)
        for j, v in enumerate(vals):
            fig.text(
                0.35 + 0.07 * j,
                y,
                f"{v:.2f}",
                color=theme.muted,
                fontproperties=font,
                fontsize=10,
                ha="right",
            )
        y -= 0.038
    for j in range(len(vals)):
        fig.text(
            0.35 + 0.07 * j,
            0.19,
            f"planet {j}",
            color=theme.muted,
            fontproperties=font,
            fontsize=8.5,
            ha="right",
        )
    out = ASSETS / f"planted-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
