"""P9h: CB-32, each region forecast by a world Field that never read a single quake of it.

Per region (and the mean), the gain over ETAS fitted there, bits per quake: accent, the world
Field's fold (craton-mini's CB-32 answer, three seeds, the spread as a line); ink, the same with
dimensionless inputs (`world_dimless2`, three seeds); muted, the best floor (`etas_local_alpha`).
Read from the accepted run, `results/ablation/dimless.json` and the floor's run: no model is run.

    uv run --group figures python scripts/figures/unseen.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

REGIONS = (
    ("central_italy", "Central Italy", "2016-17, 2009"),
    ("california", "California", "El Mayor 2010, Ridgecrest 2019"),
    ("japan", "Japan", "Tohoku 2011, Kumamoto 2016"),
    ("new_zealand", "New Zealand", "Darfield 2010, Kaikoura 2016"),
)


def _run(model: str) -> dict[str, Any]:
    """The newest accepted run of ``model`` that holds CB-32."""
    for path in sorted((ROOT / "results" / "runs").glob("*.json"), reverse=True):
        body = json.loads(path.read_text("utf-8"))
        if body.get("model") != model:
            continue
        for r in body["results"]:
            if r["task_id"] == "CB-32":
                return dict(r)
    raise SystemExit(f"no accepted CB-32 run of {model}")


def data() -> dict[str, dict[str, np.ndarray]]:
    mini = _run("craton-mini")["extras"]
    floor = _run("etas_local_alpha")["extras"]
    abl = json.loads((ROOT / "results" / "ablation" / "dimless.json").read_text("utf-8"))
    rows = [r for r in abl["rows"] if r["task"] == "CB-32" and r["field"] == "dimless"]
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, *_ in REGIONS:
        k = f"{key}.mean_gain"
        out[key] = {
            "fold": np.asarray(mini[k]),
            "dimless": np.asarray([r[k] for r in rows]),
            "floor": np.asarray(floor[k]),
        }
    out["mean"] = {
        "fold": np.asarray(_run("craton-mini")["values"]),
        "dimless": np.asarray([r["value"] for r in rows]),
        "floor": np.asarray(_run("etas_local_alpha")["values"]),
    }
    return out


def draw(theme: Theme, d: dict[str, dict[str, np.ndarray]]) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 6.4), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.92,
        "Trained without the region, it still beats ETAS fitted there",
        color=theme.ink,
        fontproperties=bold,
        fontsize=19,
    )
    fig.text(
        0.04,
        0.855,
        "Gain over ETAS fitted on each box's own past, bits per quake. Each region is forecast "
        "by a world Field trained on the\nother three only. The best simple fix (ETAS refit on "
        "the last month) gains a fifth of that.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    ax = fig.add_axes((0.24, 0.13, 0.68, 0.62), facecolor=theme.ground)
    styles = (
        ("fold", "world Field, never read the region", ACCENT),
        ("dimless", "the same, dimensionless inputs", theme.ink),
        ("floor", "ETAS refit on the last month (best floor)", theme.muted),
    )
    labels = [*REGIONS, ("mean", "CB-32", "mean of the four")]
    h = 0.24
    for i, (key, name, sub) in enumerate(labels):
        y0 = len(labels) - 1 - i
        for j, (part, _, colour) in enumerate(styles):
            v = d[key][part]
            y = y0 + (1 - j) * h
            ax.barh(y, float(v.mean()), height=h * 0.86, color=colour)
            if len(v) > 1 and float(v.max() - v.min()) > 1e-6:
                ax.plot([v.min(), v.max()], [y, y], color=theme.ground, lw=1.4)
            ax.text(
                max(float(v.mean()), 0.0) + 0.012,
                y,
                f"{v.mean():+.2f}",
                color=colour,
                fontproperties=bold,
                fontsize=9,
                va="center",
            )
        fig.text(
            0.04,
            0.13 + 0.62 * (y0 + 0.5) / len(labels),
            name,
            color=theme.ink,
            fontproperties=bold,
            fontsize=12,
            va="bottom",
        )
        fig.text(
            0.04,
            0.13 + 0.62 * (y0 + 0.5) / len(labels) - 0.006,
            sub,
            color=theme.muted,
            fontproperties=font,
            fontsize=8.5,
            va="top",
        )
    ax.axvline(0, color=theme.faint, lw=0.8)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_yticks([])
    lo = min(min(float(v.min()) for v in r.values()) for r in d.values())
    ax.set_xlim(min(lo - 0.05, -0.05), 0.85)
    ax.set_xlabel("bits per quake over ETAS fitted there", color=theme.muted, fontproperties=font)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=9)
    x = 0.04
    for _, label, colour in styles:
        fig.patches.append(
            Rectangle((x, 0.786), 0.012, 0.022, transform=fig.transFigure, color=colour)
        )
        fig.text(
            x + 0.018, 0.797, label, color=colour, fontproperties=font, fontsize=9.5, va="center"
        )
        x += 0.3
    fig.text(
        0.04,
        0.02,
        "Three seeds each; the thin line across a bar is the seed range. Central Italy is CB-02's "
        "and CB-03's windows, the others CB-22's sequences.",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    out = ASSETS / f"unseen-regions-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
