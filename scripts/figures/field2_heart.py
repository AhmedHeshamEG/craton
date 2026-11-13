"""P11: how many quakes the rebuilt heart expected, in regions it never read (CB-32).

Per sequence, the quakes expected over the test year as a share of the quakes that came (M at
the region's completeness + 1): muted, ETAS fitted on the region's own past; accent, Field 2's
rebuilt heart (the lean pass: one member, seed 0). A line at 1 is "exactly what came". Beside
each row, the region the heart's member stopped on and the bits per quake it gained or lost
over ETAS. Read from ``results/ablation/field2_heart.json``: no model is run.

    uv run --group figures python scripts/figures/field2_heart.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

SEQUENCES = (
    ("cb02", "Central Italy 2016-17", "central_italy"),
    ("cb03", "L'Aquila 2009", "central_italy"),
    ("el_mayor", "El Mayor 2010", "california"),
    ("ridgecrest", "Ridgecrest 2019", "california"),
    ("tohoku", "Tohoku 2011", "japan"),
    ("kumamoto", "Kumamoto 2016", "japan"),
    ("darfield", "Darfield 2010", "new_zealand"),
    ("kaikoura", "Kaikoura 2016", "new_zealand"),
)
NAMES = {
    "central_italy": "central Italy",
    "california": "California",
    "japan": "Japan",
    "new_zealand": "New Zealand",
}
RUN = "place_lean"


def data() -> dict[str, Any]:
    abl = json.loads((ROOT / "results" / "ablation" / "field2_heart.json").read_text("utf-8"))
    heart = abl["runs"][RUN]["CB-32"]["extras"]
    born = abl["runs"]["heart-born"]["CB-32"]["extras"]
    members = abl["members"][RUN]
    rows = []
    for key, name, region in SEQUENCES:
        came = heart[f"{key}.observed"][0]
        rows.append(
            {
                "name": name,
                "etas": born[f"{key}.expected"][0] / came,
                "heart": heart[f"{key}.expected"][0] / came,
                "gain": heart[f"{key}.gain"][0],
                "stop": NAMES[members[f"s0:{region}"][0]["stop"][0]],
            }
        )
    return {"rows": rows, "mean": abl["runs"][RUN]["CB-32"]["values"][0]}


def draw(theme: Theme, d: dict[str, Any]) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 6.8), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04,
        0.925,
        "A place that knows it is busy counts right, if it stopped on the right region",
        color=theme.ink,
        fontproperties=bold,
        fontsize=18,
    )
    fig.text(
        0.04,
        0.855,
        "Quakes expected over each test year as a share of those that came, in regions the "
        "model never read (CB-32). Field 2's rebuilt heart:\na learned state per place, "
        "neighbours, rulers, trained as tested. One member, seed 0: the lean first pass.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    ax = fig.add_axes((0.24, 0.13, 0.5, 0.62), facecolor=theme.ground)
    rows = d["rows"]
    n = len(rows)
    h = 0.34
    for i, r in enumerate(rows):
        y0 = n - 1 - i
        ax.barh(y0 + h / 2, r["etas"], height=h * 0.9, color=theme.faint)
        ax.barh(y0 - h / 2, r["heart"], height=h * 0.9, color=ACCENT)
        ax.text(
            r["heart"] + 0.02,
            y0 - h / 2,
            f"{r['heart']:.0%}",
            color=ACCENT,
            fontproperties=bold,
            fontsize=9,
            va="center",
        )
        ax.text(
            r["etas"] + 0.02,
            y0 + h / 2,
            f"{r['etas']:.0%}",
            color=theme.muted,
            fontproperties=font,
            fontsize=8.5,
            va="center",
        )
        yy = 0.13 + 0.62 * (y0 + 0.5) / n
        fig.text(
            0.04, yy, r["name"], color=theme.ink, fontproperties=font, fontsize=11.5, va="center"
        )
        fig.text(
            0.77,
            yy,
            f"stopped on {r['stop']}",
            color=theme.muted,
            fontproperties=font,
            fontsize=9.5,
            va="center",
        )
        g = r["gain"]
        fig.text(
            0.92,
            yy,
            f"{g:+.2f}",
            color=ACCENT if g > 0.005 else theme.muted,
            fontproperties=bold if g > 0.005 else font,
            fontsize=11,
            va="center",
        )
    ax.axvline(1.0, color=theme.ink, lw=1.0)
    ax.text(
        1.0,
        n - 0.35,
        "what came",
        color=theme.ink,
        fontproperties=font,
        fontsize=9,
        ha="center",
        va="bottom",
    )
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_yticks([])
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=9)
    fig.text(0.92, 0.775, "bits", color=theme.muted, fontproperties=font, fontsize=9)
    fig.text(0.24, 0.79, "ETAS", color=theme.muted, fontproperties=bold, fontsize=9.5)
    fig.text(0.29, 0.79, "the rebuilt heart", color=ACCENT, fontproperties=bold, fontsize=9.5)
    fig.text(
        0.04,
        0.035,
        f"CB-32, mean over the four regions: {d['mean']:+.3f} bits per quake over ETAS (the Field: "
        "+0.336). Stopped on California, where ETAS fitted on the past over-counts the next two "
        "years,\nthe heart learned to turn ETAS down, and did so in Japan while ETAS was already "
        "three times short. New Zealand's heart stopped at its start: ETAS itself.",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    out = ASSETS / f"field2-heart-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
