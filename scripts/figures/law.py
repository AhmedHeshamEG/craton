"""CB-26 (P9e): the rule inside the Field, and the check that it reads time.

Left, the shortest law symbolic regression found (CB-03, seed 0, one term): ETAS's rate times
exp(k · N3y / N12y), N the cell's quakes fading over 3 and 12 years. Drawn as the factor it puts
on ETAS against the ratio, bounded the Field's way. Right, the share of the network's held-out
gain the formulas keep, and what the network gains when its training answers are dealt to the
wrong days (the leak audit's control).

Reads results/laws/mini.json, the latest craton-mini run and results/ablation/leak_audit.json.

    uv run --group figures python scripts/figures/law.py
"""

from __future__ import annotations

import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.field.distill import _evaluate, input_names

LAW = ("CB-03", "0")


def data() -> dict[str, object]:
    laws = json.loads((ROOT / "results" / "laws" / "mini.json").read_text("utf-8"))
    law = laws[LAW[0]][LAW[1]]
    names = input_names()
    x = np.zeros((1, len(names)))
    k = float(_evaluate(law["expression"], x)[0])  # n_3y = n_12y: the ratio is 1
    run = sorted((ROOT / "results" / "runs").glob("*-craton-mini.json"))[-1]
    body = {r["task_id"]: r for r in json.loads(run.read_text("utf-8"))["results"]}
    ex = body["CB-26"]["extras"]
    audit = json.loads((ROOT / "results" / "ablation" / "leak_audit.json").read_text("utf-8"))
    floor_run = sorted((ROOT / "results" / "runs").glob("*-law-constant.json"))[-1]
    floor = next(
        r for r in json.loads(floor_run.read_text("utf-8"))["results"] if r["task_id"] == "CB-26"
    )
    return {
        "share": float(np.mean(body["CB-26"]["values"])),
        "floor": float(np.mean(floor["values"])),
        "expression": law["expression"],
        "k": k,
        "max_log": float(law["max_log"]),
        "share02": float(np.mean(ex["CB-02.share"])),
        "share03": float(np.mean(ex["CB-03.share"])),
        "terms02": (int(min(ex["CB-02.terms"])), int(max(ex["CB-02.terms"]))),
        "terms03": (int(min(ex["CB-03.terms"])), int(max(ex["CB-03.terms"]))),
        "shuffled": float(np.mean([r["gain"] for r in audit["audit_shuffled"].values()])),
        "mini": float(np.mean([r["gain"] for r in audit["mini"].values()])),
        "cut": float(np.mean([r["gain"] for r in audit["audit_cut"].values()])),
    }


def draw(theme: Theme, d: dict[str, object]) -> None:
    font, bold = inter(400), inter(600)
    k, b = float(d["k"]), float(d["max_log"])  # type: ignore[arg-type]
    fig = plt.figure(figsize=(12, 7.2), dpi=200, facecolor=theme.ground)
    fig.text(
        0.04, 0.92, "The rule inside the Field", color=theme.ink, fontproperties=bold, fontsize=20
    )
    fig.text(
        0.04,
        0.875,
        "Symbolic regression on what the network adds to ETAS. The shortest law it found, one "
        "term:",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    fig.text(
        0.04,
        0.78,
        f"rate  =  ETAS  ×  exp( {k:.2f} · N(3 y) / N(12 y) )",
        color=theme.ink,
        fontproperties=bold,
        fontsize=22,
    )
    fig.text(
        0.04,
        0.735,
        "N(3 y), N(12 y): the cell's quakes, each fading over 3 and over 12 years",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )

    ax = fig.add_axes((0.07, 0.12, 0.45, 0.52), facecolor=theme.ground)
    r = np.linspace(0, 1, 200)
    factor = np.exp(b * np.tanh(k * r / b))
    ax.plot(r, factor, color=ACCENT, lw=2.4)
    # (ratio, label, offset in points, alignment): every label clear of the line and the axis
    marks = (
        (0.0, "gone quiet: ETAS as it is", (10, 14), "left"),
        (0.25, "steady for 12 years", (10, -10), "left"),
        (1.0, "all of it recent: waking up", (-10, 14), "right"),
    )
    for x0, label, off, ha in marks:
        y0 = math.exp(b * math.tanh(k * x0 / b))
        ax.scatter([x0], [y0], s=36, color=ACCENT, zorder=3)
        ax.annotate(
            f"×{y0:.1f}  {label}",
            (x0, y0),
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va="center",
            color=theme.ink,
            fontproperties=font,
            fontsize=9.5,
        )
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0.85, math.exp(b * math.tanh(k / b)) + 0.45)
    ax.set_xlabel(
        "N(3 y) / N(12 y)   (how much of the last 12 years happened in the last 3)",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )
    ax.set_ylabel("factor on ETAS's rate", color=theme.muted, fontproperties=font, fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=8.5)

    x0, y = 0.62, 0.62
    fig.text(x0, y, "Kept on years it never saw", color=theme.ink, fontproperties=bold, fontsize=13)
    y -= 0.1
    lo = min(d["terms02"][0], d["terms03"][0])  # type: ignore[index]
    hi = max(d["terms02"][1], d["terms03"][1])  # type: ignore[index]
    rows = (
        (
            f"{100 * float(d['share']):.0f}%",
            f"of the network's gain over ETAS, kept by formulas\nof {lo}-{hi} terms "
            "(CB-26: Apennines 2016-17, L'Aquila 2009)",
            ACCENT,
        ),
        (
            f"{100 * float(d['floor']):.0f}%",
            "kept by one constant factor on ETAS: most of the\ngain is more quakes; the rule "
            "is the rest",
            theme.ink,
        ),
    )
    for big, about, colour in rows:
        fig.text(x0, y, big, color=colour, fontproperties=bold, fontsize=28)
        fig.text(
            x0, y - 0.015, about, color=theme.muted, fontproperties=font, fontsize=9.5, va="top"
        )
        y -= 0.16
    fig.text(x0, y, "The leak check", color=theme.ink, fontproperties=bold, fontsize=13)
    y -= 0.07
    checks = (
        (f"{float(d['mini']):+.2f}", "bits per quake over ETAS, the Field as trained"),
        (f"{float(d['cut']):+.2f}", "trained as if the data ended a year earlier"),
        (f"{float(d['shuffled']):+.2f}", "trained on answers dealt to the wrong days"),
    )
    for big, about in checks:
        big = big.replace("-", "−")  # a true minus sign
        fig.text(x0, y, big, color=theme.ink, fontproperties=bold, fontsize=15)
        fig.text(x0 + 0.075, y + 0.004, about, color=theme.muted, fontproperties=font, fontsize=9.5)
        y -= 0.05
    out = ASSETS / f"law-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    d = data()
    for theme in THEMES:
        draw(theme, d)


if __name__ == "__main__":
    main()
