"""Where Craton stands against ETAS, on everything scored so far.

Three panels. Top left: every forecasting task, as bits per quake above ETAS (the zero line),
for Craton (the accent) and the two ETAS variants that are its hardest floors; CB-29 is in
nats, converted to bits so it shares the axis. Top right: the climb on CB-02 and CB-03, one
point per accepted craton-mini run, against ETAS and ETAS-incomplete. Bottom: the fourteen
CB-22 sequences, Craton (bar over seeds) against ETAS-incomplete (ring).

Reads results/scoreboard.json and results/runs/; trains nothing.

    uv run --group figures python scripts/figures/scorecard.py
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
from craton.bench.world import SEQUENCES, ZERO_SHOT

RUNS = ROOT / "results" / "runs"
FORECASTS = (
    ("CB-02", "Apennines 2016-17, daily"),
    ("CB-03", "L'Aquila 2009, daily"),
    ("CB-09", "Physics planet"),
    ("CB-19", "Apennines, 30 days"),
    ("CB-22 seen", "World, regions trained on"),
    ("CB-22", "World, never-seen regions"),
    ("CB-29", "EarthquakeNPP, in bits"),
)
FLOORS = (("etas_incomplete", "ETAS-incomplete"), ("etas_aniso", "ETAS, stretched zones"))
#: The phase each accepted craton-mini run shipped with; later runs fall back to their date.
PHASES = {
    "20261020-105609": "P4",
    "20261021-122053": "P5",
    "20261022-014206": "P6",
    "20261022-123525": "P7",
    "20261025-132227": "P8a",
    "20261025-195238": "P8b",
    "20261026-102634": "P8d",
}
LO, HI = -0.5, 1.3


def _results(path) -> dict[str, dict]:
    return {r["task_id"]: r for r in json.loads(path.read_text("utf-8"))["results"]}


def _latest(model: str, task: str) -> dict:
    for path in sorted(RUNS.glob(f"*-{model}.json"), reverse=True):
        body = _results(path)
        if task in body:
            return body[task]
    raise KeyError((model, task))


def gains() -> dict[str, dict[str, float]]:
    """Bits per quake above ETAS, per task and method."""
    board = json.loads((ROOT / "results" / "scoreboard.json").read_text("utf-8"))
    entries, base = board["entries"], board["baselines"]
    out: dict[str, dict[str, float]] = {}
    for task, _ in FORECASTS:
        if task == "CB-22 seen":
            out[task] = {
                m: float(np.mean(_latest(m.replace("_", "-"), "CB-22")["extras"]["seen.mean_gain"]))
                for m in ("craton-mini", "etas_incomplete", "etas_aniso")
            }
            out[task]["craton"] = out[task].pop("craton-mini")
        elif task == "CB-22":
            out[task] = {"craton": entries[task]["mean"]}
            out[task] |= {m: base[task][m]["mean"] for m, _ in FLOORS if m in base[task]}
        elif task == "CB-29":
            etas = base[task]["etas"]["mean"]
            out[task] = {"craton": (etas - entries[task]["mean"]) / math.log(2)}
        else:
            etas = base[task]["etas"]["mean"]
            out[task] = {"craton": entries[task]["mean"] - etas}
            out[task] |= {m: base[task][m]["mean"] - etas for m, _ in FLOORS if m in base[task]}
    return out


def climb() -> tuple[list[str], dict[str, list[float]], dict[str, dict[str, float]]]:
    board = json.loads((ROOT / "results" / "scoreboard.json").read_text("utf-8"))
    accepted = {h["run_id"] for h in board["history"]}
    labels: list[str] = []
    series: dict[str, list[float]] = {"CB-02": [], "CB-03": []}
    for path in sorted(RUNS.glob("*-craton-mini.json")):
        run_id = path.stem.removesuffix("-craton-mini")
        if path.stem not in accepted:
            continue
        body = _results(path)
        labels.append(PHASES.get(run_id, run_id[4:8]))
        for task in series:
            series[task].append(float(np.mean(body[task]["values"])))
    refs = {
        t: {m: board["baselines"][t][m]["mean"] for m in ("etas", "etas_incomplete")}
        for t in series
    }
    return labels, series, refs


def sequences() -> list[dict]:
    craton = _latest("craton-mini", "CB-22")["extras"]
    floor = _latest("etas-incomplete", "CB-22")["extras"]
    return [
        {
            "seq": s,
            "gain": craton[f"{s.key}.gain"],
            "floor": float(np.mean(floor[f"{s.key}.gain"])),
            "counted": min(craton[f"{s.key}.qualified"]) >= 1,
        }
        for s in SEQUENCES
    ]


def _style(ax, theme: Theme) -> None:
    ax.set_facecolor(theme.ground)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=8, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(inter(400))
        label.set_fontsize(8)


def _sign(v: float, d: int = 2) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


def draw(theme: Theme, g, labels, series, refs, seqs) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(12, 13), dpi=200, facecolor=theme.ground)
    fig.text(0.05, 0.965, "Craton against ETAS", color=theme.ink, fontproperties=bold, fontsize=20)
    fig.text(
        0.05,
        0.945,
        "Bits per quake above ETAS, mean of 3 seeds. Right of zero is better; one bit is twice "
        "the probability on the quakes that happened.",
        color=theme.muted,
        fontproperties=font,
        fontsize=9.5,
    )

    # Every forecasting task.
    ax = fig.add_axes((0.25, 0.62, 0.32, 0.29))
    _style(ax, theme)
    ys = np.arange(len(FORECASTS))[::-1]
    ax.axvline(0, color=theme.mid, lw=1.2, zorder=1)
    ax.text(
        0,
        len(FORECASTS) - 0.35,
        "ETAS",
        color=theme.muted,
        ha="center",
        fontproperties=font,
        fontsize=8,
    )
    for y, (task, name) in zip(ys, FORECASTS, strict=True):
        row = g[task]
        c = row["craton"]
        ax.plot(
            [0, np.clip(c, LO, HI)], [y, y], color=ACCENT, lw=2, alpha=0.35, solid_capstyle="round"
        )
        if "etas_aniso" in row:
            ax.scatter(row["etas_aniso"], y, s=22, color=theme.faint, zorder=2)
        if "etas_incomplete" in row:
            ax.scatter(
                row["etas_incomplete"],
                y,
                s=46,
                facecolors=theme.ground,
                edgecolors=theme.muted,
                linewidths=1.4,
                zorder=3,
            )
        ax.scatter(np.clip(c, LO, HI), y, s=80, color=ACCENT, zorder=4)
        ax.text(
            LO - 0.06,
            y,
            f"{task.split()[0]}  {name}",
            ha="right",
            va="center",
            color=theme.ink,
            fontproperties=font,
            fontsize=8.5,
        )
        ax.text(
            HI + 0.05,
            y,
            _sign(c),
            va="center",
            color=ACCENT if c > 0 else theme.muted,
            fontproperties=bold,
            fontsize=9,
        )
    ax.set_xlim(LO, HI)
    ax.set_ylim(-0.6, len(FORECASTS) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xticklabels(["−0.5", "0", "+0.5", "+1.0"])
    _style(ax, theme)
    ax.grid(axis="x", color=theme.faint, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    lx, ly = 0.05, 0.595
    for i, (txt, kw) in enumerate(
        (
            ("Craton", {"s": 60, "color": ACCENT}),
            (
                "ETAS-incomplete",
                {"s": 36, "facecolors": theme.ground, "edgecolors": theme.muted, "linewidths": 1.4},
            ),
            ("ETAS, stretched zones", {"s": 18, "color": theme.faint}),
        )
    ):
        fig.text(
            lx + 0.02 + 0.15 * i,
            ly,
            txt,
            color=theme.muted,
            fontproperties=font,
            fontsize=8.5,
            va="center",
        )
        dot = fig.add_axes((lx + 0.15 * i, ly - 0.006, 0.012, 0.012))
        dot.axis("off")
        dot.scatter(0, 0, **kw)
    fig.text(
        0.05, 0.925, "Every forecasting test", color=theme.ink, fontproperties=bold, fontsize=12
    )

    # The climb.
    fig.text(
        0.67, 0.925, "The climb, phase by phase", color=theme.ink, fontproperties=bold, fontsize=12
    )
    for i, task in enumerate(("CB-02", "CB-03")):
        ax = fig.add_axes((0.67, 0.785 - 0.165 * i, 0.26, 0.12))
        _style(ax, theme)
        x = np.arange(len(labels))
        v = series[task]
        ax.axhline(refs[task]["etas"], color=theme.mid, lw=1, ls=(0, (4, 3)))
        ax.axhline(refs[task]["etas_incomplete"], color=theme.muted, lw=1, ls=(0, (1, 2)))
        ax.plot(x, v, color=ACCENT, lw=2, solid_capstyle="round", solid_joinstyle="round")
        above = [y > refs[task]["etas"] for y in v]
        ax.scatter(
            x,
            v,
            s=28,
            color=[ACCENT if a else theme.ground for a in above],
            edgecolors=ACCENT,
            linewidths=1.5,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.tick_right()
        _style(ax, theme)
        ax.grid(axis="y", color=theme.faint, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        name = "Apennines 2016-17" if task == "CB-02" else "L'Aquila 2009"
        ax.set_title(
            f"{task}  {name}", loc="left", color=theme.muted, fontproperties=font, fontsize=8.5
        )
        ax.annotate(
            f"{v[-1]:.2f}".replace("-", "−"),
            (len(labels) - 1, v[-1]),
            xytext=(-8, 6),
            textcoords="offset points",
            color=ACCENT,
            ha="right",
            va="bottom",
            fontproperties=bold,
            fontsize=8.5,
        )
    fig.text(
        0.67,
        0.595,
        "Bits per M3+ quake. Dashed: ETAS. Dotted: ETAS-incomplete.\nHollow points are below ETAS.",
        color=theme.muted,
        fontproperties=font,
        fontsize=8,
        va="center",
    )

    # The fourteen sequences.
    fig.text(
        0.05,
        0.545,
        "Fourteen sequences around the world (CB-22)",
        color=theme.ink,
        fontproperties=bold,
        fontsize=12,
    )
    fig.text(
        0.05,
        0.527,
        "Each the year after a big quake, against ETAS fitted on that box's own past. "
        "Bar: spread over seeds. Ring: ETAS-incomplete. Maule is not counted (D-091).",
        color=theme.muted,
        fontproperties=font,
        fontsize=8.5,
    )
    ax = fig.add_axes((0.3, 0.05, 0.56, 0.46))
    _style(ax, theme)
    order = [s for s in seqs if s["seq"].half != ZERO_SHOT] + [
        s for s in seqs if s["seq"].half == ZERO_SHOT
    ]
    n_seen = sum(s["seq"].half != ZERO_SHOT for s in seqs)
    lo, hi = -0.4, 1.0
    ypos = []
    y = len(order) + 1
    for i, _ in enumerate(order):
        if i == 0 or i == n_seen:
            y -= 1
        ypos.append(y)
        y -= 1
    ax.axvline(0, color=theme.mid, lw=1.2, zorder=1)
    for yy, s in zip(ypos, order, strict=True):
        seq, gs = s["seq"], s["gain"]
        mean = float(np.mean(gs))
        ink = theme.ink if s["counted"] else theme.muted
        ax.text(
            lo - 0.04,
            yy,
            f"{seq.name}  M{seq.largest:g}",
            ha="right",
            va="center",
            color=ink,
            fontproperties=font,
            fontsize=8.5,
        )
        if s["floor"] <= hi:
            ax.scatter(
                s["floor"],
                yy,
                s=40,
                facecolors=theme.ground,
                edgecolors=theme.muted,
                linewidths=1.4,
                zorder=2,
            )
        else:
            ax.text(
                hi,
                yy,
                f"ring {_sign(s['floor'], 1)}, off scale",
                ha="right",
                va="center",
                color=theme.muted,
                fontproperties=font,
                fontsize=7.5,
            )
        if mean < lo:
            ax.text(
                lo + 0.01,
                yy,
                f"{_sign(mean, 1)}, off scale, not counted",
                va="center",
                color=theme.muted,
                fontproperties=font,
                fontsize=8,
            )
        else:
            ax.plot(
                [max(lo, min(gs)), max(gs)],
                [yy, yy],
                color=ACCENT,
                lw=3,
                alpha=0.35,
                solid_capstyle="round",
            )
            ax.scatter(mean, yy, s=60, color=ACCENT, zorder=3)
        ax.text(
            hi + 0.04,
            yy,
            _sign(mean),
            va="center",
            color=ACCENT if s["counted"] and mean > 0 else theme.muted,
            fontproperties=bold,
            fontsize=8.5,
        )
    for yy, head in (
        (ypos[0] + 1, "REGIONS IT TRAINED ON"),
        (ypos[n_seen] + 1, "NEVER SEEN IN TRAINING"),
    ):
        ax.text(
            lo - 0.04,
            yy,
            head,
            ha="right",
            va="center",
            color=theme.muted,
            fontproperties=bold,
            fontsize=7.5,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(ypos) - 0.7, max(ypos) + 1.5)
    ax.set_yticks([])
    ax.set_xticks([-0.4, 0, 0.4, 0.8])
    ax.set_xticklabels(["−0.4", "0", "+0.4", "+0.8"])
    _style(ax, theme)
    ax.grid(axis="x", color=theme.faint, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    out = ASSETS / f"scorecard-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    g, (labels, series, refs), seqs = gains(), climb(), sequences()
    for theme in THEMES:
        draw(theme, g, labels, series, refs, seqs)


if __name__ == "__main__":
    main()
