"""The world Craton learned from, and the sequences it was tested on (CB-22, P8d).

Every quake a magnitude above its region's completeness, as a dot: grey where the world Field
learned (up to 2007), faint where it never looked (the zero-shot regions). On top, each held-out
sequence at its first big shock, with Craton's gain over ETAS fitted there, in bits per quake,
averaged over seeds: white rings for regions it trained on, orange for regions it never saw.
No coastlines: the plate edges draw themselves.

Reads the region snapshots and the latest craton-mini run in `results/runs/`.

    uv run --group figures python scripts/figures/world.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.adapters.region import load_region
from craton.bench.tasks import BenchContext
from craton.bench.world import SEQUENCES, ZERO_SHOT

REGIONS = ("central_italy", "california", "japan", "new_zealand", "turkey", "chile", "alaska")
TRAINED = {"central_italy", "california", "japan", "new_zealand"}
CUT = 1167609600.0  # 2007-01-01
MC = {"central_italy": 2.0}


def quakes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name in REGIONS:
        region = load_region(name, ROOT)
        mc = MC.get(name, region.completeness or 3.0)
        ev = BenchContext(ROOT, region=name).events.above(mc + 1.0)
        if name in TRAINED:
            ev = ev.select(ev.time < CUT)
        out[name] = (ev.lon, ev.lat)
    return out


def gains() -> dict[str, float]:
    runs = sorted((ROOT / "results" / "runs").glob("*-craton-mini.json"))
    for path in reversed(runs):
        body = json.loads(path.read_text("utf-8"))
        for r in body["results"]:
            if r["task_id"] == "CB-22":
                return {s.key: float(np.mean(r["extras"][f"{s.key}.gain"])) for s in SEQUENCES}
    return {}


TOP = ("central_italy", "california", "japan", "new_zealand")
BOTTOM = ("turkey", "chile", "alaska")
TITLES = {
    "central_italy": "Central Italy",
    "california": "California",
    "japan": "Japan",
    "new_zealand": "New Zealand",
    "turkey": "Turkey",
    "chile": "Chile",
    "alaska": "Alaska",
}


def _panel(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    name: str,
    theme: Theme,
    q: dict[str, tuple[np.ndarray, np.ndarray]],
    g: dict[str, float],
) -> None:
    font, bold = inter(400), inter(600)
    box = load_region(name, ROOT).box
    ax = fig.add_axes(rect, facecolor=theme.ground)
    lon, lat = q[name]
    seen = name in TRAINED
    ax.scatter(
        lon,
        lat,
        s=0.5,
        c=theme.mid if seen else ACCENT,
        linewidths=0,
        alpha=0.7 if seen else 0.35,
        rasterized=True,
    )
    accent = theme.ink if seen else ACCENT
    lines = []
    for s in (s for s in SEQUENCES if s.region == name):
        ax.scatter([s.lon], [s.lat], s=70, facecolors="none", edgecolors=accent, linewidths=1.3)
        gain = g.get(s.key)
        lines.append(s.name if gain is None else f"{s.name}   {gain:+.2f}")
    ax.set_xlim(box.lon_min, box.lon_max)
    ax.set_ylim(box.lat_min, box.lat_max)
    ax.set_aspect(
        1 / np.cos(np.radians((box.lat_min + box.lat_max) / 2)), adjustable="box", anchor="S"
    )
    ax.axis("off")
    x, y, _, h = rect
    fig.text(x, y + h + 0.012, TITLES[name], color=theme.ink, fontproperties=bold, fontsize=11)
    for i, line in enumerate(lines):
        fig.text(x, y - 0.03 - 0.026 * i, line, color=accent, fontproperties=font, fontsize=8.5)


def draw(theme: Theme, q: dict[str, tuple[np.ndarray, np.ndarray]], g: dict[str, float]) -> None:
    font, bold = inter(400), inter(600)
    fig = plt.figure(figsize=(14, 10), dpi=200, facecolor=theme.ground)
    for i, name in enumerate(TOP):
        _panel(fig, (0.03 + 0.245 * i, 0.55, 0.21, 0.28), name, theme, q, g)
    for i, name in enumerate(BOTTOM):
        _panel(fig, (0.03 + 0.245 * i, 0.12, 0.21, 0.28), name, theme, q, g)
    zero = [g[s.key] for s in SEQUENCES if s.half == ZERO_SHOT and s.key in g]
    if zero:
        by_region: dict[str, list[float]] = {}
        for s in SEQUENCES:
            if s.half == ZERO_SHOT and s.key in g:
                by_region.setdefault(s.region, []).append(g[s.key])
        headline = float(np.mean([np.mean(v) for v in by_region.values()]))
        fig.text(0.765, 0.30, f"{headline:+.2f}", color=ACCENT, fontproperties=bold, fontsize=44)
        fig.text(
            0.765,
            0.255,
            "bits per quake over ETAS,",
            color=theme.ink,
            fontproperties=font,
            fontsize=11,
        )
        fig.text(
            0.765,
            0.232,
            "never-seen regions (CB-22)",
            color=theme.ink,
            fontproperties=font,
            fontsize=11,
        )
        beaten = sum(v > 0 for v in zero)
        fig.text(
            0.765,
            0.200,
            f"{beaten} of {len(zero)} sequences above ETAS",
            color=theme.muted,
            fontproperties=font,
            fontsize=10,
        )
    fig.text(
        0.03, 0.95, "One model, seven regions", color=theme.ink, fontproperties=bold, fontsize=20
    )
    fig.text(
        0.03,
        0.918,
        "Top: regions the world Field learned from, up to 2007 (grey dots).  Bottom: regions it "
        "never saw (orange).",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    fig.text(
        0.03,
        0.895,
        "Rings: held-out sequences. The number is Craton's gain over ETAS fitted on that box's "
        "own past, in bits per quake.",
        color=theme.muted,
        fontproperties=font,
        fontsize=10,
    )
    out = ASSETS / f"world-{theme.name}.png"
    fig.savefig(out, facecolor=theme.ground)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    q, g = quakes(), gains()
    for theme in THEMES:
        draw(theme, q, g)


if __name__ == "__main__":
    main()
