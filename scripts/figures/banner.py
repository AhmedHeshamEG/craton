"""The README hero: every M5+ earthquake on Earth since 1976, and nothing else.

No coastlines and no plate boundaries are drawn. The outlines you see are the plates, drawn
by the earthquakes themselves. Shallow events (under 70 km, the brittle crust) are the accent;
deeper ones fade to grey, so subducting slabs show up as grey halos beside the trenches.

Data: USGS ComCat via the FDSN event service (public domain). Cached under data/cache/.

    uv run --group figures python scripts/figures/banner.py
"""

from __future__ import annotations

import csv
import io
import math
import urllib.parse
from dataclasses import dataclass
from datetime import date

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, CACHE, THEMES, Theme, download, inter

START, END = 1976, 2026
END_DATE = "2026-10-19"
MIN_MAG = 5.0
CENTRAL_MERIDIAN = 150.0  # Pacific-centred: the Ring of Fire in the middle
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@dataclass(frozen=True)
class Quakes:
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    mag: np.ndarray


def fetch() -> Quakes:
    """Download (once) and load the catalogue in five-year windows (the API caps at 20k)."""
    folder = CACHE / "usgs"
    folder.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[float, float, float, float]] = []
    for y in range(START, END + 1, 5):
        stop = f"{y + 5}-01-01" if y + 5 <= END else END_DATE
        path = folder / f"m{MIN_MAG:g}_{y}.csv"
        if not path.exists():
            q = urllib.parse.urlencode(
                {
                    "format": "csv",
                    "starttime": f"{y}-01-01",
                    "endtime": stop,
                    "minmagnitude": MIN_MAG,
                    "orderby": "time-asc",
                }
            )
            path.write_bytes(download(f"{USGS}?{q}"))
        for rec in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))):
            try:
                rows.append(
                    (
                        float(rec["longitude"]),
                        float(rec["latitude"]),
                        float(rec["depth"] or 0),
                        float(rec["mag"]),
                    )
                )
            except ValueError:
                continue
    a = np.array(rows)
    return Quakes(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def equal_earth(lon: np.ndarray, lat: np.ndarray, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal Earth projection (Šavrič, Patterson & Jenny 2018), equal-area."""
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    lam = np.radians(((lon - lon0 + 180) % 360) - 180)
    phi = np.radians(lat)
    theta = np.arcsin(math.sqrt(3) / 2 * np.sin(phi))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = (2 * math.sqrt(3) * lam * np.cos(theta)) / (
        3 * (9 * a4 * t6 * t2 + 7 * a3 * t6 + 3 * a2 * t2 + a1)
    )
    y = theta * (a1 + a2 * t2 + t6 * (a3 + a4 * t2))
    return x, y


def draw(q: Quakes, theme: Theme, width: int, height: int, out: str, caption: bool = True) -> None:
    x, y = equal_earth(q.lon, q.lat, CENTRAL_MERIDIAN)
    dpi = 200
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_facecolor(theme.ground)
    ax.set_axis_off()

    # Frame the map by width, crop the empty poles.
    half_w = 2.7 * 1.0
    aspect = height / width
    ax.set_xlim(-half_w, half_w)
    y_mid = 0.08
    ax.set_ylim(y_mid - half_w * aspect, y_mid + half_w * aspect)

    size = 0.6 * np.exp((q.mag - 5.0) * 1.15) * (width / 2400) ** 2
    order = np.argsort(-q.depth)  # deep first, shallow drawn on top
    shallow = q.depth[order] < 70
    mid = (q.depth[order] >= 70) & (q.depth[order] < 300)
    deep = q.depth[order] >= 300
    xs, ys, ss = x[order], y[order], size[order]

    ax.scatter(xs[deep], ys[deep], s=ss[deep], c=theme.faint, alpha=0.5, linewidths=0)
    ax.scatter(xs[mid], ys[mid], s=ss[mid], c=theme.mid, alpha=0.28, linewidths=0)
    # Shallow quakes twice: a wide faint glow, then a small core, so density reads as heat.
    ax.scatter(xs[shallow], ys[shallow], s=ss[shallow] * 6, c=ACCENT, alpha=0.035, linewidths=0)
    ax.scatter(xs[shallow], ys[shallow], s=ss[shallow], c=ACCENT, alpha=0.22, linewidths=0)

    # A soft scrim in the lower left so the words sit on calm ground.
    gy, gx = np.mgrid[0:1:200j, 0:1:400j]
    r = np.hypot(gx / 0.56, gy / 0.5)
    edge = np.clip((1.0 - r) / 0.45, 0, 1)
    fade = edge * edge * (3 - 2 * edge)  # smoothstep: a solid core, a soft edge
    rgba = np.zeros((200, 400, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(theme.ground)
    rgba[..., 3] = 0.97 * fade
    scrim = fig.add_axes((0, 0, 1, 1))
    scrim.set_axis_off()
    scrim.imshow(rgba, extent=(0, 1, 0, 1), origin="lower", aspect="auto", zorder=1)
    scrim.set_xlim(0, 1)
    scrim.set_ylim(0, 1)
    scrim.patch.set_alpha(0)

    k = width / 2400
    left, bottom = 0.045, 0.075
    fig.text(
        left,
        bottom + 0.15,
        "craton",
        fontproperties=inter(700),
        fontsize=66 * k,
        color=theme.ink,
        ha="left",
        va="baseline",
        zorder=5,
    )
    fig.text(
        left + 0.003,
        bottom + 0.09,
        "One model for the solid Earth.",
        fontproperties=inter(400),
        fontsize=21 * k,
        color=theme.ink,
        ha="left",
        va="baseline",
        zorder=5,
    )
    if caption:
        fig.text(
            left + 0.003,
            bottom + 0.03,
            f"Every M{MIN_MAG:g}+ earthquake since {START}, {len(q.mag):,} of them. "
            "No coastlines drawn.",
            fontproperties=inter(400),
            fontsize=11 * k,
            color=theme.muted,
            ha="left",
            va="baseline",
            zorder=5,
        )
        fig.text(
            left + 0.003,
            bottom,
            f"The plates draw themselves · USGS ComCat, to {date.fromisoformat(END_DATE):%B %Y}.",
            fontproperties=inter(400),
            fontsize=11 * k,
            color=theme.muted,
            ha="left",
            va="baseline",
            zorder=5,
        )
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)
    print("wrote", ASSETS / out)


def main() -> None:
    q = fetch()
    print(f"{len(q.mag):,} events, M{q.mag.min():.1f}–{q.mag.max():.1f}")
    for theme in THEMES:
        draw(q, theme, 2400, 1000, f"hero-{theme.name}.png")
    draw(q, THEMES[0], 1280, 640, "social-preview.png")


if __name__ == "__main__":
    main()
