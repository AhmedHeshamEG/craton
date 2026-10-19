"""The data coverage report: what Craton-mini can see, where and when.

Left, the map: every earthquake INGV located in the box since 1985, and every GNSS station NGL
processes there. No coastlines are drawn; the Apennines draw themselves. Right, three timelines
on one clock: earthquakes per month, GNSS stations reporting, and the catalogue's magnitude of
completeness (the smallest ML it reliably catches) per year.

Reads the latest snapshot of the region built by `craton data build`, nothing else.

    uv run --group figures python scripts/figures/coverage.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
from craton.spec.timeaxis import to_seconds
from craton.store.parquet import read_columns

REGION = "central_italy"
LABELS = [  # (event id, short name): the sequences a reader may know, placed at their main shock
    ("8863681", "Norcia 2016"),
    ("7073641", "Amatrice 2016"),
    ("1895389", "L'Aquila 2009"),
    ("8669321", "Visso 2016"),
    ("849549", "Colfiorito 1997"),
]


def load() -> dict:
    data = ROOT / "data"
    folder = data / "snapshots" / REGION
    snap_id = (folder / "LATEST").read_text(encoding="utf-8").strip()
    snap = json.loads((folder / f"{snap_id}.json").read_text(encoding="utf-8"))
    tokens = data / "tokens"

    def cols(adapter: str, names: tuple[str, ...]) -> dict[str, list]:
        out: dict[str, list] = {n: [] for n in names}
        for f in snap["adapters"][adapter]["files"]:
            c = read_columns(tokens / f["path"], names)
            for n in names:
                out[n] += c[n]
        return out

    q = cols("ingv_catalog", ("lat", "lon", "time", "value", "qualifiers"))
    g = cols("ngl_gnss", ("lat", "lon", "time", "qualifiers"))
    return {
        "snap": snap_id,
        "box": snap["box"],
        "window": snap["window"],
        "q": q,
        "g": g,
        "health": snap["adapters"]["ingv_catalog"]["health"]["seismic_catalog"],
    }


def span(window: list[str]) -> str:
    """'January 1985 to September 2026' from a half-open [start, end) window."""
    first = dt.datetime.fromisoformat(window[0].replace("Z", "+00:00"))
    last = dt.datetime.fromisoformat(window[1].replace("Z", "+00:00")) - dt.timedelta(seconds=1)
    return f"{first:%B %Y} to {last:%B %Y}"


def years(t: np.ndarray) -> np.ndarray:
    return 1970 + t / (365.2425 * 86400)


def draw(d: dict, theme: Theme, out: str) -> None:
    q, g, box = d["q"], d["g"], d["box"]
    qlat, qlon = np.array(q["lat"]), np.array(q["lon"])
    qt = np.array(q["time"])
    mag = np.array([v[0] for v in q["value"]])
    quals = [json.loads(x) for x in q["qualifiers"]]
    ids = [x.get("event_id") for x in quals]

    width_px, height_px, dpi = 2400, 1260, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)

    # ---------------------------------------------------------------- the map
    lat0 = (box["lat_min"] + box["lat_max"]) / 2
    sx = np.cos(np.radians(lat0))
    map_h = 0.80
    map_w = (
        map_h
        * (height_px / width_px)
        * (box["lon_max"] - box["lon_min"])
        * sx
        / (box["lat_max"] - box["lat_min"])
    )
    ax = fig.add_axes((0.035, 0.11, map_w, map_h))
    ax.set_facecolor(theme.ground)
    ax.set_axis_off()
    ax.set_xlim(box["lon_min"] * sx, box["lon_max"] * sx)
    ax.set_ylim(box["lat_min"], box["lat_max"])
    order = np.argsort(mag)
    x, y, m = qlon[order] * sx, qlat[order], mag[order]
    size = 0.5 * np.exp((np.clip(m, -1, 7) - 1.0) * 0.95) * k**2
    ax.scatter(x, y, s=size * 5, c=ACCENT, alpha=0.012, linewidths=0, rasterized=True)
    ax.scatter(x, y, s=size, c=ACCENT, alpha=0.10, linewidths=0, rasterized=True)

    stations: dict[str, tuple[float, float]] = {}
    for lat, lon, qq in zip(g["lat"], g["lon"], g["qualifiers"], strict=True):
        stations.setdefault(json.loads(qq)["station"], (lat, lon))
    slat = np.array([v[0] for v in stations.values()])
    slon = np.array([v[1] for v in stations.values()])
    ax.scatter(
        slon * sx,
        slat,
        s=9 * k**2,
        facecolors="none",
        edgecolors=theme.ink,
        linewidths=0.6 * k,
        alpha=0.85,
    )

    # Labels sit in a column east of the chain, joined to their main shock by a hairline.
    placed = []
    for eid, name in LABELS:
        if eid in ids:
            i = ids.index(eid)
            placed.append((qlat[i], qlon[i], name))
    placed.sort(reverse=True)
    label_x = (box["lon_min"] + 0.80 * (box["lon_max"] - box["lon_min"])) * sx
    last_y = None
    for lat, lon, name in placed:
        ty = lat if last_y is None else min(lat, last_y - 0.16)
        last_y = ty
        ax.plot(
            [lon * sx, label_x - 0.04], [lat, ty], color=theme.muted, linewidth=0.5 * k, alpha=0.8
        )
        ax.scatter(
            [lon * sx],
            [lat],
            s=14 * k**2,
            facecolors="none",
            edgecolors=theme.ink,
            linewidths=0.8 * k,
        )
        ax.text(
            label_x,
            ty,
            name,
            color=theme.ink,
            fontsize=8.5 * k,
            fontproperties=inter(400),
            va="center",
            ha="left",
        )

    # ---------------------------------------------------------------- the timelines
    left = 0.035 + map_w + 0.06
    width = 0.96 - left
    t0 = float(d["window"][0][:4])
    t1 = years(np.array([to_seconds(d["window"][1])]))[0]
    rows = [(0.66, 0.22), (0.39, 0.18), (0.12, 0.18)]

    def strip(i: int, title: str, note: str) -> plt.Axes:
        bottom, height = rows[i]
        a = fig.add_axes((left, bottom, width, height))
        a.set_facecolor(theme.ground)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(theme.faint)
        a.tick_params(colors=theme.muted, labelsize=8 * k, length=0)
        a.tick_params(which="minor", length=0)
        a.set_xlim(t0, t1)
        a.set_xticks([1990, 2000, 2010, 2020])
        if i < 2:
            a.set_xticklabels([])
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_fontproperties(inter(400))
            lab.set_fontsize(8 * k)
        fig.text(
            left,
            bottom + height + 0.012,
            title,
            color=theme.ink,
            fontsize=10.5 * k,
            fontproperties=inter(600),
            ha="left",
            va="bottom",
        )
        fig.text(
            left + width,
            bottom + height + 0.012,
            note,
            color=theme.muted,
            fontsize=8.5 * k,
            fontproperties=inter(400),
            ha="right",
            va="bottom",
        )
        return a

    # earthquakes per month
    month = np.floor((years(qt) - t0) * 12).astype(int)
    counts = np.bincount(month, minlength=int((t1 - t0) * 12) + 1)
    tm = t0 + (np.arange(len(counts)) + 0.5) / 12
    a = strip(0, "Earthquakes per month", f"{len(qt):,} located by INGV")
    a.fill_between(tm, 1, np.maximum(counts, 1), step="mid", color=ACCENT, alpha=0.85, linewidth=0)
    a.set_yscale("log")
    a.set_ylim(1, counts.max() * 1.5)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a.set_yticks([10, 1000, 10000])

    # stations reporting
    gt = years(np.array(g["time"]))
    per_station: dict[str, set[int]] = {}
    for tt, qq in zip(gt, g["qualifiers"], strict=True):
        per_station.setdefault(qq, set()).add(int(tt * 12))
    active: Counter[int] = Counter()
    for months in per_station.values():
        active.update(months)
    mm = np.array(sorted(active))
    a = strip(1, "GNSS stations reporting", f"{len(stations)} stations, daily positions")
    a.fill_between(
        mm / 12, 0, [active[v] for v in mm], step="mid", color=theme.mid, alpha=0.9, linewidth=0
    )
    a.set_ylim(0, max(active.values()) * 1.15)

    # completeness
    a = strip(2, "Smallest quake the catalogue catches", "magnitude of completeness per year")
    for mt, colour in (("Md", theme.mid), ("ML", theme.ink)):
        per_year = d["health"]["mc_per_year"].get(mt, {})
        yrs = np.array([int(y) for y in per_year]) + 0.5
        vals = [per_year[str(int(y - 0.5))] for y in yrs]
        a.plot(yrs, vals, color=colour, linewidth=1.4 * k, drawstyle="steps-mid")
        if len(yrs):
            below = mt == "ML"  # ML starts under the Md line; its tag goes underneath
            dy = -0.55 if below else 0.25
            a.text(
                yrs[0] - 0.3,
                vals[0] + dy,
                mt,
                color=colour,
                fontsize=8 * k,
                fontproperties=inter(600),
                ha="left",
            )
    a.set_ylim(0, 3.2)
    a.set_yticks([0, 1, 2, 3])

    fig.text(
        0.035,
        0.04,
        f"Central Italy, {span(d['window'])} · every dot and "
        f"every station-day is a Craton token · snapshot {d['snap']}",
        color=theme.muted,
        fontsize=8.5 * k,
        fontproperties=inter(400),
        ha="left",
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)
    print("wrote", ASSETS / out)


def main() -> None:
    d = load()
    for theme in THEMES:
        draw(d, theme, f"coverage-{theme.name}.png")


if __name__ == "__main__":
    main()
