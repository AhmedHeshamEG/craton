"""One token, exploded into its slots: a monk's account from 1349 next to a GPS day from 2016.

Every value on the figure is read from the example tokens in ``craton.spec.examples``, which
are built from the raw INGV and Nevada Geodetic Laboratory records.

uv run --group figures python scripts/figures/token_slots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import ACCENT, ASSETS, THEMES, Theme, inter

from craton.spec import SPEC_VERSION, Token
from craton.spec.examples import example

plt.rcParams["svg.fonttype"] = "path"

W, H = 16.0, 9.3


def _deg(lat: float, lon: float) -> str:
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"{abs(lat):.3f}° {ns}   {abs(lon):.3f}° {ew}"


def _span(seconds: float) -> str:
    if seconds >= 86_400:
        return f"{seconds / 86_400:g} day" + ("s" if seconds != 86_400 else "")
    return f"{seconds:g} s"


def _size(km: float) -> str:
    return f"{km * 1000:g} m" if km < 1 else f"{km:g} km"


def rows(monk: Token, gps: Token) -> list[tuple[str, str, str]]:
    """(slot, monk's value, GPS value), all read from the tokens."""
    q, g = monk.qualifiers, gps.qualifiers
    return [
        ("kind", monk.kind, gps.kind),
        ("instrument", monk.instrument, gps.instrument),
        ("lat · lon", _deg(monk.lat, monk.lon), _deg(gps.lat, gps.lon)),
        (
            "depth",
            f"{monk.depth_km:+.2f} km  ·  on a hill",
            f"{gps.depth_km:+.2f} km  ·  on a mountain",
        ),
        (
            "time",
            f"{(monk.time_iso or '')[:10]}  ·  9 Sep in his calendar",
            f"{(gps.time_iso or '')[:16].replace('T', ' ')} UTC",
        ),
        (
            "value",
            f"intensity {monk.value[0]:g} {q['scale']}",
            "  ".join(f"{c[0].upper()} {v:,.3f}" for c, v in zip("enu", gps.value, strict=True)),
        ),
        (
            "uncertainty",
            f"± {monk.uncertainty[0]:g} degree  ·  {monk.uncertainty_basis}",
            "± "
            + " · ".join(f"{s * 1000:.1f}" for s in gps.uncertainty)
            + f" mm  ·  {gps.uncertainty_basis}",
        ),
        ("qualifiers", q["place"], f"station {g['station']}  ·  {g['reference_frame']}"),
        (
            "provenance",
            f"INGV ASMI  ·  study {q['study']}",
            "Nevada Geodetic Laboratory",
        ),
        (
            "resolution",
            f"{_size(monk.resolution.spatial_km)}  ·  {_span(monk.resolution.temporal_s)}",
            f"{_size(gps.resolution.spatial_km)}  ·  {_span(gps.resolution.temporal_s)}",
        ),
    ]


def draw(theme: Theme, preview: Path | None = None) -> None:
    monk, gps = example("montecassino_1349"), example("arquata_gnss")
    fig = plt.figure(figsize=(W, H), facecolor=theme.ground)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    bold, body = inter(600), inter(400)
    x0, xm, xg = 0.8, 4.25, 10.05
    top = H - 0.75

    ax.text(x0, top, "One token", fontproperties=bold, fontsize=26, color=theme.ink, va="top")
    ax.text(
        x0,
        top - 0.72,
        f"Token Spec {SPEC_VERSION}  ·  the same slots for a chronicle and a satellite receiver",
        fontproperties=body,
        fontsize=13,
        color=theme.muted,
        va="top",
    )

    head_y = top - 1.6
    for x, year, what in (
        (xm, "1349", "Montecassino Abbey, written down by the monks"),
        (xg, "2016", "GNSS station ARQT, the day after the Mw 6.5"),
    ):
        ax.text(x, head_y, year, fontproperties=bold, fontsize=15, color=theme.ink, va="baseline")
        ax.text(
            x + 0.75,
            head_y,
            what,
            fontproperties=body,
            fontsize=12,
            color=theme.muted,
            va="baseline",
        )

    row_h, gap = 0.44, 0.13
    y = head_y - 0.45
    for slot, left, right in rows(monk, gps):
        accent = slot == "uncertainty"
        edge = ACCENT if accent else theme.faint
        ax.add_patch(
            FancyBboxPatch(
                (x0, y - row_h),
                2.95,
                row_h,
                boxstyle="round,pad=0,rounding_size=0.09",
                lw=1.1 if accent else 0.9,
                ec=edge,
                fc="none",
            )
        )
        ax.text(
            x0 + 0.18,
            y - row_h / 2,
            slot,
            fontproperties=bold if accent else body,
            fontsize=12,
            color=ACCENT if accent else theme.ink,
            va="center",
        )
        # Leader lines: the slot is exploded out to each value.
        ax.plot([x0 + 2.95, xm - 0.15], [y - row_h / 2] * 2, color=edge, lw=0.7)
        for x, text in ((xm, left), (xg, right)):
            ax.text(
                x,
                y - row_h / 2,
                text.replace("-", "−") if slot in ("depth", "value") else text,
                fontproperties=body,
                fontsize=12,
                color=ACCENT if accent else theme.muted,
                va="center",
            )
        y -= row_h + gap

    ax.plot([x0, W - x0], [y + 0.05, y + 0.05], color=theme.faint, lw=0.8)
    ax.text(
        x0,
        y - 0.4,
        "667 years apart, one shape. The uncertainty slot is what lets a model read both: "
        "it can trust the receiver to a millimetre and the monk to half a degree.",
        fontproperties=body,
        fontsize=12,
        color=theme.muted,
        va="center",
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"token-{theme.name}.svg"
    fig.savefig(out, facecolor=theme.ground, metadata={"Date": None})
    if preview is not None:
        fig.savefig(preview / out.with_suffix(".png").name, facecolor=theme.ground, dpi=100)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    where = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--preview" else None
    for theme in THEMES:
        draw(theme, where)
