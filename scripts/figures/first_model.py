"""Craton-mini v0, in one picture: how it learned, and where it looks.

Left: the masked loss in bits per hidden quake, measured on the validation blocks during
training, one line per seed, against climatology scored on the very same hidden quakes.

Right: L'Aquila, 6 April 2009, six hours after the Mw 6.3 main shock. The model never saw
2009 (it is CB-01's held-out era). One aftershock is hidden (the ring); every other quake in
its window is drawn where it happened, brighter where one attention head looked while
guessing where the hidden one was. The head shown is the one, out of all 48, that puts the
most attention on quakes within 10 km of the hidden one, relative to how many there are; the
caption gives both numbers, so the picture can't oversell it.

Needs the three CB-01 checkpoints (`craton train pretrain`, or `craton bench run craton-mini`).

    uv run --extra cu126 --group figures python scripts/figures/first_model.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style import ACCENT, ASSETS, ROOT, THEMES, Theme, inter

sys.path.insert(0, str(ROOT))
import torch

from craton.bench.tasks import BenchContext
from craton.model.heads import EVENT_BINS
from craton.spec.timeaxis import to_seconds
from craton.train.data import Catalog, MaskedWindows, window_tokens
from craton.train.loop import VAL_BATCHES
from craton.train.method import CratonMini

MAIN_SHOCK = "2009-04-06T01:32:39Z"
HOURS_AFTER = 6.0  # the hidden quake is the first one this long after the main shock
NEAR_KM = 10.0


def climatology_bits(ctx: BenchContext, mini: CratonMini) -> float:
    """Climatology's bits per hidden quake on the validation windows the curves are scored on:
    one histogram per channel over the pretrain split, plus one count per bin."""
    cat = Catalog.of(ctx.train_events("pretrain"))
    cfg = mini.config
    val = MaskedWindows(cat, cfg.model.context_tokens, cfg.train.mask_share).validation(
        cfg.train.micro_batch, VAL_BATCHES
    )
    targets = np.concatenate([b.targets.numpy() for b in val]).astype(np.float64)
    bits = np.zeros(len(targets))
    for c, b in enumerate(EVENT_BINS):
        counts = np.bincount(b.index(torch.tensor(cat.values[:, c])).numpy(), minlength=b.n)
        p = (counts + 1.0) / (counts.sum() + b.n)
        bits += -np.log2(p[b.index(torch.tensor(targets[:, c])).numpy()])
    return float(bits.mean())


def load() -> dict:
    ctx = BenchContext(ROOT)
    mini = CratonMini()
    curves = []
    for seed in mini.config.train.seeds:
        mini.model(ctx, "pretrain", seed)
        meta = mini._meta[("pretrain", seed)]
        curves.append(
            (
                [h["step"] for h in meta["history"]],
                [h["val_masked"] for h in meta["history"]],
                meta["stats"]["best_step"],
            )
        )
    clim = climatology_bits(ctx, mini)

    seed = mini.config.train.seeds[0]
    model = mini.model(ctx, "pretrain", seed)
    cat = Catalog.of(ctx.events)
    t_hidden = to_seconds(MAIN_SHOCK) + HOURS_AFTER * 3600
    target = int(np.searchsorted(cat.time, t_hidden))
    length = mini.config.model.context_tokens
    mid = length // 2
    rows = (target - mid + np.arange(length))[None, :]
    hidden = np.zeros((1, length), dtype=bool)
    hidden[0, mid] = True
    tok = window_tokens(cat, rows, hidden, cat.time[rows[:, 0]]).to(mini.device)
    vals = cat.values[rows[0]]
    km = np.hypot(
        (vals[:, 0] - vals[mid, 0]) * 111.2,
        (vals[:, 1] - vals[mid, 1]) * 111.2 * np.cos(np.radians(vals[mid, 0])),
    )
    near = km < NEAR_KM
    near[mid] = False
    with torch.no_grad():
        att = np.stack(
            [
                model.attention(tok, layer)[0, :, mid].float().cpu().numpy()
                for layer in range(model.config.model.n_layers)
            ]
        )  # (layers, heads, T)
    share = att[..., near].sum(axis=-1)  # attention on nearby quakes, per layer and head
    layer, head = np.unravel_index(int(np.argmax(share)), share.shape)
    return {
        "curves": curves,
        "climatology": clim,
        "lat": vals[:, 0],
        "lon": vals[:, 1],
        "att": att[layer, head],
        "hidden": mid,
        "head": int(head),
        "layer": int(layer),
        "near_attention": float(share[layer, head]),
        "near_share": float(near.mean()),
        "median_near_attention": float(np.median(share)),
        "span_h": (cat.time[rows[0, -1]] - cat.time[rows[0, 0]]) / 3600,
    }


def _axes(ax: plt.Axes, theme: Theme, k: float) -> None:
    ax.set_facecolor(theme.ground)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(theme.faint)
    ax.tick_params(colors=theme.muted, labelsize=8 * k, length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(inter(400))
        lab.set_fontsize(8 * k)


def draw(d: dict, theme: Theme, out: str) -> None:
    width_px, height_px, dpi = 2400, 1200, 200
    k = width_px / 2400
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=theme.ground)

    ax = fig.add_axes((0.07, 0.14, 0.38, 0.6))
    for i, (steps, val, _best) in enumerate(d["curves"]):
        ax.plot(steps, val, color=ACCENT, linewidth=1.4 * k, alpha=1.0 if i == 0 else 0.45)
    ax.axhline(d["climatology"], color=theme.muted, linewidth=1.0 * k, linestyle=(0, (3, 3)))
    ax.text(
        d["curves"][0][0][-1],
        d["climatology"] + 0.12,
        "climatology on the same hidden quakes",
        color=theme.muted,
        fontsize=8 * k,
        fontproperties=inter(400),
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("training step", color=theme.muted, fontsize=8 * k, fontproperties=inter(400))
    ax.set_ylabel(
        "bits per hidden quake (validation)",
        color=theme.muted,
        fontsize=8 * k,
        fontproperties=inter(400),
    )
    _axes(ax, theme, k)

    mx = fig.add_axes((0.55, 0.14, 0.40, 0.6))
    a = d["att"] / d["att"].max()
    order = np.argsort(a)
    others = np.ones(len(a), dtype=bool)
    others[d["hidden"]] = False
    sel = order[others[order]]
    mx.scatter(
        d["lon"][sel],
        d["lat"][sel],
        s=(2 + 40 * a[sel]) * k,
        c=a[sel],
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("att", [theme.faint, ACCENT]),
        linewidths=0,
    )
    i = d["hidden"]
    mx.scatter(
        [d["lon"][i]],
        [d["lat"][i]],
        s=160 * k,
        facecolors="none",
        edgecolors=theme.ink,
        linewidths=1.2 * k,
    )
    pad = 0.12
    mx.set_xlim(np.percentile(d["lon"], 1) - pad, np.percentile(d["lon"], 99) + pad)
    mx.set_ylim(np.percentile(d["lat"], 1) - pad, np.percentile(d["lat"], 99) + pad)
    mx.set_aspect(1 / np.cos(np.radians(42.3)))
    mx.set_xlabel("longitude °E", color=theme.muted, fontsize=8 * k, fontproperties=inter(400))
    mx.set_ylabel("latitude °N", color=theme.muted, fontsize=8 * k, fontproperties=inter(400))
    _axes(mx, theme, k)

    fig.text(
        0.07,
        0.90,
        "The first model learns, and starts to look in space",
        color=theme.ink,
        fontsize=15 * k,
        fontproperties=inter(600),
    )
    fig.text(
        0.07,
        0.845,
        "Left: masked loss on validation blocks while training, three seeds. Right: L'Aquila, "
        f"{HOURS_AFTER:g} hours after the Mw 6.3 of 6 April 2009, a year the model never saw; "
        "one quake hidden (ring).",
        color=theme.muted,
        fontsize=9 * k,
        fontproperties=inter(400),
    )
    fig.text(
        0.07,
        0.81,
        f"Head {d['head'] + 1} of layer {d['layer'] + 1}, the most local of 48, puts "
        f"{d['near_attention']:.0%} of its attention within {NEAR_KM:g} km of it, where "
        f"{d['near_share']:.0%} of the window's quakes are. The median head: "
        f"{d['median_near_attention']:.0%}. Looking, not yet hard.",
        color=theme.muted,
        fontsize=9 * k,
        fontproperties=inter(400),
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / out, dpi=dpi, facecolor=theme.ground)
    plt.close(fig)
    print("wrote", ASSETS / out)


def main() -> None:
    d = load()
    for theme in THEMES:
        draw(d, theme, f"first-model-{theme.name}.png")


if __name__ == "__main__":
    main()
