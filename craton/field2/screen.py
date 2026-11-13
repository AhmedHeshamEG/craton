"""``craton screen``: what ETAS misses, and which information explains it, before anything is built.

A model can only learn what its inputs carry. Before Field 2's heart is rebuilt (P11), this
check asks the data, in minutes and with no network to train beyond a tiny one:

1. **The miss.** On the world pool held *as tested* (``craton.field2.tested``: each day
   forecast by an ETAS fitted only before its two-year block), how far is the day's count from
   ETAS's expectation? Per region: came ÷ expected, and the spread of the daily log ratio.
2. **Who explains it.** For each candidate set of information, one small correction
   ``λ = ETAS · e^{f(x)}`` (one hidden layer, 16 wide) is fitted on three regions and scored on
   the fourth (leave one region out, as CB-32): the gain over ETAS in bits per quake at mc + 1
   (CB-32's threshold) and at mc. Every input is read from the past only and is dimensionless
   (a count over what the cell's background would give in the same time).

The sets: the cell's own quakes at mc and above (1, 7, 30, 365 days), the small ones just below
mc, the neighbours' (5 × 5 and 15 × 15 cells), how excited ETAS is (its rate over the
background, cell and box), the region's ETAS numbers (the first heart's context), the Field's
54 counters (what the Field reads), and a **yardstick**: the running came ÷ expected (30 and 365
days, the box and the 15 × 15 cells around). The yardstick is a measure of what there is to win;
it is never a model input (D-153).

Zero-count cells are sampled (``ZERO_SHARE``) and weighted back, as the pool does quiet days.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from craton.field2.heart import CONTEXT, rates, to_device
from craton.field2.tested import OriginTile, build_all

Log = Callable[[str], None]
WINDOWS = (1, 7, 30, 365)
ZERO_SHARE = 0.05
REGIONS = ("central_italy", "california", "japan", "new_zealand")


def _quiet(_: str) -> None:
    return None


@dataclass
class Rows:
    """Cell-days of one origin tile: features by group, answers, ETAS, weights."""

    region: str
    groups: dict[str, np.ndarray]
    y: np.ndarray  # (N, 2): [mc, mc+1) and >= mc+1
    etas: np.ndarray  # (N,) expected at >= mc
    w: np.ndarray  # (N,)
    big_share: np.ndarray  # (N,) 10^-b: the share of quakes at >= mc + 1


def _pool2d(x: torch.Tensor, shape: tuple[int, int], k: int) -> torch.Tensor:
    """Sum over a k × k window of cells (zero outside the box), x: (D, cells)."""
    d = x.shape[0]
    g = x.reshape(d, 1, *shape)
    out = F.avg_pool2d(g, k, stride=1, padding=k // 2, count_include_pad=True) * (k * k)
    return out.reshape(d, -1)


def tile_rows(ot: OriginTile, memory: np.ndarray | None, rng: np.random.Generator) -> Rows:
    """Every feature group for the pool days of one origin tile (GPU)."""
    from craton.field2.config import load_field2_config

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bins = load_field2_config("heart").bins
    box = ot.box
    shape = (ot.grid.n_lat, ot.grid.n_lon)
    db = to_device(box, bins, dev)
    first = math.floor(float(ot.days.min()))
    last = math.floor(float(ot.days.max()))
    cal = torch.arange(first, last + 1, dtype=torch.float64, device=dev)
    n_static = len(bins.magnitude) * len(bins.ruler)
    zero_w = torch.zeros(len(bins.age_days) * n_static, device=dev)
    with torch.no_grad():
        exp = torch.cat(
            [
                rates(db, cal[i : i + 8], zero_w, torch.zeros((), device=dev), bins)
                for i in range(0, len(cal), 8)
            ]
        ).float()  # (Dc, cells), >= mc, cascade in

    # daily counts per cell from [first - 365, last], at mc and above, and just below
    lo = first - 365
    n_days = last - lo + 1
    q = ot.quakes
    t_rel = (q.time - ot.t_ref) / 86_400.0
    day = np.floor(t_rel).astype(np.int64) - lo
    ok = (day >= 0) & (day < n_days)
    big = q.m_rel >= -1e-6
    daily = np.zeros((2, n_days, box.n_cells), np.float32)
    np.add.at(daily, (big[ok].astype(np.int64), day[ok], q.cell[ok]), 1.0)
    cum = torch.as_tensor(
        np.concatenate(
            [np.zeros((2, 1, box.n_cells), np.float32), np.cumsum(daily, axis=1)], axis=1
        ),
        device=dev,
    )

    di = torch.as_tensor(np.floor(ot.days).astype(np.int64) - lo, device=dev)  # index of day d
    bg = torch.as_tensor(box.background, dtype=torch.float32, device=dev)  # >= mc per day
    bg_w = {wd: bg[None, :] * wd for wd in WINDOWS}

    def count(kind: int, wd: int) -> torch.Tensor:
        return cum[kind, di] - cum[kind, (di - wd).clamp(min=0)]  # quakes in [d - wd, d)

    groups: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("own", "small", "neighbours", "excitement", "yardstick")
    }
    for wd in WINDOWS:
        c_big, c_small = count(1, wd), count(0, wd)
        groups["own"].append(torch.log1p(c_big / bg_w[wd]))
        groups["small"].append(torch.log1p(c_small / bg_w[wd]))
        for k in (5, 15):
            num = _pool2d(c_big + c_small, shape, k)
            den = _pool2d(bg_w[wd].expand_as(c_big), shape, k)
            groups["neighbours"].append(torch.log1p(num / den))
    e_day = exp[di - (first - lo)]  # ETAS on the pool days
    base = box.cascade * bg[None, :]
    groups["excitement"].append(torch.log(e_day / base))
    groups["excitement"].append(
        torch.log(e_day.sum(1, keepdim=True) / base.sum(1, keepdim=True)).expand_as(e_day)
    )
    # The yardstick: the running came / expected (box and 15 × 15 cells), never an input.
    exp_full = torch.zeros(n_days, box.n_cells, device=dev)
    exp_full[first - lo :] = exp
    # Days before the block are not this ETAS's: they count as expected = came (no evidence).
    came = torch.as_tensor(daily[1], device=dev)
    exp_full[: first - lo] = came[: first - lo]
    cum_e = torch.cat([torch.zeros(1, box.n_cells, device=dev), exp_full.cumsum(0)])
    cum_c = torch.cat([torch.zeros(1, box.n_cells, device=dev), came.cumsum(0)])
    for wd in (30, 365):
        ce = cum_c[di] - cum_c[(di - wd).clamp(min=0)]
        ee = cum_e[di] - cum_e[(di - wd).clamp(min=0)]
        groups["yardstick"].append(
            torch.log((ce.sum(1, keepdim=True) + 1) / (ee.sum(1, keepdim=True) + 1)).expand_as(ce)
        )
        if wd == 365:
            groups["yardstick"].append(
                torch.log((_pool2d(ce, shape, 15) + 1) / (_pool2d(ee, shape, 15) + 1))
            )

    y = torch.as_tensor(ot.counts, device=dev)  # (D, cells, 2)
    keep = (y.sum(-1) > 0) | (torch.as_tensor(rng.random(y.shape[:2]), device=dev) < ZERO_SHARE)
    wt = torch.as_tensor(ot.weights, device=dev)[:, None].expand_as(keep)
    wt = torch.where(y.sum(-1) > 0, wt, wt / ZERO_SHARE)
    di_, ci = torch.nonzero(keep, as_tuple=True)

    out = {k: torch.stack(v, -1)[di_, ci].cpu().numpy() for k, v in groups.items()}
    ctx = np.broadcast_to(box.context.astype(np.float32), (len(di_), len(CONTEXT)))
    out["context"] = np.ascontiguousarray(ctx)
    if memory is not None:
        m = np.asarray(memory[ot.pool_idx], np.float32)
        out["field"] = m[di_.cpu().numpy(), ci.cpu().numpy()]
    b = float(box.context[CONTEXT.index("b")])
    return Rows(
        ot.region,
        out,
        y[di_, ci].cpu().numpy(),
        e_day[di_, ci].cpu().numpy(),
        wt[di_, ci].cpu().numpy(),
        np.full(len(di_), 10.0 ** (-b), np.float32),
    )


#: The candidate sets, each a list of groups.
SETS: dict[str, tuple[str, ...]] = {
    "own quakes": ("own",),
    "+ small quakes": ("own", "small"),
    "+ neighbours (the state)": ("own", "small", "neighbours"),
    "+ ETAS excitement": ("own", "small", "neighbours", "excitement"),
    "ETAS excitement alone": ("excitement",),
    "region's ETAS numbers alone": ("context",),
    "state + region's ETAS numbers": ("own", "small", "neighbours", "context"),
    "the Field's 54 counters": ("field",),
    "yardstick: came / expected": ("yardstick",),
    "state + yardstick": ("own", "small", "neighbours", "yardstick"),
}


def _fit_eval(
    train: list[Rows], test: list[Rows], groups: tuple[str, ...], seed: int = 0
) -> dict[str, float]:
    """Fit λ = ETAS · e^{f(x)} on ``train``; bits per quake over ETAS on ``test``."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def stack(rows: list[Rows]) -> tuple[torch.Tensor, ...]:
        x = np.concatenate([np.concatenate([r.groups[g] for g in groups], 1) for r in rows])
        y = np.concatenate([r.y for r in rows])
        return (
            torch.as_tensor(x, device=dev),
            torch.as_tensor(y, device=dev),
            torch.as_tensor(np.concatenate([r.etas for r in rows]), device=dev),
            torch.as_tensor(np.concatenate([r.w for r in rows]), device=dev),
            torch.as_tensor(np.concatenate([r.big_share for r in rows]), device=dev),
        )

    xt, yt, et, wt, _ = stack(train)
    mean, std = xt.mean(0), xt.std(0).clamp(min=1e-3)
    torch.manual_seed(seed)
    last = nn.Linear(16, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    net = nn.Sequential(nn.Linear(xt.shape[1], 16), nn.GELU(), last).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    n_all = float((wt * yt.sum(1)).sum())
    for _ in range(400):
        opt.zero_grad()
        lam = et * torch.exp(3 * torch.tanh(net((xt - mean) / std).squeeze(-1) / 3))
        loss = (wt * (lam - yt.sum(1) * torch.log(lam.clamp(min=1e-30)))).sum() / n_all
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    xs, ys, es, ws, bs = stack(test)
    with torch.no_grad():
        f = torch.exp(3 * torch.tanh(net((xs - mean) / std).squeeze(-1) / 3))

    def gain(lam_e: torch.Tensor, y: torch.Tensor) -> float:
        lam_m = lam_e * f
        ll = (
            ws * (y * torch.log(lam_m.clamp(min=1e-30) / lam_e.clamp(min=1e-30)) - lam_m + lam_e)
        ).sum()
        n = float((ws * y).sum())
        return float(ll) / max(n, 1.0) / math.log(2)

    return {
        "gain_mc1": gain(es * bs, ys[:, 1]),
        "gain_mc": gain(es, ys.sum(1)),
        "factor_mean": float((ws * f * es).sum() / (ws * es).sum()),
    }


def miss(rows: list[Rows]) -> dict[str, dict[str, float]]:
    """ETAS's miss per region: came ÷ expected (all pool days, weighted)."""
    out = {}
    for reg in REGIONS:
        rs = [r for r in rows if r.region == reg]
        if not rs:
            continue
        came = sum(float((r.w * r.y.sum(1)).sum()) for r in rs)
        exp = sum(float((r.w * r.etas).sum()) for r in rs)
        came1 = sum(float((r.w * r.y[:, 1]).sum()) for r in rs)
        exp1 = sum(float((r.w * r.etas * r.big_share).sum()) for r in rs)
        out[reg] = {
            "came": came,
            "expected": exp,
            "ratio": came / exp,
            "came_mc1": came1,
            "expected_mc1": exp1,
            "ratio_mc1": came1 / exp1,
        }
    return out


def run(root: Path, log: Log = _quiet, years: float = 2.0, seed: int = 0) -> dict[str, Any]:
    from craton.field2.world import Field2World

    t0 = time.perf_counter()
    world = Field2World("heart", log=log)
    tiles = build_all(world, root, years, log)
    pool_dir = world.world.pool_dir(root)
    rng = np.random.default_rng(seed)
    rows = []
    for ot in tiles:
        mem = np.load(pool_dir / ot.tile / "memory.npy", mmap_mode="r")
        rows.append(tile_rows(ot, mem, rng))
    log(
        f"screen: {sum(len(r.w) for r in rows):,} cell-days from {len(tiles)} origins, "
        f"{(time.perf_counter() - t0) / 60:.1f} min"
    )
    out: dict[str, Any] = {
        "miss": miss(rows),
        "sets": {},
        "origins": len(tiles),
        "cell_days": int(sum(len(r.w) for r in rows)),
        "years": years,
    }
    for name, groups in SETS.items():
        per: dict[str, Any] = {}
        for reg in REGIONS:
            train = [r for r in rows if r.region != reg]
            test = [r for r in rows if r.region == reg]
            if not test:
                continue
            per[reg] = _fit_eval(train, test, groups, seed)
        mean1 = float(np.mean([v["gain_mc1"] for v in per.values()]))
        mean0 = float(np.mean([v["gain_mc"] for v in per.values()]))
        out["sets"][name] = {
            "groups": list(groups),
            "folds": per,
            "mean_gain_mc1": mean1,
            "mean_gain_mc": mean0,
        }
        log(
            f"  {name:32s} mc+1 {mean1:+.4f}  mc {mean0:+.4f}  "
            + "  ".join(f"{k[:5]} {v['gain_mc1']:+.3f}" for k, v in per.items())
        )
    out["seconds"] = time.perf_counter() - t0
    return out


def main(root: Path, out_path: Path, log: Log = print) -> int:
    res = run(root, log)
    if out_path.is_file():
        # the expectation is written by hand before training (P11 step 2); a rerun keeps it
        old = json.loads(out_path.read_text("utf-8"))
        if "expectation" in old:
            res["expectation"] = old["expectation"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2) + "\n", "utf-8")
    log(f"screen written to {out_path}")
    return 0
