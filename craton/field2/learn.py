"""Training the rebuilt heart as it will be tested (P11).

- **The days**: every pool day of every tile, each forecast by the ETAS of its block
  (``craton.field2.tested``): fitted before the block, never on the day.
- **The members**: ``train.members`` hearts, each with its own seed. With ``stop_outside``,
  member *m* sets one training region aside (the regions in turn) and learns from the others;
  the region it set aside decides when it stops. Early stopping on days of the regions being
  learned only measures how well it memorises them, and CB-32 asks about a region never read.
- **The loss**: the Poisson likelihood of each cell's count at or above mc, weighted as the
  pool weighs its days (quiet days were sampled), per quake.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from craton.field2.config import Field2Config
from craton.field2.heart import to_device
from craton.field2.place import PlaceHeart, quakes_to_device
from craton.field2.tested import OriginTile

Log = Callable[[str], None]
#: Days forecast in one call (memory: days × kernel pieces).
CHUNK = 8
#: Validation days per stopping tile at most (a fixed draw, the same for every epoch).
MAX_STOP_DAYS = 120


def _quiet(_: str) -> None:
    return None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Held:
    """One origin tile on the device."""

    def __init__(self, ot: OriginTile, cfg: Field2Config, dev: torch.device) -> None:
        self.ot = ot
        db = to_device(ot.box, cfg.bins, dev)
        # int32 indices: every origin of every tile lives on the 6 GB card at once
        db.e_parent = db.e_parent.int()
        db.e_cell = db.e_cell.int()
        db.e_static = db.e_static.int()
        self.db = db
        self.dq = (
            quakes_to_device(ot.quakes, ot.t_ref, cfg.net.state_below_mc, dev)
            if cfg.net.state
            else None
        )
        self.shape = (ot.grid.n_lat, ot.grid.n_lon)
        self.counts = ot.counts.sum(-1)
        self.dev = dev

    def nll(self, net: PlaceHeart, sel: np.ndarray) -> tuple[torch.Tensor, float]:
        sel = np.sort(sel)
        d = torch.as_tensor(self.ot.days[sel], dtype=torch.float64, device=self.dev)
        lam = net(self.db, self.dq, d, self.shape).clamp(min=1e-30)
        y = torch.as_tensor(self.counts[sel], device=self.dev)
        wt = torch.as_tensor(self.ot.weights[sel], device=self.dev)[:, None]
        return (wt * (lam - y * lam.log())).sum(), float((wt * y).sum())


def stop_days(ot: OriginTile, rng: np.random.Generator) -> np.ndarray:
    idx = np.nonzero(ot.is_val)[0]
    if len(idx) == 0:
        idx = np.arange(len(ot.days))
    if len(idx) > MAX_STOP_DAYS:
        idx = np.sort(rng.choice(idx, MAX_STOP_DAYS, replace=False))
    return idx


def train_place(
    train: list[OriginTile],
    stop: list[OriginTile],
    cfg: Field2Config,
    member_seed: int,
    log: Log = _quiet,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One rebuilt heart: learns from ``train``, stops on ``stop`` (its pool validation days,
    or all of a region it does not learn from)."""
    dev = _device()
    tc = cfg.train
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    net = PlaceHeart(cfg.bins, cfg.net).to(dev)
    ctxs = np.stack([t.box.context for t in train])
    net.ctx_mean.copy_(torch.as_tensor(ctxs.mean(axis=0), dtype=torch.float32))  # type: ignore[operator]
    net.ctx_std.copy_(  # type: ignore[operator]
        torch.as_tensor(np.maximum(ctxs.std(axis=0), 1e-3), dtype=torch.float32)
    )
    held = [Held(t, cfg, dev) for t in train]
    same = {id(t) for t in train}
    stop_held = [h for h in held if id(h.ot) in {id(s) for s in stop}] + [
        Held(t, cfg, dev) for t in stop if id(t) not in same
    ]
    v_rng = np.random.default_rng(10_000 + member_seed)
    stop_sel = [stop_days(h.ot, v_rng) for h in stop_held]
    # Learning days: when stopping on its own tiles' validation blocks, those are kept out.
    learn_sel = [
        np.nonzero(~h.ot.is_val)[0]
        if id(h.ot) in {id(s) for s in stop}
        else np.arange(len(h.ot.days))
        for h in held
    ]
    share = np.array([len(s) for s in learn_sel], dtype=np.float64)
    share /= share.sum()
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    def validate() -> float:
        net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for h, sel in zip(stop_held, stop_sel, strict=True):
                for i in range(0, len(sel), CHUNK):
                    loss, count = h.nll(net, sel[i : i + CHUNK])
                    tot, n = tot + float(loss), n + count
        net.train()
        return tot / max(n, 1.0)

    best = start = validate()
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad = epochs = 0
    t0 = time.perf_counter()
    for epoch in range(tc.max_epochs):
        epochs = epoch + 1
        for _ in range(tc.steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            picks = []
            for k in rng.choice(len(held), tc.tiles_per_step, p=share):
                s = learn_sel[int(k)]
                picks.append((int(k), rng.choice(s, min(tc.days_per_step, len(s)), False)))
            total = sum(
                float((held[k].ot.weights[s][:, None] * held[k].counts[s]).sum()) for k, s in picks
            )
            for k, sel in picks:
                loss, _ = held[k].nll(net, sel)
                (loss / max(total, 1.0)).backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        v = validate()
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        log(
            f"  heart {member_seed} epoch {epochs} stop-loss {v:.5f} (ETAS {start:.5f}, best "
            f"{best:.5f}) {(time.perf_counter() - t0) / 60:.1f} min"
        )
        if bad >= tc.patience:
            break
    return {k: v.cpu() for k, v in state.items()}, {
        "epochs": float(epochs),
        "validation_etas": start,
        "validation_best": best,
        "minutes": (time.perf_counter() - t0) / 60,
        "stop_regions": sorted({t.region for t in stop}),
        "learn_regions": sorted({t.region for t in train}),
    }


def members_plan(
    tiles: list[OriginTile], cfg: Field2Config
) -> list[tuple[list[OriginTile], list[OriginTile]]]:
    """(learn, stop) tiles of every member."""
    regions = sorted({t.region for t in tiles})
    out = []
    for m in range(cfg.train.members):
        if cfg.train.stop_outside and len(regions) > 1:
            aside = regions[m % len(regions)]
            out.append(
                ([t for t in tiles if t.region != aside], [t for t in tiles if t.region == aside])
            )
        else:
            out.append((tiles, tiles))
    return out


def predict_place(net: PlaceHeart, held: Held, days: np.ndarray) -> np.ndarray:
    out = np.zeros((len(days), held.ot.box.n_cells))
    d = torch.as_tensor(days, dtype=torch.float64, device=held.dev)
    with torch.no_grad():
        for i in range(0, len(days), CHUNK):
            out[i : i + CHUNK] = (
                net(held.db, held.dq, d[i : i + CHUNK], held.shape).double().cpu().numpy()
            )
    return out
