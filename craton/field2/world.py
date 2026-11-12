"""Field 2's heart on the world: trained on the world Field's tiles, asked about any box.

The heart reads exactly what the world Field reads (``configs/field/world.yaml``): the same
regions, the same cut (2007-01-01), the same tiles and the same pool of tile-days (every day
with a quake, quiet days sampled and weighted back). What differs is how a tile is held:

- each tile's ETAS is fitted as the pool fitted it (``Etas().fit`` on the tile's quakes before
  the cut, seed 0), and its parents are every quake since that fit's start (``t_ref``), exactly
  as ``Etas.forecast`` reads them;
- the pool's days before ``t_ref`` + the burn-in are dropped: there, ETAS itself has no past to
  read (central Italy's fit starts in 2005, when INGV's ML catalogue does);
- the answers are each cell's count at or above mc, and the loss is the Poisson likelihood of
  them under the heart's rate, weighted as the pool weighs its days.

A tile's kernel pieces (``heart.ring_entries``) are built once and cached beside the world
pool, in ``data/world/field2-<pool>-<bins>/``. Training, forecasting and CB-29 all go through
``heart.rates``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton import __version__
from craton.bench import etas
from craton.bench.baselines import Etas, _shares
from craton.bench.data import DAY, Events
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field.world import PoolTile, WorldField, pool_keep, tile_spec
from craton.field2.config import Field2Config, field2_hash, load_field2_config
from craton.field2.free import DeviceFree, FreeBox, free_rates, free_to_device, make_free
from craton.field2.heart import (
    CONTEXT,
    Box,
    DeviceBox,
    Entries,
    Heart,
    make_box,
    rates,
    to_device,
)
from craton.spec.timeaxis import to_seconds
from craton.train.checkpoint import checkpoints_dir

Log = Callable[[str], None]
FIELD2_VERSION = "h0.1"
#: Days forecast in one call of ``rates`` (memory: days × kernel pieces).
CHUNK_DAYS = 8


def _quiet(_: str) -> None:
    return None


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- a tile, held for the heart


@dataclass
class HeartTile:
    """One pool tile: its box, its pool days (days since ``t_ref``), answers and weights."""

    key: str
    region: str
    box: Box
    days: np.ndarray  # (D,) float64
    counts: np.ndarray  # (D, cells) float32, M ≥ mc
    weights: np.ndarray  # (D,) float32
    train: np.ndarray  # indices into days
    val: np.ndarray
    free: FreeBox | None = None


def _save_box(path: Path, box: Box, extra: dict[str, np.ndarray]) -> None:
    e = box.entries
    tmp = path.with_suffix(".tmp.npz")
    arrays: dict[str, Any] = dict(extra)
    np.savez(
        tmp,
        t=box.t,
        mag_bin=box.mag_bin,
        prod=box.prod,
        e_parent=e.parent.astype(np.int32),
        e_cell=e.cell.astype(np.int32),
        e_ring=e.ring.astype(np.int8),
        e_share=e.share.astype(np.float32),
        background=box.background,
        scalars=np.array([box.cascade, box.c, box.p, box.span, box.n_cells]),
        context=box.context,
        **arrays,
    )
    tmp.replace(path)


def _load_box(path: Path) -> tuple[Box, dict[str, np.ndarray]]:
    z = np.load(path)
    cascade, c, p, span, n_cells = (float(x) for x in z["scalars"])
    box = Box(
        t=z["t"],
        mag_bin=z["mag_bin"].astype(np.int64),
        prod=z["prod"],
        entries=Entries(
            z["e_parent"].astype(np.int64),
            z["e_cell"].astype(np.int64),
            z["e_ring"].astype(np.int64),
            z["e_share"].astype(np.float64),
        ),
        background=z["background"],
        cascade=cascade,
        c=c,
        p=p,
        span=span,
        context=z["context"],
        n_cells=int(n_cells),
    )
    return box, {k: z[k] for k in z.files if k in ("days",)}


def _save_free(path: Path, fb: FreeBox) -> None:
    e = fb.entries
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        t=fb.t,
        mag_bin=fb.mag_bin,
        e_parent=e.parent.astype(np.int32),
        e_cell=e.cell.astype(np.int32),
        e_ring=e.ring.astype(np.int8),
        e_share=e.share.astype(np.float32),
    )
    tmp.replace(path)


def _load_free(path: Path) -> FreeBox:
    z = np.load(path)
    return FreeBox(
        z["t"],
        z["mag_bin"].astype(np.int64),
        Entries(
            z["e_parent"].astype(np.int64),
            z["e_cell"].astype(np.int64),
            z["e_ring"].astype(np.int64),
            z["e_share"].astype(np.float64),
        ),
    )


def free_box(
    events: Events, mc: float, t_ref: float, t_end: float, grid: Any, cfg: Field2Config
) -> FreeBox:
    """The free head's quakes in a box, from ``t_ref`` to ``t_end`` (seconds)."""
    proj = etas.Projection.for_grid(grid)
    hist = events.window(t_ref, t_end).above(mc + cfg.net.free_magnitude[0])
    cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, t_ref)
    return make_free(cat, mc, grid, proj, cfg.bins.ruler, cfg.net.free_magnitude)


@dataclass
class Field2World:
    """Field 2's heart on the world Field's tiles; ``holdout`` leaves a region unread (CB-32)."""

    config_name: str = "heart"
    log: Log = _quiet
    holdout: str = ""
    #: Untrained (P11's born-as-ETAS check): every member is the network at step 0.
    born: bool = False
    _states: dict[int, list[dict[str, torch.Tensor]]] = field(default_factory=dict)

    @property
    def config(self) -> Field2Config:
        return load_field2_config(self.config_name)

    @property
    def world(self) -> WorldField:
        return WorldField(self.config.world_config, log=self.log)

    def prep_dir(self, root: Path) -> Path:
        cfg = self.config
        body = {"v": FIELD2_VERSION, "bins": cfg.bins.model_dump(mode="json")}
        h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]
        return root / "data" / "world" / f"field2-{self.world.pool_id()}-{h}"

    # ------------------------------------------------------------ the tiles

    def heart_tile(self, root: Path, p: PoolTile) -> HeartTile | None:
        from craton.bench.world import RegionView

        cfg = self.config
        wcfg = self.world.config
        assert wcfg.world is not None
        w = wcfg.world
        meta = json.loads((p.folder / "meta.json").read_text("utf-8"))
        path = self.prep_dir(root) / f"{p.tile.key}.npz"
        if not path.is_file():
            t0 = time.perf_counter()
            g = p.tile.grid
            view = RegionView(
                root=root,
                region=p.tile.region,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=w.cut,
            )
            fit = Etas().fit(tile_spec(p.tile, w.cut), view, 0)
            for name in ("mu", "k", "c", "p"):
                if abs(getattr(fit.params, name) / meta["etas"][name] - 1) > 1e-6:
                    raise RuntimeError(f"tile {p.tile.key}: ETAS does not refit as the pool did")
            events = view.train_events("world")
            starts_all = np.arange(
                to_seconds(wcfg.train.first_day), to_seconds(w.cut) - DAY + 1, DAY
            )
            keep, _, _ = pool_keep(p.tile, events, starts_all, w.quiet_share)
            if int(keep.sum()) != meta["days"]:
                raise RuntimeError(f"tile {p.tile.key}: pool days do not rebuild")
            proj = etas.Projection.for_grid(g)
            hist = events.window(fit.t_ref, to_seconds(w.cut)).above(fit.mc)
            cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, fit.t_ref)
            box = make_box(fit, cat, g, proj, cfg.bins)
            path.parent.mkdir(parents=True, exist_ok=True)
            _save_box(path, box, {"days": (starts_all[keep] - fit.t_ref) / DAY})
            self.log(
                f"field2 tile {p.tile.key}: {len(box.t):,} parents, {len(box.entries):,} "
                f"kernel pieces, {time.perf_counter() - t0:.0f}s"
            )
        box, extra = _load_box(path)
        days = extra["days"]
        fb = self._free_tile(root, p, box, days) if cfg.net.free_head else None
        ok = days >= etas.BURN_IN_DAYS
        if not ok.any():
            return None
        _, _, counts, weights, _ = p.arrays()
        train = np.asarray([i for i in p.train if ok[i]], np.int64)
        val = np.asarray([i for i in p.val if ok[i]], np.int64)
        if len(train) == 0 or len(val) == 0:
            return None
        return HeartTile(
            p.tile.key,
            p.tile.region,
            box,
            days,
            np.asarray(counts).sum(axis=-1).astype(np.float32),
            np.asarray(weights, np.float32),
            train,
            val,
            fb,
        )

    def _free_tile(self, root: Path, p: PoolTile, box: Box, days: np.ndarray) -> FreeBox:
        """The free head's quakes of a pool tile, cached beside its box."""
        from craton.bench.world import RegionView

        cfg = self.config
        wcfg = self.world.config
        assert wcfg.world is not None
        lo = cfg.net.free_magnitude[0]
        path = self.prep_dir(root) / f"{p.tile.key}.free{lo:+g}.npz"
        if not path.is_file():
            t0 = time.perf_counter()
            g = p.tile.grid
            view = RegionView(
                root=root,
                region=p.tile.region,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=wcfg.world.cut,
            )
            starts_all = np.arange(
                to_seconds(wcfg.train.first_day), to_seconds(wcfg.world.cut) - DAY + 1, DAY
            )
            events = view.train_events("world")
            keep, _, _ = pool_keep(p.tile, events, starts_all, wcfg.world.quiet_share)
            t_ref = float(starts_all[keep][0] - days[0] * DAY)
            mc = float(box.context[CONTEXT.index("mc")])
            fb = free_box(events, mc, t_ref, to_seconds(wcfg.world.cut), g, cfg)
            _save_free(path, fb)
            self.log(
                f"field2 free {p.tile.key}: {len(fb.t):,} quakes, {len(fb.entries):,} pieces, "
                f"{time.perf_counter() - t0:.0f}s"
            )
        return _load_free(path)

    def tiles(self, root: Path) -> list[HeartTile]:
        pool = self.world.training_pool(root) if not self.holdout else self._pool_without(root)
        out = []
        for p in pool:
            ht = self.heart_tile(root, p)
            if ht is not None:
                out.append(ht)
        return out

    def _pool_without(self, root: Path) -> list[PoolTile]:
        wf = WorldField(self.config.world_config, log=self.log, holdout=self.holdout)
        return wf.training_pool(root)

    # ------------------------------------------------------------ training

    def folder(self, root: Path, seed: int) -> Path:
        cfg = self.config
        without = f"-without-{self.holdout}" if self.holdout else ""
        return checkpoints_dir(root) / (
            f"field2-{cfg.name}-{FIELD2_VERSION}-{self.world.pool_id()}-{field2_hash(cfg)[:8]}"
            f"{without}-s{seed}"
        )

    def states(self, root: Path, seed: int) -> list[dict[str, torch.Tensor]]:
        if seed in self._states:
            return self._states[seed]
        cfg = self.config
        if self.born:
            net = Heart(cfg.bins, cfg.net)
            self._states[seed] = [{k: v.clone() for k, v in net.state_dict().items()}]
            return self._states[seed]
        path = self.folder(root, seed) / "model.pt"
        if path.is_file():
            body = torch.load(path, map_location="cpu", weights_only=True)
            self._states[seed] = list(body["members"])
            return self._states[seed]
        tiles = self.tiles(root)
        t0 = time.perf_counter()
        states, runs = [], []
        for m in range(cfg.train.members):
            st, info = train_heart(tiles, cfg, seed * 100 + m, self.log)
            states.append(st)
            runs.append(info)
        folder = self.folder(root, seed)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": seed,
            "config": cfg.model_dump(mode="json"),
            "config_hash": field2_hash(cfg),
            "field2_version": FIELD2_VERSION,
            "pool": self.world.pool_id(),
            "holdout": self.holdout or None,
            "craton_version": __version__,
            "tiles": [t.key for t in tiles],
            "training_days": int(sum(len(t.train) for t in tiles)),
            "members": runs,
            "parameters": int(
                sum(v.numel() for k, v in states[0].items() if not k.startswith(("bin_x", "ctx")))
            ),
            "train_seconds": time.perf_counter() - t0,
        }
        torch.save({"members": states}, folder / "model.pt")
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        (folder / "card.md").write_text(card(meta), "utf-8")
        self._states[seed] = states
        return states

    # ------------------------------------------------------------ forecasting

    def box_rates(
        self, box: Box, days: np.ndarray, seed: int, root: Path, fb: FreeBox | None = None
    ) -> np.ndarray:
        """(days, cells) the members' mean rate at or above mc for ``box``."""
        cfg = self.config
        dev = _device()
        db = to_device(box, cfg.bins, dev)
        fd = free_to_device(fb, len(cfg.bins.ruler), box.n_cells, dev) if fb else None
        out = np.zeros((len(days), box.n_cells))
        states = self.states(root, seed)
        for st in states:
            net = Heart(cfg.bins, cfg.net).to(dev)
            net.load_state_dict(st)
            net.eval()
            with torch.no_grad():
                out += predict(net, db, fd, days, cfg) / len(states)
        return out

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        """Daily forecasts of any box: ETAS fitted on the box's past (its numbers are the
        context), every quake since the fit's start as a parent, the heart's weights on top.
        Shaped exactly as ``Etas.forecast``, so an untrained heart scores as ETAS."""
        cfg = self.config
        f = Etas().fit(spec, ctx, seed)
        proj = etas.Projection.for_grid(spec.grid)
        hist = ctx.events.window(f.t_ref, spec.window.t1).above(f.mc)
        cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, f.t_ref)
        box = make_box(f, cat, spec.grid, proj, cfg.bins, span=spec.period_days)
        fb = None
        if cfg.net.free_head:
            fb = free_box(ctx.events, f.mc, f.t_ref, spec.window.t1, spec.grid, cfg)
        days = (spec.day_starts - f.t_ref) / DAY
        r = self.box_rates(box, days, seed, ctx.root, fb)
        scale, shares = _shares(f.b, f.mc, spec.grid)
        return GriddedForecast(r * scale, shares)

    def npp_factor(self, engine: Any, events: Events, seed: int) -> np.ndarray:
        """CB-29: the factor on EarthquakeNPP's ETAS (``craton.field2.npp``)."""
        from craton.field2.npp import factor

        return factor(self, engine, events, seed)


def heart_rates(
    net: Heart, db: DeviceBox, fd: DeviceFree | None, days: torch.Tensor, cfg: Field2Config
) -> torch.Tensor:
    """(days, cells) the heart's rate at or above mc: the ETAS head, and the free head if on."""
    log_w, log_w0 = net(db.context)
    lam = rates(db, days, log_w, log_w0, cfg.bins)
    log_f = net.free_log_rate(db.context)
    if fd is not None and log_f is not None:
        n_static = len(cfg.net.free_magnitude) * len(cfg.bins.ruler)
        lam = lam + free_rates(fd, days, log_f, cfg.bins.age_days, n_static, db.box.span)
    return lam


def predict(
    net: Heart, db: DeviceBox, fd: DeviceFree | None, days: np.ndarray, cfg: Field2Config
) -> np.ndarray:
    out = np.zeros((len(days), db.box.n_cells))
    d = torch.as_tensor(days, dtype=torch.float64, device=db.t.device)
    for i in range(0, len(days), CHUNK_DAYS):
        out[i : i + CHUNK_DAYS] = (
            heart_rates(net, db, fd, d[i : i + CHUNK_DAYS], cfg).double().cpu().numpy()
        )
    return out


def train_heart(
    tiles: list[HeartTile], cfg: Field2Config, member_seed: int, log: Log = _quiet
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """One heart trained on every tile; weighted Poisson loss per quake at or above mc, early
    stopping on every tile's validation days. The loss at step 0 is ETAS's."""
    dev = _device()
    tc = cfg.train
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    net = Heart(cfg.bins, cfg.net).to(dev)
    ctxs = np.stack([t.box.context for t in tiles])
    net.ctx_mean.copy_(torch.as_tensor(ctxs.mean(axis=0), dtype=torch.float32))  # type: ignore[operator]
    net.ctx_std.copy_(  # type: ignore[operator]
        torch.as_tensor(np.maximum(ctxs.std(axis=0), 1e-3), dtype=torch.float32)
    )
    boxes = [to_device(t.box, cfg.bins, dev) for t in tiles]
    frees = [
        free_to_device(t.free, len(cfg.bins.ruler), t.box.n_cells, dev) if t.free else None
        for t in tiles
    ]
    # The free head is born silent, so its first gradients are ~1e-12: Adam's usual eps (1e-8)
    # would swallow them and it would never wake. Its own group, with an eps below them.
    free = [p for n, p in net.named_parameters() if n.startswith("free.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("free.")]
    groups: list[dict[str, Any]] = [{"params": rest}]
    if free:
        groups.append({"params": free, "eps": 1e-16})
    opt = torch.optim.AdamW(groups, lr=tc.lr, weight_decay=tc.weight_decay)
    share = np.array([len(t.train) for t in tiles], dtype=np.float64)
    share /= share.sum()

    def nll(k: int, sel: np.ndarray) -> tuple[torch.Tensor, float]:
        t = tiles[k]
        sel = np.sort(sel)
        d = torch.as_tensor(t.days[sel], dtype=torch.float64, device=dev)
        lam = heart_rates(net, boxes[k], frees[k], d, cfg).clamp(min=1e-30)
        y = torch.as_tensor(t.counts[sel], device=dev)
        wt = torch.as_tensor(t.weights[sel], device=dev)[:, None]
        return (wt * (lam - y * lam.log())).sum(), float((wt * y).sum())

    def validate() -> float:
        net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for k, t in enumerate(tiles):
                for i in range(0, len(t.val), CHUNK_DAYS):
                    loss, count = nll(k, t.val[i : i + CHUNK_DAYS])
                    tot, n = tot + float(loss), n + count
        net.train()
        return tot / max(n, 1.0)

    best = validate()
    start = best
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad = epochs = 0
    t0 = time.perf_counter()
    for epoch in range(tc.max_epochs):
        epochs = epoch + 1
        for _ in range(tc.steps_per_epoch):
            # Several tiles per step: one tile's days are one sequence's mood.
            # The step's quake count first, then one backward per tile: the same gradient as
            # one backward over all of them, with one tile's graph in memory at a time.
            opt.zero_grad(set_to_none=True)
            picks = []
            for k in rng.choice(len(tiles), tc.tiles_per_step, p=share):
                t = tiles[int(k)]
                picks.append(
                    (int(k), rng.choice(t.train, min(tc.days_per_step, len(t.train)), False))
                )
            total = sum(
                float((tiles[k].weights[s][:, None] * tiles[k].counts[s]).sum()) for k, s in picks
            )
            for k, sel in picks:
                loss, _ = nll(k, sel)
                (loss / max(total, 1.0)).backward()  # type: ignore[no-untyped-call]
            opt.step()
        v = validate()
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        log(
            f"  heart {member_seed} epoch {epochs} validation {v:.5f} (ETAS {start:.5f}, best "
            f"{best:.5f}) {(time.perf_counter() - t0) / 60:.1f} min"
        )
        if bad >= tc.patience:
            break
    return {k: v.cpu() for k, v in state.items()}, {
        "epochs": float(epochs),
        "validation_etas": start,
        "validation_best": best,
        "minutes": (time.perf_counter() - t0) / 60,
    }


def card(meta: dict[str, Any]) -> str:
    """The model card of one Field 2 heart checkpoint."""
    runs = meta["members"]
    gain = ", ".join(f"{r['validation_etas'] - r['validation_best']:.4f}" for r in runs)
    regions = sorted({t.split("-")[0] for t in meta["tiles"]})
    return (
        f"# Field 2, the heart · seed {meta['seed']}\n\n"
        "- **What it does:** daily forecasts of any 0.1° box on Earth. Every past quake sends "
        "its ETAS weight to every cell; the heart re-weighs it by the quake's age (log clock), "
        "size and distance in its own rupture lengths, from the region's ETAS numbers. Born as "
        "ETAS: an untrained heart is ETAS exactly.\n"
        + (
            f"- **Left out:** {meta['holdout']}, never read (CB-32's fold).\n"
            if meta.get("holdout")
            else ""
        )
        + f"- **Learned from:** {len(meta['tiles'])} tiles of {', '.join(regions)}, "
        f"{meta['training_days']:,} tile-days before the world cut.\n"
        f"- **Reads:** {len(CONTEXT)} context numbers ({', '.join(CONTEXT)}), no place, no "
        "coordinate.\n"
        f"- **Size:** {meta['parameters']:,} parameters per member, {len(runs)} member(s).\n"
        f"- **Training:** validation gain over ETAS (nats per quake) {gain}; "
        f"{meta['train_seconds'] / 60:.1f} min.\n"
        f"- **Code:** Craton {meta['craton_version']}, Field 2 {meta['field2_version']}, pool "
        f"`{meta['pool']}`, config hash `{meta['config_hash']}`.\n"
    )
