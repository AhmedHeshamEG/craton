"""The world Field: one Field learned from many regions at once, then asked about any region.

The Italy Field (``craton.field.forecaster``) learns one patch of crust. Nothing in its
network knows a place, though: a cell is its memory of past quakes, ETAS's rate and where it
sits *inside its box*. So the same network can read any box on Earth, provided every region
speaks in its own magnitudes. Each region has a completeness mc (``Region.completeness``):
the memory counts from mc − 0.5, a quake of magnitude mc weighs 1 in the productivity
counters, the network learns from mc and is scored from mc + 1. Central Italy is mc = 2:
exactly the Italy Field's terms.

Training, for one seed:

1. Every region in ``world.regions`` is cut into tiles of CB-02's size (3° × 3.5°, 1,050
   cells); tiles with fewer than ``min_tile_quakes`` quakes at mc are dropped (open sea).
2. Each tile gets its own ETAS, fitted on its quakes before ``world.cut``: the starting line
   is the tile's own floor, as in Italy.
3. Days from ``train.first_day`` to the cut. Days with a quake at mc somewhere in the tile
   are all kept; quiet days are kept with probability ``quiet_share`` and weigh
   1 / ``quiet_share`` in the loss, so the loss is the same in expectation and the pool is
   small enough for 6 GB and a laptop disk. One 30-day block in ``validation_every`` is
   validation, as in Italy. The pool is data (the same for every seed), cached on disk.
4. ``train.members`` Fields are trained, each step on ``days_per_step`` days of one tile
   (tiles drawn in proportion to their training days), each member stopped early on the
   validation days of every tile.

Forecasting any box (a CB-22 sequence, CB-02): ETAS is fitted on the box's past at its
region's mc (the CB-22 reference itself), the days are built from the box's quakes, and the
members' average correction is applied, exactly as the Italy Field does.
"""

from __future__ import annotations

import contextlib
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
from craton.adapters.region import load_region
from craton.bench import csep
from craton.bench.baselines import Etas
from craton.bench.data import DAY
from craton.bench.splits import Window
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field.config import FieldConfig, field_hash, load_field_config
from craton.field.forecaster import (
    INFER_DAYS,
    VALIDATION_BLOCK_DAYS,
    _counts,
    build_days,
    magnitude_shares,
)
from craton.field.memory import MemorySpec
from craton.field.model import FieldNet
from craton.spec.timeaxis import to_iso, to_seconds
from craton.train.checkpoint import checkpoints_dir

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
Log = Callable[[str], None]

WORLD_VERSION = "w0.1"
#: The memory counts from this far below the region's completeness (Italy: 2.0 → 1.5).
MEMORY_BELOW_MC = 0.5
#: Scored from this far above it (Italy: M3).
SCORED_ABOVE_MC = 1.0
#: ETAS fits on at most this many years before the cut when a region names no start.
FIT_YEARS = 20
#: The pool's tile-days are drawn with this salt, so a pool is the same on every machine.
POOL_SALT = "craton-world-pool-v1"


def _quiet(_: str) -> None:
    return None


def memory_spec(mc: float, start: str) -> MemorySpec:
    return MemorySpec(m_min=mc - MEMORY_BELOW_MC, start=to_seconds(start), m_ref=mc)


@dataclass(frozen=True)
class Tile:
    """One box of a region the world Field learns from."""

    region: str
    grid: csep.Grid
    mc: float
    fit_from: str

    @property
    def key(self) -> str:
        g = self.grid
        return f"{self.region}-{g.lat_min:g}_{g.lat_max:g}_{g.lon_min:g}_{g.lon_max:g}"


def tile_grids(
    box: tuple[float, float, float, float], mc: float, tile_lat: float, tile_lon: float
) -> list[csep.Grid]:
    """The region's box in tiles from its south-west corner; edge tiles are what is left."""
    lat0, lat1, lon0, lon1 = box
    out = []
    la = lat0
    while la < lat1 - 1e-9:
        lo = lon0
        top = round(min(la + tile_lat, lat1), 6)
        while lo < lon1 - 1e-9:
            right = round(min(lo + tile_lon, lon1), 6)
            out.append(csep.Grid(la, top, lo, right, cell_deg=0.1, m_min=mc, m_max=9.5))
            lo = right
        la = top
    return out


def tile_spec(tile: Tile, cut: str) -> ForecastSpec:
    """The question a tile's training days answer, in its region's terms (for its ETAS)."""
    window = Window(f"world_{tile.key}", cut, to_iso(to_seconds(cut) + DAY) or cut)
    return ForecastSpec(
        task_id=f"world-{tile.key}",
        fit_split="world",
        window=window,
        grid=tile.grid,
        mc=tile.mc,
        fit_from=tile.fit_from,
        cut=cut,
        thresholds=(tile.mc + SCORED_ABOVE_MC, tile.mc),
        max_magnitude=9.5,
    )


@dataclass
class PoolTile:
    """One tile's training and validation days, read from disk as needed."""

    tile: Tile
    folder: Path
    train: IntArray
    val: IntArray

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(memory, log base, counts, weights, log inputs): the base is what the correction
        multiplies; the inputs what the Field reads (the same unless ``inputs.npy`` says
        otherwise: CB-29's base is a day's realized ETAS, its input the midnight forecast)."""
        f = self.folder
        base = np.load(f / "log_etas.npy", mmap_mode="r")
        inputs = np.load(f / "inputs.npy", mmap_mode="r") if (f / "inputs.npy").is_file() else base
        return (
            np.load(f / "memory.npy", mmap_mode="r"),
            base,
            np.load(f / "counts.npy", mmap_mode="r"),
            np.load(f / "weights.npy", mmap_mode="r"),
            inputs,
        )


@dataclass
class WorldField:
    """The world Field: trains on ``world.regions``; forecasts any box of any region."""

    config_name: str = "world"
    log: Log = _quiet
    _states: dict[int, list[dict[str, torch.Tensor]]] = field(default_factory=dict)

    @property
    def config(self) -> FieldConfig:
        cfg = load_field_config(self.config_name)
        if cfg.world is None:
            raise ValueError(f"field config {self.config_name!r} has no world section")
        return cfg

    # ------------------------------------------------------------------ tiles and the pool

    def tiles(self, root: Path) -> list[Tile]:
        cfg = self.config
        assert cfg.world is not None
        out = []
        for name, terms in cfg.world.regions.items():
            region = load_region(name, root)
            mc = terms.mc if terms.mc is not None else region.completeness
            if mc is None:
                raise ValueError(f"region {name} states no completeness and the config none")
            fit_from = terms.fit_from or (
                to_iso(max(region.t_start, to_seconds(cfg.world.cut) - FIT_YEARS * 365.25 * DAY))
                or region.start
            )
            b = region.box
            for g in tile_grids(
                (b.lat_min, b.lat_max, b.lon_min, b.lon_max),
                mc,
                cfg.world.tile_lat,
                cfg.world.tile_lon,
            ):
                out.append(Tile(name, g, mc, fit_from))
        return out

    def pool_id(self) -> str:
        """Named by what decides the pool's data (never the model's shape or the members)."""
        cfg = self.config
        assert cfg.world is not None
        body = {
            "version": WORLD_VERSION,
            "world": cfg.world.model_dump(mode="json", exclude={"steps_per_epoch"}),
            "memory_start": cfg.memory.start,
            "first_day": cfg.train.first_day,
            "validation_every": cfg.train.validation_every,
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

    def pool_dir(self, root: Path) -> Path:
        return root / "data" / "world" / f"pool-{self.config_name}-{self.pool_id()}"

    def build_pool(self, root: Path) -> list[PoolTile]:
        """Every tile's pool of days, built once and cached (``data/world/``)."""
        from craton.bench.world import RegionView

        cfg = self.config
        assert cfg.world is not None
        w = cfg.world
        out: list[PoolTile] = []
        cut = to_seconds(w.cut)
        starts_all = np.arange(to_seconds(cfg.train.first_day), cut - DAY + 1, DAY)
        block = (np.arange(len(starts_all)) // VALIDATION_BLOCK_DAYS) % cfg.train.validation_every
        is_val = block == cfg.train.validation_every - 1
        for tile in self.tiles(root):
            folder = self.pool_dir(root) / tile.key
            meta_path = folder / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text("utf-8"))
                if meta.get("skipped"):
                    continue
                out.append(
                    PoolTile(tile, folder, np.asarray(meta["train"]), np.asarray(meta["val"]))
                )
                continue
            t0 = time.perf_counter()
            g = tile.grid
            view = RegionView(
                root=root,
                region=tile.region,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=w.cut,
            )
            hist = view.train_events("world").above(tile.mc)
            n_fit = int((hist.time >= to_seconds(tile.fit_from)).sum())
            folder.mkdir(parents=True, exist_ok=True)
            if len(hist) < w.min_tile_quakes or n_fit < 50:
                meta_path.write_text(
                    json.dumps({"skipped": True, "quakes": len(hist)}) + "\n", "utf-8"
                )
                self.log(f"tile {tile.key}: {len(hist)} quakes at mc {tile.mc:g}, skipped")
                continue
            spec = tile_spec(tile, w.cut)
            fit = Etas().fit(spec, view, 0)
            events = view.train_events("world")
            scored = tile.mc + SCORED_ABOVE_MC
            day_counts = _counts(events, g, starts_all, tile.mc, 1, scored).sum(axis=(1, 2))
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{POOL_SALT}:{tile.key}".encode()).hexdigest()[:8], 16)
            )
            busy = day_counts > 0
            keep = busy | (rng.random(len(starts_all)) < w.quiet_share)
            weights = np.where(busy, 1.0, 1.0 / w.quiet_share)[keep].astype(np.float32)
            starts = starts_all[keep]
            days = build_days(
                fit,
                events,
                starts,
                g,
                cfg,
                True,
                1,
                scored,
                memory_spec(tile.mc, max(cfg.memory.start, load_region(tile.region, root).start)),
            )
            assert days.counts is not None
            np.save(folder / "memory.npy", days.memory)
            np.save(folder / "log_etas.npy", days.log_etas)
            np.save(folder / "counts.npy", days.counts)
            np.save(folder / "weights.npy", weights)
            idx = np.arange(len(starts))
            val = idx[is_val[keep]]
            # A validation block's neighbours are training days; one-day answers can't leak.
            train = idx[~is_val[keep]]
            if len(val) > w.max_validation_days:
                val = np.sort(rng.choice(val, w.max_validation_days, replace=False))
            meta = {
                "tile": tile.key,
                "region": tile.region,
                "mc": tile.mc,
                "cells": g.n_cells,
                "days": len(starts),
                "busy_days": int(busy.sum()),
                "quakes": len(hist),
                "etas": {
                    **fit.params.as_dict(),
                    "b": fit.b,
                    "mc": fit.mc,
                    "fit_from": tile.fit_from,
                },
                "train": train.tolist(),
                "val": val.tolist(),
                "seconds": time.perf_counter() - t0,
            }
            meta_path.write_text(json.dumps(meta) + "\n", "utf-8")
            self.log(
                f"tile {tile.key}: {len(hist):,} quakes, {len(starts):,} days in the pool "
                f"({int(busy.sum()):,} busy), {meta['seconds']:.0f}s"
            )
            out.append(PoolTile(tile, folder, train, val))
        return out

    # ------------------------------------------------------------------ training

    def folder(self, root: Path, seed: int) -> Path:
        cfg = self.config
        return checkpoints_dir(root) / (
            f"world-{cfg.name}-{WORLD_VERSION}-{self.pool_id()}-{field_hash(cfg)[:8]}-s{seed}"
        )

    def states(self, root: Path, seed: int) -> list[dict[str, torch.Tensor]]:
        if seed in self._states:
            return self._states[seed]
        path = self.folder(root, seed) / "model.pt"
        if path.is_file():
            body = torch.load(path, map_location="cpu", weights_only=True)
            self._states[seed] = list(body["members"])
            return self._states[seed]
        pool = self.build_pool(root)
        cfg = self.config
        t0 = time.perf_counter()
        states, runs = [], []
        for m in range(cfg.train.members):
            st, info = train_world_member(pool, cfg, seed * 100 + m, self.log)
            states.append(st)
            runs.append(info)
        folder = self.folder(root, seed)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": seed,
            "config": cfg.model_dump(mode="json"),
            "config_hash": field_hash(cfg),
            "world_version": WORLD_VERSION,
            "pool": self.pool_id(),
            "craton_version": __version__,
            "tiles": [p.tile.key for p in pool],
            "training_days": int(sum(len(p.train) for p in pool)),
            "members": runs,
            "parameters": sum(v.numel() for v in states[0].values()),
            "train_seconds": time.perf_counter() - t0,
        }
        torch.save({"members": states}, folder / "model.pt")
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        (folder / "card.md").write_text(card(meta), "utf-8")
        self._states[seed] = states
        return states

    # ------------------------------------------------------------------ forecasting

    def npp_factor(self, engine: Any, events: Any, seed: int) -> np.ndarray:
        """CB-29: the factor on EarthquakeNPP's ETAS (``craton.field.npp``)."""
        from craton.field.npp import factor

        return factor(self, engine, events, seed)

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        """Daily forecasts of any box: ETAS fitted on the box's past, corrected by the world
        Field's members (it never trained on this box after ``world.cut``)."""
        cfg = self.config
        fit = Etas().fit(spec, ctx, seed)
        states = self.states(ctx.root, seed)
        start = cfg.memory.start
        region_start = getattr(ctx, "region", "")
        if region_start:
            with contextlib.suppress(FileNotFoundError):
                start = max(start, load_region(region_start, ctx.root).start)
        days = build_days(
            fit,
            ctx.events,
            spec.day_starts,
            spec.grid,
            cfg,
            False,
            spec.period_days,
            spec.scored,
            memory_spec(spec.mc, start),
        )
        m3 = predict_world(states, days.memory, days.log_etas[..., 1], spec.grid, cfg)
        shares = magnitude_shares(fit, spec.grid, spec.max_magnitude)
        first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
        return GriddedForecast(m3.mean(axis=0) / shares[first:].sum(), shares)


def _net(n_memory: int, grid: csep.Grid, cfg: FieldConfig, dev: torch.device) -> FieldNet:
    return FieldNet(n_memory, grid, cfg.model).to(dev)


def _without_place(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weights only: the place code belongs to the box being read, never to the model."""
    return {k: v for k, v in state.items() if k != "place"}


def train_world_member(
    pool: list[PoolTile],
    cfg: FieldConfig,
    member_seed: int,
    log: Log = _quiet,
    init: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """One world Field trained on every tile's pool; weighted Poisson loss, early stopping on
    every tile's validation days. ``init`` starts from trained weights (fine-tuning)."""
    from craton.field.model import place_code

    assert cfg.world is not None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg.train
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    data = [p.arrays() for p in pool]
    n_feat = data[0][0].shape[-1]
    places = [place_code(p.tile.grid).to(dev) for p in pool]
    net = _net(n_feat, pool[0].tile.grid, cfg, dev)
    if init is not None:
        net.load_state_dict({**init, "place": net.place})
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    share = np.array([len(p.train) for p in pool], dtype=np.float64)
    share /= share.sum()

    def nll(k: int, sel: IntArray) -> tuple[torch.Tensor, torch.Tensor]:
        mem, log_e, y, w, log_in = data[k]
        sel = np.sort(sel)
        net.place = places[k]
        lb = torch.from_numpy(np.asarray(log_e[sel])).to(dev)
        li = lb if log_in is log_e else torch.from_numpy(np.asarray(log_in[sel])).to(dev)
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            corr = net(torch.from_numpy(np.asarray(mem[sel])).to(dev).float(), li[..., 1])
        lam = lb + corr
        yy = torch.from_numpy(np.asarray(y[sel])).to(dev)
        ww = torch.from_numpy(np.asarray(w[sel])).to(dev)[:, None, None]
        return -(ww * (yy * lam - lam.exp())).sum(), (ww * yy).sum()

    def validate() -> float:
        net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for k, p in enumerate(pool):
                for i in range(0, len(p.val), INFER_DAYS):
                    loss, count = nll(k, p.val[i : i + INFER_DAYS])
                    tot, n = tot + float(loss), n + float(count)
        net.train()
        return tot / max(n, 1.0)

    best = validate()
    start_loss = best
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad = epochs = 0
    t0 = time.perf_counter()
    for epoch in range(tc.max_epochs):
        epochs = epoch + 1
        for _ in range(cfg.world.steps_per_epoch):
            k = int(rng.choice(len(pool), p=share))
            sel = rng.choice(pool[k].train, tc.days_per_step, replace=False)
            loss, _ = nll(k, sel)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss / (tc.days_per_step * 1000.0)).backward()  # type: ignore[no-untyped-call]
            scaler.step(opt)
            scaler.update()
        v = validate()
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        log(
            f"  world member {member_seed} epoch {epochs} validation {v:.4f} (best {best:.4f}) "
            f"{(time.perf_counter() - t0) / 60:.1f} min"
        )
        if bad >= tc.patience:
            break
    return {k: v.cpu() for k, v in _without_place(state).items()}, {
        "epochs": float(epochs),
        "validation_start": start_loss,
        "validation_best": best,
        "minutes": (time.perf_counter() - t0) / 60,
    }


def predict_world(
    states: list[dict[str, torch.Tensor]],
    memory: np.ndarray,
    log_etas_scored: np.ndarray,
    grid: csep.Grid,
    cfg: FieldConfig,
    corrections: bool = False,
) -> FloatArray:
    """(members, days, cells) each member's rate of scored quakes on ``grid``; or, with
    ``corrections``, (members, days, cells, 2) its log corrections of both bins."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (len(states), memory.shape[0], grid.n_cells) + ((2,) if corrections else ())
    out = np.zeros(shape)
    mem = torch.from_numpy(np.asarray(memory))
    log_e = torch.from_numpy(np.asarray(log_etas_scored))
    for k, st in enumerate(states):
        net = _net(memory.shape[-1], grid, cfg, dev)
        net.load_state_dict({**st, "place": net.place})
        net.eval()
        with torch.no_grad():
            for i in range(0, memory.shape[0], INFER_DAYS):
                lb = log_e[i : i + INFER_DAYS].to(dev)
                corr = net(mem[i : i + INFER_DAYS].to(dev).float(), lb)
                if corrections:
                    out[k, i : i + INFER_DAYS] = corr.double().cpu().numpy()
                else:
                    out[k, i : i + INFER_DAYS] = (lb + corr[..., 1]).exp().double().cpu().numpy()
    return out


def card(meta: dict[str, Any]) -> str:
    """The model card of one world Field checkpoint."""
    runs = meta["members"]
    gain = ", ".join(f"{r['validation_start'] - r['validation_best']:.4f}" for r in runs)
    regions = sorted({t.split("-")[0] for t in meta["tiles"]})
    w = meta["config"]["world"]
    return (
        f"# World Field · seed {meta['seed']}\n\n"
        "- **What it does:** daily forecasts of any 0.1° box on Earth, starting from ETAS "
        "fitted there and correcting it cell by cell, at most "
        f"{meta['config']['model']['max_nudge']:g}x either way.\n"
        f"- **Learned from:** {len(meta['tiles'])} tiles of {', '.join(regions)}, "
        f"{meta['training_days']:,} tile-days from {meta['config']['train']['first_day'][:10]} "
        f"to {w['cut'][:10]}; quiet days sampled at {w['quiet_share']:g} and weighted back.\n"
        f"- **Magnitudes:** each region's own completeness mc; memory from mc − "
        f"{MEMORY_BELOW_MC:g}, scored from mc + {SCORED_ABOVE_MC:g}.\n"
        f"- **Size:** {meta['parameters']:,} parameters per member, {len(runs)} members.\n"
        f"- **Training:** validation gain over ETAS (nats per quake) {gain}; "
        f"{meta['train_seconds'] / 60:.1f} min.\n"
        f"- **Code:** Craton {meta['craton_version']}, world {meta['world_version']}, pool "
        f"`{meta['pool']}`, config hash `{meta['config_hash']}`.\n"
    )
