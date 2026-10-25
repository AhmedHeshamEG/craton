"""The world Field on CB-29 (EarthquakeNPP): fine-tuned on their training window, their ETAS
as the starting line.

The world Field learned to correct *our* ETAS, one day ahead, from catalogues complete half a
magnitude below mc. EarthquakeNPP is a different game: their ETAS, a continuous-time score,
and a catalogue cut at mc. So the world Field is the starting point and each member is
fine-tuned on CB-29's own training days (1981-2007), the way a pretrained model is
fine-tuned on a downstream task:

- the Field reads, at each midnight, its memory of the quakes before it and their ETAS's
  expected count in each cell from the parents before midnight (``Engine.fore``);
- its correction multiplies their ETAS's realized intensity that day (``Engine.real``: the
  day's own quakes trigger too, as in their likelihood), and the loss is the Poisson
  likelihood of each cell's count under it: the point-process likelihood of the factor;
- the factor of a cell is the members' average of (1 − s)·e^{c₀} + s·e^{c₁}, s being the
  Gutenberg-Richter share of quakes a magnitude above mc, so every magnitude moves together
  as the likelihood (which ignores magnitudes) needs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from craton.bench.data import Events
from craton.field.config import field_hash
from craton.field.forecaster import VALIDATION_BLOCK_DAYS, _counts
from craton.field.memory import memory
from craton.field.world import (
    POOL_SALT,
    SCORED_ABOVE_MC,
    WORLD_VERSION,
    PoolTile,
    Tile,
    WorldField,
    memory_spec,
    predict_world,
    tile_grids,
    train_world_member,
)
from craton.spec.timeaxis import to_seconds
from craton.train.checkpoint import checkpoints_dir

if TYPE_CHECKING:
    from craton.bench.npp import Engine

NPP_VERSION = "p0.1"


def share_above() -> float:
    """Gutenberg-Richter share of their quakes a magnitude or more above mc (their β)."""
    from craton.bench.npp import BETA

    return float(math.exp(-BETA * SCORED_ABOVE_MC))


def _tiles(world: WorldField) -> list[tuple[Tile, np.ndarray]]:
    """CB-29's grid in the world Field's tiles, each with its cells' global indices."""
    from craton.bench.npp import AUX_START, GRID, MC

    cfg = world.config
    assert cfg.world is not None
    out = []
    for g in tile_grids(
        (GRID.lat_min, GRID.lat_max, GRID.lon_min, GRID.lon_max),
        MC,
        cfg.world.tile_lat,
        cfg.world.tile_lon,
    ):
        i0 = round((g.lat_min - GRID.lat_min) / GRID.cell_deg)
        j0 = round((g.lon_min - GRID.lon_min) / GRID.cell_deg)
        rows = i0 + np.arange(g.n_lat)
        cols = j0 + np.arange(g.n_lon)
        cells = (rows[:, None] * GRID.n_lon + cols[None, :]).ravel()
        out.append((Tile("cb29", g, MC, AUX_START), cells))
    return out


def _log_bins(mass: np.ndarray, s: float) -> np.ndarray:
    m = np.maximum(np.asarray(mass, np.float64), 1e-30)
    return np.stack([np.log(m * (1 - s)), np.log(m * s)], axis=-1).astype(np.float32)


def pool_dir(world: WorldField, eng: Engine) -> Path:
    cfg = world.config
    assert cfg.world is not None
    body = {"v": NPP_VERSION, "engine": eng.folder.name, "world": cfg.world.model_dump(mode="json")}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return eng.folder.parent / f"cb29-pool-{h}"


def build_pool(world: WorldField, eng: Engine, events: Events) -> list[PoolTile]:
    from craton.bench.npp import AUX_START, MC

    cfg = world.config
    assert cfg.world is not None
    w = cfg.world
    s = share_above()
    root = pool_dir(world, eng)
    n_train = eng.test_day0
    starts_all = eng.day_starts()[:n_train]
    block = (np.arange(n_train) // VALIDATION_BLOCK_DAYS) % cfg.train.validation_every
    is_val = block == cfg.train.validation_every - 1
    arrays: dict[str, Any] = {}
    out = []
    for tile, cells in _tiles(world):
        folder = root / tile.key
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text("utf-8"))
            if not meta.get("skipped"):
                out.append(
                    PoolTile(tile, folder, np.asarray(meta["train"]), np.asarray(meta["val"]))
                )
            continue
        g = tile.grid
        train_ev = events.select(events.time < to_seconds(eng_test_start()))
        in_tile = g.cell_of(train_ev.lat, train_ev.lon) >= 0
        folder.mkdir(parents=True, exist_ok=True)
        if int(in_tile.sum()) < w.min_tile_quakes:
            meta_path.write_text(json.dumps({"skipped": True, "quakes": int(in_tile.sum())}) + "\n")
            continue
        if not arrays:
            a = eng.arrays()
            arrays["real"] = np.asarray(a["real"][:n_train])
            arrays["fore"] = np.asarray(a["fore"][:n_train])
        counts_all = _counts(train_ev, g, starts_all, MC, 1, MC + SCORED_ABOVE_MC)
        busy = counts_all.sum(axis=(1, 2)) > 0
        rng = np.random.default_rng(
            int(hashlib.sha256(f"{POOL_SALT}:cb29:{tile.key}".encode()).hexdigest()[:8], 16)
        )
        keep = busy | (rng.random(n_train) < w.quiet_share)
        weights = np.where(busy, 1.0, 1.0 / w.quiet_share)[keep].astype(np.float32)
        days = np.nonzero(keep)[0]
        mem = memory(train_ev, g, starts_all[days], memory_spec(MC, AUX_START))
        np.save(folder / "memory.npy", mem)
        np.save(folder / "log_etas.npy", _log_bins(arrays["real"][days][:, cells], s))
        np.save(folder / "inputs.npy", _log_bins(arrays["fore"][days][:, cells], s))
        np.save(folder / "counts.npy", counts_all[days])
        np.save(folder / "weights.npy", weights)
        idx = np.arange(len(days))
        val, train = idx[is_val[days]], idx[~is_val[days]]
        if len(val) > w.max_validation_days:
            val = np.sort(rng.choice(val, w.max_validation_days, replace=False))
        meta = {
            "tile": tile.key,
            "days": len(days),
            "busy": int(busy.sum()),
            "train": train.tolist(),
            "val": val.tolist(),
        }
        meta_path.write_text(json.dumps(meta) + "\n", "utf-8")
        world.log(f"CB-29 tile {tile.key}: {int(in_tile.sum()):,} quakes, {len(days):,} pool days")
        out.append(PoolTile(tile, folder, train, val))
    return out


def eng_test_start() -> str:
    from craton.bench.npp import TEST_START

    return TEST_START


def checkpoint(world: WorldField, eng: Engine, seed: int) -> Path:
    cfg = world.config
    return checkpoints_dir(eng.folder.parents[2]) / (
        f"cb29-{cfg.name}-{NPP_VERSION}-{WORLD_VERSION}-{pool_dir(world, eng).name[-12:]}-"
        f"{field_hash(cfg)[:8]}-s{seed}"
    )


def states(
    world: WorldField, eng: Engine, events: Events, seed: int
) -> list[dict[str, torch.Tensor]]:
    folder = checkpoint(world, eng, seed)
    if (folder / "model.pt").is_file():
        return list(
            torch.load(folder / "model.pt", map_location="cpu", weights_only=True)["members"]
        )
    root = eng.folder.parents[2]
    base = world.states(root, seed)
    pool = build_pool(world, eng, events)
    cfg = world.config
    t0 = time.perf_counter()
    out, runs = [], []
    for m, init in enumerate(base):
        st, info = train_world_member(pool, cfg, seed * 100 + 50 + m, world.log, init=init)
        out.append(st)
        runs.append(info)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"members": out}, folder / "model.pt")
    meta = {
        "seed": seed,
        "fine_tuned_from": world.folder(root, seed).name,
        "tiles": [p.tile.key for p in pool],
        "members": runs,
        "train_seconds": time.perf_counter() - t0,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    return out


def factor(world: WorldField, eng: Engine, events: Events, seed: int) -> np.ndarray:
    """(test days, cells) e^c: the members' average factor on their ETAS (1 off the tiles)."""
    from craton.bench.npp import AUX_START, GRID, MC

    sts = states(world, eng, events, seed)
    cfg = world.config
    s = share_above()
    d0 = eng.test_day0
    starts = eng.day_starts()[d0:]
    fore = eng.arrays()["fore"]
    out = np.ones((len(starts), GRID.n_cells), np.float64)
    for tile, cells in _tiles(world):
        if not (pool_dir(world, eng) / tile.key / "memory.npy").is_file():
            continue
        g = tile.grid
        mem = memory(events, g, starts, memory_spec(MC, AUX_START))
        inp = _log_bins(np.asarray(fore[d0:])[:, cells], s)
        corr = predict_world(sts, mem, inp[..., 1], g, cfg, corrections=True)
        f = (1 - s) * np.exp(corr[..., 0]) + s * np.exp(corr[..., 1])
        out[:, cells] = f.mean(axis=0)
    return out
