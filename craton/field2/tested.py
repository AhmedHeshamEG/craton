"""Train as tested (P11): every pool day forecast by an ETAS that never saw it.

The first heart learned on days its tiles' ETAS had been fitted on. There ETAS's totals are
right by construction (a maximum-likelihood fit matches the count it was fitted on), so the
one thing the test asks for, how much ETAS under-counts in a period it has not read, was
invisible in training. On CB-32 ETAS is fitted on a box's past and asked about the year after.

So each tile's pool days are cut into blocks of ``origin_years`` (from the tile's first
possible origin to the world cut). A block is one *origin*: ETAS fitted on the tile's quakes
from its ``fit_from`` up to the block's first day, then run over the block day by day with
every quake known before each day as a parent, exactly as the bench runs it. Each origin is
its own ``Box``; a pool day belongs to one origin only.

The place state (``craton.field2.state``) hears quakes from mc − 0.5 in the tile, stored once
per tile (``StateQuakes``), in seconds, for every origin to read up to its own days.

Cached in ``<prep_dir>/tested-<years>y/``: one ``<tile>@<origin>.npz`` per origin (the box,
its days since its ``t_ref`` and their indices in the tile's pool) and one
``<tile>.quakes.npz`` per tile.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from craton.bench import etas
from craton.bench.csep import Grid
from craton.bench.data import DAY, Events
from craton.field2.heart import Box, make_box
from craton.spec.timeaxis import to_iso, to_seconds

if TYPE_CHECKING:
    from craton.field.world import PoolTile
    from craton.field2.world import Field2World

YEAR = 365.25 * DAY
#: An origin needs at least this much catalogue behind it (ETAS also burns in 90 days).
MIN_FIT_YEARS = 1.0
#: An origin counts only if its ETAS was fitted on this many quakes: CB-22's rule for its
#: reference (``craton.bench.world.MIN_FIT_QUAKES``). Fitted on fewer, ETAS often comes out
#: supercritical (branching ratio near one) and over-forecasts a block twenty-fold, a miss
#: the bench never asks about.
MIN_FIT_QUAKES = 200
#: A block shorter than this (the sliver before the cut) is not worth a fit.
MIN_BLOCK_DAYS = 30
#: The state hears quakes down to this far below the region's completeness (the Field's
#: memory does too).
STATE_BELOW_MC = 0.5


@dataclass(frozen=True)
class StateQuakes:
    """Quakes a place state hears, inside one box: absolute seconds, cell, m − mc."""

    time: np.ndarray  # (Q,) float64 seconds
    cell: np.ndarray  # (Q,) int64
    m_rel: np.ndarray  # (Q,) float32

    def __len__(self) -> int:
        return len(self.time)


def state_quakes(events: Events, grid: Grid, mc: float, t0: float, t1: float) -> StateQuakes:
    """Every quake in ``grid`` from mc − 0.5, in [t0, t1) seconds."""
    ev = events.window(t0, t1).above(mc - STATE_BELOW_MC)
    cell = grid.cell_of(ev.lat, ev.lon)
    ok = cell >= 0
    return StateQuakes(
        np.asarray(ev.time[ok], np.float64),
        np.asarray(cell[ok], np.int64),
        np.asarray(ev.mag[ok] - mc, np.float32),
    )


@dataclass
class OriginTile:
    """One origin of one pool tile: an ETAS fitted before it, the days after it."""

    key: str  # "<tile>@<origin date>"
    tile: str
    region: str
    box: Box
    t_ref: float  # seconds: the box's day 0
    days: np.ndarray  # (D,) days since t_ref
    pool_idx: np.ndarray  # (D,) indices into the tile's pool arrays
    quakes: StateQuakes
    counts: np.ndarray  # (D, cells, 2) [mc, mc+1) and >= mc+1
    weights: np.ndarray  # (D,)
    is_val: np.ndarray  # (D,) bool: the pool's validation blocks
    grid: Grid


def origins_of(fit_from: float, first_day: float, cut: float, years: float) -> list[float]:
    """Block starts (seconds): from the first day an origin has a year of catalogue behind
    it, every ``years`` to the cut."""
    o = max(first_day, fit_from + MIN_FIT_YEARS * YEAR)
    o = float(np.floor(o / DAY) * DAY)
    out = []
    while o < cut - DAY:
        out.append(o)
        o = float(np.floor((o + years * YEAR) / DAY) * DAY)
    return out


def _save(path: Path, ot: OriginTile) -> None:
    from craton.field2.world import _save_box

    _save_box(
        path,
        ot.box,
        {
            "days": ot.days,
            "pool_idx": ot.pool_idx,
            "t_ref_s": np.array([ot.t_ref]),
        },
    )


def tested_dir(world: Field2World, root: Path, years: float) -> Path:
    return world.prep_dir(root) / f"tested-{years:g}y"


def build_tile(
    world: Field2World, root: Path, p: PoolTile, years: float, log: Any
) -> list[OriginTile]:
    """Every origin of one pool tile, built once and cached."""
    from craton.bench.world import RegionView
    from craton.field.world import pool_keep
    from craton.field2.world import _load_box

    cfg = world.config
    wcfg = world.world.config
    assert wcfg.world is not None
    w = wcfg.world
    folder = tested_dir(world, root, years)
    folder.mkdir(parents=True, exist_ok=True)
    g = p.tile.grid
    cut = to_seconds(w.cut)
    first = to_seconds(wcfg.train.first_day)
    starts_all = np.arange(first, cut - DAY + 1, DAY)
    origins = origins_of(to_seconds(p.tile.fit_from), first, cut, years)
    qpath = folder / f"{p.tile.key}.quakes.npz"
    events: Events | None = None

    def load_events() -> Events:
        nonlocal events
        if events is None:
            view = RegionView(
                root=root,
                region=p.tile.region,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=w.cut,
            )
            events = view.train_events("world")
        return events

    if not qpath.is_file():
        q = state_quakes(load_events(), g, p.tile.mc, to_seconds(p.tile.fit_from), cut)
        tmp = qpath.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, time=q.time, cell=q.cell, m_rel=q.m_rel)
        tmp.replace(qpath)
    z = np.load(qpath)
    quakes = StateQuakes(z["time"], z["cell"].astype(np.int64), z["m_rel"])

    meta = json.loads((p.folder / "meta.json").read_text("utf-8"))
    keep_path = folder / f"{p.tile.key}.days.npy"
    if not keep_path.is_file():
        keep, _, _ = pool_keep(p.tile, load_events(), starts_all, w.quiet_share)
        if int(keep.sum()) != meta["days"]:
            raise RuntimeError(f"tile {p.tile.key}: pool days do not rebuild")
        np.save(keep_path, starts_all[keep])
    starts = np.load(keep_path)
    val_mask = np.zeros(len(starts), bool)
    val_mask[np.asarray(meta["val"], np.int64)] = True
    _, _, counts, weights, _ = p.arrays()

    out: list[OriginTile] = []
    for k, o in enumerate(origins):
        end = origins[k + 1] if k + 1 < len(origins) else cut
        idx = np.nonzero((starts >= o) & (starts < end))[0]
        if len(idx) == 0 or end - o < MIN_BLOCK_DAYS * DAY:
            continue
        name = f"{p.tile.key}@{(to_iso(o) or '')[:10]}"
        path = folder / f"{name}.npz"
        skip = folder / f"{name}.skipped.json"
        if skip.is_file():
            continue
        if not path.is_file():
            t0 = time.perf_counter()
            ev = load_events()
            t_ref = to_seconds(p.tile.fit_from)
            proj = etas.Projection.for_grid(g)
            fit_ev = ev.window(t_ref, o).above(p.tile.mc)
            if len(fit_ev) < MIN_FIT_QUAKES:
                skip.write_text(json.dumps({"why": f"{len(fit_ev)} quakes"}) + "\n", "utf-8")
                continue
            cat = etas.to_catalog(fit_ev.time, fit_ev.lat, fit_ev.lon, fit_ev.mag, proj, t_ref)
            window = etas.FitWindow(etas.BURN_IN_DAYS, (o - t_ref) / DAY)
            try:
                fit = etas.fit(cat, p.tile.mc, window, etas.box_km(g, proj), t_ref, seed=0)
            except ValueError as err:
                skip.write_text(json.dumps({"why": str(err)}) + "\n", "utf-8")
                log(f"origin {name}: skipped ({err})")
                continue
            hist = ev.window(t_ref, end).above(p.tile.mc)
            hcat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, t_ref)
            box = make_box(fit, hcat, g, proj, cfg.bins)
            ot = OriginTile(
                name,
                p.tile.key,
                p.tile.region,
                box,
                t_ref,
                (starts[idx] - t_ref) / DAY,
                idx,
                quakes,
                np.zeros(0),
                np.zeros(0),
                np.zeros(0, bool),
                g,
            )
            _save(path, ot)
            log(
                f"origin {name}: ETAS on {fit.n_targets:,} quakes, {len(box.t):,} parents, "
                f"{len(idx):,} days, {time.perf_counter() - t0:.0f}s"
            )
        box, extra = _load_box(path)
        t_ref_o = float(extra["t_ref_s"][0])
        n_fit = int(((box.t >= etas.BURN_IN_DAYS) & (box.t < (o - t_ref_o) / DAY)).sum())
        if n_fit < MIN_FIT_QUAKES:
            continue
        pidx = extra["pool_idx"].astype(np.int64)
        out.append(
            OriginTile(
                name,
                p.tile.key,
                p.tile.region,
                box,
                float(extra["t_ref_s"][0]),
                extra["days"],
                pidx,
                quakes,
                np.asarray(counts[pidx], np.float32),
                np.asarray(weights[pidx], np.float32),
                val_mask[pidx],
                g,
            )
        )
    return out


def build_all(world: Field2World, root: Path, years: float, log: Any) -> list[OriginTile]:
    """Every origin of every pool tile (all regions; folds choose among them)."""
    out: list[OriginTile] = []
    for p in world.world.training_pool(root):
        out += build_tile(world, root, p, years, log)
    return out
