"""Field 2's heart on CB-29 (EarthquakeNPP), zero-shot: no fine-tuning on their data.

CB-29 scores a factor e^{c(day, cell)} on *their* ETAS. The heart learned to bend *our* ETAS,
so here it does exactly that and hands over the ratio:

1. their grid is cut into the world tiles; in each tile our ETAS is fitted on their catalogue
   (M2.5+, 1971 as burn-in, targets 1981-2007: their training window), seed 0;
2. every quake from 1971 on is a parent, as in their likelihood;
3. the factor of a cell on a test day is the heart's rate over the same box's rate with every
   weight at one (our ETAS): how far the heart would move ETAS there. Off the tiles, 1.

An untrained heart gives 1 everywhere, which *is* their ETAS: born equal on CB-29 too. The
heart's weights were learned on the world pool (California among it, to the same 2007 cut), so
nothing after 2007 is read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from craton.bench import etas
from craton.bench.data import DAY, Events
from craton.field.npp import _tiles
from craton.field2.heart import Heart, make_box, to_device
from craton.field2.world import (
    FIELD2_VERSION,
    Field2World,
    _device,
    _load_box,
    _save_box,
    free_box,
    predict,
)
from craton.spec.timeaxis import to_seconds

if TYPE_CHECKING:
    from craton.bench.npp import Engine


def box_dir(world: Field2World, eng: Engine) -> Path:
    return (
        eng.folder.parent
        / f"cb29-field2-{FIELD2_VERSION}-{world.prep_dir(eng.folder.parents[2]).name[-8:]}"
    )


def boxes(world: Field2World, eng: Engine, events: Events) -> list[tuple[Any, np.ndarray, Any]]:
    """(box, global cells, grid) of every CB-29 tile with enough quakes, built once."""
    from craton.bench.npp import AUX_START, MC, TEST_END, TEST_START, TRAIN_START

    wcfg = world.world.config
    assert wcfg.world is not None
    folder = box_dir(world, eng)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    t_aux = to_seconds(AUX_START)
    for tile, cells in _tiles(world.world):
        path = folder / f"{tile.key}.npz"
        skip = folder / f"{tile.key}.skipped.json"
        if skip.is_file():
            continue
        if not path.is_file():
            t0 = time.perf_counter()
            g = tile.grid
            ev = events.select(g.cell_of(events.lat, events.lon) >= 0).above(MC)
            fit_ev = ev.select(ev.time < to_seconds(TEST_START))
            n_target = int((fit_ev.time >= to_seconds(TRAIN_START)).sum())
            if n_target < wcfg.world.min_tile_quakes:
                skip.write_text(json.dumps({"quakes": n_target}) + "\n", "utf-8")
                continue
            proj = etas.Projection.for_grid(g)
            cat = etas.to_catalog(fit_ev.time, fit_ev.lat, fit_ev.lon, fit_ev.mag, proj, t_aux)
            window = etas.FitWindow(
                (to_seconds(TRAIN_START) - t_aux) / DAY, (to_seconds(TEST_START) - t_aux) / DAY
            )
            fit = etas.fit(cat, MC, window, etas.box_km(g, proj), t_aux, seed=0)
            hist = ev.window(t_aux, to_seconds(TEST_END))
            hcat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, t_aux)
            _save_box(path, make_box(fit, hcat, g, proj, world.config.bins), {})
            world.log(
                f"CB-29 field2 tile {tile.key}: {len(hcat):,} parents, "
                f"{time.perf_counter() - t0:.0f}s"
            )
        box, _ = _load_box(path)
        out.append((box, cells, tile.grid))
    return out


def factor(world: Field2World, eng: Engine, events: Events, seed: int) -> np.ndarray:
    """(test days, cells) the heart's rate over our ETAS's, per CB-29 cell (1 off the tiles)."""
    from craton.bench.npp import AUX_START, GRID, MC, TEST_END

    cfg = world.config
    d0 = eng.test_day0
    starts = eng.day_starts()[d0:]
    t_aux = to_seconds(AUX_START)
    days = (starts - t_aux) / DAY
    out = np.ones((len(starts), GRID.n_cells), np.float64)
    root = eng.folder.parents[2]
    dev = _device()
    born = Heart(cfg.bins, cfg.net).to(dev).eval()
    for box, cells, grid in boxes(world, eng, events):
        fb = None
        if cfg.net.free_head:
            inside = events.select(grid.cell_of(events.lat, events.lon) >= 0)
            fb = free_box(inside, MC, t_aux, to_seconds(TEST_END), grid, cfg)
        heart = world.box_rates(box, days, seed, root, fb)
        with torch.no_grad():
            base = predict(born, to_device(box, cfg.bins, dev), None, days, cfg)
        out[:, cells] = heart / np.maximum(base, 1e-30)
    return out
