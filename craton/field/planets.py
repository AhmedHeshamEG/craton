"""Fake planets as the first stage of the curriculum (P8f): planets → regions → central Italy.

An LLM learns grammar from a lot of text before it learns a task from a little. The Field's
equivalent of "a lot" is simulated: the physics planet (``craton.sim.rate_state``) makes as
many 33-year catalogues of central Italy's box as anyone wants, each with its faults, stress
shadows and fluid sources. Real data is the scarce thing; fake data is free.

The planet Field is a world Field (``craton.field.world``) whose tiles are planets instead of
boxes of real regions:

1. Planets ``first_seed`` … ``first_seed + count − 1`` of the configured kind, simulated once
   and cached in ``data/world/planets/``. Every seed is 1000 or more, so no planet CB-09 scores
   (its first qualifying seeds under 200) is ever trained on.
2. Each planet is read exactly as central Italy is: completeness M2, memory from M1.5, scored
   from M3, ETAS fitted from 2005 on the planet's own past, training days from 1991.
3. **Blind twins.** Every planet also comes through a network that goes blind after a big
   quake (Helmstetter et al. 2006: after a magnitude M, quakes below M − 4.5 − 0.75·log10 t,
   t in days, go unrecorded). The twin's memory and its ETAS read the blind catalogue; its
   answers are the full one. The Field then sees both a complete and a blind past with the
   same future, so a catalogue's blindness is something it can read in its input instead of a
   constant it learns once (D-096: the Field on CB-20 multiplied a filled-in catalogue's ETAS
   as if it were still INGV's blind one).

A planet's answers come from the simulator, so a planet Field trains on the whole calendar to
2018: it's another planet, nothing about the real Earth is in it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from craton.bench import csep
from craton.bench.baselines import Etas
from craton.bench.data import DAY, Events
from craton.bench.forcing import Pushes
from craton.bench.planets import KINDS, PlanetContext, events_table, read_events, simulate_planet
from craton.bench.tasks import _write_table
from craton.field.config import FieldConfig, load_field_config
from craton.field.forecaster import VALIDATION_BLOCK_DAYS, _counts, build_days
from craton.field.world import (
    POOL_SALT,
    SCORED_ABOVE_MC,
    WORLD_VERSION,
    Log,
    PoolTile,
    Tile,
    WorldField,
    _quiet,
    memory_spec,
    tile_spec,
)
from craton.paths import data_dir
from craton.spec.timeaxis import to_seconds

#: Central Italy's terms, which every planet copies.
PLANET_MC = 2.0
PLANET_FIT_FROM = "2005-01-01T00:00:00Z"
#: Helmstetter, Kagan & Jackson (2006), California: mc(t) = M − 4.5 − 0.75 log10(t / day).
BLIND_OFFSET = 4.5
BLIND_SLOPE = 0.75
#: Quakes this big or bigger blind the network; a smaller one hides nothing above M1.5.
BLINDING_MAGNITUDE = 4.0
#: After this long nothing above M1.5 is hidden by even an M7 (7 − 4.5 − 0.75·log10 t < 1.5).
BLIND_DAYS = 25.0
#: Pumped planets: a push per 30 days; 30-day starts every few days from 1991 to the cut.
PUSH_PERIOD_DAYS = 30
PUMPED_FIRST_DAY = "1991-01-01T00:00:00Z"
PUMPED_CUT = "2018-01-01T00:00:00Z"
PUMPED_STRIDE_DAYS = 6


def blind(ev: Events) -> Events:
    """The catalogue a network that goes blind after big quakes would have recorded."""
    t_days = ev.time / DAY
    hidden = np.zeros(len(ev), bool)
    for i in np.flatnonzero(ev.mag >= BLINDING_MAGNITUDE - 1e-9):
        lo = np.searchsorted(t_days, t_days[i], side="right")
        hi = np.searchsorted(t_days, t_days[i] + BLIND_DAYS, side="right")
        dt = np.maximum(t_days[lo:hi] - t_days[i], 1e-6)
        mc = ev.mag[i] - BLIND_OFFSET - BLIND_SLOPE * np.log10(dt)
        hidden[lo:hi] |= ev.mag[lo:hi] < mc - 1e-9
    return ev.select(~hidden)


def planet_grid() -> csep.Grid:
    """CB-02's box read at central Italy's completeness, as the world Field reads tiles."""
    from craton.bench.tasks import forecast_grid

    g = forecast_grid()
    return csep.Grid(g.lat_min, g.lat_max, g.lon_min, g.lon_max, 0.1, PLANET_MC, 9.5)


def planet_events(root: Path, kind: str, seed: int, log: Log = _quiet) -> Events:
    """Planet ``seed`` of ``kind``, simulated once and cached in ``data/world/planets/``."""
    path = data_dir(root) / "world" / "planets" / f"{kind}-{seed}.parquet"
    if not path.is_file():
        t0 = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_table(path, events_table(simulate_planet(kind, seed)))
        log(f"planet {kind} {seed} simulated in {time.perf_counter() - t0:.0f}s")
    return read_events(path)


@dataclass
class PlanetView(PlanetContext):
    """A planet with a cut-off: ``train_events`` is its past before ``cut``."""

    cut: str = ""

    def train_events(self, split: str) -> Events:
        return self.events.select(self.events.time < to_seconds(self.cut))


@dataclass
class PlanetField(WorldField):
    """A world Field whose tiles are fake planets (``planets`` in its config)."""

    config_name: str = "planets"
    _events: dict[int, Events] = field(default_factory=dict)

    @property
    def config(self) -> FieldConfig:
        cfg = load_field_config(self.config_name)
        if cfg.planets is None or cfg.world is None:
            raise ValueError(f"field config {self.config_name!r} needs planets and world sections")
        return cfg

    def tiles(self, root: Path) -> list[Tile]:
        p = self.config.planets
        assert p is not None
        out = []
        for seed in range(p.first_seed, p.first_seed + p.count):
            out.append(Tile(f"planet_{p.kind}_{seed}", planet_grid(), PLANET_MC, PLANET_FIT_FROM))
            if p.blind_twins:
                out.append(
                    Tile(f"planet_{p.kind}_{seed}_blind", planet_grid(), PLANET_MC, PLANET_FIT_FROM)
                )
        return out

    def pool_id(self) -> str:
        cfg = self.config
        assert cfg.world is not None
        assert cfg.planets is not None
        body = {
            "version": WORLD_VERSION,
            "planets": cfg.planets.model_dump(mode="json"),
            "world": cfg.world.model_dump(mode="json", exclude={"steps_per_epoch"}),
            "memory_start": cfg.memory.start,
            "first_day": cfg.train.first_day,
            "validation_every": cfg.train.validation_every,
            "blind": [BLIND_OFFSET, BLIND_SLOPE, BLINDING_MAGNITUDE],
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

    def planet(self, root: Path, seed: int) -> Events:
        """Planet ``seed`` of the configured kind, simulated once and cached."""
        if seed not in self._events:
            p = self.config.planets
            assert p is not None
            self._events[seed] = planet_events(root, p.kind, seed, self.log)
        return self._events[seed]

    def build_pool(self, root: Path) -> list[PoolTile]:
        cfg = self.config
        assert cfg.world is not None
        assert cfg.planets is not None
        w = cfg.world
        out: list[PoolTile] = []
        cut = to_seconds(w.cut)
        starts_all = np.arange(to_seconds(cfg.train.first_day), cut - DAY + 1, DAY)
        block = (np.arange(len(starts_all)) // VALIDATION_BLOCK_DAYS) % cfg.train.validation_every
        is_val = block == cfg.train.validation_every - 1
        scored = PLANET_MC + SCORED_ABOVE_MC
        for tile in self.tiles(root):
            folder = self.pool_dir(root) / tile.key
            meta_path = folder / "meta.json"
            if meta_path.is_file():
                done = json.loads(meta_path.read_text("utf-8"))
                out.append(
                    PoolTile(tile, folder, np.asarray(done["train"]), np.asarray(done["val"]))
                )
                continue
            t0 = time.perf_counter()
            is_blind = tile.region.endswith("_blind")
            seed = int(tile.region.removesuffix("_blind").rsplit("_", 1)[1])
            truth = self.planet(root, seed)
            seen = blind(truth) if is_blind else truth
            view = PlanetView(root=root, planet=seen, planet_id=tile.region, cut=w.cut)
            fit = Etas().fit(tile_spec(tile, w.cut), view, 0)
            g = tile.grid
            day_counts = _counts(truth, g, starts_all, scored, 1, scored).sum(axis=(1, 2))
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{POOL_SALT}:{tile.key}".encode()).hexdigest()[:8], 16)
            )
            busy = day_counts > 0
            keep = busy | (rng.random(len(starts_all)) < w.quiet_share)
            weights = np.where(busy, 1.0, 1.0 / w.quiet_share)[keep].astype(np.float32)
            starts = starts_all[keep]
            mem = memory_spec(PLANET_MC, cfg.memory.start)
            days = build_days(fit, seen, starts, g, cfg, False, 1, scored, mem)
            # The blind twin reads the blind past but answers to the full catalogue.
            counts = _counts(truth, g, starts, PLANET_MC, 1, scored)
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "memory.npy", days.memory)
            np.save(folder / "log_etas.npy", days.log_etas)
            np.save(folder / "counts.npy", counts)
            np.save(folder / "weights.npy", weights)
            idx = np.arange(len(starts))
            val = idx[is_val[keep]]
            train = idx[~is_val[keep]]
            if len(val) > w.max_validation_days:
                val = np.sort(rng.choice(val, w.max_validation_days, replace=False))
            meta: dict[str, Any] = {
                "tile": tile.key,
                "planet": seed,
                "kind": KINDS[cfg.planets.kind][0],
                "blind": is_blind,
                "quakes": len(seen.above(PLANET_MC)),
                "hidden": len(truth.above(PLANET_MC)) - len(seen.above(PLANET_MC)),
                "days": len(starts),
                "busy_days": int(busy.sum()),
                "etas": {**fit.params.as_dict(), "b": fit.b, "mc": fit.mc},
                "train": train.tolist(),
                "val": val.tolist(),
                "seconds": time.perf_counter() - t0,
            }
            meta_path.write_text(json.dumps(meta) + "\n", "utf-8")
            self.log(
                f"tile {tile.key}: {meta['quakes']:,} quakes at M{PLANET_MC:g} "
                f"({meta['hidden']:,} hidden), {len(starts):,} days, {meta['seconds']:.0f}s"
            )
            out.append(PoolTile(tile, folder, train, val))
        return out


# ---------------------------------------------------------------- pumped planets (CB-30)


def pumped_pushes(kind: str, seed: int, grid: csep.Grid) -> Pushes:
    """The physics planet's fluid sources as forcing, month by month: each source pushes
    ``p0`` (MPa) for every day it's on, in its own cell. A source switches on and, one to five
    years later, off: the cut in the push that Groningen and Oklahoma's training years lack."""
    from craton.bench.planets import PLANET_START, WINDOW
    from craton.sim import SIMULATORS

    sim = SIMULATORS[KINDS[kind][0]]
    t0, t1 = to_seconds(PLANET_START), WINDOW.t1
    box = (grid.lat_min, grid.lat_max, grid.lon_min, grid.lon_max)
    p = sim.planet(box, seed)  # type: ignore[attr-defined]
    burn = sim.burn_in_days  # type: ignore[attr-defined]
    total = (t1 - t0) / DAY + burn
    sources = sim.fluids(p, total, np.random.default_rng([seed, 3]))  # type: ignore[attr-defined]
    month = PUSH_PERIOD_DAYS
    ends = t0 + month * DAY * np.arange(1, int((t1 - t0) / DAY // month) + 1)
    rows: list[tuple[float, int, float]] = []
    for s in sources:
        on, off = t0 + (s.t_on - burn) * DAY, t0 + (s.t_off - burn) * DAY
        cell = int(grid.cell_of(np.array([s.lat]), np.array([s.lon]))[0])
        if cell < 0:
            continue
        for end in ends:
            days = (min(end, off) - max(end - month * DAY, on)) / DAY
            if days > 0:
                rows.append((float(end), cell, s.p0 * days))
    rows.sort()
    return Pushes(
        np.array([r[0] for r in rows], dtype=np.float64),
        np.full(len(rows), month * DAY),
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=np.float64),
    )


def build_pumped_pool(
    field_: WorldField, root: Path, folder: Path, period_days: int, with_forcing: bool = True
) -> list[PoolTile]:
    """The pumped planets of a forced Field's ``planets`` section, as extra training cases:
    30-day forecasts read the Omori memory plus the forcing features, as CB-30's cases do
    (the memory alone for the forcing-blind ablation, so both see the same planets)."""
    from craton.bench.forcing import Scales, features, raw

    cfg = field_.config
    p = cfg.planets
    assert p is not None
    out: list[PoolTile] = []
    every = cfg.train.validation_every
    scored = PLANET_MC + SCORED_ABOVE_MC
    cut = PUMPED_CUT
    for seed in range(p.first_seed, p.first_seed + p.count):
        grid = planet_grid()
        tile = Tile(f"pumped_{p.kind}_{seed}", grid, PLANET_MC, PLANET_FIT_FROM)
        sub = folder / tile.key
        meta_path = sub / "meta.json"
        if meta_path.is_file():
            done = json.loads(meta_path.read_text("utf-8"))
            out.append(PoolTile(tile, sub, np.asarray(done["train"]), np.asarray(done["val"])))
            continue
        t0 = time.perf_counter()
        truth = planet_events(root, p.kind, seed, field_.log)
        view = PlanetView(root=root, planet=truth, planet_id=tile.key, cut=cut)
        fit = Etas().fit(tile_spec(tile, cut), view, 0)
        last = to_seconds(cut) - period_days * DAY
        starts = np.arange(to_seconds(PUMPED_FIRST_DAY), last + 1, PUMPED_STRIDE_DAYS * DAY)
        mem = memory_spec(PLANET_MC, cfg.memory.start)
        days = build_days(fit, truth, starts, grid, cfg, True, period_days, scored, mem)
        assert days.counts is not None
        push = pumped_pushes(p.kind, seed, grid)
        r = raw(push, grid, starts)
        scales = Scales.of(r)
        memory = days.memory
        if with_forcing:
            memory = np.concatenate([memory, features(r, grid, scales)], axis=-1)
        idx = np.arange(len(starts))
        # Validation in 30-day blocks of calendar time; a start whose period reaches into
        # one would have seen its answers.
        block = ((starts - starts[0]) / DAY // 30).astype(np.int64)
        is_val = block % every == every - 1
        reach_end = ((starts - starts[0]) / DAY + period_days) // 30
        sees_val = (reach_end.astype(np.int64) % every == every - 1) | is_val
        train, val = idx[~sees_val], idx[is_val]
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / "memory.npy", memory)
        np.save(sub / "log_etas.npy", days.log_etas)
        np.save(sub / "counts.npy", days.counts)
        np.save(sub / "weights.npy", np.ones(len(starts), np.float32))
        meta: dict[str, Any] = {
            "tile": tile.key,
            "case": tile.key,
            "planet": seed,
            "quakes": len(truth.above(PLANET_MC)),
            "pushes": len(push),
            "scales": {"rate": scales.rate, "total": scales.total},
            "features": int(memory.shape[-1]),
            "etas": {**fit.params.as_dict(), "b": fit.b, "mc": fit.mc},
            "train": train.tolist(),
            "val": val.tolist(),
            "seconds": time.perf_counter() - t0,
        }
        meta_path.write_text(json.dumps(meta) + "\n", "utf-8")
        field_.log(f"{tile.key}: {len(starts):,} starts, {meta['seconds']:.0f}s")
        out.append(PoolTile(tile, sub, train, val))
    return out
