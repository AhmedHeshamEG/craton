"""The forced Field: the Field that also reads what people did to the rock (CB-30, P8e).

Everywhere else the Field corrects ETAS from the quakes alone. In Groningen and Oklahoma the
push is measured (``craton.bench.forcing``), so each cell's token carries it too: next to its
Omori memory, the push in the cell and around it over the last 90 and 365 days, the total
since the series began, and how old the newest report is. Same network, same ETAS start line,
same bounded correction; only the cell token is longer. If the Field can learn
"pressure → quakes", it has to come from those extra numbers, and the ablation without them
(``with_forcing=False``) says how much they are worth.

Training, for one seed: each CB-30 case is one tile, its box and cells exactly the task's.
Every midnight from the case's ``train_from`` to 30 days before the cut answers with the next
30 days' counts (as CB-19's Field does, D-084); one 30-day block in ``validation_every`` is
validation, and a start whose 30 days reach into a validation block is dropped. The members
learn from both cases at once (``craton.field.world.train_world_member``), and the forcing
features are divided by each box's own scale from its training days, so "twice the usual
push" reads the same in both places.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton import __version__
from craton.adapters.region import load_region
from craton.bench import csep
from craton.bench.baselines import Etas
from craton.bench.data import DAY
from craton.bench.forcing import FEATURES, ForcingSpec, Scales, features, pushes, raw
from craton.bench.induced import CASES, CUT, PERIOD_DAYS, Case, case_of, case_spec
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field.config import field_hash
from craton.field.forecaster import VALIDATION_BLOCK_DAYS, build_days, magnitude_shares
from craton.field.world import (
    PoolTile,
    Tile,
    WorldField,
    memory_spec,
    predict_world,
    train_world_member,
)
from craton.spec.timeaxis import to_seconds
from craton.train.checkpoint import checkpoints_dir

FORCED_VERSION = "f0.1"


def _start(case: Case, root: Path, memory_start: str) -> str:
    region = load_region(case.region, root)
    return memory_start if to_seconds(memory_start) >= region.t_start else region.start


@dataclass
class ForcedField(WorldField):
    """CB-30 through the Field, with (or, for the ablation, without) the forcing."""

    config_name: str = "forced"
    with_forcing: bool = True

    # ------------------------------------------------------------------ tiles and the pool

    def tiles(self, root: Path) -> list[Tile]:
        out = []
        for case in CASES:
            spec = case_spec(case)
            out.append(Tile(case.region, spec.grid, case.mc, case.fit_from))
        return out

    def pool_id(self) -> str:
        cfg = self.config
        assert cfg.world is not None
        body = {
            "version": FORCED_VERSION,
            "cases": [
                {**{k: v for k, v in c.__dict__.items() if k != "forcing"}, **c.forcing.as_dict()}
                for c in CASES
            ],
            "cut": CUT,
            "memory_start": cfg.memory.start,
            "validation_every": cfg.train.validation_every,
            "forcing": list(FEATURES) if self.with_forcing else [],
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

    def _forcing(
        self, view: BenchContext, case: Case, grid: csep.Grid, starts: np.ndarray, scales: Scales
    ) -> np.ndarray:
        return features(raw(pushes(view, case.forcing, grid), grid, starts), grid, scales)

    def build_pool(self, root: Path) -> list[PoolTile]:
        """Each case's training starts, built once and cached (``data/world/``)."""
        from craton.bench.world import RegionView

        cfg = self.config
        every = cfg.train.validation_every
        out: list[PoolTile] = []
        for case in CASES:
            spec = case_spec(case)
            tile = Tile(case.region, spec.grid, case.mc, case.fit_from)
            folder = self.pool_dir(root) / tile.key
            meta_path = folder / "meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text("utf-8"))
                out.append(
                    PoolTile(tile, folder, np.asarray(meta["train"]), np.asarray(meta["val"]))
                )
                continue
            t0 = time.perf_counter()
            g = spec.grid
            view = RegionView(
                root=root,
                region=case.region,
                box=(g.lat_min, g.lat_max, g.lon_min, g.lon_max),
                cut=CUT,
            )
            fit = Etas().fit(spec, view, 0)
            events = view.train_events("induced")
            last = to_seconds(CUT) - PERIOD_DAYS * DAY
            starts = np.arange(to_seconds(case.train_from), last + 1, DAY)
            days = build_days(
                fit,
                events,
                starts,
                g,
                cfg,
                True,
                PERIOD_DAYS,
                case.scored,
                memory_spec(case.mc, _start(case, root, cfg.memory.start)),
            )
            assert days.counts is not None
            memory = days.memory
            scales = Scales(1.0, 1.0)
            if self.with_forcing:
                r = raw(pushes(view, case.forcing, g), g, starts)
                scales = Scales.of(r)
                memory = np.concatenate([memory, features(r, g, scales)], axis=-1)
            idx = np.arange(len(starts))
            is_val = (idx // VALIDATION_BLOCK_DAYS) % every == every - 1
            # A start whose 30 days reach into a validation block would have seen its answers.
            reach = np.convolve(is_val.astype(np.int64)[::-1], np.ones(PERIOD_DAYS, np.int64))
            sees_val = reach[: len(idx)][::-1] > 0
            train, val = idx[~sees_val], idx[is_val]
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "memory.npy", memory)
            np.save(folder / "log_etas.npy", days.log_etas)
            np.save(folder / "counts.npy", days.counts)
            np.save(folder / "weights.npy", np.ones(len(starts), np.float32))
            meta = {
                "tile": tile.key,
                "case": case.key,
                "region": case.region,
                "mc": case.mc,
                "scored": case.scored,
                "cells": g.n_cells,
                "days": len(starts),
                "quakes": len(events.above(case.mc)),
                "scales": {"rate": scales.rate, "total": scales.total},
                "features": int(memory.shape[-1]),
                "etas": {**fit.params.as_dict(), "b": fit.b, "mc": fit.mc},
                "train": train.tolist(),
                "val": val.tolist(),
                "seconds": time.perf_counter() - t0,
            }
            meta_path.write_text(json.dumps(meta) + "\n", "utf-8")
            self.log(
                f"{case.key}: {len(starts):,} starts, {memory.shape[-1]} features per cell, "
                f"{meta['seconds']:.0f}s"
            )
            out.append(PoolTile(tile, folder, train, val))
        return out

    def pool_dir(self, root: Path) -> Path:
        return root / "data" / "world" / f"pool-{self.config_name}-{self.pool_id()}"

    def scales(self, root: Path, case: Case) -> Scales:
        for tile in self.build_pool(root):
            meta = json.loads((tile.folder / "meta.json").read_text("utf-8"))
            if meta["case"] == case.key:
                return Scales(meta["scales"]["rate"], meta["scales"]["total"])
        raise KeyError(case.key)

    # ------------------------------------------------------------------ training

    def folder(self, root: Path, seed: int) -> Path:
        cfg = self.config
        return checkpoints_dir(root) / (
            f"forced-{cfg.name}-{FORCED_VERSION}-{self.pool_id()}-{field_hash(cfg)[:8]}-s{seed}"
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
            "forced_version": FORCED_VERSION,
            "with_forcing": self.with_forcing,
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

    def forecast_forced(
        self, spec: ForecastSpec, ctx: BenchContext, forcing: ForcingSpec, seed: int
    ) -> GriddedForecast:
        """CB-30: ETAS fitted on the box's past, corrected by the members, each period
        reading the quakes and the forcing known at its start."""
        case = case_of(spec)
        cfg = self.config
        fit = Etas().fit(spec, ctx, seed)
        states = self.states(ctx.root, seed)
        days = build_days(
            fit,
            ctx.events,
            spec.day_starts,
            spec.grid,
            cfg,
            False,
            spec.period_days,
            spec.scored,
            memory_spec(spec.mc, _start(case, ctx.root, cfg.memory.start)),
        )
        memory = days.memory
        if self.with_forcing:
            extra = self._forcing(
                ctx, case, spec.grid, spec.day_starts, self.scales(ctx.root, case)
            )
            memory = np.concatenate([memory, extra], axis=-1)
        m3 = predict_world(states, memory, days.log_etas[..., 1], spec.grid, cfg)
        shares = magnitude_shares(fit, spec.grid, spec.max_magnitude)
        first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
        return GriddedForecast(m3.mean(axis=0) / shares[first:].sum(), shares)


def card(meta: dict[str, Any]) -> str:
    """The model card of one forced Field checkpoint."""
    runs = meta["members"]
    gain = ", ".join(f"{r['validation_start'] - r['validation_best']:.4f}" for r in runs)
    reads = "the quakes and the measured forcing" if meta["with_forcing"] else "the quakes only"
    return (
        f"# Forced Field{'' if meta['with_forcing'] else ' without the forcing (ablation)'}"
        f" · seed {meta['seed']}\n\n"
        "- **What it does:** 30-day forecasts of CB-30's two boxes (Groningen, Oklahoma), "
        "starting from ETAS fitted there and correcting it cell by cell, at most "
        f"{meta['config']['model']['max_nudge']:g}x either way.\n"
        f"- **Reads:** {reads}, as known at each period's start.\n"
        f"- **Learned from:** {', '.join(meta['tiles'])}, {meta['training_days']:,} training "
        f"starts, all before {CUT[:10]}.\n"
        f"- **Size:** {meta['parameters']:,} parameters per member, {len(runs)} members.\n"
        f"- **Training:** validation gain over ETAS (nats per quake) {gain}; "
        f"{meta['train_seconds'] / 60:.1f} min.\n"
        f"- **Code:** Craton {meta['craton_version']}, forced {meta['forced_version']}, pool "
        f"`{meta['pool']}`, config hash `{meta['config_hash']}`.\n"
    )
