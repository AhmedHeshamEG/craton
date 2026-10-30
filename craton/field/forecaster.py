"""The Field as a forecaster: data, training, checkpoints, and CB-02 / CB-03 forecasts.

For a task and a seed:

1. ETAS is fitted on the task's split (the ``etas`` baseline's own fit, so the starting line
   is exactly the floor).
2. Every day from ``train.first_day`` to the split's end is a training day. Its memory, its
   ETAS rates and its answers (M2-3 and M3+ counts per cell) are all built from the split's
   quakes, each midnight reading only the quakes before it.
3. ``train.members`` Fields are trained on those days (Poisson log-likelihood of the counts,
   the benchmark's own score), each stopped early on one 30-day block in
   ``train.validation_every``.
4. Test days are built from the snapshot the same way (each midnight still reads only its
   past) and the forecast is the members' average M3+ rate. Magnitudes follow ETAS's
   Gutenberg-Richter b-value, as in the ``etas`` baseline, up to ``MAX_MAGNITUDE`` (P7: an
   unbounded Gutenberg-Richter law breaks the region's slip budget, ``craton.trust.physics``).
5. (P7) The test-time catalogue passes the guard first (``craton.trust.guard``): duplicates
   and magnitude-scale glitches on the live edge are removed before the Field reads it.
6. (P7) ``predictive`` adds what the rates alone can't say: every member's rates, and the
   negative binomial dispersion of day totals and of cell counts, fitted on the training
   split's validation days (``calibration.json`` beside the checkpoint).

Checkpoints live in ``results/checkpoints/field-...`` (gitignored), named by everything that
made them, with a model card beside each.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton import __version__
from craton.bench import csep, etas
from craton.bench.baselines import Etas
from craton.bench.data import DAY, Events
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field.base import etas_daily
from craton.field.config import FieldConfig, field_hash, load_field_config
from craton.field.memory import MemorySpec, memory
from craton.field.model import FieldNet
from craton.spec.timeaxis import to_seconds
from craton.train.checkpoint import checkpoints_dir
from craton.trust import calibration, guard
from craton.trust.predictive import Predictive

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
Log = Callable[[str], None]

#: Bumped whenever the Field's code changes what a checkpoint would contain.
FIELD_VERSION = "f0.1"
#: The scored threshold, and the smallest magnitude the loss learns from (ETAS's mc).
M_SCORED = 3.0
VALIDATION_BLOCK_DAYS = 30
INFER_DAYS = 16
#: The largest magnitude the Field forecasts: CPTI15's Mw 7.1 for Avezzano 1915, the largest
#: quake on record in the box, plus 0.2 for what history may have missed.
MAX_MAGNITUDE = 7.3
CALIBRATION_VERSION = "c2"


def _quiet(_: str) -> None:
    return None


@dataclass
class Days:
    """Everything the Field reads and is graded on, for a run of consecutive days."""

    starts: FloatArray  # seconds since 1970, midnights
    memory: np.ndarray  # (days, cells, F) float16
    log_etas: np.ndarray  # (days, cells, 2) float32: log ETAS rate of M2-3 and M3+
    counts: np.ndarray | None  # (days, cells, 2) float32, training days only
    #: Each start forecasts this many days from its midnight (CB-19: 30).
    span_days: int = 1


def _counts(
    ev: Events,
    grid: csep.Grid,
    starts: FloatArray,
    mc: float,
    span_days: int = 1,
    scored: float = M_SCORED,
) -> np.ndarray:
    if span_days != 1:
        return _span_counts(ev, grid, starts, mc, span_days, scored)
    ev = ev.above(mc).window(float(starts[0]), float(starts[-1]) + DAY)
    cell = grid.cell_of(ev.lat, ev.lon)
    # Each quake to the day it falls in; ``starts`` need not be consecutive (validation days).
    day = np.searchsorted(starts, ev.time, side="right") - 1
    inside = (day >= 0) & (ev.time < starts[np.maximum(day, 0)] + DAY)
    big = (ev.mag >= scored - 1e-9).astype(np.int64)
    ok = (cell >= 0) & inside
    out = np.zeros((len(starts), grid.n_cells, 2), np.float32)
    np.add.at(out, (day[ok], cell[ok], big[ok]), 1.0)
    return out


def _span_counts(
    ev: Events, grid: csep.Grid, starts: FloatArray, mc: float, span_days: int, scored: float
) -> np.ndarray:
    """Counts in the ``span_days`` from each midnight; the spans overlap, so every day is
    counted once and the spans are read off the running sum."""
    first = float(starts.min())
    n_days = round((float(starts.max()) - first) / DAY) + span_days
    ev = ev.above(mc).window(first, first + n_days * DAY)
    cell = grid.cell_of(ev.lat, ev.lon)
    day = np.floor((ev.time - first) / DAY).astype(np.int64)
    big = (ev.mag >= scored - 1e-9).astype(np.int64)
    ok = cell >= 0
    daily = np.zeros((n_days + 1, grid.n_cells, 2), np.float32)
    np.add.at(daily, (day[ok] + 1, cell[ok], big[ok]), 1.0)
    run = np.cumsum(daily, axis=0)
    i = np.round((starts - first) / DAY).astype(np.int64)
    return np.asarray(run[i + span_days] - run[i], dtype=np.float32)


def build_days(
    fit: etas.EtasFit,
    events: Events,
    starts: FloatArray,
    grid: csep.Grid,
    cfg: FieldConfig,
    with_counts: bool,
    span_days: int = 1,
    scored: float = M_SCORED,
    memory_spec: MemorySpec | None = None,
) -> Days:
    """Memory, ETAS rates and (optionally) answers for ``starts``, reading only ``events``
    before each midnight; each start's rates and answers cover ``span_days`` days. Answers
    split at ``scored`` (central Italy: M3); the memory follows ``cfg`` unless given."""
    proj = etas.Projection.for_grid(grid)
    spec = memory_spec or MemorySpec(m_min=cfg.memory.m_min, start=to_seconds(cfg.memory.start))
    mem = memory(events, grid, starts, spec)
    # ETAS reads every quake since the memory starts, also before its own fit window.
    hist = events.window(min(fit.t_ref, spec.start), float(starts[-1])).above(fit.mc)
    cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, fit.t_ref)
    rates = etas_daily(fit, cat, (starts - fit.t_ref) / DAY, grid, proj, span=span_days)
    big = 10.0 ** (-fit.b * (scored - fit.mc))  # GR share of the scored among M ≥ mc
    log_r = np.log(np.maximum(rates, 1e-30))
    log_etas = np.stack([log_r + math.log(1 - big), log_r + math.log(big)], axis=-1)
    counts = _counts(events, grid, starts, fit.mc, span_days, scored) if with_counts else None
    return Days(starts, mem, log_etas.astype(np.float32), counts, span_days)


def train_member(
    days: Days,
    grid: csep.Grid,
    cfg: FieldConfig,
    member_seed: int,
    log: Log = _quiet,
    init: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """One Field trained on ``days``; returns its weights and how training went. ``init``
    starts it from an earlier stage's weights (the curriculum) instead of from ETAS."""
    assert days.counts is not None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg.train
    idx = np.arange(len(days.starts))
    val_mask = (idx // VALIDATION_BLOCK_DAYS) % tc.validation_every == tc.validation_every - 1
    # A training start whose span reaches into a validation block would have seen its answers.
    reach = np.convolve(val_mask.astype(np.int64)[::-1], np.ones(days.span_days, np.int64))
    sees_val = reach[: len(idx)][::-1] > 0
    train_d, val_d = idx[~sees_val], idx[val_mask]
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    net = FieldNet(days.memory.shape[-1], grid, cfg.model).to(dev)
    if init is not None:
        net.load_state_dict({**init, "place": net.place})
    opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    mem = torch.from_numpy(days.memory)
    log_e = torch.from_numpy(days.log_etas)
    y = torch.from_numpy(days.counts)

    def nll(sel: IntArray) -> tuple[torch.Tensor, torch.Tensor]:
        lb = log_e[sel].to(dev)
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            corr = net(mem[sel].to(dev).float(), lb[..., 1])
        lam = lb + corr
        yy = y[sel].to(dev)
        return -(yy * lam - lam.exp()).sum(), yy.sum()

    def validate() -> float:
        net.eval()
        tot = n = 0.0
        with torch.no_grad():
            for i in range(0, len(val_d), INFER_DAYS):
                loss, count = nll(val_d[i : i + INFER_DAYS])
                tot, n = tot + float(loss), n + float(count)
        net.train()
        return tot / max(n, 1.0)

    best = validate()
    start_loss = best
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    bad = epochs = 0
    for epoch in range(tc.max_epochs):
        epochs = epoch + 1
        order = rng.permutation(train_d)
        for i in range(0, len(order), tc.days_per_step):
            loss, _ = nll(order[i : i + tc.days_per_step])
            opt.zero_grad(set_to_none=True)
            # Scaled to a few units per step; the Poisson sum over 8 days is in the thousands.
            scaler.scale(loss / (tc.days_per_step * 1000.0)).backward()  # type: ignore[no-untyped-call]
            scaler.step(opt)
            scaler.update()
        v = validate()
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        log(f"  member {member_seed} epoch {epochs} validation {v:.4f} (best {best:.4f})")
        if bad >= tc.patience:
            break
    return {k: v.cpu() for k, v in state.items()}, {
        "epochs": float(epochs),
        "validation_start": start_loss,
        "validation_best": best,
    }


def predict(
    states: list[dict[str, torch.Tensor]], days: Days, grid: csep.Grid, cfg: FieldConfig
) -> FloatArray:
    """(days, cells) the members' average M3+ rate."""
    return np.asarray(predict_members(states, days, grid, cfg).mean(axis=0), dtype=np.float64)


def predict_members(
    states: list[dict[str, torch.Tensor]], days: Days, grid: csep.Grid, cfg: FieldConfig
) -> FloatArray:
    """(members, days, cells) each member's M3+ rate."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros((len(states), len(days.starts), grid.n_cells))
    mem = torch.from_numpy(days.memory)
    log_e = torch.from_numpy(days.log_etas[..., 1])
    for k, state in enumerate(states):
        net = FieldNet(days.memory.shape[-1], grid, cfg.model)
        net.load_state_dict(state)
        net.to(dev).eval()
        with torch.no_grad():
            for i in range(0, len(days.starts), INFER_DAYS):
                lb = log_e[i : i + INFER_DAYS].to(dev)
                corr = net(mem[i : i + INFER_DAYS].to(dev).float(), lb)
                rate = (lb + corr[..., 1]).exp().double().cpu().numpy()
                out[k, i : i + INFER_DAYS] = rate
    return out


@dataclass
class FieldForecaster:
    """CB-02 and CB-03 through the Field. Used by ``craton-mini`` for its forecasts."""

    config_name: str = "mini"
    log: Log = _quiet
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def config(self) -> FieldConfig:
        return load_field_config(self.config_name)

    def folder(
        self, ctx: BenchContext, spec: ForecastSpec, seed: int, config: FieldConfig | None = None
    ) -> Path:
        cfg = config or self.config
        manifest = json.loads((ctx.split_dir(spec.fit_split) / "manifest.json").read_text("utf-8"))
        name = (
            f"field-{cfg.name}-{FIELD_VERSION}-{spec.trained_as}-{manifest['id'][:12]}-"
            f"{field_hash(cfg)[:8]}-s{seed}"
        )
        return checkpoints_dir(ctx.root) / name

    def grown_from(
        self, ctx: BenchContext, spec: ForecastSpec, seed: int
    ) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]], str]:
        """The largest ensemble already trained with this exact config but fewer members:
        an ensemble grows, it is never retrained (constructive-only, D-076). Members are
        trained one by one from their own seeds, so the first k of a bigger ensemble are the
        same job as a k-member ensemble."""
        tc = self.config.train
        for k in range(tc.members - 1, 0, -1):
            smaller = self.config.model_copy(update={"train": tc.model_copy(update={"members": k})})
            folder = self.folder(ctx, spec, seed, smaller)
            if (folder / "model.pt").is_file():
                body = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
                meta = json.loads((folder / "meta.json").read_text("utf-8"))
                return list(body["members"]), list(meta["members"]), folder.name
        return [], [], ""

    def states(
        self, spec: ForecastSpec, ctx: BenchContext, seed: int, fit: etas.EtasFit
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
        folder = self.folder(ctx, spec, seed)
        path = folder / "model.pt"
        if path.is_file():
            body = torch.load(path, map_location="cpu", weights_only=True)
            meta = json.loads((folder / "meta.json").read_text("utf-8"))
            return list(body["members"]), meta
        cfg = self.config
        t0 = time.perf_counter()
        cut = to_seconds(spec.fit_end)
        first = to_seconds(cfg.train.first_day)
        # Every midnight whose whole period ends before the cut-off is a training example.
        starts = np.arange(first, cut - spec.period + 1, DAY)
        self.log(f"Field {spec.task_id} seed {seed}: building {len(starts)} training days")
        days = build_days(
            fit, ctx.train_events(spec.fit_split), starts, spec.grid, cfg, True, spec.period_days
        )
        states, runs, base = self.grown_from(ctx, spec, seed)
        if base:
            self.log(f"Field {spec.task_id} seed {seed}: keeping {len(states)} members of {base}")
        starts_from = self.starts_from(ctx, seed)
        for m in range(len(states), cfg.train.members):
            init = starts_from[m % len(starts_from)] if starts_from else None
            st, info = train_member(days, spec.grid, cfg, seed * 100 + m, self.log, init)
            states.append(st)
            runs.append(info)
        meta = {
            "task": spec.task_id,
            "split": spec.fit_split,
            "seed": seed,
            "config": cfg.model_dump(mode="json"),
            "config_hash": field_hash(cfg),
            "field_version": FIELD_VERSION,
            "craton_version": __version__,
            "training_days": len(starts),
            "first_day": cfg.train.first_day,
            "members": runs,
            "grown_from": base or None,
            "starts_from": cfg.curriculum.start_from if cfg.curriculum else None,
            "etas": {**fit.params.as_dict(), "b": fit.b, "mc": fit.mc},
            "parameters": sum(v.numel() for v in states[0].values()),
            "train_seconds": time.perf_counter() - t0,
        }
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"members": states}, path)
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
        (folder / "card.md").write_text(card(meta), "utf-8")
        return states, meta

    def starts_from(self, ctx: BenchContext, seed: int) -> list[dict[str, torch.Tensor]]:
        """The curriculum's previous stage (P8f): the members of the Field this config starts
        from, trained with the same seed; none for a Field that starts from ETAS alone."""
        cur = self.config.curriculum
        if cur is None:
            return []
        from craton.field.world import field_stage

        return field_stage(cur.start_from, self.log).states(ctx.root, seed)

    def inputs(self, spec: ForecastSpec, ctx: BenchContext) -> Events:
        """The test-time catalogue after the guard, which watches the live edge: the test
        window and the ``guard.LIVE_EDGE_DAYS`` before it."""
        since = spec.window.t0 - guard.LIVE_EDGE_DAYS * DAY
        key = f"guarded:{since:.0f}"
        if key not in ctx._cache:
            ctx._cache[key] = guard.clean(ctx.events, since)
        out: tuple[Events, guard.GuardReport] = ctx._cache[key]
        return out[0]

    def members(
        self, spec: ForecastSpec, ctx: BenchContext, seed: int
    ) -> tuple[FloatArray, FloatArray, etas.EtasFit, list[dict[str, torch.Tensor]]]:
        """(members, days, cells) M1.5+ rates, the magnitude shares, the ETAS fit, the states."""
        fit = Etas().fit(spec, ctx, seed)
        states, _ = self.states(spec, ctx, seed, fit)
        days = build_days(
            fit,
            self.inputs(spec, ctx),
            spec.day_starts,
            spec.grid,
            self.config,
            False,
            spec.period_days,
        )
        m3 = predict_members(states, days, spec.grid, self.config)
        shares = magnitude_shares(fit, spec.grid, spec.max_magnitude)
        first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
        return m3 / shares[first:].sum(), shares, fit, states

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        rates, shares, _, _ = self.members(spec, ctx, seed)
        return GriddedForecast(rates.mean(axis=0), shares)

    def predictive(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> Predictive:
        rates, shares, fit, states = self.members(spec, ctx, seed)
        cal = self.calibration(spec, ctx, seed, fit, states)
        return Predictive(
            GriddedForecast(rates.mean(axis=0), shares),
            members=rates,
            day_k=cal["day_k"],
            day_gamma=cal["day_gamma"],
            cell_k=cal["cell_k"],
        )

    def calibration(
        self,
        spec: ForecastSpec,
        ctx: BenchContext,
        seed: int,
        fit: etas.EtasFit,
        states: list[dict[str, torch.Tensor]],
    ) -> dict[str, Any]:
        """Negative binomial sizes of day totals and cell counts (M3+), fitted on the
        training split's validation days (never the test). Cached beside the checkpoint."""
        path = self.folder(ctx, spec, seed) / "calibration.json"
        if path.is_file():
            body: dict[str, Any] = json.loads(path.read_text("utf-8"))
            if body.get("version") == CALIBRATION_VERSION:
                return body
        cfg = self.config
        cut = to_seconds(spec.fit_end)
        starts = np.arange(to_seconds(cfg.train.first_day), cut - DAY + 1, DAY)
        idx = np.arange(len(starts))
        every = cfg.train.validation_every
        val = (idx // VALIDATION_BLOCK_DAYS) % every == every - 1
        days = build_days(fit, ctx.train_events(spec.fit_split), starts[val], spec.grid, cfg, True)
        assert days.counts is not None
        mean = predict_members(states, days, spec.grid, cfg).mean(axis=0)
        n = days.counts[..., 1].astype(np.float64)
        body = {
            "version": CALIBRATION_VERSION,
            **dict(
                zip(
                    ("day_k", "day_gamma"),
                    calibration.fit_day_dispersion(n.sum(axis=1), mean.sum(axis=1)),
                    strict=True,
                )
            ),
            "cell_k": calibration.fit_cell_dispersion(n > 0, mean),
            "validation_days": float(val.sum()),
        }
        path.write_text(json.dumps(body, indent=2) + "\n", "utf-8")
        return body


def magnitude_shares(
    fit: etas.EtasFit, grid: csep.Grid, max_magnitude: float = MAX_MAGNITUDE
) -> FloatArray:
    """Gutenberg-Richter shares over the grid's bins with ETAS's b-value, nothing above
    ``max_magnitude`` (central Italy: ``MAX_MAGNITUDE``)."""
    gr = csep.gutenberg_richter_bins(fit.b, fit.mc, grid)
    gr = np.where(grid.magnitudes <= max_magnitude + 1e-9, gr, 0.0)
    return np.asarray(gr / gr.sum(), dtype=np.float64)


def card(meta: dict[str, Any]) -> str:
    """The model card of one Field checkpoint."""
    runs = meta["members"]
    epochs = ", ".join(f"{r['epochs']:.0f}" for r in runs)
    gain = ", ".join(f"{r['validation_start'] - r['validation_best']:.4f}" for r in runs)
    return (
        f"# Field · {meta['task']} · seed {meta['seed']}\n\n"
        f"- **What it does:** daily forecasts for {meta['task']}, starting from ETAS and "
        "correcting it cell by cell, at most "
        f"{meta['config']['model']['max_nudge']:g}x either way.\n"
        f"- **Learned from:** split `{meta['split']}`, {meta['training_days']:,} days from "
        f"{meta['first_day'][:10]}; every midnight reads only the quakes before it.\n"
        f"- **Size:** {meta['parameters']:,} parameters per member, {len(runs)} members "
        "averaged.\n"
        f"- **Training:** epochs per member {epochs}; validation gain over ETAS (nats per "
        f"quake) {gain}; {meta['train_seconds'] / 60:.1f} min.\n"
        f"- **Starting line:** ETAS with "
        + ", ".join(f"{k} {v:.4g}" for k, v in meta["etas"].items())
        + ".\n"
        f"- **Code:** Craton {meta['craton_version']}, Field {meta['field_version']}, config "
        f"hash `{meta['config_hash']}`.\n"
    )
