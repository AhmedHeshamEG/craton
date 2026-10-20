"""The training loop. Short enough to read in one sitting, on purpose.

    for each step:
        cut a batch of masked windows and a batch of next-day questions (on the GPU a worker
        process cuts the next ones while the GPU works on these)
        forward pass under fp16 autocast, loss in bits for each objective
        weighted sum → backward (scaled, so small fp16 gradients don't vanish)
        clip the gradient, step AdamW, move the learning rate along its schedule
    every ``eval_every`` steps:
        score the fixed validation set; keep the weights if it's the best so far
        stop when ``patience`` checks in a row brought nothing

The learning rate warms up linearly, then follows a cosine down to a tenth of its peak, the
schedule most LLM pretraining runs use. Peak VRAM, tokens per second and wall-clock are
measured on every run: CB-15 reads them.
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from craton.config import CratonConfig
from craton.hardware import plan_precision, probe
from craton.model.craton import CratonModel
from craton.seed import derive_seed, seed_everything
from craton.train.context import Batch, Sources
from craton.train.data import Masked, NextDay, NextDayBatch
from craton.train.flops import flops_per_token

Log = Callable[[str], None]
#: Next-day questions per micro-batch: shorter than a masked batch, each is one whole window.
NEXT_DAY_BATCH = 16
VAL_BATCHES = 8


def _quiet(_: str) -> None:
    return None


@dataclass
class TrainResult:
    state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    stats: dict[str, Any] = field(default_factory=dict)


def lr_at(step: int, config: CratonConfig) -> float:
    t = config.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    span = max(t.max_steps - t.warmup_steps, 1)
    progress = min((step - t.warmup_steps) / span, 1.0)
    return t.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stream(torch.utils.data.IterableDataset[Any]):
    """Endless (masked batch, next-day batch) pairs, cut with one seeded generator.

    It runs in one worker process, so cutting batches never competes with the training loop
    for Python's interpreter lock, and the order is exactly the order a single loop would cut
    them in: a seed still fixes every batch.
    """

    def __init__(
        self, masked: Masked | None, nextday: NextDay | None, micro: int, seed: int
    ) -> None:
        self.masked, self.nextday, self.micro, self.seed = masked, nextday, micro, seed

    def __iter__(self) -> Iterator[tuple[Batch | None, NextDayBatch | None]]:
        rng = np.random.default_rng(self.seed)
        while True:
            mb = self.masked.sample(self.micro, rng) if self.masked else None
            nb = self.nextday.sample(NEXT_DAY_BATCH, rng) if self.nextday else None
            yield mb, nb


def batches(stream: Stream, workers: int) -> Iterator[tuple[Batch | None, NextDayBatch | None]]:
    """The stream through a worker process (``workers`` = 1), or inline (0, for the CPU and
    the tests, where starting a process costs more than it saves)."""
    if workers == 0:
        return iter(stream)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=workers, prefetch_factor=4, pin_memory=True
    )
    return iter(loader)


def train(
    config: CratonConfig,
    src: Sources,
    seed: int,
    next_day_window: tuple[float, float] | None,
    log: Log = _quiet,
    device: torch.device | None = None,
    windows: dict[str, float] | None = None,
) -> TrainResult:
    """Train one model on one split's sources. ``next_day_window`` = (start, end) of the
    midnights the next-day head may learn from, or None to leave that objective out."""
    seed_everything(seed)
    # D-046: the fast attention kernel's backward pass is not bitwise deterministic; known.
    warnings.filterwarnings("ignore", message="Memory Efficient attention defaults")
    device = device or _device()
    t = config.train
    length = config.model.context_tokens
    weights = {k: w for k, w in t.objectives.items() if w > 0}
    if next_day_window is None or not src.quakes:
        weights.pop("next_day", None)
    if src.gnss is None:
        weights.pop("gnss", None)
    if not src.quakes:
        weights.pop("masked", None)
    masked = (
        Masked(src, t.mask_share, windows or t.windows)
        if {"masked", "gnss"} & set(weights)
        else None
    )
    nextday = None
    if "next_day" in weights and next_day_window is not None:
        from craton.model.heads import forecast_grid

        nextday = NextDay(src, forecast_grid(), *next_day_window)

    model = CratonModel(config).to(device)
    plan = plan_precision(t.precision, probe()) if device.type == "cuda" else None
    use_amp = plan is not None and plan.dtype == "fp16"

    # Matrices decay; biases, norms, embeddings and the persistence line don't.
    def decays(name: str, p: torch.Tensor) -> bool:
        skip = ("embed", "mag_type", "recent", "modality")
        return p.ndim >= 2 and not any(k in name for k in skip)

    decay = [p for n, p in model.named_parameters() if decays(n, p)]
    no_decay = [p for n, p in model.named_parameters() if not decays(n, p)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": t.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=t.lr,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",  # one kernel for the whole update
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_m = masked.validation(t.micro_batch, VAL_BATCHES) if masked else []
    val_n = nextday.validation(NEXT_DAY_BATCH) if nextday else []

    def gpu(x: torch.Tensor) -> torch.Tensor:
        # Never a blocking copy: one would make the CPU wait for the forward pass before it
        # could queue the loss, and the GPU would sit idle meanwhile.
        return x.to(device, non_blocking=True)

    def losses(m_batch: Batch | None, n_batch: NextDayBatch | None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            if m_batch is not None:
                tok = m_batch.tokens.to(device)
                # Hidden tokens are picked by positions the CPU already knows: no GPU sync.
                h = model(tok).flatten(0, 1)
                if "masked" in weights and len(m_batch.quake_targets):
                    out["masked"] = model.masked.loss_bits(
                        h[gpu(m_batch.quake_pos)],
                        gpu(m_batch.quake_targets),
                        gpu(m_batch.quake_weight),
                    )
                if "gnss" in weights and len(m_batch.gnss_lo):
                    out["gnss"] = model.gnss.loss_bits(
                        h[gpu(m_batch.gnss_pos)],
                        gpu(m_batch.gnss_years),
                        gpu(m_batch.gnss_lo),
                        gpu(m_batch.gnss_hi),
                        gpu(m_batch.gnss_weight),
                    )
            if n_batch is not None:
                tok = n_batch.tokens.to(device)
                h = model(tok)[:, -1]
                bits, n = model.next_day.loss_bits(
                    h,
                    gpu(n_batch.recent),
                    gpu(n_batch.counts),
                    gpu(n_batch.cells),
                    gpu(n_batch.mags),
                    gpu(n_batch.day_of),
                    n_batch.log_factorials,
                )
                out["next_day"] = bits / torch.clamp(n, min=1.0)
                out["_next_day_bits"], out["_next_day_n"] = bits, n
        return out

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        res: dict[str, float] = {}
        if val_m:
            parts_m = [losses(b, None) for b in val_m]
            for key in ("masked", "gnss"):
                got = [p[key].item() for p in parts_m if key in p]
                if got:
                    res[f"val_{key}"] = float(np.mean(got))
        if val_n:
            parts = [losses(None, b) for b in val_n]
            bits = sum(p["_next_day_bits"].item() for p in parts)
            n = sum(p["_next_day_n"].item() for p in parts)
            res["val_next_day"] = bits / max(n, 1.0)
        model.train()
        return res

    def objective(v: dict[str, float]) -> float:
        return sum(weights[k] * v[f"val_{k}"] for k in weights if f"val_{k}" in v)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, float]] = []
    best, best_state, best_step, stale = math.inf, copy.deepcopy(model.state_dict()), 0, 0
    tokens_seen, train_time = 0, 0.0
    started = segment = time.perf_counter()
    model.train()
    stream = Stream(masked, nextday, t.micro_batch, derive_seed(seed, "batches"))
    feed = batches(stream, 1 if device.type == "cuda" else 0)
    step = 0
    running: dict[str, torch.Tensor] = {}
    since = 0
    for step in range(t.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, config)
        since += 1
        for _ in range(t.grad_accum):
            mb, nb = next(feed)
            parts = losses(mb, nb)
            loss = sum(weights[k] * parts[k] for k in parts if not k.startswith("_"))
            assert isinstance(loss, torch.Tensor)
            scaled: torch.Tensor = scaler.scale(loss / t.grad_accum)
            scaled.backward()  # type: ignore[no-untyped-call]
            for k, part in parts.items():
                if not k.startswith("_"):  # summed on the GPU, read only at validation
                    part = part.detach() / t.grad_accum
                    running[k] = running[k] + part if k in running else part
            tokens_seen += (mb.tokens.shape[0] * length if mb else 0) + (
                NEXT_DAY_BATCH * length if nb else 0
            )
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % t.eval_every == 0 or step + 1 == t.max_steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_time += time.perf_counter() - segment
            v = validate()
            mean = {k: x.item() / since for k, x in running.items()}
            running, since = {}, 0
            score = objective(v)
            row = {"step": float(step + 1), "lr": lr_at(step, config), **mean, **v}
            history.append(row)
            shown = " · ".join(f"{k} {x:.3f}" for k, x in row.items() if k != "step")
            log(f"step {step + 1} · {shown}")
            if score < best - 1e-4:
                best, best_state, best_step, stale = (
                    score,
                    copy.deepcopy(model.state_dict()),
                    step + 1,
                    0,
                )
            else:
                stale += 1
                if stale >= t.patience:
                    log(f"stopping: {t.patience} checks without progress; best at step {best_step}")
                    break
            segment = time.perf_counter()

    del feed  # stops the worker
    wall = time.perf_counter() - started
    tps = tokens_seen / max(train_time, 1e-9)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    stats: dict[str, Any] = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": "fp16 + GradScaler" if use_amp else "fp32",
        "steps": step + 1,
        "best_step": best_step,
        "best_val": best,
        "tokens_seen": tokens_seen,
        "tokens_per_s": tps,
        "train_seconds": train_time,
        "wall_seconds": wall,
        "peak_vram_bytes": int(peak),
        "flops_per_token": flops_per_token(config),
        "params": model.count(),
        "objectives": weights,
        "inputs": list(config.model.inputs),
    }
    return TrainResult({k: v.detach().cpu() for k, v in best_state.items()}, history, stats)
