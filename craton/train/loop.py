"""The training loop. Short enough to read in one sitting, on purpose.

    for each step:
        cut a batch of masked windows and a batch of next-day questions
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from craton.config import CratonConfig
from craton.hardware import plan_precision, probe
from craton.model.craton import CratonModel
from craton.seed import derive_seed, seed_everything
from craton.train.data import Catalog, MaskedWindows, NextDay
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


def train(
    config: CratonConfig,
    cat: Catalog,
    seed: int,
    next_day_window: tuple[float, float] | None,
    log: Log = _quiet,
    device: torch.device | None = None,
) -> TrainResult:
    """Train one model on one catalogue. ``next_day_window`` = (start, end) of the midnights
    the next-day head may learn from, or None to train the masked objective alone."""
    seed_everything(seed)
    # D-046: the fast attention kernel's backward pass is not bitwise deterministic; known.
    warnings.filterwarnings("ignore", message="Memory Efficient attention defaults")
    device = device or _device()
    rng = np.random.default_rng(derive_seed(seed, "batches"))
    t = config.train
    length = config.model.context_tokens
    weights = dict(t.objectives)
    if next_day_window is None:
        weights["next_day"] = 0.0
    masked = MaskedWindows(cat, length, t.mask_share) if weights.get("masked", 0) > 0 else None
    nextday = None
    if weights.get("next_day", 0) > 0 and next_day_window is not None:
        from craton.model.heads import forecast_grid

        nextday = NextDay(cat, forecast_grid(), length, *next_day_window)

    model = CratonModel(config).to(device)
    plan = plan_precision(t.precision, probe()) if device.type == "cuda" else None
    use_amp = plan is not None and plan.dtype == "fp16"

    # Matrices decay; biases, norms, embeddings and the persistence line don't.
    def decays(name: str, p: torch.Tensor) -> bool:
        return p.ndim >= 2 and not any(k in name for k in ("embed", "mag_type", "recent"))

    decay = [p for n, p in model.named_parameters() if decays(n, p)]
    no_decay = [p for n, p in model.named_parameters() if not decays(n, p)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": t.weight_decay},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=t.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_m = masked.validation(t.micro_batch, VAL_BATCHES) if masked else []
    val_n = nextday.validation(NEXT_DAY_BATCH) if nextday else []

    def losses(m_batch: Any, n_batch: Any) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            if m_batch is not None:
                tok = m_batch.tokens.to(device)
                h = model(tok)
                sel = tok.hidden[..., 0]
                out["masked"] = model.masked.loss_bits(h[sel], m_batch.targets.to(device))
            if n_batch is not None:
                tok = n_batch.tokens.to(device)
                h = model(tok)[:, -1]
                bits, n = model.next_day.loss_bits(
                    h,
                    n_batch.recent.to(device),
                    n_batch.counts.to(device),
                    n_batch.cells.to(device),
                    n_batch.mags.to(device),
                    n_batch.day_of.to(device),
                )
                out["next_day"] = bits / torch.clamp(n, min=1.0)
                out["_next_day_bits"], out["_next_day_n"] = bits, n
        return out

    @torch.no_grad()
    def validate() -> dict[str, float]:
        model.eval()
        res: dict[str, float] = {}
        if val_m:
            res["val_masked"] = float(np.mean([losses(b, None)["masked"].item() for b in val_m]))
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
    started = time.perf_counter()
    model.train()
    step = 0
    for step in range(t.max_steps):
        tick = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, config)
        running: dict[str, float] = {}
        for _ in range(t.grad_accum):
            mb = masked.sample(t.micro_batch, rng) if masked else None
            nb = nextday.sample(NEXT_DAY_BATCH, rng) if nextday else None
            parts = losses(mb, nb)
            loss = sum(weights[k] * parts[k] for k in parts if not k.startswith("_"))
            assert isinstance(loss, torch.Tensor)
            scaled: torch.Tensor = scaler.scale(loss / t.grad_accum)
            scaled.backward()  # type: ignore[no-untyped-call]
            for k, part in parts.items():
                if not k.startswith("_"):
                    running[k] = running.get(k, 0.0) + part.item() / t.grad_accum
            tokens_seen += (t.micro_batch * length if mb else 0) + (
                NEXT_DAY_BATCH * length if nb else 0
            )
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_time += time.perf_counter() - tick

        if (step + 1) % t.eval_every == 0 or step + 1 == t.max_steps:
            v = validate()
            score = objective(v)
            row = {"step": float(step + 1), "lr": lr_at(step, config), **running, **v}
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
    }
    return TrainResult({k: v.detach().cpu() for k, v in best_state.items()}, history, stats)
