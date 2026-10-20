"""Model-size configs.

One codebase, three sizes, and the only difference between them is a YAML file. Each config
declares the parameter range it claims to be in, and loading fails if the shape doesn't add
up, so "mini" can never quietly grow into something the laptop can't hold.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from craton.hardware import Precision
from craton.paths import configs_dir, find_root

Status = Literal["active", "planned"]
Positional = Literal["rope4d", "sinusoidal"]
#: Training objectives a config may weight.
OBJECTIVES = ("masked", "gnss", "next_day")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelShape(_Strict):
    d_model: int = Field(gt=0, description="Width of every token vector (features per token).")
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    d_ff: int | None = Field(default=None, gt=0, description="Feed-forward width; 4 x d_model.")
    context_tokens: int = Field(gt=0, description="How many tokens the model sees at once.")
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    positional: Positional = Field(
        default="rope4d", description="4D rotary encoding, or the sinusoidal fallback."
    )
    inputs: tuple[str, ...] = Field(
        default=("seismic_catalog",),
        min_length=1,
        description="Instruments the model reads, by registry name (one encoder each).",
    )

    @model_validator(mode="after")
    def _heads_divide_width(self) -> ModelShape:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        return self

    @property
    def ff_width(self) -> int:
        return self.d_ff if self.d_ff is not None else 4 * self.d_model


class ContextPolicy(_Strict):
    """How a window of ``context_tokens`` is filled (``craton.train.context``)."""

    tides: int = Field(default=25, ge=0, description="Hourly tide samples around the anchor.")
    tides_ahead: int = Field(
        default=24, ge=0, description="Hours of tide after midnight in a next-day window."
    )
    gnss: int = Field(default=160, ge=0, description="GNSS days in a window, at most.")
    gnss_quake_window: int = Field(
        default=48, ge=0, description="GNSS days in a window about quakes: quakes keep the room."
    )
    gnss_days: int = Field(default=7, ge=0, description="Days either side of the anchor day.")
    gnss_stations: int = Field(default=12, gt=0, description="Nearest stations, at most.")
    gnss_km: float = Field(default=100.0, gt=0, description="Farthest station from the anchor.")
    next_day_gnss_days: int = Field(default=5, ge=0, description="Days before midnight.")
    big_quakes: int = Field(
        default=32, ge=0, description="Big quakes of the past year kept in a next-day window."
    )
    big_magnitude: float = Field(default=3.5, description="What counts as big for that.")


class TrainSettings(_Strict):
    precision: Precision = "fp16"
    micro_batch: int = Field(gt=0)
    grad_accum: int = Field(default=1, gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    seeds: tuple[int, ...] = Field(min_length=3, description="Benchmarks average over these.")
    max_steps: int = Field(default=2000, gt=0, description="Optimiser steps at most.")
    warmup_steps: int = Field(default=100, ge=0)
    eval_every: int = Field(default=100, gt=0, description="Steps between validation passes.")
    patience: int = Field(default=4, gt=0, description="Validations without progress, then stop.")
    mask_share: float = Field(default=0.15, gt=0, lt=1, description="Share of quakes hidden.")
    windows: dict[str, float] = Field(
        default_factory=lambda: {"quake": 0.55, "gnss": 0.45},
        description="How often each kind of masked window is drawn (craton.train.data).",
    )
    split_windows: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="A different mix for some splits: each game trained where it is tested.",
    )

    def windows_for(self, split: str) -> dict[str, float]:
        return self.split_windows.get(split, self.windows)

    objectives: dict[str, float] = Field(
        default_factory=lambda: {"masked": 1.0, "next_day": 1.0},
        description="Training objectives and their weights.",
    )

    @model_validator(mode="after")
    def _distinct_seeds(self) -> TrainSettings:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        unknown = set(self.objectives) - set(OBJECTIVES)
        if unknown:
            raise ValueError(f"unknown objectives: {sorted(unknown)}")
        if not any(w > 0 for w in self.objectives.values()):
            raise ValueError("at least one objective needs a positive weight")
        return self


class CratonConfig(_Strict):
    name: str
    status: Status
    description: str
    target_hardware: str
    params_range: tuple[int, int] = Field(description="Claimed trunk size, [low, high].")
    model: ModelShape
    train: TrainSettings
    context: ContextPolicy = Field(default_factory=ContextPolicy)

    @model_validator(mode="after")
    def _size_is_honest(self) -> CratonConfig:
        low, high = self.params_range
        n = trunk_params(self.model)
        if not low <= n <= high:
            raise ValueError(
                f"config '{self.name}' claims {low:,}-{high:,} parameters but its shape has "
                f"{n:,}. Fix the shape or the claim."
            )
        return self

    @property
    def params(self) -> int:
        return trunk_params(self.model)

    @property
    def effective_batch(self) -> int:
        return self.train.micro_batch * self.train.grad_accum


def trunk_params(shape: ModelShape) -> int:
    """Learnable parameters in the transformer trunk.

    Per layer: attention has four d x d matrices (query, key, value, output) plus biases; the
    feed-forward block has d x d_ff in and d_ff x d out plus biases; two layer norms have a
    scale and a shift each. With d_ff = 4d that is about 12 d^2 per layer, the rule of thumb.
    Encoders and heads are counted separately once they exist (P4).
    """
    d, f = shape.d_model, shape.ff_width
    attention = 4 * d * d + 4 * d
    feed_forward = 2 * d * f + f + d
    norms = 4 * d
    return shape.n_layers * (attention + feed_forward + norms) + 2 * d  # + the final norm


def load_config(name_or_path: str | Path, root: Path | None = None) -> CratonConfig:
    """Load ``configs/<name>.yaml`` (or a path to any YAML) and validate it."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CratonConfig.model_validate(data)


def available_configs(root: Path | None = None) -> list[str]:
    return sorted(p.stem for p in configs_dir(root or find_root()).glob("*.yaml"))


def config_hash(config: CratonConfig) -> str:
    """Fingerprint of a config's content, recorded with every checkpoint and result."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
