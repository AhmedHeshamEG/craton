"""The Field's config: ``configs/field/<size>.yaml``, separate from the trunk's."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from craton.paths import configs_dir, find_root


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FieldShape(_Strict):
    d_model: int = Field(gt=0, description="Features per forecast cell.")
    n_layers: int = Field(ge=0, description="Attention layers between cells (0 = none).")
    n_heads: int = Field(gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    max_nudge: float = Field(gt=1, description="Largest factor the Field may move ETAS by.")


class FieldMemory(_Strict):
    m_min: float = Field(description="Smallest magnitude the memory counts.")
    start: str = Field(description="When the counters start (ISO 8601, UTC).")


class FieldTrain(_Strict):
    first_day: str
    validation_every: int = Field(gt=1)
    days_per_step: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    members: int = Field(gt=0, description="Fields averaged into one forecast.")
    seeds: tuple[int, ...] = Field(min_length=3)


class FieldConfig(_Strict):
    name: str
    description: str
    model: FieldShape
    memory: FieldMemory
    train: FieldTrain


def load_field_config(name_or_path: str | Path, root: Path | None = None) -> FieldConfig:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / "field" / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return FieldConfig.model_validate(yaml.safe_load(f))


def field_hash(config: FieldConfig) -> str:
    body = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
