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


class WorldRegion(_Strict):
    """One region the world Field learns from; empty fields take the region's own terms."""

    mc: float | None = Field(default=None, description="Completeness; else the region's.")
    fit_from: str | None = Field(default=None, description="ETAS fit start; else 20 years back.")


class FieldWorld(_Strict):
    """How one Field learns from many regions at once (P8d): every region is cut into tiles of
    CB-02's size, each tile a place the Field reads like central Italy."""

    regions: dict[str, WorldRegion] = Field(description="Real regions; none for planets only.")
    cut: str = Field(description="Nothing at or after this moment is trained on, anywhere.")
    tile_lat: float = Field(default=3.0, gt=0)
    tile_lon: float = Field(default=3.5, gt=0)
    min_tile_quakes: int = Field(default=300, ge=0, description="M ≥ mc quakes a tile needs.")
    quiet_share: float = Field(default=0.15, gt=0, le=1, description="Quiet days kept.")
    steps_per_epoch: int = Field(default=1000, gt=0)
    max_validation_days: int = Field(default=200, gt=0, description="Kept per tile.")


class FieldPlanets(_Strict):
    """Fake planets as a stage of the curriculum (P8f): simulated catalogues of central Italy's
    box, never the ones CB-09 scores (those are its first seeds under 200)."""

    kind: str = Field(default="physics_planet", description="A CB-09 kind of planet.")
    first_seed: int = Field(default=1000, ge=200, description="Above every CB-09 planet's seed.")
    count: int = Field(default=6, gt=0)
    blind_twins: bool = Field(
        default=True,
        description="Each planet also read through a network that goes blind after big quakes, "
        "answers from the full catalogue, so completeness is something the Field reads.",
    )


class FieldCurriculum(_Strict):
    """Where a Field starts (P8f): every member from a member of an earlier stage's Field,
    trained on more of the Earth, instead of from ETAS alone."""

    start_from: str = Field(description="The Field config whose members this one starts from.")


class FieldConfig(_Strict):
    name: str
    description: str
    model: FieldShape
    memory: FieldMemory
    train: FieldTrain
    world: FieldWorld | None = None
    curriculum: FieldCurriculum | None = None
    planets: FieldPlanets | None = None


def load_field_config(name_or_path: str | Path, root: Path | None = None) -> FieldConfig:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / "field" / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return FieldConfig.model_validate(yaml.safe_load(f))


def field_hash(config: FieldConfig) -> str:
    dump = config.model_dump(mode="json")
    for later in ("world", "curriculum", "planets"):
        if dump.get(later) is None:
            dump.pop(later, None)  # a config from before P8d / P8f keeps its hash and checkpoints
    body = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
