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


class FieldHidden(_Strict):
    """The hidden-state layer (P9c): two blocks that start as the physics. Off, each block is
    the P6 Field exactly; on, its last layer starts at zero, so the untrained block is too."""

    clock: bool = Field(
        default=False,
        description="Clock-shift memory: a learned per-cell, per-day shift of the counter "
        "lifetimes along log-time (how fast time runs there: the stressing rate).",
    )
    max_shift: float = Field(
        default=2.0, gt=0, description="Largest shift, in steps of the lifetime bank (4x each)."
    )
    steer: bool = Field(
        default=False,
        description="Steerable kernel: a per-cell angle and stretch of the neighbourhood "
        "memory, from a rotation-equivariant head (the angle is CB-23's SHmax answer).",
    )
    max_stretch: float = Field(
        default=0.8, gt=0, lt=1, description="Largest e in the kernel's 1 + e cos 2(phi - theta)."
    )
    steer_sigma_km: float = Field(default=15.0, gt=0, description="The moments' Gaussian width.")


class FieldWiring(_Strict):
    """The wiring (P9d): Omori attention over ETAS's own sum of parents. Its learned scores
    start at zero, so the untrained block is ETAS exactly."""

    attention: bool = Field(
        default=False,
        description="Omori attention: each source cell sends the aftershocks ETAS gives it "
        "(no more, no fewer) to the target cells a learned query-key score picks.",
    )
    d_key: int = Field(default=16, gt=0, description="Size of the queries and keys.")
    max_bias: float = Field(
        default=4.0, gt=1, description="Largest factor one key's weight may move by."
    )


class FieldCamera(_Strict):
    """The camera (P9d): the world apart from one network's eyes (``craton.field.camera``)."""

    world: bool = Field(
        default=False,
        description="Count the world: every quake below its catalogue's completeness at the "
        "time stands for the ones missed, and the M2-3 answers go back through the eyes.",
    )


class FieldAudit(_Strict):
    """The leak audit (P9e): controls that must behave a known way if nothing leaks. Never a
    forecaster for the board; only the ablation scripts read these configs."""

    cut: str | None = Field(
        default=None,
        description="Train (ETAS's fit too) as if the split ended here, earlier than its own "
        "cut; the test window stays the task's.",
    )
    shuffle_answers: bool = Field(
        default=False,
        description="Hand every training day another day's answers (whole days, one fixed "
        "permutation per member seed): a Field that still gains learned no rule in time.",
    )


class FieldMeter(_Strict):
    """The stress meter (P9f, ``craton.field.meter``): one stress per cell per day is all the
    Field outputs; rates follow from it through rate-and-state. Off, the P6 Field."""

    stress: bool = Field(
        default=False,
        description="A Dieterich bottleneck: the cell's stress anomaly over ETAS's moves every "
        "magnitude's rate alike, and quakes beyond ETAS's use up the cell's faults.",
    )
    t_a_years: float = Field(
        default=5.0,
        gt=0,
        description="The aftershock duration t_a = A sigma / stressing rate "
        "every cell starts from, years.",
    )
    t_a_factor: float = Field(
        default=8.0, gt=1, description="How far each cell's learned t_a may move either way."
    )
    deplete: bool = Field(
        default=True,
        description="Quakes beyond ETAS's use up the cell's faults (off: one stress for every "
        "magnitude, nothing else; the ablation of the two halves).",
    )
    max_stress: float = Field(
        default=3.0, gt=0, description="Largest stress anomaly, in units of A sigma."
    )


class FieldConfig(_Strict):
    name: str
    description: str
    model: FieldShape
    memory: FieldMemory
    train: FieldTrain
    world: FieldWorld | None = None
    curriculum: FieldCurriculum | None = None
    planets: FieldPlanets | None = None
    hidden: FieldHidden | None = None
    wiring: FieldWiring | None = None
    camera: FieldCamera | None = None
    audit: FieldAudit | None = None
    meter: FieldMeter | None = None


def load_field_config(name_or_path: str | Path, root: Path | None = None) -> FieldConfig:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / "field" / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return FieldConfig.model_validate(yaml.safe_load(f))


def field_hash(config: FieldConfig) -> str:
    dump = config.model_dump(mode="json")
    for later in ("world", "curriculum", "planets", "hidden", "wiring", "camera", "audit", "meter"):
        if dump.get(later) is None:
            # a config from before P8d / P8f / P9c / P9d / P9e / P9f keeps its hash and checkpoints
            dump.pop(later, None)
    meter = dump.get("meter")
    if meter is not None and meter.get("deplete") is True:
        meter.pop("deplete")  # added after the first meter run; on (the default) keeps its hash
    body = json.dumps(dump, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
