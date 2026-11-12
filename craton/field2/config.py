"""Field 2's config: ``configs/field2/<name>.yaml``.

The heart (P11) has three groups of settings: the bins its counting attention reads (``bins``),
the small network that weighs them (``net``) and how it learns (``train``). The world it learns
from (regions, cut, tiles, pool days) is the world Field's, named by ``world_config``, so both
models read exactly the same tile-days and their scores can be compared day for day.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from craton.paths import configs_dir, find_root


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HeartBins(_Strict):
    """Lower edges of the bins; the last bin of each runs to infinity."""

    age_days: tuple[float, ...] = Field(
        description="A parent's age at the forecast's midnight (days): the log clock."
    )
    magnitude: tuple[float, ...] = Field(description="m − mc of the parent.")
    ruler: tuple[float, ...] = Field(
        description="Distance from the parent in its own rupture lengths (Wells-Coppersmith)."
    )


class HeartNet(_Strict):
    hidden: int = Field(gt=0, description="Width of the network that weighs each bin.")
    layers: int = Field(ge=1)
    cap: float = Field(gt=0, description="Largest |log weight| on one bin (e^cap either way).")
    context: bool = Field(
        default=True, description="Read the region's ETAS numbers; off = one rule for the Earth."
    )
    free_head: bool = Field(
        default=False, description="A second counting head with a kernel of its own (P11)."
    )
    free_magnitude: tuple[float, ...] = Field(
        default=(-0.5, 0.0, 0.5, 1.0, 2.0, 3.0),
        description="m − mc bins of the free head; it hears quakes from the first edge up.",
    )


class HeartTrain(_Strict):
    days_per_step: int = Field(gt=0, description="Days of one tile in a step.")
    tiles_per_step: int = Field(default=1, gt=0, description="Tiles in one step.")
    steps_per_epoch: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    members: int = Field(gt=0, description="Hearts averaged into one forecast.")
    seeds: tuple[int, ...] = Field(min_length=3)


class Field2Config(_Strict):
    name: str
    description: str = ""
    world_config: str = Field(
        description="The world Field config whose tiles and pool days it reads."
    )
    bins: HeartBins
    net: HeartNet
    train: HeartTrain


def load_field2_config(name_or_path: str | Path, root: Path | None = None) -> Field2Config:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / "field2" / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return Field2Config.model_validate(yaml.safe_load(f))


def field2_hash(config: Field2Config) -> str:
    dump = config.model_dump(mode="json")
    dump.pop("description", None)
    if not config.net.free_head:
        # a heart without the free head keeps the name it had before the switch existed
        dump["net"].pop("free_head")
        dump["net"].pop("free_magnitude")
    if config.net.context:
        dump["net"].pop("context")
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest()[:12]
