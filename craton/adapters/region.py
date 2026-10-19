"""Regions: which piece of the Earth, which years, which sources.

A region is a YAML file in ``configs/regions/``. Its time window has a fixed end, so two builds
of the same region from the same downloads agree byte for byte; moving the end forward is a new
version of the region file, and a new snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from craton.paths import configs_dir, find_root
from craton.spec.timeaxis import to_seconds


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Box(_Strict):
    """A latitude-longitude rectangle (WGS84 degrees, edges included)."""

    lat_min: float = Field(ge=-90, le=90)
    lat_max: float = Field(ge=-90, le=90)
    lon_min: float = Field(ge=-180, lt=180)
    lon_max: float = Field(ge=-180, lt=180)

    @model_validator(mode="after")
    def _ordered(self) -> Box:
        if not (self.lat_min < self.lat_max and self.lon_min < self.lon_max):
            raise ValueError("box needs lat_min < lat_max and lon_min < lon_max")
        return self

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    @property
    def centre(self) -> tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2


class Region(_Strict):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,47}$")
    description: str
    box: Box
    start: str = Field(description="First moment included, ISO 8601 UTC.")
    end: str = Field(description="First moment excluded, ISO 8601 UTC. Fixed, never 'today'.")
    adapters: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _window(self) -> Region:
        if to_seconds(self.start) >= to_seconds(self.end):
            raise ValueError("start must be before end")
        return self

    @property
    def t_start(self) -> float:
        return to_seconds(self.start)

    @property
    def t_end(self) -> float:
        return to_seconds(self.end)

    def holds(self, lat: float, lon: float, t: float) -> bool:
        """Inside the box and the half-open window [start, end)."""
        return self.box.contains(lat, lon) and self.t_start <= t < self.t_end

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def regions_dir(root: Path) -> Path:
    return configs_dir(root) / "regions"


def available_regions(root: Path | None = None) -> list[str]:
    return sorted(p.stem for p in regions_dir(root or find_root()).glob("*.yaml"))


def load_region(name_or_path: str | Path, root: Path | None = None) -> Region:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = regions_dir(root or find_root()) / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return Region.model_validate(yaml.safe_load(f))
