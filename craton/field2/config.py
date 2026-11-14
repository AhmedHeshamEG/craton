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
    # (P11, redrawn) The rebuilt heart: each part behind its own switch. All off = the first
    # heart, number for number (``craton.field2.place``).
    state: int = Field(
        default=0, ge=0, description="Channels of each place's learned state; 0 = no state."
    )
    state_below_mc: float = Field(
        default=0.5, ge=0, description="The state hears quakes from mc minus this."
    )
    neighbours: int = Field(default=0, ge=0, description="Rounds of places reading neighbours.")
    rank: int = Field(
        default=2, ge=0, description="How many ways a place's state bends each bin's weight."
    )
    rulers: bool = Field(
        default=False,
        description="Time in the region's Omori c, rates over its background: no context.",
    )
    smooth_clock: bool = Field(
        default=False, description="Weights blended between neighbouring ages; no jumps."
    )
    additive: bool = Field(
        default=False, description="A plus beside the times: quakes a place's state adds."
    )
    excitement: bool = Field(
        default=False, description="The state also reads ETAS's rate over the background."
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
    # (P11, redrawn) Train as tested.
    as_tested: bool = Field(
        default=False,
        description="Each day forecast by an ETAS fitted before its block (craton.field2.tested).",
    )
    origin_years: float = Field(default=2.0, gt=0, description="Length of one block.")
    stop_outside: bool = Field(
        default=False,
        description="Each member stops early on a training region it does not train on.",
    )


class SchemeNet(_Strict):
    """(P11, the converged scheme, 2026-11-13 evening) Tanks per place, attention between
    places with a relative-distance bias, and the ETAS line with a gain, a recent-quake
    network and a background, all born silent (``craton.field2.scheme``)."""

    tanks: int = Field(default=16, gt=0, description="Count tanks per place (and as many size).")
    tau_min_days: float = Field(default=1.0 / 24.0, gt=0, description="Shortest lifetime at birth.")
    tau_max_days: float = Field(default=10957.0, gt=0, description="Longest lifetime at birth.")
    learn_lifetimes: bool = Field(
        default=True, description="Lifetimes learned; off = a fixed log-spaced grid."
    )
    hears_below_mc: float = Field(default=0.5, ge=0, description="Tanks hear from mc minus this.")
    max_age_days: float = Field(
        default=10957.0, gt=0, description="Quakes older than this no longer fill a tank."
    )
    d_model: int = Field(default=128, gt=0)
    n_layers: int = Field(default=3, ge=0)
    n_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    recent_days: float = Field(
        default=10.0, gt=0, description="Quakes younger than this also pass through N."
    )
    n_hidden: int = Field(default=64, gt=0, description="Width of N, the recent-quake MLP.")
    max_nudge: float = Field(
        default=4.0, gt=1, description="g, N and b each move ETAS at most this many times."
    )
    gain: bool = Field(default=True, description="g(s): one gain per place on the ETAS line.")
    recent: bool = Field(default=True, description="N: the free function of the recent past.")
    background: bool = Field(default=True, description="b(s): a quiet place may speak.")
    distance_bias: bool = Field(
        default=True, description="Attention biased by relative distance (cells, rupture lengths)."
    )
    neighbours: list[int] = Field(
        default_factory=list,
        description="(P11 second run) the tanks also read pooled over k x k squares of places.",
    )
    reads_etas: bool = Field(
        default=False,
        description="(P11 second run) each place reads ETAS's own rate there: over its quiet, "
        "and its share of the box.",
    )


class Field2Config(_Strict):
    name: str
    description: str = ""
    world_config: str = Field(
        description="The world Field config whose tiles and pool days it reads."
    )
    bins: HeartBins
    net: HeartNet
    train: HeartTrain
    scheme: SchemeNet | None = Field(
        default=None, description="(P11) The converged scheme in place of the heart."
    )


_P11_REDRAW = {
    "net": (
        "state",
        "state_below_mc",
        "neighbours",
        "rank",
        "rulers",
        "smooth_clock",
        "additive",
        "excitement",
    ),
    "train": ("as_tested", "origin_years", "stop_outside"),
}


def rebuilt(net: HeartNet) -> bool:
    """Does this heart use any part of the P11 redraw (``craton.field2.place``)?"""
    return bool(
        net.state
        or net.neighbours
        or net.rulers
        or net.smooth_clock
        or net.additive
        or net.excitement
    )


def load_field2_config(name_or_path: str | Path, root: Path | None = None) -> Field2Config:
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = configs_dir(root or find_root()) / "field2" / f"{name_or_path}.yaml"
    with path.open(encoding="utf-8") as f:
        return Field2Config.model_validate(yaml.safe_load(f))


_SCHEME_SECOND = ("neighbours", "reads_etas")


def field2_hash(config: Field2Config) -> str:
    dump = config.model_dump(mode="json")
    dump.pop("description", None)
    if dump.get("scheme") is None:
        dump.pop("scheme", None)  # the hearts keep the names they had
    else:
        # switches added after the first scheme run keep its name when off
        for key in _SCHEME_SECOND:
            if dump["scheme"].get(key) == SchemeNet.model_fields[key].get_default(
                call_default_factory=True
            ):
                dump["scheme"].pop(key)
    if not config.net.free_head:
        # a heart without the free head keeps the name it had before the switch existed
        dump["net"].pop("free_head")
        dump["net"].pop("free_magnitude")
    if config.net.context:
        dump["net"].pop("context")
    # switches added in the P11 redraw keep the first heart's names when off
    for section, model in (("net", HeartNet), ("train", HeartTrain)):
        for key in _P11_REDRAW[section]:
            if dump[section].get(key) == model.model_fields[key].default:
                dump[section].pop(key)
    return hashlib.sha256(json.dumps(dump, sort_keys=True).encode()).hexdigest()[:12]
