"""CB-23 through the Field: the steerable kernel's angle, read as the direction of SHmax.

The steer head (``craton.field.hidden``) was trained for one thing only: forecasting central
Italy's quakes. It reads nothing but a cell's own counters and the quadrupole moments of the
quakes around it, never where the cell is, so it runs on any catalogue anywhere. For a CB-23
region this module:

1. lays 0.1° cells over the region's scored cells (with a margin, so moments near the edge
   still see their neighbours),
2. builds the memory and the moments at the cut, from the history CB-23 allows (quakes before
   the cut, at or above the region's completeness, which is also the productivity
   counters' unit there),
3. runs every member's steer head and adds the stretch vectors w of the 0.1° cells inside each
   0.5° cell and over the members, each cell weighted by its own quakes (the twelve-year
   counter). A cell beside a fault sees the fault side-on, so its moments point across it;
   the cells the quakes happen in are the ones whose forecasts taught the head (fixed before
   any CB-23 score was seen; with no quakes inside, the cells count equally),
4. answers half the angle of the sum. A cell with no stretch at all gets no answer (NaN,
   charged as a coin's 45°).

No WSM reading, no focal mechanism and no stress label is ever read: the angle is whatever
the forecaster found useful, and CB-23 asks whether that is the crust's squeeze.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton.bench.csep import Grid
from craton.bench.stress import CELL_DEG, StressQuery
from craton.field.config import FieldConfig, field_hash, load_field_config
from craton.field.forecaster import FIELD_VERSION
from craton.field.hidden import HiddenFieldNet, K, W
from craton.field.memory import EPS, MemorySpec, memory
from craton.field.steer import azimuth_of, quadrupole
from craton.paths import find_root
from craton.spec.timeaxis import to_seconds
from craton.train.checkpoint import checkpoints_dir

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

FINE = 0.1
MARGIN_DEG = 0.5
#: The Field whose steer head answers: the one trained for this task.
TRAINED_ON = "CB-02"


def steer_states(cfg: FieldConfig, seed: int, root: Path | None = None) -> list[dict[str, Any]]:
    """The members of ``cfg``'s Field trained for ``TRAINED_ON`` with ``seed``."""
    pattern = f"field-{cfg.name}-{FIELD_VERSION}-{TRAINED_ON}-*-{field_hash(cfg)[:8]}-s{seed}"
    found = sorted(checkpoints_dir(root or find_root()).glob(pattern))
    if not found:
        raise FileNotFoundError(f"no {pattern}: train it first (its CB-02 forecast does)")
    body = torch.load(found[-1] / "model.pt", map_location="cpu", weights_only=True)
    return list(body["members"])


def fine_grid(lat: FloatArray, lon: FloatArray) -> Grid:
    """0.1° cells over the 0.5° query cells and a margin, edges on the 0.1° lattice."""
    half = CELL_DEG / 2 + MARGIN_DEG

    def lo(v: float) -> float:
        return round(math.floor(round(v / FINE, 6)) * FINE, 6)

    def hi(v: float) -> float:
        return round(math.ceil(round(v / FINE, 6)) * FINE, 6)

    return Grid(
        lo(float(lat.min()) - half),
        hi(float(lat.max()) + half),
        lo(float(lon.min()) - half),
        hi(float(lon.max()) + half),
        cell_deg=FINE,
    )


def stretch_map(
    states: list[dict[str, Any]], cfg: FieldConfig, query: StressQuery, grid: Grid
) -> tuple[FloatArray, FloatArray]:
    """(members, fine cells, 2) every member's stretch w at the cut, and (fine cells,) each
    cell's own quakes on the twelve-year counter."""
    assert cfg.hidden is not None
    assert cfg.hidden.steer
    spec = MemorySpec(m_min=query.mc, start=to_seconds(cfg.memory.start), m_ref=query.mc)
    at = np.array([to_seconds(query.cut)])
    mem = torch.from_numpy(memory(query.history, grid, at, spec)).float()
    quad = torch.from_numpy(quadrupole(query.history, grid, at, spec, cfg.hidden.steer_sigma_km))
    out = []
    for st in states:
        net = HiddenFieldNet(mem.shape[-1], Grid(0, 1, 0, 1), cfg.model, cfg.hidden)
        steer = {k[len("steer.") :]: v for k, v in st.items() if k.startswith("steer.")}
        net.steer.load_state_dict(steer)
        with torch.no_grad():
            w = net.eval().stretch(mem, quad)[0].double().numpy()
        out.append(w)
    own = np.exp(mem[0, :, (K - 1) * W].double().numpy()) - EPS  # 12 years, every quake 1
    return np.stack(out), np.maximum(own, 0.0)


@dataclass
class FieldStress:
    """A StressMapper for CB-23 from a steering Field config's trained members."""

    config_name: str = "hidden_steer"
    name: str = "field-steer"
    kind: str = "model"

    def stress_azimuths(self, query: StressQuery, seed: int) -> FloatArray:
        cfg = load_field_config(self.config_name)
        grid = fine_grid(query.lat, query.lon)
        members, own = stretch_map(steer_states(cfg, seed), cfg, query, grid)
        w = members.sum(axis=0)  # (cells, 2)
        clat, clon = grid.cell_centres()
        out = np.full(len(query.lat), math.nan)
        h = CELL_DEG / 2
        for k, (la, lo) in enumerate(zip(query.lat, query.lon, strict=True)):
            inside = (np.abs(clat - la) < h) & (np.abs(clon - lo) < h)
            weight = own[inside] if own[inside].sum() > 0 else np.ones(int(inside.sum()))
            v = (w[inside] * weight[:, None]).sum(axis=0)
            if np.hypot(v[0], v[1]) > 1e-9:
                out[k] = float(azimuth_of(v))
        return out
