"""The referee for maps: a map that matches the truth has to survive the tests a fluke can't.

`craton referee` (``craton.trust.referee``) was built for clocks: a tidal "discovery" that a
24-hour clock matched just as well. Maps fool the same way. Any smooth map agrees a little with
any other smooth map, and a map that knows only the region's average looks right everywhere
the average is right. A claim ("this inferred map matches that truth") passes only if it
survives all four:

1. **Better than a coin.** Mean angle under 45° for directions; rank correlation above 0.
2. **Better than the floor.** The task's best simple method, on the same cells.
3. **Shifted maps.** The answer slid across the truth, 200 times, by at least two cells in a
   random direction, wrapping around the panel's bounding box (a toroidal shift, Fortin & Dale
   2005), each cell taking the answer of the cell nearest its shifted point. The same map,
   texture and all, with any real alignment broken: the claim must beat 95% of them. A map
   that knows only the regional trend scores the same shifted and fails here.
4. **A locked half.** The cells are split in two by position. The claim's lead over the floor
   on the first half is written down and hashed before the second half is read; on the
   second, the lead must have the same sign.

The verdicts of every map shown in the README are in ``results/referee/maps.json``
(``scripts/referee_maps.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.stats import spearmanr

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

SHIFTS = 200
SHIFT_QUANTILE = 0.95
MIN_SHIFT_CELLS = 2.0
KINDS = ("axial", "rank")


@dataclass(frozen=True)
class MapPanel:
    """One map to compare: a region, a day or a year. Same cells for all three arrays."""

    lat: FloatArray
    lon: FloatArray
    truth: FloatArray
    answer: FloatArray
    floor: FloatArray
    #: Only these cells are graded (CB-33 grades cells off steady); all cells, if None.
    graded: np.ndarray[Any, Any] | None = None


@dataclass(frozen=True)
class MapClaim:
    name: str
    kind: str  # "axial" (degrees, lower error better) or "rank" (Spearman, higher better)
    cell_deg: float
    panels: list[MapPanel]
    statement: str = ""


@dataclass
class MapVerdict:
    name: str
    statement: str
    kind: str
    skill: float  # higher is better: −degrees for axial, ρ for rank
    floor_skill: float
    coin_skill: float
    shifted_quantile: float
    lock_hash: str
    locked_lead: float
    held_out_lead: float
    tests: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.tests.values())

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["passed"] = self.passed
        return out


def axial_error(a: FloatArray, b: FloatArray) -> FloatArray:
    d = np.abs((np.asarray(a, dtype=np.float64) - b + 90.0) % 180.0 - 90.0)
    return np.where(np.isfinite(d), d, 45.0)


def _panel_skill(kind: str, answer: FloatArray, truth: FloatArray) -> float:
    if kind == "axial":
        return -float(axial_error(answer, truth).mean())
    if len(truth) < 3 or np.ptp(answer) == 0:
        return 0.0
    rho = spearmanr(answer, truth).statistic
    return float(rho) if np.isfinite(rho) else 0.0


def _sel(p: MapPanel, arr: FloatArray, keep: np.ndarray[Any, Any] | None = None) -> FloatArray:
    m = np.ones(len(p.lat), dtype=bool) if p.graded is None else p.graded
    if keep is not None:
        m = m & keep
    return np.asarray(arr[m], dtype=np.float64)


def skill(claim: MapClaim, answers: list[FloatArray], keep: list[Any] | None = None) -> float:
    """Mean over panels of the panel's skill (only its graded cells, and ``keep`` if given)."""
    vals = []
    for i, (p, a) in enumerate(zip(claim.panels, answers, strict=True)):
        k = None if keep is None else keep[i]
        t = _sel(p, p.truth, k)
        if len(t) == 0:
            continue
        vals.append(_panel_skill(claim.kind, _sel(p, a, k), t))
    return float(np.mean(vals)) if vals else math.nan


def shifted(p: MapPanel, cell_deg: float, rng: np.random.Generator) -> FloatArray:
    """The answer slid around the panel's bounding box, read at the nearest cell."""
    lat0, lat1 = p.lat.min() - cell_deg / 2, p.lat.max() + cell_deg / 2
    lon0, lon1 = p.lon.min() - cell_deg / 2, p.lon.max() + cell_deg / 2
    h, w = lat1 - lat0, lon1 - lon0
    while True:
        dy, dx = rng.uniform(0, h), rng.uniform(0, w)
        # Distance of the shift on the torus, in cells: at least MIN_SHIFT_CELLS.
        ty, tx = min(dy, h - dy), min(dx, w - dx)
        if math.hypot(ty, tx) >= MIN_SHIFT_CELLS * cell_deg or max(h, w) < 3 * cell_deg:
            break
    sy = (p.lat - lat0 + dy) % h + lat0
    sx = (p.lon - lon0 + dx) % w + lon0
    d = (sy[:, None] - p.lat[None]) ** 2 + (sx[:, None] - p.lon[None]) ** 2
    return np.asarray(p.answer[np.argmin(d, axis=1)], dtype=np.float64)


def halves(p: MapPanel, cell_deg: float) -> np.ndarray[Any, Any]:
    """True for the first half: cells whose (row + column) on the cell grid is even."""
    i = np.floor(p.lat / cell_deg + 1e-9).astype(np.int64)
    j = np.floor(p.lon / cell_deg + 1e-9).astype(np.int64)
    return np.asarray((i + j) % 2 == 0)


def lock_effect(name: str, lead: float) -> str:
    body = json.dumps({"claim": name, "lead_on_first_half": round(lead, 6)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def judge(claim: MapClaim, seed: int = 0) -> MapVerdict:
    if claim.kind not in KINDS:
        raise ValueError(f"map kind must be one of {KINDS}")
    rng = np.random.default_rng(seed)
    answers = [p.answer for p in claim.panels]
    floors = [p.floor for p in claim.panels]
    s, f = skill(claim, answers), skill(claim, floors)
    coin = -45.0 if claim.kind == "axial" else 0.0
    fake = np.array(
        [
            skill(claim, [shifted(p, claim.cell_deg, rng) for p in claim.panels])
            for _ in range(SHIFTS)
        ]
    )
    quantile = float((fake < s).mean())
    first = [halves(p, claim.cell_deg) for p in claim.panels]
    second = [~h for h in first]
    lead_a = skill(claim, answers, first) - skill(claim, floors, first)
    lock = lock_effect(claim.name, lead_a)  # written down before the second half is read
    lead_b = skill(claim, answers, second) - skill(claim, floors, second)
    tests = {
        "beats_coin": s > coin,
        "beats_floor": s > f,
        "beats_shifted_maps": quantile >= SHIFT_QUANTILE,
        "locked_half_holds": bool(np.sign(lead_b) == np.sign(lead_a) and lead_a != 0),
    }
    return MapVerdict(
        claim.name, claim.statement, claim.kind, s, f, coin, quantile, lock, lead_a, lead_b, tests
    )
