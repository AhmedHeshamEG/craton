"""The guard: input hygiene between the raw catalogue and a forecast.

Two glitches are common in real catalogues and look, to a naive model, like a sequence
getting going:

* **Duplicates.** Two networks merged without care list the same quake twice, seconds apart.
  The guard drops a quake if an earlier one lies within 5 s, 3 km and 0.3 magnitude units.
* **A magnitude shift.** A miscalibrated network reads every quake a few tenths too high. A
  shift moves the whole magnitude distribution at once while the number of quakes stays the
  same. A real sequence is the opposite: thousands more quakes, and the small ones lost in the
  coda of the big ones. So, per 5-day stretch, the guard compares the stretch's magnitudes
  with the year before it. If the 25th, 50th and 75th percentiles all jumped by 0.2 or more
  (the median by 0.25 or more), with at least 20 quakes and no more than twice the usual
  number, and the jump is shared across the map (in at least two-thirds of the 0.5° blocks
  holding three or more of the stretch's quakes, at least three such blocks, each block's
  median up by 0.2 against its own year), the median jump is read as a calibration error and
  taken off the stretch. Either test alone fires about once a year on the real catalogue;
  together, never (``guard_report``).

The guard watches the **live edge**: only quakes after ``since`` are inspected (the forecast
window and the 60 days before it), because that is where glitches live; a catalogue's past
has been reviewed. On the real catalogue the guard drops 7 quakes before CB-02 and 3 before
CB-03 as duplicates and shifts nothing in either test window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from craton.bench.data import DAY, Events

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

DUP_SECONDS = 5.0
DUP_KM = 3.0
DUP_MAG = 0.3
STRETCH_DAYS = 5
REFERENCE_DAYS = 365
MIN_QUAKES = 20
QUARTILE_JUMP = 0.2
SHIFT_FLAG = 0.25
COUNT_FACTOR = 2.0
BLOCK_DEG = 0.5
BLOCK_MIN = 3
MIN_BLOCKS = 3
AGREE_SHARE = 2 / 3
LIVE_EDGE_DAYS = 60


@dataclass(frozen=True)
class GuardReport:
    duplicates: int
    shifted_stretches: int
    shifted_quakes: int


def duplicate_mask(ev: Events) -> np.ndarray:
    """True for every quake that repeats an earlier one (time-sorted input)."""
    n = len(ev)
    drop = np.zeros(n, dtype=bool)
    j0 = 0
    coslat = np.cos(np.radians(ev.lat))
    for i in range(1, n):
        while ev.time[i] - ev.time[j0] > DUP_SECONDS:
            j0 += 1
        if j0 >= i:
            continue
        j = np.arange(j0, i)
        j = j[~drop[j]]
        if not len(j):
            continue
        dy = (ev.lat[j] - ev.lat[i]) * 111.195
        dx = (ev.lon[j] - ev.lon[i]) * 111.195 * coslat[i]
        close = (np.hypot(dx, dy) <= DUP_KM) & (np.abs(ev.mag[j] - ev.mag[i]) <= DUP_MAG + 1e-9)
        drop[i] = bool(close.any())
    return drop


def magnitude_shifts(ev: Events, since: float) -> tuple[FloatArray, int]:
    """(correction to add to each quake's magnitude, stretches flagged), for quakes after
    ``since``."""
    corr = np.zeros(len(ev))
    if not len(ev):
        return corr, 0
    mag = ev.mag.copy()  # corrected as we go, so a shifted stretch never becomes a reference
    block = np.floor(ev.lat / BLOCK_DEG).astype(np.int64) * 1_000 + np.floor(
        ev.lon / BLOCK_DEG
    ).astype(np.int64)
    first = int(np.floor(since / (STRETCH_DAYS * DAY)))
    last = int(np.floor(ev.time[-1] / (STRETCH_DAYS * DAY)))
    q = (0.25, 0.5, 0.75)
    flagged = 0
    for st in range(first, last + 1):
        t0 = st * STRETCH_DAYS * DAY
        edges = [t0 - REFERENCE_DAYS * DAY, t0, t0 + STRETCH_DAYS * DAY]
        i0, i1, i2 = (int(i) for i in np.searchsorted(ev.time, edges))
        n_here, n_ref = i2 - i1, i1 - i0
        usual = n_ref * STRETCH_DAYS / REFERENCE_DAYS
        if n_here < MIN_QUAKES or n_ref < 2 * MIN_QUAKES or n_here > COUNT_FACTOR * usual:
            continue
        jump = np.quantile(mag[i1:i2], q) - np.quantile(mag[i0:i1], q)
        if not (np.all(jump >= QUARTILE_JUMP - 1e-9) and jump[1] >= SHIFT_FLAG - 1e-9):
            continue
        if _shared(block[i0:i2], mag[i0:i2], i1 - i0):
            corr[i1:i2] = -np.round(jump[1], 1)
            mag[i1:i2] += corr[i1:i2]
            flagged += 1
    return corr, flagged


def _shared(block: np.ndarray, mag: FloatArray, n_ref: int) -> bool:
    """Whether most blocks of the stretch (after the first ``n_ref`` reference quakes) moved."""
    ref_b, ref_m = block[:n_ref], mag[:n_ref]
    here_b, here_m = block[n_ref:], mag[n_ref:]
    moved = []
    for b in np.unique(here_b):
        mh, mr = here_m[here_b == b], ref_m[ref_b == b]
        if len(mh) >= BLOCK_MIN and len(mr) >= 2 * BLOCK_MIN:
            moved.append(float(np.median(mh) - np.median(mr)) >= QUARTILE_JUMP - 1e-9)
    return len(moved) >= MIN_BLOCKS and float(np.mean(moved)) >= AGREE_SHARE


def clean(ev: Events, since: float) -> tuple[Events, GuardReport]:
    """The catalogue with duplicates dropped and magnitude shifts taken off, after ``since``."""
    live = ev.time >= since
    dup = np.zeros(len(ev), dtype=bool)
    dup[live] = duplicate_mask(ev.select(live))
    out = ev.select(~dup)
    corr, flagged = magnitude_shifts(out, since)
    if flagged:
        out = replace(out, mag=np.round(out.mag + corr, 1))
    return out, GuardReport(int(dup.sum()), flagged, int(np.count_nonzero(corr)))
