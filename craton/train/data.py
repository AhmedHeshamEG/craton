"""Training examples cut from the catalogue: masked windows and next-day questions.

Two games, both played on windows of consecutive quakes sorted by time:

    masked    take ``T`` quakes in a row, hide where and how big 15% of them were (their time
              and magnitude type stay visible, as in CB-01), ask for them back.
    next_day  stand at a midnight, take the ``T − 1`` quakes before it plus a query token at
              midnight, ask for the next 24 hours: how many, where, how big (CB-02's question).

A share of 30-day blocks, picked by a salted hash, is kept for validation: the loop watches
its loss to know when to stop. Nothing here reads the benchmark answers; every example comes
from one training split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.model.craton import Tokens
from craton.model.encoders import magnitude_type_id
from craton.model.heads import RATE_SCALES

IntArray = np.ndarray[Any, np.dtype[np.int64]]
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

BLOCK_DAYS = 30
VAL_SHARE = 0.1


def in_validation(t: FloatArray, salt: str = "craton-val") -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Whether each moment falls in a validation block (30-day blocks, 10% by salted hash)."""
    block = np.floor(t / (BLOCK_DAYS * DAY)).astype(np.int64)
    out = np.zeros(len(block), dtype=bool)
    for b in np.unique(block):
        h = int(hashlib.sha256(f"{salt}:{b}".encode()).hexdigest()[:8], 16) / 16**8
        out[block == b] = h < VAL_SHARE
    return out


@dataclass(frozen=True)
class Catalog:
    """A catalogue as arrays the batch builders index into, sorted by time."""

    time: FloatArray
    values: FloatArray  # (n, 4) lat, lon, depth, magnitude
    mag_type: IntArray

    @staticmethod
    def of(ev: Events) -> Catalog:
        values = np.stack([ev.lat, ev.lon, ev.depth_km, ev.mag], axis=1)
        types = np.array([magnitude_type_id(str(t)) for t in ev.mag_type], dtype=np.int64)
        return Catalog(ev.time.astype(np.float64), values.astype(np.float64), types)

    def __len__(self) -> int:
        return len(self.time)


def window_tokens(
    cat: Catalog,
    rows: IntArray,
    hidden: np.ndarray[Any, np.dtype[np.bool_]],
    t_ref: FloatArray,
    query_at: FloatArray | None = None,
) -> Tokens:
    """Tokens for windows of catalogue rows. ``rows`` (B, L), ``hidden`` (B, L) marks quakes
    whose place and size are hidden. With ``query_at`` (B,), a query token is appended to each
    window at that moment. Times are relative to ``t_ref`` (B,)."""
    b, n = rows.shape
    times = cat.time[rows]
    prev = np.where(rows[:, :1] > 0, cat.time[np.maximum(rows[:, :1] - 1, 0)], times[:, :1])
    gaps = np.diff(times, axis=1, prepend=prev)
    hid4 = np.repeat(hidden[..., None], 4, axis=-1)
    values = np.where(hid4, 0.0, cat.values[rows])  # a hidden quake's numbers never leave here
    coords = np.concatenate([(times - t_ref[:, None])[..., None], values[..., :3]], -1)
    known = np.ones((b, n, 4), dtype=bool)
    known[..., 1:] = ~hidden[..., None]
    mag_type = cat.mag_type[rows]
    query = np.zeros((b, n), dtype=bool)
    clock = np.stack([np.abs(times - t_ref[:, None]), gaps], axis=-1)
    if query_at is not None:
        values = np.concatenate([values, np.zeros((b, 1, 4))], axis=1)
        hid4 = np.concatenate([hid4, np.ones((b, 1, 4), dtype=bool)], axis=1)
        q_coords = np.zeros((b, 1, 4))
        q_coords[:, 0, 0] = query_at - t_ref
        coords = np.concatenate([coords, q_coords], axis=1)
        q_known = np.zeros((b, 1, 4), dtype=bool)
        q_known[..., 0] = True
        known = np.concatenate([known, q_known], axis=1)
        mag_type = np.concatenate([mag_type, np.zeros((b, 1), dtype=np.int64)], axis=1)
        query = np.concatenate([query, np.ones((b, 1), dtype=bool)], axis=1)
        q_clock = np.stack([np.abs(query_at - t_ref), query_at - times[:, -1]], axis=-1)
        clock = np.concatenate([clock, q_clock[:, None, :]], axis=1)
    return Tokens(
        values=torch.from_numpy(values.astype(np.float32)),
        hidden=torch.from_numpy(hid4),
        mag_type=torch.from_numpy(mag_type),
        coords=torch.from_numpy(coords.astype(np.float64)),
        known=torch.from_numpy(known),
        query=torch.from_numpy(query),
        clock=torch.from_numpy(np.log10(1.0 + np.maximum(clock, 0.0)).astype(np.float32)),
    )


# ---------------------------------------------------------------- masked windows


@dataclass
class MaskedBatch:
    tokens: Tokens
    targets: torch.Tensor  # (N, 4) true values of the hidden quakes, in window order


class MaskedWindows:
    """Random windows of ``length`` quakes, a fresh mask every time."""

    def __init__(self, cat: Catalog, length: int, mask_share: float) -> None:
        if len(cat) < length:
            raise ValueError(f"{len(cat)} quakes is fewer than one window of {length}")
        self.cat, self.length, self.mask_share = cat, length, mask_share
        starts = np.arange(len(cat) - length + 1)
        val = in_validation(cat.time[starts + length // 2])
        self.train_starts, self.val_starts = starts[~val], starts[val]

    def batch(self, starts: IntArray, rng: np.random.Generator) -> MaskedBatch:
        rows = starts[:, None] + np.arange(self.length)[None, :]
        hidden = rng.random(rows.shape) < self.mask_share
        hidden[np.arange(len(starts)), rng.integers(0, self.length, len(starts))] = True
        tok = window_tokens(self.cat, rows, hidden, self.cat.time[rows[:, 0]])
        targets = torch.from_numpy(self.cat.values[rows][hidden].astype(np.float32))
        return MaskedBatch(tok, targets)

    def sample(self, size: int, rng: np.random.Generator) -> MaskedBatch:
        return self.batch(rng.choice(self.train_starts, size), rng)

    def validation(self, size: int, n_batches: int, seed: int = 12345) -> list[MaskedBatch]:
        """A fixed set of validation windows and masks, the same at every check."""
        rng = np.random.default_rng(seed)
        if not len(self.val_starts):
            return []
        picks = rng.choice(self.val_starts, size * n_batches)
        return [self.batch(picks[i * size : (i + 1) * size], rng) for i in range(n_batches)]


# ---------------------------------------------------------------- next-day questions


@dataclass
class NextDayBatch:
    tokens: Tokens
    counts: torch.Tensor  # (B,) quakes in each next day (in the grid, M ≥ grid minimum)
    cells: torch.Tensor  # (N,) cell of every one of those quakes
    mags: torch.Tensor  # (N,) magnitude bin
    day_of: torch.Tensor  # (N,) which row of the batch it belongs to
    recent: torch.Tensor  # (B, scales) log quakes per day just before each midnight


def context_rows(cat: Catalog, day_starts: FloatArray, length: int) -> IntArray:
    """For each midnight, the ``length`` quakes just before it. Needs enough history."""
    first = np.searchsorted(cat.time, day_starts, side="left")
    if np.any(first < length):
        raise ValueError("a midnight has fewer quakes before it than the context needs")
    return np.asarray(first[:, None] - length + np.arange(length)[None, :], dtype=np.int64)


def recent_rates(cat: Catalog, day_starts: FloatArray) -> torch.Tensor:
    """Natural-log rate per day just before each midnight, for every (k, magnitude) in
    ``RATE_SCALES``: k / (age of the k-th last quake at or above that magnitude). Where fewer
    than k such quakes exist, the age runs from the catalogue's first quake. Reads only quakes
    before the midnight."""
    out = np.zeros((len(day_starts), len(RATE_SCALES)))
    for j, (k, m_min) in enumerate(RATE_SCALES):
        t = cat.time[cat.values[:, 3] >= m_min - 1e-9]
        first = np.searchsorted(t, day_starts, side="left")
        oldest = np.where(first >= k, t[np.maximum(first - k, 0)], cat.time[0])
        age = np.maximum(day_starts - oldest, 1.0) / DAY
        out[:, j] = np.log(k / age)
    return torch.from_numpy(out.astype(np.float32))


def query_tokens(cat: Catalog, day_starts: FloatArray, length: int) -> Tokens:
    """Context of ``length − 1`` quakes, then the query at midnight. Times relative to it."""
    rows = context_rows(cat, day_starts, length - 1)
    return window_tokens(cat, rows, np.zeros(rows.shape, dtype=bool), day_starts, day_starts)


class NextDay:
    """Every midnight with enough history and a whole next day inside ``[start, end)``."""

    def __init__(
        self, cat: Catalog, grid: csep.Grid, length: int, start: float, end: float
    ) -> None:
        self.cat, self.grid, self.length = cat, grid, length
        need = max(length - 1, max(k for k, _ in RATE_SCALES))  # quakes needed behind midnight
        t_min = cat.time[need] if len(cat) > need else np.inf
        first_day = max(np.ceil(start / DAY), np.ceil(t_min / DAY)) * DAY
        days = np.arange(first_day, end - DAY + 1e-6, DAY)
        if not len(days):
            raise ValueError("no whole day with enough history in the training window")
        cell = grid.cell_of(cat.values[:, 0], cat.values[:, 1])
        mbin = grid.mag_bin_of(cat.values[:, 3])
        keep = (cell >= 0) & (mbin >= 0)
        self.t, self.cell, self.mbin = cat.time[keep], cell[keep], mbin[keep]
        val = in_validation(days)
        self.train_days, self.val_days = days[~val], days[val]

    def batch(self, days: FloatArray) -> NextDayBatch:
        tok = query_tokens(self.cat, days, self.length)
        lo = np.searchsorted(self.t, days, side="left")
        hi = np.searchsorted(self.t, days + DAY, side="left")
        idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi, strict=True)])
        idx = idx.astype(np.int64)
        day_of = np.repeat(np.arange(len(days)), hi - lo)
        return NextDayBatch(
            tokens=tok,
            counts=torch.from_numpy((hi - lo).astype(np.int64)),
            cells=torch.from_numpy(self.cell[idx].astype(np.int64)),
            mags=torch.from_numpy(self.mbin[idx].astype(np.int64)),
            day_of=torch.from_numpy(day_of.astype(np.int64)),
            recent=recent_rates(self.cat, days),
        )

    def sample(self, size: int, rng: np.random.Generator) -> NextDayBatch:
        return self.batch(rng.choice(self.train_days, size))

    def validation(self, size: int) -> list[NextDayBatch]:
        days = self.val_days
        return [self.batch(days[i : i + size]) for i in range(0, len(days), size)]
