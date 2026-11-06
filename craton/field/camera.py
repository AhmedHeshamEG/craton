"""The camera (P9d): what happened (the world) apart from what one network saw (the eyes).

A seismic network doesn't see every quake. Its completeness magnitude ``mc`` is the size
above which it catches them all, and it moves: down over the years as stations are added,
and up by one or two units for hours to days after a big shock, when the records of a
thousand aftershocks overlap and the small ones drown (short-term aftershock
incompleteness). A forecaster that reads raw counts learns one network's blindness as if it
were the crust. Give it a better network, say a machine-learning catalogue that sees three
times as many small quakes after Norcia, and it reads the extra quakes as extra danger
(CB-20, P8c: the Field double-counted).

The camera splits the two:

- **The eyes** are read off the catalogue itself, causally: at each moment, ``mc`` is the
  maximum-curvature magnitude of the last ``WINDOW`` quakes before it, plus ``MAXC_SHIFT``
  (Woessner & Wiemer 2005: maximum curvature runs low by about 0.2). Nothing after the
  moment is used, and nothing is fitted: a new network brings its own eyes.
- **The world** is what the eyes imply was there, counted down to the memory's floor
  (M1.5). Above ``mc`` the network missed nothing, so the quakes it saw there are a fair
  sample of the sizes: by Gutenberg-Richter, each stands for ``10^(b (mc − floor))``
  quakes at or above the floor (at most ``MAX_GAP`` units' worth). Between the floor and
  ``mc`` the network saw only some, and those already stand inside that weight, so they
  count zero. When ``mc`` is at or below the floor every weight is exactly one. This is
  Horvitz-Thompson counting with the eyes as the inclusion probability. The memory's
  count counters use it; its productivity counters (10^m per quake) are carried by the big
  quakes every network sees, and stay as they were.
- The training answers are still one network's, so the M2-3 rate the Field is graded on
  goes back through the eyes: its share above ``mc`` that day. M3+ (scored) is left alone.

With the camera off nothing changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from craton.bench.data import Events

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Quakes in the moving window that estimates the completeness.
WINDOW = 200
BIN = 0.1
#: Maximum curvature's usual underestimate (Woessner & Wiemer 2005).
MAXC_SHIFT = 0.2
#: The largest magnitude gap a quake may stand in for: 10^(b · 1) ≈ 10 quakes.
MAX_GAP = 1.0
M_LOW = -1.0
M_HIGH = 8.0


def _bins(mag: FloatArray) -> np.ndarray:
    n = round((M_HIGH - M_LOW) / BIN)
    return np.clip(np.floor((mag - M_LOW) / BIN + 1e-6).astype(np.int64), 0, n - 1)


def completeness_before(
    mag: FloatArray, time: FloatArray, at: FloatArray, device: torch.device | None = None
) -> FloatArray:
    """The completeness at each moment of ``at``: maximum curvature of the ``WINDOW``
    quakes (``mag``, ``time``: the whole catalogue, any order) strictly before it, plus
    ``MAXC_SHIFT``. NaN where fewer than ``WINDOW`` quakes came before."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    order = np.argsort(time, kind="stable")
    t, b = time[order], _bins(mag[order])
    n_bins = round((M_HIGH - M_LOW) / BIN)
    k = np.searchsorted(t, at, side="left")  # quakes before each moment
    out = np.full(len(at), np.nan)
    ok = k >= WINDOW
    if not ok.any():
        return out
    onehot = torch.zeros((len(t) + 1, n_bins), dtype=torch.int32, device=dev)
    onehot[torch.arange(1, len(t) + 1, device=dev), torch.from_numpy(b).to(dev)] = 1
    cum = onehot.cumsum(0)
    del onehot
    kk = torch.from_numpy(k[ok]).to(dev)
    out_ok = np.empty(int(ok.sum()))
    for s in range(0, len(kk), 65_536):
        part = kk[s : s + 65_536]
        hist = cum[part] - cum[part - WINDOW]
        mode = hist.argmax(-1).cpu().numpy()
        out_ok[s : s + len(part)] = M_LOW + mode * BIN + MAXC_SHIFT
    out[ok] = out_ok
    return out


def world_weights(
    events: Events, b: float, floor: float, device: torch.device | None = None
) -> FloatArray:
    """How many quakes at or above ``floor`` each quake of ``events`` stands for, under the
    eyes of the catalogue it came from (``events``: that whole catalogue, any order)."""
    mc = completeness_before(events.mag, events.time, events.time, device)
    edge = np.maximum(np.nan_to_num(mc, nan=floor), floor)
    gap = np.minimum(edge - floor, MAX_GAP)
    w = np.where(events.mag >= edge - 1e-9, 10.0 ** (b * gap), 0.0)
    return np.asarray(w, dtype=np.float64)


def band_seen(mc: FloatArray, b: float, lo: float, hi: float) -> FloatArray:
    """The share of a Gutenberg-Richter band [lo, hi) a network with completeness ``mc``
    records (everything above ``mc``, nothing below), at least 1%."""
    edge = np.clip(np.nan_to_num(mc, nan=lo), lo, hi)
    full = 10.0 ** (-b * lo) - 10.0 ** (-b * hi)
    seen = 10.0 ** (-b * edge) - 10.0 ** (-b * hi)
    return np.asarray(np.maximum(seen / full, 0.01), dtype=np.float64)
