"""Probability distributions over one number, as benchmark answers.

CratonBench never asks for a single number. A method answers a masked token with a distribution
per channel, and the benchmark asks it one question: how much probability did you put on the
bin the truth fell in? Anything with a ``mass(lo, hi)`` method can answer; these are the two the
baselines use. Models add their own (a mixture density head is a sum of Gaussians).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.special import ndtr

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


class ChannelDist(Protocol):
    """One distribution per query, vectorised over the queries."""

    def mass(self, lo: FloatArray, hi: FloatArray) -> FloatArray:
        """Probability of [lo, hi) for each query."""
        ...


@dataclass(frozen=True)
class Gaussian:
    mean: FloatArray
    std: FloatArray

    def mass(self, lo: FloatArray, hi: FloatArray) -> FloatArray:
        # Centre first: a GNSS position is millions of metres, its bins one millimetre.
        a = (lo - self.mean) / self.std
        b = (hi - self.mean) / self.std
        upper = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
        return np.asarray(np.maximum(upper, 0.0), dtype=np.float64)


@dataclass(frozen=True)
class Histogram:
    """Piecewise-uniform density: ``probs[q, i]`` on [edges[i], edges[i+1]).

    ``probs`` is (queries, bins) or (bins,) shared by every query. Mass outside the edges is 0.
    """

    edges: FloatArray
    probs: FloatArray

    def _cdf(self, x: FloatArray) -> FloatArray:
        probs = np.atleast_2d(self.probs)
        cum = np.concatenate([np.zeros((probs.shape[0], 1)), np.cumsum(probs, axis=1)], axis=1)
        pos = np.clip(x, self.edges[0], self.edges[-1])
        i = np.clip(np.searchsorted(self.edges, pos, side="right") - 1, 0, len(self.edges) - 2)
        frac = (pos - self.edges[i]) / (self.edges[i + 1] - self.edges[i])
        rows = np.arange(len(x)) if probs.shape[0] > 1 else np.zeros(len(x), dtype=np.int64)
        out = cum[rows, i] + frac * (cum[rows, i + 1] - cum[rows, i])
        return np.asarray(out, dtype=np.float64)

    def mass(self, lo: FloatArray, hi: FloatArray) -> FloatArray:
        return np.maximum(self._cdf(hi) - self._cdf(lo), 0.0)
