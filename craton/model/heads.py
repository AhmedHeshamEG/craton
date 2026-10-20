"""Heads: what comes out of the trunk, always as a probability distribution.

Craton never answers with a single number. Two heads exist in v0:

    masked     for a hidden quake: a probability for every bin of latitude, longitude, depth
               and magnitude, the same bins CB-01 scores (BERT's masked-word head, for rock)
    next_day   at midnight: how many quakes tomorrow (a Poisson rate), where (a probability
               per 0.1° cell) and how big (a probability per 0.1 magnitude bin). The expected
               count in a cell and bin is rate x where x how big, exactly what CB-02 asks.

Both losses are measured in bits, and both are the benchmark's own score: a binned head's
cross-entropy is CB-01's bits on those bins, and the next-day loss is CB-02's Poisson
log-likelihood. The model trains on the thing it is judged by.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from craton.bench import csep

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
LN2 = math.log(2.0)
#: A sliver of probability spread evenly over every bin, so no answer is ever impossible:
#: the worst a channel can cost is log2(bins / FLOOR) bits, about 20.
FLOOR = 1e-3


@dataclass(frozen=True)
class Bins:
    """``n`` bins of ``width`` starting at ``lo``: bin k is [lo + k·w, lo + (k+1)·w)."""

    channel: str
    lo: float
    width: float
    n: int

    @property
    def edges(self) -> FloatArray:
        return np.asarray(np.round(self.lo + self.width * np.arange(self.n + 1), 6), np.float64)

    def index(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.floor((x.double() - self.lo) / self.width + 1e-9).long()
        return k.clamp(0, self.n - 1)


#: CB-01's bins over the region (``craton.bench.tasks.CB01_BINS``), with a range for each.
EVENT_BINS = (
    Bins("lat", 41.0, 0.01, 300),
    Bins("lon", 11.5, 0.01, 350),
    Bins("depth_km", -5.0, 1.0, 705),
    Bins("magnitude", -1.05, 0.1, 92),
)


def floored(logits: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of (1 − FLOOR)·softmax + FLOOR/n, computed stably."""
    n = logits.shape[-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    return torch.logaddexp(logp + math.log1p(-FLOOR), torch.full_like(logp, math.log(FLOOR / n)))


class MaskedHead(nn.Module):
    def __init__(self, d_model: int, bins: tuple[Bins, ...] = EVENT_BINS) -> None:
        super().__init__()
        self.bins = bins
        self.proj = nn.ModuleList(nn.Linear(d_model, b.n) for b in bins)

    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        """(N, d) → one (N, bins) log-probability tensor per channel."""
        return [floored(p(h)) for p in self.proj]

    def loss_bits(self, h: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Mean bits per masked token, summed over channels. values (N, channels)."""
        total = torch.zeros((), device=h.device)
        for c, (logp, b) in enumerate(zip(self(h), self.bins, strict=True)):
            target = b.index(values[:, c])
            total = total + F.nll_loss(logp, target) / LN2
        return total


def forecast_grid() -> csep.Grid:
    from craton.bench.tasks import forecast_grid as grid

    return grid()


#: The recent rate is measured from the last k quakes at or above a magnitude before midnight,
#: for each (k, magnitude) here. All sizes at four scales; M3+ at two, because big quakes
#: trigger more aftershocks than small ones (ETAS's productivity law).
RATE_SCALES: tuple[tuple[int, float], ...] = (
    (1, -10.0),
    (10, -10.0),
    (100, -10.0),
    (1000, -10.0),
    (1, 3.0),
    (10, 3.0),
)


class NextDayHead(nn.Module):
    """Reads the query token placed at midnight and forecasts the next 24 hours.

    The log rate has two paths. A straight line from the recent rates (``RATE_SCALES``, in
    log quakes per day) starts the model at "tomorrow looks like now"; the trunk's vector adds
    the correction. Attention averages, so it can't count how crowded its window is, and the
    trunk's final layer norm squashes any sense of "huge"; the straight path keeps both, and
    extrapolates like a power law past the busiest day in the training data.
    """

    def __init__(self, d_model: int, grid: csep.Grid | None = None) -> None:
        super().__init__()
        self.grid = grid or forecast_grid()
        self.rate = nn.Linear(d_model, 1)
        self.recent = nn.Linear(len(RATE_SCALES), 1)
        self.where = nn.Linear(d_model, self.grid.n_cells)
        self.size = nn.Linear(d_model, self.grid.n_mags)
        self._start_from_persistence()

    def _start_from_persistence(self) -> None:
        with torch.no_grad():
            self.recent.weight.zero_()
            self.recent.weight[0, :4] = 0.25  # start from the all-sizes rate
            self.recent.bias.zero_()

    def forward(
        self, h: torch.Tensor, recent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (B, d), recent (B, scales) log rates → log rate (B,), log p(cell) (B, cells),
        log p(magnitude bin) (B, mags)."""
        log_rate = self.rate(h).float().squeeze(-1) + self.recent(recent.float()).squeeze(-1)
        return log_rate.clamp(-12.0, 12.0), floored(self.where(h)), floored(self.size(h))

    def loss_bits(
        self,
        h: torch.Tensor,
        recent: torch.Tensor,
        counts: torch.Tensor,
        cells: torch.Tensor,
        mags: torch.Tensor,
        day_of: torch.Tensor,
        log_factorials: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Negative Poisson log-likelihood of the next days' quakes, in bits, summed.

        ``counts`` (B,) quakes per day; ``cells``, ``mags`` and ``day_of`` list every quake.
        The joint likelihood of a Poisson rate split over cells and bins factorises into the
        day's total (n log rate − rate) plus the log-probability of each quake's cell and bin,
        minus log n! for every (cell, bin) count, exactly as CB-02 scores it.
        ``log_factorials``, the sum of log n! (a constant), can come precomputed from the CPU
        so the GPU never has to count. Returns (summed bits, number of quakes).
        """
        log_rate, logp_cell, logp_mag = self(h, recent)
        n = counts.float()
        count_ll = n * log_rate - log_rate.exp()
        quake_ll = logp_cell[day_of, cells].sum() + logp_mag[day_of, mags].sum()
        # log n! per (day, cell, bin): a constant for training, but it makes the number exact.
        if log_factorials is None:
            key = (day_of * self.grid.n_cells + cells) * self.grid.n_mags + mags
            _, per_bin = torch.unique(key, return_counts=True)
            quake_ll = quake_ll - torch.lgamma(per_bin.float() + 1).sum()
        else:
            quake_ll = quake_ll - log_factorials
        bits = -(count_ll.sum() + quake_ll) / LN2
        return bits, n.sum()


#: Logistic components in a GNSS answer, per channel.
GNSS_MIX = 4
#: GNSS answers are in millimetres, the unit CB-01 bins them in.
GNSS_BIN_MM = 1.0
#: A sliver of probability on one wide logistic (1 m scale) around the baseline, so a day far
#: from every component (a jump the model never saw coming) still costs a finite number of bits.
GNSS_FLOOR = 1e-3
GNSS_WIDE_MM = 1000.0
#: No channel costs more than CB-01's cap.
MAX_BITS = 64.0


def log_logistic_mass(
    lo: torch.Tensor, hi: torch.Tensor, mu: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """log P(lo ≤ x < hi) for logistic(mu, s), stable in both tails (lo < hi everywhere)."""
    a, b = (lo - mu) / s, (hi - mu) / s
    upper = a > 0  # both edges right of the centre: use the mirrored survival function
    la = torch.where(upper, F.logsigmoid(-a), F.logsigmoid(b))
    lb = torch.where(upper, F.logsigmoid(-b), F.logsigmoid(a))
    gap = (lb - la).clamp(max=-1e-12)
    return la + torch.log1p(-torch.exp(gap))


class GnssHead(nn.Module):
    """For a hidden GNSS day: a distribution over its distance from a known baseline, in mm.

    The baseline is the station's long-term straight line, plus the level of its visible days
    in the window when it has any (``craton.train.context``). The head answers, per channel
    (east, north, up), with a mixture of ``GNSS_MIX`` logistic distributions (the output layer
    of PixelCNN++) whose centres all share one straight ramp, ``b`` mm/yr times the years
    since 2015. The ramp is how a station nobody has seen gets a velocity of its own: its
    baseline is borrowed from its neighbours, and ``b`` is the model's correction.

    The loss is −log2 of the probability of the true 1 mm bin, which is CB-01's score.
    """

    def __init__(self, d_model: int, k: int = GNSS_MIX) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(d_model, 3 * (3 * k + 1))

    def _start_small(self) -> None:
        """Start every component near the baseline, a few millimetres wide."""
        with torch.no_grad():
            self.proj.weight.mul_(0.1)
            self.proj.bias.zero_()

    def forward(
        self, h: torch.Tensor, years: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (N, d), years (N,) since 2015 → log weights, centres (mm), scales (mm), each
        (N, 3, K)."""
        k = self.k
        p = self.proj(h).float().view(-1, 3, 3 * k + 1)
        logw = F.log_softmax(p[..., :k], dim=-1)
        ramp = 10.0 * p[..., 3 * k : 3 * k + 1] * years.float()[:, None, None]
        mu = 10.0 * p[..., k : 2 * k] + ramp
        s = torch.exp(p[..., 2 * k : 3 * k] + math.log(3.0)).clamp(0.05, 1.0e4)
        return logw, mu, s

    def bin_bits(
        self, h: torch.Tensor, years: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        """Bits per channel (N, 3) for the true bins [lo, hi) (mm from the baseline)."""
        logw, mu, s = self(h, years)
        mix = torch.logsumexp(logw + log_logistic_mass(lo[..., None], hi[..., None], mu, s), -1)
        wide = log_logistic_mass(lo, hi, torch.zeros_like(lo), torch.full_like(lo, GNSS_WIDE_MM))
        logp = torch.logaddexp(mix + math.log1p(-GNSS_FLOOR), wide + math.log(GNSS_FLOOR))
        return torch.clamp(-logp / LN2, max=MAX_BITS)

    def loss_bits(
        self, h: torch.Tensor, years: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        """Mean bits per hidden GNSS day, summed over the three channels."""
        return self.bin_bits(h, years, lo, hi).sum(-1).mean()

    def mean_mm(self, h: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
        """(N, 3) the mixture's mean, mm from the baseline: the point answer CB-04 asks for."""
        logw, mu, _ = self(h, years)
        out: torch.Tensor = (logw.exp() * mu).sum(-1)
        return out


HEADS = ("masked", "gnss", "next_day")
