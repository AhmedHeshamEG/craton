"""dv/v from noise: correlations find the delay, stretching finds the velocity change."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from craton.adapters.velocity import (
    correlate,
    daily_correlation,
    design,
    fit_correction,
    hour_spectra,
    pair_correlations,
    pair_series,
    stretch,
    stretch_error,
    trailing_stack,
    whiten_onebit,
)

CPU = torch.device("cpu")


def noise(n: int, seed: int = 0) -> torch.Tensor:
    return torch.as_tensor(np.random.default_rng(seed).normal(size=n), dtype=torch.float64)


def test_a_delayed_copy_peaks_at_its_delay() -> None:
    x = noise(3_600 + 50)
    # a[t] = x[t + 50], b[t] = x[t] = a[t - 50]: b hears everything 50 s after a
    a, b = x[50:][None, :], x[:-50][None, :]
    sa, sb = whiten_onebit(a), whiten_onebit(b)
    cc = correlate(sa, sb, 3_600)[0].numpy()
    lag = np.argmax(cc) - (len(cc) - 1) // 2
    assert lag == 50
    assert 0.2 < cc.max() <= 1.0


def test_whitening_keeps_only_the_band() -> None:
    spec = whiten_onebit(noise(3_600)[None, :])[0]
    f = torch.fft.rfftfreq(7_200, d=1.0)
    amp = spec.abs()
    assert amp[(f > 0.15) & (f < 0.4)].min() > 0.99
    assert amp[(f < 0.07) | (f > 0.48)].max() == 0.0


def coda(seed: int = 0) -> np.ndarray:
    """A symmetric band-limited coda decaying over a minute, lags -120..120 s at 1 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(-120, 121, dtype=np.float64)
    spec = np.fft.rfft(rng.normal(size=4_096))
    f = np.fft.rfftfreq(4_096)
    spec[(f < 0.1) | (f > 0.45)] = 0
    w = np.fft.irfft(spec)[: len(t)]
    return w * np.exp(-np.abs(t) / 40.0)


def resample(x: np.ndarray, factor: float) -> np.ndarray:
    t = np.arange(-120, 121, dtype=np.float64)
    out: np.ndarray = np.interp(t * factor, t, x)
    return out


def test_stretching_finds_a_known_velocity_change() -> None:
    ref = coda()
    rows = []
    for dvv in (-0.002, 0.0, 0.001):
        rows.append(resample(ref, 1 + dvv))  # faster rock: arrivals come at t (1 - dvv)
    cur = torch.as_tensor(np.array(rows))
    eps, coh = stretch(torch.as_tensor(ref), cur, 1.0, (10.0, 80.0))
    assert eps.numpy() == pytest.approx([-0.002, 0.0, 0.001], abs=4e-5)
    assert (coh.numpy() > 0.99).all()


def test_the_error_grows_as_the_match_fades() -> None:
    sig = stretch_error(np.array([0.9, 0.5, 0.2]), (10.0, 80.0))
    assert sig[0] < sig[1] < sig[2]
    assert 1e-5 < sig[0] < 1e-3


def test_trailing_stacks_only_look_back() -> None:
    daily = np.array([[1.0], [3.0], [np.nan], [5.0]])
    counts = np.array([24, 24, 0, 12])
    out = trailing_stack(daily, counts, 2)
    assert out[:, 0].tolist() == pytest.approx([1.0, 2.0, 3.0, 5.0])
    assert np.isnan(trailing_stack(daily[2:3], counts[2:3], 1)).all()


def test_all_pairs_at_once_match_one_pair_at_a_time() -> None:
    rng = np.random.default_rng(3)
    grid = rng.normal(size=(3 * 3, 86_400))  # three stations, Z N E each
    grid[4, 7_200:9_000] = np.nan  # station 1, N: hour 2 incomplete
    spec, ok = hour_spectra(grid, 1.0, CPU)
    spec, ok = spec.reshape(3, 3, 24, -1), ok.reshape(3, 3, 24)
    pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
    cc, n = pair_correlations(spec, ok, pairs, chunk=2)
    one, hours = daily_correlation((spec[0, 0], ok[0, 0]), (spec[1, 1], ok[1, 1]))
    assert cc[0, 1].numpy() == pytest.approx(one, abs=1e-12)  # Z-N is 1; N-N is 4
    assert int(n[0, 1]) == hours == 23
    assert n[2].tolist() == [24, 24, 24, 23, 23, 23, 24, 24, 24]


def test_a_pair_finds_a_change_shared_by_its_nine_components() -> None:
    days, n_lag = 120, 241
    refs = [coda(seed) for seed in range(9)]
    cc = np.empty((9, days, n_lag))
    truth = np.where(np.arange(days) >= 80, -0.002, 0.0)
    for k in range(9):
        for d in range(days):
            cc[k, d] = resample(refs[k], 1 + truth[d])
    hours = np.full((9, days), 24)
    ref = np.arange(days) < 60
    dvv, sigma, coh = pair_series(cc, hours, ref, 1, (10.0, 80.0), CPU)
    assert dvv[:80] == pytest.approx(0.0, abs=4e-5)
    assert dvv[80:] == pytest.approx(-0.002, abs=4e-5)
    assert (sigma > 0).all()
    assert (coh > 0.99).all()
    # a ten-day stack spreads the step over ten days, never before it
    slow, _, _ = pair_series(cc, hours, ref, 10, (10.0, 80.0), CPU)
    assert slow[79] == pytest.approx(0.0, abs=4e-5)
    assert -0.002 < slow[84] < 0.0


def test_the_correction_removes_what_the_ocean_put_in() -> None:
    rng = np.random.default_rng(1)
    days = np.arange(400) * 86_400.0
    micro = -130 + 5 * np.sin(np.arange(400) / 20)
    crust = np.where(np.arange(400) >= 300, -0.001, 0.0)
    dvv = crust + 2e-4 * (micro + 130) + rng.normal(0, 2e-5, 400)
    noise = {"microseism": micro, "balance": np.zeros(400)}
    x = design(days, ("microseism",), noise, 1)
    beta = fit_correction(dvv, np.full(400, 2e-5), x, np.arange(400) < 250)
    assert beta is not None
    fixed = dvv - x @ beta
    assert np.std(fixed[:250]) < 3e-5
    assert fixed[300:].mean() == pytest.approx(-0.001, abs=5e-5)
    assert fit_correction(dvv, np.full(400, 2e-5), x, np.arange(400) < 10) is None
