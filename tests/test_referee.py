"""The referee: the replays of a false discovery and a planted true one."""

from __future__ import annotations

import math

import numpy as np

from craton.bench.data import DAY, Events
from craton.sim import SIMULATORS, SimSpec, omori_burst, thin_periodic
from craton.trust import referee


def _calm_catalogue(seed: int = 0) -> Events:
    return SIMULATORS["etas"].simulate(SimSpec((41.0, 44.0, 11.5, 15.0), 3 * 365, t0=1.2e9), seed)


def test_referee_rejects_the_artifact_and_passes_the_planted_signal() -> None:
    claim = referee.Claim("M2", referee.M2_HOURS)
    fake = referee.judge(claim, referee.artifact_catalogue(0), referee.artifact_catalogue(1))
    assert fake.raw.log10_p < -50  # it looks like a discovery…
    assert not fake.passed  # …and it isn't
    real = referee.judge(claim, referee.planted_catalogue(0), referee.planted_catalogue(1))
    assert real.passed
    assert real.lock_hash


def test_shifted_series_catch_an_unevenly_spread_phase() -> None:
    # A "tide" whose phase lingers near 0: every catalogue looks locked to it.
    t_grid = 1.2e9 + np.arange(0, 3650 * 24) * 3600.0
    base = 2 * math.pi * (t_grid / 3600 / referee.M2_HOURS % 1.0)
    lingering = np.mod(base + 0.8 * np.sin(base), 2 * math.pi)

    def phase_of(t: np.ndarray) -> np.ndarray:
        return np.asarray(np.interp(t, t_grid, np.unwrap(lingering)) % (2 * math.pi))

    ev = _calm_catalogue(3)
    v = referee.judge(
        referee.Claim("lingering", referee.M2_HOURS), ev, _calm_catalogue(4), phase_of=phase_of
    )
    assert v.tests["declustered"]  # fooled on its own
    assert not v.tests["fake clocks"]
    assert not v.passed


def test_lock_hash_depends_on_the_effect() -> None:
    c = referee.Claim("x", 12.0)
    a = referee.schuster(np.zeros(10))
    b = referee.schuster(np.full(10, 1.0))
    assert referee.lock(c, a)[0] != referee.lock(c, b)[0]


def test_omori_burst_and_thinning() -> None:
    b = omori_burst(1e9, 42.0, 13.0, 6.0, 2000, 0)
    assert len(b) == 2001
    assert b.mag.max() == 6.0
    early = np.mean(b.time < 1e9 + DAY)
    assert early > 0.3  # Omori: most of it in the first day
    th = thin_periodic(b, 0.5, 0.9, 0)
    assert 0.3 * len(b) < len(th) < len(b)
