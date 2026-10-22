"""`craton referee`: a discovery has to survive the tests that killed every false one before it.

On 2026-10-21 aftershocks in central Italy looked locked to the tide's phase with p = 10⁻²¹.
A plain 24-hour clock locked just as hard. Nothing tidal was going on: aftershocks come in
bursts, so hundreds of quakes share one phase and any clock "locks". The referee exists so that
can't happen again. A claim ("quakes prefer a phase of a cycle with this period") passes only if
it survives all five:

1. **Raw signal.** Schuster's test on the catalogue as it is: p = exp(−R²/N), R the length of
   the sum of the quakes' phase vectors. Reported, never trusted alone (it assumes quakes are
   independent, and they aren't).
2. **Declustered.** Gardner and Knopoff (1974) windows remove the aftershocks of every quake;
   the signal must still be there, p < 0.01, on the mainshocks alone.
3. **Fake clocks.** The same statistic for 200 random periods between 4 hours and 3 days (and
   for the solar day and half-day, named): the claimed period must beat 99% of them on the
   declustered catalogue. This is the Schuster spectrum of Ader and Avouac (2013): a single
   p-value can't prove a period, a period that stands above the spectrum can. When the cycle
   is a computed series (the real tide is a sum of many periods), the fake clocks are that
   series slid by random lags of 30 days to 10 years: shifted tides, the same rhythm with any
   real alignment broken.
4. **Shuffled time of day.** Every quake keeps its day and gets a random time within it: a
   real sub-daily cycle dies, day-scale clustering survives. The claimed R² must beat 99% of
   200 shuffles.
5. **A locked prediction.** The effect (phase and amplitude) measured on the claim's own
   catalogue is written down and hashed *before* a held-out catalogue is read. On the held-out
   one, declustered, the signal must appear again (p < 0.05) within 45° of the locked phase.

Its own tests replay both sides: the clustered catalogue with day-night detection (the
tidal artifact's mechanism, and the real 2016-17 sequence when the data is there) must be
rejected; a planted 12.42-hour signal must pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from craton.bench.data import DAY, Events

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: The principal lunar semidiurnal tide, M2, in hours.
M2_HOURS = 12.4206012
ALPHA_DECLUSTERED = 0.01
ALPHA_HOLDOUT = 0.05
FAKE_CLOCKS = 200
SHUFFLES = 200
SPECTRUM_HOURS = (4.0, 72.0)
CLOCK_QUANTILE = 0.99
PHASE_TOLERANCE_DEG = 45.0
NAMED_CLOCKS = {"solar day": 24.0, "solar half-day": 12.0}


# ---------------------------------------------------------------- statistics


def phases(time: FloatArray, period_hours: float) -> FloatArray:
    """Phase in radians of each moment on a clock with this period (zero at 1970-01-01)."""
    return np.asarray(2 * math.pi * ((time / 3600.0) / period_hours % 1.0), dtype=np.float64)


@dataclass(frozen=True)
class Schuster:
    n: int
    r2_over_n: float  # R² / N: ~1 for random phases, large for a locked cycle
    log10_p: float
    phase_deg: float  # preferred phase
    amplitude: float  # rate swings ±amplitude around the mean (first-harmonic estimate, 2R/N)

    @property
    def p(self) -> float:
        return float(10.0**self.log10_p)


def schuster(ph: FloatArray) -> Schuster:
    n = len(ph)
    if n == 0:
        return Schuster(0, 0.0, 0.0, 0.0, 0.0)
    s = complex(np.sum(np.cos(ph)), np.sum(np.sin(ph)))
    r2 = abs(s) ** 2
    return Schuster(
        n,
        r2 / n,
        -r2 / n / math.log(10),
        math.degrees(math.atan2(s.imag, s.real)) % 360,
        2 * abs(s) / n,
    )


def gardner_knopoff(ev: Events) -> np.ndarray:
    """True for every mainshock: quakes not inside the window of a bigger (or equal, earlier)
    quake. Windows: L = 10^(0.1238 M + 0.983) km; T = 10^(0.032 M + 2.7389) days for M ≥ 6.5,
    10^(0.5409 M − 0.547) days below (Gardner and Knopoff 1974, the usual fit)."""
    n = len(ev)
    keep = np.ones(n, dtype=bool)
    order = np.lexsort((ev.time, -ev.mag))
    big_l = 10 ** (0.1238 * ev.mag + 0.983)
    big_t = np.where(
        ev.mag >= 6.5, 10 ** (0.032 * ev.mag + 2.7389), 10 ** (0.5409 * ev.mag - 0.547)
    )
    coslat = np.cos(np.radians(ev.lat))
    for i in order:
        if not keep[i]:
            continue
        j0, j1 = np.searchsorted(ev.time, [ev.time[i], ev.time[i] + big_t[i] * DAY], "right")
        if j1 <= j0:
            continue
        j = np.arange(j0, j1)
        dy = (ev.lat[j] - ev.lat[i]) * 111.195
        dx = (ev.lon[j] - ev.lon[i]) * 111.195 * coslat[i]
        inside = (np.hypot(dx, dy) <= big_l[i]) & (ev.mag[j] <= ev.mag[i])
        keep[j[inside]] = False
    return keep


def fake_periods(claimed: float, n: int, seed: int) -> FloatArray:
    """``n`` log-uniform periods in SPECTRUM_HOURS, none within 3% of the claim, its double
    or its half."""
    rng = np.random.default_rng(seed)
    out: list[float] = []
    lo, hi = math.log(SPECTRUM_HOURS[0]), math.log(SPECTRUM_HOURS[1])
    while len(out) < n:
        p = math.exp(rng.uniform(lo, hi))
        if all(abs(p / c - 1) > 0.03 for c in (claimed, claimed / 2, claimed * 2)):
            out.append(p)
    return np.asarray(out)


def shuffle_time_of_day(time: FloatArray, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    day = np.floor(time / DAY) * DAY
    return np.asarray(day + rng.uniform(0, DAY, len(time)), dtype=np.float64)


# ---------------------------------------------------------------- the verdict


@dataclass(frozen=True)
class Claim:
    """What is claimed: quakes (M ≥ ``m_min``) prefer a phase of a cycle of this period."""

    name: str
    period_hours: float
    m_min: float = 1.5


@dataclass
class Verdict:
    claim: Claim
    raw: Schuster
    declustered: Schuster
    mainshock_share: float
    clock_quantile: float  # share of fake clocks the claim beats
    named_clocks: dict[str, float]  # log10 p of the solar clocks, declustered
    shuffle_quantile: float
    lock_hash: str = ""
    holdout: Schuster | None = None
    phase_gap_deg: float | None = None
    tests: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.tests) and all(self.tests.values())

    def as_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["passed"] = self.passed
        return body


def lock(claim: Claim, effect: Schuster) -> tuple[str, str]:
    """(sha256, canonical JSON) of the claim and its measured effect: written before any
    held-out quake is read."""
    body = json.dumps(
        {
            "claim": asdict(claim),
            "phase_deg": round(effect.phase_deg, 3),
            "amplitude": round(effect.amplitude, 5),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode()).hexdigest(), body


PhaseFn = Callable[[FloatArray], FloatArray]
#: Shifted tides: the claim's own phase series slid by a random lag in this range (days).
SHIFT_DAYS = (30.0, 3650.0)


def clock(period_hours: float) -> PhaseFn:
    return lambda t: phases(t, period_hours)


def shifted(phase_of: PhaseFn, n: int, seed: int) -> list[PhaseFn]:
    """``n`` copies of a phase series slid by random lags: the same rhythm, any real
    alignment with the quakes broken (the "shifted tides" control)."""
    lags = np.random.default_rng(seed).uniform(*SHIFT_DAYS, n) * DAY

    def slid(lag: float) -> PhaseFn:
        def f(t: FloatArray) -> FloatArray:
            return phase_of(t + lag)

        return f

    return [slid(float(lag)) for lag in lags]


def judge(
    claim: Claim,
    ev: Events,
    holdout: Events | None = None,
    seed: int = 0,
    phase_of: PhaseFn | None = None,
) -> Verdict:
    """Run every test on ``ev``; with ``holdout``, lock the effect first, then test on it.

    ``phase_of`` gives each moment's phase when the cycle is not a plain clock (the computed
    tide, say). Then the fake clocks are that same series slid by random lags (shifted tides):
    a fixed period is no null for a signal that isn't one.
    """
    ph = phase_of or clock(claim.period_hours)
    ev = ev.above(claim.m_min)
    raw = schuster(ph(ev.time))
    main = ev.select(gardner_knopoff(ev))
    dec = schuster(ph(main.time))
    if phase_of is None:
        fakes = [clock(p) for p in fake_periods(claim.period_hours, FAKE_CLOCKS, seed)]
    else:
        fakes = shifted(phase_of, FAKE_CLOCKS, seed)
    fake_r = np.array([schuster(f(main.time)).r2_over_n for f in fakes])
    named = {k: schuster(phases(main.time, p)).log10_p for k, p in NAMED_CLOCKS.items()}
    shuffled = np.array(
        [schuster(ph(shuffle_time_of_day(main.time, seed + i))).r2_over_n for i in range(SHUFFLES)]
    )
    v = Verdict(
        claim,
        raw,
        dec,
        len(main) / max(len(ev), 1),
        float(np.mean(fake_r < dec.r2_over_n)),
        named,
        float(np.mean(shuffled < dec.r2_over_n)),
    )
    v.tests = {
        "declustered": dec.p < ALPHA_DECLUSTERED,
        "fake clocks": v.clock_quantile >= CLOCK_QUANTILE,
        "shuffled time of day": v.shuffle_quantile >= CLOCK_QUANTILE,
    }
    if holdout is not None:
        v.lock_hash, _ = lock(claim, dec)
        h = holdout.above(claim.m_min)
        h = h.select(gardner_knopoff(h))
        hs = schuster(ph(h.time))
        gap = abs((hs.phase_deg - dec.phase_deg + 180) % 360 - 180)
        v.holdout, v.phase_gap_deg = hs, gap
        v.tests["locked prediction"] = hs.p < ALPHA_HOLDOUT and gap <= PHASE_TOLERANCE_DEG
    else:
        v.tests["locked prediction"] = False
    return v


def tide_phase(time: FloatArray, up: FloatArray) -> PhaseFn:
    """The phase of the computed vertical solid-Earth tide (its analytic signal, Hilbert), as
    a function of time: the way a "quakes prefer the tide's peak" claim is usually measured."""
    from scipy.signal import hilbert

    unwrapped = np.unwrap(np.angle(hilbert(up - up.mean())))
    return lambda t: np.asarray(
        np.mod(np.interp(t, time, unwrapped), 2 * math.pi), dtype=np.float64
    )


# ---------------------------------------------------------------- the replays


def artifact_catalogue(seed: int) -> Events:
    """The 2026-10-21 mechanism on a fake planet: a clustered catalogue (ETAS background and one
    M6.5 sequence of 20,000 aftershocks) seen by a network that misses 40% of the small quakes
    in daylight. Nothing in it follows the Moon."""
    from craton.sim import SIMULATORS, SimSpec, omori_burst

    t0 = 1.2e9
    base = SIMULATORS["etas"].simulate(SimSpec((41.0, 44.0, 11.5, 15.0), 3650, t0=t0), seed)
    burst = omori_burst(t0 + 1800.3 * DAY, 42.8, 13.2, 6.5, 20_000, seed + 1)
    ev = Events.concat([base, burst])
    hour = (ev.time / 3600 + 1) % 24  # local time, UTC+1
    rng = np.random.default_rng(seed + 2)
    missed = (hour >= 7) & (hour < 19) & (ev.mag < 2.5) & (rng.random(len(ev)) < 0.4)
    return ev.select(~missed)


def planted_catalogue(seed: int, amplitude: float = 0.3) -> Events:
    """The same kind of clustered planet with a true 12.42-hour modulation planted in every
    quake's chance of happening."""
    from craton.sim import SIMULATORS, SimSpec, thin_periodic

    ev = SIMULATORS["etas"].simulate(SimSpec((41.0, 44.0, 11.5, 15.0), 3650, t0=1.2e9), seed)
    return thin_periodic(ev, M2_HOURS / 24, amplitude, seed + 3)
