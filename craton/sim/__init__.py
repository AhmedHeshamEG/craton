"""Fake planets: simulators that make synthetic catalogues with known answers.

Lock 5 says never trust one simulator. This is the hook every simulator plugs into, so P8 can
add physics-based ones (rate-and-state fault models, the planets humans made) next to the
first: an ETAS planet, where the true model is ETAS itself. A synthetic catalogue comes back
as ordinary ``Events`` with the magnitude type ``sim`` and token hashes that say which
simulator and seed made each quake, so nothing synthetic can pass for real data.

    from craton.sim import SIMULATORS, SimSpec
    ev = SIMULATORS["etas"].simulate(SimSpec(box=(41, 44, 11.5, 15), days=3650), seed=0)

The referee's tests use it (a planted periodic signal on a clustered catalogue).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from craton.bench import etas
from craton.bench.data import DAY, Events

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class SimSpec:
    """What to simulate: a lat/lon box, how many days from ``t0`` (seconds since 1970), the
    smallest magnitude."""

    box: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    days: float
    t0: float = 0.0
    m_min: float = 1.5


@runtime_checkable
class Simulator(Protocol):
    @property
    def name(self) -> str: ...

    def simulate(self, spec: SimSpec, seed: int) -> Events: ...


#: A central-Italy-like ETAS: ~0.6 M1.5+ background quakes a day, b = 1, strongly clustered.
CENTRAL_ITALY_LIKE = etas.EtasParams(
    mu=0.6, k=0.03, a=1.9, c=0.004, p=1.12, d=0.8, gamma=0.9, q=1.6
)


@dataclass(frozen=True)
class EtasPlanet:
    """A planet where ETAS is the truth."""

    name: str = "etas"
    params: etas.EtasParams = field(default=CENTRAL_ITALY_LIKE)
    b: float = 1.0
    m_max: float = 7.3

    def simulate(self, spec: SimSpec, seed: int) -> Events:
        lat0, lat1, lon0, lon1 = spec.box
        proj = etas.Projection((lat0 + lat1) / 2, (lon0 + lon1) / 2)
        x0, y0 = proj.to_km(lat0, lon0)
        x1, y1 = proj.to_km(lat1, lon1)
        cat = etas.simulate(
            self.params,
            self.b,
            spec.m_min,
            spec.days,
            (float(x0), float(x1), float(y0), float(y1)),
            seed,
            self.m_max,
        )
        lat = proj.lat0 + cat.y / proj.ky
        lon = proj.lon0 + cat.x / proj.kx
        n = len(cat)
        rng = np.random.default_rng(seed + 1)
        return Events(
            time=spec.t0 + cat.t * DAY,
            lat=lat,
            lon=lon,
            depth_km=rng.uniform(5.0, 15.0, n),
            mag=np.round(cat.m, 1).astype(np.float64),
            mag_type=np.asarray(["sim"] * n, dtype=np.str_),
            token_hash=np.asarray([f"sim-{self.name}-{seed}-{i}" for i in range(n)], dtype=np.str_),
        )


SIMULATORS: dict[str, Simulator] = {}


def register_simulator(sim: Simulator) -> Simulator:
    if sim.name in SIMULATORS:
        raise ValueError(f"a simulator called {sim.name!r} exists")
    SIMULATORS[sim.name] = sim
    return sim


register_simulator(EtasPlanet())

#: Central Italy as ETAS sees it: the ``etas`` floor's fit on CB-02's split (M ≥ 2, 2005 to
#: Amatrice), rounded. A planet where those numbers are the whole truth (CB-09).
ITALY_AS_ETAS = etas.EtasParams(
    mu=0.575, k=0.477, a=1.225, c=0.018, p=1.186, d=0.541, gamma=0.695, q=2.035
)

register_simulator(EtasPlanet(name="etas_italy", params=ITALY_AS_ETAS, b=1.16))

# The physics planet (P8) registers through the same hook.
from craton.sim.rate_state import RateStatePlanet  # noqa: E402

register_simulator(RateStatePlanet())


def thin_periodic(
    ev: Events, period_days: float, amplitude: float, seed: int, phase: float = 0.0
) -> Events:
    """Keep each quake with probability (1 + A·cos(2π t / P − phase)) / (1 + A): a planted
    periodic modulation of amplitude ``amplitude`` (0.2 = the rate swings ±20%)."""
    rng = np.random.default_rng(seed)
    w = 2 * math.pi * (ev.time / DAY) / period_days - phase
    keep = rng.random(len(ev)) < (1 + amplitude * np.cos(w)) / (1 + amplitude)
    return ev.select(keep)


def omori_burst(
    t_main: float,
    lat: float,
    lon: float,
    m_main: float,
    n: int,
    seed: int,
    c_days: float = 0.01,
    p: float = 1.1,
    days: float = 60.0,
    b: float = 1.0,
    m_min: float = 1.5,
) -> Events:
    """A mainshock and ``n`` aftershocks fading as Omori's law (t + c)^-p over ``days``,
    within ~5 km, magnitudes Gutenberg-Richter below the mainshock."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    # inverse CDF of the truncated Omori density on [0, days]
    a0, a1 = c_days ** (1 - p), (days + c_days) ** (1 - p)
    dt = (a0 + u * (a1 - a0)) ** (1 / (1 - p)) - c_days
    beta = b * math.log(10)
    span = 1 - math.exp(-beta * (m_main - 0.1 - m_min))
    mags = m_min - np.log(1 - rng.random(n) * span) / beta
    t = np.concatenate([[t_main], t_main + dt * DAY])
    la = np.concatenate([[lat], lat + rng.normal(0, 5 / 111.195, n)])
    lo = np.concatenate(
        [[lon], lon + rng.normal(0, 5 / (111.195 * math.cos(math.radians(lat))), n)]
    )
    m = np.round(np.concatenate([[m_main], mags]), 1).astype(np.float64)
    order = np.argsort(t, kind="stable")
    return Events(
        time=t[order],
        lat=la[order],
        lon=lo[order],
        depth_km=rng.uniform(5.0, 15.0, n + 1)[order],
        mag=m[order],
        mag_type=np.asarray(["sim"] * (n + 1), dtype=np.str_),
        token_hash=np.asarray([f"sim-burst-{seed}-{i}" for i in order], dtype=np.str_),
    )
