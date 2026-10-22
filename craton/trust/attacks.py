"""Fake precursors: signals injected into the inputs to see whether the forecast takes the bait.

Every precursor claim in history (radon, animals, b-value drops, GPS wiggles) died on an
artifact. A forecaster worth trusting must not raise an alarm because of a glitch. CB-07
plants five fakes, each inside a 30-day stretch of early 2009 (the months before L'Aquila,
the most famous "precursor" season there is) at a place at least 60 km from L'Aquila, both
drawn from the seed:

    counted (they carry no information about the Earth; the forecast must ignore them)
    gps_step         the five nearest GNSS stations jump 5 cm east and stay there
    duplicates       every quake in the box is listed twice, a second or so apart
                     (two networks merged without care)
    magnitude_shift  every magnitude in the box reads 0.4 too high (a miscalibrated network;
                     INGV's own switch from Md to ML in 2005 moved the whole catalogue)

    reported (real-looking seismicity; reacting to it can be physics, so it is shown, not
    counted)
    b_drop           small quakes within 50 km grow, so the b-value falls from ~1 to ~0.7:
                     the classic "the big one is coming" signature, with no big one behind it
    fake_swarm       150 small quakes within 5 km over 10 days

Attacks register by name; a new one is one function and one ``register_attack`` call.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from craton.bench.data import DAY, Events, Gnss
from craton.bench.tasks import BenchContext
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Where and when the attacks may go.
SEASON = (to_seconds("2009-01-05T00:00:00Z"), to_seconds("2009-03-01T00:00:00Z"))
ATTACK_DAYS = 30
ZONE_KM = 50.0
LAQUILA = (42.342, 13.380)
KEEP_AWAY_KM = 60.0
BOX = (41.3, 43.7, 11.8, 14.7)  # where the attack site may fall, inside the forecast box


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    t0: float
    t1: float


def km(lat1: float, lon1: float, lat2: Any, lon2: Any) -> Any:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def site(seed: int) -> Site:
    rng = np.random.default_rng(7_000 + seed)
    while True:
        lat, lon = rng.uniform(BOX[0], BOX[1]), rng.uniform(BOX[2], BOX[3])
        if km(LAQUILA[0], LAQUILA[1], lat, lon) >= KEEP_AWAY_KM:
            break
    t0 = SEASON[0] + DAY * float(rng.integers(0, round((SEASON[1] - SEASON[0]) / DAY)))
    return Site(float(lat), float(lon), t0, t0 + ATTACK_DAYS * DAY)


@dataclass(frozen=True)
class Attack:
    name: str
    counted: bool
    what: str
    apply: Callable[[Events, Gnss, Site, np.random.Generator], tuple[Events, Gnss]]


ATTACKS: dict[str, Attack] = {}


def register_attack(attack: Attack) -> Attack:
    if attack.name in ATTACKS:
        raise ValueError(f"an attack called {attack.name!r} exists")
    ATTACKS[attack.name] = attack
    return attack


def _zone(ev: Events, s: Site, radius: float = ZONE_KM) -> np.ndarray:
    near = km(s.lat, s.lon, ev.lat, ev.lon) <= radius
    return np.asarray(near & (ev.time >= s.t0) & (ev.time < s.t1))


def _gps_step(ev: Events, g: Gnss, s: Site, rng: np.random.Generator) -> tuple[Events, Gnss]:
    codes = sorted(set(g.station.tolist()))
    sites = {
        c: (float(np.median(g.lat[g.station == c])), float(np.median(g.lon[g.station == c])))
        for c in codes
    }
    live = [c for c in codes if np.any((g.station == c) & (g.time >= s.t0) & (g.time < s.t1))]
    near = sorted(live, key=lambda c: float(km(s.lat, s.lon, *sites[c])))[:5]
    enu = g.enu.copy()
    enu[np.isin(g.station, near) & (g.time >= s.t0), 0] += 0.05
    return ev, dataclasses.replace(g, enu=enu)


def _in_period(ev: Events, s: Site) -> np.ndarray:
    return np.asarray((ev.time >= s.t0) & (ev.time < s.t1))


def _duplicates(ev: Events, g: Gnss, s: Site, rng: np.random.Generator) -> tuple[Events, Gnss]:
    sel = _in_period(ev, s)
    d = ev.select(sel)
    n = len(d)
    dup = Events(
        d.time + rng.uniform(0.5, 3.0, n),
        d.lat + rng.normal(0, 0.005, n),
        d.lon + rng.normal(0, 0.005, n),
        d.depth_km + rng.normal(0, 0.5, n),
        d.mag.copy(),
        d.mag_type.copy(),
        np.asarray([h + "-dup" for h in d.token_hash], dtype=np.str_),
    )
    return Events.concat([ev, dup]), g


def _magnitude_shift(ev: Events, g: Gnss, s: Site, rng: np.random.Generator) -> tuple[Events, Gnss]:
    mag = ev.mag.copy()
    mag[_in_period(ev, s)] += 0.4
    return dataclasses.replace(ev, mag=np.round(mag, 1)), g


def _b_drop(ev: Events, g: Gnss, s: Site, rng: np.random.Generator) -> tuple[Events, Gnss]:
    mag = ev.mag.copy()
    sel = _zone(ev, s) & (mag >= 1.5) & (mag < 3.0)
    mag[sel] = np.minimum(1.5 + (mag[sel] - 1.5) / 0.7, 2.9)
    return dataclasses.replace(ev, mag=np.round(mag, 1)), g


def _fake_swarm(ev: Events, g: Gnss, s: Site, rng: np.random.Generator) -> tuple[Events, Gnss]:
    n = 150
    t = np.sort(rng.uniform(s.t0, s.t0 + 10 * DAY, n))
    r = 5.0 * np.sqrt(rng.uniform(size=n))
    a = rng.uniform(0, 2 * np.pi, n)
    lat = s.lat + r * np.cos(a) / 111.195
    lon = s.lon + r * np.sin(a) / (111.195 * np.cos(np.radians(s.lat)))
    mag = np.minimum(1.5 + rng.exponential(1 / np.log(10), n), 3.0)
    fake = Events(
        t,
        lat,
        lon,
        rng.uniform(8, 12, n),
        np.round(mag, 1).astype(np.float64),
        np.asarray(["ML"] * n, dtype=np.str_),
        np.asarray([f"fake-swarm-{i}" for i in range(n)], dtype=np.str_),
    )
    return Events.concat([ev, fake]), g


register_attack(Attack("gps_step", True, "five nearest GNSS stations jump 5 cm east", _gps_step))
register_attack(Attack("duplicates", True, "every quake in the box listed twice", _duplicates))
register_attack(
    Attack("magnitude_shift", True, "all magnitudes in the box read 0.4 high", _magnitude_shift)
)
register_attack(Attack("b_drop", False, "small quakes grow: b falls from ~1 to ~0.7", _b_drop))
register_attack(Attack("fake_swarm", False, "150 small quakes within 5 km in 10 days", _fake_swarm))


def attacked(ctx: BenchContext, attack: Attack, seed: int) -> BenchContext:
    """A view of ``ctx`` whose snapshot inputs carry the attack. Training splits are untouched
    (and so is every fit already cached on ``ctx``); test answers come from the frozen files."""
    s = site(seed)
    rng = np.random.default_rng(11_000 + seed)
    ev, g = attack.apply(ctx.events, ctx.gnss, s, rng)
    cache = {k: v for k, v in ctx._cache.items() if k.startswith(("etas-fit:", "field-states:"))}
    cache.update({"events": ev, "gnss": g, "attack": attack.name})
    return dataclasses.replace(ctx, _cache=cache)
