"""The physics planet: rate-and-state faults, Coulomb stress transfer and a fluid source.

ETAS describes clustering; it doesn't say why quakes cluster. This planet says why, with
the physics seismology uses for it, and nothing in it is ETAS:

1. **Faults.** An Apennine-shaped network of normal-fault segments (strike along the
   chain's axis, lengths 8-35 km, the largest quake each can host from its length, Wells
   and Coppersmith 1994), broken into ~2 km patches, plus a coarse grid of diffuse sources
   between them. Each patch stands for a population of nucleation sites.
2. **Rate-and-state seismicity** (Dieterich 1994). Each patch has a state γ. Its quake rate
   is R = r / (γ · τ̇), with r its background rate and τ̇ its tectonic stressing rate.
   Under steady loading γ settles at 1/τ̇ and R = r. A sudden Coulomb stress change ΔS
   multiplies γ by exp(−ΔS / Aσ): a positive step makes the rate jump and decay as Omori's
   law over the aftershock duration ta = Aσ / τ̇; a negative one opens a **stress shadow**,
   a rate below background that ETAS can't represent.
3. **Stress transfer.** Every quake moves the stress of every patch: positive lobes off
   the rupture's ends along strike, negative ones beside it, a drop on the rupture itself,
   falling as distance⁻³, with per-patch heterogeneity (Helmstetter and Shaw 2006: the
   heterogeneity is what lets aftershocks land in the shadows too). Triggered quakes
   transfer stress in turn.
4. **A fluid source.** A pressure pulse diffusing from a point at depth for a few years,
   p = P0 · erfc(r / √(4Dt)), raising the Coulomb stress by μ' · p along a front that
   moves like √t. Nothing in the catalogue says where it is.

Between quakes γ follows its loading law exactly (``state_after``), and the expected
number of quakes in a step is its exact integral (``expected_count``), so the step only
decides when triggered quakes pass their stress on. Steps are an hour, five minutes for
two days after an M5+.

Everything is checked against Dieterich's closed form in ``tests/test_sim.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import erfc

from craton.bench.data import DAY, Events
from craton.sim import SimSpec

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

YEAR = 365.25
KM_PER_DEG = 111.195

#: The chain's axis, north-west to south-east: Mugello, Gubbio, Colfiorito, Norcia,
#: L'Aquila, Sulmona, Matese (lat, lon). The faults line up along it.
APENNINE_AXIS = (
    (43.95, 11.45),
    (43.35, 12.60),
    (43.00, 12.85),
    (42.83, 13.10),
    (42.35, 13.40),
    (42.05, 13.90),
    (41.40, 14.40),
)


def max_magnitude(length_km: FloatArray) -> FloatArray:
    """Largest magnitude a fault of this surface length hosts (Wells & Coppersmith 1994,
    all slip types: M = 5.08 + 1.16 log₁₀ L)."""
    return np.asarray(5.08 + 1.16 * np.log10(length_km), dtype=np.float64)


def rupture_length_km(m: FloatArray) -> FloatArray:
    """The same relation the other way: rupture length of a magnitude-``m`` quake."""
    return np.asarray(10.0 ** ((m - 5.08) / 1.16), dtype=np.float64)


# ---------------------------------------------------------------- the exact loading law


def state_after(gamma: FloatArray, rate: FloatArray, dt: float, a_sigma: FloatArray) -> FloatArray:
    """γ after ``dt`` days of constant stressing ``rate`` (MPa/day), no quakes in between:
    dγ/dt = (1 − γ·rate) / Aσ, solved exactly, for any sign of ``rate``."""
    x = rate * dt / a_sigma
    decay = np.exp(-x)
    return np.asarray(gamma * decay + (dt / a_sigma) * _phi(x), dtype=np.float64)


def _phi(x: FloatArray) -> FloatArray:
    """(1 − e^{−x}) / x, 1 at x = 0."""
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.asarray(np.where(small, 1.0 - x / 2, -np.expm1(-safe) / safe), dtype=np.float64)


def expected_count(
    gamma: FloatArray, rate: FloatArray, dt: float, a_sigma: FloatArray, r_over_tau: FloatArray
) -> FloatArray:
    """Expected quakes over ``dt`` days from state ``gamma`` under constant stressing
    ``rate``: ∫ r / (γ(t) τ̇) dt, exactly, finite for any sign of ``rate``."""
    return np.asarray(r_over_tau * _cumulative(gamma, rate, dt, a_sigma), dtype=np.float64)


def _cumulative(gamma: FloatArray, rate: FloatArray, t: Any, a_sigma: FloatArray) -> FloatArray:
    """∫₀ᵗ dt' / γ(t'), exactly: Aσ · ln(1 + (t/Aσ) · ψ(x) / γ), x = rate·t/Aσ, where
    ψ(x) = (e^x − 1)/x."""
    x = rate * t / a_sigma
    return np.asarray(a_sigma * np.log1p((t / a_sigma) * _psi(x) / gamma), dtype=np.float64)


def _psi(x: FloatArray) -> FloatArray:
    """(e^x − 1) / x, 1 at x = 0."""
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.asarray(np.where(small, 1.0 + x / 2, np.expm1(safe) / safe), dtype=np.float64)


def time_of_count(
    gamma: FloatArray, rate: FloatArray, a_sigma: FloatArray, integral: FloatArray
) -> FloatArray:
    """Inverse of ``_cumulative``: the time at which ∫ dt'/γ reaches ``integral``."""
    # Aσ ln(1 + (e^{rate t/Aσ} − 1) / (γ rate)) = I  →  t = (Aσ/rate) ln(1 + γ rate (e^{I/Aσ} − 1))
    g = np.expm1(integral / a_sigma)
    y = gamma * rate * g
    small = np.abs(y) < 1e-10
    t = np.where(
        small, gamma * g * a_sigma, (a_sigma / np.where(rate == 0, 1.0, rate)) * np.log1p(y)
    )
    return np.asarray(t, dtype=np.float64)


# ---------------------------------------------------------------- the planet


@dataclass(frozen=True)
class Patches:
    """Every nucleation population on the planet."""

    lat: FloatArray
    lon: FloatArray
    strike: FloatArray  # radians, clockwise from north
    r: FloatArray  # background quakes per day at M ≥ m_min
    tau_dot: FloatArray  # tectonic stressing rate, MPa/day
    a_sigma: FloatArray  # MPa
    m_max: FloatArray
    on_fault: np.ndarray

    def __len__(self) -> int:
        return len(self.lat)


@dataclass(frozen=True)
class FluidSource:
    lat: float
    lon: float
    t_on: float  # days since the simulation start
    t_off: float
    p0: float  # MPa


@dataclass(frozen=True)
class RateStatePlanet:
    """A planet where rate-and-state friction and stress transfer are the truth."""

    name: str = "rate_state"
    #: Background M ≥ m_min quakes per day on the faults and in the diffuse grid.
    fault_rate: float = 6.0
    diffuse_rate: float = 2.0
    n_faults: int = 36
    patch_km: float = 2.0
    diffuse_deg: float = 0.1
    #: Tectonic stressing rates (MPa/year) and the direct-effect term Aσ (MPa).
    fault_tau_dot: float = 0.01
    diffuse_tau_dot: float = 0.004
    a_sigma: float = 0.05
    stress_drop: float = 3.0
    heterogeneity: float = 1.0
    #: The largest stress step (MPa) a patch takes from one quake: the patch averages its
    #: population, so it never sees the full near-tip concentration.
    max_step: float = 1.0
    diffuse_m_max: float = 5.6
    b: float = 1.0
    #: Fluid: sources per planet, pressure (MPa), diffusivity (m²/s), friction, depth (km).
    fluid_sources: int = 2
    fluid_p0: float = 1.0
    diffusivity: float = 0.5
    friction: float = 0.6
    fluid_depth_km: float = 6.0
    burn_in_days: float = 5 * YEAR
    dt_days: float = 1.0 / 24
    fine_dt_days: float = 1.0 / 288
    fine_after: float = 5.0
    fine_for_days: float = 2.0
    min_stress_magnitude: float = 2.5

    # ------------------------------------------------------------ geometry

    def patches(self, box: tuple[float, float, float, float], rng: np.random.Generator) -> Patches:
        lat0, lat1, lon0, lon1 = box
        axis = np.asarray(APENNINE_AXIS)
        seg = np.diff(axis, axis=0)
        kx = KM_PER_DEG * math.cos(math.radians((lat0 + lat1) / 2))
        seg_km = np.hypot(seg[:, 0] * KM_PER_DEG, seg[:, 1] * kx)
        cum = np.concatenate([[0.0], np.cumsum(seg_km)])
        lats, lons, strikes, mmax = [], [], [], []
        for _ in range(self.n_faults):
            s = rng.uniform(0, cum[-1])
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            f = (s - cum[i]) / seg_km[i]
            c_lat, c_lon = axis[i] + f * seg[i]
            az = math.atan2(seg[i, 1] * kx, seg[i, 0] * KM_PER_DEG) + rng.normal(0, 0.14)
            off = rng.normal(0, 12.0)  # km across the chain
            c_lat += off * -math.sin(az) / KM_PER_DEG
            c_lon += off * math.cos(az) / kx
            length = float(np.clip(rng.lognormal(math.log(16.0), 0.4), 8.0, 35.0))
            n = max(round(length / self.patch_km), 1)
            along = (np.arange(n) + 0.5 - n / 2) * self.patch_km
            lats.append(c_lat + along * math.cos(az) / KM_PER_DEG)
            lons.append(c_lon + along * math.sin(az) / kx)
            strikes.append(np.full(n, az))
            mmax.append(np.full(n, float(max_magnitude(np.array([length]))[0])))
        f_lat, f_lon = np.concatenate(lats), np.concatenate(lons)
        f_strike, f_mmax = np.concatenate(strikes), np.concatenate(mmax)
        g_lat, g_lon = np.meshgrid(
            np.arange(lat0 + self.diffuse_deg / 2, lat1, self.diffuse_deg),
            np.arange(lon0 + self.diffuse_deg / 2, lon1, self.diffuse_deg),
            indexing="ij",
        )
        g_lat, g_lon = g_lat.ravel(), g_lon.ravel()
        n_f, n_g = len(f_lat), len(g_lat)
        f_w = rng.lognormal(0, 0.8, n_f)
        g_w = rng.lognormal(0, 0.8, n_g)
        r = np.concatenate([self.fault_rate * f_w / f_w.sum(), self.diffuse_rate * g_w / g_w.sum()])
        tau = np.concatenate(
            [
                self.fault_tau_dot * rng.uniform(0.5, 2.0, n_f),
                self.diffuse_tau_dot * rng.uniform(0.5, 2.0, n_g),
            ]
        )
        inside = np.concatenate(
            [
                (f_lat >= lat0) & (f_lat < lat1) & (f_lon >= lon0) & (f_lon < lon1),
                np.ones(n_g, bool),
            ]
        )
        return Patches(
            lat=np.concatenate([f_lat, g_lat])[inside],
            lon=np.concatenate([f_lon, g_lon])[inside],
            strike=np.concatenate([f_strike, rng.normal(2.44, 0.6, n_g)])[inside],
            r=r[inside],
            tau_dot=tau[inside] / YEAR,
            a_sigma=np.full(int(inside.sum()), self.a_sigma),
            m_max=np.concatenate([f_mmax, np.full(n_g, self.diffuse_m_max)])[inside],
            on_fault=np.concatenate([np.ones(n_f, bool), np.zeros(n_g, bool)])[inside],
        )

    def fluids(self, p: Patches, days: float, rng: np.random.Generator) -> list[FluidSource]:
        out = []
        faults = np.flatnonzero(p.on_fault)
        for _ in range(self.fluid_sources):
            i = int(rng.choice(faults))
            t_on = rng.uniform(self.burn_in_days, days)
            out.append(
                FluidSource(
                    float(p.lat[i]),
                    float(p.lon[i]),
                    t_on,
                    t_on + rng.uniform(1.0, 5.0) * YEAR,
                    self.fluid_p0 * rng.uniform(0.5, 1.5),
                )
            )
        return out

    # ------------------------------------------------------------ stress

    def stress_steps(
        self,
        p: Patches,
        e_lat: FloatArray,
        e_lon: FloatArray,
        e_mag: FloatArray,
        e_strike: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """(patches,) summed Coulomb stress change (MPa) from a batch of quakes."""
        kx = KM_PER_DEG * np.cos(np.radians(e_lat))[:, None]
        dy = (p.lat[None, :] - e_lat[:, None]) * KM_PER_DEG
        dx = (p.lon[None, :] - e_lon[:, None]) * kx
        s_n, s_e = np.cos(e_strike)[:, None], np.sin(e_strike)[:, None]
        along = dy * s_n + dx * s_e
        across = -dy * s_e + dx * s_n
        half = rupture_length_km(e_mag)[:, None] / 2
        s2 = (self.patch_km / 2) ** 2
        r2 = along**2 + across**2 + half**2 + s2
        amp = self.stress_drop * (half**2 / r2) ** 1.5
        shape = (along**2 - across**2 - half**2) / r2
        noise = rng.normal(0.0, self.heterogeneity, shape.shape)
        step = np.minimum(amp * (shape + noise), self.max_step)
        return np.asarray(step.sum(axis=0), dtype=np.float64)

    def pressure(self, p: Patches, fluids: list[FluidSource], t: float) -> FloatArray:
        """Pore pressure (MPa) at every patch at time ``t`` (days)."""
        out = np.zeros(len(p))
        d_km2 = self.diffusivity * DAY / 1e6
        for f in fluids:
            if t <= f.t_on:
                continue
            kx = KM_PER_DEG * math.cos(math.radians(f.lat))
            dist = np.sqrt(
                ((p.lat - f.lat) * KM_PER_DEG) ** 2
                + ((p.lon - f.lon) * kx) ** 2
                + self.fluid_depth_km**2
            )
            on = f.p0 * erfc(dist / np.sqrt(4 * d_km2 * (t - f.t_on)))
            if t > f.t_off:
                on -= f.p0 * erfc(dist / np.sqrt(4 * d_km2 * (t - f.t_off)))
            out += on
        return out

    # ------------------------------------------------------------ the run

    def planet(self, box: tuple[float, float, float, float], seed: int) -> Patches:
        return self.patches(box, np.random.default_rng([seed, 1]))

    def simulate(self, spec: SimSpec, seed: int) -> Events:
        rng = np.random.default_rng([seed, 2])
        p = self.planet(spec.box, seed)
        total = spec.days + self.burn_in_days
        fluids = self.fluids(p, total, np.random.default_rng([seed, 3]))
        beta = self.b * math.log(10)
        gamma = 1.0 / p.tau_dot
        r_over_tau = p.r / p.tau_dot
        span = 1 - np.exp(-beta * (p.m_max - spec.m_min))
        t = 0.0
        fine_until = -1.0
        pres = self.pressure(p, fluids, 0.0)
        times: list[FloatArray] = []
        where: list[IntArray] = []
        mags: list[FloatArray] = []
        while t < total:
            dt = self.fine_dt_days if t < fine_until else self.dt_days
            dt = min(dt, total - t)
            nxt = self.pressure(p, fluids, t + dt)
            rate = p.tau_dot + self.friction * (nxt - pres) / dt
            lam = expected_count(gamma, rate, dt, p.a_sigma, r_over_tau)
            n = rng.poisson(lam)
            g0 = gamma
            gamma = state_after(gamma, rate, dt, p.a_sigma)
            if n.any():
                idx = np.repeat(np.flatnonzero(n), n[n > 0])
                # Each quake's time inside the step, from the step's own rate profile.
                full = _cumulative(g0[idx], rate[idx], dt, p.a_sigma[idx])
                u = rng.random(len(idx)) * full
                tt = t + time_of_count(g0[idx], rate[idx], p.a_sigma[idx], u)
                m = spec.m_min - np.log(1 - rng.random(len(idx)) * span[idx]) / beta
                times.append(tt)
                where.append(idx)
                mags.append(m)
                big = m >= self.min_stress_magnitude
                if big.any():
                    ds = self.stress_steps(
                        p, p.lat[idx[big]], p.lon[idx[big]], m[big], p.strike[idx[big]], rng
                    )
                    gamma = gamma * np.exp(-ds / p.a_sigma)
                if m.max() >= self.fine_after:
                    fine_until = t + dt + self.fine_for_days
            pres = nxt
            t += dt
        tt = np.concatenate(times) if times else np.zeros(0)
        ii = np.concatenate(where) if where else np.zeros(0, np.int64)
        mm = np.concatenate(mags) if mags else np.zeros(0)
        keep = tt >= self.burn_in_days
        tt, ii, mm = tt[keep] - self.burn_in_days, ii[keep], mm[keep]
        order = np.argsort(tt, kind="stable")
        tt, ii, mm = tt[order], ii[order], mm[order]
        count = len(tt)
        jitter = rng.normal(0, 0.6, (count, 2))  # km: a patch is a population, not a point
        kx = KM_PER_DEG * np.cos(np.radians(p.lat[ii]))
        return Events(
            time=spec.t0 + tt * DAY,
            lat=p.lat[ii] + jitter[:, 0] / KM_PER_DEG,
            lon=p.lon[ii] + jitter[:, 1] / kx,
            depth_km=rng.uniform(5.0, 15.0, count),
            mag=np.round(mm, 1).astype(np.float64),
            mag_type=np.asarray(["sim"] * count, dtype=np.str_),
            token_hash=np.asarray(
                [f"sim-{self.name}-{seed}-{i}" for i in range(count)], dtype=np.str_
            ),
        )


def dieterich_rate(
    t: FloatArray, r: float, step: float, a_sigma: float, tau_dot: float
) -> FloatArray:
    """Dieterich (1994) eq. 12: the rate after a stress step ``step`` at t = 0 on a population
    at steady state under constant loading, R = r / (1 + (e^{−ΔS/Aσ} − 1) e^{−t/ta})."""
    ta = a_sigma / tau_dot
    return np.asarray(r / (1 + (math.exp(-step / a_sigma) - 1) * np.exp(-t / ta)), dtype=np.float64)
