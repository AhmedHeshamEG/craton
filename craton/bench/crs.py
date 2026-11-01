"""The Coulomb rate-and-state floor: aftershocks as the crust's answer to a stress step.

ETAS says quakes make quakes, without saying how. The physics answer (King et al. 1994,
Dieterich 1994) is that every quake moves the stress around it: it pushes some faults
closer to failing and pulls others back. Each fault patch then answers its stress step on a
clock set by rate-and-state friction. A step up makes a burst that fades like Omori's law
over a time ``t_a``; a step down makes a quiet shadow for as long.

This floor does exactly that, from the catalogue alone, anywhere:

1. **Receivers.** Every 0.1° forecast cell holds 3 × 3 × 4 points down to 11 km
   (``configs/floors/crs.yaml``). Where the region states its faults' orientation (central Italy:
   NW-SE normal faults dipping SW, Mancini et al. 2019's CRS-1), the stress is resolved on that
   plane and slip direction.
2. **Sources.** Every quake of the scored size before a forecast's day moves the stress.
   Quakes from M5.4 are rectangles of uniform slip (Wells & Coppersmith 1994 size, centred on
   the hypocentre, the region's orientation), computed with triangular dislocations in an
   elastic half-space (``cutde``, Nikkhoo & Walter 2015). Smaller ones, and every quake where
   no orientation is known, are isotropic: ΔCFF = M0 / (6π r³) (Chen et al. 2013, Mancini's
   CRS-7 for M3-4).
3. **Rates.** At every point, Dieterich's state variable takes each step and relaxes towards
   steady state; the expected count of a day is the background rate times the day's average
   of R/r, integrated in closed form. The background is ETAS's own declustered one, so the
   floor differs from ETAS only in how quakes make quakes.
4. **Fit.** Only (A·σ, t_a) is chosen, by likelihood on the fit split's days, over Mancini's
   ranges (A·σ 0.01-0.1 MPa, t_a 1-1000 years). The one pass over time that scores the fit
   days also writes the test days.

What it is not: Mancini's CRS-7, which also used published finite-fault slip models for the
six largest shocks and receiver planes from pre-sequence focal mechanisms. Neither is in the
token store (slip models are papers; mechanisms aren't tokens), so the floor is the version
any catalogue can run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from craton.bench import csep, etas
from craton.bench.data import DAY
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.paths import configs_dir
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
YEAR_DAYS = 365.25


class Orientation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strike: float
    dip: float
    rake: float


class CrsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    friction: float
    shear_modulus_mpa: float
    poisson: float
    stress_drop_mpa: float
    finite_from: float
    max_stress_mpa: float
    depths_km: tuple[float, ...]
    subcells: int
    default_depth_km: float
    a_sigma_mpa: tuple[float, ...]
    ta_years: tuple[float, ...]
    regions: dict[str, Orientation] = {}


@lru_cache(maxsize=4)
def load_config(root: Path) -> CrsConfig:
    with (configs_dir(root) / "floors" / "crs.yaml").open(encoding="utf-8") as f:
        return CrsConfig.model_validate(yaml.safe_load(f))


# ---------------------------------------------------------------- geometry (east, north, up; km)


def _enu(n: float, e: float, d: float) -> FloatArray:
    return np.array([e, n, -d], dtype=np.float64)


@dataclass(frozen=True)
class Plane:
    """A fault's unit vectors: along strike, down dip, the normal pointing into the hanging
    wall, and the hanging wall's slip direction (Aki & Richards)."""

    strike: FloatArray
    down_dip: FloatArray
    normal: FloatArray
    slip: FloatArray

    @staticmethod
    def of(o: Orientation) -> Plane:
        f, d, r = (math.radians(v) for v in (o.strike, o.dip, o.rake))
        return Plane(
            _enu(math.cos(f), math.sin(f), 0.0),
            _enu(-math.sin(f) * math.cos(d), math.cos(f) * math.cos(d), math.sin(d)),
            _enu(-math.sin(d) * math.sin(f), math.sin(d) * math.cos(f), -math.cos(d)),
            _enu(
                math.cos(r) * math.cos(f) + math.sin(r) * math.cos(d) * math.sin(f),
                math.cos(r) * math.sin(f) - math.sin(r) * math.cos(d) * math.cos(f),
                -math.sin(r) * math.sin(d),
            ),
        )


def moment_nm(m: float) -> float:
    """Seismic moment (N·m) of moment magnitude ``m`` (Hanks & Kanamori 1979)."""
    return float(10.0 ** (1.5 * m + 9.1))


def rupture_km(m: float) -> tuple[float, float]:
    """Subsurface rupture length and width (km), all slip types (Wells & Coppersmith 1994)."""
    return float(10.0 ** (-2.44 + 0.59 * m)), float(10.0 ** (-1.01 + 0.32 * m))


def rectangle(
    x: float, y: float, depth: float, m: float, plane: Plane, cfg: CrsConfig
) -> tuple[FloatArray, FloatArray]:
    """Two triangles (2, 3, 3) of a uniform-slip rectangle centred on the hypocentre, and
    their slip vectors (2, 3) in cutde's (strike, dip, tensile) frame, km."""
    length, width = rupture_km(m)
    slip_km = moment_nm(m) / (cfg.shear_modulus_mpa * 1e6 * length * width * 1e6) / 1e3
    top = 0.5 + 0.5 * width * float(plane.down_dip[2] * -1.0)  # shallowest centre allowed
    c = np.array([x, y, -max(depth, top)], dtype=np.float64)
    hl, hw = length / 2, width / 2
    s, dd = plane.strike, plane.down_dip
    p = [c - hl * s - hw * dd, c + hl * s - hw * dd, c + hl * s + hw * dd, c - hl * s + hw * dd]
    tris = np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])
    slips = np.empty((2, 3))
    for k, t in enumerate(tris):
        vn = np.cross(t[1] - t[0], t[2] - t[0])
        vn /= np.linalg.norm(vn)
        vs = np.cross([0.0, 0.0, 1.0], vn)
        vs /= np.linalg.norm(vs)
        vd = np.cross(vn, vs)
        # cutde's jump (the +vn side minus the -vn side) equals its slip vector, and vn points
        # into the footwall here: so the hanging wall's slip enters with a minus sign.
        u = slip_km * plane.slip
        slips[k] = [-(u @ vs), -(u @ vd), -(u @ vn)]
    return tris, slips


def dcff_rectangle(
    points: FloatArray, tris: FloatArray, slips: FloatArray, plane: Plane, cfg: CrsConfig
) -> FloatArray:
    """Coulomb stress change (MPa) on the receiver plane at every point (P, 3)."""
    import cutde.halfspace as hs
    from cutde.geometry import strain_to_stress

    strain = np.einsum("pktd,td->pk", hs.strain_matrix(points, tris, cfg.poisson), slips)
    s = strain_to_stress(strain, cfg.shear_modulus_mpa, cfg.poisson)  # xx yy zz xy xz yz
    sig = np.empty((len(points), 3, 3))
    sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    sig[:, 0, 1] = sig[:, 1, 0] = s[:, 3]
    sig[:, 0, 2] = sig[:, 2, 0] = s[:, 4]
    sig[:, 1, 2] = sig[:, 2, 1] = s[:, 5]
    traction = sig @ plane.normal
    shear = traction @ plane.slip
    normal = traction @ plane.normal  # tension positive: unclamping helps failure
    out = shear + cfg.friction * normal
    return np.asarray(np.where(np.isfinite(out), out, 0.0), dtype=np.float64)


def dcff_isotropic(
    points: FloatArray, x: float, y: float, depth: float, m: float, cfg: CrsConfig
) -> FloatArray:
    """M0 / (6π r³) in MPa, r the hypocentral distance, no closer than the source radius."""
    m0 = moment_nm(m)
    r_min = (7.0 * m0 / (16.0 * cfg.stress_drop_mpa * 1e6)) ** (1.0 / 3.0)  # m
    r = 1e3 * np.sqrt(
        (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2 + (points[:, 2] + depth) ** 2
    )
    return np.asarray(m0 / (6.0 * math.pi * np.maximum(r, r_min) ** 3) / 1e6, dtype=np.float64)


def receivers(grid: csep.Grid, proj: etas.Projection, cfg: CrsConfig) -> FloatArray:
    """(cells × subcells² × depths, 3) receiver points, km, cell by cell."""
    lat, lon = grid.cell_centres()
    k = cfg.subcells
    off = (np.arange(k) - (k - 1) / 2) * grid.cell_deg / k
    dla, dlo = np.meshgrid(off, off, indexing="ij")
    la = (lat[:, None] + dla.ravel()[None, :]).ravel()
    lo = (lon[:, None] + dlo.ravel()[None, :]).ravel()
    x, y = proj.to_km(la, lo)
    z = -np.asarray(cfg.depths_km, dtype=np.float64)
    n = len(z)
    return np.stack([np.repeat(x, n), np.repeat(y, n), np.tile(z, len(x))], axis=1)


# ---------------------------------------------------------------- Dieterich's clock


@dataclass
class StateBank:
    """log(γ·τ̇) per receiver and per (A·σ, t_a) pair: 0 in steady state."""

    a_sigma: FloatArray  # (C,)
    ta_days: FloatArray  # (C,)
    lg: FloatArray  # (P, C)
    t: float  # days

    def decay_to(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        x = dt / self.ta_days
        self.lg = np.logaddexp(self.lg - x, np.log(-np.expm1(-x)))
        self.t = t

    def step(self, dcff: FloatArray) -> None:
        self.lg = self.lg - dcff[:, None] / self.a_sigma[None, :]

    def mean_rate(self, span: float) -> FloatArray:
        """(P, C) average of R/r over the next ``span`` days, nothing new happening."""
        x = span / self.ta_days
        inner = np.logaddexp(self.lg - x, np.log(-np.expm1(-x)))
        return np.asarray(1.0 + (inner - self.lg) / x, dtype=np.float64)


# ---------------------------------------------------------------- the floor


@dataclass(frozen=True)
class CrsRun:
    rates: FloatArray  # (days, cells) expected quakes at M >= mc, the chosen pair
    a_sigma: float
    ta_years: float
    log_likelihood: FloatArray  # per pair, on the fit days
    finite_sources: int


def region_of(ctx: BenchContext) -> str:
    return str(getattr(ctx, "region", ""))


class Crs:
    """Coulomb rate-and-state, Dieterich (1994), from the catalogue alone."""

    name = "crs"
    kind = "baseline"
    long_periods: ClassVar[bool] = False

    def run(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> CrsRun:
        # Deterministic: one run serves every seed.
        key = f"crs-run:{spec.task_id}:{getattr(ctx, 'view_id', '')}"
        if key in ctx._cache:
            out: CrsRun = ctx._cache[key]
            return out
        from craton.bench.baselines import Etas

        cfg = load_config(ctx.root)
        f = Etas().fit(spec, ctx, seed)
        grid = spec.grid
        proj = etas.Projection.for_grid(grid)
        pts = receivers(grid, proj, cfg)
        per_cell = len(pts) // grid.n_cells
        r0 = f.params.mu * f.background.cell_mass(grid, proj)  # declustered, per day, M >= mc
        orient = cfg.regions.get(region_of(ctx))
        plane = Plane.of(orient) if orient is not None else None

        aa, tt = np.meshgrid(cfg.a_sigma_mpa, cfg.ta_years, indexing="ij")
        bank = StateBank(aa.ravel(), tt.ravel() * YEAR_DAYS, np.zeros((len(pts), aa.size)), 0.0)
        fit_end = (to_seconds(spec.fit_end) - f.t_ref) / DAY
        # Sources: every quake of the scored size, from the fit's start to the test's end.
        hist = ctx.events.window(f.t_ref, spec.window.t1).above(spec.scored)
        src_t = (hist.time - f.t_ref) / DAY
        sx, sy = proj.to_km(hist.lat, hist.lon)
        depth = np.where(np.isfinite(hist.depth_km), hist.depth_km, cfg.default_depth_km)
        depth = np.clip(depth, 0.5, None)

        # Fit days: every whole day after the burn-in, before the cut.
        fit_days = np.arange(etas.BURN_IN_DAYS, math.floor(fit_end))
        train = ctx.train_events(spec.fit_split).window(f.t_ref, to_seconds(spec.fit_end))
        train = train.above(f.mc)
        tc = grid.cell_of(train.lat, train.lon)
        td = np.floor((train.time - f.t_ref) / DAY).astype(np.int64)
        ok = tc >= 0
        counts: dict[int, np.ndarray[Any, Any]] = {}
        for d, c in zip(td[ok].tolist(), tc[ok].tolist(), strict=True):
            counts.setdefault(d, np.zeros(grid.n_cells))[c] += 1
        test_days = (spec.day_starts - f.t_ref) / DAY

        days = np.concatenate([fit_days, test_days])
        is_test = np.concatenate([np.zeros(len(fit_days), bool), np.ones(len(test_days), bool)])
        ll = np.zeros(aa.size)
        test_rates = np.empty((len(test_days), grid.n_cells, aa.size))
        j, n_finite, k_test = 0, 0, 0
        for day, test in zip(days.tolist(), is_test.tolist(), strict=True):
            while j < len(src_t) and src_t[j] < day:
                bank.decay_to(float(src_t[j]))
                m = float(hist.mag[j])
                if plane is not None and m >= cfg.finite_from - 1e-9:
                    tris, slips = rectangle(
                        float(sx[j]), float(sy[j]), float(depth[j]), m, plane, cfg
                    )
                    dc = dcff_rectangle(pts, tris, slips, plane, cfg)
                    n_finite += 1
                else:
                    dc = dcff_isotropic(pts, float(sx[j]), float(sy[j]), float(depth[j]), m, cfg)
                bank.step(np.clip(dc, -cfg.max_stress_mpa, cfg.max_stress_mpa))
                j += 1
            bank.decay_to(day)
            rate = bank.mean_rate(1.0).reshape(grid.n_cells, per_cell, -1).mean(axis=1)
            lam = r0[:, None] * rate
            if test:
                test_rates[k_test] = lam
                k_test += 1
            else:
                n = counts.get(int(day))
                ll -= lam.sum(axis=0)
                if n is not None:
                    hit = n > 0
                    ll += (n[hit, None] * np.log(np.maximum(lam[hit], 1e-300))).sum(axis=0)
        best = int(np.argmax(ll))
        out = CrsRun(
            np.ascontiguousarray(test_rates[:, :, best]),
            float(bank.a_sigma[best]),
            float(bank.ta_days[best] / YEAR_DAYS),
            ll,
            n_finite,
        )
        ctx._cache[key] = out
        ctx._cache[f"crs:{spec.task_id}"] = {"a_sigma_mpa": out.a_sigma, "ta_years": out.ta_years}
        return out

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        from craton.bench.baselines import Etas, _shares

        f = Etas().fit(spec, ctx, seed)
        r = self.run(spec, ctx, seed)
        scale, shares = _shares(f.b, f.mc, spec.grid)
        return GriddedForecast(r.rates * scale, shares)
