"""CB-29, EarthquakeNPP: an outside benchmark, with its data, its splits and its ETAS untouched.

EarthquakeNPP (Stockman, Lawson & Werner, TMLR 2026) scored five neural point processes on
California and found none of them better than ETAS. CB-29 is its ComCat_25 dataset: every
M2.5+ quake inside California's polygon, ETAS fitted by its authors on 1971 (burn-in) /
1981-2007, tested on 2007-01-01 → 2020-01-17. The metric is theirs: the point-process
log-likelihood of the test quakes per quake, in nats (lower negative log-likelihood is
better), with time in days and space in km².

**The ETAS line is theirs.** Their fitted parameters (``ETAS_PARAMETERS``, from their
repository) define λ_ETAS(t, x), and this task computes its likelihood with their formula;
the reproduction differs from their published 7.2554 by 1·10⁻⁴ nats only because their first
test interval starts at the first test quake rather than at the window's start.

**How Craton enters a continuous-time benchmark.** Craton forecasts by the day and by the
0.1° cell. Its intensity here is λ_C(t, x) = λ_ETAS(t, x) · e^{c(day, cell)}: their ETAS,
moved in each cell by a factor fixed at that day's midnight from what was known then. That is
a proper point process (the factor is predictable), so its likelihood is exact:
Σ log λ_C(t_i, x_i) − ∫ λ_C. Within a day it still reacts to every quake the way ETAS does.
The factor multiplies the background and every parent's kernel inside the cells within
``R_CUT_KM`` of it; the kernel's far tail (and everything outside the grid) is left at
ETAS's value, which keeps the integral exact. A method with c = 0 *is* their ETAS.

Each parent's mass in each nearby cell is integrated exactly along 360 rays from the parent
(the kernel's radial distribution is known in closed form), on a plane tangent at the parent.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import quad

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.bench.tasks import BenchContext, Score, Task, _write_table, register_task
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

DATASET = "ComCat_25"
REGION = "earthquakenpp_comcat"
AUX_START = "1971-01-01T00:00:00Z"
TRAIN_START = "1981-01-01T00:00:00Z"
TEST_START = "2007-01-01T00:00:00Z"
TEST_END = "2020-01-17T00:00:00Z"
MC = 2.5
DELTA_M = 0.1
EARTH_RADIUS_KM = 6378.1
#: California's polygon, (lat, lon), from their ``california_shape.npy``.
POLYGON = (
    (43.5, -125.7),
    (43.5, -118.5),
    (39.7, -118.5),
    (36.1, -113.6),
    (34.6, -112.6),
    (34.3, -112.6),
    (32.7, -113.1),
    (31.8, -113.2),
    (31.2, -114.5),
    (31.0, -117.1),
    (31.1, -117.4),
    (31.5, -118.3),
    (32.4, -118.8),
    (33.3, -121.3),
    (34.0, -122.0),
    (37.5, -124.3),
    (40.0, -125.9),
    (40.5, -125.9),
    (43.0, -125.7),
    (43.5, -125.7),
)
#: Their polygon's area (km²) and fitted ETAS (``output_data_ComCat_25/parameters_0.json``).
AREA_KM2 = 959822.9591782562
ETAS_PARAMETERS = {
    "log10_mu": -6.3329886729199405,
    "log10_k0": -2.6717871682913943,
    "a": 1.5556888441764327,
    "log10_c": -2.79828186440062,
    "omega": -0.06185408129678495,
    "log10_tau": 3.723786279418681,
    "log10_d": -0.7725791509539909,
    "gamma": 1.010873536606162,
    "rho": 0.5565963092026862,
}
#: Gutenberg-Richter β (natural log) from the same fit.
BETA = 2.1471442086213064
#: Their published test scores (``ll_scores.json``), nats per quake.
PUBLISHED = {"etas_nll": 7.2554275527505645, "poisson_nll": 13.261863460288378}
#: The grid the factor lives on: 0.1° cells over the polygon's box.
GRID = csep.Grid(31.0, 43.5, -126.0, -112.5, cell_deg=0.1, m_min=MC, m_max=9.5)
R_CUT_KM = 50.0
RAYS = 360
ENGINE_VERSION = "n0.1"


@dataclass(frozen=True)
class Kernel:
    """Their ETAS: λ(t, x) = μ + Σ k0 e^{a(m−mc)} e^{−Δt/τ} (Δt + c)^{−1−ω} (r² + D)^{−1−ρ},
    D = d e^{γ(m−mc)}."""

    mu: float
    k0: float
    a: float
    c: float
    omega: float
    tau: float
    d: float
    gamma: float
    rho: float

    @staticmethod
    def of(p: dict[str, float]) -> Kernel:
        return Kernel(
            10 ** p["log10_mu"],
            10 ** p["log10_k0"],
            p["a"],
            10 ** p["log10_c"],
            p["omega"],
            10 ** p["log10_tau"],
            10 ** p["log10_d"],
            p["gamma"],
            p["rho"],
        )

    def productivity(self, m: FloatArray) -> FloatArray:
        return np.asarray(self.k0 * np.exp(self.a * (m - MC)), dtype=np.float64)

    def zone(self, m: FloatArray) -> FloatArray:
        return np.asarray(self.d * np.exp(self.gamma * (m - MC)), dtype=np.float64)

    def space_mass(self, m: FloatArray) -> FloatArray:
        """∫ (r² + D)^{−1−ρ} over the plane."""
        return np.asarray(math.pi / (self.rho * self.zone(m) ** self.rho), dtype=np.float64)

    def time_mesh(self) -> tuple[FloatArray, FloatArray]:
        """(t, ∫₀ᵗ time decay), log-spaced to a million days, for interpolation."""

        def f(x: float) -> float:
            return float(math.exp(-x / self.tau) / (x + self.c) ** (1 + self.omega))

        mesh = np.concatenate([[0.0], np.logspace(-9, 6, 30001)])
        cum = np.zeros(len(mesh))
        for i in range(1, len(mesh)):
            cum[i] = cum[i - 1] + quad(f, mesh[i - 1], mesh[i])[0]
        return mesh, cum


KERNEL = Kernel.of(ETAS_PARAMETERS)


def polygon_mask(lat: FloatArray, lon: FloatArray) -> np.ndarray:
    """Points inside the polygon (even-odd rule, in (lat, lon) as they define it)."""
    poly = np.asarray(POLYGON)
    inside = np.zeros(len(lat), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        (ya, xa), (yb, xb) = poly[i], poly[j]
        hit = ((xa > lon) != (xb > lon)) & (
            lat < (ya - yb) * (lon - xb) / np.where(xa == xb, 1e-12, xa - xb) + yb
        )
        inside ^= hit
        j = i
    return inside


def round_half_up(m: FloatArray) -> FloatArray:
    return np.asarray(np.floor(m / DELTA_M + 0.5) * DELTA_M, dtype=np.float64)


def catalogue(ev: Events) -> Events:
    """Their filter: polygon, [aux start, test end), magnitudes rounded half up, ≥ mc."""
    ev = ev.window(to_seconds(AUX_START), to_seconds(TEST_END))
    ev = ev.select(polygon_mask(ev.lat, ev.lon))
    m = round_half_up(ev.mag)
    keep = m >= MC - 1e-9
    out = ev.select(keep)
    return Events(out.time, out.lat, out.lon, out.depth_km, m[keep], out.mag_type, out.token_hash)


def cell_areas() -> FloatArray:
    """Each cell's area inside the polygon (km², 20 × 20 samples a cell), scaled to sum to
    their polygon's area so the background integrates exactly as theirs."""
    lat, lon = GRID.cell_centres()
    k = 20
    off = (np.arange(k) + 0.5) / k - 0.5
    la = lat[:, None, None] + off[None, :, None] * GRID.cell_deg
    lo = lon[:, None, None] + off[None, None, :] * GRID.cell_deg
    la, lo = np.broadcast_arrays(la, lo)
    inside = polygon_mask(la.ravel(), lo.ravel()).reshape(la.shape).mean(axis=(1, 2))
    ky = EARTH_RADIUS_KM * math.pi / 180
    area = inside * (GRID.cell_deg * ky) ** 2 * np.cos(np.radians(lat))
    return np.asarray(area * AREA_KM2 / area.sum(), dtype=np.float64)


# ---------------------------------------------------------------- each parent's mass per cell


def near_masses(
    lat: FloatArray, lon: FloatArray, m: FloatArray, kernel: Kernel = KERNEL
) -> tuple[np.ndarray, np.ndarray, FloatArray, FloatArray]:
    """(parent, cell, mass) of every parent in every cell whose centre is within
    ``R_CUT_KM``, and each parent's far mass (the plane minus those cells). Masses are
    ∫ (r² + D)^{−1−ρ} dA, so ``mass · productivity · time decay`` is the expected count."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ky = EARTH_RADIUS_KM * math.pi / 180
    clat, clon = GRID.cell_centres()
    dr = int(R_CUT_KM / (ky * GRID.cell_deg)) + 2
    theta = (np.arange(RAYS) + 0.5) * 2 * math.pi / RAYS + 1e-3
    cos_t = torch.tensor(np.cos(theta), device=dev)
    sin_t = torch.tensor(np.sin(theta), device=dev)
    rows_p, rows_c, rows_m = [], [], []
    zone = kernel.zone(m)
    full = kernel.space_mass(m)
    near_total = np.zeros(len(lat))
    i_lat = np.floor((lat - GRID.lat_min) / GRID.cell_deg).astype(np.int64)
    i_lon = np.floor((lon - GRID.lon_min) / GRID.cell_deg).astype(np.int64)
    for p0 in range(0, len(lat), 2048):
        p1 = min(p0 + 2048, len(lat))
        kx = ky * np.cos(np.radians(lat[p0:p1]))
        dc = (R_CUT_KM / (kx * GRID.cell_deg)).astype(np.int64) + 2
        cand_p, cand_c = [], []
        for q in range(p1 - p0):
            j = p0 + q
            r = np.arange(max(i_lat[j] - dr, 0), min(i_lat[j] + dr + 1, GRID.n_lat))
            c = np.arange(max(i_lon[j] - dc[q], 0), min(i_lon[j] + dc[q] + 1, GRID.n_lon))
            if not len(r) or not len(c):
                continue
            cells = (r[:, None] * GRID.n_lon + c[None, :]).ravel()
            dx = (clon[cells] - lon[j]) * kx[q]
            dy = (clat[cells] - lat[j]) * ky
            cells = cells[dx**2 + dy**2 < R_CUT_KM**2]
            cand_p.append(np.full(len(cells), j))
            cand_c.append(cells)
        if not cand_p:
            continue
        pp, cc = np.concatenate(cand_p), np.concatenate(cand_c)
        kxp = ky * np.cos(np.radians(lat[pp]))
        half = GRID.cell_deg / 2
        x0 = torch.tensor((clon[cc] - half - lon[pp]) * kxp, device=dev)
        x1 = torch.tensor((clon[cc] + half - lon[pp]) * kxp, device=dev)
        y0 = torch.tensor((clat[cc] - half - lat[pp]) * ky, device=dev)
        y1 = torch.tensor((clat[cc] + half - lat[pp]) * ky, device=dev)
        zz = torch.tensor(zone[pp], device=dev)[:, None]
        frac = torch.zeros(len(pp), dtype=torch.float64, device=dev)
        for s0 in range(0, len(pp), 4096):
            s = slice(s0, s0 + 4096)
            tx0, tx1 = x0[s, None] / cos_t, x1[s, None] / cos_t
            ty0, ty1 = y0[s, None] / sin_t, y1[s, None] / sin_t
            t_in = torch.clamp(
                torch.maximum(torch.minimum(tx0, tx1), torch.minimum(ty0, ty1)), min=0.0
            )
            t_out = torch.minimum(torch.maximum(tx0, tx1), torch.maximum(ty0, ty1))
            z = zz[s]
            cdf_out = 1 - (z / (t_out.clamp(min=0) ** 2 + z)) ** kernel.rho
            cdf_in = 1 - (z / (t_in**2 + z)) ** kernel.rho
            frac[s] = torch.where(t_out > t_in, cdf_out - cdf_in, 0.0).mean(dim=1)
        mass = frac.cpu().numpy() * full[pp]
        np.add.at(near_total, pp, mass)
        rows_p.append(pp)
        rows_c.append(cc)
        rows_m.append(mass)
    far = np.maximum(full - near_total, 0.0)
    return (
        np.concatenate(rows_p),
        np.concatenate(rows_c),
        np.concatenate(rows_m),
        far,
    )


# ---------------------------------------------------------------- the engine


@dataclass
class Engine:
    """Everything a CB-29 score needs about their ETAS, computed once and cached on disk.

    Days run from ``TRAIN_START`` to ``TEST_END``. For each (day, cell):
    ``real`` is ETAS's expected count that day (background + the near kernels of every
    parent before the day ends, the day's own quakes included), ``fore`` the same from the
    parents before the day's midnight only (what a forecaster knows at midnight); ``far`` is
    each day's mass left at ETAS's value. For each test quake: μ + its near kernels, and its
    far kernels."""

    folder: Path
    t: FloatArray  # days since AUX_START, every quake ≥ mc
    lat: FloatArray
    lon: FloatArray
    m: FloatArray

    @property
    def day0(self) -> float:
        return (to_seconds(TRAIN_START) - to_seconds(AUX_START)) / DAY

    @property
    def n_days(self) -> int:
        return round((to_seconds(TEST_END) - to_seconds(TRAIN_START)) / DAY)

    @property
    def test_day0(self) -> int:
        return round((to_seconds(TEST_START) - to_seconds(TRAIN_START)) / DAY)

    def day_starts(self) -> FloatArray:
        """Midnights (seconds since 1970) of every engine day."""
        return np.asarray(to_seconds(TRAIN_START) + DAY * np.arange(self.n_days), np.float64)

    def build(self, log: Any = print) -> None:
        if (self.folder / "done.json").is_file():
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k = KERNEL
        t0 = time.perf_counter()
        pp, cc, mass, far = near_masses(self.lat, self.lon, self.m)
        log(f"CB-29 engine: {len(pp):,} parent-cell masses in {time.perf_counter() - t0:.0f}s")
        mesh, cum = k.time_mesh()
        mesh_t = torch.tensor(mesh, device=dev)
        cum_t = torch.tensor(cum, device=dev)

        def integral(x: torch.Tensor) -> torch.Tensor:
            x = x.clamp(min=0.0, max=float(mesh[-1]))
            i = torch.searchsorted(mesh_t, x).clamp(1, len(mesh) - 1)
            a, b = mesh_t[i - 1], mesh_t[i]
            return cum_t[i - 1] + (cum_t[i] - cum_t[i - 1]) * (x - a) / (b - a)

        prod = torch.tensor(k.productivity(self.m), device=dev)
        tp = torch.tensor(self.t, device=dev)
        bg = k.mu * cell_areas()
        n_cells = GRID.n_cells
        real = np.lib.format.open_memmap(
            self.folder / "real.npy", "w+", np.float32, (self.n_days, n_cells)
        )
        fore = np.lib.format.open_memmap(
            self.folder / "fore.npy", "w+", np.float32, (self.n_days, n_cells)
        )
        far_day = np.zeros(self.n_days)
        order = np.argsort(pp, kind="stable")
        pp, cc, mass = pp[order], cc[order], mass[order]
        s = (
            torch.sparse_coo_tensor(
                torch.tensor(np.stack([pp, cc])), torch.tensor(mass), (len(self.t), n_cells)
            )
            .coalesce()
            .to(dev)
        )
        far_t = torch.tensor(far, device=dev)
        for d0 in range(0, self.n_days, 128):
            d1 = min(d0 + 128, self.n_days)
            a = torch.tensor(self.day0 + np.arange(d0, d1), device=dev)[:, None]
            b = a + 1.0
            alive = tp < float(self.day0 + d1)
            tj = tp[alive][None, :]
            w_real = (
                torch.where(tj < b, integral(b - tj) - integral(torch.maximum(a, tj) - tj), 0.0)
                * prod[alive][None, :]
            )
            w_fore = torch.where(tj < a, integral(b - tj) - integral(a - tj), 0.0) * prod[alive]
            n_alive = int(alive.sum())
            sub = torch.sparse_coo_tensor(
                s.indices()[:, s.indices()[0] < n_alive],
                s.values()[s.indices()[0] < n_alive],
                (n_alive, n_cells),
            )
            # alive is a prefix: the catalogue is sorted by time.
            real[d0:d1] = (
                (torch.sparse.mm(sub.t(), w_real.t()).t() + torch.tensor(bg, device=dev))
                .float()
                .cpu()
                .numpy()
            )
            fore[d0:d1] = (
                (torch.sparse.mm(sub.t(), w_fore.t()).t() + torch.tensor(bg, device=dev))
                .float()
                .cpu()
                .numpy()
            )
            far_day[d0:d1] = (w_real @ far_t[:n_alive]).cpu().numpy()
        real.flush()
        fore.flush()
        np.save(self.folder / "far_day.npy", far_day)
        log(f"CB-29 engine: day masses in {time.perf_counter() - t0:.0f}s")
        self._events(pp, cc, log)
        (self.folder / "done.json").write_text(
            json.dumps({"seconds": time.perf_counter() - t0, "version": ENGINE_VERSION}) + "\n",
            "utf-8",
        )

    def _events(self, pp: np.ndarray, cc: np.ndarray, log: Any) -> None:
        """μ + near and far kernels at every test quake (the near test: is the quake's cell
        within ``R_CUT_KM`` of the parent, the same rule as the cell masses)."""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        k = KERNEL
        test0 = (to_seconds(TEST_START) - to_seconds(AUX_START)) / DAY
        idx = np.nonzero(self.t >= test0)[0]
        cell = GRID.cell_of(self.lat, self.lon)
        clat, clon = GRID.cell_centres()
        ky = EARTH_RADIUS_KM * math.pi / 180
        tt = torch.tensor(self.t, device=dev)
        la = torch.tensor(np.radians(self.lat), device=dev)
        lo = torch.tensor(np.radians(self.lon), device=dev)
        lat_d = torch.tensor(self.lat, device=dev)
        lon_d = torch.tensor(self.lon, device=dev)
        prod = torch.tensor(k.productivity(self.m), device=dev)
        zone = torch.tensor(k.zone(self.m), device=dev)
        near = np.zeros(len(idx))
        farv = np.zeros(len(idx))
        for q, i in enumerate(idx):
            dt = tt[i] - tt[:i]
            h = (
                torch.sin((la[i] - la[:i]) / 2) ** 2
                + torch.cos(la[i]) * torch.cos(la[:i]) * torch.sin((lo[i] - lo[:i]) / 2) ** 2
            )
            r2 = (2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(h))) ** 2
            g = (
                prod[:i]
                * torch.exp(-dt / k.tau)
                / (dt + k.c) ** (1 + k.omega)
                / (r2 + zone[:i]) ** (1 + k.rho)
            )
            c = int(cell[i])
            kx = ky * torch.cos(torch.deg2rad(lat_d[:i]))
            ddx = (float(clon[c]) - lon_d[:i]) * kx
            ddy = (float(clat[c]) - lat_d[:i]) * ky
            is_near = ddx**2 + ddy**2 < R_CUT_KM**2
            near[q] = k.mu + float(g[is_near].sum())
            farv[q] = float(g[~is_near].sum())
        np.savez(self.folder / "events.npz", idx=idx, near=near, far=farv, cell=cell[idx])
        log(f"CB-29 engine: {len(idx):,} test quakes")

    # ---------------------------------------------------------------- reading it back

    def arrays(self) -> dict[str, Any]:
        ev = np.load(self.folder / "events.npz")
        return {
            "real": np.load(self.folder / "real.npy", mmap_mode="r"),
            "fore": np.load(self.folder / "fore.npy", mmap_mode="r"),
            "far_day": np.load(self.folder / "far_day.npy"),
            **{k: ev[k] for k in ev.files},
        }

    def score(self, factor: np.ndarray | None) -> dict[str, float]:
        """Negative log-likelihood per test quake (nats) of ETAS × ``factor``: a (test days,
        cells) array of e^c, or None for ETAS itself."""
        a = self.arrays()
        d0 = self.test_day0
        days = self.n_days - d0
        test_t = self.t[a["idx"]]
        day = np.floor(test_t - (self.day0 + d0)).astype(np.int64)
        f_at = 1.0 if factor is None else factor[day, a["cell"]]
        log_lam = np.log(f_at * a["near"] + a["far"])
        total = float(a["far_day"][d0:].sum())
        for i in range(0, days, 256):
            r = np.asarray(a["real"][d0 + i : d0 + i + 256], np.float64)
            total += float((r if factor is None else r * factor[i : i + 256]).sum())
        n = len(a["idx"])
        return {"nll": float(-(log_lam.sum() - total) / n), "quakes": float(n)}


def engine_dir(root: Path, ev: Events) -> Path:
    h = hashlib.sha256(
        json.dumps(
            {
                "v": ENGINE_VERSION,
                "n": len(ev),
                "t": float(ev.time.sum()),
                "r": R_CUT_KM,
                "rays": RAYS,
            }
        ).encode()
    ).hexdigest()[:12]
    return root / "data" / "world" / f"cb29-engine-{h}"


def engine_for(root: Path, ev: Events) -> Engine:
    t = (ev.time - to_seconds(AUX_START)) / DAY
    return Engine(engine_dir(root, ev), t, ev.lat, ev.lon, ev.mag)


# ---------------------------------------------------------------- the task


class EarthquakeNppTask(Task):
    task_id = "CB-29"
    name = "EarthquakeNPP"
    metric = "negative log-likelihood per test quake (nats), ComCat_25, their ETAS and splits"
    direction = "lower"

    def definition(self) -> dict[str, Any]:
        return {
            "benchmark": "EarthquakeNPP (Stockman, Lawson & Werner, TMLR 2026), "
            "https://github.com/ss15859/EarthquakeNPP",
            "dataset": DATASET,
            "region": REGION,
            "polygon": [list(p) for p in POLYGON],
            "windows": {
                "auxiliary": AUX_START,
                "train": TRAIN_START,
                "test": [TEST_START, TEST_END],
            },
            "mc": MC,
            "etas": {"parameters": ETAS_PARAMETERS, "area_km2": AREA_KM2, "beta": BETA},
            "published": PUBLISHED,
            "form": "λ = λ_ETAS · e^{c(day, cell)}, c fixed at each midnight; c = 0 is their ETAS",
            "grid": GRID.__dict__,
            "near": f"cells within {R_CUT_KM:g} km of a parent, masses on {RAYS} rays",
            "units": "nats per quake; days, km²",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.planets import events_table

        view = BenchContext(root=ctx.root, region=REGION)
        cat = catalogue(view.events)
        sha = _write_table(ctx.task_dir(self.task_id) / "catalogue.parquet", events_table(cat))
        test = cat.time >= to_seconds(TEST_START)
        return {
            "files": {"catalogue.parquet": sha},
            "summary": {
                "snapshot": view.snapshot_id,
                "quakes": len(cat),
                "test_quakes": int(test.sum()),
                "largest_test": float(cat.mag[test].max()),
            },
        }

    def catalogue(self, ctx: BenchContext) -> Events:
        from craton.bench.planets import read_events

        return read_events(ctx.task_dir(self.task_id) / "catalogue.parquet")

    def engine(self, ctx: BenchContext) -> Engine:
        key = "cb29-engine"
        if key not in ctx._cache:
            eng = engine_for(ctx.root, self.catalogue(ctx))
            eng.build()
            ctx._cache[key] = eng
        out: Engine = ctx._cache[key]
        return out

    def supports(self, method: Any) -> bool:
        return getattr(method, "name", "") == "etas" or callable(
            getattr(method, "world_forecaster", None)
        )

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        eng = self.engine(ctx)
        etas_nll = eng.score(None)["nll"]
        if getattr(method, "name", "") == "etas":
            factor = None
        else:
            factor = method.world_forecaster().npp_factor(eng, self.catalogue(ctx), seed)
        out = eng.score(factor)
        extras = {
            "quakes": out["quakes"],
            "etas_nll": etas_nll,
            "gain_over_etas": etas_nll - out["nll"],
            "published_etas_nll": PUBLISHED["etas_nll"],
            "published_poisson_nll": PUBLISHED["poisson_nll"],
        }
        if factor is not None:
            extras["factor_median"] = float(np.median(factor))
        return Score(out["nll"], extras)


register_task(EarthquakeNppTask())
