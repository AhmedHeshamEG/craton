"""ETAS: the standard model of how earthquakes trigger earthquakes.

The Epidemic-Type Aftershock Sequence model (Ogata 1988, 1998) says the rate of earthquakes at
time t and place (x, y) is a steady background plus a contribution from every earlier quake:

    λ(t, x, y) = μ·u(x, y) + Σ_{t_i < t}  k·e^{a(m_i − Mc)} · h(t − t_i) · f(x − x_i, y − y_i | m_i)

* ``k·e^{a(m − Mc)}``: how many direct aftershocks a quake of magnitude m has, on average
  (bigger quakes have exponentially more).
* ``h``: Omori-Utsu decay in time, ``(p−1)·c^{p−1}·(t + c)^{−p}``, normalised to 1.
* ``f``: a power-law spatial kernel whose size grows with magnitude,
  ``(q−1)/(π·σ)·(1 + r²/σ)^{−q}`` with ``σ = d·e^{γ(m − Mc)}`` (km²), cut at ``R_MAX`` and
  renormalised so it still integrates to 1.
* ``u``: where background quakes happen, a smoothed map of the catalogue weighted by each
  quake's probability of being background (stochastic declustering, Zhuang et al. 2002).

Magnitudes follow Gutenberg-Richter with a b-value fitted separately (Aki-Utsu).

Fitting maximises the exact log-likelihood over the fit window with automatic gradients
(PyTorch, float64, CPU) and L-BFGS-B. Forecasts integrate the intensity over each day and each
grid cell; quakes triggered *inside* the day are added with a mean-field cascade factor.
``simulate`` draws synthetic catalogues from the same model, which is how the tests check that a
fit recovers parameters it was given.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import ndtr

from craton.bench.csep import Grid, b_value

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: The spatial kernel is cut here (km) and renormalised; pairs further apart don't interact.
R_MAX = 50.0
#: Background smoothing bandwidth (km) and the share of background spread evenly over the box.
BANDWIDTH_KM = 5.0
BACKGROUND_FLOOR = 0.02
#: Days at the start of the fit window that act only as parents, never as targets.
BURN_IN_DAYS = 90.0
#: Gutenberg-Richter is truncated here when a mean over magnitudes is needed.
M_MAX = 8.0
#: Rounds of "fit, decluster, re-smooth the background".
BACKGROUND_ROUNDS = 3

DAY = 86_400.0


# ---------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Projection:
    """Degrees → km on a plane tangent at the box centre (equirectangular, fixed latitude)."""

    lat0: float
    lon0: float

    @property
    def kx(self) -> float:
        return 111.32 * math.cos(math.radians(self.lat0))

    ky: float = 110.57

    def to_km(self, lat: Any, lon: Any) -> tuple[FloatArray, FloatArray]:
        x = (np.asarray(lon, dtype=np.float64) - self.lon0) * self.kx
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) * self.ky
        return x, y

    @staticmethod
    def for_grid(grid: Grid) -> Projection:
        return Projection((grid.lat_min + grid.lat_max) / 2, (grid.lon_min + grid.lon_max) / 2)


@dataclass(frozen=True)
class Catalog:
    """A catalogue in model units: days since ``t_ref`` (seconds), km, magnitude."""

    t: FloatArray
    x: FloatArray
    y: FloatArray
    m: FloatArray

    def __len__(self) -> int:
        return len(self.t)

    def select(self, mask: Any) -> Catalog:
        return Catalog(self.t[mask], self.x[mask], self.y[mask], self.m[mask])


def to_catalog(
    time_s: FloatArray,
    lat: FloatArray,
    lon: FloatArray,
    mag: FloatArray,
    proj: Projection,
    t_ref: float,
) -> Catalog:
    x, y = proj.to_km(lat, lon)
    order = np.argsort(time_s, kind="stable")
    return Catalog(((time_s - t_ref) / DAY)[order], x[order], y[order], mag[order])


# ---------------------------------------------------------------- background


@dataclass(frozen=True)
class Background:
    """Smoothed background density over the box (per km²), integrating to exactly 1.

    A weighted sum of Gaussians, each renormalised by the share of it inside the box, mixed with
    a small uniform floor so no part of the box is impossible.
    """

    x: FloatArray
    y: FloatArray
    w: FloatArray
    box_km: tuple[float, float, float, float]  # x0, x1, y0, y1
    h: float = BANDWIDTH_KM
    floor: float = BACKGROUND_FLOOR

    @property
    def area(self) -> float:
        x0, x1, y0, y1 = self.box_km
        return (x1 - x0) * (y1 - y0)

    def _mass_x(self, a: float | FloatArray, b: float | FloatArray) -> FloatArray:
        return np.asarray(ndtr((b - self.x) / self.h) - ndtr((a - self.x) / self.h))

    def _mass_y(self, a: float | FloatArray, b: float | FloatArray) -> FloatArray:
        return np.asarray(ndtr((b - self.y) / self.h) - ndtr((a - self.y) / self.h))

    @property
    def _norm(self) -> float:
        x0, x1, y0, y1 = self.box_km
        return float(np.sum(self.w * self._mass_x(x0, x1) * self._mass_y(y0, y1)))

    def density(self, px: FloatArray, py: FloatArray, chunk: int = 2048) -> FloatArray:
        out = np.empty(len(px))
        norm = 2 * math.pi * self.h**2 * self._norm
        for s in range(0, len(px), chunk):
            dx = px[s : s + chunk, None] - self.x[None, :]
            dy = py[s : s + chunk, None] - self.y[None, :]
            k = np.exp(-(dx * dx + dy * dy) / (2 * self.h**2)) @ self.w
            out[s : s + chunk] = k / norm
        return (1 - self.floor) * out + self.floor / self.area

    def cell_mass(self, grid: Grid, proj: Projection) -> FloatArray:
        """Integral of the density over every grid cell (sums to 1 over the box)."""
        lat_e = grid.lat_min + grid.cell_deg * np.arange(grid.n_lat + 1)
        lon_e = grid.lon_min + grid.cell_deg * np.arange(grid.n_lon + 1)
        xe, _ = proj.to_km(np.zeros_like(lon_e), lon_e)
        _, ye = proj.to_km(lat_e, np.zeros_like(lat_e))
        mx = np.stack([self._mass_x(xe[j], xe[j + 1]) for j in range(grid.n_lon)])  # lon × N
        my = np.stack([self._mass_y(ye[i], ye[i + 1]) for i in range(grid.n_lat)])  # lat × N
        smooth = (my * self.w) @ mx.T  # lat × lon
        cell_area = np.outer(np.diff(ye), np.diff(xe))
        mass = (1 - self.floor) * smooth / self._norm + self.floor * cell_area / self.area
        return np.asarray(mass.ravel(), dtype=np.float64)


def box_km(grid: Grid, proj: Projection) -> tuple[float, float, float, float]:
    x0, y0 = proj.to_km(np.array([grid.lat_min]), np.array([grid.lon_min]))
    x1, y1 = proj.to_km(np.array([grid.lat_max]), np.array([grid.lon_max]))
    return float(x0[0]), float(x1[0]), float(y0[0]), float(y1[0])


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class EtasParams:
    mu: float  # background events per day in the whole box, at or above mc
    k: float  # productivity
    a: float  # productivity growth with magnitude (per magnitude unit, natural log)
    c: float  # days
    p: float  # Omori exponent
    d: float  # km², spatial scale of an mc quake
    gamma: float  # spatial scale growth with magnitude
    q: float  # spatial power-law exponent

    NAMES = ("mu", "k", "a", "c", "p", "d", "gamma", "q")

    def to_vector(self) -> FloatArray:
        return np.array(
            [
                math.log(self.mu),
                math.log(self.k),
                self.a,
                math.log(self.c),
                math.log(self.p - 1),
                math.log(self.d),
                self.gamma,
                math.log(self.q - 1),
            ]
        )

    @staticmethod
    def from_vector(v: FloatArray) -> EtasParams:
        return EtasParams(
            mu=math.exp(v[0]),
            k=math.exp(v[1]),
            a=float(v[2]),
            c=math.exp(v[3]),
            p=1 + math.exp(v[4]),
            d=math.exp(v[5]),
            gamma=float(v[6]),
            q=1 + math.exp(v[7]),
        )

    def as_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in self.NAMES}


#: Search box for L-BFGS-B, in the vector's units.
BOUNDS = (
    (math.log(1e-4), math.log(1e3)),
    (math.log(1e-4), math.log(10.0)),
    (0.0, 3.5),
    (math.log(1e-5), math.log(1.0)),
    (math.log(1e-3), math.log(2.0)),
    (math.log(1e-2), math.log(1e2)),
    (0.0, 3.0),
    (math.log(1e-2), math.log(3.0)),
)


def omori_integral(t: FloatArray, c: float, p: float) -> FloatArray:
    """H(t): share of a quake's aftershocks that arrive within t days (0 for t <= 0)."""
    tt = np.maximum(t, 0.0)
    return np.asarray(1.0 - (c / (tt + c)) ** (p - 1), dtype=np.float64)


def mean_productivity(params: EtasParams, b: float, mc: float, m_max: float = M_MAX) -> float:
    """Mean of k·e^{a(m−mc)} over Gutenberg-Richter magnitudes in [mc, m_max]."""
    beta = b * math.log(10)
    span = m_max - mc
    if abs(params.a - beta) < 1e-9:
        mean = beta * span / (1 - math.exp(-beta * span))
    else:
        mean = (
            beta
            * (math.exp((params.a - beta) * span) - 1)
            / ((params.a - beta) * (1 - math.exp(-beta * span)))
        )
    return params.k * mean


# ---------------------------------------------------------------- likelihood


@dataclass(frozen=True)
class Pairs:
    """Every (parent, target) pair closer than ``R_MAX``, parent strictly earlier."""

    parent: IntArray
    target: IntArray  # index into the target list
    dt: FloatArray
    r2: FloatArray


def find_pairs(cat: Catalog, targets: IntArray, chunk: int = 256) -> Pairs:
    """Pairs between each target and every earlier event within ``R_MAX`` km."""
    parents, tgt, dts, r2s = [], [], [], []
    for s in range(0, len(targets), chunk):
        tj = targets[s : s + chunk]
        hi = int(tj.max()) + 1
        dx = cat.x[None, :hi] - cat.x[tj, None]
        dy = cat.y[None, :hi] - cat.y[tj, None]
        r2 = dx * dx + dy * dy
        dt = cat.t[tj, None] - cat.t[None, :hi]
        ok = (r2 <= R_MAX**2) & (dt > 0)
        rows, cols = np.nonzero(ok)
        parents.append(cols.astype(np.int64))
        tgt.append((rows + s).astype(np.int64))
        dts.append(dt[rows, cols])
        r2s.append(r2[rows, cols])
    return Pairs(
        np.concatenate(parents), np.concatenate(tgt), np.concatenate(dts), np.concatenate(r2s)
    )


@dataclass(frozen=True)
class FitWindow:
    """Targets are events in [t_start, t_end) days; earlier events act only as parents."""

    t_start: float
    t_end: float


def _make_nll(
    cat: Catalog, mc: float, window: FitWindow, targets: IntArray, pairs: Pairs, u: FloatArray
) -> Callable[[FloatArray], tuple[float, FloatArray]]:
    import torch

    f64 = torch.float64
    mp = torch.tensor(cat.m - mc, dtype=f64)
    pi = torch.tensor(pairs.parent)
    tj = torch.tensor(pairs.target)
    dt = torch.tensor(pairs.dt, dtype=f64)
    r2 = torch.tensor(pairs.r2, dtype=f64)
    u_t = torch.tensor(u, dtype=f64)
    n_t = len(targets)
    before = cat.t < window.t_end
    tp = torch.tensor(cat.t[before], dtype=f64)
    mpi = mp[torch.tensor(np.nonzero(before)[0])]
    span = window.t_end - window.t_start
    m_pair = mp[pi]
    big_r2 = R_MAX**2

    def nll(v: FloatArray) -> tuple[float, FloatArray]:
        th = torch.tensor(v, dtype=f64, requires_grad=True)
        log_mu, log_k, a, log_c, log_pm1, log_d, gamma, log_qm1 = th
        p = 1 + torch.exp(log_pm1)
        q = 1 + torch.exp(log_qm1)
        c = torch.exp(log_c)
        log_sigma = log_d + gamma * m_pair
        sigma = torch.exp(log_sigma)
        trunc = -torch.expm1((1 - q) * torch.log1p(big_r2 / sigma))
        log_g = (
            log_k
            + a * m_pair
            + log_pm1
            + (p - 1) * log_c
            - p * torch.log(dt + c)
            + log_qm1
            - math.log(math.pi)
            - log_sigma
            - q * torch.log1p(r2 / sigma)
            - torch.log(trunc)
        )
        lam = torch.exp(log_mu) * u_t
        lam = lam.index_add(0, tj, torch.exp(log_g))
        h_end = 1 - (c / (window.t_end - tp + c)) ** (p - 1)
        h_start = 1 - (c / (torch.clamp(window.t_start - tp, min=0.0) + c)) ** (p - 1)
        expected = torch.exp(log_mu) * span + torch.sum(
            torch.exp(log_k + a * mpi) * (h_end - h_start)
        )
        loss = -(torch.sum(torch.log(lam)) - expected) / n_t
        loss.backward()
        assert th.grad is not None
        return float(loss.item()), th.grad.numpy().astype(np.float64)

    return nll


@dataclass(frozen=True)
class EtasFit:
    params: EtasParams
    background: Background
    mc: float
    b: float
    t_ref: float  # seconds since 1970 of day 0
    window: FitWindow
    n_targets: int
    log_likelihood: float
    converged: bool

    @property
    def branching_ratio(self) -> float:
        """Average number of direct aftershocks per quake; < 1 or the model explodes."""
        return mean_productivity(self.params, self.b, self.mc)


DEFAULT_START = EtasParams(mu=1.0, k=0.2, a=1.5, c=0.01, p=1.1, d=1.0, gamma=1.0, q=1.5)


def fit(
    cat: Catalog,
    mc: float,
    window: FitWindow,
    box: tuple[float, float, float, float],
    t_ref: float,
    seed: int = 0,
    start: EtasParams = DEFAULT_START,
    rounds: int = BACKGROUND_ROUNDS,
    max_iter: int = 500,
    mag_step: float = 0.1,
) -> EtasFit:
    """Maximum-likelihood ETAS on events at or above ``mc``.

    ``seed`` jitters the starting point (±10% in every log-parameter), so three seeds are three
    honest restarts of the optimiser.
    """
    cat = cat.select((cat.m >= mc - 1e-9) & (cat.t < window.t_end))
    targets = np.nonzero(cat.t >= window.t_start)[0]
    if len(targets) < 50:
        raise ValueError(f"only {len(targets)} target events; ETAS needs more")
    pairs = find_pairs(cat, targets)
    b = b_value(cat.m, mc, mag_step)
    rng = np.random.default_rng(seed)
    v = start.to_vector()
    v = v + rng.normal(0, 0.1, size=v.shape) * np.maximum(np.abs(v), 0.5)
    span = window.t_end - window.t_start
    v[0] = math.log(max(len(targets) / span / 2, 1e-3))
    v = np.clip(v, [lo for lo, _ in BOUNDS], [hi for _, hi in BOUNDS])

    weights = np.ones(len(cat))
    result = None
    for _ in range(rounds):
        bg = Background(cat.x, cat.y, weights, box)
        u = bg.density(cat.x[targets], cat.y[targets])
        nll = _make_nll(cat, mc, window, targets, pairs, u)
        result = minimize(
            nll,
            v,
            jac=True,
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-7},
        )
        v = result.x
        params = EtasParams.from_vector(v)
        lam_bg = params.mu * u
        trig = _pair_rates(params, cat, pairs, mc)
        lam = lam_bg + np.bincount(pairs.target, weights=trig, minlength=len(targets))
        weights = np.zeros(len(cat))
        weights[targets] = lam_bg / lam
        # Parents before the window have no probability of their own; treat them as background.
        weights[: targets[0]] = 1.0
    assert result is not None
    bg = Background(cat.x, cat.y, weights, box)
    return EtasFit(
        params=EtasParams.from_vector(result.x),
        background=bg,
        mc=mc,
        b=b,
        t_ref=t_ref,
        window=window,
        n_targets=len(targets),
        log_likelihood=-float(result.fun) * len(targets),
        converged=bool(result.success),
    )


def _spatial(params: EtasParams, r2: FloatArray, m_rel: FloatArray) -> FloatArray:
    sigma = params.d * np.exp(params.gamma * m_rel)
    trunc = 1 - (1 + R_MAX**2 / sigma) ** (1 - params.q)
    dens = (params.q - 1) / (math.pi * sigma) * (1 + r2 / sigma) ** (-params.q) / trunc
    return np.asarray(dens, dtype=np.float64)


def _pair_rates(params: EtasParams, cat: Catalog, pairs: Pairs, mc: float) -> FloatArray:
    m_rel = cat.m[pairs.parent] - mc
    prod = params.k * np.exp(params.a * m_rel)
    omori = (params.p - 1) * params.c ** (params.p - 1) * (pairs.dt + params.c) ** (-params.p)
    return np.asarray(prod * omori * _spatial(params, pairs.r2, m_rel), dtype=np.float64)


# ---------------------------------------------------------------- forecasting


def _fine_points(grid: Grid, proj: Projection, per_cell: int = 5) -> tuple[Any, ...]:
    """Quadrature points: ``per_cell``² per cell, extended ``R_MAX`` beyond the box."""
    step = grid.cell_deg / per_cell
    margin_lat = math.ceil(R_MAX / proj.ky / step) + 1
    margin_lon = math.ceil(R_MAX / proj.kx / step) + 1
    ilat = np.arange(-margin_lat, grid.n_lat * per_cell + margin_lat)
    ilon = np.arange(-margin_lon, grid.n_lon * per_cell + margin_lon)
    lat = grid.lat_min + step * (ilat + 0.5)
    lon = grid.lon_min + step * (ilon + 0.5)
    _, ys = proj.to_km(lat, np.zeros_like(lat))
    xs, _ = proj.to_km(np.zeros_like(lon), lon)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    cells = grid.cell_of(la.ravel(), lo.ravel()).reshape(la.shape)
    return xs, ys, cells, step


def spatial_matrix(
    params: EtasParams, cat: Catalog, mc: float, grid: Grid, proj: Projection, chunk: int = 256
) -> sp.csr_matrix:
    """(events × cells) share of each event's aftershock kernel landing in each cell.

    The kernel is sampled on a fine grid and renormalised to its in-disc total, so a narrow
    kernel keeps its full mass instead of vanishing between quadrature points. Mass that falls
    outside the box is lost, as it should be: those aftershocks are outside the forecast.
    """
    xs, ys, cells, _ = _fine_points(grid, proj)
    dxs, dys = xs[1] - xs[0], ys[1] - ys[0]
    rx, ry = math.ceil(R_MAX / dxs) + 1, math.ceil(R_MAX / dys) + 1
    grid_y, grid_x = np.meshgrid(np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij")
    oy, ox = grid_y.ravel(), grid_x.ravel()
    rows, cols, vals = [], [], []
    for s in range(0, len(cat), chunk):
        sl = slice(s, s + chunk)
        ix = np.rint((cat.x[sl] - xs[0]) / dxs).astype(np.int64)
        iy = np.rint((cat.y[sl] - ys[0]) / dys).astype(np.int64)
        gx = ix[:, None] + ox[None, :]
        gy = iy[:, None] + oy[None, :]
        valid = (gx >= 0) & (gx < len(xs)) & (gy >= 0) & (gy < len(ys))
        gxc, gyc = np.clip(gx, 0, len(xs) - 1), np.clip(gy, 0, len(ys) - 1)
        r2 = (xs[gxc] - cat.x[sl, None]) ** 2 + (ys[gyc] - cat.y[sl, None]) ** 2
        inside = valid & (r2 <= R_MAX**2)
        m_rel = np.broadcast_to((cat.m[sl] - mc)[:, None], r2.shape)
        w = np.where(inside, _spatial(params, r2, m_rel), 0.0)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        cell = np.where(inside, cells[gyc, gxc], -1)
        keep = cell >= 0
        r, cidx = np.nonzero(keep)
        rows.append(r + s)
        cols.append(cell[r, cidx])
        vals.append(w[r, cidx])
    mat = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(cat), grid.n_cells),
    )
    return mat.tocsr()


def cascade_factor(params: EtasParams, b: float, mc: float, span_days: float) -> float:
    """How much quakes triggered inside the window add, in the mean-field approximation.

    A quake at a uniform random moment of the window has, on average, ``n_w`` direct
    aftershocks still inside it; those have their own, and so on: 1 + n_w + n_w² + … .
    """
    s = np.linspace(0, span_days, 201)[:-1] + span_days / 400
    inside = float(np.mean(omori_integral(span_days - s, params.c, params.p)))
    n_w = min(mean_productivity(params, b, mc) * inside, 0.95)
    return 1.0 / (1.0 - n_w)


@dataclass(frozen=True)
class DailyForecast:
    """Expected counts per (day, cell) at or above ``mc``; magnitudes follow GR with ``b``."""

    rates: FloatArray  # (days, cells)
    mc: float
    b: float


def forecast_days(
    fit: EtasFit,
    history: Catalog,
    day_starts: FloatArray,
    grid: Grid,
    proj: Projection,
) -> DailyForecast:
    """Daily ETAS forecasts. ``history`` holds every quake known by each day (model units).

    Day ``d`` covers [day_starts[d], day_starts[d] + 1) and sees only quakes before it starts.
    """
    pr = fit.params
    parents = history.select(history.m >= fit.mc - 1e-9)
    smat = spatial_matrix(pr, parents, fit.mc, grid, proj)
    prod = pr.k * np.exp(pr.a * (parents.m - fit.mc))
    t0 = day_starts[:, None]
    seen = parents.t[None, :] < t0
    w = prod[None, :] * (
        omori_integral(t0 + 1 - parents.t[None, :], pr.c, pr.p)
        - omori_integral(t0 - parents.t[None, :], pr.c, pr.p)
    )
    w = np.where(seen, w, 0.0)
    triggered = np.asarray((smat.T @ w.T).T)
    background = pr.mu * fit.background.cell_mass(grid, proj)
    factor = cascade_factor(pr, fit.b, fit.mc, 1.0)
    return DailyForecast((background[None, :] + triggered) * factor, fit.mc, fit.b)


# ---------------------------------------------------------------- simulation


def simulate(
    params: EtasParams,
    b: float,
    mc: float,
    days: float,
    box: tuple[float, float, float, float],
    seed: int,
    m_max: float = M_MAX,
) -> Catalog:
    """A synthetic catalogue from the model: uniform background, branching aftershocks.

    Quakes are generated on an unbounded plane and kept only inside ``box``, so the fitted
    catalogue has the same edge effects as a real one.
    """
    rng = np.random.default_rng(seed)
    beta = b * math.log(10)

    def mags(n: int) -> FloatArray:
        u = rng.random(n)
        return mc - np.log(1 - u * (1 - math.exp(-beta * (m_max - mc)))) / beta

    x0, x1, y0, y1 = box
    n0 = rng.poisson(params.mu * days)
    gen = Catalog(
        rng.uniform(0, days, n0), rng.uniform(x0, x1, n0), rng.uniform(y0, y1, n0), mags(n0)
    )
    parts = [gen]
    while len(gen):
        n_kids = rng.poisson(params.k * np.exp(params.a * (gen.m - mc)))
        idx = np.repeat(np.arange(len(gen)), n_kids)
        if not len(idx):
            break
        u = rng.random(len(idx))
        dt = params.c * ((1 - u) ** (-1 / (params.p - 1)) - 1)
        sigma = params.d * np.exp(params.gamma * (gen.m[idx] - mc))
        z = 1 - (1 + R_MAX**2 / sigma) ** (1 - params.q)
        r2 = sigma * ((1 - rng.random(len(idx)) * z) ** (1 / (1 - params.q)) - 1)
        ang = rng.uniform(0, 2 * math.pi, len(idx))
        r = np.sqrt(r2)
        kids = Catalog(
            gen.t[idx] + dt,
            gen.x[idx] + r * np.cos(ang),
            gen.y[idx] + r * np.sin(ang),
            mags(len(idx)),
        )
        gen = kids.select(kids.t < days)
        parts.append(gen)
    allq = Catalog(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("t", "x", "y", "m")))
    inside = (allq.x >= x0) & (allq.x < x1) & (allq.y >= y0) & (allq.y < y1)
    allq = allq.select(inside)
    return allq.select(np.argsort(allq.t, kind="stable"))


def with_mu(params: EtasParams, mu: float) -> EtasParams:
    return replace(params, mu=mu)
