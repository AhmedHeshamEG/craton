"""A second ETAS referee: ETH Zurich's ETAS (lmizrahi/etas, MIT) on CB-02 and CB-03.

Every gain Craton claims is measured against ETAS, and the ETAS is Craton's own code (D-033). A
reviewer's first question is whether that ETAS is weak. This one answers it with the code the
Swiss Seismological Service's forecasts are built on (Mizrahi et al. 2021, 2024):

1. ``--export`` writes the catalogue; ``scripts/reference/eth_etas_invert.py`` (a throwaway
   environment; ETH's package pins its own stack) fits ETH's ETAS on exactly CB-02's / CB-03's
   training data: INGV M2+, central Italy's box, primary window 2005 to the cut, every quake
   since 1990 as a possible parent. It writes ``results/reference/eth_etas-<task>.json``.
2. This script turns them into the same daily 0.1° forecasts every method makes: a uniform
   background plus every quake known before midnight, each with ETH's kernels (Omori with an
   exponential taper in time; a magnitude-dependent power law in space), integrated over the day
   and the cell. First generation only, as Craton's ETAS forecast is.
3. Scored by ``score_forecast``, next to Craton's own ETAS.

    uv run python scripts/ablations/eth_etas_referee.py CB-02 CB-03
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.baselines import Etas, _shares
from craton.bench.data import DAY
from craton.bench.tasks import TASKS, BenchContext, GriddedForecast, score_forecast

ROOT = Path(__file__).resolve().parents[2]
KM = 111.32
SUB = 5  # sub-points per cell side for the space integral
NODES, WEIGHTS = np.polynomial.legendre.leggauss(24)


def time_integral(x0: np.ndarray, x1: np.ndarray, c: float, omega: float, tau: float) -> np.ndarray:
    """∫ e^(−x/τ) (x + c)^(−1−ω) dx from x0 to x1 (days), by Gauss-Legendre in log(x + c)."""
    u0, u1 = np.log(x0 + c), np.log(x1 + c)
    mid, half = (u0 + u1) / 2, (u1 - u0) / 2
    u = mid[..., None] + half[..., None] * NODES
    y = np.exp(u)  # x + c
    f = np.exp(-(y - c) / tau) * y ** (-1 - omega) * y  # dx = y du
    return np.asarray(half * (f * WEIGHTS).sum(-1), dtype=np.float64)


def space_matrix(
    lat: np.ndarray, lon: np.ndarray, m: np.ndarray, th: dict, mc: float, grid
) -> np.ndarray:
    """(events, cells) ∫ over each cell of (r² + d e^(γ(m − mc)))^(−1−ρ) dA (km²)."""
    d, gamma, rho = 10 ** th["log10_d"], th["gamma"], th["rho"]
    clat, clon = grid.cell_centres()
    step = grid.cell_deg
    off = (np.arange(SUB) + 0.5) / SUB - 0.5
    plat = (clat[:, None, None] + off[None, :, None] * step + 0 * off[None, None, :]).reshape(
        len(clat), -1
    )
    plon = (clon[:, None, None] + 0 * off[None, :, None] + off[None, None, :] * step).reshape(
        len(clon), -1
    )
    kx = KM * np.cos(np.radians(clat))
    area = (step * KM) * (step * kx)  # km² per cell
    out = np.zeros((len(m), len(clat)), np.float32)
    own = grid.cell_of(lat, lon)
    for s in range(0, len(m), 400):
        sl = slice(s, s + 400)
        dd = d * np.exp(gamma * (m[sl] - mc))
        dy = (plat[None] - lat[sl, None, None]) * KM
        dx = (plon[None] - lon[sl, None, None]) * kx[None, :, None]
        dens = (dx**2 + dy**2 + dd[:, None, None]) ** (-1 - rho)
        out[sl] = dens.mean(-1) * area[None]
        # The event's own cell: the kernel is far narrower than a sub-point there, so its mass
        # is the total minus what lies beyond a disc of the cell's area.
        r2 = area[own[sl].clip(0)] / math.pi
        inner = math.pi / rho * (dd ** (-rho) - (r2 + dd) ** (-rho))
        rows = np.flatnonzero(own[sl] >= 0)
        out[s + rows, own[sl][rows]] = inner[rows]
    return out


def forecast(task_id: str, ctx: BenchContext) -> tuple[GriddedForecast, dict]:
    task = TASKS[task_id]
    spec = task.spec  # type: ignore[attr-defined]
    ref = json.loads((ROOT / "results" / "reference" / f"eth_etas-{task_id}.json").read_text())
    th, mc, beta = ref["theta"], ref["mc"], ref["beta"]
    g = spec.grid
    ev = ctx.events
    inside = (
        (ev.lat >= g.lat_min) & (ev.lat < g.lat_max) & (ev.lon >= g.lon_min) & (ev.lon < g.lon_max)
    )
    ev = ev.select(inside & (ev.mag >= mc - 0.05) & (ev.time >= 631_152_000.0))  # 1990 on
    ev = ev.select(ev.time < spec.window.t1)
    k0, a = 10 ** th["log10_k0"], th["a"]
    c, omega, tau = 10 ** th["log10_c"], th["omega"], 10 ** th["log10_tau"]
    starts = spec.day_starts
    kmat = space_matrix(ev.lat, ev.lon, ev.mag, th, mc, g)
    prod = k0 * np.exp(a * (ev.mag - mc))
    clat, _ = g.cell_centres()
    area = (g.cell_deg * KM) ** 2 * np.cos(np.radians(clat))
    rates = np.zeros((len(starts), g.n_cells))
    for i, t in enumerate(starts):
        seen = np.flatnonzero(ev.time < t)
        x0 = (t - ev.time[seen]) / DAY
        w = np.zeros(len(ev), np.float32)
        w[seen] = prod[seen] * time_integral(x0, x0 + spec.period_days, c, omega, tau)
        rates[i] = w @ kmat + 10 ** th["log10_mu"] * area * spec.period_days
    scale, shares = _shares(beta / math.log(10), mc, g)
    return GriddedForecast(rates * scale, shares), {"n_parents": len(ev), **th, "beta": beta}


def export(ctx: BenchContext) -> None:
    """Central Italy's M2+ quakes in CB-02's box, for the inversion's own environment."""
    import datetime as dt

    g = TASKS["CB-02"].spec.grid  # type: ignore[attr-defined]
    ev = ctx.events
    inside = (ev.lat >= g.lat_min) & (ev.lat < g.lat_max) & (ev.lon >= g.lon_min)
    ev = ev.select(inside & (ev.lon < g.lon_max) & (ev.mag >= 1.95))
    path = ROOT / "results" / "reference" / "central_italy_m2.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("id,latitude,longitude,time,magnitude\n")
        for i in range(len(ev)):
            t = dt.datetime.fromtimestamp(float(ev.time[i]), dt.UTC).replace(tzinfo=None)
            f.write(f"{i},{ev.lat[i]:.5f},{ev.lon[i]:.5f},{t.isoformat()},{ev.mag[i]:.1f}\n")
    print(f"wrote {path.relative_to(ROOT)}: {len(ev):,} quakes")


def main(tasks: list[str]) -> None:
    ctx = BenchContext(root=ROOT)
    out = {}
    for task_id in tasks:
        task = TASKS[task_id]
        spec = task.spec  # type: ignore[attr-defined]
        obs = task.observed(ctx)  # type: ignore[attr-defined]
        fc, info = forecast(task_id, ctx)
        eth = score_forecast(fc, obs, spec, 0)
        ours = score_forecast(Etas().forecast(spec, ctx, 0), obs, spec, 0)
        e3 = float(fc.expected(int(spec.grid.mag_bin_of(np.array([3.0]))[0])).sum())
        out[task_id] = {
            "eth_etas": eth.value,
            "craton_etas": ours.value,
            "eth_minus_craton": eth.value - ours.value,
            "eth_expected_m3": e3,
            "eth_how_many": eth.extras["M3.how_many_bits"],
            "eth_where": eth.extras["M3.where_bits"],
            "craton_how_many": ours.extras["M3.how_many_bits"],
            "craton_where": ours.extras["M3.where_bits"],
            "parameters": info,
        }
        print(
            f"{task_id}: ETH ETAS {eth.value:.3f}, Craton's ETAS {ours.value:.3f} "
            f"(ETH - ours {eth.value - ours.value:+.3f}; how many "
            f"{eth.extras['M3.how_many_bits'] - ours.extras['M3.how_many_bits']:+.3f}, where "
            f"{eth.extras['M3.where_bits'] - ours.extras['M3.where_bits']:+.3f}); "
            f"ETH expects {e3:.0f} M3+",
            flush=True,
        )
    path = ROOT / "results" / "ablation" / "eth_etas_referee.json"
    path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--export"]:
        export(BenchContext(root=ROOT))
    else:
        main(sys.argv[1:] or ["CB-02", "CB-03"])
