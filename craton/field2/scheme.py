"""Field 2, the converged scheme (P11, decided with Hesham on 2026-11-13 evening).

Raw quakes in, pure numbers all the way, ETAS as the straight line the network bends:

1. **Tanks.** Every place (a 0.1° cell) holds 16 count tanks and 16 size tanks. A quake from
   mc − 0.5 pours into the tanks of the place it falls in; tank k leaks with its own lifetime
   τ_k (learned, born log-spaced from an hour to thirty years), so a tank holds
   Σ e^{−age/τ_k}. Exponentials of many lifetimes add up to Omori's 1/t: the bank *is* a
   multi-timescale memory with no fixed boxes. A size tank pours 10^{β_k (m − mc)} instead of 1
   (β_k learned). Each tank is read over what the place's usual rate would have poured in:
   a pure number, "how full, over usual".
2. **What else a place says** (pure numbers): the largest quake it heard (size − mc, its age in
   Omori c), its mean depth over the region's thickness, its usual rate over the box's, and the
   region's mc, b and branching number.
3. **Places see each other.** 128 numbers per place, 3 pre-norm encoder layers, 4 heads, full
   attention over the tile's places with a learned bias on relative distance: in cells, and in
   rupture lengths of the box's largest quake of the last 30 days. No place code, no latitude
   or longitude.
4. **The equation layer.** With s a place's 128 numbers after attention,

       λ/cascade = e^{b(s)}·background + Σ_i ETAS_i · e^{g(s) + 1[age_i < T]·N(π_i, s)}

   where ETAS_i is quake i's ETAS weight in that place on that day (its productivity, the Omori
   integral over the day and its kernel's share there: ETAS's straight line in the logs), g a
   gain per place, N a two-layer network on the quake's pure numbers π (age in c, size − mc,
   distance in rupture lengths) and the place, only for quakes younger than T (ten days), and
   b the background's own say. The last layer of g, N and b starts at zero: **the untrained
   model is ETAS exactly** (``tests/test_field2.py``). Each is bounded smoothly to ×max_nudge
   either way, as the old Field was.

Training is the old world Field's: the same pool of tile-days, ETAS fitted in-sample on each
tile, lr 3e-4, early stopping on held-back days of every training region
(``train_scheme``). One tile per step; the days of a step pass in small chunks, each with its
own backward, so a big sequence's recent-quake pairs fit on a 6 GB card.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from craton.bench.csep import Grid
from craton.bench.data import DAY, Events
from craton.field2.config import Field2Config, HeartBins, SchemeNet
from craton.field2.heart import CONTEXT, Box, DeviceBox, to_device

Log = Callable[[str], None]

#: The box's largest quake of this many days sets the attention's rupture-length ruler.
RULER_DAYS = 30.0
#: Lower edges of the attention's distance bins: in cells (0.1°, ~11 km) ...
CELL_EDGES = (0.0, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0, 64.0)
#: ... and in rupture lengths of the box's largest recent quake.
RUPTURE_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
#: Days of a training step that pass forward and backward together.
TRAIN_CHUNK = 2
#: Days forecast together.
INFER_CHUNK = 4
KM_PER_DEG = 111.195
#: Size tanks' β at birth (amount 10^{β(m − mc)}), from below counting to above energy.
BETA_RANGE = (0.25, 1.5)


def _quiet(_: str) -> None:
    return None


# ---------------------------------------------------------------- what the tanks hear


@dataclass
class TankQuakes:
    """Quakes a box's tanks hear: days since its ETAS's t_ref, cell, m − mc, depth (km)."""

    t: np.ndarray  # float64
    cell: np.ndarray  # int64
    m_rel: np.ndarray  # float32
    depth: np.ndarray  # float32

    def __len__(self) -> int:
        return len(self.t)


def tank_quakes(
    events: Events, grid: Grid, mc: float, below: float, t_ref: float, t0: float, t1: float
) -> TankQuakes:
    """Every quake in ``grid`` from mc − ``below``, in [t0, t1) seconds."""
    ev = events.window(t0, t1).above(mc - below)
    cell = grid.cell_of(ev.lat, ev.lon)
    ok = cell >= 0
    return TankQuakes(
        np.asarray((ev.time[ok] - t_ref) / DAY, np.float64),
        np.asarray(cell[ok], np.int64),
        np.asarray(ev.mag[ok] - mc, np.float32),
        np.asarray(np.nan_to_num(ev.depth_km[ok], nan=10.0), np.float32),
    )


def save_quakes(path: Path, q: TankQuakes) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, t=q.t, cell=q.cell.astype(np.int32), m_rel=q.m_rel, depth=q.depth)
    tmp.replace(path)


def load_quakes(path: Path) -> TankQuakes:
    z = np.load(path)
    return TankQuakes(z["t"], z["cell"].astype(np.int64), z["m_rel"], z["depth"])


def thickness_km(q: TankQuakes, before: float) -> float:
    """The region's thickness, the depth ruler: 90% of its quakes before ``before`` (days)
    are shallower."""
    d = q.depth[q.t < before]
    return float(max(np.percentile(d, 90), 5.0)) if len(d) else 15.0


def parent_m_rel(box: Box, q: TankQuakes, bins: HeartBins) -> np.ndarray:
    """m − mc of every ETAS parent of ``box``, read from the tanks' quakes (the same events);
    a parent with no twin (never seen so far) falls back on its size bin's lower edge."""
    order = np.argsort(q.t, kind="stable")
    ts = q.t[order]
    i = np.clip(np.searchsorted(ts, box.t), 0, max(len(ts) - 1, 0))
    out = np.asarray(bins.magnitude, np.float32)[box.mag_bin]
    if len(ts):
        hit = np.abs(ts[i] - box.t) < 1e-6
        out = np.where(hit, q.m_rel[order][i], out).astype(np.float32)
    return out


# ---------------------------------------------------------------- one box on the device


@dataclass
class SchemeBox:
    """A box ready for ``SchemeModel``: ETAS's pieces, the tanks' quakes and the rulers."""

    db: DeviceBox
    q_t: torch.Tensor  # float64 days
    q_cell: torch.Tensor  # int64
    q_m: torch.Tensor  # float32
    q_depth: torch.Tensor  # float32
    p_m: torch.Tensor  # float32 parents' m − mc
    e_ring_x: torch.Tensor  # float32 each piece's ring centre on the log ruler, about ±1
    dist_km: torch.Tensor  # (C, C) float32
    cell_bin: torch.Tensor  # (C, C) int64
    usual: torch.Tensor  # (C,) quakes a day from mc − below, the place's background
    usual_x: torch.Tensor  # (C,) log10 of it over the box's mean, /2
    mc: float
    b: float
    branching: float
    thickness: float
    n_cells: int


def ring_centres(edges: tuple[float, ...]) -> np.ndarray:
    e = np.asarray(edges, np.float64)
    hi = np.append(e[1:], e[-1] * 3.0)
    lo = np.where(e > 0, e, hi / 2)
    return np.asarray(np.log10(np.sqrt(lo * hi)) / 1.5, np.float32)


def scheme_box(
    box: Box,
    q: TankQuakes,
    grid: Grid,
    bins: HeartBins,
    net: SchemeNet,
    thickness: float,
    dev: torch.device,
) -> SchemeBox:
    db = to_device(box, bins, dev)
    n_rings = len(bins.ruler)
    ring = box.entries.ring if len(box.entries) else np.zeros(0, np.int64)
    la, lo = (np.asarray(v, np.float64) for v in grid.cell_centres())  # one per cell
    cos = math.cos(math.radians(float(la.mean())))
    dy = (la[:, None] - la[None, :]) * KM_PER_DEG
    dx = (lo[:, None] - lo[None, :]) * KM_PER_DEG * cos
    dist = np.sqrt(dx**2 + dy**2).astype(np.float32)
    cells = dist / (grid.cell_deg * KM_PER_DEG)
    cell_bin = np.searchsorted(np.asarray(CELL_EDGES), cells, side="right") - 1
    ctx = box.context
    b = float(ctx[CONTEXT.index("b")])
    usual = box.background * 10.0 ** (b * net.hears_below_mc)
    usual = np.maximum(usual, 1e-9)
    p_m = parent_m_rel(box, q, bins)
    assert n_rings > 0
    return SchemeBox(
        db=db,
        q_t=torch.as_tensor(q.t, dtype=torch.float64, device=dev),
        q_cell=torch.as_tensor(q.cell, dtype=torch.int64, device=dev),
        q_m=torch.as_tensor(q.m_rel, dtype=torch.float32, device=dev),
        q_depth=torch.as_tensor(q.depth, dtype=torch.float32, device=dev),
        p_m=torch.as_tensor(p_m, dtype=torch.float32, device=dev),
        e_ring_x=torch.as_tensor(ring_centres(bins.ruler)[ring], device=dev),
        dist_km=torch.as_tensor(dist, device=dev),
        cell_bin=torch.as_tensor(cell_bin, dtype=torch.int64, device=dev),
        usual=torch.as_tensor(usual, dtype=torch.float32, device=dev),
        usual_x=torch.as_tensor(
            np.log10(usual / usual.mean()) / 2.0, dtype=torch.float32, device=dev
        ),
        mc=float(ctx[CONTEXT.index("mc")]),
        b=b,
        branching=float(ctx[CONTEXT.index("branching")]),
        thickness=thickness,
        n_cells=box.n_cells,
    )


# ---------------------------------------------------------------- the network


class BiasedLayer(nn.Module):
    """A pre-norm encoder layer whose attention takes an additive bias per head."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        n_d, n_c, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n_d, n_c, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if bias is None else bias.to(q.dtype),
            dropout_p=self.p if self.training else 0.0,
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(n_d, n_c, d)))
        out: torch.Tensor = x + self.drop(self.mlp(self.ln2(x)))
        return out


def _bounded(d_in: int, d_out: int = 1) -> nn.Linear:
    lin = nn.Linear(d_in, d_out)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


#: Features per place besides the tanks: largest (size, age, heard), depth, usual, mc, b, n.
N_EXTRA = 8
#: π, a recent quake's pure numbers: age in c, size − mc, distance in rupture lengths.
N_PI = 3


class SchemeModel(nn.Module):
    """The converged scheme. ``forward`` gives (days, cells) expected quakes at or above mc."""

    def __init__(self, net: SchemeNet) -> None:
        super().__init__()
        self.net = net
        k = net.tanks
        self.cap = math.log(net.max_nudge)
        log_tau = torch.linspace(math.log(net.tau_min_days), math.log(net.tau_max_days), k)
        beta = torch.linspace(*BETA_RANGE, k)
        if net.learn_lifetimes:
            self.log_tau = nn.Parameter(log_tau)
            self.beta = nn.Parameter(beta)
        else:
            # the fallback: a fixed grid, learned mixing lives in the embedding's first layer
            self.register_buffer("log_tau", log_tau)
            self.register_buffer("beta", beta)
        d = net.d_model
        n_in = 2 * k + N_EXTRA
        self.inp = nn.Sequential(nn.LayerNorm(n_in), nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(
            [BiasedLayer(d, net.n_heads, net.dropout) for _ in range(net.n_layers)]
        )
        n_l = max(net.n_layers, 1)
        self.cell_tab = nn.Parameter(torch.zeros(n_l, len(CELL_EDGES), net.n_heads))
        self.rup_tab = nn.Parameter(torch.zeros(n_l, len(RUPTURE_EDGES), net.n_heads))
        self.ln_out = nn.LayerNorm(d)
        self.g = _bounded(d)
        self.b = _bounded(d)
        self.n_pi = nn.Linear(N_PI, net.n_hidden)
        self.n_s = nn.Linear(d, net.n_hidden, bias=False)
        self.n_out = _bounded(net.n_hidden)

    # ------------------------------------------------------------ the places

    def tau(self) -> torch.Tensor:
        lt: torch.Tensor = self.log_tau
        return lt.exp()

    def places(self, sb: SchemeBox, days: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(days, cells, features) what every place says at each midnight, and (days,) the
        rupture length (km) of the box's largest quake of the last ``RULER_DAYS``."""
        n_d, n_c = len(days), sb.n_cells
        net = self.net
        age = days[:, None] - sb.q_t[None, :]
        on = (age > 0) & (age < net.max_age_days)
        d_i, q_i = torch.nonzero(on, as_tuple=True)
        a = age[d_i, q_i].float()
        cell = sb.q_cell.index_select(0, q_i)
        m = sb.q_m.index_select(0, q_i)
        flat = d_i * n_c + cell
        tau = self.tau()
        beta: torch.Tensor = self.beta
        dec = torch.exp(-a[:, None] / tau[None, :])
        amt = torch.exp((beta[None, :] * m[:, None] * math.log(10.0)).clamp(max=30.0))
        count = torch.zeros(n_d * n_c, len(tau), device=a.device).index_add_(0, flat, dec)
        size = torch.zeros_like(count).index_add_(0, flat, dec * amt)
        usual = sb.usual.repeat(n_d)[:, None] * tau[None, :]
        tanks = torch.cat([torch.log1p(count / usual), torch.log1p(size / usual)], -1)
        with torch.no_grad():
            mx = torch.full((n_d * n_c,), -1e9, device=a.device)
            mx = mx.scatter_reduce(0, flat, m, "amax")
            heard = mx > -1e8
            top = m >= mx.index_select(0, flat) - 1e-6
            age_mx = torch.full((n_d * n_c,), 1e9, device=a.device)
            age_mx = age_mx.scatter_reduce(0, flat[top], a[top], "amin")
            n_q = torch.zeros(n_d * n_c, device=a.device).index_add_(0, flat, torch.ones_like(a))
            dep = torch.zeros(n_d * n_c, device=a.device).index_add_(
                0, flat, sb.q_depth.index_select(0, q_i)
            )
            c = float(sb.db.box.c)
            extra = torch.stack(
                [
                    torch.where(heard, mx / 2.0, torch.full_like(mx, -1.0)),
                    torch.where(
                        heard,
                        (torch.log10(age_mx.clamp(min=1e-6) / c) - 3.0) / 3.0,
                        torch.zeros_like(mx),
                    ),
                    heard.float(),
                    torch.where(n_q > 0, dep / n_q.clamp(min=1) / sb.thickness, 0.0),
                    sb.usual_x.repeat(n_d),
                    torch.full_like(mx, sb.mc / 3.0),
                    torch.full_like(mx, sb.b),
                    torch.full_like(mx, sb.branching),
                ],
                -1,
            )
            recent = a < RULER_DAYS
            biggest = torch.full((n_d,), 0.0, device=a.device)
            if bool(recent.any()):
                biggest = biggest.scatter_reduce(0, d_i[recent], m[recent], "amax")
            length = 10.0 ** (-2.44 + 0.59 * (biggest + sb.mc))
        x = torch.cat([tanks, extra], -1).view(n_d, n_c, -1)
        return x, length

    def bias(self, sb: SchemeBox, length: torch.Tensor, layer: int) -> torch.Tensor | None:
        if not self.net.distance_bias:
            return None
        n_c = sb.n_cells
        cell_tab: torch.Tensor = self.cell_tab
        rup_tab: torch.Tensor = self.rup_tab
        cb = cell_tab[layer].index_select(0, sb.cell_bin.reshape(-1)).view(n_c, n_c, -1)
        edges = torch.as_tensor(RUPTURE_EDGES, device=length.device)
        rb = torch.bucketize(sb.dist_km[None] / length[:, None, None], edges, right=True) - 1
        rr = rup_tab[layer].index_select(0, rb.reshape(-1)).view(len(length), n_c, n_c, -1)
        out: torch.Tensor = (cb[None] + rr).permute(0, 3, 1, 2)
        return out

    def state(self, sb: SchemeBox, days: torch.Tensor) -> torch.Tensor:
        """(days, cells, d_model) every place's numbers after attention."""
        x, length = self.places(sb, days)
        dev = x.device
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            h = self.inp(x)
            for i, layer in enumerate(self.layers):
                h = layer(h, self.bias(sb, length, i))
            s: torch.Tensor = self.ln_out(h)
        return s.float()

    # ------------------------------------------------------------ the equation layer

    def forward(self, sb: SchemeBox, days: torch.Tensor) -> torch.Tensor:
        db = sb.db
        box = db.box
        n_d, n_c = len(days), sb.n_cells
        cap = self.cap
        s = self.state(sb, days)
        g = cap * torch.tanh(self.g(s).squeeze(-1) / cap) if self.net.gain else None
        bb = cap * torch.tanh(self.b(s).squeeze(-1) / cap) if self.net.background else None

        days64 = days.double()
        age = days64[:, None] - db.t[None, :]  # (D, P)
        seen = age > 0

        def omori(x: torch.Tensor) -> torch.Tensor:
            return 1.0 - (box.c / (x.clamp(min=0.0) + box.c)) ** (box.p - 1)

        w = torch.where(seen, db.prod[None, :] * (omori(age + box.span) - omori(age)), 0.0)
        w = w.float()
        piece = w.index_select(1, db.e_parent) * db.e_share[None, :]
        trig = torch.zeros(n_d, n_c, device=w.device).index_add_(1, db.e_cell, piece)
        if self.net.recent:
            recent = seen & (age < self.net.recent_days)
            r_pieces = recent.index_select(1, db.e_parent)
            d_i, e_i = torch.nonzero(r_pieces, as_tuple=True)
            if len(d_i):
                par = db.e_parent.index_select(0, e_i)
                cell = db.e_cell.index_select(0, e_i)
                a = age[d_i, par].float()
                pi = torch.stack(
                    [
                        (torch.log10(a.clamp(min=1e-6) / box.c) - 3.0) / 3.0,
                        sb.p_m.index_select(0, par) / 2.0,
                        sb.e_ring_x.index_select(0, e_i),
                    ],
                    -1,
                )
                s_proj = self.n_s(s).reshape(n_d * n_c, -1)
                hid = F.gelu(self.n_pi(pi) + s_proj.index_select(0, d_i * n_c + cell))
                n = cap * torch.tanh(self.n_out(hid).squeeze(-1) / cap)
                extra = w[d_i, par] * db.e_share.index_select(0, e_i) * torch.expm1(n)
                trig = trig + torch.zeros(n_d * n_c, device=w.device).index_add_(
                    0, d_i * n_c + cell, extra
                ).view(n_d, n_c)
        if g is not None:
            trig = trig * g.exp()
        bg = (db.background * box.span)[None, :].expand(n_d, n_c)
        if bb is not None:
            bg = bg * bb.exp()
        return box.cascade * (bg + trig)


def n_parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ---------------------------------------------------------------- training


@dataclass
class SchemeTile:
    """One pool tile for the scheme: its box on the device and its days."""

    key: str
    region: str
    sb: SchemeBox
    days: np.ndarray
    counts: np.ndarray  # (D, cells) at or above mc
    weights: np.ndarray
    train: np.ndarray
    val: np.ndarray


def _nll(
    model: SchemeModel, t: SchemeTile, sel: np.ndarray, dev: torch.device
) -> tuple[torch.Tensor, float]:
    d = torch.as_tensor(t.days[sel], dtype=torch.float64, device=dev)
    lam = model(t.sb, d).clamp(min=1e-30)
    y = torch.as_tensor(t.counts[sel], device=dev)
    wt = torch.as_tensor(t.weights[sel], device=dev)[:, None]
    return (wt * (lam - y * lam.log())).sum(), float((wt * y).sum())


def train_scheme(
    tiles: list[SchemeTile],
    cfg: Field2Config,
    member_seed: int,
    log: Log = _quiet,
    max_minutes: float | None = None,
    max_steps: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One scheme trained like the old world Field: a tile per step, weighted Poisson loss,
    early stopping on every tile's held-back days. ``max_minutes`` / ``max_steps`` cut the
    run short (the timing check)."""
    assert cfg.scheme is not None
    dev = tiles[0].sb.db.t.device
    tc = cfg.train
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    model = SchemeModel(cfg.scheme).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    share = np.array([len(t.train) for t in tiles], dtype=np.float64)
    share /= share.sum()

    def validate() -> float:
        model.eval()
        tot = n = 0.0
        with torch.no_grad():
            for t in tiles:
                for i in range(0, len(t.val), INFER_CHUNK):
                    loss, count = _nll(model, t, np.sort(t.val[i : i + INFER_CHUNK]), dev)
                    tot, n = tot + float(loss), n + count
        model.train()
        return tot / max(n, 1.0)

    t0 = time.perf_counter()
    best = validate()
    start = best
    val_minutes = (time.perf_counter() - t0) / 60
    log(f"  scheme {member_seed}: ETAS's validation loss {start:.5f} ({val_minutes:.1f} min)")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = epochs = steps = 0
    peak = 0
    stopped = ""
    for epoch in range(tc.max_epochs):
        epochs = epoch + 1
        for _ in range(tc.steps_per_epoch):
            k = int(rng.choice(len(tiles), p=share))
            t = tiles[k]
            sel = np.sort(rng.choice(t.train, min(tc.days_per_step, len(t.train)), False))
            total = max(float((t.weights[sel][:, None] * t.counts[sel]).sum()), 1.0)
            opt.zero_grad(set_to_none=True)
            for i in range(0, len(sel), TRAIN_CHUNK):
                loss, _ = _nll(model, t, sel[i : i + TRAIN_CHUNK], dev)
                (loss / total).backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps += 1
            if dev.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(dev))
            if max_steps is not None and steps >= max_steps:
                stopped = "max_steps"
                break
            if max_minutes is not None and (time.perf_counter() - t0) / 60 > max_minutes:
                stopped = "max_minutes"
                break
        if stopped:
            break
        v = validate()
        if v < best - 1e-5:
            best, bad = v, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        log(
            f"  scheme {member_seed} epoch {epochs} validation {v:.5f} (ETAS {start:.5f}, best "
            f"{best:.5f}) {(time.perf_counter() - t0) / 60:.1f} min"
        )
        if bad >= tc.patience:
            break
    return {k: v.cpu() for k, v in state.items()}, {
        "epochs": float(epochs),
        "steps": float(steps),
        "validation_etas": start,
        "validation_best": best,
        "validation_minutes": val_minutes,
        "minutes": (time.perf_counter() - t0) / 60,
        "peak_gb": peak / 1e9,
        "stopped": stopped,
        "parameters": float(n_parameters(model)),
        "tau_days": [float(x) for x in model.tau().detach().cpu()],
    }


def predict_scheme(
    states: list[dict[str, torch.Tensor]], cfg: Field2Config, sb: SchemeBox, days: np.ndarray
) -> np.ndarray:
    """(days, cells) the members' mean rate at or above mc."""
    assert cfg.scheme is not None
    dev = sb.db.t.device
    out = np.zeros((len(days), sb.n_cells))
    d = torch.as_tensor(days, dtype=torch.float64, device=dev)
    for st in states:
        model = SchemeModel(cfg.scheme).to(dev)
        model.load_state_dict(st)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(days), INFER_CHUNK):
                lam = model(sb, d[i : i + INFER_CHUNK])
                out[i : i + INFER_CHUNK] += lam.double().cpu().numpy() / len(states)
    return out


def describe(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2)
