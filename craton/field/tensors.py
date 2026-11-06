"""The tensor field: masked modelling of how quakes slip, and the squeeze it implies (CB-34).

The trunk learns quakes by hiding one and predicting it from the rest. The tensor field does
the same with moment tensors: hide one Global CMT mechanism and predict its shape (its unit,
trace-free tensor) from the mechanisms around it: where they are relative to it, how deep, how
big, how long before or after, and how they slipped. No stress measurement is ever read.

What it learns is a field: ask it at any point and it answers the mechanism a quake there would
most likely have. Asked at the centre of a World Stress Map cell, its answer's squeeze axis is
the cell's SHmax (Lund & Townend 2007 on minus the tensor, ``craton.bench.mechanisms``). That
is how Craton answers CB-34, graded only on stress records not read from quakes.

Three choices keep the test honest (D-133):

- **No copying.** A mechanism's own sequence (within 20 km and 30 days) is hidden with it: a
  borehole has no aftershock twin to copy, so training must not reward copying one.
- **Geometry, not memory.** Every training example is rotated about the vertical by a random
  angle (positions and tensors together), so the field can't learn "Japan squeezes east-west";
  it has to read the neighbours.
- **Never the answers, never the zero-shot regions.** Training reads mechanisms before 2007
  outside Turkey, Chile and Alaska (CB-22's never-seen regions). The epochs are chosen on
  mechanisms of 2003-2006, never on stress records.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from craton.bench.mechanisms import MechanismQuery, Mechanisms, load_mechanisms, shmax_of
from craton.paths import configs_dir, find_root, results_dir
from craton.seed import seed_everything
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
KM_PER_DEG = 111.195
YEAR_S = 365.25 * 86_400.0
VERSION = "tensors-v1"


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class TensorConfig:
    name: str = "mini"
    description: str = ""
    # data
    max_depth_km: float = 40.0
    train_end: str = "2007-01-01T00:00:00Z"
    validation_start: str = "2003-01-01T00:00:00Z"
    exclude_regions: tuple[str, ...] = ("turkey", "chile", "alaska")
    radius_km: float = 300.0
    neighbours: int = 48
    twin_km: float = 20.0
    twin_days: float = 30.0
    # model
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    # training
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch: int = 256
    max_epochs: int = 60
    patience: int = 6
    seeds: tuple[int, ...] = (0, 1, 2)
    # answering
    query_depths_km: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0)

    @property
    def hash(self) -> str:
        body = asdict(self)
        body.pop("description")
        body.pop("seeds")
        text = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:8]


def load_tensor_config(name: str, root: Path | None = None) -> TensorConfig:
    path = configs_dir(root or find_root()) / "tensors" / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat: dict[str, Any] = {
        "name": raw.get("name", name),
        "description": raw.get("description", ""),
    }
    for section in ("data", "model", "train", "answer"):
        flat.update(raw.get(section, {}) or {})
    for key in ("exclude_regions", "seeds", "query_depths_km"):
        if key in flat:
            flat[key] = tuple(flat[key])
    return TensorConfig(**flat)


# ---------------------------------------------------------------- tensors in (up, north, east)


def une_of(m_rtp: FloatArray) -> FloatArray:
    """(..., 6) rr, tt, pp, rt, rp, tp → (..., 6) uu, nn, ee, un, ue, ne (t points south)."""
    m = np.asarray(m_rtp, dtype=np.float64)
    return np.stack([m[..., 0], m[..., 1], m[..., 2], -m[..., 3], m[..., 4], -m[..., 5]], -1)


def rtp_of(m_une: FloatArray) -> FloatArray:
    """The inverse of :func:`une_of` (the same sign flips)."""
    return une_of(m_une)


def _sym(v: torch.Tensor) -> torch.Tensor:
    """(..., 6) uu, nn, ee, un, ue, ne → (..., 3, 3)."""
    uu, nn_, ee, un, ue, ne = v.unbind(-1)
    rows = [
        torch.stack([uu, un, ue], -1),
        torch.stack([un, nn_, ne], -1),
        torch.stack([ue, ne, ee], -1),
    ]
    return torch.stack(rows, -2)


def _vec(m: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], m[..., 0, 1], m[..., 0, 2], m[..., 1, 2]], -1
    )


def unit_deviatoric(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → trace removed, Frobenius norm 1 (the shape of the slip)."""
    eye = torch.eye(3, device=m.device, dtype=m.dtype)
    dev = m - eye * (m.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0)[..., None, None]
    out: torch.Tensor = dev / dev.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
    return out


def rotate(
    m: torch.Tensor, xy: torch.Tensor, theta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate tensors (B, K, 3, 3) and offsets (B, K, 2: east, north) about the vertical by
    ``theta`` (B,), azimuth increasing from north to east. Same rotation for both."""
    c, s = torch.cos(theta), torch.sin(theta)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    # In (u, n, e): n' = n c − e s, e' = n s + e c.
    r = torch.stack(
        [
            torch.stack([one, zero, zero], -1),
            torch.stack([zero, c, -s], -1),
            torch.stack([zero, s, c], -1),
        ],
        -2,
    )  # (B, 3, 3)
    rk = r[:, None]
    m2 = rk @ m @ rk.transpose(-1, -2)
    e, n = xy[..., 0], xy[..., 1]
    xy2 = torch.stack([n * s[:, None] + e * c[:, None], n * c[:, None] - e * s[:, None]], -1)
    return m2, xy2


# ---------------------------------------------------------------- neighbours, on the GPU


@dataclass
class Pool:
    """Mechanisms as tensors on one device, with each one's unit deviatoric shape."""

    lat: torch.Tensor
    lon: torch.Tensor
    depth: torch.Tensor
    time: torch.Tensor
    mw: torch.Tensor
    shape: torch.Tensor  # (n, 3, 3) in (u, n, e)

    @staticmethod
    def of(mech: Mechanisms, device: torch.device) -> Pool:
        m = torch.as_tensor(une_of(mech.m), dtype=torch.float64, device=device)
        mw = (2.0 / 3.0) * (np.log10(np.maximum(mech.m0, 1.0)) - 9.1)

        def t(x: Any) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x, dtype=np.float64), device=device)

        return Pool(
            t(mech.lat),
            t(mech.lon),
            t(mech.depth_km),
            t(mech.time),
            t(mw),
            unit_deviatoric(_sym(m)).float(),
        )

    def __len__(self) -> int:
        return int(self.lat.shape[0])


def offsets_km(
    lat0: torch.Tensor, lon0: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor
) -> torch.Tensor:
    """(Q, N, 2) east, north km from each query to each point (tangent plane, dateline-safe)."""
    dlon = (lon[None, :] - lon0[:, None] + 180.0) % 360.0 - 180.0
    east = dlon * torch.cos(torch.deg2rad(lat0))[:, None] * KM_PER_DEG
    north = (lat[None, :] - lat0[:, None]) * KM_PER_DEG
    return torch.stack([east, north], -1)


@dataclass
class Batch:
    feats: torch.Tensor  # (B, K, F) without the tensor
    shapes: torch.Tensor  # (B, K, 3, 3)
    xy: torch.Tensor  # (B, K, 2) km
    mask: torch.Tensor  # (B, K) True = padding
    query_depth: torch.Tensor  # (B,)


def gather(
    cfg: TensorConfig,
    pool: Pool,
    lat: torch.Tensor,
    lon: torch.Tensor,
    depth: torch.Tensor,
    time: torch.Tensor | None,
    hide: torch.Tensor | None = None,
    chunk: int = 512,
) -> Batch:
    """The ``neighbours`` nearest pool mechanisms within ``radius_km`` of each query point.

    ``time`` set (a training target): its twins (within twin_km and twin_days) and ``hide``
    (the target's own index) are left out. ``time`` None (a stress cell): everything counts.
    """
    k = cfg.neighbours
    parts: list[Batch] = []
    for a in range(0, len(lat), chunk):
        sl = slice(a, a + chunk)
        xy = offsets_km(lat[sl], lon[sl], pool.lat, pool.lon)  # (Q, N, 2)
        d = xy.norm(dim=-1)
        bad = d > cfg.radius_km
        if time is not None:
            dt = (pool.time[None, :] - time[sl, None]).abs()
            bad |= (d < cfg.twin_km) & (dt < cfg.twin_days * 86_400.0)
        if hide is not None:
            bad[torch.arange(d.shape[0], device=d.device), hide[sl]] = True
        d = d.masked_fill(bad, math.inf)
        if d.shape[1] < k:  # fewer mechanisms than slots: the rest are padding
            fill = torch.full((d.shape[0], k - d.shape[1]), math.inf, device=d.device)
            d = torch.cat([d, fill.to(d.dtype)], dim=1)
            xy = torch.cat(
                [xy, torch.zeros(xy.shape[0], fill.shape[1], 2, device=d.device, dtype=xy.dtype)], 1
            )
        dist, idx = d.topk(k, dim=1, largest=False)
        idx_pool = idx.clamp_max(len(pool) - 1)  # padding slots point anywhere; masked below
        pad = ~torch.isfinite(dist)
        sel_xy = torch.gather(xy, 1, idx[..., None].expand(-1, -1, 2))
        dz = (pool.depth[idx_pool] - depth[sl, None]) / 20.0
        if time is not None:
            dt_y = (pool.time[idx_pool] - time[sl, None]) / YEAR_S
        else:
            dt_y = torch.zeros_like(dz)
        feats = torch.stack(
            [
                dist.masked_fill(pad, 0.0) / 100.0,
                dz,
                (pool.mw[idx_pool] - 6.0),
                torch.sign(dt_y) * torch.log1p(dt_y.abs()),
            ],
            -1,
        )
        parts.append(
            Batch(
                feats.float().masked_fill(pad[..., None], 0.0),
                pool.shape[idx_pool].masked_fill(pad[..., None, None], 0.0),
                (sel_xy / 100.0).float().masked_fill(pad[..., None], 0.0),
                pad,
                (depth[sl] / 20.0).float(),
            )
        )
    return Batch(*(torch.cat([getattr(p, f) for p in parts]) for f in Batch.__dataclass_fields__))


# ---------------------------------------------------------------- the network


class TensorFieldNet(nn.Module):
    """Neighbours → one query token by cross-attention → a unit trace-free tensor."""

    def __init__(self, cfg: TensorConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Sequential(nn.Linear(4 + 2 + 6, d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            for _ in range(cfg.n_layers)
        )
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(2 * cfg.n_layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            for _ in range(cfg.n_layers)
        )
        self.out = nn.Linear(d, 6)

    def forward(self, b: Batch, shapes: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(B, 3, 3) unit deviatoric tensors in (u, n, e)."""
        x = self.embed(torch.cat([b.feats, xy, _vec(shapes)], -1))
        q = self.query(b.query_depth[:, None])[:, None, :]
        mask = b.mask.clone()
        empty = mask.all(dim=1)
        mask[empty, 0] = False  # attention needs one key; such rows are answered NaN outside
        for i, att in enumerate(self.attn):
            h, _ = att(self.norm[2 * i](q), x, x, key_padding_mask=mask, need_weights=False)
            q = q + h
            q = q + self.mlp[i](self.norm[2 * i + 1](q))
        return unit_deviatoric(_sym(self.out(q[:, 0]).float()))


def alignment(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product of unit tensors: 1 = the same slip, −1 = the opposite."""
    return (pred * target).flatten(-2).sum(-1)


# ---------------------------------------------------------------- training


def training_pool(cfg: TensorConfig, root: Path) -> tuple[Mechanisms, str]:
    """Crustal mechanisms before ``train_end``, outside the excluded regions."""
    mech, snap = load_mechanisms(root)
    keep = ~(mech.depth_km > cfg.max_depth_km) & (mech.time < to_seconds(cfg.train_end))
    keep &= ~excluded_mask(mech, cfg.exclude_regions, root)
    return mech.select(keep), snap


def excluded_mask(mech: Mechanisms, regions: tuple[str, ...], root: Path) -> np.ndarray[Any, Any]:
    """True for every mechanism inside any of the regions' boxes."""
    from craton.adapters.region import load_region

    out = np.zeros(len(mech), dtype=bool)
    for name in regions:
        b = load_region(name, root).box
        out |= (
            (mech.lat >= b.lat_min)
            & (mech.lat < b.lat_max)
            & (mech.lon >= b.lon_min)
            & (mech.lon < b.lon_max)
        )
    return out


@dataclass
class Trained:
    net: TensorFieldNet
    meta: dict[str, Any] = field(default_factory=dict)


def checkpoint_path(cfg: TensorConfig, seed: int, root: Path) -> Path:
    return results_dir(root) / "checkpoints" / f"tensors-{cfg.name}-{cfg.hash}-s{seed}" / "model.pt"


def card(meta: dict[str, Any], n_params: int) -> str:
    """The model card of one tensor field checkpoint."""
    c = meta["config"]
    h = meta["history"]
    best = min(h, key=lambda r: r["val"])
    return "\n".join(
        [
            f"## tensors-{c['name']}-{meta['config_hash']}-s{meta['seed']}",
            "",
            "The tensor field: hide a Global CMT moment tensor, predict its shape from the "
            "mechanisms around it. It answers CB-34 (the squeeze at a stress cell, from the "
            "field's tensor there). It never reads a stress record.",
            "",
            "| | |",
            "|---|---|",
            f"| Trained on | `moment_tensor`, gcmt_world snapshot `{meta['mechanism_snapshot']}`, "
            f"crust above {c['max_depth_km']:g} km, before {c['train_end'][:10]}, outside "
            f"{', '.join(c['exclude_regions'])} |",
            f"| Examples | {meta['pool']:,} mechanisms; {meta['train_targets']:,} training "
            f"targets, {meta['validation_targets']:,} validation targets (from "
            f"{c['validation_start'][:10]}) |",
            f"| Shape | d = {c['d_model']}, {c['n_layers']} cross-attention layers, {c['n_heads']} "
            f"heads, {c['neighbours']} neighbours within {c['radius_km']:g} km |",
            f"| Parameters | {n_params:,} |",
            f"| Seed | {meta['seed']} |",
            f"| Training | best at epoch {best['epoch']} of {len(h)}, misfit "
            f"{meta['best_validation_misfit']:.4f} (1 − alignment of unit tensors) |",
            f"| Code | {meta['version']} |",
            "",
            "**Not for.** Any decision about safety: a statistical smoother of mechanisms, "
            "graded against a few hundred stress records.",
            "",
        ]
    )


def train_field(cfg: TensorConfig, seed: int, root: Path, log: Any = print) -> Trained:
    """Train one tensor field (or load it from its checkpoint)."""
    path = checkpoint_path(cfg, seed, root)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path.is_file():
        blob = torch.load(path, map_location=dev, weights_only=False)
        net = TensorFieldNet(cfg).to(dev)
        net.load_state_dict(blob["state"])
        net.eval()
        if not (path.parent / "card.md").is_file():
            n = sum(p.numel() for p in net.parameters())
            (path.parent / "card.md").write_text(card(blob["meta"], n), encoding="utf-8")
        return Trained(net, blob["meta"])
    seed_everything(seed, deterministic=False)
    mech, snap = training_pool(cfg, root)
    pool = Pool.of(mech, dev)
    t = pool.time
    is_val = t >= to_seconds(cfg.validation_start)
    idx_all = torch.arange(len(pool), device=dev)
    data = gather(cfg, pool, pool.lat, pool.lon, pool.depth, pool.time, hide=idx_all)
    has = ~data.mask.all(dim=1)
    tr = idx_all[~is_val & has]
    va = idx_all[is_val & has]
    log(
        f"tensor field {cfg.name} seed {seed}: pool {len(pool):,}, "
        f"train {len(tr):,}, val {len(va):,}"
    )
    net = TensorFieldNet(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def pick(rows: torch.Tensor) -> Batch:
        return Batch(*(getattr(data, f)[rows] for f in Batch.__dataclass_fields__))

    def run(rows: torch.Tensor, train: bool) -> torch.Tensor:
        b = pick(rows)
        theta = (
            torch.rand(len(rows), device=dev) * 2 * math.pi
            if train
            else torch.zeros(len(rows), device=dev)
        )
        shapes, xy = rotate(b.shapes, b.xy, theta)
        target, _ = rotate(
            pool.shape[rows][:, None], torch.zeros(len(rows), 1, 2, device=dev), theta
        )
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pred = net(b, shapes, xy)
        return (1.0 - alignment(pred.float(), target[:, 0])).mean()

    best, best_state, wait, history = math.inf, None, 0, []
    for epoch in range(cfg.max_epochs):
        net.train()
        order = tr[torch.randperm(len(tr), generator=gen).to(dev)]
        losses = []
        for a in range(0, len(order), cfg.batch):
            loss = run(order[a : a + cfg.batch], True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            val = float(
                np.mean([float(run(va[a : a + 1024], False)) for a in range(0, len(va), 1024)])
            )
        history.append({"epoch": epoch, "train": float(np.mean(losses)), "val": val})
        log(f"  epoch {epoch:2d}  train {np.mean(losses):.4f}  val {val:.4f}")
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    meta = {
        "version": VERSION,
        "config": asdict(cfg),
        "config_hash": cfg.hash,
        "seed": seed,
        "mechanism_snapshot": snap,
        "pool": len(pool),
        "train_targets": len(tr),
        "validation_targets": len(va),
        "best_validation_misfit": best,
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "meta": meta}, path)
    (path.parent / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    n = sum(p.numel() for p in net.parameters())
    (path.parent / "card.md").write_text(card(meta, n), encoding="utf-8")
    return Trained(net, meta)


# ---------------------------------------------------------------- answering CB-34


def predict_shapes(
    trained: Trained, cfg: TensorConfig, mech: Mechanisms, lat: FloatArray, lon: FloatArray
) -> tuple[FloatArray, np.ndarray[Any, Any]]:
    """The field's mean unit tensor (rr, tt, pp, rt, rp, tp) at each point, averaged over the
    answer depths, and whether any mechanism was in reach."""
    dev = next(trained.net.parameters()).device
    crust = mech.select(~(mech.depth_km > cfg.max_depth_km))
    if len(crust) == 0:
        return np.full((len(lat), 6), math.nan), np.zeros(len(lat), dtype=bool)
    pool = Pool.of(crust, dev)
    la = torch.as_tensor(lat, dtype=torch.float64, device=dev)
    lo = torch.as_tensor(lon, dtype=torch.float64, device=dev)
    acc = torch.zeros(len(lat), 3, 3, device=dev)
    reach = torch.zeros(len(lat), dtype=torch.bool, device=dev)
    with torch.no_grad():
        for z in cfg.query_depths_km:
            b = gather(cfg, pool, la, lo, torch.full_like(la, z), None)
            reach |= ~b.mask.all(dim=1)
            acc += trained.net(b, b.shapes, b.xy)
    out = rtp_of(_vec(acc / len(cfg.query_depths_km)).double().cpu().numpy())
    ok = reach.cpu().numpy()
    out[~ok] = math.nan
    return out, ok


@dataclass
class TensorFieldMapper:
    """CB-34's answer from the tensor field: one field per seed, trained on first use."""

    config_name: str = "mini"
    root: Path | None = None
    log: Any = print
    _fields: dict[int, Trained] = field(default_factory=dict)

    @property
    def config(self) -> TensorConfig:
        return load_tensor_config(self.config_name, self.root)

    def field_for(self, seed: int) -> Trained:
        if seed not in self._fields:
            self._fields[seed] = train_field(self.config, seed, self.root or find_root(), self.log)
        return self._fields[seed]

    def mechanism_stress_azimuths(self, query: MechanismQuery, seed: int) -> FloatArray:
        shapes, _ = predict_shapes(
            self.field_for(seed), self.config, query.mechanisms, query.lat, query.lon
        )
        return shmax_of(shapes)
