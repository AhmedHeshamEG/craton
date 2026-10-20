"""Craton-mini as a CratonBench method: ``craton bench run craton-mini``.

One model per (split, seed), trained on first use and cached as a checkpoint:

    CB-01   the model trained on ``pretrain`` (masked objective; the split has holes where the
            eras were, so midnights there would ask about days that aren't in it)
    CB-02   the model trained on ``cb02_fit`` (everything before 24 August 2016), both objectives
    CB-03   the model trained on ``cb03_fit`` (everything before 2009), both objectives
    CB-15   the measurements of the CB-01 model's training run

How each task is read:

    CB-01   every quake in the snapshot except the hidden ones is visible; the masked quakes go
            back in at their times, place and size hidden, and the model reads windows of
            1,024 around them. GNSS queries get the climatology answer: v0 reads quakes only,
            and says so in the run's extras.
    CB-02/3 at each midnight, the 1,023 quakes before it and a query token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from craton.bench.baselines import Climatology, register_method
from craton.bench.data import Events
from craton.bench.dists import ChannelDist, Histogram
from craton.bench.splits import SPLITS
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    ForecastSpec,
    GeoPerplexity,
    GriddedForecast,
    MaskedQueries,
)
from craton.config import CratonConfig, config_hash, load_config
from craton.model.craton import CratonModel
from craton.model.encoders import EVENT_CHANNELS, magnitude_type_id
from craton.model.heads import EVENT_BINS
from craton.spec.timeaxis import to_seconds
from craton.train import checkpoint
from craton.train.data import Catalog, query_tokens, recent_rates, window_tokens
from craton.train.loop import Log, train

#: Next-day training starts here: one year in, so every midnight has a full context.
NEXT_DAY_FROM = "1986-01-01T00:00:00Z"
INFER_BATCH = 16


def _quiet(_: str) -> None:
    return None


@dataclass
class CratonMini:
    name: str = "craton-mini"
    kind: str = "model"
    config_name: str = "mini"
    log: Log = _quiet
    _models: dict[tuple[str, int], CratonModel] = field(default_factory=dict)
    _meta: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)

    @property
    def config(self) -> CratonConfig:
        return load_config(self.config_name)

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------ models

    def key(self, ctx: BenchContext, split: str, seed: int) -> checkpoint.CheckpointKey:
        import json

        manifest = json.loads((ctx.split_dir(split) / "manifest.json").read_text("utf-8"))
        return checkpoint.CheckpointKey(split, manifest["id"], self.config, seed)

    def model(self, ctx: BenchContext, split: str, seed: int) -> CratonModel:
        """The trained model for a split and seed: from its checkpoint, or trained now."""
        if (split, seed) in self._models:
            return self._models[(split, seed)]
        key = self.key(ctx, split, seed)
        found = checkpoint.load(ctx.root, key)
        if found is None:
            self.log(f"training {key.name}")
            state, meta = self.train(ctx, split, seed)
        else:
            state, meta = found
        model = CratonModel(self.config)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self._models[(split, seed)] = model
        self._meta[(split, seed)] = meta
        return model

    def train(
        self, ctx: BenchContext, split: str, seed: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        ev = ctx.train_events(split)
        cat = Catalog.of(ev)
        before = SPLITS[split].before
        window = (to_seconds(NEXT_DAY_FROM), to_seconds(before)) if before else None
        result = train(self.config, cat, seed, window, log=self.log)
        key = self.key(ctx, split, seed)
        data = {
            "snapshot": ctx.snapshot_id,
            "quakes": len(ev),
            "first": _day(ev.time[0]),
            "last": _day(ev.time[-1]),
            "next_day_window": [NEXT_DAY_FROM, before] if before else None,
        }
        checkpoint.save(ctx.root, key, result.state, result.history, result.stats, data)
        found = checkpoint.load(ctx.root, key)
        assert found is not None
        return found

    # ------------------------------------------------------------ CB-01

    def predict_masked(
        self, queries: MaskedQueries, ctx: BenchContext, seed: int
    ) -> dict[str, dict[str, ChannelDist]]:
        task = TASKS["CB-01"]
        assert isinstance(task, GeoPerplexity)
        out = Climatology().predict_masked(queries, ctx, seed)  # GNSS: v0 reads quakes only
        idx = queries.where("seismic_catalog")
        if not len(idx):
            return out
        model = self.model(ctx, "pretrain", seed)
        hidden = task.hidden(ctx)
        ev = ctx.events
        visible = ev.select(~np.isin(ev.token_hash, sorted(hidden)))
        logp = masked_logprobs(model, visible, queries, idx, self.device)
        out["seismic_catalog"] = {
            ch: Histogram(b.edges, np.exp(lp))
            for ch, b, lp in zip(EVENT_CHANNELS, EVENT_BINS, logp, strict=True)
        }
        return out

    # ------------------------------------------------------------ CB-02, CB-03

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        model = self.model(ctx, spec.fit_split, seed)
        cat = Catalog.of(ctx.events)  # each midnight reads only the quakes before it
        length = self.config.model.context_tokens
        days = spec.day_starts
        rates, shares = [], []
        with torch.no_grad():
            for i in range(0, len(days), INFER_BATCH):
                tok = query_tokens(cat, days[i : i + INFER_BATCH], length).to(self.device)
                with torch.autocast(self.device.type, torch.float16, self.device.type == "cuda"):
                    h = model(tok)[:, -1]
                recent = recent_rates(cat, days[i : i + INFER_BATCH]).to(self.device)
                log_rate, logp_cell, logp_mag = model.next_day(h, recent)
                rates.append((log_rate[:, None] + logp_cell).exp().double().cpu().numpy())
                shares.append(logp_mag.exp().double().cpu().numpy())
        mag = np.concatenate(shares)
        mag = mag / mag.sum(axis=1, keepdims=True)
        return GriddedForecast(np.concatenate(rates), mag[:, None, :])

    def write_cards(self, root: Any) -> Any:
        """docs/models/<config>.md: the cards of every checkpoint this method has loaded."""
        folders = [self.key_from_meta(meta).folder(root) for meta in self._meta.values()]
        return checkpoint.write_index(
            root, folders, root / "docs" / "models" / f"craton-{self.config_name}.md"
        )

    def key_from_meta(self, meta: dict[str, Any]) -> checkpoint.CheckpointKey:
        return checkpoint.CheckpointKey(meta["split"], meta["split_id"], self.config, meta["seed"])

    # ------------------------------------------------------------ CB-15

    def efficiency(self, ctx: BenchContext, seed: int) -> dict[str, Any]:
        self.model(ctx, "pretrain", seed)
        stats: dict[str, Any] = self._meta[("pretrain", seed)]["stats"]
        return stats


def _day(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(float(t), dt.UTC).strftime("%Y-%m-%d")


def masked_logprobs(
    model: CratonModel,
    visible: Events,
    queries: MaskedQueries,
    idx: np.ndarray[Any, np.dtype[np.int64]],
    device: torch.device,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    """Log-probabilities per channel for the masked quakes ``idx`` of the queries.

    The hidden quakes are rebuilt from what the query reveals (time, magnitude type) and put
    back among the visible ones; their place and size are never read from anywhere.
    """
    n_vis, n_q = len(visible), len(idx)
    time = np.concatenate([visible.time, queries.time[idx]])
    values = np.zeros((n_vis + n_q, 4))
    values[:n_vis] = np.stack([visible.lat, visible.lon, visible.depth_km, visible.mag], axis=1)
    types = [magnitude_type_id(str(t)) for t in visible.mag_type]
    types += [magnitude_type_id(queries.qualifiers[i]["magnitude_type"]) for i in idx]
    is_q = np.concatenate([np.zeros(n_vis, dtype=bool), np.ones(n_q, dtype=bool)])
    q_id = np.concatenate([np.full(n_vis, -1), np.arange(n_q)])
    order = np.argsort(time, kind="stable")
    cat = Catalog(time[order], values[order], np.asarray(types, dtype=np.int64)[order])
    is_q, q_id = is_q[order], q_id[order]

    length = model.config.model.context_tokens
    if len(cat) < length:
        raise ValueError("fewer quakes than one context window")
    stride = length // 2
    starts = np.arange(0, len(cat) - length + 1, stride)
    if starts[-1] != len(cat) - length:
        starts = np.append(starts, len(cat) - length)
    pos = np.nonzero(is_q)[0]
    centres = starts + length / 2
    best = np.abs(pos[:, None] - centres[None, :]).argmin(axis=1)
    out = [np.zeros((n_q, b.n)) for b in EVENT_BINS]
    used = np.unique(best)
    filled = 0
    with torch.no_grad():
        for i in range(0, len(used), INFER_BATCH):
            chunk = used[i : i + INFER_BATCH]
            rows = starts[chunk][:, None] + np.arange(length)[None, :]
            hid = is_q[rows]
            tok = window_tokens(cat, rows, hid, cat.time[rows[:, 0]]).to(device)
            with torch.autocast(device.type, torch.float16, device.type == "cuda"):
                h = model(tok)
            for j, w in enumerate(chunk):
                mine = (best == w) & (pos >= starts[w]) & (pos < starts[w] + length)
                local = pos[mine] - starts[w]
                logp = model.masked(h[j, torch.as_tensor(local, device=device)])
                for c, lp in enumerate(logp):
                    out[c][q_id[pos[mine]]] = lp.double().cpu().numpy()
                filled += int(mine.sum())
    if filled != n_q:
        raise RuntimeError(f"{n_q - filled} masked quakes got no window")
    return out


register_method(CratonMini())
