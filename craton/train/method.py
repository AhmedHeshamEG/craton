"""Craton-mini as a CratonBench method: ``craton bench run craton-mini``.

One model per (split, seed), trained on first use and cached as a checkpoint:

    CB-01   the model trained on ``pretrain`` (masked objectives; the split has holes where the
            eras were, so midnights there would ask about days that aren't in it)
    CB-02   the model trained on ``cb02_fit`` (everything before 24 August 2016)
    CB-03   the model trained on ``cb03_fit`` (everything before 2009)
    CB-04   the ``pretrain`` model again: a held-out station is a station hidden whole
    CB-15   the measurements of the CB-01 model's training run
    CB-17   the model trained on ``cb17_fit`` (``pretrain`` without the CB-17 region)

How each task is read:

    CB-01   every snapshot token except the hidden ones and the held-out stations is visible.
            The masked quakes go back in at their times, place and size hidden, and the model
            reads windows around them; a masked GNSS day goes back in at its station and day,
            its position hidden, and the model reads the station's other days and its
            neighbours'. An instrument the config doesn't read is answered by climatology.
    CB-02/3 at each midnight, the fused context before it (the policy in
            ``craton.train.context``) and a query token.
    CB-04   windows of a held-out station's days, every one hidden, its line borrowed.
    CB-17   the region's first N days of quakes, then its test year's quakes, hidden.

Every station line comes from the training split the model learned on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from craton.bench.baselines import Climatology, register_method
from craton.bench.data import DAY, Events, Gnss
from craton.bench.dists import ChannelDist, Histogram, LogisticMixture
from craton.bench.splits import SPLITS, station_sites
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    ForecastSpec,
    GapQueries,
    GeoPerplexity,
    GriddedForecast,
    MaskedQueries,
    RegionQueries,
)
from craton.config import CratonConfig, config_hash, load_config
from craton.model.craton import CratonModel
from craton.model.encoders import EVENT_CHANNELS, magnitude_type_id
from craton.model.heads import EVENT_BINS
from craton.spec.timeaxis import to_seconds
from craton.train import checkpoint
from craton.train.context import (
    Batch,
    Catalog,
    GnssDays,
    Lines,
    Sources,
    Window,
    fit_lines,
    gnss_rows,
    quakes_around,
    tide_rows,
    where_now,
    write,
)
from craton.train.data import midnight_window, recent_rates, region_context
from craton.train.loop import Log, train

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]

#: Next-day training starts here: one year in, so every midnight has a full context.
NEXT_DAY_FROM = "1986-01-01T00:00:00Z"
INFER_BATCH = 16
#: CB-01 GNSS queries are answered in groups of this many stations sharing a day.
GNSS_GROUP = 8
#: CB-04 rebuilds a held-out station this many days per window.
GAP_CHUNK_DAYS = 15
GNSS_CHANNELS = ("east", "north", "up")


def _quiet(_: str) -> None:
    return None


def _sites(g: Gnss) -> dict[str, tuple[float, float]]:
    return {c: (lat, lon) for c, (lat, lon, _) in station_sites(g).items()}


@dataclass
class CratonMini:
    name: str = "craton-mini"
    kind: str = "model"
    config_name: str = "mini"
    #: A config to use instead of ``config_name`` (the ablation runner's variants).
    override: CratonConfig | None = None
    log: Log = _quiet
    _models: dict[tuple[str, int], CratonModel] = field(default_factory=dict)
    _meta: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    _lines: dict[str, Lines] = field(default_factory=dict)

    @property
    def config(self) -> CratonConfig:
        return self.override if self.override is not None else load_config(self.config_name)

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.config.model.inputs

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------ models

    def key(self, ctx: BenchContext, split: str, seed: int) -> checkpoint.CheckpointKey:
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

    def lines(self, ctx: BenchContext, split: str) -> Lines:
        """Station lines fitted on a split, for every station it has."""
        if split not in self._lines:
            g = ctx.train_gnss(split)
            self._lines[split] = fit_lines(g, _sites(g))
        return self._lines[split]

    def sources(
        self,
        ctx: BenchContext,
        events: Events,
        gnss: GnssDays | None,
        split_for_tides: str | None = None,
    ) -> Sources:
        cfg = self.config
        tides = ctx.train_tides(split_for_tides) if split_for_tides else ctx.tides
        return Sources.build(
            cfg.model.inputs, cfg.context, cfg.model.context_tokens, events, gnss, tides
        )

    def train(
        self, ctx: BenchContext, split: str, seed: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        ev = ctx.train_events(split)
        gnss = None
        if "gnss_daily" in self.inputs:
            gnss = GnssDays.of(ctx.train_gnss(split), self.lines(ctx, split))
        src = self.sources(ctx, ev, gnss, split)
        before = SPLITS[split].before
        window = (to_seconds(NEXT_DAY_FROM), to_seconds(before)) if before else None
        result = train(self.config, src, seed, window, log=self.log)
        key = self.key(ctx, split, seed)
        data = {
            "snapshot": ctx.snapshot_id,
            "inputs": list(self.inputs),
            "quakes": len(ev),
            "gnss_days": len(gnss) if gnss is not None else 0,
            "tide_hours": len(src.tides) if src.tides is not None else 0,
            "first": _day(ev.time[0]),
            "last": _day(ev.time[-1]),
            "next_day_window": [NEXT_DAY_FROM, before] if before else None,
        }
        checkpoint.save(ctx.root, key, result.state, result.history, result.stats, data)
        found = checkpoint.load(ctx.root, key)
        assert found is not None
        return found

    def _run(self, model: CratonModel, batch: Batch) -> torch.Tensor:
        tok = batch.tokens.to(self.device)
        with (
            torch.no_grad(),
            torch.autocast(self.device.type, torch.float16, self.device.type == "cuda"),
        ):
            out: torch.Tensor = model(tok)
        return out

    def _gnss_answers(
        self, model: CratonModel, batch: Batch, h: torch.Tensor
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """logw, mu, s (Ng, 3, K) and the baseline (Ng, 3) for every hidden GNSS day."""
        tok = batch.tokens.to(self.device)
        sel = tok.hidden[..., 0] & tok.of("gnss_daily")
        with torch.no_grad():
            logw, mu, s = model.gnss(h[sel], batch.gnss_years.to(self.device))
        return (
            logw.double().cpu().numpy(),
            mu.double().cpu().numpy(),
            s.double().cpu().numpy(),
            batch.gnss_base,
        )

    # ------------------------------------------------------------ CB-01

    def predict_masked(
        self, queries: MaskedQueries, ctx: BenchContext, seed: int
    ) -> dict[str, dict[str, ChannelDist]]:
        task = TASKS["CB-01"]
        assert isinstance(task, GeoPerplexity)
        out = Climatology().predict_masked(queries, ctx, seed)  # what the model doesn't read
        model = self.model(ctx, "pretrain", seed)
        hidden = task.hidden(ctx)
        ev = ctx.events
        visible = ev.select(~np.isin(ev.token_hash, sorted(hidden)))
        g = ctx.gnss
        g = g.select(~np.isin(g.token_hash, sorted(hidden)) & ~np.isin(g.station, ctx.heldout))
        lines = self.lines(ctx, "pretrain")
        days = GnssDays.of(g, lines) if "gnss_daily" in self.inputs else None
        src = self.sources(ctx, visible, days)
        idx = queries.where("seismic_catalog")
        if len(idx) and "seismic_catalog" in self.inputs:
            logp = self.masked_logprobs(model, src, visible, queries, idx)
            out["seismic_catalog"] = {
                ch: Histogram(b.edges, np.exp(lp))
                for ch, b, lp in zip(EVENT_CHANNELS, EVENT_BINS, logp, strict=True)
            }
        gi = queries.where("gnss_daily")
        if len(gi) and days is not None:
            out["gnss_daily"] = self.gnss_dists(model, src, days, queries, gi)
        return out

    def masked_logprobs(
        self,
        model: CratonModel,
        src: Sources,
        visible: Events,
        queries: MaskedQueries,
        idx: IntArray,
    ) -> list[FloatArray]:
        """Log-probabilities per channel for the masked quakes ``idx`` of the queries.

        The hidden quakes are rebuilt from what the query reveals (time, magnitude type) and
        put back among the visible ones; their place and size are never read from anywhere.
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

        p, length = src.policy, src.length
        n_quakes = length - (p.tides if src.tides is not None else 0)
        n_quakes -= p.gnss if src.gnss is not None else 0
        if len(cat) < n_quakes:
            raise ValueError("fewer quakes than one context window")
        stride = n_quakes // 2
        starts = np.arange(0, len(cat) - n_quakes + 1, stride)
        if starts[-1] != len(cat) - n_quakes:
            starts = np.append(starts, len(cat) - n_quakes)
        pos = np.nonzero(is_q)[0]
        centres = starts + n_quakes / 2
        best = np.abs(pos[:, None] - centres[None, :]).argmin(axis=1)
        out = [np.zeros((n_q, b.n)) for b in EVENT_BINS]
        used = np.unique(best)
        filled = 0
        for i in range(0, len(used), INFER_BATCH):
            chunk = used[i : i + INFER_BATCH]
            windows = []
            for w in chunk:
                rows = np.arange(starts[w], starts[w] + n_quakes, dtype=np.int64)
                t = float(cat.time[rows[n_quakes // 2]])
                win = Window(t_ref=t, quakes=rows, quake_hidden=is_q[rows], cat=cat)
                win.tides = tide_rows(
                    src, t - 0.5 * p.tides * 3600, t + 0.5 * p.tides * 3600, p.tides
                )
                if src.gnss is not None:
                    lat, lon = where_now(cat, rows, is_q[rows])
                    day = np.floor(t / DAY) * DAY
                    span = p.gnss_days * DAY
                    win.gnss = gnss_rows(src, lat, lon, day - span, day + span + DAY)
                    win.gnss_hidden = np.zeros(len(win.gnss), bool)
                windows.append(win)
            batch = write(src, windows)
            h = self._run(model, batch)
            tok = batch.tokens.to(self.device)
            sel = tok.hidden[..., 0] & tok.of("seismic_catalog")
            with torch.no_grad():
                logp_all = model.masked(h[sel])
            # Hidden quakes come out window by window, in token order.
            at = 0
            for w in chunk:
                rows = np.arange(starts[w], starts[w] + n_quakes)
                mine_rows = rows[is_q[rows]]
                n_here = len(mine_rows)
                mine = best[np.searchsorted(pos, mine_rows)] == w
                for c, lp in enumerate(logp_all):
                    part = lp[at : at + n_here][torch.as_tensor(mine, device=lp.device)]
                    out[c][q_id[mine_rows[mine]]] = part.double().cpu().numpy()
                filled += int(mine.sum())
                at += n_here
        if filled != n_q:
            raise RuntimeError(f"{n_q - filled} masked quakes got no window")
        return out

    def gnss_dists(
        self,
        model: CratonModel,
        src: Sources,
        visible: GnssDays,
        queries: MaskedQueries,
        gi: IntArray,
    ) -> dict[str, ChannelDist]:
        """A logistic mixture per channel for every masked GNSS day ``gi`` of the queries."""
        lines = visible.lines
        code_ix = {c: i for i, c in enumerate(lines.codes)}
        q_station = np.array([code_ix[queries.qualifiers[i]["station"]] for i in gi])
        q_time = queries.time[gi]
        n_vis = len(visible)
        days = GnssDays(
            np.concatenate([visible.time, q_time]),
            np.concatenate([visible.station, q_station]),
            np.concatenate([visible.enu, np.full((len(gi), 3), np.nan)]),
            lines,
        )
        order = np.argsort(days.time, kind="stable")
        days = GnssDays(days.time[order], days.station[order], days.enu[order], lines)
        is_q = order >= n_vis
        q_of = np.where(is_q, order - n_vis, -1)
        src = Sources(src.cat, days, src.tides, src.policy, src.length, src.quakes)

        groups: list[IntArray] = []
        q_rows = np.nonzero(is_q)[0]
        q_day = np.floor(days.time[q_rows] / DAY)
        for d in np.unique(q_day):
            rows = q_rows[q_day == d]
            rows = rows[np.argsort(lines.site[days.station[rows], 1])]
            groups += [rows[i : i + GNSS_GROUP] for i in range(0, len(rows), GNSS_GROUP)]

        k = model.gnss.k
        logw = np.zeros((len(gi), 3, k))
        mu = np.zeros((len(gi), 3, k))
        s = np.ones((len(gi), 3, k))
        base = np.zeros((len(gi), 3))
        done = np.zeros(len(gi), bool)
        p = src.policy
        for i in range(0, len(groups), INFER_BATCH):
            windows = []
            for rows in groups[i : i + INFER_BATCH]:
                t = float(days.time[rows[0]])
                lat, lon = lines.site[days.station[rows]].mean(0)
                stations = tuple(int(x) for x in np.unique(days.station[rows]))
                day = np.floor(t / DAY) * DAY
                span = p.gnss_days * DAY
                win = Window(t_ref=t)
                win.tides = tide_rows(
                    src, t - 0.5 * p.tides * 3600, t + 0.5 * p.tides * 3600, p.tides
                )
                budget = max(p.gnss, len(stations) * (2 * p.gnss_days + 1))
                win.gnss = gnss_rows(
                    src, lat, lon, day - span, day + span + DAY, first=stations, budget=budget
                )
                win.gnss_hidden = is_q[win.gnss]
                if src.quakes:
                    n = src.length - win.size
                    win.quakes = quakes_around(src.cat, t, max(n, 0))
                    win.quake_hidden = np.zeros(len(win.quakes), bool)
                windows.append(win)
            batch = write(src, windows)
            h = self._run(model, batch)
            lw, m, sc, b = self._gnss_answers(model, batch, h)
            qi = q_of[batch.gnss_rows]
            take = (qi >= 0) & ~done[qi]
            logw[qi[take]], mu[qi[take]], s[qi[take]] = lw[take], m[take], sc[take]
            base[qi[take]] = b[take]
            done[qi[take]] = True
        if not done.all():
            raise RuntimeError(f"{int((~done).sum())} masked GNSS days got no window")
        return {
            ch: LogisticMixture(base[:, c], logw[:, c], mu[:, c], s[:, c])
            for c, ch in enumerate(GNSS_CHANNELS)
        }

    # ------------------------------------------------------------ CB-04

    def fill(self, queries: GapQueries, ctx: BenchContext, seed: int) -> FloatArray:
        """Every day of every held-out station, rebuilt with its line borrowed."""
        if "gnss_daily" not in self.inputs:
            raise ValueError(f"{self.name} reads no GNSS; it cannot rebuild a station")
        model = self.model(ctx, "pretrain", seed)
        train_g = ctx.train_gnss("pretrain")
        sites = _sites(train_g)
        q_station = np.asarray(queries.station)
        for code in sorted(set(queries.station)):
            first = int(np.nonzero(q_station == code)[0][0])
            sites[code] = (float(queries.site_lat[first]), float(queries.site_lon[first]))
        lines = fit_lines(train_g, sites)
        g = ctx.gnss
        g = g.select(~np.isin(g.station, ctx.heldout))
        visible = GnssDays.of(g, lines)
        code_ix = {c: i for i, c in enumerate(lines.codes)}
        st = np.array([code_ix[c] for c in queries.station])
        n_vis = len(visible)
        time = np.concatenate([visible.time, queries.time])
        order = np.argsort(time, kind="stable")
        days = GnssDays(
            time[order],
            np.concatenate([visible.station, st])[order],
            np.concatenate([visible.enu, np.full((len(st), 3), np.nan)])[order],
            lines,
        )
        q_of = np.where(order >= n_vis, order - n_vis, -1)
        unseen = frozenset(int(code_ix[c]) for c in set(queries.station))
        src = self.sources(ctx, ctx.events, days)
        p = src.policy

        windows_rows: list[IntArray] = []
        for code in sorted(set(queries.station)):
            rows = np.nonzero((days.station == code_ix[code]) & (q_of >= 0))[0]
            day = np.floor(days.time[rows] / DAY)
            block = np.floor((day - day[0]) / GAP_CHUNK_DAYS)
            windows_rows += [rows[block == b] for b in np.unique(block)]

        out = np.zeros((len(st), 3))
        done = np.zeros(len(st), bool)
        for i in range(0, len(windows_rows), INFER_BATCH):
            windows = []
            for rows in windows_rows[i : i + INFER_BATCH]:
                s_ix = int(days.station[rows[0]])
                lat, lon = lines.site[s_ix]
                t0, t1 = float(days.time[rows[0]]), float(days.time[rows[-1]])
                t = 0.5 * (t0 + t1)
                span = p.gnss_days * DAY
                win = Window(t_ref=t, unseen=unseen)
                win.tides = tide_rows(
                    src, t - 0.5 * p.tides * 3600, t + 0.5 * p.tides * 3600, p.tides
                )
                budget = p.gnss + len(rows)
                win.gnss = gnss_rows(
                    src, lat, lon, t0 - span, t1 + span + DAY, first=(s_ix,), budget=budget
                )
                win.gnss_hidden = np.zeros(len(win.gnss), bool)
                if src.quakes:
                    n = src.length - win.size
                    win.quakes = quakes_around(src.cat, t, max(n, 0))
                    win.quake_hidden = np.zeros(len(win.quakes), bool)
                windows.append(win)
            batch = write(src, windows)
            h = self._run(model, batch)
            tok = batch.tokens.to(self.device)
            sel = tok.hidden[..., 0] & tok.of("gnss_daily")
            with torch.no_grad():
                mean = model.gnss.mean_mm(h[sel], batch.gnss_years.to(self.device))
            pred = batch.gnss_base + mean.double().cpu().numpy() / 1000.0
            qi = q_of[batch.gnss_rows]
            take = (qi >= 0) & ~done[np.maximum(qi, 0)]
            # Each day is answered by the window built around its own chunk.
            own = np.zeros(len(qi), bool)
            at = 0
            for rows, win in zip(windows_rows[i : i + INFER_BATCH], windows, strict=True):
                n_hidden = int(np.isin(days.station[win.gnss], list(unseen)).sum())
                here = batch.gnss_rows[at : at + n_hidden]
                own[at : at + n_hidden] = np.isin(here, rows)
                at += n_hidden
            take &= own
            out[qi[take]] = pred[take]
            done[qi[take]] = True
        if not done.all():
            raise RuntimeError(f"{int((~done).sum())} held-out days got no window")
        return out

    # ------------------------------------------------------------ CB-17

    def predict_in_context(
        self, queries: RegionQueries, days: int, ctx: BenchContext, seed: int
    ) -> dict[str, ChannelDist]:
        """The region's first ``days`` days of quakes (the latest ``region_context`` of them),
        then its test quakes, all hidden, in chunks that fill each window: the same shape as
        the region windows it trained on. Nothing else of the region is read; the model
        learned on ``cb17_fit``, which never held a token from inside it."""
        if "seismic_catalog" not in self.inputs:
            raise ValueError(f"{self.name} reads no quakes; it cannot adapt to a region from them")
        model = self.model(ctx, "cb17_fit", seed)
        c_time, c_values, c_types = queries.context(days)
        keep = min(len(c_time), region_context(self.config.model.context_tokens))
        c_time, c_values = c_time[len(c_time) - keep :], c_values[len(c_values) - keep :]
        c_types = c_types[len(c_types) - keep :]
        chunk = self.config.model.context_tokens - keep
        c_label = np.array([magnitude_type_id(t) for t in c_types], dtype=np.int64)
        q_label = np.array([magnitude_type_id(t) for t in queries.magnitude_type], dtype=np.int64)
        empty = Catalog(np.zeros(0), np.zeros((0, 4)), np.zeros(0, np.int64))
        src = Sources(empty, None, None, self.config.context, self.config.model.context_tokens)
        windows = []
        for i in range(0, len(queries.time), chunk):
            q = slice(i, i + chunk)
            n_q = len(queries.time[q])
            cat = Catalog(
                np.concatenate([c_time, queries.time[q]]),
                np.concatenate([c_values, np.zeros((n_q, 4))]),
                np.concatenate([c_label, q_label[q]]),
            )
            hidden = np.concatenate([np.zeros(len(c_time), bool), np.ones(n_q, bool)])
            rows = np.arange(len(cat), dtype=np.int64)
            windows.append(
                Window(t_ref=float(queries.time[i]), quakes=rows, quake_hidden=hidden, cat=cat)
            )
        logp: list[list[FloatArray]] = [[] for _ in EVENT_BINS]
        for i in range(0, len(windows), INFER_BATCH):
            batch = write(src, windows[i : i + INFER_BATCH])
            h = self._run(model, batch)
            tok = batch.tokens.to(self.device)
            sel = tok.hidden[..., 0] & tok.of("seismic_catalog")
            with torch.no_grad():
                for c, lp in enumerate(model.masked(h[sel])):
                    logp[c].append(lp.double().cpu().numpy())
        return {
            ch: Histogram(b.edges, np.exp(np.concatenate(lp)))
            for ch, b, lp in zip(EVENT_CHANNELS, EVENT_BINS, logp, strict=True)
        }

    # ------------------------------------------------------------ CB-02, CB-03

    def forecast(self, spec: ForecastSpec, ctx: BenchContext, seed: int) -> GriddedForecast:
        model = self.model(ctx, spec.fit_split, seed)
        gnss = None
        if "gnss_daily" in self.inputs:
            g = ctx.gnss  # each midnight reads only the days before it
            gnss = GnssDays.of(
                g.select(~np.isin(g.station, ctx.heldout)), self.lines(ctx, spec.fit_split)
            )
        src = self.sources(ctx, ctx.events, gnss)
        days = spec.day_starts
        rates, shares = [], []
        for i in range(0, len(days), INFER_BATCH):
            chunk = days[i : i + INFER_BATCH]
            batch = write(src, [midnight_window(src, float(d)) for d in chunk])
            h = self._run(model, batch)[:, -1]
            recent = recent_rates(src.cat, chunk).to(self.device)
            with torch.no_grad():
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


register_method(CratonMini())
