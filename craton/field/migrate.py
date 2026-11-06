"""CB-24 through a Field: rank the next cell to light by tomorrow's forecast.

Before each lighting the Field reads what CB-24 allows (the box's quakes before the cut, plus
the quakes that lit the cells so far) as if that were the whole catalogue, and forecasts the
next day from the moment of the last lighting. Its rate of quakes at the lighting threshold
in each cell is the ranking. ETAS alone (the starting line) ranks the same way with the
Field switched off, so the three rankers differ only in what is learned:

- ``etas``: ETAS's own sum; the wiring's G = 0;
- a Field config (``mini``): the P6 correction on top of ETAS;
- ``wiring``: the same, plus Omori attention re-weighting which parents feed which cell.

No Field is trained for CB-24. Central Italy's sequences read the Field trained for the
forecast task whose cut comes before them (Amatrice-Norcia: CB-02's, cut 2016-08-24;
L'Aquila: CB-03's, cut 2009-01-01), with that task's ETAS fit and seed. Every other sequence
is zero-shot: CB-02's Field, ETAS fitted on that sequence's own box (CB-22's fit), the
memory counting from the region's completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton.bench import etas
from craton.bench.baselines import Etas
from craton.bench.data import Events
from craton.bench.migration import MigrationQuery, cases
from craton.bench.tasks import TASKS, BenchContext, SequenceForecast
from craton.field.base import KernelCache
from craton.field.config import load_field_config
from craton.field.forecaster import FieldForecaster, build_days, predict_members
from craton.field.memory import MemorySpec
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Central Italy's sequences and the forecast task whose Field and ETAS answer them.
ITALY = {"amatrice_norcia": "CB-02", "laquila": "CB-03"}
ZERO_SHOT_FROM = "CB-02"


def _concat(a: Events, b: Events) -> Events:
    keys = ("time", "lat", "lon", "depth_km", "mag", "mag_type", "token_hash")
    both = {k: np.concatenate([getattr(a, k), getattr(b, k)]) for k in keys}
    order = np.argsort(both["time"], kind="stable")
    return Events(**{k: v[order] for k, v in both.items()})


@dataclass
class FieldRanker:
    """A MigrationRanker from a Field config's trained members (``config=None``: ETAS)."""

    config: str | None = "wiring"
    root: Path = field(default_factory=lambda: Path.cwd())
    kind: str = "model"
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"field-{self.config}" if self.config else "etas-ranker"

    def _case(self, key: str, seed: int) -> tuple[Any, etas.EtasFit, list[Any]]:
        """The case's view (its events), its ETAS fit and the Field members that answer it."""
        ck = f"{key}:{seed}"
        if ck in self._cache:
            out: tuple[Any, etas.EtasFit, list[Any]] = self._cache[ck]
            return out
        ctx = BenchContext(root=self.root)
        task = ITALY.get(key)
        forecast_task = TASKS[task or ZERO_SHOT_FROM]
        assert isinstance(forecast_task, SequenceForecast)
        own = forecast_task.spec
        if task:
            spec, view = own, ctx
        else:
            from craton.adapters.region import load_region
            from craton.bench.world import SEQUENCES, sequence_spec, view_for

            seq = next(s for s in SEQUENCES if s.key == key)
            spec = sequence_spec(seq, load_region(seq.region, self.root))
            view = view_for(self.root, seq.region, spec)
        fit = Etas().fit(spec, view, seed)
        states: list[Any] = []
        if self.config:
            fc = FieldForecaster(self.config)
            own_fit = fit if task else Etas().fit(own, ctx, seed)
            states, _ = fc.states(own, ctx, seed, own_fit)
        self._cache[ck] = (view, fit, states)
        return view, fit, states

    def _kernels(self, query: MigrationQuery, fit: etas.EtasFit) -> KernelCache:
        key = f"kernels:{query.case}:{id(fit)}"
        if key not in self._cache:
            self._cache[key] = KernelCache(fit, query.grid, etas.Projection.for_grid(query.grid))
        out: KernelCache = self._cache[key]
        return out

    def next_cell_scores(self, query: MigrationQuery, seed: int) -> FloatArray:
        _, fit, states = self._case(query.case, seed)
        cfg = load_field_config(self.config or "mini")
        events = _concat(query.history, query.lit)
        start = np.array([float(query.lit.time.max()) + 1.0])
        mspec = MemorySpec(
            m_min=min(cfg.memory.m_min, query.mc),
            start=to_seconds(cfg.memory.start),
            m_ref=query.mc,
        )
        days = build_days(
            fit,
            events,
            start,
            query.grid,
            cfg,
            False,
            scored=query.threshold,
            memory_spec=mspec,
            kernel_cache=self._kernels(query, fit),
        )
        if not states:
            return np.asarray(days.log_etas[0, :, 1], dtype=np.float64)
        rates = predict_members(states, days, query.grid, cfg).mean(axis=0)[0]
        return np.asarray(np.log(np.maximum(rates, 1e-300)), dtype=np.float64)


def wiring_graph(
    config: str, key: str, seed: int, root: Path, at: str | None = None
) -> tuple[FloatArray, Any]:
    """(cells, cells + 2) the members' mean learned declustering matrix for sequence ``key``
    on the day of its cut (or ``at``): row = target cell, columns = source cells, the outside,
    the background. With the case's grid."""
    from craton.field.wiring import WiredFieldNet

    case = next(c for c in cases(root) if c.key == key)
    view, fit, states = FieldRanker(config, root)._case(key, seed)
    cfg = load_field_config(config)
    ev: Events = view.events
    t = to_seconds(at or case.cut)
    hist = ev.select(ev.time < t).above(case.mc)
    mspec = MemorySpec(m_min=case.mc, start=to_seconds(cfg.memory.start), m_ref=case.mc)
    days = build_days(
        fit, hist, np.array([t]), case.grid, cfg, False, scored=case.threshold, memory_spec=mspec
    )
    assert days.wiring is not None
    src, out, bg, kern = days.wiring
    mats = []
    for st in states:
        net = WiredFieldNet(days.memory.shape[-1], case.grid, cfg.model, cfg.wiring, kern)  # type: ignore[arg-type]
        net.load_state_dict(st)
        net.eval()
        with torch.no_grad():
            mem = torch.from_numpy(days.memory).float()
            le = torch.from_numpy(days.log_etas[..., 1])
            h = net.field.encode(mem, le)
            arrays = [torch.from_numpy(a) for a in (src, out, bg)]
            mats.append(net.pair_matrix(h, *arrays)[0].double().numpy())
    return np.mean(mats, axis=0), case.grid
