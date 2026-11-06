"""The distilled law (P9e, CB-26): a short formula pulled out of the Field's correction to ETAS.

The Field moves ETAS's rate in every cell on every day by a factor e^c, where c comes out of a
transformer reading 54 fading counters. If what it learned is a rule, a short formula of the
same counters should do most of what the network does, also on days neither ever saw. If it is
memory of the training years, no short formula will.

How (Cranmer et al. 2020, "Discovering symbolic models from deep learning with inductive
biases": fit the formula to the network, never to the data):

1. Every other training day, every cell: the counters, ETAS's log rate, and the network's
   correction c (the members' average rate over ETAS's, in logs).
2. Rows are drawn in proportion to the network's expected M3+ count. For a Poisson count, the
   price of answering f where the network says c is about ½ λ (f − c)², so drawing rows by λ and
   fitting squared error is fitting the network where its bits are.
3. PySR (symbolic regression) searches for formulas of the ten most useful inputs.
4. From PySR's list of best formulas at each size, the one kept is the one with at most
   ``MAX_TERMS`` added terms that gains most over ETAS on the training split's validation days
   (its own held-out blocks). The test window is never looked at to pick.
5. The formula forecasts the task's test window the way the Field does: ETAS's rate times
   e^f, with the Field's own bound on f.

The formula and how it was picked are cached beside the Field's checkpoint (``law.json``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from craton.bench import csep
from craton.bench.data import DAY
from craton.bench.tasks import BenchContext, ForecastSpec, GriddedForecast
from craton.field.forecaster import (
    VALIDATION_BLOCK_DAYS,
    Days,
    FieldForecaster,
    build_days,
    magnitude_shares,
    predict_members,
)
from craton.field.memory import LIFETIMES_DAYS, SPREADS
from craton.spec.timeaxis import to_seconds

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

LAW_VERSION = "l1"
#: A law has at most this many added terms (the plan's "≤ 10 terms").
MAX_TERMS = 10
#: The longest formula PySR may try, in nodes (variables, constants and operators).
MAX_SIZE = 30
#: Rows the search fits on, and inputs it keeps (by PySR's own random-forest screen).
ROWS = 4000
INPUTS = 10
ITERATIONS = 40
#: Every ``DAY_STRIDE``-th training day is read (memory at every midnight is a lot of bytes).
DAY_STRIDE = 2


def _lifetime_name(days: float) -> str:
    if days < 1:
        return f"{round(days * 24)}h"
    if days < 365:
        return f"{round(days)}d"
    return f"{round(days / 365.25)}y"


def input_names() -> list[str]:
    """Names of the formula's possible inputs, in the memory's order, then ETAS's two.

    ``n_<lifetime>_<spread>`` is the log count of quakes fading with that lifetime;
    ``p_...`` weighs each by its aftershock productivity 10^(m − 2). Spreads: the cell itself
    (``c``), ~10 km (``k10``), ~30 km (``k30``). ``etas`` is ETAS's log M3+ rate in the cell,
    ``box`` its log rate summed over the whole box."""
    spread = {0.0: "c", 1.0: "k10", 3.0: "k30"}
    out = []
    for s in SPREADS:
        for life in LIFETIMES_DAYS:
            for w in ("n", "p"):
                out.append(f"{w}_{_lifetime_name(life)}_{spread[s]}")
    return [*out, "etas", "box"]


def inputs_of(days: Days) -> np.ndarray:
    """(days, cells, inputs) float32: the memory, ETAS's log M3+ rate, and the box's."""
    le = days.log_etas[..., 1]
    box = np.broadcast_to(np.log(np.exp(le.astype(np.float64)).sum(axis=1))[:, None], le.shape)
    return np.concatenate(
        [days.memory.astype(np.float32), le[..., None], box[..., None].astype(np.float32)],
        axis=-1,
    )


def network_correction(
    ff: FieldForecaster, states: list[Any], days: Days, grid: csep.Grid
) -> FloatArray:
    """(days, cells) c: log of the members' average M3+ rate over ETAS's."""
    rate = predict_members(states, days, grid, ff.config).mean(axis=0)
    return np.asarray(np.log(rate) - days.log_etas[..., 1].astype(np.float64), dtype=np.float64)


@dataclass(frozen=True)
class Law:
    """A formula for the correction, and the record of how it was chosen."""

    expression: str  # sympy, over ``input_names()``
    terms: int
    complexity: int
    max_log: float  # the Field's bound: ln(max_nudge)
    info: dict[str, Any]

    def correction(self, x: np.ndarray) -> FloatArray:
        """(…, inputs) → (…) the bounded log correction."""
        f = _evaluate(self.expression, x)
        return np.asarray(self.max_log * np.tanh(f / self.max_log), dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": LAW_VERSION,
            "expression": self.expression,
            "terms": self.terms,
            "complexity": self.complexity,
            "max_log": self.max_log,
            **self.info,
        }


def _evaluate(expression: str, x: np.ndarray) -> FloatArray:
    import sympy

    names = input_names()
    syms = sympy.symbols(names)
    fn = sympy.lambdify(
        syms, sympy.sympify(expression, locals=dict(zip(names, syms, strict=True))), "numpy"
    )
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(fn(*flat.T), dtype=np.float64), (flat.shape[0],))
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out.reshape(x.shape[:-1]), dtype=np.float64)


def narrowed(law: Law, max_complexity: int) -> Law:
    """The best formula of the same search no longer than ``max_complexity`` nodes, picked the
    same way (validation days). CB-26's floor is one node: a constant factor on ETAS."""
    front = [
        r
        for r in law.info["front"]
        if r["complexity"] <= max_complexity and r["terms"] <= MAX_TERMS
    ]
    best = max(front, key=lambda r: float(r["validation_gain"]))
    info = {**law.info, "validation_gain": best["validation_gain"]}
    return Law(best["expression"], best["terms"], best["complexity"], law.max_log, info)


def count_terms(expression: str) -> int:
    """Added terms at the top of the formula, as written (nothing expanded)."""
    import sympy

    return len(sympy.Add.make_args(sympy.sympify(expression)))


def poisson_gain(log_base: FloatArray, f: FloatArray, counts: np.ndarray) -> float:
    """Bits per quake gained over ``log_base`` by moving it by ``f`` (M3+ counts)."""
    n = float(counts.sum())
    lam0, lam1 = np.exp(log_base), np.exp(log_base + f)
    ll = float(np.sum(counts * f) - np.sum(lam1 - lam0))
    return ll / (max(n, 1.0) * math.log(2))


def training_days(
    ff: FieldForecaster, spec: ForecastSpec, ctx: BenchContext, seed: int
) -> tuple[Days, np.ndarray, Any, list[Any]]:
    """Every ``DAY_STRIDE``-th training day of the Field, a mask of its validation days, the
    ETAS fit and the Field's members."""
    from craton.bench.baselines import Etas

    spec = ff.audited(spec)
    fit = Etas().fit(spec, ctx, seed)
    states, _ = ff.states(spec, ctx, seed, fit)
    cfg = ff.config
    starts = np.arange(to_seconds(cfg.train.first_day), to_seconds(spec.fit_end) - DAY + 1, DAY)
    idx = np.arange(len(starts))
    every = cfg.train.validation_every
    val = (idx // VALIDATION_BLOCK_DAYS) % every == every - 1
    pick = idx % DAY_STRIDE == 0
    days = build_days(fit, ctx.train_events(spec.fit_split), starts[pick], spec.grid, cfg, True)
    return days, val[pick], fit, states


def distill(
    ff: FieldForecaster, spec: ForecastSpec, ctx: BenchContext, seed: int, log: Any = print
) -> Law:
    """The Field's law for one task and seed (cached as ``law.json`` beside its checkpoint)."""
    path = ff.folder(ctx, spec, seed) / "law.json"
    max_log = math.log(ff.config.model.max_nudge)
    if path.is_file():
        body = json.loads(path.read_text("utf-8"))
        if body.get("version") == LAW_VERSION:
            info = {
                k: v
                for k, v in body.items()
                if k not in ("version", "expression", "terms", "complexity", "max_log")
            }
            return Law(body["expression"], body["terms"], body["complexity"], max_log, info)
    days, val, _, states = training_days(ff, spec, ctx, seed)
    assert days.counts is not None
    c = network_correction(ff, states, days, spec.grid)
    x = inputs_of(days)
    le = days.log_etas[..., 1].astype(np.float64)
    weight = np.exp(le + c)
    weight[val] = 0.0  # the search fits on training days; validation days pick the formula
    rng = np.random.default_rng(seed)
    flat_w = weight.ravel() / weight.sum()
    rows = rng.choice(flat_w.size, size=ROWS, replace=True, p=flat_w)
    xs = x.reshape(-1, x.shape[-1])[rows].astype(np.float64)
    ys = c.ravel()[rows]
    log(f"CB-26 seed {seed}: PySR on {ROWS} rows of {int((~val).sum())} days")
    table = _search(xs, ys, seed)
    n3 = days.counts[val][..., 1]
    net_val = poisson_gain(le[val], c[val], n3)
    best: dict[str, Any] | None = None
    front = []
    for _, r in table.iterrows():
        expr = str(r["sympy_format"])
        terms = count_terms(expr)
        f = max_log * np.tanh(_evaluate(expr, x[val]) / max_log)
        g = poisson_gain(le[val], f, n3)
        front.append(
            {
                "expression": expr,
                "complexity": int(r["complexity"]),
                "terms": terms,
                "validation_gain": g,
            }
        )
        if terms <= MAX_TERMS and (best is None or g > best["validation_gain"]):
            best = front[-1]
    assert best is not None
    info = {
        "task": spec.task_id,
        "seed": seed,
        "rows": ROWS,
        "days": len(days.starts),
        "network_validation_gain": net_val,
        "validation_gain": best["validation_gain"],
        "front": front,
    }
    law = Law(best["expression"], best["terms"], best["complexity"], max_log, info)
    path.write_text(json.dumps(law.as_dict(), indent=2) + "\n", "utf-8")
    return law


def _search(x: FloatArray, y: FloatArray, seed: int) -> Any:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=ITERATIONS,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp"],
        maxsize=MAX_SIZE,
        select_k_features=INPUTS,
        parallelism="serial",
        deterministic=True,
        random_state=seed,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    model.fit(x, y, variable_names=input_names())
    return model.equations_


def law_forecast(
    law: Law, ff: FieldForecaster, spec: ForecastSpec, ctx: BenchContext, seed: int
) -> GriddedForecast:
    """The task's test window forecast by ETAS times e^f, f the law (the Field's way)."""
    from craton.bench.baselines import Etas

    fit = Etas().fit(ff.audited(spec), ctx, seed)
    days = build_days(
        fit, ff.inputs(spec, ctx), spec.day_starts, spec.grid, ff.config, False, spec.period_days
    )
    m3 = np.exp(days.log_etas[..., 1].astype(np.float64) + law.correction(inputs_of(days)))
    shares = magnitude_shares(fit, spec.grid, spec.max_magnitude)
    first = int(spec.grid.mag_bin_of(np.array([spec.scored]))[0])
    return GriddedForecast(m3 / shares[first:].sum(), shares)
