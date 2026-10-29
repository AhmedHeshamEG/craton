"""CB-30, human-made quakes: forecasts where the cause is measured.

Everywhere else Craton forecasts quakes whose push nobody sees: plates move, stress builds,
and the only clue is the quakes before. In two places the push was made by people and written
down month by month:

- **Groningen** (the Netherlands): gas pumped out of a sandstone 3 km down since 1963. The
  reservoir's pressure fell from 35 to under 10 MPa, the rock compacted, and from 1991 it
  started to crack. Production was cut from 2014 and stopped in 2023. The push: the pressure
  model of the reservoir (NAM), one map a year.
- **Oklahoma** (USA): salty wastewater from oil and gas wells pumped into the Arbuckle
  formation, right above the basement faults. From a couple of M3 quakes a year before 2009
  to over 900 in 2015; volumes were cut in 2015-16 and the quakes fell with them. The push:
  every Arbuckle disposal well's monthly volume (Oklahoma Corporation Commission).

The test: from 2017-01-01 on, the years when both pushes were being turned down, one forecast
every 30 days for eight years (97 periods), scored like CB-19 in bits per quake. Every method
fits on the box's past only, and may read the forcing only as far as it was known at each
period's start (``craton.bench.forcing``). The reference is ETAS fitted on the box's own past,
blind to the push; the physics floor is ``etas_forced``, ETAS whose background follows the
push (the seismogenic-index idea of Shapiro et al. 2010 and Langenbruch & Zoback 2016). The
primary number is the gain over ETAS, mean of the two cases. A model that has learned
"pressure → quakes" beats ETAS; one that has learned it better than the textbook law beats
``etas_forced`` too.

Groningen's pressures come from a model matched in 2023 to the measurements of every year,
so its past is read with hindsight (as CB-20's catalogue is); Oklahoma's volumes are the
operators' reports, used as if known at the end of each month.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
from scipy.optimize import minimize

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.forcing import (
    SPREADS,
    ForcingSpec,
    pushes,
    rate_field,
    raw,
)
from craton.bench.splits import Window
from craton.bench.tasks import (
    BenchContext,
    Forecaster,
    ForecastSpec,
    GriddedForecast,
    Score,
    Task,
    _write_table,
    register_task,
    score_forecast,
)
from craton.spec.timeaxis import to_iso, to_seconds

if TYPE_CHECKING:
    from craton.bench.world import RegionView

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Nothing at or after this moment is fitted or trained on, in either case.
CUT = "2017-01-01T00:00:00Z"
PERIOD_DAYS = 30
N_PERIODS = 97  # eight years: to 2024-12-19
MAX_MAGNITUDE = 9.5
#: The physics floor fits its two numbers on 30-day periods from this long after the
#: forcing's first known period (a year, so its 365-day rate is whole).
WARM_UP_DAYS = 365


@dataclass(frozen=True)
class Case:
    """One place where people pushed the crust and wrote down how hard."""

    key: str
    name: str
    region: str
    box: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    cell_deg: float
    mc: float  # completeness: fitted and learned from here
    scored: float  # the primary magnitude
    fit_from: str  # ETAS's fit window starts here
    train_from: str  # the Field's first training day
    forcing: ForcingSpec
    about: str


CASES = (
    Case(
        "groningen",
        "Groningen gas field",
        "groningen",
        (53.0, 53.6, 6.4, 7.3),
        0.05,
        1.0,
        1.5,
        "2006-01-01T00:00:00Z",
        "2006-01-01T00:00:00Z",
        ForcingSpec("reservoir_pressure"),
        "gas production; the reservoir's modelled pressure, one map a year",
    ),
    Case(
        "oklahoma",
        "Oklahoma wastewater",
        "oklahoma",
        (34.0, 37.0, -99.5, -96.0),
        0.1,
        2.5,
        3.0,
        "2010-01-01T00:00:00Z",
        "2012-01-01T00:00:00Z",
        ForcingSpec("fluid_injection", ("2D",), ("ARBUCKLE",)),
        "wastewater disposal into the Arbuckle; every such well's (class 2D) monthly volume",
    ),
)
CASE_BY_KEY = {c.key: c for c in CASES}


def case_spec(case: Case) -> ForecastSpec:
    lat0, lat1, lon0, lon1 = case.box
    grid = csep.Grid(lat0, lat1, lon0, lon1, cell_deg=case.cell_deg, m_min=case.mc, m_max=9.5)
    t0 = to_seconds(CUT)
    window = Window(f"cb30_{case.key}", CUT, to_iso(t0 + N_PERIODS * PERIOD_DAYS * DAY) or CUT)
    return ForecastSpec(
        task_id=f"CB-30-{case.key}",
        fit_split="induced",
        window=window,
        grid=grid,
        period_days=PERIOD_DAYS,
        mc=case.mc,
        fit_from=case.fit_from,
        cut=CUT,
        thresholds=(case.scored, case.mc),
        max_magnitude=MAX_MAGNITUDE,
    )


def case_of(spec: ForecastSpec) -> Case:
    key = spec.task_id.removeprefix("CB-30-")
    if key not in CASE_BY_KEY:
        raise KeyError(f"{spec.task_id} is not a CB-30 case")
    return CASE_BY_KEY[key]


def forecast_of(
    method: Any, spec: ForecastSpec, view: RegionView, case: Case, seed: int
) -> GriddedForecast:
    """A method's forecast of one case: with the forcing if it reads forcing, else blind."""
    forced = getattr(method, "forced_forecaster", None)
    if callable(forced):
        out: GriddedForecast = forced().forecast_forced(spec, view, case.forcing, seed)
        return out
    if hasattr(method, "forecast_forced"):
        return cast(GriddedForecast, method.forecast_forced(spec, view, case.forcing, seed))
    return cast(GriddedForecast, method.forecast(spec, view, seed))


# ---------------------------------------------------------------- the physics floor


def period_counts(ev: Events, grid: csep.Grid, starts: FloatArray, span_days: int) -> np.ndarray:
    """(periods, cells) quakes at or above the grid's m_min in each period."""
    out = np.zeros((len(starts), grid.n_cells))
    ev = ev.above(grid.m_min)
    cell = grid.cell_of(ev.lat, ev.lon)
    k = np.searchsorted(starts, ev.time, side="right") - 1
    ok = (cell >= 0) & (k >= 0)
    ok &= ev.time < starts[np.maximum(k, 0)] + span_days * DAY
    np.add.at(out, (k[ok], cell[ok]), 1.0)
    return out


#: A cell's push is compared with its own past push plus this share of the box's mean, so a
#: cell nobody pumped near reads a steady 1 instead of 0/0.
PUSH_FLOOR = 0.05
#: The largest exponent the background may follow the push with.
MAX_EXPONENT = 4.0


@dataclass(frozen=True)
class ForcedFit:
    """ETAS's triggering, and ETAS's background map turned up and down by the push:
    background(cell, period) = b · ETAS's background(cell) · (push now / its past push)^k."""

    b: float
    k: float
    spread: float
    floor: float  # added to every push: PUSH_FLOOR × the box's mean push over the fit
    past: np.ndarray  # (cells,) each cell's mean push over the fit periods, plus the floor
    loglik: float


@dataclass
class EtasForced:
    """ETAS whose background follows the forcing: the physics floor of CB-30.

    The seismogenic index (Shapiro et al. 2010; Langenbruch & Zoback 2016) says induced
    quakes come at a rate set by the push, scaled by how ready the faults under it are. ETAS's
    smoothed background, fitted on the box's past, is the map of how ready they are; this floor
    keeps that map and lets each cell's rate move with its push relative to its own past:
    ``(push now / past push)^k``. ``k`` = 1 is the textbook proportional law, ``k`` = 0 is
    ETAS; ``b`` and ``k`` are fitted by maximum likelihood on the past's 30-day periods, the
    push read as the 365-day rate in the cell or blurred over 1 or 3 cells, whichever fits
    best. The triggering is ETAS's own. Answers CB-30 only: without a forcing it has nothing
    to follow.
    """

    name: str = "etas_forced"
    kind: str = "baseline"
    long_periods: ClassVar[bool] = True

    def _pieces(
        self, spec: ForecastSpec, view: BenchContext, fit: etas.EtasFit, starts: FloatArray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """(triggered rates, ETAS's background per cell, cascade factor) for ``starts``."""
        proj = etas.Projection.for_grid(spec.grid)
        hist = view.events.window(fit.t_ref, float(starts.max())).above(fit.mc)
        cat = etas.to_catalog(hist.time, hist.lat, hist.lon, hist.mag, proj, fit.t_ref)
        days = (starts - fit.t_ref) / DAY
        rates = etas.forecast_days(fit, cat, days, spec.grid, proj, span=spec.period_days).rates
        factor = etas.cascade_factor(fit.params, fit.b, fit.mc, spec.period_days)
        bg = spec.period_days * fit.params.mu * fit.background.cell_mass(spec.grid, proj)
        return np.maximum(rates - bg[None] * factor, 0.0), bg, factor

    def fit_forced(
        self, spec: ForecastSpec, view: BenchContext, forcing: ForcingSpec, seed: int
    ) -> tuple[etas.EtasFit, ForcedFit]:
        from craton.bench.baselines import Etas

        key = f"etas-forced:{spec.task_id}:{seed}"
        if key in view._cache:
            out: tuple[etas.EtasFit, ForcedFit] = view._cache[key]
            return out
        fit = Etas().fit(spec, view, seed)
        p = pushes(view, forcing, spec.grid)
        first = max(to_seconds(spec.fit_from), float(p.known_until(p.t_end)[0]))
        first = first + WARM_UP_DAYS * DAY
        end = to_seconds(spec.fit_end) - spec.period
        starts = np.arange(first, end + 1, spec.period)
        trig, bg, factor = self._pieces(spec, view, fit, starts)
        n = period_counts(view.train_events(spec.fit_split), spec.grid, starts, spec.period_days)
        r = raw(p, spec.grid, starts)
        best: ForcedFit | None = None
        for s in SPREADS:
            push = rate_field(r, spec.grid, 365, s)
            floor = PUSH_FLOOR * float(push.mean())
            if floor <= 0:
                continue
            past = push.mean(axis=0) + floor
            ratio = np.log((push + floor) / past[None])
            y = factor * bg[None]

            def nll(theta: FloatArray, ratio: np.ndarray = ratio, y: np.ndarray = y) -> float:
                lam = trig + np.exp(theta[0] + theta[1] * ratio) * y
                return float(-(n * np.log(np.maximum(lam, 1e-300)) - lam).sum())

            res = minimize(
                nll,
                np.array([0.0, 1.0]),
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0), (0.0, MAX_EXPONENT)],
            )
            cand = ForcedFit(
                float(np.exp(res.x[0])), float(res.x[1]), s, floor, past, -float(res.fun)
            )
            if best is None or cand.loglik > best.loglik:
                best = cand
        if best is None:
            raise ValueError(f"{spec.task_id}: no push inside the box before the cut")
        view._cache[key] = (fit, best)
        return fit, best

    def forecast_forced(
        self, spec: ForecastSpec, view: BenchContext, forcing: ForcingSpec, seed: int
    ) -> GriddedForecast:
        from craton.bench.baselines import _shares

        fit, ff = self.fit_forced(spec, view, forcing, seed)
        starts = spec.day_starts
        trig, bg, factor = self._pieces(spec, view, fit, starts)
        r = raw(pushes(view, forcing, spec.grid), spec.grid, starts)
        push = rate_field(r, spec.grid, 365, ff.spread)
        mod = ((push + ff.floor) / ff.past[None]) ** ff.k
        rates = trig + factor * ff.b * bg[None] * mod
        scale, shares = _shares(fit.b, fit.mc, spec.grid)
        return GriddedForecast(rates * scale, shares)


# ---------------------------------------------------------------- the task


def _mean(v: list[float]) -> float:
    return float(np.mean(v)) if v else math.nan


class InducedQuakes(Task):
    task_id = "CB-30"
    name = "Human-made quakes"
    metric = "bits per quake above ETAS fitted on the box's own past, mean of the two cases"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        return {
            "cases": [
                {
                    **{k: v for k, v in c.__dict__.items() if k != "forcing"},
                    "forcing": c.forcing.as_dict(),
                }
                for c in CASES
            ],
            "cut": CUT,
            "periods": f"{N_PERIODS} periods of {PERIOD_DAYS} days from the cut",
            "magnitudes": "from each case's completeness; scored from its `scored` (primary) "
            "and from its completeness",
            "reference": "ETAS fitted on the box's quakes from fit_from to the cut, run period "
            "by period",
            "physics_floor": "etas_forced: ETAS with a background that follows the forcing",
            "forcing": "read only as far as known at each period's start (a period's push is "
            "known once the period has ended)",
            "primary": "mean over the cases of the gain over the reference, bits per quake "
            "at the case's scored magnitude",
            "scoring": "as CB-19 (score_forecast), max magnitude 9.5",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        from craton.bench.planets import events_table
        from craton.bench.world import observed, view_for

        folder = ctx.task_dir(self.task_id)
        files: dict[str, str] = {}
        summary: dict[str, Any] = {"snapshots": {}}
        for case in CASES:
            spec = case_spec(case)
            view = view_for(ctx.root, case.region, spec)
            summary["snapshots"][case.region] = view.snapshot_id
            obs = observed(view, spec)
            files[f"{case.key}.parquet"] = _write_table(
                folder / f"{case.key}.parquet", events_table(obs)
            )
            hist = view.train_events("induced").above(spec.mc)
            summary[case.key] = {
                "box": list(view.box),
                "window": spec.window.as_dict(),
                "mc": spec.mc,
                f"M>={spec.scored:g}": int((obs.mag >= spec.scored - 1e-9).sum()),
                f"M>={spec.mc:g}": int((obs.mag >= spec.mc - 1e-9).sum()),
                "largest": float(obs.mag.max()) if len(obs) else None,
                "fit_quakes": int((hist.time >= to_seconds(spec.fit_from)).sum()),
            }
        return {"files": files, "summary": summary}

    def supports(self, method: Any) -> bool:
        if callable(getattr(method, "forced_forecaster", None)):
            return True
        if hasattr(method, "forecast_forced"):
            return True
        return (
            isinstance(method, Forecaster)
            and getattr(method, "long_periods", False)
            and not getattr(method, "regional_only", False)
        )

    def frozen_snapshots(self, ctx: BenchContext) -> dict[str, str]:
        from craton.bench.tasks import load_frozen

        frozen = load_frozen(ctx.region) or {}
        task = frozen.get("tasks", {}).get(self.task_id, {})
        out: dict[str, str] = task.get("summary", {}).get("snapshots", {})
        return out

    def observed_of(self, ctx: BenchContext, case: Case) -> Events:
        from craton.bench.planets import read_events

        return read_events(ctx.task_dir(self.task_id) / f"{case.key}.parquet")

    def view(self, ctx: BenchContext, case: Case, spec: ForecastSpec) -> RegionView:
        from craton.bench.world import view_for

        key = f"cb30-view:{case.key}"
        if key not in ctx._cache:
            snaps = self.frozen_snapshots(ctx)
            ctx._cache[key] = view_for(ctx.root, case.region, spec, snaps.get(case.region, ""))
        out: RegionView = ctx._cache[key]
        return out

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        from craton.bench.baselines import Etas
        from craton.bench.world import _check

        extras: dict[str, float] = {}
        gains: list[float] = []
        for case in CASES:
            spec = case_spec(case)
            view = self.view(ctx, case, spec)
            obs = self.observed_of(ctx, case)
            fc = forecast_of(method, spec, view, case, seed)
            _check(fc, spec)
            sc = score_forecast(fc, obs, spec, seed)
            refs = {}
            for ref_name, ref in (("etas", Etas()), ("physics", EtasForced())):
                ref_key = f"cb30-{ref_name}:{case.key}:{seed}"
                if ref_key not in ctx._cache:
                    ctx._cache[ref_key] = score_forecast(
                        forecast_of(ref, spec, view, case, seed), obs, spec, seed
                    ).value
                refs[ref_name] = float(ctx._cache[ref_key])
            gain = sc.value - refs["etas"]
            gains.append(gain)
            tag = f"M{spec.scored:g}"
            extras[f"{case.key}.bits"] = sc.value
            extras[f"{case.key}.gain"] = gain
            extras[f"{case.key}.gain_over_physics"] = sc.value - refs["physics"]
            extras[f"{case.key}.physics_gain"] = refs["physics"] - refs["etas"]
            extras[f"{case.key}.observed"] = sc.extras[f"{tag}.observed"]
            extras[f"{case.key}.expected"] = sc.extras[f"{tag}.expected"]
            for part in ("how_many_bits", "where_bits", "how_big_bits"):
                extras[f"{case.key}.{part}"] = sc.extras.get(f"{tag}.{part}", math.nan)
            low = f"M{spec.mc:g}"
            extras[f"{case.key}.bits_mc"] = sc.extras.get(f"{low}.bits_per_eq", math.nan)
        return Score(_mean(gains), extras)


register_task(InducedQuakes())
