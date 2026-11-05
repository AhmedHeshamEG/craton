"""CB-26, the distilled law: does what the network learned fit in a short formula?

A method that forecasts CB-02 and CB-03 through a Field (``craton.field``) hands that Field to
symbolic regression (``craton.field.distill``): a formula of at most ten added terms over the
same counters the network reads is fitted to the network's correction to ETAS on the training
years only, picked on the training split's own validation days, and then forecasts each task's
test window exactly as the network does, ETAS times e^formula.

The number is the share of the network's gain over ETAS that the formula keeps on those test
windows, mean over CB-02 and CB-03: 1 is a formula as good as the network, 0 is ETAS. The plan
asks for ~0.8. A rule the network found survives the squeeze into ten terms; memory of the
training years doesn't. (Cranmer et al. 2020: their formula generalised better than the network
it came from.)

No new test data: the windows, quakes and scores are CB-02's and CB-03's, frozen in P3.
"""

from __future__ import annotations

from typing import Any

from craton.bench.baselines import Etas
from craton.bench.tasks import TASKS, BenchContext, Score, SequenceForecast, Task, register_task

SOURCES = ("CB-02", "CB-03")


class _Fixed:
    """A forecast already made, as a method (so the source task scores it its own way)."""

    def __init__(self, name: str, fn: Any) -> None:
        self.name, self.fn = name, fn

    def forecast(self, spec: Any, ctx: BenchContext, seed: int) -> Any:
        return self.fn(spec, ctx, seed)


class DistilledLaw(Task):
    task_id = "CB-26"
    name = "Distilled law"
    metric = "share of the network's held-out gain over ETAS a <=10-term formula keeps"
    direction = "higher"

    def definition(self) -> dict[str, Any]:
        from craton.field import distill

        return {
            "sources": list(SOURCES),
            "formula": f"at most {distill.MAX_TERMS} added terms, at most {distill.MAX_SIZE} "
            "nodes; inputs: the Field's counters, ETAS's log rate in the cell and in the box",
            "fit": "the network's correction on the training split, rows drawn by its expected "
            "M3+ count; picked on the split's validation days; test windows never read",
            "score": "(formula - ETAS) / (network - ETAS) in bits per M3+ quake on each "
            "source's test window, mean over the sources",
        }

    def prepare(self, ctx: BenchContext) -> dict[str, Any]:
        return {"files": {}, "summary": {"sources": list(SOURCES)}}

    def supports(self, method: Any) -> bool:
        return bool(getattr(method, "field_config", ""))

    def score(self, method: Any, ctx: BenchContext, seed: int) -> Score:
        from craton.field.distill import distill, law_forecast
        from craton.field.forecaster import FieldForecaster

        ff = FieldForecaster(method.field_config, log=getattr(method, "log", print))
        extras: dict[str, float] = {}
        shares = []
        for tid in SOURCES:
            task = TASKS[tid]
            assert isinstance(task, SequenceForecast)
            law = distill(ff, task.spec, ctx, seed)
            tag = f"{tid}.{ff.config.name}"
            sl = task.score(
                _Fixed(f"law:{tag}", lambda s, c, k, law=law: law_forecast(law, ff, s, c, k)),
                ctx,
                seed,
            ).value
            sn = task.score(_Fixed(f"field:{tag}", ff.forecast), ctx, seed).value
            se = task.score(_Fixed(f"etas:{tid}", Etas().forecast), ctx, seed).value
            share = (sl - se) / (sn - se)
            shares.append(share)
            extras.update(
                {
                    f"{tid}.share": share,
                    f"{tid}.law_gain_bits": sl - se,
                    f"{tid}.network_gain_bits": sn - se,
                    f"{tid}.terms": float(law.terms),
                    f"{tid}.complexity": float(law.complexity),
                    # the same share on the training split's validation days, which picked it
                    f"{tid}.validation_share": law.info["validation_gain"]
                    / law.info["network_validation_gain"],
                }
            )
        return Score(sum(shares) / len(shares), extras)


register_task(DistilledLaw())
