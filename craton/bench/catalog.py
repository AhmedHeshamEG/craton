"""The planned CratonBench tasks.

This is the published list of what Craton will be measured on and when each test arrives.
The tasks themselves (data, splits, metrics) are built phase by phase; until a task is built
the scoreboard shows it as planned. An ID, once given, is never reused or redefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["lower", "higher"]


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    name: str
    measures: str
    phase: str
    headline: bool = False


CATALOG: tuple[PlannedTask, ...] = (
    PlannedTask(
        "CB-01", "Geo-perplexity", "bits per masked token in the held-out eras", "P3", True
    ),
    PlannedTask(
        "CB-02",
        "Apennines 2016-17 forecast",
        "bits per M3+ quake, daily CSEP-style forecasts",
        "P3",
        True,
    ),
    PlannedTask(
        "CB-03", "L'Aquila 2009 era holdout", "forecast skill trained only on data up to 2008", "P3"
    ),
    PlannedTask("CB-04", "GPS gap-fill", "RMSE (mm) rebuilding hidden GNSS stations", "P3"),
    PlannedTask("CB-05", "Calibration", "ECE, PIT uniformity, CRPS", "P7"),
    PlannedTask("CB-06", "Unseen region", "skill and abstention on Calabria + Sicily", "P7"),
    PlannedTask("CB-07", "Fake-precursor attack", "false alarms from injected fake signals", "P7"),
    PlannedTask(
        "CB-08", "Physics sanity", "slip budget, Gutenberg-Richter, Omori violations", "P7"
    ),
    PlannedTask("CB-09", "Sim-to-real gap", "synthetic vs real score gap per generator", "P8"),
    PlannedTask("CB-10", "Planted discovery", "tidal periods recovered by the ghost channel", "P9"),
    PlannedTask(
        "CB-11",
        "Predictability meter",
        "width of the 90% window in days, km, magnitude",
        "P7",
        True,
    ),
    PlannedTask(
        "CB-12", "Interpreter faithfulness", "numbers traced to tool calls, abstentions", "P10"
    ),
    PlannedTask("CB-13", "Microscope", "planted hidden variable found by a sparse feature", "P11"),
    PlannedTask("CB-14", "Deep time", "plate positions vs GPlates reconstructions", "P12"),
    PlannedTask("CB-15", "Efficiency", "tokens/s and peak VRAM on the RTX 2060", "P4"),
    PlannedTask("CB-16", "Scaling-law fit", "next size's loss, predicted before training", "P13"),
    PlannedTask(
        "CB-17", "In-context adaptation", "CB-01 gain from N days of a new region in context", "P5"
    ),
    PlannedTask("CB-18", "Ladder rung 3", "timing of repeating small earthquakes", "P13"),
)

BY_ID = {t.task_id: t for t in CATALOG}
