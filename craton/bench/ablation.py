"""Ablations: what each instrument is worth to the model.

Train Craton-mini again with one instrument left out, score it on the same frozen tasks, and
the difference from the full model is that instrument's **share**: how much worse the model
gets without it. The plan's question for P5 was whether GPS and the tides (the Moon, really)
carry anything the earthquake catalogue doesn't. This is the measurement.

    craton bench ablate     every variant, 3 seeds each (checkpoints are cached)
                            → results/ablation/<variant>.json → docs/ablation.md

A variant only answers what it can: one without GNSS cannot rebuild a GNSS station (CB-04),
one without quakes cannot forecast them (CB-02, CB-03). CB-01 is answered by every variant,
with climatology standing in for any instrument the variant doesn't read (as in P4, D-042),
so its quake and GNSS halves are reported separately too.

Ablation runs never go on the scoreboard: they are a measurement of the model, not a model.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from craton.bench import harness
from craton.bench.ratchet import NOISE_SIGMAS, improvement
from craton.bench.results import RunResults
from craton.bench.tasks import BenchContext
from craton.paths import results_dir

Log = Callable[[str], None]

QUAKES, GNSS, TIDES = "seismic_catalog", "gnss_daily", "solid_earth_tide"
ALL = (QUAKES, GNSS, TIDES)

#: Every variant: the instruments it reads, and the tasks it can answer.
VARIANTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "all": (ALL, ("CB-01", "CB-02", "CB-03", "CB-04")),
    "no_tides": ((QUAKES, GNSS), ("CB-01", "CB-02", "CB-03", "CB-04")),
    "no_gnss": ((QUAKES, TIDES), ("CB-01", "CB-02", "CB-03")),
    "no_quakes": ((GNSS, TIDES), ("CB-01", "CB-04")),
}
#: Which variant leaves out which instrument.
LEFT_OUT = {QUAKES: "no_quakes", GNSS: "no_gnss", TIDES: "no_tides"}
NAMES = {QUAKES: "Earthquakes", GNSS: "GNSS", TIDES: "Tides"}

#: The rows of the table: (task, extra or None for the score, label, unit, direction).
ROWS: tuple[tuple[str, str | None, str, str, str], ...] = (
    ("CB-01", None, "CB-01 geo-perplexity", "bits", "lower"),
    ("CB-01", "seismic_catalog_bits_per_token", "  · a hidden quake", "bits", "lower"),
    ("CB-01", "gnss_daily_bits_per_token", "  · a hidden GNSS day", "bits", "lower"),
    ("CB-02", None, "CB-02 Apennines 2016-17", "bits/quake", "higher"),
    ("CB-03", None, "CB-03 L'Aquila 2009", "bits/quake", "higher"),
    ("CB-04", None, "CB-04 GPS gap-fill", "mm", "lower"),
)


def ablation_dir(root: Path) -> Path:
    return results_dir(root) / "ablation"


def method(variant: str) -> object:
    """Craton-mini reading only the variant's instruments (its own checkpoints, by config)."""
    from craton.train.method import CratonMini

    inputs, _ = VARIANTS[variant]
    base = CratonMini().config
    if variant == "all":
        return CratonMini()
    cfg = base.model_copy(update={"model": base.model.model_copy(update={"inputs": inputs})})
    return CratonMini(name=f"craton-mini-{variant.replace('_', '-')}", override=cfg)


def run(
    ctx: BenchContext,
    variants: Sequence[str],
    seeds: Sequence[int],
    log: Log,
) -> dict[str, RunResults]:
    """Run each variant on its tasks and write ``results/ablation/<variant>.json``."""
    out = {}
    folder = ablation_dir(ctx.root)
    folder.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        m = method(variant)
        if hasattr(m, "log"):
            m.log = log
        _, tasks = VARIANTS[variant]
        log(f"variant {variant}: {', '.join(tasks)}")
        res = harness.run(m, ctx, seeds, tasks, log=log, config_hash=getattr(m, "config_hash", "?"))
        path = folder / f"{variant}.json"
        path.write_text(res.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        out[variant] = res
    return out


def load(root: Path) -> dict[str, RunResults]:
    return {
        p.stem: RunResults.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(ablation_dir(root).glob("*.json"))
        if p.stem in VARIANTS
    }


@dataclass(frozen=True)
class Cell:
    """A number with its spread over seeds."""

    mean: float
    std: float


def cell(run: RunResults, task: str, extra: str | None) -> Cell | None:
    for r in run.results:
        if r.task_id != task:
            continue
        values = r.values if extra is None else r.extras.get(extra)
        if not values:
            return None
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return Cell(statistics.fmean(values), sd)
    return None


@dataclass(frozen=True)
class Share:
    """What leaving one instrument out costs on one row: positive = the instrument helps."""

    value: float
    noise: float

    @property
    def beyond_noise(self) -> bool:
        return abs(self.value) > self.noise


def share(runs: dict[str, RunResults], row: int, instrument: str) -> Share | None:
    task, extra, _, _, direction = ROWS[row]
    full = runs.get("all")
    without = runs.get(LEFT_OUT[instrument])
    if full is None or without is None:
        return None
    a, b = cell(full, task, extra), cell(without, task, extra)
    if a is None or b is None:
        return None
    return Share(improvement(direction, b.mean, a.mean), NOISE_SIGMAS * max(a.std, b.std))


def _num(x: float) -> str:
    return f"{x:+.2f}" if abs(x) < 100 else f"{x:+.0f}"


def render(runs: dict[str, RunResults]) -> str:
    """docs/ablation.md, from the JSON alone."""
    lines = [
        "<!-- Generated by `craton bench ablate` from results/ablation/*.json. Do not edit. -->",
        "",
        "# What each instrument is worth",
        "",
        "Craton-mini trained again with one instrument left out, scored on the same frozen "
        "tasks. A **share** is how much worse the model gets without that instrument, in the "
        "task's own unit (positive = the instrument helps). It counts only beyond noise: two "
        "standard deviations over 3 seeds, the ratchet's rule. Where a variant can't answer "
        "a task (no GNSS: no station to rebuild; no quakes: nothing to forecast), the cell "
        "is blank.",
        "",
        "| Task | All three | " + " | ".join(f"without {NAMES[i]}" for i in ALL) + " |",
        "|---|---|" + "---|" * len(ALL),
    ]
    full = runs.get("all")
    for k, (task, extra, label, unit, _) in enumerate(ROWS):
        base = cell(full, task, extra) if full else None
        head = f"{base.mean:.2f} ± {base.std:.2f} {unit}" if base else "·"
        cells = []
        for inst in ALL:
            s = share(runs, k, inst)
            if s is None:
                cells.append("·")
            elif s.beyond_noise:
                cells.append(f"**{_num(s.value)}**")
            else:
                cells.append(f"{_num(s.value)} (noise ±{s.noise:.2f})")
        lines.append(f"| {label.strip()} | {head} | " + " | ".join(cells) + " |")
    lines += ["", "Runs: " + ", ".join(f"`{r.run_id}`" for r in runs.values()) + "", ""]
    return "\n".join(lines)


def write_doc(root: Path) -> Path:
    path = root / "docs" / "ablation.md"
    path.write_text(render(load(root)), encoding="utf-8", newline="\n")
    return path
