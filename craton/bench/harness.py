"""CratonBench's harness: prepare and freeze the tests, check for leaks, run a method.

    craton bench prepare           cut the splits, write the test files, compare with the frozen
    craton bench prepare --freeze  …and freeze it: the whole record the first time, later only
                                   tasks and splits it doesn't have yet (everything else must
                                   match)
    craton bench leaks             read every training shard back and look for held-out tokens
    craton bench run etas          score a method on every task it supports, 3 seeds
                                   → results/pending/<run>.json, judged by `craton ratchet`

The frozen record, ``craton/bench/frozen/<region>.json``, is committed. It names the snapshot,
the held-out stations, every split's manifest id and every task's fingerprint and file hashes.
Rebuilding the data must reproduce it exactly, or the benchmark refuses to run.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from craton.bench.results import RunResults, TaskResult
from craton.bench.splits import (
    SPLITS,
    definition,
    definition_problems,
    grown_from,
    leak_problems,
    materialise,
)
from craton.bench.tasks import (
    TASKS,
    BenchContext,
    GeoPerplexity,
    GpsGapFill,
    InContextAdaptation,
    Task,
    frozen_path,
    load_frozen,
)
from craton.paths import pending_dir

Log = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


def prepare(ctx: BenchContext, freeze: bool = False, log: Log = _quiet) -> list[str]:
    """Cut every split and write every task's test files. Returns problems (empty = all good).

    Without ``freeze`` the result must equal the committed frozen record. With it, the record
    is written, but only if none exists yet: a frozen benchmark is never re-frozen.
    """
    files = []
    for adapter in ctx.snapshot["adapters"].values():
        files += [ctx.tokens_root / f["path"] for f in adapter["files"]]
    files.sort()
    heldout = ctx.heldout
    splits = {}
    for name, spec in SPLITS.items():
        log(f"split {name}")
        m = materialise(spec, files, ctx.tokens_root, ctx.split_dir(name), heldout)
        splits[name] = {"id": m["id"], "tokens": m["tokens"], "left_out": m["left_out"]}
    tasks = {}
    for task_id, task in TASKS.items():
        log(f"task {task_id}")
        info = task.prepare(ctx)
        tasks[task_id] = {
            "name": task.name,
            "metric": task.metric,
            "direction": task.direction,
            "fingerprint": task.fingerprint(info["files"]),
            "files": info["files"],
            "summary": info["summary"],
        }
    body = {
        "region": ctx.region,
        "snapshot": ctx.snapshot_id,
        "definition": definition(),
        "heldout_stations": heldout,
        "splits": splits,
        "tasks": tasks,
    }
    frozen = load_frozen(ctx.region)
    path = frozen_path(ctx.region)
    if frozen is not None and freeze:
        return _add_tasks(frozen, body, path, log)
    if frozen is None:
        if not freeze:
            return [
                f"no frozen record yet; run `craton bench prepare --freeze` to write {path.name}"
            ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")
        log(f"froze {path.name}")
        return []
    return _diff(frozen, body)


def _add_tasks(frozen: dict[str, Any], body: dict[str, Any], path: Path, log: Log) -> list[str]:
    """Freeze tasks and splits the record doesn't have yet. Everything already frozen must
    match exactly: a frozen benchmark grows, it never changes."""
    if not grown_from(frozen["definition"], body["definition"]):
        return ["the split definition changed beyond adding splits; a changed split needs new IDs"]
    new = {t: v for t, v in body["tasks"].items() if t not in frozen["tasks"]}
    new_splits = {s: v for s, v in body["splits"].items() if s not in frozen["splits"]}
    known = {
        **body,
        "definition": frozen["definition"],
        "splits": {s: v for s, v in body["splits"].items() if s in frozen["splits"]},
        "tasks": {t: v for t, v in body["tasks"].items() if t in frozen["tasks"]},
    }
    problems = _diff(frozen, known)
    if problems:
        return problems
    if not new and not new_splits:
        log("nothing new to freeze")
        return []
    out = {
        **frozen,
        "definition": body["definition"],
        "splits": {**frozen["splits"], **new_splits},
        "tasks": {**frozen["tasks"], **new},
    }
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    log(f"froze {', '.join([*new_splits, *new])} into {path.name}")
    return []


def _diff(frozen: dict[str, Any], body: dict[str, Any], prefix: str = "") -> list[str]:
    problems = []
    for key in sorted(set(frozen) | set(body)):
        a, b = frozen.get(key), body.get(key)
        where = f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            problems += _diff(a, b, where + ".")
        elif a != b:
            problems.append(f"{where}: frozen {a!r}, rebuilt {b!r}")
    return problems


def verify(ctx: BenchContext | None, root: Path) -> tuple[list[str], str]:
    """Problems with the frozen benchmark, and a note on how deep the check went."""
    frozen = load_frozen()
    if frozen is None:
        return ["CratonBench is not frozen yet (craton/bench/frozen/ is empty)"], ""
    problems = definition_problems(frozen)
    if ctx is None or not ctx.bench_dir.is_dir():
        return problems, "definition only: no prepared data on this machine"
    for task_id, task in TASKS.items():
        if task.files(ctx) != frozen["tasks"][task_id]["files"]:
            problems.append(f"{task_id}: prepared test files differ from the frozen hashes")
    cb01, cb04, cb17 = TASKS["CB-01"], TASKS["CB-04"], TASKS["CB-17"]
    assert isinstance(cb01, GeoPerplexity)
    assert isinstance(cb04, GpsGapFill)
    assert isinstance(cb17, InContextAdaptation)
    gap_answers = cb04.answer_hashes(ctx)
    forbidden = {
        "pretrain": cb01.hidden(ctx) | gap_answers,
        "cb17_fit": cb01.hidden(ctx) | gap_answers | cb17.answer_hashes(ctx),
    }
    for name in SPLITS:
        if not (ctx.split_dir(name) / "manifest.json").is_file():
            problems.append(f"split {name} is not prepared")
            continue
        manifest = json.loads((ctx.split_dir(name) / "manifest.json").read_text("utf-8"))
        if manifest["id"] != frozen["splits"][name]["id"]:
            problems.append(f"split {name}: manifest {manifest['id']} is not the frozen one")
        problems += leak_problems(
            ctx.split_dir(name), frozen["heldout_stations"], forbidden.get(name, gap_answers)
        )
    return problems, "full: every shard read back"


# ---------------------------------------------------------------- runs


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(
    method: Any,
    ctx: BenchContext,
    seeds: Sequence[int],
    task_ids: Sequence[str] | None = None,
    log: Log = _quiet,
    config_hash: str = "baseline",
) -> RunResults:
    """Score a method on every task it supports (or the ones named), one score per seed."""
    frozen = load_frozen(ctx.region)
    if frozen is None:
        raise RuntimeError("CratonBench is not frozen; run `craton bench prepare --freeze`")
    chosen: list[Task] = [
        t
        for tid, t in TASKS.items()
        if (task_ids is None or tid in task_ids) and t.supports(method)
    ]
    if not chosen:
        raise ValueError(f"{method.name} supports none of the tasks asked for")
    results = []
    for task in chosen:
        fp = task.fingerprint(task.files(ctx))
        if fp != frozen["tasks"][task.task_id]["fingerprint"]:
            raise RuntimeError(f"{task.task_id}: the prepared test data is not the frozen one")
        values: list[float] = []
        extras: dict[str, list[float]] = {}
        for seed in seeds:
            score = task.score(method, ctx, seed)
            log(f"{task.task_id} seed {seed}: {score.value:.4f}")
            values.append(score.value)
            for k, v in score.extras.items():
                extras.setdefault(k, []).append(v)
        clean = {k: tuple(v) for k, v in extras.items() if all(math.isfinite(x) for x in v)}
        results.append(
            TaskResult(
                task_id=task.task_id,
                metric=task.metric,
                direction=task.direction,  # type: ignore[arg-type]
                fingerprint=fp,
                seeds=tuple(seeds),
                values=tuple(values),
                extras=clean,
            )
        )
    stamp = datetime.now(UTC).replace(microsecond=0)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", method.name).strip("-")
    return RunResults(
        run_id=f"{stamp:%Y%m%d-%H%M%S}-{slug}",
        model=method.name,
        kind=getattr(method, "kind", "model"),
        config_hash=config_hash,
        git_commit=git_commit(ctx.root),
        created=stamp,
        results=tuple(results),
    )


def save_pending(root: Path, results: RunResults) -> Path:
    path = pending_dir(root) / f"{results.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
