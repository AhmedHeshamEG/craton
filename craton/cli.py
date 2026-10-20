"""The ``craton`` command.

craton doctor        what this machine can run, and how
craton config NAME   a model size, with its parameter count
craton check         everything that must pass before a change lands
craton ratchet       judge pending benchmark runs (``--accept`` to put them on the board)
craton scoreboard    regenerate SCOREBOARD.md from results/scoreboard.json
craton spec          the token spec: slots, instruments, examples, validation
craton data          adapters and regions: list, fetch, build, cards
craton bench         CratonBench: list, prepare, leaks, run METHOD, cards
craton train         train Craton-mini on a split: a checkpoint and its model card
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from craton import __version__
from craton.bench import ratchet as ratchet_mod
from craton.bench import scoreboard as scoreboard_mod
from craton.bench.results import RunResults, Scoreboard
from craton.config import CratonConfig, available_configs, config_hash, load_config
from craton.hardware import HardwareReport, plan_precision, probe, smoke_test
from craton.paths import (
    data_health_json,
    find_root,
    pending_dir,
    raw_dir,
    runs_dir,
    scoreboard_json,
    snapshots_dir,
    tokens_dir,
)
from craton.records import append_only_problems
from craton.spec import REGISTRY, SPEC_VERSION, Token, TokenError, sequence_problems, validate_token
from craton.spec import docs as spec_docs
from craton.spec.examples import EXAMPLES, example
from craton.spec.policy import spec_problems

ACCENT = "#ff6b35"
console = Console(highlight=False)


def _title(command: str) -> None:
    console.print()
    console.print(
        Text.assemble(("craton", f"bold {ACCENT}"), (f" {__version__} · {command}", "dim"))
    )
    console.print()


def _human(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


def _grid() -> Table:
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2, 0, 0))
    t.add_column(style="dim", no_wrap=True)
    t.add_column()
    return t


# ---------------------------------------------------------------- doctor


def _precision_line(cfg: CratonConfig, report: HardwareReport) -> str:
    plan = plan_precision(cfg.train.precision, report)
    scaler = " + scaler" if plan.grad_scaler else ""
    cpu = " (cpu)" if plan.device == "cpu" else ""
    swap = f"{cfg.train.precision} → " if plan.dtype != cfg.train.precision else ""
    return f"{swap}{plan.dtype}{scaler}{cpu}"


def cmd_doctor(args: argparse.Namespace) -> int:
    _title("doctor")
    report = probe()
    t = _grid()
    ram = f" · {report.ram_gib} GiB RAM" if report.ram_gib else ""
    t.add_row("machine", f"{report.os} · Python {report.python}{ram}")
    if report.torch_version:
        cuda = f" (CUDA {report.torch_cuda})" if report.torch_cuda else " (CPU build)"
        t.add_row("pytorch", f"{report.torch_version}{cuda}")
    else:
        t.add_row("pytorch", Text("not installed", style="yellow"))

    gpu = report.gpu
    if gpu:
        cc = f"sm_{gpu.compute_capability[0]}{gpu.compute_capability[1]}"
        seen = "" if report.cuda_usable else "  [yellow](PyTorch can't use it)[/]"
        t.add_row("gpu", f"{gpu.name} · {gpu.vram_gib:g} GiB · {gpu.architecture} {cc}{seen}")
        fp16 = (
            "[green]✓[/] tensor cores" if gpu.fp16_tensor_cores else "[yellow]~[/] no tensor cores"
        )
        t.add_row("fp16", f"{fp16} · trains with gradient scaling")
        bf16 = (
            "[green]✓[/] native"
            if gpu.bf16
            else f"[{ACCENT}]✗[/] not on {gpu.architecture} · a bf16 request becomes fp16"
        )
        t.add_row("bf16", bf16)
    else:
        t.add_row("gpu", Text("none found · runs fall back to fp32 on the CPU", style="yellow"))
    console.print(t)

    root = _root_or_none()
    if root is not None:
        console.print()
        configs = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
        for col in ("size", "status", "trunk", "d × layers", "context", "precision here"):
            configs.add_column(col)
        sizes = sorted(
            (load_config(n, root) for n in available_configs(root)), key=lambda c: c.params
        )
        for cfg in sizes:
            status = f"[{ACCENT}]active[/]" if cfg.status == "active" else "[dim]planned[/]"
            shape = f"{cfg.model.d_model} × {cfg.model.n_layers}"
            configs.add_row(
                cfg.name,
                status,
                _human(cfg.params),
                shape,
                f"{cfg.model.context_tokens:,}",
                _precision_line(cfg, report),
            )
        console.print(configs)

    for note in report.notes:
        console.print(f"[yellow]![/] {note}")
    if args.smoke:
        console.print()
        if report.torch_version is None:
            console.print("[yellow]smoke test skipped:[/] PyTorch is not installed.")
        else:
            plan = plan_precision("fp16", report)
            with console.status("[dim]training a tiny network…[/]"):
                res = smoke_test(plan)
            vram = f" · peak {res.peak_vram_mib:.1f} MiB" if res.peak_vram_mib is not None else ""
            mark = "[green]✓[/]" if res.learned else "[red]✗[/]"
            console.print(
                f"{mark} smoke: {res.steps} {res.dtype} steps on {res.device} · loss "
                f"{res.first_loss:.3f} → {res.last_loss:.3f}{vram} · {res.seconds:.2f}s"
            )
            if not res.learned:
                console.print(
                    "[red]the loss didn't go down; something is wrong with this setup.[/]"
                )
                return 1
    mini_ok = report.cuda_usable and gpu is not None and gpu.vram_gib >= 5.5
    console.print(
        "[green]ready:[/] Craton-mini trains on this GPU."
        if mini_ok
        else "[yellow]heads-up:[/] no 6 GB CUDA GPU visible; CPU runs work, just slowly."
    )
    console.print()
    return 0


def _root_or_none() -> Path | None:
    try:
        return find_root()
    except RuntimeError:
        return None


# ---------------------------------------------------------------- config


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.name)
    _title(f"config {cfg.name}")
    t = _grid()
    t.add_row("status", cfg.status)
    t.add_row("about", cfg.description)
    t.add_row("hardware", cfg.target_hardware)
    m = cfg.model
    t.add_row(
        "shape", f"d_model {m.d_model} · {m.n_layers} layers · {m.n_heads} heads · ff {m.ff_width}"
    )
    t.add_row("context", f"{m.context_tokens:,} tokens")
    low, high = cfg.params_range
    t.add_row(
        "trunk", f"{cfg.params:,} parameters (claims {_human(low)}–{_human(high)}) ≈ 12·d²·layers"
    )
    tr = cfg.train
    t.add_row(
        "train",
        f"{tr.precision} · batch {tr.micro_batch} × {tr.grad_accum} = {cfg.effective_batch} "
        f"· lr {tr.lr:g} · seeds {list(tr.seeds)}",
    )
    t.add_row("hash", config_hash(cfg))
    console.print(t)
    console.print()
    return 0


# ---------------------------------------------------------------- scoreboard & ratchet


def cmd_scoreboard(_: argparse.Namespace) -> int:
    out = scoreboard_mod.write(find_root())
    console.print(f"wrote {out.name}")
    return 0


def _pending(root: Path) -> list[Path]:
    folder = pending_dir(root)
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _print_report(report: ratchet_mod.Report) -> None:
    colours = {
        "improved": "green",
        "new": ACCENT,
        "same": "dim",
        "regressed": "red",
        "missing": "red",
        "invalid": "red",
    }
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim", title_justify="left")
    for col in ("task", "verdict", "champion", "candidate", "why"):
        t.add_column(col)
    for v in report.verdicts:
        t.add_row(
            v.task_id,
            f"[{colours[v.status]}]{v.status}[/]",
            "·" if v.champion is None else f"{v.champion:.4g}",
            "·" if v.candidate is None else f"{v.candidate:.4g}",
            v.detail,
        )
    console.print(f"run [bold]{report.run_id}[/]: {report.summary}")
    if report.verdicts:
        console.print(t)


def cmd_ratchet(args: argparse.Namespace) -> int:
    root = find_root()
    _title("ratchet")
    files = [Path(p) for p in args.runs] if args.runs else _pending(root)
    board = Scoreboard.load(scoreboard_json(root))
    if not files:
        console.print(
            f"no pending runs · the board holds {len(board.entries)} task(s) · [green]pass[/]"
        )
        return 0

    failed = False
    for path in files:
        run = RunResults.load(path)
        report = ratchet_mod.judge(board, run)
        _print_report(report)
        if not report.passed:
            failed = True
            continue
        if args.accept:
            board = ratchet_mod.accept(board, run, report)
            dest = runs_dir(root) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), dest)
            console.print(f"[green]accepted[/] → {dest.relative_to(root).as_posix()}")
    if args.accept:
        board.dump(scoreboard_json(root))
        scoreboard_mod.write(root)
    console.print("[red]fail[/]" if failed else "[green]pass[/]")
    return 1 if failed else 0


# ---------------------------------------------------------------- spec


def cmd_spec(args: argparse.Namespace) -> int:
    action = args.action or "show"
    if action == "show":
        return _spec_show()
    if action == "example":
        return _spec_example(args.target)
    if action == "validate":
        if not args.target:
            console.print("usage: craton spec validate FILE (.json, .jsonl or .parquet)")
            return 2
        return _spec_validate(Path(args.target))
    out = spec_docs.write(find_root())
    console.print(f"wrote {out.relative_to(find_root()).as_posix()}")
    return 0


def _spec_show() -> int:
    _title(f"spec {SPEC_VERSION}")
    console.print(
        "[dim]kind · instrument · lat, lon, depth · time · value · uncertainty · provenance · "
        "resolution[/]"
    )
    console.print()
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
    for col in ("instrument", "kinds", "channels", "qualifiers"):
        t.add_column(col)
    for inst in REGISTRY:
        chans = escape(", ".join(f"{c.name} [{c.unit}]" for c in inst.channels))
        if len(inst.channels) > 3:
            chans = f"{len(inst.channels)} class probabilities"
        quals = ", ".join(q.name + ("*" if q.required else "") for q in inst.qualifiers)
        t.add_row(inst.name, ", ".join(inst.kinds), chans, quals or "·")
    console.print(t)
    console.print(f"[dim]* required · examples: {', '.join(EXAMPLES)}[/]")
    console.print()
    return 0


def _spec_example(name: str | None) -> int:
    if name is None or name not in EXAMPLES:
        if name is not None:
            console.print(f"[red]no example {name!r}[/]")
        console.print("examples: " + ", ".join(EXAMPLES))
        return 0 if name is None else 1
    print(json.dumps(example(name).to_dict(), indent=2, ensure_ascii=False))
    return 0


def _load_any(path: Path) -> list[object]:
    if path.suffix == ".parquet":
        from craton.store import read_tokens

        return list(read_tokens(path))
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


def _spec_validate(path: Path) -> int:
    _title("spec validate")
    try:
        items = _load_any(path)
    except (OSError, ValueError, TokenError) as exc:
        console.print(f"[red]✗[/] {path.name}: {exc}")
        return 1
    good: list[Token] = []
    bad = 0
    for i, item in enumerate(items):
        if isinstance(item, Token):
            good.append(item)
            continue
        try:
            good.append(validate_token(item))  # type: ignore[arg-type]
        except TokenError as exc:
            bad += 1
            for p in exc.problems:
                console.print(f"[red]✗[/] token {i} · [bold]{p.code}[/] · {p.field}: {p.message}")
    seq = sequence_problems(good)
    for p in seq:
        console.print(f"[yellow]![/] {p.code} · {p.field}: {p.message}")
    ok = bad == 0 and not seq
    mark = "[green]✓[/]" if ok else "[red]✗[/]"
    console.print(f"{mark} {len(good)} valid, {bad} invalid · {path.name}")
    return 0 if ok else 1


# ---------------------------------------------------------------- data


def _mib(n: int) -> str:
    return f"{n / 2**20:,.1f} MiB"


def cmd_data(args: argparse.Namespace) -> int:
    from craton.adapters import ADAPTERS, get_adapter
    from craton.adapters import build as build_mod
    from craton.adapters import cards as cards_mod
    from craton.adapters.http import Http
    from craton.adapters.region import available_regions, load_region
    from craton.store.raw import RawStore

    root = find_root()
    action = args.action
    if action == "cards":
        for path in cards_mod.write(root):
            console.print(f"wrote {path.relative_to(root).as_posix()}")
        return 0
    if action == "list":
        _title("data")
        t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
        for col in ("adapter", "version", "makes", "needs", "licence"):
            t.add_column(col)
        for name in sorted(ADAPTERS):
            a = get_adapter(name)
            card = a.card()
            needs = ", ".join(a.requires) or "·"
            t.add_row(name, a.version, ", ".join(a.instruments), needs, card.license)
        console.print(t)
        for rname in available_regions(root):
            r = load_region(rname, root)
            snap = build_mod.latest_snapshot(snapshots_dir(root), r.name)
            built = f"snapshot [bold]{snap.id}[/]" if snap else "[dim]not built yet[/]"
            console.print(
                f"region [bold]{r.name}[/] · {r.start[:10]} → {r.end[:10]} · "
                f"{', '.join(r.adapters)} · {built}"
            )
        console.print()
        return 0

    region = load_region(args.region or "central_italy", root)
    raw = RawStore(raw_dir(root))
    if action == "fetch":
        _title(f"data fetch {region.name}")
        reports = build_mod.fetch(
            region, raw, Http(), refresh=args.refresh, log=lambda m: console.print(f"[dim]{m}[/]")
        )
        for name, rep in reports.items():
            console.print(
                f"[green]✓[/] {name}: {rep.downloaded} downloaded ({_mib(rep.bytes)}), "
                f"{rep.cached} cached"
            )
            for note in rep.notes:
                console.print(f"  [dim]{note}[/]")
        console.print()
        return 0

    # build
    _title(f"data build {region.name}")
    start = time.perf_counter()
    snap = build_mod.build(
        region,
        raw,
        tokens_dir(root),
        snapshots_dir(root),
        log=lambda m: console.print(f"[dim]{m}[/]"),
    )
    build_mod.write_health(data_health_json(root), snap)
    scoreboard_mod.write(root)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
    for col in ("adapter", "tokens", "files", "raw", "rejected"):
        t.add_column(col)
    for name, a in snap.body["adapters"].items():
        rej = sum(a["rejects"].values())
        t.add_row(
            name,
            f"{a['tokens']:,}",
            str(len(a["files"])),
            f"{a['raw_files']} · {_mib(a['raw_bytes'])}",
            f"{rej:,}" if rej else "·",
        )
    console.print(t)
    took = time.perf_counter() - start
    console.print(
        f"snapshot [bold {ACCENT}]{snap.id}[/] · {snap.path.relative_to(root).as_posix()} "
        f"· {took:.0f}s"
    )
    console.print("[dim]same raw files → same snapshot id. Health → results/data_health.json[/]")
    console.print()
    return 0


# ---------------------------------------------------------------- bench


def cmd_bench(args: argparse.Namespace) -> int:
    from craton.bench import harness
    from craton.bench.baselines import METHODS
    from craton.bench.tasks import TASKS, BenchContext, load_frozen
    from craton.train import method as _craton_mini  # noqa: F401  registers craton-mini

    root = find_root()
    action = args.action
    if action == "cards":
        from craton.bench import cards as bench_cards

        out = bench_cards.write(root)
        console.print(f"wrote {out.relative_to(root).as_posix()}")
        return 0
    if action == "list":
        _title("bench")
        frozen = load_frozen() or {"tasks": {}}
        t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
        for col in ("task", "name", "metric", "better", "fingerprint", "methods"):
            t.add_column(col)
        for tid, task in TASKS.items():
            fp = frozen["tasks"].get(tid, {}).get("fingerprint", "")
            methods = ", ".join(n for n, m in METHODS.items() if task.supports(m))
            better = "↓ lower" if task.direction == "lower" else "↑ higher"
            t.add_row(tid, task.name, task.metric, better, fp[:12] or "·", methods)
        console.print(t)
        console.print()
        return 0

    ctx = BenchContext(root)
    log = lambda m: console.print(f"[dim]{m}[/]")  # noqa: E731
    if action == "prepare":
        _title(f"bench prepare · snapshot {ctx.snapshot_id}")
        problems = harness.prepare(ctx, freeze=args.freeze, log=log)
        for p in problems:
            console.print(f"[red]✗[/] {p}")
        if not problems:
            console.print("[green]✓[/] splits and test files match the frozen record")
        console.print()
        return 1 if problems else 0
    if action == "leaks":
        _title("bench leaks")
        problems, depth = harness.verify(ctx, root)
        for p in problems:
            console.print(f"[red]✗[/] {p}")
        mark = "[red]✗[/]" if problems else "[green]✓[/]"
        console.print(f"{mark} {len(problems)} problem(s) · {depth}")
        console.print()
        return 1 if problems else 0

    # run
    if not args.method or args.method not in METHODS:
        console.print("methods: " + ", ".join(METHODS))
        return 2
    method = METHODS[args.method]
    _title(f"bench run {method.name}")
    seeds = tuple(args.seeds) if args.seeds else load_config("mini", root).train.seeds
    start = time.perf_counter()
    if hasattr(method, "log"):
        method.log = log
    results = harness.run(
        method,
        ctx,
        seeds,
        args.tasks or None,
        log=log,
        config_hash=getattr(method, "config_hash", "baseline"),
    )
    path = harness.save_pending(root, results)
    if hasattr(method, "write_cards"):
        cards = method.write_cards(root)
        console.print(f"[dim]model cards: {cards.relative_to(root).as_posix()}[/]")
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim")
    for col in ("task", "score", "metric"):
        t.add_column(col)
    for r in results.results:
        t.add_row(r.task_id, f"{r.mean:.4g} ± {r.std:.2g}", r.metric)
    console.print(t)
    console.print(
        f"wrote {path.relative_to(root).as_posix()} · {time.perf_counter() - start:.0f}s · "
        "[dim]`craton ratchet --accept` puts it on the board[/]"
    )
    console.print()
    return 0


# ---------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> int:
    from craton.bench.tasks import BenchContext
    from craton.train import checkpoint
    from craton.train.method import CratonMini

    root = find_root()
    ctx = BenchContext(root)
    log = lambda m: console.print(f"[dim]{m}[/]")  # noqa: E731
    mini = CratonMini(config_name=args.config, log=log)
    for seed in args.seeds or mini.config.train.seeds:
        key = mini.key(ctx, args.split, seed)
        _title(f"train {key.name}")
        if checkpoint.load(root, key) is not None and not args.again:
            console.print("[green]✓[/] already trained (--again to retrain)")
            continue
        start = time.perf_counter()
        mini.train(ctx, args.split, seed)
        card = key.folder(root) / "card.md"
        console.print(
            f"[green]✓[/] {time.perf_counter() - start:.0f}s · card: "
            f"{card.relative_to(root).as_posix()}"
        )
    console.print()
    return 0


# ---------------------------------------------------------------- check


@dataclass
class Step:
    name: str
    run: Callable[[], tuple[bool, str]]


def _tool(*argv: str) -> Callable[[], tuple[bool, str]]:
    def run() -> tuple[bool, str]:
        proc = subprocess.run(
            [sys.executable, "-m", *argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        out = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, out

    return run


def cmd_check(args: argparse.Namespace) -> int:
    root = find_root()
    _title("check")

    def scoreboard_step() -> tuple[bool, str]:
        problem = scoreboard_mod.integrity_problem(root)
        return problem is None, problem or ""

    def records_step() -> tuple[bool, str]:
        problems = append_only_problems(root, args.base)
        return not problems, "\n".join(problems)

    def spec_step() -> tuple[bool, str]:
        problems = spec_problems()
        drift = spec_docs.drift_problem(root)
        if drift:
            problems.append(drift)
        return not problems, "\n".join(problems)

    def cards_step() -> tuple[bool, str]:
        from craton.adapters import cards as cards_mod

        problems = cards_mod.drift_problems(root)
        return not problems, "\n".join(problems)

    def bench_step() -> tuple[bool, str]:
        from craton.bench import harness
        from craton.bench.tasks import BenchContext

        try:
            ctx: BenchContext | None = BenchContext(root)
        except FileNotFoundError:
            ctx = None
        if ctx is not None and not ctx.has_data:
            ctx = None
        from craton.bench import cards as bench_cards

        problems, depth = harness.verify(ctx, root)
        drift = bench_cards.drift_problem(root)
        if drift:
            problems.append(drift)
        return not problems, "\n".join(problems) or depth

    def ratchet_step() -> tuple[bool, str]:
        board = Scoreboard.load(scoreboard_json(root))
        msgs, ok = [], True
        for path in _pending(root):
            report = ratchet_mod.judge(board, RunResults.load(path))
            ok &= report.passed
            msgs.append(f"{path.name}: {report.summary}")
        return ok, "\n".join(msgs)

    steps = [
        Step("lint", _tool("ruff", "check", ".")),
        Step("format", _tool("ruff", "format", "--check", ".")),
        Step("types", _tool("mypy")),
        Step("tests", _tool("pytest", "-q", "-p", "no:cacheprovider")),
        Step("scoreboard is generated", scoreboard_step),
        Step(f"records append-only vs {args.base}", records_step),
        Step(f"token spec {SPEC_VERSION} add-only, docs generated", spec_step),
        Step("data cards generated", cards_step),
        Step("benchmarks frozen, no leaks", bench_step),
        Step("ratchet", ratchet_step),
    ]

    failures = 0
    for step in steps:
        start = time.perf_counter()
        with console.status(f"[dim]{step.name}…[/]"):
            ok, out = step.run()
        took = time.perf_counter() - start
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"{mark} {step.name} [dim]{took:.1f}s[/]")
        if not ok:
            failures += 1
            console.print(Text(out[-4000:], style="dim"))
    console.print()
    if failures:
        console.print(f"[red]{failures} check(s) failed.[/] Nothing lands until they pass.")
        return 1
    console.print(f"[bold {ACCENT}]all green.[/] The foundation holds.")
    console.print()
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="craton", description="One model for the solid Earth.")
    p.add_argument("--version", action="version", version=f"craton {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="what this machine can run, and how")
    d.add_argument("--smoke", action="store_true", help="also train a tiny network in fp16")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("config", help="show a model size and its parameter count")
    c.add_argument("name", nargs="?", default="mini")
    c.set_defaults(func=cmd_config)

    k = sub.add_parser("check", help="lint, types, tests, records, scoreboard, ratchet")
    k.add_argument("--base", default="HEAD", help="git ref the append-only records compare with")
    k.set_defaults(func=cmd_check)

    r = sub.add_parser("ratchet", help="judge benchmark runs against the scoreboard")
    r.add_argument("runs", nargs="*", help="run JSON files (default: results/pending/*.json)")
    r.add_argument("--accept", action="store_true", help="put passing runs on the scoreboard")
    r.set_defaults(func=cmd_ratchet)

    sub.add_parser("scoreboard", help="regenerate SCOREBOARD.md").set_defaults(func=cmd_scoreboard)

    s = sub.add_parser("spec", help="the token spec: show, example NAME, validate FILE, docs")
    s.add_argument("action", nargs="?", choices=("show", "example", "validate", "docs"))
    s.add_argument("target", nargs="?", help="example name or file to validate")
    s.set_defaults(func=cmd_spec)

    dd = sub.add_parser("data", help="adapters and regions: list, fetch, build, cards")
    dd.add_argument("action", choices=("list", "fetch", "build", "cards"))
    dd.add_argument("region", nargs="?", help="region name (default: central_italy)")
    dd.add_argument("--refresh", action="store_true", help="fetch again even if cached")
    dd.set_defaults(func=cmd_data)

    b = sub.add_parser("bench", help="CratonBench: list, prepare, leaks, run METHOD, cards")
    b.add_argument("action", choices=("list", "prepare", "leaks", "run", "cards"))
    b.add_argument("method", nargs="?", help="baseline to run (see `craton bench list`)")
    b.add_argument("--freeze", action="store_true", help="freeze the record, or its new tasks")
    b.add_argument("--tasks", nargs="*", help="only these task IDs")
    b.add_argument("--seeds", nargs="*", type=int, help="default: the mini config's seeds")
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("train", help="train Craton-mini on a split (checkpoint + model card)")
    t.add_argument("split", choices=("pretrain", "cb02_fit", "cb03_fit"))
    t.add_argument("--config", default="mini")
    t.add_argument("--seeds", nargs="*", type=int, help="default: the config's seeds")
    t.add_argument("--again", action="store_true", help="retrain even if a checkpoint exists")
    t.set_defaults(func=cmd_train)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
