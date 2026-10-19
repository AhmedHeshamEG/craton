"""The ``craton`` command.

craton doctor        what this machine can run, and how
craton config NAME   a model size, with its parameter count
craton check         everything that must pass before a change lands
craton ratchet       judge pending benchmark runs (``--accept`` to put them on the board)
craton scoreboard    regenerate SCOREBOARD.md from results/scoreboard.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from craton import __version__
from craton.bench import ratchet as ratchet_mod
from craton.bench import scoreboard as scoreboard_mod
from craton.bench.results import RunResults, Scoreboard
from craton.config import CratonConfig, available_configs, config_hash, load_config
from craton.hardware import HardwareReport, plan_precision, probe, smoke_test
from craton.paths import find_root, pending_dir, runs_dir, scoreboard_json
from craton.records import append_only_problems

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
