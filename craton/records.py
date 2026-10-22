"""Append-only records: the logbook, the decisions and the ledger of locked forecasts.

History is never rewritten. A wrong decision gets a new decision that supersedes it; a wrong
logbook entry gets a correction below it. The check compares each record with the version at
a git reference (``HEAD`` locally, the base branch on CI) and fails if any committed line
changed or disappeared.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

APPEND_ONLY = ("LOGBOOK.md", "DECISIONS.md", "forecasts/central_italy/ledger.jsonl")


def _git_show(root: Path, ref: str, rel: str) -> str | None:
    """File content at ``ref``, or None if the ref or the file doesn't exist there."""
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel}"],
            cwd=root,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8")


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n")


def append_only_problems(root: Path, ref: str = "HEAD") -> list[str]:
    """One message per record whose committed content was edited instead of appended to."""
    problems = []
    for rel in APPEND_ONLY:
        path = root / rel
        before = _git_show(root, ref, rel)
        if before is None:
            continue
        if not path.exists():
            problems.append(f"{rel} was deleted. It is append-only.")
            continue
        # Line by line: "D-001" edited into "D-001 (typo)" is still an edit.
        old = _normalise(before).rstrip("\n").split("\n")
        new = _normalise(path.read_text(encoding="utf-8")).split("\n")
        if new[: len(old)] != old:
            line = _first_changed_line(old, new)
            problems.append(
                f"{rel}: line {line} was changed or removed, compared with {ref}. "
                "It is append-only: add a new entry instead of editing an old one."
            )
    return problems


def _first_changed_line(old: list[str], new: list[str]) -> int:
    for i, (a, b) in enumerate(zip(old, new, strict=False), start=1):
        if a != b:
            return i
    return min(len(old), len(new)) + 1


def locked_problems(root: Path, ref: str = "HEAD") -> list[str]:
    """One message per locked forecast or grade that was edited or deleted since ``ref``, and
    per seal that doesn't match its file. Adding new ones is the only change allowed."""
    from craton import live

    problems = live.verify(root)
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-status", "--no-renames", ref, "--", "forecasts"],
            cwd=root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return problems
    if proc.returncode != 0:
        return problems
    for line in proc.stdout.decode("utf-8").splitlines():
        status, _, rel = line.partition("\t")
        if rel.endswith(".json") and status in ("M", "D"):
            what = "edited" if status == "M" else "deleted"
            problems.append(f"{rel} was {what}. A locked forecast and its grade never change.")
    return problems
