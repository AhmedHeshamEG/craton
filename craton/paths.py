"""Where things live in the repo."""

from __future__ import annotations

from pathlib import Path

_MARKER = "pyproject.toml"


class RepoNotFoundError(RuntimeError):
    """Raised when a command that needs the repo runs outside it."""


def find_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: the working directory) to the Craton repo root.

    The root is the first folder holding both ``pyproject.toml`` and ``configs/``.
    """
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / _MARKER).is_file() and (folder / "configs").is_dir():
            return folder
    raise RepoNotFoundError(
        f"No Craton repo found at or above {here}. Run this from inside the cloned repo."
    )


def configs_dir(root: Path) -> Path:
    return root / "configs"


def results_dir(root: Path) -> Path:
    return root / "results"


def scoreboard_json(root: Path) -> Path:
    return results_dir(root) / "scoreboard.json"


def scoreboard_md(root: Path) -> Path:
    return root / "SCOREBOARD.md"


def pending_dir(root: Path) -> Path:
    """Benchmark runs waiting for the ratchet's verdict."""
    return results_dir(root) / "pending"


def runs_dir(root: Path) -> Path:
    """Benchmark runs the ratchet has accepted, kept forever."""
    return results_dir(root) / "runs"
