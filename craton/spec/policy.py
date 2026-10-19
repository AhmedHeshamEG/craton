"""The add-only promise, enforced.

Each spec release freezes a snapshot (``craton/spec/frozen/<version>.json``): the token's field
schema and every registered instrument. ``craton check`` compares the live code with every
snapshot ever taken and fails if anything frozen was removed or redefined. Adding is always
allowed: a new instrument, a new word in a closed vocabulary, a new *optional* token field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from craton.spec.instruments import REGISTRY
from craton.spec.registry import Registry, frozen_problems
from craton.spec.token import SPEC_VERSION, Token

FROZEN_DIR = Path(__file__).parent / "frozen"


def snapshot(registry: Registry | None = None) -> dict[str, Any]:
    """What a release freezes: the token schema and the instruments."""
    schema = Token.model_json_schema()
    return {
        "spec": SPEC_VERSION,
        "token": {
            "properties": schema["properties"],
            "required": sorted(schema.get("required", [])),
            "defs": schema.get("$defs", {}),
        },
        "instruments": (registry or REGISTRY).snapshot(),
    }


def freeze(path: Path | None = None) -> Path:
    """Write the snapshot for the current spec version. Refuses to overwrite one."""
    out = path or FROZEN_DIR / f"{SPEC_VERSION}.json"
    if out.exists():
        raise FileExistsError(f"{out.name} is frozen; a new release needs a new version number")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _token_problems(frozen: dict[str, Any], current: dict[str, Any], version: str) -> list[str]:
    problems = []
    for name, schema in frozen["properties"].items():
        if name not in current["properties"]:
            problems.append(f"token field {name!r} was removed (frozen in spec {version})")
        elif current["properties"][name] != schema:
            problems.append(f"token field {name!r} was redefined after spec {version}")
    for name, schema in frozen["defs"].items():
        if current["defs"].get(name) != schema:
            problems.append(f"token part {name!r} was redefined after spec {version}")
    if set(current["required"]) != set(frozen["required"]):
        problems.append(
            f"required token fields changed after spec {version}: new fields must be optional"
        )
    return problems


def spec_problems(frozen_dir: Path | None = None, registry: Registry | None = None) -> list[str]:
    """Every way the live spec breaks a frozen snapshot. Empty means add-only holds."""
    folder = frozen_dir or FROZEN_DIR
    snapshots = sorted(folder.glob("*.json"))
    if not snapshots:
        return [f"no frozen spec snapshot in {folder}"]
    current = snapshot(registry)
    problems = []
    for path in snapshots:
        frozen = json.loads(path.read_text(encoding="utf-8"))
        version = frozen["spec"]
        if int(version.split(".")[0]) != int(SPEC_VERSION.split(".")[0]):
            continue  # a major version brings its own migration; see SPEC_POLICY.md
        problems += _token_problems(frozen["token"], current["token"], version)
        problems += frozen_problems(registry or REGISTRY, path)
    return problems
