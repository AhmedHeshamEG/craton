"""From raw downloads to an immutable snapshot of token files.

    craton data fetch central_italy      provider → data/raw/   (network, cached)
    craton data build central_italy      data/raw/ → data/tokens/ + data/snapshots/   (offline)

A **snapshot** is one JSON file naming every token file a build wrote, with its hash, and every
raw file it read. Its own name is the hash of its content, so two builds that agree byte for
byte get the same name: that is the reproducibility test, and ``craton data build`` reports it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from craton.adapters import get_adapter
from craton.adapters.base import Adapter, BuildContext, FetchReport
from craton.adapters.health import Collector, canonical, collector_for
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.spec.token import SPEC_VERSION
from craton.store.parquet import TokenFile, write_tokens
from craton.store.raw import RawStore

Log = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


def build_order(names: tuple[str, ...]) -> list[Adapter]:
    """Adapters so that each comes after the ones it reads. Missing requirements are an error."""
    adapters = {n: get_adapter(n) for n in names}
    done: list[Adapter] = []
    seen: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> None:
        if name in seen:
            return
        if name in path:
            loop = " → ".join((*path, name))
            raise ValueError(f"adapters require each other in a loop: {loop}")
        if name not in adapters:
            raise ValueError(f"{path[-1]} needs adapter {name!r}; add it to the region")
        for req in adapters[name].requires:
            visit(req, (*path, name))
        seen.add(name)
        done.append(adapters[name])

    for n in names:
        visit(n, ())
    return done


def fetch(
    region: Region, raw: RawStore, http: Fetcher, refresh: bool = False, log: Log = _quiet
) -> dict[str, FetchReport]:
    return {a.name: a.fetch(region, raw, http, refresh, log) for a in build_order(region.adapters)}


@dataclass
class AdapterBuild:
    name: str
    version: str
    files: list[TokenFile] = field(default_factory=list)
    rejects: Counter[str] = field(default_factory=Counter)
    raw: dict[str, Any] = field(default_factory=dict)
    health: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return sum(f.tokens for f in self.files)

    def as_dict(self, root: Path) -> dict[str, Any]:
        return {
            "version": self.version,
            "tokens": self.tokens,
            "rejects": dict(sorted(self.rejects.items())),
            "raw_files": len(self.raw),
            "raw_bytes": sum(r["size"] for r in self.raw.values()),
            "raw": [self.raw[k] for k in sorted(self.raw)],
            "files": [
                {
                    "path": f.path.relative_to(root).as_posix(),
                    "sha256": f.sha256,
                    "tokens": f.tokens,
                    "instrument": f.instrument,
                    "year": f.year,
                }
                for f in sorted(self.files, key=lambda f: f.path.as_posix())
            ],
            "health": self.health,
        }


@dataclass(frozen=True)
class Snapshot:
    id: str
    path: Path
    body: dict[str, Any]


def build(
    region: Region, raw: RawStore, tokens_root: Path, snapshots_root: Path, log: Log = _quiet
) -> Snapshot:
    """Build every adapter of the region, offline, from the raw store. Returns the snapshot."""
    results: dict[str, AdapterBuild] = {}
    for adapter in build_order(region.adapters):
        inputs = {req: [f.path for f in results[req].files] for req in adapter.requires}
        ctx = BuildContext(region=region, raw=raw, inputs=inputs)
        out = AdapterBuild(adapter.name, adapter.version)
        collectors: dict[str, Collector] = {}
        for batch in adapter.tokens(ctx):
            if not batch:
                continue
            files = write_tokens(batch, tokens_root)
            out.files += files
            for tok_instrument in {t.instrument for t in batch}:
                col = collectors.setdefault(tok_instrument, collector_for(tok_instrument))
                col.observe([t for t in batch if t.instrument == tok_instrument])
            log(f"{adapter.name}  {files[0].year}  {sum(f.tokens for f in files):,} tokens")
        out.rejects = ctx.rejects
        out.raw = {
            k: {"label": r.label, "sha256": r.sha256, "size": r.size, "retrieved": r.retrieved}
            for k, r in ctx.raw_used.items()
        }
        out.health = {k: c.result() for k, c in sorted(collectors.items())}
        results[adapter.name] = out

    body = {
        "region": region.name,
        "region_fingerprint": region.fingerprint(),
        "box": region.box.model_dump(),
        "window": [region.start, region.end],
        "spec": SPEC_VERSION,
        "adapters": {n: results[n].as_dict(tokens_root) for n in sorted(results)},
    }
    snap_id = snapshot_id(body)
    folder = snapshots_root / region.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{snap_id}.json"
    if not path.exists():
        path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (folder / "LATEST").write_text(snap_id + "\n", encoding="utf-8")
    return Snapshot(snap_id, path, body)


def snapshot_id(body: dict[str, Any]) -> str:
    """The hash of what a build *is*: region, files and raw inputs. Health numbers are left out,
    so a better statistic never renames unchanged data."""
    data = {
        **body,
        "adapters": {
            n: {k: v for k, v in a.items() if k != "health"} for n, a in body["adapters"].items()
        },
    }
    return hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()[:16]


def latest_snapshot(snapshots_root: Path, region: str) -> Snapshot | None:
    folder = snapshots_root / region
    pointer = folder / "LATEST"
    if not pointer.is_file():
        return None
    snap_id = pointer.read_text(encoding="utf-8").strip()
    path = folder / f"{snap_id}.json"
    return Snapshot(snap_id, path, json.loads(path.read_text(encoding="utf-8")))


def health_entry(snapshot: Snapshot) -> dict[str, Any]:
    """What goes into ``results/data_health.json`` for one region."""
    b = snapshot.body
    return {
        "snapshot": snapshot.id,
        "window": b["window"],
        "box": b["box"],
        "adapters": {
            name: {
                "version": a["version"],
                "tokens": a["tokens"],
                "files": len(a["files"]),
                "raw_files": a["raw_files"],
                "raw_bytes": a["raw_bytes"],
                "rejects": a["rejects"],
                "health": a["health"],
            }
            for name, a in b["adapters"].items()
        },
    }


def write_health(path: Path, snapshot: Snapshot) -> None:
    """Put one region's entry into the health report, keeping the other regions."""
    report: dict[str, Any] = {"schema_version": 1, "regions": {}}
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
    report["regions"][snapshot.body["region"]] = health_entry(snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
