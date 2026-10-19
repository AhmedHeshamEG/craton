"""Token files: Parquet, partitioned by instrument and year, named by their own hash.

    data/tokens/instrument=gnss_daily/year=2016/3f9a1c0b7e2d4a65.parquet

The file name is the first 16 hex characters of the file's sha256. That gives three promises
for free:

* **Immutable.** A file can't be edited in place: different bytes would have a different name.
  Corrections arrive as new files.
* **Fingerprinted.** Anyone can recompute a file's hash and check it against its name.
* **Reproducible.** The same tokens always produce the same bytes (sorted, fixed writer
  settings), so re-running an adapter gives the same names. A test holds this.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from craton.spec.registry import Registry
from craton.spec.sequence import sequence_problems
from craton.spec.timeaxis import year_of
from craton.spec.token import SPEC_VERSION, Token, TokenError, validate_token

HASH_CHARS = 16

SCHEMA = pa.schema(
    [
        ("spec", pa.string()),
        ("kind", pa.string()),
        ("instrument", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("depth_km", pa.float64()),
        ("time", pa.float64()),
        ("time_iso", pa.string()),
        ("value", pa.list_(pa.float64())),
        ("uncertainty", pa.list_(pa.float64())),
        ("uncertainty_basis", pa.string()),
        ("loc_horizontal_km", pa.float64()),
        ("loc_vertical_km", pa.float64()),
        ("loc_time_s", pa.float64()),
        ("qualifiers", pa.string()),  # canonical JSON object
        ("source", pa.string()),
        ("dataset_version", pa.string()),
        ("license", pa.string()),
        ("adapter_version", pa.string()),
        ("content_hash", pa.string()),
        ("record", pa.string()),
        ("synthetic", pa.bool_()),
        ("spatial_km", pa.float64()),
        ("temporal_s", pa.float64()),
        ("token_hash", pa.string()),
    ]
)

_WRITE_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 9,
    "version": "2.6",
    "row_group_size": 65_536,
    "use_dictionary": ["spec", "kind", "instrument", "uncertainty_basis", "source", "license"],
    "write_statistics": True,
}


@dataclass(frozen=True)
class TokenFile:
    """One written token file and its fingerprint."""

    path: Path
    sha256: str
    tokens: int
    instrument: str
    year: int


def _row(t: Token) -> dict[str, Any]:
    lu = t.location_uncertainty
    p = t.provenance
    return {
        "spec": t.spec,
        "kind": t.kind,
        "instrument": t.instrument,
        "lat": t.lat,
        "lon": t.lon,
        "depth_km": t.depth_km,
        "time": t.time,
        "time_iso": t.time_iso,
        "value": list(t.value),
        "uncertainty": list(t.uncertainty),
        "uncertainty_basis": t.uncertainty_basis,
        "loc_horizontal_km": lu.horizontal_km if lu else None,
        "loc_vertical_km": lu.vertical_km if lu else None,
        "loc_time_s": lu.time_s if lu else None,
        "qualifiers": json.dumps(t.qualifiers, sort_keys=True, separators=(",", ":")),
        "source": p.source,
        "dataset_version": p.dataset_version,
        "license": p.license,
        "adapter_version": p.adapter_version,
        "content_hash": p.content_hash,
        "record": p.record,
        "synthetic": p.synthetic,
        "spatial_km": t.resolution.spatial_km,
        "temporal_s": t.resolution.temporal_s,
        "token_hash": t.token_hash(),
    }


def _token_dict(r: dict[str, Any]) -> dict[str, Any]:
    loc = {
        k: r[c]
        for k, c in (
            ("horizontal_km", "loc_horizontal_km"),
            ("vertical_km", "loc_vertical_km"),
            ("time_s", "loc_time_s"),
        )
        if r[c] is not None
    }
    d: dict[str, Any] = {
        "spec": r["spec"],
        "kind": r["kind"],
        "instrument": r["instrument"],
        "lat": r["lat"],
        "lon": r["lon"],
        "depth_km": r["depth_km"],
        "time": r["time"],
        "value": r["value"],
        "uncertainty": r["uncertainty"],
        "uncertainty_basis": r["uncertainty_basis"],
        "qualifiers": json.loads(r["qualifiers"]),
        "provenance": {
            "source": r["source"],
            "dataset_version": r["dataset_version"],
            "license": r["license"],
            "adapter_version": r["adapter_version"],
            "content_hash": r["content_hash"],
            "synthetic": r["synthetic"],
            **({"record": r["record"]} if r["record"] is not None else {}),
        },
        "resolution": {"spatial_km": r["spatial_km"], "temporal_s": r["temporal_s"]},
    }
    if r["time_iso"] is not None:
        d["time_iso"] = r["time_iso"]
    if loc:
        d["location_uncertainty"] = loc
    return d


def encode(tokens: Sequence[Token]) -> bytes:
    """The exact bytes of a token file for these tokens (sorted by time, then hash)."""
    ordered = sorted(tokens, key=lambda t: (t.time, t.token_hash()))
    problems = sequence_problems(ordered)
    if problems:
        raise TokenError(problems)
    table = pa.Table.from_pylist([_row(t) for t in ordered], schema=SCHEMA)
    table = table.replace_schema_metadata(
        {"craton.spec": SPEC_VERSION, "craton.tokens": str(len(ordered))}
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, **_WRITE_OPTIONS)
    return buf.getvalue()


def write_tokens(tokens: Iterable[Token], root: Path) -> list[TokenFile]:
    """Write tokens under ``root``, one file per (instrument, year). Never overwrites.

    A file whose name already exists has the same hash and therefore the same bytes, so writing
    it again is a no-op. Returns every file the tokens live in, sorted by path.
    """
    groups: dict[tuple[str, int], list[Token]] = defaultdict(list)
    for t in tokens:
        groups[(t.instrument, year_of(t.time))].append(t)

    written = []
    for (instrument, year), group in sorted(groups.items()):
        data = encode(group)
        digest = hashlib.sha256(data).hexdigest()
        folder = root / f"instrument={instrument}" / f"year={year}"
        path = folder / f"{digest[:HASH_CHARS]}.parquet"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        written.append(TokenFile(path, digest, len(group), instrument, year))
    return written


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_problem(path: Path) -> str | None:
    """None if the file's bytes still match the hash in its name."""
    digest = file_sha256(path)
    if not digest.startswith(path.stem) or len(path.stem) != HASH_CHARS:
        return f"{path.name}: content hash {digest[:HASH_CHARS]} does not match its name"
    return None


def read_tokens(path: Path, registry: Registry | None = None) -> list[Token]:
    """Read a token file back, checking its fingerprint and re-validating every token."""
    problem = fingerprint_problem(path)
    if problem:
        raise ValueError(problem)
    table = pq.read_table(path, schema=SCHEMA)
    tokens = []
    for r in table.to_pylist():
        tok = validate_token(_token_dict(r), registry)
        if tok.token_hash() != r["token_hash"]:
            raise ValueError(f"{path.name}: a token no longer matches its stored hash")
        tokens.append(tok)
    return tokens
