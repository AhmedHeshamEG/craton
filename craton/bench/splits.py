"""Frozen splits: what a model may learn from, and what it is tested on.

Three things are held out of Craton's training data, forever:

* **Era A**, the Amatrice-Visso-Norcia-Campotosto sequence: 2016-08-24 → 2018-01-01.
* **Era B**, L'Aquila: the whole of 2009.
* **Held-out GNSS stations**: one station in ten, chosen once by a hash, never trained on.

A **split** is a set of training shards cut from a data snapshot: token files with every
held-out token removed. Each split is written to disk, every file named by its hash, with a
manifest. The **leak check** reads those shards back and refuses the run if a single held-out
token got in.

    pretrain   everything except eras A and B and the held-out stations      (CB-01, CB-04, P4+)
    cb02_fit   everything before 2016-08-24, except the held-out stations    (CB-02 fits)
    cb03_fit   everything before 2009-01-01, except the held-out stations    (CB-03 fits)

The forecast tasks fit only on the past of their test window (pseudo-prospective). Their
forecasts may still *read* quakes inside the window as they happen, day by day: that is the
input of a forecast, not training data.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from craton.bench.data import Gnss
from craton.spec.timeaxis import to_seconds
from craton.store.parquet import _WRITE_OPTIONS, HASH_CHARS, fingerprint_problem

SPLIT_VERSION = "1"


@dataclass(frozen=True)
class Window:
    name: str
    start: str  # ISO 8601, UTC
    end: str

    @property
    def t0(self) -> float:
        return to_seconds(self.start)

    @property
    def t1(self) -> float:
        return to_seconds(self.end)

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "start": self.start, "end": self.end}


ERA_A = Window("era_a", "2016-08-24T00:00:00Z", "2018-01-01T00:00:00Z")
ERA_B = Window("era_b", "2009-01-01T00:00:00Z", "2010-01-01T00:00:00Z")
ERAS = (ERA_A, ERA_B)

#: One station in ten is held out.
HELDOUT_SHARE = 0.10
#: A station can be held out only if it has this many days and enough neighbours to rebuild it.
HELDOUT_MIN_DAYS = 730
HELDOUT_NEIGHBOURS = 3
HELDOUT_NEIGHBOUR_KM = 50.0
HELDOUT_SALT = "craton-heldout-stations-v1"
#: A station that jumps more than this (m) between two consecutive solutions is not one
#: monument: NGL reuses a few codes for sites metres to hundreds of km apart. The largest real
#: jump in the region is ARQT's 0.447 m in the Norcia earthquake; the next is 3.8 m (TOD3).
MAX_JUMP_M = 1.0


@dataclass(frozen=True)
class SplitSpec:
    name: str
    purpose: str
    excluded: tuple[Window, ...]  # tokens inside these windows are left out
    before: str | None = None  # if set, only tokens strictly before this moment

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "excluded": [w.as_dict() for w in self.excluded],
            "before": self.before,
            "heldout_stations_excluded": True,
        }


SPLITS: dict[str, SplitSpec] = {
    s.name: s
    for s in (
        SplitSpec("pretrain", "general training; CB-01 and CB-04 baselines", ERAS),
        SplitSpec("cb02_fit", "fitting for CB-02 (pseudo-prospective)", (), ERA_A.start),
        SplitSpec("cb03_fit", "fitting for CB-03 (pseudo-prospective)", (), ERA_B.start),
    )
}


def definition() -> dict[str, Any]:
    """Everything that defines the splits, as data. Frozen with the hashes."""
    return {
        "version": SPLIT_VERSION,
        "eras": [w.as_dict() for w in ERAS],
        "heldout": {
            "share": HELDOUT_SHARE,
            "min_days": HELDOUT_MIN_DAYS,
            "neighbours": HELDOUT_NEIGHBOURS,
            "neighbour_km": HELDOUT_NEIGHBOUR_KM,
            "salt": HELDOUT_SALT,
            "max_jump_m": MAX_JUMP_M,
        },
        "splits": {name: s.as_dict() for name, s in SPLITS.items()},
    }


# ---------------------------------------------------------------- held-out stations


def _km(lat1: float, lon1: float, lat2: Any, lon2: Any) -> Any:
    """Great-circle distance (haversine, km)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def station_sites(gnss: Gnss) -> dict[str, tuple[float, float, int]]:
    """Median (lat, lon) and day count of every station."""
    out = {}
    for code in gnss.stations:
        sel = gnss.station == code
        out[code] = (
            float(np.median(gnss.lat[sel])),
            float(np.median(gnss.lon[sel])),
            int(sel.sum()),
        )
    return out


def stable_stations(gnss: Gnss) -> set[str]:
    """Stations that never jump more than ``MAX_JUMP_M`` between consecutive solutions."""
    out = set()
    for code in gnss.stations:
        enu = gnss.enu[gnss.station == code]
        if len(enu) < 2 or float(np.abs(np.diff(enu, axis=0)).max()) <= MAX_JUMP_M:
            out.add(code)
    return out


def choose_heldout(sites: dict[str, tuple[float, float, int]], stable: set[str]) -> list[str]:
    """One station in ten, by a salted hash, among stations that can be rebuilt.

    Eligible: one monument (``MAX_JUMP_M``), at least ``HELDOUT_MIN_DAYS`` days, and at
    least ``HELDOUT_NEIGHBOURS`` other eligible stations within ``HELDOUT_NEIGHBOUR_KM`` km that
    are *not* held out. Stations are
    taken in hash order, and a pick that would leave an earlier pick with too few neighbours is
    skipped, so every held-out station keeps enough visible neighbours.
    """
    eligible = sorted(c for c, (_, _, n) in sites.items() if n >= HELDOUT_MIN_DAYS and c in stable)
    target = round(HELDOUT_SHARE * len(sites))
    order = sorted(
        eligible, key=lambda c: hashlib.sha256(f"{HELDOUT_SALT}:{c}".encode()).hexdigest()
    )
    lat = np.array([sites[c][0] for c in eligible])
    lon = np.array([sites[c][1] for c in eligible])
    near = {
        c: {
            eligible[j]
            for j in np.nonzero(_km(sites[c][0], sites[c][1], lat, lon) <= HELDOUT_NEIGHBOUR_KM)[0]
            if eligible[j] != c
        }
        for c in eligible
    }
    chosen: list[str] = []
    for c in order:
        if len(chosen) == target:
            break
        trial = {*chosen, c}
        if all(len(near[s] - trial) >= HELDOUT_NEIGHBOURS for s in trial):
            chosen.append(c)
    return sorted(chosen)


# ---------------------------------------------------------------- shards


def _station_column(table: pa.Table) -> pa.Array:
    """The station code of each row ("" for tokens without one), from the qualifiers JSON."""
    quals = table.column("qualifiers").to_pylist()
    return pa.array([json.loads(q).get("station", "") for q in quals], type=pa.string())


def keep_mask(table: pa.Table, spec: SplitSpec, heldout: Iterable[str]) -> pa.Array:
    """True for the rows of a token table that belong in the split."""
    t = table.column("time")
    keep = pc.invert(pc.is_in(_station_column(table), pa.array(sorted(heldout))))
    for w in spec.excluded:
        inside = pc.and_(pc.greater_equal(t, w.t0), pc.less(t, w.t1))
        keep = pc.and_(keep, pc.invert(inside))
    if spec.before is not None:
        keep = pc.and_(keep, pc.less(t, to_seconds(spec.before)))
    return keep


def _encode(table: pa.Table) -> bytes:
    meta = dict(table.schema.metadata or {})
    meta[b"craton.tokens"] = str(table.num_rows).encode()
    table = table.replace_schema_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(table, buf, **_WRITE_OPTIONS)
    return buf.getvalue()


@dataclass(frozen=True)
class Shard:
    path: str  # relative to the split folder
    sha256: str
    tokens: int


def materialise(
    spec: SplitSpec, files: list[Path], tokens_root: Path, out: Path, heldout: list[str]
) -> dict[str, Any]:
    """Write one split's shards under ``out`` and return its manifest (also written)."""
    shards: list[Shard] = []
    left_out = 0
    for path in files:
        problem = fingerprint_problem(path)
        if problem:
            raise ValueError(problem)
        table = pq.read_table(path)
        kept = table.filter(keep_mask(table, spec, heldout))
        left_out += table.num_rows - kept.num_rows
        if kept.num_rows == 0:
            continue
        data = _encode(kept)
        digest = hashlib.sha256(data).hexdigest()
        rel = path.parent.relative_to(tokens_root) / f"{digest[:HASH_CHARS]}.parquet"
        dest = out / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
        shards.append(Shard(rel.as_posix(), digest, kept.num_rows))
    body = {
        "split": spec.as_dict(),
        "heldout_stations": heldout,
        "shards": [s.__dict__ for s in shards],
        "tokens": sum(s.tokens for s in shards),
        "left_out": left_out,
    }
    body["id"] = manifest_id(body)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return body


def manifest_id(body: dict[str, Any]) -> str:
    content = {k: body[k] for k in ("split", "heldout_stations", "shards")}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_CHARS]


def shard_paths(split_dir: Path) -> list[Path]:
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    return [split_dir / s["path"] for s in manifest["shards"]]


# ---------------------------------------------------------------- leak check


def leak_problems(
    split_dir: Path, heldout: list[str], forbidden_hashes: set[str] | None = None
) -> list[str]:
    """Read a split's shards back and list every way held-out data got in.

    Checks each shard's fingerprint, that no token sits inside an excluded window or after the
    cut-off, that no token comes from a held-out station, and that no token is one of
    ``forbidden_hashes`` (benchmark answers).
    """
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = SplitSpec(
        manifest["split"]["name"],
        manifest["split"]["purpose"],
        tuple(Window(**w) for w in manifest["split"]["excluded"]),
        manifest["split"]["before"],
    )
    problems: list[str] = []
    if manifest_id(manifest) != manifest["id"]:
        problems.append(f"{spec.name}: manifest id does not match its content")
    if sorted(manifest["heldout_stations"]) != sorted(heldout):
        problems.append(f"{spec.name}: built with a different held-out station list")
    forbidden = forbidden_hashes or set()
    for shard in manifest["shards"]:
        path = split_dir / shard["path"]
        if not path.is_file():
            problems.append(f"{spec.name}: missing shard {shard['path']}")
            continue
        fp = fingerprint_problem(path)
        if fp:
            problems.append(f"{spec.name}: {fp}")
            continue
        table = pq.read_table(path, columns=["time", "qualifiers", "token_hash"])
        bad = pc.sum(pc.invert(keep_mask(table, spec, heldout))).as_py() or 0
        if bad:
            problems.append(f"{spec.name}: {bad} held-out token(s) in {shard['path']}")
        if forbidden:
            hits = sum(h in forbidden for h in table.column("token_hash").to_pylist())
            if hits:
                problems.append(f"{spec.name}: {hits} benchmark answer(s) in {shard['path']}")
    return problems


def definition_problems(frozen: dict[str, Any]) -> list[str]:
    """Checks that need no data: the frozen definition still matches the code, and is sane."""
    problems = []
    if frozen.get("definition") != definition():
        problems.append(
            "the split definition in the code differs from the frozen one; a changed split "
            "needs new task IDs"
        )
    stations = frozen.get("heldout_stations", [])
    if len(stations) != len(set(stations)):
        problems.append("a held-out station is listed twice")
    for a in ERAS:
        for b in ERAS:
            if a.name < b.name and a.t0 < b.t1 and b.t0 < a.t1:
                problems.append(f"eras {a.name} and {b.name} overlap")
    return problems
