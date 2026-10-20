"""Token files as NumPy columns, for benchmarks and baselines.

Benchmarks read hundreds of thousands of tokens at a time and only need a few columns, so they
skip pydantic and read Arrow straight into arrays. Every file is still fingerprint-checked
against the hash in its name before a byte of it is used.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from craton.store.parquet import fingerprint_problem

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.int64]]
StrArray = np.ndarray[Any, np.dtype[np.str_]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

DAY = 86_400.0


def read_table(paths: Iterable[Path], columns: Sequence[str]) -> pa.Table:
    """Concatenate some columns of several token files, each one fingerprint-checked."""
    tables = []
    for path in paths:
        problem = fingerprint_problem(path)
        if problem:
            raise ValueError(problem)
        tables.append(pq.read_table(path, columns=list(columns)))
    if not tables:
        raise ValueError("no token files to read")
    return pa.concat_tables(tables)


def instrument_files(root: Path, instrument: str) -> list[Path]:
    """Every token file of one instrument under a token root, in a fixed order."""
    return sorted((root / f"instrument={instrument}").glob("year=*/*.parquet"))


def _floats(table: pa.Table, name: str) -> FloatArray:
    return np.asarray(table.column(name).to_numpy(), dtype=np.float64)


def _strings(table: pa.Table, name: str) -> StrArray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.str_)


def _first(table: pa.Table, name: str, width: int) -> FloatArray:
    flat = np.asarray(pa.compute.list_flatten(table.column(name)).to_numpy(), dtype=np.float64)
    return flat.reshape(-1, width)


@dataclass(frozen=True)
class Events:
    """Earthquakes from ``seismic_catalog`` tokens, sorted by time."""

    time: FloatArray  # seconds since 1970 (UTC)
    lat: FloatArray
    lon: FloatArray
    depth_km: FloatArray
    mag: FloatArray
    mag_type: StrArray
    token_hash: StrArray

    def __len__(self) -> int:
        return len(self.time)

    def select(self, mask: BoolArray | IntArray) -> Events:
        return Events(
            self.time[mask],
            self.lat[mask],
            self.lon[mask],
            self.depth_km[mask],
            self.mag[mask],
            self.mag_type[mask],
            self.token_hash[mask],
        )

    def window(self, start: float, end: float) -> Events:
        return self.select((self.time >= start) & (self.time < end))

    def above(self, magnitude: float) -> Events:
        # Magnitudes are reported to 0.1; the epsilon keeps 3.0 >= 3.0 exact in binary.
        return self.select(self.mag >= magnitude - 1e-9)

    @staticmethod
    def concat(parts: Sequence[Events]) -> Events:
        def cat(name: str) -> Any:
            return np.concatenate([getattr(p, name) for p in parts])

        out = Events(*(cat(n) for n in Events.__dataclass_fields__))
        return out.select(np.argsort(out.time, kind="stable"))


EVENT_COLUMNS = ("time", "lat", "lon", "depth_km", "value", "qualifiers", "token_hash")


def events_from_table(table: pa.Table) -> Events:
    quals = [json.loads(q) for q in table.column("qualifiers").to_pylist()]
    ev = Events(
        time=_floats(table, "time"),
        lat=_floats(table, "lat"),
        lon=_floats(table, "lon"),
        depth_km=_floats(table, "depth_km"),
        mag=_first(table, "value", 1)[:, 0],
        mag_type=np.asarray([q["magnitude_type"] for q in quals], dtype=np.str_),
        token_hash=_strings(table, "token_hash"),
    )
    return ev.select(np.argsort(ev.time, kind="stable"))


def load_events(root: Path) -> Events:
    """All catalogue events under a token root (a snapshot's token folder or a split shard)."""
    files = instrument_files(root, "seismic_catalog")
    return events_from_table(read_table(files, EVENT_COLUMNS))


@dataclass(frozen=True)
class Gnss:
    """Daily GNSS positions from ``gnss_daily`` tokens, sorted by station then time."""

    time: FloatArray
    station: StrArray
    lat: FloatArray
    lon: FloatArray
    enu: FloatArray  # (n, 3) metres: east, north, up from the series origin
    sigma: FloatArray  # (n, 3) metres, 1σ
    token_hash: StrArray

    def __len__(self) -> int:
        return len(self.time)

    def select(self, mask: BoolArray | IntArray) -> Gnss:
        return Gnss(
            self.time[mask],
            self.station[mask],
            self.lat[mask],
            self.lon[mask],
            self.enu[mask],
            self.sigma[mask],
            self.token_hash[mask],
        )

    @property
    def stations(self) -> list[str]:
        return sorted(set(self.station.tolist()))


GNSS_COLUMNS = ("time", "lat", "lon", "value", "uncertainty", "qualifiers", "token_hash")


def gnss_from_table(table: pa.Table) -> Gnss:
    quals = [json.loads(q) for q in table.column("qualifiers").to_pylist()]
    g = Gnss(
        time=_floats(table, "time"),
        station=np.asarray([q["station"] for q in quals], dtype=np.str_),
        lat=_floats(table, "lat"),
        lon=_floats(table, "lon"),
        enu=_first(table, "value", 3),
        sigma=_first(table, "uncertainty", 3),
        token_hash=_strings(table, "token_hash"),
    )
    return g.select(np.lexsort((g.time, g.station)))


def load_gnss(root: Path) -> Gnss:
    files = instrument_files(root, "gnss_daily")
    return gnss_from_table(read_table(files, GNSS_COLUMNS))


def snapshot_files(snapshot: dict[str, Any], tokens_root: Path) -> list[Path]:
    """Every token file a snapshot names, as paths under the token root."""
    out = []
    for adapter in snapshot["adapters"].values():
        out += [tokens_root / f["path"] for f in adapter["files"]]
    return sorted(out)
