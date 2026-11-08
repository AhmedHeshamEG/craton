"""Daily noise correlations for every station pair of a region, computed once and cached.

The slow half of dv/v. For each month: decode every station's 1 Hz waveforms, whiten each hour
(:func:`craton.adapters.velocity.hour_spectra`), correlate every pair of stations closer than
``max_pair_km`` on all nine component pairs, and keep the day's mean over the hours both sides
recorded whole. One month of one region is one file under ``data/cache/velocity/``, named by
the raw files that went in and the settings, so a rebuild reads it back and a changed input
or setting computes it again.

Region YAML (all optional):

    sources:
      fdsn_waveforms:
        velocity:
          max_pair_km: 80
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton.adapters.noise import COMPONENTS, _days_in, _month_start, component_of, day_grid, decode
from craton.adapters.region import Region
from craton.adapters.velocity import BAND, HOURS, MAX_LAG_S, hour_spectra, pair_correlations
from craton.adapters.waveforms import SOURCE, Channel, options, parse_channels, stations_label
from craton.store.raw import RawFile, RawStore

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

VERSION = "1"
"""Bump when the correlation recipe changes: every cached month is then recomputed."""
MAX_PAIR_KM = 80.0
RATE = 1.0


@dataclass(frozen=True)
class Station:
    station_id: str
    lat: float
    lon: float
    elevation_m: float


@dataclass
class Correlations:
    """A region's daily correlations: ``cc`` (P, 9, D, L) float32, ``hours`` (P, 9, D)."""

    days: FloatArray
    stations: list[Station]
    pairs: list[tuple[int, int]]
    cc: np.ndarray[Any, np.dtype[np.float32]]
    hours: np.ndarray[Any, np.dtype[np.int16]]
    used: list[RawFile]

    def distance_km(self, p: int) -> float:
        a, b = self.pairs[p]
        return km(self.stations[a], self.stations[b])


def km(a: Station, b: Station) -> float:
    """Great-circle distance, km."""
    la1, lo1, la2, lo2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((la2 - la1) / 2) ** 2
        + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def settings(region: Region) -> dict[str, Any]:
    opts = options(region).get("velocity", {}) or {}
    return {"max_pair_km": float(opts.get("max_pair_km", MAX_PAIR_KM))}


def stations_of(channels: list[Channel]) -> list[Station]:
    """Every station with at least one Z/N/E channel at 1 Hz, sorted by code."""
    out: dict[str, Station] = {}
    for c in channels:
        if component_of(c.channel) is None or abs(c.sample_rate - RATE) > 1e-6:
            continue
        out.setdefault(c.station_id, Station(c.station_id, c.lat, c.lon, c.elevation_m))
    return [out[k] for k in sorted(out)]


def pairs_of(stations: list[Station], max_km: float) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i, j in itertools.combinations(range(len(stations)), 2)
        if km(stations[i], stations[j]) <= max_km
    ]


def _cache_dir(raw: RawStore) -> Path:
    return Path(raw.root).parent / "cache" / "velocity"


def _key(region: Region, month: str, files: list[RawFile], cfg: dict[str, Any]) -> str:
    body = {
        "version": VERSION,
        "band": BAND,
        "max_lag_s": MAX_LAG_S,
        "cfg": cfg,
        "month": month,
        "files": sorted(f.sha256 for f in files),
        "t": [region.t_start, region.t_end],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _pick_location(segments: list[Any]) -> str:
    """The location code with the most samples: one sensor per station, never two mixed."""
    total: dict[str, int] = {}
    for s in segments:
        total[s.location] = total.get(s.location, 0) + len(s.samples)
    return max(sorted(total), key=lambda k: total[k])


def _month(
    raw: RawStore,
    region: Region,
    month: str,
    stations: list[Station],
    pairs: torch.Tensor,
    files: dict[str, RawFile],
    device: torch.device,
) -> tuple[FloatArray, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    by: list[dict[str, list[Any]]] = []
    for st in stations:
        f = files.get(f"{st.station_id}.{month}")
        comps: dict[str, list[Any]] = {}
        if f is not None:
            segs = [s for s in decode(raw.read(f))[0] if abs(s.rate - RATE) < 1e-6]
            if segs:
                loc = _pick_location(segs)
                for s in segs:
                    comp = component_of(s.channel)
                    if s.location == loc and comp is not None:
                        comps.setdefault(comp, []).append(s)
        by.append(comps)
    days = [d for d in _days_in(_month_start("x." + month)) if region.t_start <= d < region.t_end]
    n_lag = 2 * round(MAX_LAG_S * RATE) + 1
    cc = np.full((len(pairs), 9, len(days), n_lag), np.nan, dtype=np.float32)
    hours = np.zeros((len(pairs), 9, len(days)), dtype=np.int16)
    for di, d0 in enumerate(days):
        grid = np.stack(
            [day_grid(comps.get(comp, []), d0, RATE) for comps in by for comp in COMPONENTS]
        )
        spec, ok = hour_spectra(grid, RATE, device)
        n_st = len(stations)
        day_cc, day_n = pair_correlations(
            spec.reshape(n_st, 3, HOURS, -1), ok.reshape(n_st, 3, HOURS), pairs, RATE, chunk=8
        )
        cc[:, :, di] = day_cc.numpy().astype(np.float32)
        hours[:, :, di] = day_n.numpy().astype(np.int16)
    return np.asarray(days, dtype=np.float64), cc, hours


def correlations(
    region: Region,
    raw: RawStore,
    device: torch.device | None = None,
    log: Any = None,
) -> Correlations:
    """The region's daily correlations, from the cache where it can."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = options(region)
    cfg = settings(region)
    files = raw.current(SOURCE)
    table = files.get(stations_label(opts, region))
    if table is None:
        raise FileNotFoundError(f"{region.name}: station table not fetched")
    stations = stations_of(parse_channels(raw.read(table).decode("utf-8", "replace")))
    ids = {s.station_id for s in stations}
    pairs = pairs_of(stations, cfg["max_pair_km"])
    pairs_t = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
    months = sorted(
        {
            label.rsplit(".", 1)[1]
            for label in files
            if label.rsplit(".", 1)[0] in ids
            and region.t_start <= _month_start(label) < region.t_end
        }
    )
    cache = _cache_dir(raw) / region.name
    cache.mkdir(parents=True, exist_ok=True)
    parts: list[tuple[FloatArray, Any, Any]] = []
    used: list[RawFile] = []
    for month in months:
        mine = [
            files[f"{s.station_id}.{month}"] for s in stations if f"{s.station_id}.{month}" in files
        ]
        used += mine
        path = cache / f"{month}-{_key(region, month, mine, cfg)}.npz"
        if path.exists():
            z = np.load(path)
            parts.append((z["days"], z["cc"], z["hours"]))
            continue
        days, cc, hours = _month(raw, region, month, stations, pairs_t, files, dev)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, days=days, cc=cc, hours=hours)
        tmp.replace(path)
        parts.append((days, cc, hours))
        if log is not None:
            log(f"correlations {region.name} {month}: {len(days)} days, {len(pairs)} pairs")
    if not parts:
        empty_days: FloatArray = np.zeros(0)
        return Correlations(
            empty_days,
            stations,
            pairs,
            np.zeros((len(pairs), 9, 0, 0), np.float32),
            np.zeros((len(pairs), 9, 0), np.int16),
            used,
        )
    return Correlations(
        days=np.concatenate([p[0] for p in parts]),
        stations=stations,
        pairs=pairs,
        cc=np.concatenate([p[1] for p in parts], axis=2),
        hours=np.concatenate([p[2] for p in parts], axis=2),
        used=used,
    )
