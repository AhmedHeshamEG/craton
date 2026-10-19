"""Data health: what a build actually holds, measured while it is written.

Counts, time span, rejects, and one or two numbers per instrument that say whether the data can
be trusted for what comes next: the catalogue's magnitude of completeness per year, the GNSS
stations' gaps, the size of the tides. Every number lands in the snapshot and in
``results/data_health.json``, and from there on the scoreboard.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from craton.spec.timeaxis import SECONDS_PER_DAY, to_iso
from craton.spec.token import Token

GAP_DAYS = 30
MC_BIN = 0.1
MC_CORRECTION = 0.2
"""Maximum curvature underestimates completeness; +0.2 is the usual fix (Woessner & Wiemer 2005)."""
MC_MIN_EVENTS = 50


def _day(t: float) -> str:
    return (to_iso(t) or "")[:10]


def max_curvature_mc(mags: np.ndarray) -> float | None:
    """Magnitude of completeness: the most populated 0.1 bin, plus 0.2. None if too few events."""
    if len(mags) < MC_MIN_EVENTS:
        return None
    bins = np.round(mags / MC_BIN).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    return round(float(values[np.argmax(counts)]) * MC_BIN + MC_CORRECTION, 1)


class Collector:
    """Watches every token an adapter writes; ``result()`` is the health entry."""

    def __init__(self) -> None:
        self.count = 0
        self.first: float | None = None
        self.last: float | None = None
        self.per_year: Counter[int] = Counter()

    def observe(self, tokens: list[Token]) -> None:
        for t in tokens:
            self.count += 1
            self.first = t.time if self.first is None else min(self.first, t.time)
            self.last = t.time if self.last is None else max(self.last, t.time)
            self.per_year[int((t.time_iso or "0000")[:4])] += 1
            self.see(t)

    def see(self, t: Token) -> None:
        """Instrument-specific bookkeeping."""

    def extra(self) -> dict[str, Any]:
        return {}

    def result(self) -> dict[str, Any]:
        return {
            "tokens": self.count,
            "first": _day(self.first) if self.first is not None else None,
            "last": _day(self.last) if self.last is not None else None,
            "per_year": {str(y): n for y, n in sorted(self.per_year.items())},
            **self.extra(),
        }


class CatalogCollector(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []
        self.mags: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.types: Counter[str] = Counter()
        self.largest: tuple[float, str, str, str] | None = None

    def see(self, t: Token) -> None:
        mt = t.qualifiers["magnitude_type"]
        m = t.value[0]
        self.times.append(t.time)
        self.types[mt] += 1
        self.mags[mt].append((int((t.time_iso or "0000")[:4]), m))
        if self.largest is None or m > self.largest[0]:
            self.largest = (m, mt, _day(t.time), t.qualifiers.get("event_id", ""))

    def extra(self) -> dict[str, Any]:
        times = np.sort(np.array(self.times))
        gap = float(np.diff(times).max() / SECONDS_PER_DAY) if len(times) > 1 else None
        ml = self.mags.get("ML", [])
        years = sorted({y for y, _ in ml})
        mc = {}
        for y in years:
            value = max_curvature_mc(np.array([m for yy, m in ml if yy == y]))
            if value is not None:
                mc[str(y)] = value
        largest = None
        if self.largest is not None:
            m, mt, day, eid = self.largest
            largest = {"magnitude": m, "type": mt, "day": day, "event_id": eid}
        return {
            "magnitude_types": dict(sorted(self.types.items())),
            "largest": largest,
            "longest_gap_days": round(gap, 2) if gap is not None else None,
            "mc_ml_per_year": mc,
        }


class GnssCollector(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.days: dict[str, list[float]] = defaultdict(list)

    def see(self, t: Token) -> None:
        code = t.qualifiers["station"]
        self.days[code].append(t.time)

    def extra(self) -> dict[str, Any]:
        gaps = 0
        coverage = []
        for times in self.days.values():
            ts = np.sort(np.array(times))
            d = np.diff(ts) / SECONDS_PER_DAY
            gaps += int((d > GAP_DAYS).sum())
            span = (ts[-1] - ts[0]) / SECONDS_PER_DAY + 1
            coverage.append(len(ts) / span)
        active: Counter[int] = Counter()
        for times in self.days.values():
            stamps = np.array(times).astype("datetime64[s]").astype("datetime64[Y]")
            active.update((np.unique(stamps).astype(int) + 1970).tolist())
        per_station = [len(v) for v in self.days.values()]
        return {
            "stations": len(self.days),
            "median_days_per_station": int(np.median(per_station)) if per_station else 0,
            "median_coverage": round(float(np.median(coverage)), 3) if coverage else None,
            f"gaps_over_{GAP_DAYS}_days": gaps,
            "stations_per_year": {str(y): n for y, n in sorted(active.items())},
        }


class TideCollector(Collector):
    def __init__(self) -> None:
        super().__init__()
        self.kinds: Counter[str] = Counter()
        self.peak = np.zeros(3)

    def see(self, t: Token) -> None:
        at_event = (t.provenance.record or "").startswith("event=")
        self.kinds["at_events" if at_event else "hourly"] += 1
        self.peak = np.maximum(self.peak, np.abs(np.array(t.value)))

    def extra(self) -> dict[str, Any]:
        return {
            "at_events": self.kinds["at_events"],
            "hourly": self.kinds["hourly"],
            "peak_m": {k: round(float(v), 4) for k, v in zip("enu", self.peak, strict=True)},
        }


COLLECTORS: dict[str, type[Collector]] = {
    "seismic_catalog": CatalogCollector,
    "gnss_daily": GnssCollector,
    "solid_earth_tide": TideCollector,
}


def collector_for(instrument: str) -> Collector:
    """The richest collector known for an instrument; plain counts otherwise."""
    return COLLECTORS.get(instrument, Collector)()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
