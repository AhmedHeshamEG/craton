"""Continuous waveforms from any FDSN data centre: the raw material of the noise instruments.

Quakes are rare and the ground's hum never stops. Every broadband station records it, every day,
and FDSN ``dataselect`` serves it in the same miniSEED from INGV, IRIS/EarthScope, GEOFON, RESIF
or GeoNet. This module downloads it; the instruments that read it (``seismic_noise_power``,
``seismic_velocity_change``) share one cache, so the waveforms come down once.

    sources:
      fdsn_waveforms:
        station_service: https://webservices.ingv.it/fdsnws/station/1/query
        dataselect_service: https://webservices.ingv.it/fdsnws/dataselect/1/query
        networks: [IV, MN]
        channels: LH?          # 1 Hz broadband: 0.46 MB per station-day for three components
        provider: INGV (EIDA Italia)
        license: CC-BY-4.0

Stations are found by the region's box and window (any station with the channel, no list kept
by hand). Each station-month is one request and one raw file (``NET.STA.YYYY-MM``); a month the
centre has no data for is kept as an empty file, so it is not asked for again. A 100 Hz ``HH``
day is 30 times larger than an ``LH`` day, which is why the default is 1 Hz: correlations in
0.1–0.45 Hz, the band most ambient-noise monitoring of fault zones uses.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from craton.adapters.base import FetchReport, Log
from craton.adapters.http import Fetcher, Http, HttpError
from craton.adapters.ingv import month_windows
from craton.adapters.region import Region
from craton.spec.timeaxis import to_iso, to_seconds
from craton.store.raw import RawFile, RawStore

SOURCE = "fdsn_waveforms"
"""The raw-store folder every noise instrument reads."""
#: A month of 1 Hz data takes about a minute and a half to arrive from INGV.
TIMEOUT_S = 900.0
WORKERS = 4
"""Requests in flight at once. EIDA nodes serve a few parallel clients; more is impolite."""


@dataclass(frozen=True)
class Channel:
    """One channel epoch from the station service's text format (level=channel)."""

    network: str
    station: str
    location: str
    channel: str
    lat: float
    lon: float
    elevation_m: float
    depth_m: float
    azimuth: float
    dip: float
    sensor: str
    scale: float
    """Counts per (m/s) at ``scale_freq``: the overall sensitivity."""
    scale_freq: float
    scale_units: str
    sample_rate: float
    start: float
    end: float

    @property
    def station_id(self) -> str:
        return f"{self.network}.{self.station}"

    @property
    def seed_id(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}"


def options(region: Region) -> dict[str, Any]:
    opts = region.options(SOURCE)
    for key in ("station_service", "dataselect_service", "networks", "provider", "license"):
        if not opts.get(key):
            raise ValueError(f"region {region.name}: sources.{SOURCE} needs {key!r}")
    opts.setdefault("channels", "LH?")
    return opts


def _iso(t: float) -> str:
    return (to_iso(t) or "")[:19]


def station_url(opts: dict[str, Any], region: Region) -> str:
    box = region.box
    params = {
        "network": ",".join(opts["networks"]),
        "channel": opts["channels"],
        "minlatitude": f"{box.lat_min:g}",
        "maxlatitude": f"{box.lat_max:g}",
        "minlongitude": f"{box.lon_min:g}",
        "maxlongitude": f"{box.lon_max:g}",
        "starttime": _iso(region.t_start),
        "endtime": _iso(region.t_end),
        "level": "channel",
        "format": "text",
    }
    return f"{opts['station_service']}?{urllib.parse.urlencode(params, safe=':,?')}"


def dataselect_url(opts: dict[str, Any], network: str, station: str, a: float, b: float) -> str:
    params = {
        "network": network,
        "station": station,
        "channel": opts["channels"],
        "starttime": _iso(a),
        "endtime": _iso(b),
    }
    return f"{opts['dataselect_service']}?{urllib.parse.urlencode(params, safe=':,?')}"


def _time(text: str, default: float) -> float:
    text = text.strip()
    if not text:
        return default
    return to_seconds(datetime.fromisoformat(text).replace(tzinfo=UTC))


def parse_channels(text: str) -> list[Channel]:
    """The station service's text table, one row per channel epoch. Malformed rows raise."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = [x.strip() for x in line.split("|")]
        if len(f) != 17:
            raise ValueError(f"station text row has {len(f)} columns, expected 17: {line!r}")
        out.append(
            Channel(
                network=f[0],
                station=f[1],
                location=f[2],
                channel=f[3],
                lat=float(f[4]),
                lon=float(f[5]),
                elevation_m=float(f[6] or 0),
                depth_m=float(f[7] or 0),
                azimuth=float(f[8] or 0),
                dip=float(f[9] or 0),
                sensor=f[10],
                scale=float(f[11] or 0),
                scale_freq=float(f[12] or 0),
                scale_units=f[13],
                sample_rate=float(f[14]),
                start=_time(f[15], float("-inf")),
                end=_time(f[16], float("inf")),
            )
        )
    return out


def month_label(network: str, station: str, a: float) -> str:
    return f"{network}.{station}.{_iso(a)[:7]}"


def wanted_months(region: Region, channels: list[Channel]) -> list[tuple[str, str, float, float]]:
    """(network, station, start, end) for every month a station has a channel epoch in."""
    by_station: dict[tuple[str, str], list[Channel]] = {}
    for ch in channels:
        by_station.setdefault((ch.network, ch.station), []).append(ch)
    out = []
    for (net, sta), chans in sorted(by_station.items()):
        for a, b in month_windows(region):
            if any(c.start < b and c.end > a for c in chans):
                out.append((net, sta, a, b))
    return out


def stations_label(opts: dict[str, Any], region: Region) -> str:
    """The station table's raw label, named after its query: regions that share the waveform
    cache each keep their own table."""
    return "stations." + hashlib.sha256(station_url(opts, region).encode()).hexdigest()[:12]


def stations_file(raw: RawStore, opts: dict[str, Any], region: Region) -> RawFile | None:
    return raw.current(SOURCE).get(stations_label(opts, region))


def fetch_waveforms(
    region: Region,
    raw: RawStore,
    http: Fetcher,
    refresh: bool,
    log: Log,
    workers: int = WORKERS,
    make_http: Callable[[], Fetcher] | None = None,
) -> FetchReport:
    """Download the region's station table and every station-month of waveforms.

    ``http`` fetches the station table; each worker thread gets its own fetcher from
    ``make_http`` (a polite :class:`Http` by default) so per-host pacing holds per connection.
    """
    opts = options(region)
    report = FetchReport()
    today = datetime.now(UTC).date().isoformat()
    lock = threading.Lock()

    cached = stations_file(raw, opts, region)
    if cached is None or refresh:
        url = station_url(opts, region)
        data = http.get(url)
        raw.put(SOURCE, stations_label(opts, region), url, data, today, ".txt")
        report.downloaded += 1
        report.bytes += len(data)
        cached = stations_file(raw, opts, region)
    assert cached is not None
    channels = parse_channels(raw.read(cached).decode("utf-8", "replace"))
    stations = sorted({c.station_id for c in channels})
    log(f"{SOURCE}: {len(stations)} stations with {opts['channels']}: {' '.join(stations)}")

    have = raw.current(SOURCE)
    todo = []
    for net, sta, a, b in wanted_months(region, channels):
        label = month_label(net, sta, a)
        if label in have and not refresh:
            report.cached += 1
        else:
            todo.append((label, net, sta, a, b))
    log(f"{SOURCE}: {report.cached} station-months cached, {len(todo)} to download")

    local = threading.local()

    def fetcher() -> Fetcher:
        if not hasattr(local, "http"):
            local.http = make_http() if make_http else Http(timeout_s=TIMEOUT_S)
        f: Fetcher = local.http
        return f

    def one(label: str, net: str, sta: str, a: float, b: float) -> int:
        """Bytes stored, or -1 when the month failed (it is asked for again next run)."""
        url = dataselect_url(opts, net, sta, a, b)
        try:
            data = fetcher().get(url)
        except HttpError as exc:
            if exc.status not in (204, 404):
                log(f"{SOURCE}: {label} failed (HTTP {exc.status}); retried next run")
                return -1
            data = b""  # the centre holds nothing for this month
        except OSError as exc:
            log(f"{SOURCE}: {label} failed ({exc.__class__.__name__}); retried next run")
            return -1
        with lock:
            raw.put(SOURCE, label, url, data, today, ".mseed")
        return len(data)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, *job): job[0] for job in todo}
        for fut in as_completed(futures):
            size = fut.result()
            done += 1
            if size < 0:
                report.notes.append(f"{futures[fut]} failed")
                continue
            report.downloaded += 1
            report.bytes += size
            if done % 20 == 0 or done == len(todo):
                log(f"{SOURCE}: {done}/{len(todo)} ({report.bytes / 1e9:.2f} GB)")
    return report
