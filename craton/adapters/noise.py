"""Seismic noise power: how loudly the ground hummed at each station, every day.

The ground never stops moving. Ocean waves pressing on the sea floor make the microseisms
(0.05-0.3 Hz), wind and traffic add the rest, and a seismometer records all of it between
quakes. This adapter turns the continuous 1 Hz waveforms (``craton.adapters.waveforms``) into
one token per station, day and component:

    1. place the day's samples on a 1 Hz grid (gaps stay empty)
    2. cut it into 24 hours; keep only complete hours
    3. per hour, on the GPU: remove the trend, taper (Hann), FFT, divide by the sensor's gain
       (counts per m/s), turn velocity into acceleration ((2πf)^2) and average the power in
       three bands
    4. the day's value = the median over its hours, in dB, so a quake that shook a few hours
       does not move it; the uncertainty = the spread of the hours over sqrt(hours)

It is McNamara & Buland's (2004) recipe at one window per hour and three bands instead of the
full spectrum, which is what a forecaster can read. Nothing here knows a place: the region
file names the data centre, networks and channel.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import torch

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.adapters.waveforms import (
    SOURCE,
    Channel,
    fetch_waveforms,
    options,
    parse_channels,
    stations_label,
)
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

SPEC = "1.4.0"
DAY_S = 86_400
HOUR_S = 3_600
BANDS: tuple[tuple[float, float], ...] = ((0.05, 0.1), (0.1, 0.3), (0.3, 0.45))
"""Hz, one per channel of ``seismic_noise_power``: primary, secondary, short."""
MIN_HOURS = 8
"""A day needs a third of its hours complete, or it says nothing (shock days lose many)."""
MIN_SIGMA_DB = 0.1
COMPONENTS = ("Z", "N", "E")

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class Segment:
    """A run of evenly spaced samples from one channel."""

    location: str
    channel: str
    start: float
    rate: float
    samples: FloatArray


def record_lengths(data: bytes) -> list[int]:
    """Byte length of each miniSEED 2 record, read from its blockette 1000.

    A record whose header can't be read is taken to be as long as the one before it (512 for
    the first), so one damaged header costs one record, not the rest of the month."""
    out = []
    pos, last = 0, 512
    while pos + 40 <= len(data):
        if data[pos : pos + 3] == b"MS\x03":  # miniSEED 3: lengths sit in the fixed header
            sid, extra, payload = struct.unpack("<BHI", data[pos + 33 : pos + 40])
            length = 40 + sid + extra + payload
            out.append(min(length, len(data) - pos))
            pos += length
            continue
        if pos + 48 > len(data):
            break
        length = last
        year_be = struct.unpack(">H", data[pos + 20 : pos + 22])[0]
        order = ">" if 1900 <= year_be <= 2200 else "<"
        off = struct.unpack(order + "H", data[pos + 46 : pos + 48])[0]
        for _ in range(8):  # walk the blockette chain
            if off < 48 or pos + off + 8 > len(data):
                break
            kind, nxt = struct.unpack(order + "HH", data[pos + off : pos + off + 4])
            if kind == 1000:
                exp = data[pos + off + 6]
                if 7 <= exp <= 20:
                    length = 1 << exp
                break
            off = nxt
        out.append(min(length, len(data) - pos))
        pos += length
        last = length
    return out


def _segments(data: bytes) -> list[Segment]:
    import pymseed

    out = []
    traces = pymseed.MS3TraceList.from_buffer(data, unpack_data=True)
    for sid in traces.sourceids():
        _, _, loc, cha = pymseed.sourceid2nslc(sid)
        trace = traces.get_traceid(sid)
        if trace is None:
            continue
        for seg in trace:
            out.append(
                Segment(
                    loc,
                    cha,
                    float(seg.starttime_seconds),
                    float(seg.samprate),
                    np.asarray(seg.datasamples, dtype=np.float64),
                )
            )
    return out


def decode(data: bytes, group: int = 64) -> tuple[list[Segment], int]:
    """Every segment in a miniSEED buffer, and how many records were unreadable.

    The whole buffer is decoded at once; if libmseed refuses it (one bad timestamp is enough),
    it is decoded in groups of records, and a group that fails record by record, dropping
    only the records that fail. Empty bytes (a month with no data) is no segments."""
    if not data:
        return [], 0
    from pymseed import MiniSEEDError

    try:
        return _segments(data), 0
    except MiniSEEDError:
        pass
    lengths = record_lengths(data)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    out: list[Segment] = []
    bad = 0
    for g in range(0, len(lengths), group):
        a, b = starts[g], starts[min(g + group, len(lengths))]
        try:
            out.extend(_segments(data[a:b]))
            continue
        except MiniSEEDError:
            pass
        for i in range(g, min(g + group, len(lengths))):
            try:
                out.extend(_segments(data[starts[i] : starts[i + 1]]))
            except MiniSEEDError:
                bad += 1
    return out, bad


def day_grid(segments: list[Segment], day0: float, rate: float = 1.0) -> FloatArray:
    """One day of one channel on a regular grid; NaN where nothing was recorded.

    Samples are placed by rounding their time to the grid: at 1 Hz, the sub-second offsets
    data centres leave between records are far below the hour-long windows that read them."""
    n = round(DAY_S * rate)
    grid = np.full(n, np.nan)
    for seg in segments:
        first = round((seg.start - day0) * rate)
        lo, hi = max(first, 0), min(first + len(seg.samples), n)
        if lo < hi:
            grid[lo:hi] = seg.samples[lo - first : hi - first]
    return grid


def complete_hours(grid: FloatArray, rate: float = 1.0) -> FloatArray:
    """The day's hours with no missing sample, shape (hours, samples per hour)."""
    per = round(HOUR_S * rate)
    hours = grid[: per * 24].reshape(24, per)
    return hours[~np.isnan(hours).any(axis=1)]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def band_power_db(
    hours: FloatArray,
    counts_per_m_s: float,
    rate: float = 1.0,
    bands: tuple[tuple[float, float], ...] = BANDS,
    device: torch.device | None = None,
) -> FloatArray:
    """Acceleration power per hour and band, dB re 1 (m/s^2)^2/Hz; shape (hours, bands).

    One-sided PSD of a Hann-tapered, linearly detrended window, normalised so that white
    noise of variance s^2 sampled at ``rate`` gives 2 s^2 / rate per Hz."""
    if hours.shape[0] == 0:
        return np.zeros((0, len(bands)))
    dev = device or _device()
    x = torch.as_tensor(hours, dtype=torch.float64, device=dev) / counts_per_m_s
    n = x.shape[1]
    t = torch.linspace(-1.0, 1.0, n, dtype=torch.float64, device=dev)
    x = x - x.mean(dim=1, keepdim=True)
    x = x - (x @ t / (t @ t))[:, None] * t[None, :]
    w = torch.hann_window(n, periodic=False, dtype=torch.float64, device=dev)
    spec = torch.fft.rfft(x * w, dim=1)
    psd = 2.0 * spec.abs() ** 2 / (rate * (w * w).sum())
    f = torch.fft.rfftfreq(n, d=1.0 / rate).to(dev, torch.float64)
    psd = psd * (2 * math.pi * f) ** 2  # velocity → acceleration
    cols = []
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi)
        cols.append(psd[:, sel].mean(dim=1))
    power = torch.stack(cols, dim=1).clamp_min(1e-30)
    out: FloatArray = (10.0 * torch.log10(power)).cpu().numpy()
    return out


def summarise(hourly_db: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The day's median per band and its 1σ: 1.4826·MAD over sqrt(hours), at least 0.1 dB."""
    med = np.median(hourly_db, axis=0)
    mad = 1.4826 * np.median(np.abs(hourly_db - med), axis=0)
    sigma = np.maximum(mad / math.sqrt(hourly_db.shape[0]), MIN_SIGMA_DB)
    return med, sigma


def component_of(channel: str) -> str | None:
    c = channel[-1:].upper()
    return c if c in COMPONENTS else None


def epoch_at(channels: list[Channel], location: str, channel: str, t: float) -> Channel | None:
    for c in channels:
        if c.location == location and c.channel == channel and c.start <= t < c.end:
            return c
    return None


def _month_start(label: str) -> float:
    y, m = (int(x) for x in label.rsplit(".", 1)[-1].split("-"))
    return to_seconds(datetime(y, m, 1, tzinfo=UTC))


def _days_in(t0: float) -> list[float]:
    first = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=t0)
    out = []
    d = first
    while d.month == first.month:
        out.append(to_seconds(d))
        d += timedelta(days=1)
    return out


@register
class SeismicNoisePower(Adapter):
    name = "seismic_noise_power"
    version = "1.0.0"
    instruments = ("seismic_noise_power",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Daily seismic noise power at broadband stations, from FDSN waveforms",
            provider="The data centre the region names (central Italy: INGV, EIDA Italia)",
            homepage="https://www.fdsn.org/webservices/",
            instruments=self.instruments,
            license="CC-BY-4.0",
            terms=(
                "Each region names its data centre's licence (INGV's Italian networks: CC BY "
                "4.0, cite the network DOI). Tokens carry the licence the region states."
            ),
            citations=(
                "INGV Seismological Data Centre (2006). Rete Sismica Nazionale (RSN). "
                "doi:10.13127/SD/X0FXNH7QFY",
                "MedNet Project Partner Institutions (1988). Mediterranean Very Broadband "
                "Seismographic Network (MedNet). doi:10.13127/SD/FBBBTDTD6Q",
                "McNamara, D. E., Buland, R. P. (2004). Ambient noise levels in the "
                "continental United States. BSSA 94, 1517-1527. doi:10.1785/012003001",
                "Peterson, J. (1993). Observations and modeling of seismic background noise. "
                "USGS Open-File Report 93-322.",
            ),
            access=(
                "FDSN station (text, level=channel) for coordinates and gains, then dataselect "
                "one station-month per request, 1 Hz LH channels (~14 MB a month, ~90 s from "
                "INGV), four requests in flight, cached by hash under raw/fdsn_waveforms."
            ),
            mapping=(
                ("kind", "`observation`"),
                ("lat, lon", "the station, from the station service"),
                ("depth_km", "(sensor depth - elevation) / 1000; elevation is above sea level"),
                ("time", "midnight UTC starting the day; the value covers the whole day"),
                ("value", "median over complete hours of mean band power, 3 bands, dB"),
                ("uncertainty", "1.4826·MAD of the hours / sqrt(hours), at least 0.1 dB"),
                ("qualifiers", "station NET.STA, component Z/N/E, channel LOC.CHA"),
                ("resolution", "the station (1 m) and one day"),
            ),
            uncertainty=(
                "Derived from how much the day's hours disagree. It says how well the day's "
                "level is known, not how far the gain is off (a fixed offset per channel)."
            ),
            known_issues=(
                "One gain per channel epoch (the station service's overall sensitivity at its "
                "reference frequency): exact for flat broadband sensors in these bands; "
                "Trillium 40 s sensors read ~1 dB low at 0.05 Hz.",
                "A sensor swap starts a new epoch with a new gain; a level step across the "
                "swap date is the sensor, not the Earth.",
                "The primary and secondary bands are mostly the ocean: storms in the "
                "Mediterranean and the Atlantic move them by 10-20 dB in winter.",
                "A record libmseed cannot read (one INGV month holds a timestamp of 10^9 ns) "
                "is dropped and counted; the hours it held then count as incomplete.",
                "Hours with any missing sample are dropped; a day with fewer than 8 hours is "
                "left out.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        return fetch_waveforms(region, raw, http, refresh, log)

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        opts = options(ctx.region)
        files = ctx.raw.current(SOURCE)
        table = files.get(stations_label(opts, ctx.region))
        if table is None:
            ctx.reject("stations_not_fetched")
            return
        channels = parse_channels(ctx.use(table).decode("utf-8", "replace"))
        stations = {c.station_id for c in channels}
        months = sorted(
            (f for label, f in files.items() if label.rsplit(".", 1)[0] in stations),
            key=lambda f: (_month_start(f.label), f.label),
        )
        by_year: dict[int, list[Token]] = {}
        for f in months:
            t0 = _month_start(f.label)
            if not ctx.region.t_start <= t0 < ctx.region.t_end:
                continue
            for tok in self._month(ctx, f, channels, opts):
                by_year.setdefault(year_of(tok.time), []).append(tok)
        for yr in sorted(by_year):
            yield sorted(
                by_year[yr],
                key=lambda tk: (tk.time, tk.qualifiers["station"], tk.qualifiers["component"]),
            )

    def _month(
        self, ctx: BuildContext, f: RawFile, channels: list[Channel], opts: dict[str, Any]
    ) -> Iterator[Token]:
        network, station = f.label.split(".")[:2]
        mine = [c for c in channels if c.network == network and c.station == station]
        by_channel: dict[tuple[str, str], list[Segment]] = {}
        segments, bad = decode(ctx.use(f))
        if bad:
            ctx.reject("unreadable_record", bad)
        for seg in segments:
            by_channel.setdefault((seg.location, seg.channel), []).append(seg)
        for (loc, cha), segs in sorted(by_channel.items()):
            comp = component_of(cha)
            if comp is None:
                ctx.reject("component_not_zne")
                continue
            rates = {s.rate for s in segs}
            if len(rates) != 1:
                ctx.reject("mixed_sample_rates")
                continue
            rate = rates.pop()
            for day0 in _days_in(_month_start(f.label)):
                if not ctx.region.t_start <= day0 < ctx.region.t_end:
                    continue
                epoch = epoch_at(mine, loc, cha, day0 + DAY_S / 2)
                if epoch is None:
                    ctx.reject("no_channel_epoch")
                    continue
                if epoch.scale_units.lower() != "m/s" or epoch.scale <= 0:
                    ctx.reject("not_a_velocity_sensor")
                    continue
                hours = complete_hours(day_grid(segs, day0, rate), rate)
                if hours.shape[0] < MIN_HOURS:
                    ctx.reject("too_few_hours")
                    continue
                value, sigma = summarise(band_power_db(hours, epoch.scale, rate))
                try:
                    yield validate_token(
                        self._token(epoch, comp, day0, value, sigma, hours.shape[0], f, opts)
                    )
                except (TokenError, ValueError) as exc:
                    codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                    ctx.reject("invalid:" + ",".join(codes))

    def _token(
        self,
        ch: Channel,
        comp: str,
        day0: float,
        value: FloatArray,
        sigma: FloatArray,
        n_hours: int,
        f: RawFile,
        opts: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "spec": SPEC,
            "kind": "observation",
            "instrument": "seismic_noise_power",
            "lat": ch.lat,
            "lon": ch.lon,
            "depth_km": (ch.depth_m - ch.elevation_m) / 1000.0,
            "time": day0,
            "time_iso": to_iso(day0),
            "value": [round(float(v), 3) for v in value],
            "uncertainty": [round(float(s), 3) for s in sigma],
            "uncertainty_basis": "derived",
            "qualifiers": {
                "station": ch.station_id,
                "component": comp,
                "channel": f"{ch.location}.{ch.channel}",
            },
            "provenance": {
                "source": str(opts["provider"]),
                "dataset_version": f"{f.label}, retrieved {f.retrieved}",
                "license": str(opts["license"]),
                "adapter_version": self.version,
                "content_hash": f.content_hash,
                "record": f"{ch.seed_id}, {n_hours} complete hours",
                "synthetic": False,
            },
            "resolution": {"spatial_km": 0.001, "temporal_s": float(DAY_S)},
        }
