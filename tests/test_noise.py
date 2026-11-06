"""The ear: FDSN waveforms in, one noise-power token per station, day and component out."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from craton.adapters import get_adapter
from craton.adapters.base import BuildContext
from craton.adapters.http import HttpError
from craton.adapters.noise import (
    BANDS,
    Segment,
    band_power_db,
    complete_hours,
    day_grid,
    decode,
    summarise,
)
from craton.adapters.region import Region
from craton.adapters.waveforms import (
    SOURCE,
    dataselect_url,
    fetch_waveforms,
    month_label,
    parse_channels,
    station_url,
    stations_label,
    wanted_months,
)
from craton.spec.timeaxis import to_seconds
from craton.store.raw import RawStore

CPU = torch.device("cpu")
GAIN = 1.0e9  # counts per m/s
STATIONS = (
    "#Network | Station | Location | Channel | Latitude | Longitude | Elevation | Depth | "
    "Azimuth | Dip | SensorDescription | Scale | ScaleFreq | ScaleUnits | SampleRate | "
    "StartTime | EndTime\n"
    "XX|ABC||LHZ|42.5|13.0|1000|0|0|-90|TEST SENSOR|1000000000|0.2|m/s|1|"
    "2020-01-01T00:00:00|2020-02-15T00:00:00\n"
    "XX|ABC||LHN|42.5|13.0|1000|0|0|0|TEST SENSOR|1000000000|0.2|m/s|1|"
    "2020-01-01T00:00:00|\n"
)


def region(**extra: object) -> Region:
    body: dict[str, object] = {
        "name": "noise_test",
        "description": "test",
        "box": {"lat_min": 42.0, "lat_max": 43.0, "lon_min": 12.5, "lon_max": 13.5},
        "start": "2020-01-01T00:00:00Z",
        "end": "2020-03-01T00:00:00Z",
        "adapters": ["seismic_noise_power"],
        "sources": {
            SOURCE: {
                "station_service": "https://example.org/fdsnws/station/1/query",
                "dataselect_service": "https://example.org/fdsnws/dataselect/1/query",
                "networks": ["XX"],
                "provider": "Test centre",
                "license": "CC-BY-4.0",
            }
        },
    }
    body.update(extra)
    return Region.model_validate(body)


def white_velocity_day(sigma_m_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(0.0, sigma_m_s * GAIN, 86_400)).astype(np.int32)


def mseed(sourceid: str, samples: np.ndarray, start: str) -> bytes:
    import pymseed

    traces = pymseed.MS3TraceList()
    traces.add_data(sourceid, samples.astype(np.int32), "i", 1.0, starttime_str=start)
    return b"".join(traces.generate(max_record_length=512))


# ---------------------------------------------------------------- the station table


def test_station_table_parses_and_open_ended_epochs_run_forever() -> None:
    ch = parse_channels(STATIONS)
    assert [c.seed_id for c in ch] == ["XX.ABC..LHZ", "XX.ABC..LHN"]
    assert ch[0].scale == GAIN
    assert ch[0].scale_units == "m/s"
    assert ch[0].end == to_seconds("2020-02-15T00:00:00Z")
    assert ch[1].end == math.inf


def test_station_table_refuses_a_short_row() -> None:
    with pytest.raises(ValueError, match="17"):
        parse_channels("XX|ABC||LHZ|42.5|13.0")


def test_months_follow_the_channel_epochs() -> None:
    months = wanted_months(region(), parse_channels(STATIONS[: STATIONS.index("XX|ABC||LHN")]))
    labels = [month_label(n, s, a) for n, s, a, _ in months]
    assert labels == ["XX.ABC.2020-01", "XX.ABC.2020-02"]
    one = wanted_months(region(start="2020-02-15T00:00:00Z"), parse_channels(STATIONS)[:1])
    assert one == []  # the LHZ epoch ended at the start of the window


def test_urls_name_the_box_window_and_channels() -> None:
    r = region()
    opts = r.sources[SOURCE] | {"channels": "LH?"}
    url = station_url(opts, r)
    assert "minlatitude=42" in url
    assert "level=channel" in url
    assert "channel=LH?" in url
    url = dataselect_url(opts, "XX", "ABC", r.t_start, r.t_start + 86_400)
    assert "starttime=2020-01-01T00:00:00" in url
    assert "station=ABC" in url


class FakeHttp:
    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.calls: list[str] = []
        self.missing = missing

    def get(self, url: str) -> bytes:
        self.calls.append(url)
        if "station" in url and "dataselect" not in url:
            return STATIONS.encode()
        if any(m in url for m in self.missing):
            raise HttpError(url, 204, "")
        return b"MSEED"


def test_fetch_downloads_each_month_once_and_remembers_empty_months(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    http = FakeHttp(missing=("2020-02-01",))
    report = fetch_waveforms(region(), raw, http, False, lambda s: None, 1, lambda: http)
    assert report.downloaded == 3  # the table and two months
    files = raw.current(SOURCE)
    assert files["XX.ABC.2020-02"].size == 0
    again = fetch_waveforms(region(), raw, http, False, lambda s: None, 1, lambda: http)
    assert again.downloaded == 0
    assert again.cached == 2


# ---------------------------------------------------------------- the spectra


def test_samples_land_on_the_grid_and_gaps_stay_empty() -> None:
    day0 = to_seconds("2020-01-01T00:00:00Z")
    segs = [
        Segment("", "LHZ", day0 - 10, 1.0, np.arange(20, dtype=np.int32)),
        Segment("", "LHZ", day0 + 7_200, 1.0, np.ones(3_600, dtype=np.int32)),
    ]
    grid = day_grid(segs, day0)
    assert grid[:10].tolist() == list(range(10, 20))
    assert np.isnan(grid[10:7_200]).all()
    hours = complete_hours(grid)
    assert hours.shape == (1, 3_600)  # only hour 2 is whole


def test_white_noise_gives_its_textbook_power() -> None:
    sigma = 1e-7  # m/s
    hours = white_velocity_day(sigma).astype(np.float64).reshape(24, 3_600)
    db = band_power_db(hours, GAIN, device=CPU)
    for k, (lo, hi) in enumerate(BANDS):
        f = np.arange(int(lo * 3_600), int(np.ceil(hi * 3_600))) / 3_600
        f = f[(f >= lo) & (f < hi)]
        expected = 10 * np.log10(np.mean(2 * sigma**2 * (2 * np.pi * f) ** 2))
        assert np.median(db[:, k]) == pytest.approx(expected, abs=0.3)


def test_ten_times_the_shaking_is_twenty_decibels() -> None:
    quiet = white_velocity_day(1e-8).astype(np.float64).reshape(24, 3_600)
    loud = white_velocity_day(1e-7).astype(np.float64).reshape(24, 3_600)
    gap = np.median(band_power_db(loud, GAIN, device=CPU) - band_power_db(quiet, GAIN, device=CPU))
    assert gap == pytest.approx(20.0, abs=0.2)


def test_a_quake_in_two_hours_does_not_move_the_day() -> None:
    hourly = np.full((24, 3), -130.0) + np.random.default_rng(1).normal(0, 0.5, (24, 3))
    calm, _ = summarise(hourly)
    hourly[5:7] = -80.0
    shaken, sigma = summarise(hourly)
    assert np.abs(shaken - calm).max() < 0.3
    assert (sigma >= 0.1).all()


# ---------------------------------------------------------------- the adapter


def test_tokens_one_per_day_and_component_and_short_days_refused(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    r = region()
    label = stations_label(r.sources[SOURCE] | {"channels": "LH?"}, r)
    raw.put(SOURCE, label, "u", STATIONS.encode(), "2026-11-06", ".txt")
    day = white_velocity_day(1e-7)
    data = mseed("FDSN:XX_ABC__L_H_Z", day, "2020-01-01T00:00:00.000000Z")
    data += mseed("FDSN:XX_ABC__L_H_N", day[: 3 * 3_600], "2020-01-02T00:00:00.000000Z")
    raw.put(SOURCE, "XX.ABC.2020-01", "u", data, "2026-11-06", ".mseed")
    raw.put(SOURCE, "XX.ABC.2020-02", "u", b"", "2026-11-06", ".mseed")
    assert len(decode(data)[0]) == 2

    ctx = BuildContext(region=r, raw=raw)
    years = list(get_adapter("seismic_noise_power").tokens(ctx))
    toks = [t for y in years for t in y]
    assert len(toks) == 1  # Jan 1 vertical; the north channel has only three hours on Jan 2
    tok = toks[0]
    assert tok.instrument == "seismic_noise_power"
    assert tok.spec == "1.4.0"
    assert tok.qualifiers == {"station": "XX.ABC", "component": "Z", "channel": ".LHZ"}
    assert tok.depth_km == pytest.approx(-1.0)
    assert tok.time == to_seconds("2020-01-01T00:00:00Z")
    assert -140 < tok.value[1] < -100
    assert ctx.rejects["too_few_hours"] >= 1


@pytest.mark.parametrize("version", [2, 3])
def test_one_damaged_record_costs_one_record(version: int) -> None:
    import pymseed

    from craton.adapters.noise import record_lengths

    x = white_velocity_day(1e-7)[:20_000]
    traces = pymseed.MS3TraceList()
    traces.add_data("FDSN:XX_ABC__L_H_Z", x, "i", 1.0, starttime_str="2020-01-01T00:00:00Z")
    data = bytearray(b"".join(traces.generate(max_record_length=512, format_version=version)))
    lengths = record_lengths(bytes(data))
    assert sum(lengths) == len(data)
    data[sum(lengths[:3]) + (22 if version == 2 else 8)] = 99  # record 4 says hour 99
    segments, bad = decode(bytes(data))
    assert bad == 1
    assert len(x) - 512 < sum(len(s.samples) for s in segments) < len(x)
