"""P8d: the world's catalogues, regions as YAML, the world Field's tiles, CB-22 and CB-29."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from craton.adapters.base import BuildContext
from craton.adapters.build import fetch
from craton.adapters.earthquakenpp import parse_time
from craton.adapters.fdsn import FdsnCatalog, magnitude_type, query_url
from craton.adapters.http import HttpError
from craton.adapters.jma import depth_km, magnitude, record
from craton.adapters.region import Box, Region, load_region
from craton.spec.timeaxis import to_seconds
from craton.store.raw import RawStore

FIX = Path(__file__).parent / "fixtures" / "world"
RIDGECREST = (FIX / "comcat_ridgecrest.txt").read_bytes()
BOX = Box(lat_min=35.0, lat_max=36.5, lon_min=-118.5, lon_max=-117.0)
COMCAT = {
    "service": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "provider": "USGS ANSS ComCat",
    "license": "LicenseRef-USGS-public-domain",
    "min_magnitude": 2.0,
}


class FakeHttp:
    def __init__(self, pages: dict[str, bytes | int]) -> None:
        self.pages = pages
        self.asked: list[str] = []

    def get(self, url: str) -> bytes:
        self.asked.append(url)
        page = self.pages.get(url, 404)
        if isinstance(page, int):
            raise HttpError(url, page, "fake")
        return page


def _log(_: str) -> None:
    return None


def _region(**opts: object) -> Region:
    return Region(
        name="ridgecrest_test",
        description="t",
        box=BOX,
        start="2019-07-06T03:19:00Z",
        end="2019-07-06T03:20:00Z",
        adapters=("fdsn_catalog",),
        completeness=2.5,
        sources={"fdsn_catalog": {**COMCAT, **opts}},
    )


# ---------------------------------------------------------------- regions


def test_a_region_written_before_p8d_keeps_its_fingerprint() -> None:
    # The value central Italy's snapshot was built with in P2; new optional fields stay out.
    assert load_region("central_italy").fingerprint() == "ff5911599a372269"


def test_every_world_region_is_a_yaml_file_that_loads() -> None:
    for name in ("california", "japan", "new_zealand", "turkey", "chile", "alaska"):
        region = load_region(name)
        assert region.completeness is not None
        assert region.sources or region.adapters == ("jma_hypocenter",)


# ---------------------------------------------------------------- FDSN


def test_fdsn_reads_comcat_and_keeps_its_spelling(tmp_path: Path) -> None:
    region = _region()
    raw = RawStore(tmp_path)
    fetch(region, raw, FakeHttp({query_url(COMCAT, BOX, region.t_start, region.t_end): RIDGECREST}))
    ctx = BuildContext(region=region, raw=raw)
    toks = [t for batch in FdsnCatalog().tokens(ctx) for t in batch]
    main = max(toks, key=lambda t: t.value[0])
    assert main.value == [7.1]
    assert main.qualifiers == {"magnitude_type": "Mw", "event_id": "ci38457511"}
    assert main.provenance.record == "eventid=ci38457511;author=ci;magtype=mw"
    assert main.provenance.license == "LicenseRef-USGS-public-domain"
    assert main.spec == "1.0.0"  # Mw needs nothing newer
    assert main.time == pytest.approx(to_seconds("2019-07-06T03:19:53.040Z"))


def test_fdsn_words_and_what_it_refuses(tmp_path: Path) -> None:
    assert magnitude_type("mww") == "Mw"
    assert magnitude_type("Mlv") == "MLv"
    assert magnitude_type("M") == "M"
    assert magnitude_type("ma") is None
    region = _region()
    header = RIDGECREST.decode().splitlines()[0] + "|EventType"
    rows = [
        "a1|2019-07-06T03:19:10|35.7|-117.6|8|GNS|||a1|M|2.5|GNS|x|unknown",
        "a2|2019-07-06T03:19:20|35.7|-117.6|8|GNS|||a2|ma|2.5|GNS|x|earthquake",
        "a3|2019-07-06T03:19:30|35.7|-117.6|8|GNS|||a3|ML|2.5|GNS|x|landslide",
        "a4|2019-07-06T03:19:40|35.7|-117.6||GNS|||a4|ML|2.5|GNS|x|earthquake",
    ]
    raw = RawStore(tmp_path)
    label = FdsnCatalog().label(BOX, region.t_start, region.t_end)
    raw.put("fdsn_catalog", label, "u", ("\n".join([header, *rows]) + "\n").encode(), "d", "")
    ctx = BuildContext(region=region, raw=raw)
    (tok,) = [t for batch in FdsnCatalog().tokens(ctx) for t in batch]
    assert tok.qualifiers == {"magnitude_type": "M", "event_id": "a1"}  # "unknown" said nothing
    assert tok.spec == "1.1.0"  # "M" arrived in 1.1.0
    assert dict(ctx.rejects) == {"magnitude_type:ma": 1, "event_type:landslide": 1, "no_depth": 1}


def test_fdsn_halves_a_month_comcat_calls_too_big(tmp_path: Path) -> None:
    region = _region()
    a, b = region.t_start, region.t_end
    mid = a + round((b - a) / 2)
    lines = RIDGECREST.decode().splitlines()
    pages: dict[str, bytes | int] = {
        query_url(COMCAT, BOX, a, b): 400,
        query_url(COMCAT, BOX, a, mid): ("\n".join(lines[:1]) + "\n").encode(),
        query_url(COMCAT, BOX, mid, b): ("\n".join(lines) + "\n").encode(),
    }
    raw = RawStore(tmp_path)
    fetch(region, raw, FakeHttp(pages))
    (stored,) = raw.current("fdsn_catalog").values()
    assert raw.read(stored).decode().count("#EventID") == 1


def test_a_region_names_its_service_or_fails_loudly(tmp_path: Path) -> None:
    region = _region(service="")
    with pytest.raises(ValueError, match="service"):
        fetch(region, RawStore(tmp_path), FakeHttp({}))


# ---------------------------------------------------------------- JMA


def test_jma_fixed_width_records() -> None:
    lines = (FIX / "jma_2016_head.txt").read_text().splitlines()
    assert record(lines[0]) == "low_frequency"
    r = record(lines[1])
    assert isinstance(r, dict)
    # 2016-01-01 00:05:00.41 JST is 2015-12-31 15:05:00.41 UTC.
    assert r["time"] == pytest.approx(to_seconds("2015-12-31T15:05:00.41Z"))
    assert r["lat"] == pytest.approx(33 + 3.49 / 60)
    assert r["lon"] == pytest.approx(131 + 1.40 / 60)
    assert r["depth_km"] == pytest.approx(9.34)
    assert (r["mag"], r["type"]) == (0.7, "Mj")
    small = record(lines[3])
    assert isinstance(small, dict)
    assert small["mag"] == pytest.approx(-0.3)
    assert small["depth_km"] == 0.0  # a 1 km depth slice at 0


def test_jma_magnitude_letters_and_depth_slices() -> None:
    assert magnitude("A5") == pytest.approx(-1.5)
    assert magnitude("B0") == pytest.approx(-2.0)
    assert magnitude("-9") == pytest.approx(-0.9)
    assert magnitude("73") == pytest.approx(7.3)
    assert magnitude("  ") is None
    assert depth_km(" 2342") == pytest.approx(23.42)
    assert depth_km(" 10  ") == 10.0


# ---------------------------------------------------------------- EarthquakeNPP


def test_npp_times_are_read_to_the_nanosecond() -> None:
    assert parse_time("1971-01-01 20:36:17.720") == pytest.approx(
        to_seconds("1971-01-01T20:36:17.72Z")
    )
    assert parse_time("2020-01-16 16:56:39.140000000") == pytest.approx(
        to_seconds("2020-01-16T16:56:39.14Z")
    )


def test_npp_polygon_area_and_rounding() -> None:
    from craton.bench.npp import AREA_KM2, cell_areas, polygon_mask, round_half_up

    assert cell_areas().sum() == pytest.approx(AREA_KM2)
    assert polygon_mask(np.array([36.0, 36.0]), np.array([-119.0, -110.0])).tolist() == [
        True,
        False,
    ]
    # Their own rounding (etas.mc_b_est.round_half_up), float quirks included.
    assert round_half_up(np.array([2.45, 2.449, 2.66])).tolist() == pytest.approx([2.5, 2.4, 2.7])


def test_npp_cell_masses_add_up_to_the_kernel() -> None:
    from craton.bench.npp import GRID, KERNEL, R_CUT_KM, near_masses

    lat, lon, m = np.array([36.03, 36.0]), np.array([-119.04, -119.0]), np.array([2.5, 6.0])
    pp, cc, mass, far = near_masses(lat, lon, m)
    full = KERNEL.space_mass(m)
    for j in range(2):
        assert mass[pp == j].sum() + far[j] == pytest.approx(full[j])
        assert 0 < far[j] / full[j] < 0.1
    # A small quake's mass sits almost all in its own cell.
    own = GRID.cell_of(lat[:1], lon[:1])[0]
    assert mass[(pp == 0) & (cc == own)].sum() / full[0] > 0.9
    assert R_CUT_KM == 50.0


# ---------------------------------------------------------------- the world Field and CB-22


def test_tiles_cover_a_box_exactly() -> None:
    from craton.field.world import tile_grids

    tiles = tile_grids((31.0, 43.5, -126.0, -113.0), 2.5, 3.0, 3.5)
    assert sum(g.n_cells for g in tiles) == 125 * 130
    assert max(g.n_cells for g in tiles) == 1050
    assert all(g.m_min == 2.5 for g in tiles)


def test_central_italy_is_one_tile_of_the_world() -> None:
    from craton.field.world import tile_grids

    (g,) = tile_grids((41.0, 44.0, 11.5, 15.0), 2.0, 3.0, 3.5)
    assert (g.n_lat, g.n_lon) == (30, 35)


def test_the_world_section_leaves_old_field_configs_alone() -> None:
    from craton.field.config import field_hash, load_field_config

    assert field_hash(load_field_config("mini")).startswith("b99e49ae")


def test_every_sequence_box_is_cb02_sized_and_inside_its_region() -> None:
    from craton.bench.world import SEQUENCES, ZERO_SHOT, sequence_spec

    halves = {s.half for s in SEQUENCES}
    assert halves == {"seen", ZERO_SHOT}
    for seq in SEQUENCES:
        region = load_region(seq.region)
        spec = sequence_spec(seq, region)
        g = spec.grid
        assert (g.n_lat, g.n_lon) == (30, 35)
        assert region.box.contains(g.lat_min, g.lon_min)
        assert region.box.contains(g.lat_max, g.lon_max)
        assert spec.scored == pytest.approx(spec.mc + 1)
        assert to_seconds(spec.fit_end) < spec.window.t0
        assert len(spec.day_starts) == 365
        # Seen regions trained up to 2007: no sequence may start before.
        assert spec.window.t0 > to_seconds("2007-01-01T00:00:00Z")
        assert not math.isnan(spec.max_magnitude)
