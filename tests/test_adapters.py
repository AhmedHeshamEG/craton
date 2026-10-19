"""Adapters, offline: real excerpts from INGV and NGL, a fake network, the whole pipeline."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from craton.adapters import ADAPTERS, get_adapter, register
from craton.adapters.base import BuildContext
from craton.adapters.build import build, build_order, fetch, write_health
from craton.adapters.http import HttpError
from craton.adapters.ingv import IngvCatalog, month_windows, query_url
from craton.adapters.ngl import HOLDINGS_URL, TENV3_URL, NglGnss, parse_holdings, stations_in
from craton.adapters.region import Box, Region
from craton.spec.examples import arquata_gnss
from craton.spec.timeaxis import to_seconds
from craton.store import read_tokens
from craton.store.raw import RawStore

FIX = Path(__file__).parent / "fixtures"
BOX = Box(lat_min=41.0, lat_max=44.0, lon_min=11.5, lon_max=15.0)
NORCIA = (FIX / "ingv" / "norcia.txt").read_bytes()


class FakeHttp:
    """URL → bytes, or an HTTP status to raise. Records every request."""

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


def _region(start: str, end: str, adapters: tuple[str, ...]) -> Region:
    return Region(name="test", description="t", box=BOX, start=start, end=end, adapters=adapters)


QUAKE = _region("2016-10-30T06:35:00Z", "2016-10-30T06:50:00Z", ("ingv_catalog", "iers_tides"))
GNSS = _region("2016-10-28T00:00:00Z", "2016-11-03T00:00:00Z", ("ngl_gnss",))


def _quake_http() -> FakeHttp:
    return FakeHttp({query_url(BOX, QUAKE.t_start, QUAKE.t_end): NORCIA})


def _gnss_http() -> FakeHttp:
    pages: dict[str, bytes | int] = {HOLDINGS_URL: (FIX / "ngl" / "DataHoldings.txt").read_bytes()}
    for code in ("ARQT", "CAT0"):
        pages[TENV3_URL.format(station=code)] = (FIX / "ngl" / f"{code}.tenv3").read_bytes()
    return FakeHttp(pages)  # NRCI is missing on purpose: the server says 404


# ---------------------------------------------------------------- raw store


def test_raw_store_keeps_first_retrieval_and_every_version(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    a = raw.put("src", "2016-10", "u", b"one", "2026-10-20", ".txt")
    again = raw.put("src", "2016-10", "u", b"one", "2026-11-01", ".txt")
    assert again == a
    assert len(raw.entries("src")) == 1
    b = raw.put("src", "2016-10", "u", b"two", "2026-11-01", ".txt")
    assert raw.current("src")["2016-10"] == b
    assert raw.path_of(a).read_bytes() == b"one"  # the old version stays
    back = raw.put("src", "2016-10", "u", b"one", "2026-12-01", ".txt")
    assert back.retrieved == "2026-10-20"  # same bytes, same date, same tokens


def test_raw_store_notices_tampering(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    f = raw.put("src", "x", "u", b"data", "2026-10-20", ".txt")
    raw.path_of(f).write_bytes(b"DATA")
    with pytest.raises(ValueError, match="no longer matches"):
        raw.read(f)


# ---------------------------------------------------------------- INGV


def test_month_windows_cover_the_region_exactly() -> None:
    full = _region("1985-01-01T00:00:00Z", "2026-10-01T00:00:00Z", ("ingv_catalog",))
    w = month_windows(full)
    assert len(w) == 41 * 12 + 9
    assert w[0][0] == full.t_start
    assert w[-1][1] == full.t_end
    assert all(a[1] == b[0] for a, b in itertools.pairwise(w))
    assert month_windows(QUAKE) == [(QUAKE.t_start, QUAKE.t_end)]


def test_ingv_tokens_from_the_norcia_minutes(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    fetch(QUAKE, raw, _quake_http())
    ctx = BuildContext(region=QUAKE, raw=raw)
    toks = [t for batch in IngvCatalog().tokens(ctx) for t in batch]
    assert len(toks) == 12
    main = max(toks, key=lambda t: t.value[0])
    assert main.value == [6.5]
    assert main.qualifiers == {
        "magnitude_type": "Mw",
        "event_id": "8863681",
        "event_type": "earthquake",
    }
    assert main.provenance.record == "eventid=8863681;author=BULLETIN-INGV"
    (raw_file,) = raw.current("ingv_catalog").values()
    assert main.provenance.content_hash == raw_file.content_hash
    assert main.uncertainty == [0.1]
    assert {t.qualifiers["magnitude_type"] for t in toks} == {"Mw", "ML"}
    assert not ctx.rejects


def test_ingv_counts_what_it_refuses(tmp_path: Path) -> None:
    bad = [
        "1|2016-10-30T06:41:00.000000|42.8|13.1|9.0|SURVEY-INGV||||Mx|2.0|--|x|earthquake",
        "2|2016-10-30T06:41:01.000000|42.8|13.1|9.0|SURVEY-INGV||||ML||--|x|earthquake",
        "3|2016-10-30T06:41:02.000000|42.8|13.1|9.0|SURVEY-INGV||||ML|2.0|--|x|landslide",
        "4|2016-10-30T06:41:03.000000|42.8|13.1|9.0",
        "5|2016-10-30T06:41:04.000000|42.8|13.1|9.0|SURVEY-INGV||||ML|12.0|--|x|earthquake",
        "6|2016-10-30T06:41:05.000000|45.9|13.1|9.0|SURVEY-INGV||||ML|2.0|--|x|earthquake",
        "8863681|2016-10-30T06:40:17.320000|42.8303|13.1092|10.0|BULLETIN-INGV||||Mw|6.5|--|"
        "4 km NE Norcia (PG)|earthquake",
    ]
    data = NORCIA + ("\n".join(bad) + "\n").encode()
    raw = RawStore(tmp_path)
    label = IngvCatalog().label(BOX, QUAKE.t_start, QUAKE.t_end)
    raw.put("ingv_catalog", label, "u", data, "d", "")
    ctx = BuildContext(region=QUAKE, raw=raw)
    toks = [t for batch in IngvCatalog().tokens(ctx) for t in batch]
    assert len(toks) == 12
    assert dict(ctx.rejects) == {
        "magnitude_type:Mx": 1,
        "no_magnitude": 1,
        "event_type:landslide": 1,
        "field_count": 1,
        "invalid:value_range": 1,
        "outside_region": 1,
        "duplicate": 1,
    }


def test_ingv_splits_a_month_the_service_finds_too_big(tmp_path: Path) -> None:
    a, b = QUAKE.t_start, QUAKE.t_end
    mid = a + round((b - a) / 2)
    lines = NORCIA.decode().splitlines()
    header, body = lines[0], lines[1:]
    early = [x for x in body if x.split("|")[1] < "2016-10-30T06:42:30"]
    late = [x for x in body if x not in early]
    http = FakeHttp(
        {
            query_url(BOX, a, b): 413,
            query_url(BOX, a, mid): ("\n".join([header, *early]) + "\n").encode(),
            query_url(BOX, mid, b): ("\n".join([header, *late]) + "\n").encode(),
        }
    )
    raw = RawStore(tmp_path)
    fetch(_region(QUAKE.start, QUAKE.end, ("ingv_catalog",)), raw, http)
    (stored,) = raw.current("ingv_catalog").values()
    text = raw.read(stored).decode()
    assert text.count("#EventID") == 1
    assert len(text.splitlines()) == len(lines)
    assert len(http.asked) == 3


def test_fetch_uses_the_cache(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    http = _quake_http()
    fetch(QUAKE, raw, http)
    fetch(QUAKE, raw, http)
    assert len(http.asked) == 1
    fetch(QUAKE, raw, http, refresh=True)
    assert len(http.asked) == 2


# ---------------------------------------------------------------- NGL


def test_station_list_picks_the_box_and_the_window() -> None:
    stations = parse_holdings((FIX / "ngl" / "DataHoldings.txt").read_bytes())
    assert [s.code for s in stations] == ["00NA", "ARQT", "CAT0", "NRCI"]
    assert [s.code for s in stations_in(GNSS, stations)] == ["ARQT", "CAT0", "NRCI"]
    early = _region("1990-01-01T00:00:00Z", "2000-01-01T00:00:00Z", ("ngl_gnss",))
    assert stations_in(early, stations) == []


def test_gnss_tokens_and_the_night_arqt_dropped(tmp_path: Path) -> None:
    raw = RawStore(tmp_path)
    report = fetch(GNSS, raw, _gnss_http())["ngl_gnss"]
    assert report.notes == ["NRCI: no IGS20 tenv3 file (404)"]
    ctx = BuildContext(region=GNSS, raw=raw)
    toks = [t for batch in NglGnss().tokens(ctx) for t in batch]
    assert dict(ctx.rejects) == {"station_not_fetched": 1}
    arqt = {t.time_iso: t for t in toks if t.qualifiers["station"] == "ARQT"}
    assert len(arqt) == 6  # 28 Oct to 2 Nov
    # Same numbers as the P1 example token, read from the same line.
    example = arquata_gnss()
    day = arqt["2016-10-31T12:00:00Z"]
    assert day.value == example["value"]
    assert day.uncertainty == example["uncertainty"]
    assert day.lon == example["lon"]
    assert day.provenance.record == "ARQT.tenv3 16OCT31 reflon=13.2"
    drop = arqt["2016-10-29T12:00:00Z"].value[2] - arqt["2016-10-30T12:00:00Z"].value[2]
    assert 0.4 < drop < 0.5
    cat0 = [t for t in toks if t.qualifiers["station"] == "CAT0"]
    assert len(cat0) == 4
    assert all(t.time_iso and t.time_iso.endswith("T12:00:00Z") for t in toks)


# ---------------------------------------------------------------- tides and the pipeline


def test_tides_at_every_event_and_every_hour(tmp_path: Path) -> None:
    raw = RawStore(tmp_path / "raw")
    fetch(QUAKE, raw, _quake_http())
    snap = build(QUAKE, raw, tmp_path / "tokens", tmp_path / "snaps")
    tide_files = snap.body["adapters"]["iers_tides"]["files"]
    tides = [t for f in tide_files for t in read_tokens(tmp_path / "tokens" / f["path"])]
    events = [
        t
        for f in snap.body["adapters"]["ingv_catalog"]["files"]
        for t in read_tokens(tmp_path / "tokens" / f["path"])
    ]
    at_events = [t for t in tides if (t.provenance.record or "").startswith("event=")]
    assert len(at_events) == len(events) == 12
    assert {t.provenance.content_hash for t in at_events} == {e.token_hash() for e in events}
    assert [t for t in tides if t.provenance.record == "hourly=centre"] == []  # no full hour
    main = next(t for t in at_events if t.provenance.record == "event=8863681")
    assert main.time == to_seconds("2016-10-30T06:40:17.32Z")
    assert all(abs(v) < 0.5 for v in main.value)
    health = snap.body["adapters"]["iers_tides"]["health"]["solid_earth_tide"]
    assert health["at_events"] == 12


def test_hourly_series_has_one_token_per_hour(tmp_path: Path) -> None:
    region = _region("2016-10-30T00:00:00Z", "2016-10-31T00:00:00Z", ("iers_tides",))
    region = region.model_copy(update={"adapters": ("ingv_catalog", "iers_tides")})
    raw = RawStore(tmp_path)
    label = IngvCatalog().label(BOX, region.t_start, region.t_end)
    raw.put("ingv_catalog", label, "u", b"#header\n", "2026-10-20", ".txt")
    snap = build(region, raw, tmp_path / "t", tmp_path / "s")
    assert snap.body["adapters"]["iers_tides"]["tokens"] == 24


def test_rebuilding_gives_the_same_bytes(tmp_path: Path) -> None:
    raw = RawStore(tmp_path / "raw")
    fetch(QUAKE, raw, _quake_http())
    fetch(GNSS, raw, _gnss_http())
    snaps = []
    for run in ("a", "b"):
        for region in (QUAKE, GNSS):
            snaps.append(build(region, raw, tmp_path / run / "tok", tmp_path / run / "snap"))
    assert snaps[0].id == snaps[2].id
    assert snaps[1].id == snaps[3].id
    for f in snaps[1].body["adapters"]["ngl_gnss"]["files"]:
        a = (tmp_path / "a" / "tok" / f["path"]).read_bytes()
        b = (tmp_path / "b" / "tok" / f["path"]).read_bytes()
        assert a == b


def test_health_report_keeps_every_region(tmp_path: Path) -> None:
    import json

    raw = RawStore(tmp_path / "raw")
    fetch(QUAKE, raw, _quake_http())
    snap = build(QUAKE, raw, tmp_path / "t", tmp_path / "s")
    out = tmp_path / "health.json"
    write_health(out, snap)
    report = json.loads(out.read_text(encoding="utf-8"))
    entry = report["regions"]["test"]
    assert entry["snapshot"] == snap.id
    cat = entry["adapters"]["ingv_catalog"]["health"]["seismic_catalog"]
    assert cat["largest"] == {
        "magnitude": 6.5,
        "type": "Mw",
        "day": "2016-10-30",
        "event_id": "8863681",
    }
    assert cat["magnitude_types"] == {"ML": 11, "Mw": 1}


# ---------------------------------------------------------------- the registry


def test_build_order_puts_requirements_first() -> None:
    order = [a.name for a in build_order(("iers_tides", "ingv_catalog", "ngl_gnss"))]
    assert order.index("ingv_catalog") < order.index("iers_tides")
    with pytest.raises(ValueError, match="needs adapter 'ingv_catalog'"):
        build_order(("iers_tides",))


def test_a_name_belongs_to_one_adapter() -> None:
    class Impostor(IngvCatalog):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register(Impostor)
    assert ADAPTERS["ingv_catalog"] is IngvCatalog


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_every_card_is_complete(name: str) -> None:
    card = get_adapter(name).card()
    assert card.name == name
    assert card.citations
    assert card.known_issues
    assert card.mapping
    assert card.adapter_version == get_adapter(name).version
    assert set(card.instruments) == set(get_adapter(name).instruments)


def test_scoreboard_shows_the_data(tmp_path: Path) -> None:
    import json

    from craton.bench.results import Scoreboard
    from craton.bench.scoreboard import render

    raw = RawStore(tmp_path / "raw")
    fetch(QUAKE, raw, _quake_http())
    snap = build(QUAKE, raw, tmp_path / "t", tmp_path / "s")
    out = tmp_path / "health.json"
    write_health(out, snap)
    md = render(Scoreboard(), json.loads(out.read_text(encoding="utf-8")))
    assert f"### test · snapshot `{snap.id}`" in md
    assert "largest Mw 6.5 on 2016-10-30" in md
    assert "12 at earthquakes + 0 hourly" in md


def test_a_stranger_adds_a_source_without_touching_the_core(tmp_path: Path) -> None:
    from collections.abc import Iterator

    from craton.adapters.base import Adapter, DataCard, FetchReport, Log
    from craton.adapters.http import Fetcher
    from craton.spec import Token, validate_token
    from craton.spec.examples import limestone_porosity

    class Borehole(Adapter):
        name = "test_borehole"
        version = "0.1.0"
        instruments = ("rock_porosity",)

        def card(self) -> DataCard:
            return DataCard(
                "test_borehole",
                "t",
                "p",
                "h",
                self.instruments,
                "CC0-1.0",
                "t",
                ("c",),
                "a",
                (("value", "x"),),
                "u",
                ("i",),
                self.version,
            )

        def fetch(
            self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
        ) -> FetchReport:
            raw.put(self.name, "core-1", "u", b"porosity 0.05", "2026-10-20", ".txt")
            return FetchReport(downloaded=1)

        def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
            (f,) = ctx.raw.current(self.name).values()
            ctx.use(f)
            d = limestone_porosity()
            d["provenance"] = {**d["provenance"], "content_hash": f.content_hash}
            yield [validate_token(d)]

    register(Borehole)
    try:
        region = _region("2015-01-01T00:00:00Z", "2017-01-01T00:00:00Z", ("test_borehole",))
        raw = RawStore(tmp_path / "raw")
        fetch(region, raw, FakeHttp({}))
        snap = build(region, raw, tmp_path / "t", tmp_path / "s")
        assert snap.body["adapters"]["test_borehole"]["tokens"] == 1
    finally:
        ADAPTERS.pop("test_borehole")
