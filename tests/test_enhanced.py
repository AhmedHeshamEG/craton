"""P8c: research catalogues from Zenodo, and CB-20 / CB-21's union of two catalogues."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from craton.adapters.base import BuildContext
from craton.adapters.build import fetch
from craton.adapters.region import Box, Region, load_region
from craton.adapters.zenodo import ZenodoCatalog, parse_iso, url
from craton.bench.data import Events
from craton.bench.enhanced import CALIBRATE_BELOW, MagnitudeMap, fit_map, match, union
from craton.bench.tasks import TASKS
from craton.spec.timeaxis import to_seconds
from craton.store.raw import RawStore

TAN_CSV = (
    "year,month,day,hour,minute,second,lat,lon,dep,EH1,EH2,AZ,EZ,id,ml_n,ml_mean,ml_median,"
    "ml_std,mw,split\n"
    "2016,8,24,1,36,32.5,42.70,13.23,8.1,0.05,0.04,100,0.09,1,15,5.9,6.0,0.2,6.0,0\n"
    "2016,8,24,1,40,1.25,42.71,13.24,7.0,0.30,0.10,100,0.50,2,15,1.9,2.0,0.2,2.0,1\n"
    "2016,8,24,1,41,0.0,42.72,13.25,6.0,0.20,0.10,100,0.40,3,3,,,,,0\n"
    "2016,8,24,1,42,59.9,45.00,13.25,6.0,0.20,0.10,100,0.40,4,3,1.0,1.0,0.1,1.5,0\n"
)
OPTS = {
    "record": 4736089,
    "file": "Amatrice_CAT5.v20210504.csv",
    "provider": "Tan et al. (2021)",
    "license": "CC-BY-4.0",
    "citation": "Tan et al. (2021)",
    "magnitude_type": "ML",
    "columns": {
        "lat": "lat",
        "lon": "lon",
        "depth": "dep",
        "magnitude": "ml_median",
        "id": "id",
        "horizontal_err": ["EH1", "EH2"],
        "vertical_err": "EZ",
    },
    "time_parts": ["year", "month", "day", "hour", "minute", "second"],
    "keep": {"split": "0"},
}


class FakeHttp:
    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = pages

    def get(self, u: str) -> bytes:
        return self.pages[u]


def _log(_: str) -> None:
    return None


def _region(data: bytes, **extra: object) -> Region:
    opts = {**OPTS, "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(), **extra}
    return Region(
        name="tan_test",
        description="t",
        box=Box(lat_min=42.0, lat_max=44.0, lon_min=11.5, lon_max=14.0),
        start="2016-08-15T00:00:00Z",
        end="2017-08-16T00:00:00Z",
        adapters=("zenodo_catalog",),
        sources={"zenodo_catalog": opts},
    )


def test_zenodo_reads_a_catalogue_by_its_column_names(tmp_path: Path) -> None:
    data = TAN_CSV.encode()
    region = _region(data)
    raw = RawStore(tmp_path)
    fetch(region, raw, FakeHttp({url(OPTS): data}), log=_log)
    ctx = BuildContext(region=region, raw=raw)
    toks = [t for batch in ZenodoCatalog().tokens(ctx) for t in batch]
    # Row 2 is flagged split, row 3 has no magnitude, row 4 is outside the box.
    assert len(toks) == 1
    (t,) = toks
    assert t.value == [6.0]
    assert t.time == pytest.approx(to_seconds("2016-08-24T01:36:32.5Z"))
    assert t.time_iso is not None
    assert t.time_iso.startswith("2016-08-24T01:36:32")
    assert t.qualifiers == {"magnitude_type": "ML", "event_id": "zenodo4736089:1"}
    assert t.location_uncertainty is not None
    assert t.location_uncertainty.horizontal_km == pytest.approx(0.05)
    assert t.location_uncertainty.vertical_km == pytest.approx(0.09)
    assert t.provenance.license == "CC-BY-4.0"
    assert sum(ctx.rejects.values()) == 3


def test_a_changed_record_fails_its_md5(tmp_path: Path) -> None:
    region = _region(b"something else")
    with pytest.raises(ValueError, match="md5"):
        fetch(region, RawStore(tmp_path), FakeHttp({url(OPTS): TAN_CSV.encode()}), log=_log)


def test_iso_times_without_a_zone_are_utc() -> None:
    assert parse_iso("2009-01-07T04:41:22.63") == pytest.approx(
        to_seconds("2009-01-07T04:41:22.63Z")
    )
    assert parse_iso("2009-01-07 04:41:22") == to_seconds("2009-01-07T04:41:22Z")


def test_the_three_machine_learning_regions_load() -> None:
    for name in ("central_italy_ml_2016", "laquila_ml_2009", "alto_tiberina_ml"):
        region = load_region(name)
        assert region.adapters == ("zenodo_catalog",)
        assert len(region.options("zenodo_catalog")["md5"]) == 32


# ---------------------------------------------------------------- the union


def _events(t: list[float], lat: list[float], mag: list[float], tag: str) -> Events:
    n = len(t)
    return Events(
        np.asarray(t, float),
        np.asarray(lat, float),
        np.full(n, 13.2),
        np.full(n, 10.0),
        np.asarray(mag, float),
        np.full(n, "ML"),
        np.asarray([f"{tag}{i}" for i in range(n)]),
    )


def test_quakes_match_within_three_seconds_and_ten_km() -> None:
    base = _events([0.0, 100.0, 200.0], [42.7, 42.7, 42.7], [2.0, 2.0, 2.0], "b")
    extra = _events([1.0, 100.0, 205.0, 300.0], [42.7, 42.9, 42.7, 42.7], [1.8, 1.8, 1.8, 1.8], "x")
    idx, hit = match(base, extra)
    # Same quake 1 s apart; 22 km apart is another quake; 5 s apart too; nothing near 300 s.
    assert idx.tolist() == [0, -1, -1, -1]
    assert hit.tolist() == [True, False, False, False]


def test_the_magnitude_map_ignores_the_scored_quakes() -> None:
    rng = np.random.default_rng(0)
    m = rng.uniform(1.0, 4.0, 2000)
    t = np.arange(len(m)) * 100.0
    base = _events(list(t), [42.7] * len(m), list(m), "b")
    # The second catalogue reads 0.3 low below M3 and 1.0 low above it.
    shifted = np.where(m < CALIBRATE_BELOW, m - 0.3, m - 1.0)
    extra = _events(list(t + 1), [42.7] * len(m), list(shifted), "x")
    idx, _ = match(base, extra)
    mmap = fit_map(base, extra, idx)
    # Only the pairs below M3 shaped it, so the shift it learned is 0.3, not a blend.
    assert mmap(np.array([2.0]))[0] == pytest.approx(2.3, abs=0.05)
    # Beyond the fitted range it goes on with slope one.
    assert mmap(np.array([5.0]))[0] - mmap(np.array([4.0]))[0] == pytest.approx(1.0)


def test_the_union_keeps_every_base_quake_and_adds_the_unmatched() -> None:
    n = 400
    t = np.arange(n) * 100.0
    base = _events(list(t), [42.7] * n, list(np.linspace(1.0, 2.9, n)), "b")
    extra_t = np.concatenate([t + 1, t + 50])  # a twin of each, and a new quake between
    extra = _events(list(extra_t), [42.7] * (2 * n), [1.0] * (2 * n), "x")
    extra = extra.select(np.argsort(extra.time, kind="stable"))
    u = union(base, extra)
    assert u.matched == n
    assert len(u.added) == n
    assert len(u.events) == 2 * n
    assert set(base.token_hash) <= set(u.events.token_hash)
    assert np.all(np.diff(u.events.time) >= 0)


def test_a_map_is_monotone_and_says_what_it_does() -> None:
    mm = MagnitudeMap(np.array([0.0, 1.0, 2.0]), np.array([0.5, 1.5, 2.2]))
    out = mm(np.array([-1.0, 0.5, 1.5, 3.0]))
    assert np.all(np.diff(out) > 0)
    assert mm.as_dict()["2"] == 2.2


def test_cb20_and_cb21_score_their_base_tasks_quakes() -> None:
    for new, base in (("CB-20", "CB-02"), ("CB-21", "CB-03")):
        a, b = TASKS[new], TASKS[base]
        assert a.window == b.window  # type: ignore[attr-defined]
        assert a.spec.trained_as == base  # type: ignore[attr-defined]
        assert a.spec.fit_end == b.spec.fit_end  # type: ignore[attr-defined]
