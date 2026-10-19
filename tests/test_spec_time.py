from datetime import UTC, date, datetime

import pytest

from craton.spec.timeaxis import (
    EARTH_AGE_S,
    julian_to_gregorian,
    to_iso,
    to_seconds,
    year_of,
)


def test_epoch_is_zero() -> None:
    assert to_seconds("1970-01-01T00:00:00Z") == 0.0


def test_iso_round_trip_keeps_milliseconds() -> None:
    t = to_seconds("2016-10-30T06:40:17.32Z")
    assert t == 1_477_809_617.32
    assert to_iso(t) == "2016-10-30T06:40:17.32Z"


def test_offsets_are_converted_to_utc() -> None:
    assert to_seconds("2016-10-30T07:40:17+01:00") == to_seconds("2016-10-30T06:40:17Z")


def test_naive_times_are_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        to_seconds("2016-10-30T06:40:17")
    with pytest.raises(ValueError, match="timezone"):
        to_seconds(datetime(2016, 10, 30))


def test_medieval_dates_work() -> None:
    t = to_seconds(datetime(1349, 9, 17, 12, tzinfo=UTC))
    assert t < 0
    assert to_iso(t) == "1349-09-17T12:00:00Z"
    assert to_iso(to_seconds("0001-01-01T00:00:00Z")) == "0001-01-01T00:00:00Z"


@pytest.mark.parametrize(
    ("julian", "gregorian"),
    [
        ((1582, 10, 4), date(1582, 10, 14)),  # the day the Gregorian calendar began
        ((1349, 9, 9), date(1349, 9, 17)),  # the Montecassino earthquake
        ((1200, 3, 1), date(1200, 3, 8)),
        ((1700, 2, 18), date(1700, 2, 28)),
        ((1900, 3, 1), date(1900, 3, 14)),
    ],
)
def test_julian_to_gregorian(julian: tuple[int, int, int], gregorian: date) -> None:
    assert julian_to_gregorian(*julian) == gregorian


def test_deep_time_has_no_iso_but_a_year() -> None:
    million_years_ago = -1e6 * 365.2425 * 86_400
    assert to_iso(million_years_ago) is None
    assert year_of(million_years_ago) == 1970 - 1_000_000
    assert to_iso(EARTH_AGE_S) is None


def test_year_of_is_exact_near_new_year() -> None:
    assert year_of(to_seconds("2016-12-31T23:59:59Z")) == 2016
    assert year_of(to_seconds("2017-01-01T00:00:00Z")) == 2017
    assert year_of(to_seconds("1349-09-17T12:00:00Z")) == 1349
