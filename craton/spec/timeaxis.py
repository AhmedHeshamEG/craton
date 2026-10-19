"""Craton's one clock.

Every token's ``time`` is a single float: seconds since 1970-01-01T00:00:00 UTC on the proleptic
Gregorian calendar, leap seconds not counted (the POSIX convention). One float covers a
seismogram sample in 2016 (precise to well under a microsecond) and a plate position a billion
years ago (precise to about a minute), so the same axis runs from rupture to deep time.

Two things this module handles so adapters don't get them wrong:

* **Calendars.** Before 15 October 1582 European sources date events in the Julian calendar.
  Craton's axis is Gregorian everywhere, so a medieval date has to be converted first. The
  Montecassino report of 9 September 1349 (Julian) is 17 September 1349 on the axis.
* **Out-of-range dates.** Python's ``datetime`` stops at year 1. Earlier times are still valid
  floats; they just have no ISO 8601 string.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.2425 * SECONDS_PER_DAY  # mean Gregorian year
EARTH_AGE_S = 4.6e9 * SECONDS_PER_YEAR
"""The axis runs ±4.6 billion years around 1970: older than any rock, wide enough to forecast."""

_MIN_ISO = datetime(1, 1, 1, tzinfo=UTC)
_MAX_ISO = datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)


def to_seconds(moment: datetime | str) -> float:
    """Seconds on the axis for a timezone-aware datetime or an ISO 8601 string.

    Naive datetimes and strings without an offset are refused: a time without a zone is a
    guess, and the axis is UTC.
    """
    dt = (
        datetime.fromisoformat(moment.replace("Z", "+00:00")) if isinstance(moment, str) else moment
    )
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{moment!r} has no timezone; Craton times are UTC")
    delta = dt.astimezone(UTC) - EPOCH
    # Integer days and microseconds keep full precision before converting to float.
    return delta.days * SECONDS_PER_DAY + delta.seconds + delta.microseconds / 1e6


def to_iso(seconds: float) -> str | None:
    """The ISO 8601 UTC string for a time on the axis, or None outside years 1-9999."""
    if not math.isfinite(seconds):
        return None
    try:
        dt = EPOCH + timedelta(seconds=seconds)
    except OverflowError:
        return None
    if not _MIN_ISO <= dt <= _MAX_ISO:
        return None
    text = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if dt.microsecond:
        text += f".{dt.microsecond:06d}".rstrip("0")
    # strftime drops leading zeros on years below 1000 on some platforms.
    return f"{dt.year:04d}{text[text.index('-') :]}Z"


def julian_to_gregorian(year: int, month: int, day: int) -> date:
    """Convert a Julian-calendar date to the proleptic Gregorian date of the same day.

    Goes through the Julian Day Number, so it is exact for every year from 1 onward.
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - 32083
    # Gregorian date from the JDN (Fliegel & Van Flandern).
    f = jdn + 68569
    e = 4 * f // 146097
    f -= (146097 * e + 3) // 4
    g = 4000 * (f + 1) // 1461001
    f = f - 1461 * g // 4 + 31
    h = 80 * f // 2447
    d = f - 2447 * h // 80
    f = h // 11
    mo = h + 2 - 12 * f
    yr = 100 * (e - 49) + g + f
    return date(yr, mo, d)


def year_of(seconds: float) -> int:
    """The calendar year a time falls in, used to partition token files.

    Exact inside years 1-9999; outside, the mean Gregorian year (deep time has no calendar).
    """
    iso = to_iso(seconds)
    if iso is not None:
        return int(iso[:4])
    return 1970 + math.floor(seconds / SECONDS_PER_YEAR)
