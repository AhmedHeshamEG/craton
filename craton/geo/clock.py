"""Astronomical time scales from Craton's clock.

Craton's ``time`` is UTC seconds since 1970 (``craton.spec.timeaxis``). Astronomy wants two more:

* **Julian Date**: days since noon, 1 January 4713 BC, the astronomer's running day count.
* **TT** (Terrestrial Time): the smooth clock the Moon and the Sun move by. TT = UTC + leap
  seconds so far + 32.184 s. The leap-second count is the table below.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from craton.spec.timeaxis import SECONDS_PER_DAY, to_seconds

JD_UNIX_EPOCH = 2_440_587.5
"""Julian Date of 1970-01-01T00:00:00 UTC."""
JD_J2000 = 2_451_545.0
"""Julian Date of 2000-01-01T12:00:00 TT, the reference epoch of the formulas."""
DAYS_PER_CENTURY = 36_525.0
TT_MINUS_TAI_S = 32.184

# TAI − UTC from the day it took effect (IERS Bulletin C). No leap second since 2017; a new one
# is announced six months ahead and goes at the end of this list.
_LEAPS = (
    ("1972-01-01", 10),
    ("1972-07-01", 11),
    ("1973-01-01", 12),
    ("1974-01-01", 13),
    ("1975-01-01", 14),
    ("1976-01-01", 15),
    ("1977-01-01", 16),
    ("1978-01-01", 17),
    ("1979-01-01", 18),
    ("1980-01-01", 19),
    ("1981-07-01", 20),
    ("1982-07-01", 21),
    ("1983-07-01", 22),
    ("1985-07-01", 23),
    ("1988-01-01", 24),
    ("1990-01-01", 25),
    ("1991-01-01", 26),
    ("1992-07-01", 27),
    ("1993-07-01", 28),
    ("1994-07-01", 29),
    ("1996-01-01", 30),
    ("1997-07-01", 31),
    ("1999-01-01", 32),
    ("2006-01-01", 33),
    ("2009-01-01", 34),
    ("2012-07-01", 35),
    ("2015-07-01", 36),
    ("2017-01-01", 37),
)
_LEAP_T = np.array([to_seconds(d + "T00:00:00Z") for d, _ in _LEAPS])
_LEAP_S = np.array([s for _, s in _LEAPS], dtype=float)


def tai_minus_utc(t: ArrayLike) -> NDArray[np.float64]:
    """Leap seconds in force at UTC time ``t`` (Craton seconds).

    Before 1972 UTC had no whole-second steps; 10 s is used there. The error is a few seconds of
    time, which moves a tide by well under a micrometre.
    """
    idx = np.searchsorted(_LEAP_T, np.asarray(t, dtype=float), side="right") - 1
    out: NDArray[np.float64] = np.where(idx >= 0, _LEAP_S[np.clip(idx, 0, None)], _LEAP_S[0])
    return out


def julian_date(t: ArrayLike) -> NDArray[np.float64]:
    """Julian Date (UTC) of Craton seconds."""
    out: NDArray[np.float64] = np.asarray(t, dtype=float) / SECONDS_PER_DAY + JD_UNIX_EPOCH
    return out


def tt_offset_s(t: ArrayLike) -> NDArray[np.float64]:
    """TT − UTC in seconds."""
    return tai_minus_utc(t) + TT_MINUS_TAI_S


def centuries_tt(t: ArrayLike) -> NDArray[np.float64]:
    """Julian centuries of TT since J2000: the time argument of every series below."""
    days = julian_date(t) - JD_J2000 + tt_offset_s(t) / SECONDS_PER_DAY
    out: NDArray[np.float64] = days / DAYS_PER_CENTURY
    return out
