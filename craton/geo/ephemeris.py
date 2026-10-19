"""Where the Sun and the Moon are, seen from the Earth's centre.

Low-precision analytic series from Montenbruck & Gill, *Satellite Orbits* (2000), section 3.3.2
(equations 3.43-3.51), with Greenwich sidereal time from section 2.3.1. The Moon comes out good
to a few arcminutes, the Sun better. For a body tide that is plenty: checked against the JPL
DE440 ephemeris, the tide moves by under 3 mm (``tests/test_solid_tide.py``).

Positions are in ECEF metres: they turn with the Earth, ready to sit next to a station.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from craton.geo.clock import JD_J2000, centuries_tt, julian_date

Vec = NDArray[np.float64]

OBLIQUITY_J2000_DEG = 23.43929111
"""Tilt of the Earth's axis against the ecliptic at J2000."""
PRECESSION_DEG_PER_CENTURY = 1.3972
"""How fast the equinox slides along the ecliptic (general precession in longitude)."""
_ARCSEC = 1 / 3600


def _sin(deg: Vec) -> Vec:
    out: Vec = np.sin(np.radians(deg))
    return out


def _cos(deg: Vec) -> Vec:
    out: Vec = np.cos(np.radians(deg))
    return out


def greenwich_angle(t: ArrayLike) -> Vec:
    """Greenwich mean sidereal angle, radians: how far the Earth has turned (UTC ≈ UT1)."""
    days = julian_date(t) - JD_J2000
    deg = 280.46061837504 + 360.9856473662862 * days
    out: Vec = np.radians(np.mod(deg, 360.0))
    return out


def _ecliptic_to_ecef(lon_deg: Vec, lat_deg: Vec, dist_m: Vec, t: ArrayLike) -> Vec:
    """Spherical ecliptic (mean equinox of date) → equatorial → Earth-fixed."""
    x = dist_m * _cos(lon_deg) * _cos(lat_deg)
    y = dist_m * _sin(lon_deg) * _cos(lat_deg)
    z = dist_m * _sin(lat_deg)
    eps = np.radians(OBLIQUITY_J2000_DEG)
    # rotate about x by the obliquity: ecliptic → equator
    ye = y * np.cos(eps) - z * np.sin(eps)
    ze = y * np.sin(eps) + z * np.cos(eps)
    # rotate about z by the sidereal angle: inertial → Earth-fixed
    g = greenwich_angle(t)
    xf = x * np.cos(g) + ye * np.sin(g)
    yf = -x * np.sin(g) + ye * np.cos(g)
    out: Vec = np.stack([xf, yf, ze], axis=-1)
    return out


def sun_ecef(t: ArrayLike) -> Vec:
    """The Sun's geocentric position (..., 3), metres, ECEF. ``t`` in Craton seconds (UTC)."""
    tc = centuries_tt(t)
    m = 357.5256 + 35999.049 * tc  # mean anomaly
    dist = (149.619 - 2.499 * _cos(m) - 0.021 * _cos(2 * m)) * 1e9
    lon = 282.9400 + m + (6892 * _sin(m) + 72 * _sin(2 * m)) * _ARCSEC
    lon = lon + PRECESSION_DEG_PER_CENTURY * tc
    return _ecliptic_to_ecef(lon, np.zeros_like(lon), dist, t)


def moon_ecef(t: ArrayLike) -> Vec:
    """The Moon's geocentric position (..., 3), metres, ECEF. ``t`` in Craton seconds (UTC)."""
    tc = centuries_tt(t)
    l0 = 218.31617 + 481267.88088 * tc - PRECESSION_DEG_PER_CENTURY * tc  # mean longitude
    lm = 134.96292 + 477198.86753 * tc  # Moon's mean anomaly
    ls = 357.52543 + 35999.04944 * tc  # Sun's mean anomaly
    f = 93.27283 + 483202.01873 * tc  # distance from the ascending node
    d = 297.85027 + 445267.11135 * tc  # elongation from the Sun
    lon = l0 + _ARCSEC * (
        22640 * _sin(lm)
        + 769 * _sin(2 * lm)
        - 4586 * _sin(lm - 2 * d)
        + 2370 * _sin(2 * d)
        - 668 * _sin(ls)
        - 412 * _sin(2 * f)
        - 212 * _sin(2 * lm - 2 * d)
        - 206 * _sin(lm + ls - 2 * d)
        + 192 * _sin(lm + 2 * d)
        - 165 * _sin(ls - 2 * d)
        + 148 * _sin(lm - ls)
        - 125 * _sin(d)
        - 110 * _sin(lm + ls)
        - 55 * _sin(2 * f - 2 * d)
    )
    q = _ARCSEC * (412 * _sin(2 * f) + 541 * _sin(ls))
    lat = _ARCSEC * (
        18520 * _sin(f + lon - l0 + q)
        - 526 * _sin(f - 2 * d)
        + 44 * _sin(lm + f - 2 * d)
        - 31 * _sin(-lm + f - 2 * d)
        - 25 * _sin(-2 * lm + f)
        - 23 * _sin(ls + f - 2 * d)
        + 21 * _sin(-lm + f)
        + 11 * _sin(-ls + f - 2 * d)
    )
    dist = 1000.0 * (
        385000
        - 20905 * _cos(lm)
        - 3699 * _cos(2 * d - lm)
        - 2956 * _cos(2 * d)
        - 570 * _cos(2 * lm)
        + 246 * _cos(2 * lm - 2 * d)
        - 205 * _cos(ls - 2 * d)
        - 171 * _cos(lm + 2 * d)
        - 152 * _cos(lm + ls - 2 * d)
    )
    lon = lon + PRECESSION_DEG_PER_CENTURY * tc
    return _ecliptic_to_ecef(lon, lat, dist, t)
