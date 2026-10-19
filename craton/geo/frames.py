"""Where a point is, in the two coordinate systems the tide needs.

* **Geodetic** (lat, lon, height above the WGS84 ellipsoid): what tokens carry.
* **ECEF** (x, y, z in metres from the Earth's centre, z through the North Pole, x through the
  Greenwich meridian): where vector arithmetic with the Sun and the Moon is easy.

A displacement computed in ECEF is turned back into east, north and up with :func:`to_enu`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

WGS84_A = 6_378_137.0
"""Equatorial radius of the WGS84 ellipsoid, metres."""
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

Vec = NDArray[np.float64]


def geodetic_to_ecef(lat: ArrayLike, lon: ArrayLike, height_m: ArrayLike = 0.0) -> Vec:
    """ECEF position (..., 3) in metres of geodetic points on WGS84."""
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    h = np.asarray(height_m, dtype=float)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)  # prime-vertical radius
    x = (n + h) * np.cos(phi) * np.cos(lam)
    y = (n + h) * np.cos(phi) * np.sin(lam)
    z = (n * (1 - WGS84_E2) + h) * np.sin(phi)
    out: Vec = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    return out


def enu_basis(lat: ArrayLike, lon: ArrayLike) -> Vec:
    """Unit vectors east, north, up (..., 3, 3) at geodetic points, rows in that order."""
    phi = np.radians(np.asarray(lat, dtype=float))
    lam = np.radians(np.asarray(lon, dtype=float))
    phi, lam = np.broadcast_arrays(phi, lam)
    zero = np.zeros_like(phi)
    east = np.stack([-np.sin(lam), np.cos(lam), zero], axis=-1)
    north = np.stack([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)], axis=-1)
    up = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)
    out: Vec = np.stack([east, north, up], axis=-2)
    return out


def to_enu(vec_ecef: ArrayLike, lat: ArrayLike, lon: ArrayLike) -> Vec:
    """An ECEF vector (..., 3) as east, north, up components (..., 3) at a geodetic point."""
    basis = enu_basis(lat, lon)
    out: Vec = np.einsum("...ij,...j->...i", basis, np.asarray(vec_ecef, dtype=float))
    return out
