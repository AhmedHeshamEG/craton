"""Projected coordinates → WGS84 latitude and longitude, for the grids sources publish.

Reservoir models and national surveys publish on national grids, not in degrees. Each grid a
source uses gets its conversion here, by EPSG code, so an adapter reads any of them from a
line of YAML. No projection library: the formulas below are the ones the grid's own
authority publishes, accurate far below the size of any cell they are used for.

- **EPSG:28992, Amersfoort / RD New** (the Netherlands): the approximation series of
  Schreutelkamp & Strang van Hees (2001), "Benaderingsformules voor de transformatie tussen
  RD- en WGS84-kaartcoördinaten", Geodesia 43(2), 64-69; about 1 m inside the Netherlands.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# RD New: false origin at the Onze Lieve Vrouwetoren, Amersfoort.
_RD_X0, _RD_Y0 = 155_000.0, 463_000.0
_RD_LAT0, _RD_LON0 = 52.15517440, 5.38720621
#: (p, q, coefficient): latitude += K · dx^p · dy^q (arc seconds).
_RD_LAT = (
    (0, 1, 3235.65389),
    (2, 0, -32.58297),
    (0, 2, -0.24750),
    (2, 1, -0.84978),
    (0, 3, -0.06550),
    (2, 2, -0.01709),
    (1, 0, -0.00738),
    (4, 0, 0.00530),
    (2, 3, -0.00039),
    (4, 1, 0.00033),
    (1, 1, -0.00012),
)
#: (p, q, coefficient): longitude += L · dx^p · dy^q (arc seconds).
_RD_LON = (
    (1, 0, 5260.52916),
    (1, 1, 105.94684),
    (1, 2, 2.45656),
    (3, 0, -0.81885),
    (1, 3, 0.05594),
    (3, 1, -0.05607),
    (0, 1, 0.01199),
    (3, 2, -0.00256),
    (1, 4, 0.00128),
    (0, 2, 0.00022),
    (2, 0, -0.00022),
    (5, 0, 0.00026),
)


def _rd_new(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    dx = (x - _RD_X0) * 1e-5
    dy = (y - _RD_Y0) * 1e-5
    lat = sum(k * dx**p * dy**q for p, q, k in _RD_LAT)
    lon = sum(k * dx**p * dy**q for p, q, k in _RD_LON)
    return np.asarray(_RD_LAT0 + lat / 3600.0), np.asarray(_RD_LON0 + lon / 3600.0)


CONVERSIONS = {"EPSG:28992": _rd_new}


def to_wgs84(crs: str, x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(lat, lon) in degrees of points given in ``crs`` (metres east, north)."""
    if crs not in CONVERSIONS:
        raise ValueError(f"no conversion from {crs}; known: {', '.join(sorted(CONVERSIONS))}")
    lat, lon = CONVERSIONS[crs](np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
