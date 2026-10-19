"""Reference solid-Earth tides from two independent codes, for tests/test_solid_tide.py.

Craton computes the body tide with its own NumPy code (craton/geo/solid_tide.py). This script
asks two other programs the same questions so the test can compare:

* PySolid 0.3.4 (Milbert's Fortran, IERS 2003/2010 model, low-precision Sun and Moon). It has
  no Windows wheels, so this script runs in a Linux container.
* pyTMD 3.0.9 with the JPL DE440 ephemeris (the precise Sun and Moon), and with its own
  Montenbruck ephemeris. pyTMD's "N" points south (colatitude); it is flipped to north here.

Six places and dates, hourly for a day each. Run from the repo root:

    docker run --rm -v "$PWD:/w" -w /w python:3.12-slim bash -c \
      "pip install -q pysolid==0.3.4 pyTMD==3.0.9 jplephem && \
       python scripts/reference/tides_reference.py"

It writes tests/fixtures/tides/reference.csv.
"""

import csv
import datetime as dt

import numpy as np
import pysolid
import pyTMD.compute

CASES = [
    ("arquata_norcia", 42.75497, 13.19872, "2016-10-30"),
    ("laquila", 42.342, 13.380, "2009-04-06"),
    ("equator", 0.0, -60.0, "2000-03-20"),
    ("sydney", -33.9, 151.2, "2024-07-01"),
    ("svalbard", 78.2, 15.6, "1990-01-01"),
    ("tokyo", 35.7, 139.7, "2026-10-20"),
]
OUT = "tests/fixtures/tides/reference.csv"


def pytmd_enu(lat, lon, t64, ephemerides):
    n = len(t64)
    ds = pyTMD.compute.SET_displacements(
        np.full(n, lon),
        np.full(n, lat),
        t64,
        standard="datetime",
        type="trajectory",
        ephemerides=ephemerides,
        variable=["N", "E", "R"],
    )
    east, south, up = (np.asarray(ds[k]).ravel() for k in ("E", "N", "R"))
    return east, -south, up


def main():
    rows = []
    for name, lat, lon, day in CASES:
        t0 = dt.datetime.fromisoformat(day)
        t1 = t0 + dt.timedelta(hours=23)
        times, e, n, u = pysolid.calc_solid_earth_tides_point(
            lat, lon, t0, t1, step_sec=3600, display=False, verbose=False
        )
        t64 = np.array([np.datetime64(t) for t in times], dtype="datetime64[ns]")
        jpl = pytmd_enu(lat, lon, t64, "JPL")
        mont = pytmd_enu(lat, lon, t64, "Montenbruck")
        for i, t in enumerate(times):
            values = [e[i], n[i], u[i], *(c[i] for c in jpl), *(c[i] for c in mont)]
            rows.append(
                [name, t.strftime("%Y-%m-%dT%H:%M:%SZ"), lat, lon, *(f"{v:.9f}" for v in values)]
            )
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "case",
                "time",
                "lat",
                "lon",
                "pysolid_e",
                "pysolid_n",
                "pysolid_u",
                "jpl_e",
                "jpl_n",
                "jpl_u",
                "mont_e",
                "mont_n",
                "mont_u",
            ]
        )
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
