"""The body tide, checked three ways: the IERS test cases, PySolid, and the JPL ephemeris."""

import csv
from pathlib import Path

import numpy as np
import pytest

from craton.geo.clock import tai_minus_utc
from craton.geo.ephemeris import moon_ecef, sun_ecef
from craton.geo.frames import enu_basis, geodetic_to_ecef, to_enu
from craton.geo.solid_tide import solid_tide_enu, tide_ecef
from craton.spec.timeaxis import to_seconds

FIXTURE = Path(__file__).parent / "fixtures" / "tides" / "reference.csv"

# The test cases published in the header of the IERS routine DEHANTTIDEINEL (station, Sun and
# Moon in ECEF metres; expected displacement in metres). The header has a fourth case whose Sun
# sits 8.4 million km away and whose expected output repeats the third case's; it is left out.
IERS_CASES = [
    (
        (4075578.385, 931852.890, 4801570.154),
        (137859926952.015, 54228127881.4350, 23509422341.6960),
        (-179996231.920342, -312468450.131567, -169288918.592160),
        "2009-04-13",
        (0.7700420357108125891e-01, 0.6304056321824967613e-01, 0.5516568152597246810e-01),
    ),
    (
        (1112189.660, -4842955.026, 3985352.284),
        (-54537460436.2357, 130244288385.279, 56463429031.5996),
        (300396716.912, 243238281.451, 120548075.939),
        "2012-07-13",
        (-0.2036831479592075833e-01, 0.5658254776225972449e-01, -0.7597679676871742227e-01),
    ),
    (
        (1112200.5696, -4842957.8511, 3985345.9122),
        (100210282451.6279, 103055630398.3160, 56855096480.4475),
        (369817604.4348, 1897917.5258, 120804980.8284),
        "2015-07-15",
        (0.00509570869172363845, 0.0828663025983528700, -0.0636634925404189617),
    ),
]


@pytest.mark.parametrize(("xsta", "xsun", "xmon", "day", "expected"), IERS_CASES)
def test_reproduces_the_iers_test_cases(
    xsta: tuple[float, ...],
    xsun: tuple[float, ...],
    xmon: tuple[float, ...],
    day: str,
    expected: tuple[float, ...],
) -> None:
    got = tide_ecef(xsta, xsun, xmon, to_seconds(day + "T00:00:00Z"))[0]
    np.testing.assert_allclose(got, expected, atol=1e-12)


def _reference() -> tuple[np.ndarray, np.ndarray]:
    with FIXTURE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["lon"]) for r in rows])
    t = np.array([to_seconds(r["time"]) for r in rows])
    ours = solid_tide_enu(lat, lon, t)
    cols = ["pysolid_e", "pysolid_n", "pysolid_u", "jpl_e", "jpl_n", "jpl_u"]
    ref = np.array([[float(r[c]) for c in cols] for r in rows])
    return ours, ref


def test_matches_pysolid_to_half_a_millimetre() -> None:
    ours, ref = _reference()
    assert np.abs(ours - ref[:, 0:3]).max() < 0.0005


def test_matches_the_jpl_ephemeris_inside_the_stated_uncertainty() -> None:
    # The tide tokens carry 1 mm per channel; the low-precision Sun and Moon must stay inside it.
    ours, ref = _reference()
    assert np.abs(ours - ref[:, 3:6]).max() < 0.001


def test_the_ground_moves_tens_of_centimetres() -> None:
    t = to_seconds("2016-10-30T00:00:00Z") + 600.0 * np.arange(6 * 24 * 30)
    up = solid_tide_enu(42.75, 13.2, t)[:, 2]
    assert 0.2 < up.max() - up.min() < 0.7
    assert abs(up.mean()) < 0.05


def test_sun_and_moon_are_where_they_should_be() -> None:
    t = to_seconds("2016-10-30T06:40:17Z")
    au = np.linalg.norm(sun_ecef(t))
    moon = np.linalg.norm(moon_ecef(t))
    assert 1.47e11 < au < 1.53e11
    assert 3.56e8 < moon < 4.07e8


def test_new_moon_puts_sun_and_moon_on_the_same_side() -> None:
    # New moon, 30 October 2016 17:38 UTC: the two directions nearly line up.
    t = to_seconds("2016-10-30T17:38:00Z")
    s, m = sun_ecef(t), moon_ecef(t)
    cos = s @ m / np.linalg.norm(s) / np.linalg.norm(m)
    assert cos > 0.99


def test_frames_round_trip() -> None:
    x = geodetic_to_ecef(0.0, 0.0, 0.0)
    np.testing.assert_allclose(x, [6378137.0, 0.0, 0.0])
    pole = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert pole[2] == pytest.approx(6356752.314, abs=1e-3)
    basis = enu_basis(42.0, 13.0)
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    up = basis[2]
    np.testing.assert_allclose(to_enu(up, 42.0, 13.0), [0, 0, 1], atol=1e-12)


def test_leap_seconds() -> None:
    assert tai_minus_utc(to_seconds("1971-06-01T00:00:00Z")) == 10
    assert tai_minus_utc(to_seconds("2016-12-31T23:59:59Z")) == 36
    assert tai_minus_utc(to_seconds("2017-01-01T00:00:00Z")) == 37
    assert tai_minus_utc(to_seconds("2026-10-20T00:00:00Z")) == 37
