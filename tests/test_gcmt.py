"""The Global CMT adapter: ndk entries become valid moment_tensor tokens, in N·m."""

from __future__ import annotations

import math

import pytest

from craton.adapters import get_adapter
from craton.adapters.gcmt import Solution, moment_magnitude, parse_ndk, rupture_km
from craton.spec.token import validate_token
from craton.store.raw import RawFile

# Two entries from GCMT's own format notes, and one whose centroid numbers touch ("1.060.0").
NDK = """\
PDE  2005/01/01 01:20:05.4  13.78  -88.78 193.1 5.0 0.0 EL SALVADOR
C200501010120A   B:  4    4  40 S: 27   33  50 M:  0    0   0 CMT: 1 TRIHD:  0.6
CENTROID:     -0.3 0.9  13.76 0.06  -89.08 0.09 162.8 12.5 FREE S-20050322125201
23  0.838 0.201 -0.005 0.231 -0.833 0.270  1.050 0.121 -0.369 0.161  0.044 0.240
V10   1.581 56  12  -0.537 23 140  -1.044 24 241   1.312   9 29  142 133 72   66
PDE  2005/01/01 01:42:24.9   7.29   93.92  30.0 5.1 0.0 NICOBAR ISLANDS, INDIA R
C200501010142A   B: 17   27  40 S: 41   58  50 M:  0    0   0 CMT: 1 TRIHD:  0.7
CENTROID:     -1.1 0.8   7.24 0.04   93.96 0.04  12.0  0.0 BDY  S-20050322125628
23 -1.310 0.212  2.320 0.166 -1.010 0.241  0.013 0.535 -2.570 0.668  1.780 0.151
V10   3.376 16 149   0.611 43  44  -3.987 43 254   3.681 282 48  -23  28 73 -136
PDE  1996/11/20 19:42:56.1  10.30  127.43  33.0 5.6 5.3 PHILIPPINE ISLANDS REGIO
B112096E         B: 35   58  45 S:  0    0   0 M:  0    0   0 CMT: 1 BOXHD: 22.0
CENTROID:      1.060.0  10.47 0.03  127.42 0.03  15.0  0.0 BDY  O-00000000000000
24 -4.983 0.133 -0.092 0.153  5.075 0.173 -0.433 0.443  0.715 0.469 -2.898 0.141
V10   6.430  4 246  -1.390  2 156  -5.050 86  45   5.740 338 41  -88 154 49  -92
"""

RAW = RawFile(
    source="gcmt",
    label="test.ndk",
    url="https://example.org/test.ndk",
    sha256="0" * 64,
    size=len(NDK),
    retrieved="2026-11-06",
    suffix=".ndk",
)


def _solutions() -> list[Solution]:
    out = list(parse_ndk(NDK))
    assert all(isinstance(s, Solution) for s in out)
    return [s for s in out if isinstance(s, Solution)]


def test_parse_reads_centroid_and_elements_in_newton_metres() -> None:
    s = _solutions()[0]
    assert s.event_id == "C200501010120A"
    assert (s.lat, s.lon, s.depth_km) == (13.76, -89.08, 162.8)
    assert s.elements[0] == pytest.approx(0.838e23 * 1e-7)  # Mrr, dyne·cm → N·m
    assert s.elements[5] == pytest.approx(0.044e23 * 1e-7)  # Mtp
    assert s.moment == pytest.approx(1.312e23 * 1e-7)
    assert (s.source_type, s.depth_type) == ("deviatoric", "free")
    assert s.half_duration_s == 0.6


def test_parse_centroid_time_is_reference_plus_shift() -> None:
    a, b = _solutions()[:2]
    assert b.time - a.time == pytest.approx((22 * 60 + 19.5) + (-1.1 - -0.3))


def test_touching_numbers_are_read_by_column() -> None:
    s = _solutions()[2]
    assert s.time_sigma_s == 60.0
    assert s.lat == 10.47
    assert s.depth_sigma_km == 0.0


def test_tokens_validate_and_constrained_elements_get_a_positive_sigma() -> None:
    ad = get_adapter("gcmt")
    for s in _solutions():
        tok = validate_token(ad._token(s, RAW, "LicenseRef-GCMT-cite"))  # type: ignore[attr-defined]
        assert tok.instrument == "moment_tensor"
        assert tok.spec == "1.3.0"
        assert len(tok.value) == 7
        assert min(tok.uncertainty) > 0
    third = validate_token(ad._token(_solutions()[2], RAW, "x"))  # type: ignore[attr-defined]
    assert third.location_uncertainty is not None
    assert third.location_uncertainty.vertical_km is None  # depth held fixed: σ 0 is left out


def test_bad_entry_is_reported_not_raised() -> None:
    broken = NDK.replace("23  0.838", "xx  0.838", 1)
    out = list(parse_ndk(broken))
    assert out[0] == "unparseable"
    assert isinstance(out[1], Solution)


def test_moment_magnitude_and_rupture_size() -> None:
    assert moment_magnitude(10 ** (1.5 * 6 + 9.1)) == pytest.approx(6.0)
    assert rupture_km(7.0) == pytest.approx(10 ** (-2.44 + 4.13))
    assert math.isfinite(rupture_km(moment_magnitude(1.312e16)))
