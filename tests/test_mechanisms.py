"""P9i: CB-34, SHmax from moment tensors, graded on stress records not read from quakes."""

from __future__ import annotations

import numpy as np
import pytest

from craton.bench.baselines import METHODS
from craton.bench.data import Events
from craton.bench.mechanisms import (
    MechanismKostrov,
    MechanismQuery,
    Mechanisms,
    MechanismStressMapper,
    StressFromMechanisms,
    non_quake,
    non_quake_cells,
    shmax_of,
)
from craton.bench.stress import StressTruth, axial_error
from craton.bench.tasks import TASKS
from craton.bench.truth import parse_wsm


def _tensor_squeezing(azimuth_deg: float, regime: str = "thrust") -> np.ndarray:
    """A double couple in GCMT's (r, t, p) frame whose pressure axis is horizontal at
    ``azimuth_deg`` (thrust or strike-slip), or whose tension axis is (normal fault: SHmax is
    then perpendicular to it)."""
    a = np.radians(azimuth_deg)
    n = np.array([0.0, -np.cos(a), np.sin(a)])  # (r, t, p): t points south
    perp = np.array([0.0, -np.cos(a + np.pi / 2), np.sin(a + np.pi / 2)])
    up = np.array([1.0, 0.0, 0.0])
    if regime == "thrust":
        m = -np.outer(n, n) + np.outer(up, up)
    elif regime == "strike_slip":
        m = -np.outer(n, n) + np.outer(perp, perp)
    else:  # normal: vertical pressure, tension across azimuth + 90
        m = -np.outer(up, up) + np.outer(perp, perp)
    return np.array([m[0, 0], m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2]])


@pytest.mark.parametrize("regime", ["thrust", "strike_slip", "normal"])
@pytest.mark.parametrize("azimuth", [0.0, 30.0, 45.0, 100.0, 170.0])
def test_shmax_recovers_the_squeeze_in_every_regime(regime: str, azimuth: float) -> None:
    got = float(shmax_of(_tensor_squeezing(azimuth, regime)))
    assert float(axial_error(np.array([got]), np.array([azimuth]))[0]) < 1e-6


def test_isotropic_horizontal_block_has_no_direction() -> None:
    assert np.isnan(shmax_of(np.array([-2.0, 1.0, 1.0, 0.0, 0.0, 0.0])))


def _mechs(az: list[float], lat: list[float], lon: list[float], scale: float = 1e17) -> Mechanisms:
    m = np.stack([_tensor_squeezing(a) for a in az]) * scale
    n = len(az)
    return Mechanisms(
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.full(n, 10.0),
        np.arange(n, dtype=np.float64),
        m,
        np.full(n, scale),
    )


def _query(mech: Mechanisms) -> MechanismQuery:
    empty = Events(
        *(np.zeros(0) for _ in range(5)), np.zeros(0, dtype=np.str_), np.zeros(0, dtype=np.str_)
    )
    return MechanismQuery(
        region="x",
        half="seen",
        lat=np.array([42.0, 60.0]),
        lon=np.array([13.0, 13.0]),
        cut="2007-01-01T00:00:00Z",
        mc=2.0,
        history=empty,
        mechanisms=mech,
    )


def test_kostrov_reads_nearby_tensors_and_says_nothing_far_away() -> None:
    mech = _mechs([40.0, 50.0, 45.0], [42.05, 41.95, 42.0], [13.0, 13.05, 12.95])
    got = MechanismKostrov().mechanism_stress_azimuths(_query(mech), 0)
    assert got[0] == pytest.approx(45.0, abs=1.0)
    assert np.isnan(got[1])  # nothing within 150 km of 60° N


def test_kostrov_weighs_shapes_not_sizes() -> None:
    big = _mechs([0.0], [42.0], [13.0], scale=1e20)
    small = _mechs([90.0, 90.0], [42.01, 41.99], [13.0, 13.0])
    both = Mechanisms(
        *(
            np.concatenate([getattr(big, k), getattr(small, k)])
            for k in ("lat", "lon", "depth_km", "time")
        ),
        np.concatenate([big.m, small.m]),
        np.concatenate([big.m0, small.m0]),
    )
    assert MechanismKostrov().mechanism_stress_azimuths(_query(both), 0)[0] == pytest.approx(
        90.0, abs=1.0
    )


def test_truth_leaves_out_every_record_read_from_a_quake() -> None:
    rows = [
        "a,42.1,13.1,10,FMS,10,A,TF",
        "b,42.2,13.2,20,FMF,10,A,TF",
        "c,42.3,13.3,30,BO,2,B,TF",
        "d,42.4,13.4,40,HF,1,C,TF",
        "e,42.6,13.6,50,FMA,10,A,TF",
    ]
    text = ("ID,LAT,LON,AZI,TYPE,DEPTH,QUALITY,REGIME\n" + "\n".join(rows) + "\n").encode()
    rec = parse_wsm(text)
    assert sorted(non_quake(rec).kind.tolist()) == ["BO", "HF"]
    cells = non_quake_cells(rec, (42.0, 43.0, 13.0, 14.0))
    assert len(cells) == 1
    assert cells.n.tolist() == [2]


def test_cb34_is_registered_and_cb23_never_sees_mechanism_methods() -> None:
    assert isinstance(TASKS["CB-34"], StressFromMechanisms)
    kostrov = METHODS["mechanism_kostrov"]
    assert isinstance(kostrov, MechanismStressMapper)
    assert TASKS["CB-34"].supports(kostrov)
    assert not StressTruth().supports(kostrov)  # CB-23 reads no mechanisms, ever
    for name in ("stress_regional", "stress_interpolation", "quake_lineation"):
        assert TASKS["CB-34"].supports(METHODS[name])
