import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from craton.bench import csep

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "csep" / "reference.json").read_text(encoding="utf-8")
)
CASES = REFERENCE["cases"]


def grid(**kw: float) -> csep.Grid:
    return csep.Grid(41.0, 44.0, 11.5, 15.0, **kw)


def test_grid_shape_and_cells() -> None:
    g = grid()
    assert (g.n_lat, g.n_lon, g.n_cells) == (30, 35, 1050)
    cells = g.cell_of(np.array([41.0, 43.999, 44.0, 40.9]), np.array([11.5, 14.999, 12.0, 12.0]))
    assert cells.tolist() == [0, 1049, -1, -1]
    lat, lon = g.cell_centres()
    assert g.cell_of(lat, lon).tolist() == list(range(1050))


def test_magnitude_bins_put_reported_values_in_their_own_bin() -> None:
    g = grid(m_min=3.0, m_max=8.0)
    assert g.n_mags == 51
    bins = g.mag_bin_of(np.array([2.9, 2.95, 3.0, 3.04, 3.1, 6.5, 9.0]))
    assert bins.tolist() == [-1, 0, 0, 0, 1, 35, 50]


def test_counts() -> None:
    g = grid(m_min=3.0)
    c = g.counts(
        np.array([41.05, 41.05, 43.95]), np.array([11.55, 11.55, 14.95]), np.array([3.0, 3.0, 2.0])
    )
    assert c.sum() == 2
    assert c[0, 0] == 2


def test_gutenberg_richter_shares() -> None:
    g = grid(m_min=2.0, m_max=8.0)
    shares = csep.gutenberg_richter_bins(1.0, 2.0, g)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] / shares[1] == pytest.approx(10**0.1)
    # Extrapolating below the fit magnitude: M1.5+ is 10^0.5 times M2.0+ for b = 1.
    low = csep.gutenberg_richter_bins(1.0, 2.0, grid(m_min=1.5, m_max=8.0))
    assert low.sum() == pytest.approx(10**0.5)


def test_b_value_recovers_a_known_one() -> None:
    rng = np.random.default_rng(0)
    beta = 1.1 * math.log(10)
    mags = np.round(1.95 + rng.exponential(1 / beta, 200_000), 1)
    assert csep.b_value(mags, 2.0) == pytest.approx(1.1, abs=0.02)
    with pytest.raises(ValueError, match="at least two"):
        csep.b_value(np.array([1.0]), 2.0)


def test_poisson_ll_matches_scipy() -> None:
    rates = np.array([[0.5, 2.0], [0.1, 3.0]])
    counts = np.array([[1, 0], [0, 4]])
    expected = float(np.sum(poisson.logpmf(counts, rates)))
    assert csep.poisson_ll(rates, counts) == pytest.approx(expected)
    assert csep.poisson_ll(np.array([0.0]), np.array([1.0])) == -math.inf
    with pytest.raises(ValueError, match="negative"):
        csep.poisson_ll(np.array([-1.0]), np.array([0.0]))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"seed{c['seed']}")
def test_matches_pycsep(case: dict) -> None:
    rates, counts = np.array(case["rates"]), np.array(case["counts"])
    assert csep.poisson_ll(rates, counts) == pytest.approx(case["joint_ll"], rel=1e-12)
    n = csep.n_test(float(rates.sum()), int(counts.sum()))
    assert (n.delta_1, n.delta_2) == pytest.approx(case["n_test"], rel=1e-9)
    for name, fn in (("s_test", csep.s_test), ("m_test", csep.m_test)):
        ours = fn(rates, counts, n_sims=REFERENCE["n_sims"], seed=case["seed"])
        ref = case[name]
        # Same statistic, exactly; the simulations use different random numbers.
        assert ours.observed_ll == pytest.approx(ref["observed_ll"], rel=1e-10)
        assert ours.quantile == pytest.approx(ref["quantile"], abs=0.04)
        assert ours.simulated_mean == pytest.approx(ref["mean"], rel=0.02)


def test_bits_per_earthquake() -> None:
    assert csep.bits_per_earthquake(-math.log(2) * 10, 5) == pytest.approx(-2.0)
    with pytest.raises(ValueError, match="no observed"):
        csep.bits_per_earthquake(-1.0, 0)
