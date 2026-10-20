import math

import numpy as np
import pytest

from craton.bench import csep, etas

GRID = csep.Grid(41.0, 44.0, 11.5, 15.0, m_min=2.0)
PROJ = etas.Projection.for_grid(GRID)
BOX = etas.box_km(GRID, PROJ)
TRUE = etas.EtasParams(mu=1.0, k=0.2, a=1.5, c=0.01, p=1.15, d=1.0, gamma=1.2, q=1.6)


@pytest.fixture(scope="module")
def planet() -> etas.Catalog:
    """A synthetic catalogue where ETAS is the truth: 2,400 days of a fake central Italy."""
    return etas.simulate(TRUE, 1.0, 2.0, 2400, BOX, seed=11)


@pytest.fixture(scope="module")
def fitted(planet: etas.Catalog) -> etas.EtasFit:
    return etas.fit(planet, 2.0, etas.FitWindow(90, 2000), BOX, 0.0, seed=0, mag_step=0.0)


def test_params_round_trip() -> None:
    assert etas.EtasParams.from_vector(TRUE.to_vector()).as_dict() == pytest.approx(TRUE.as_dict())


def test_omori_integral() -> None:
    h = etas.omori_integral(np.array([-1.0, 0.0, 1e9]), 0.01, 1.2)
    assert h[0] == 0
    assert h[1] == 0
    assert h[2] == pytest.approx(1.0, abs=0.03)


def test_simulation_has_the_right_size(planet: etas.Catalog) -> None:
    # Expected total = background / (1 - branching ratio), minus what lands outside the box.
    n = etas.mean_productivity(TRUE, 1.0, 2.0)
    expected = TRUE.mu * 2400 / (1 - n)
    assert 0.75 * expected < len(planet) < 1.1 * expected
    assert np.all(np.diff(planet.t) >= 0)


def test_fit_recovers_the_planted_parameters(fitted: etas.EtasFit) -> None:
    got = fitted.params
    assert fitted.converged
    assert got.mu == pytest.approx(TRUE.mu, rel=0.25)
    assert got.a == pytest.approx(TRUE.a, abs=0.25)
    assert got.p == pytest.approx(TRUE.p, abs=0.1)
    assert got.gamma == pytest.approx(TRUE.gamma, abs=0.3)
    assert fitted.b == pytest.approx(1.0, abs=0.05)
    # Aftershocks that land outside the box are never seen, so the fit sees fewer of them.
    truth = etas.mean_productivity(TRUE, 1.0, 2.0)
    assert truth - 0.2 < fitted.branching_ratio <= truth + 0.05


def test_background_integrates_to_one(planet: etas.Catalog) -> None:
    bg = etas.Background(planet.x, planet.y, np.ones(len(planet)), BOX)
    assert bg.cell_mass(GRID, PROJ).sum() == pytest.approx(1.0, abs=1e-9)
    x0, x1, y0, y1 = BOX
    xs, ys = np.meshgrid(np.linspace(x0, x1, 81), np.linspace(y0, y1, 81))
    dens = bg.density(xs.ravel(), ys.ravel())
    assert dens.mean() * bg.area == pytest.approx(1.0, rel=0.05)


def test_spatial_matrix_keeps_mass_inside_and_loses_it_outside() -> None:
    lat = np.array([42.5, 41.0])  # the centre, and on the southern edge
    lon = np.array([13.25, 13.25])
    x, y = PROJ.to_km(lat, lon)
    cat = etas.Catalog(np.zeros(2), x, y, np.array([2.0, 2.0]))
    rows = np.asarray(etas.spatial_matrix(TRUE, cat, 2.0, GRID, PROJ).sum(axis=1)).ravel()
    assert rows[0] == pytest.approx(1.0, abs=1e-9)
    assert 0.3 < rows[1] < 0.7


def test_cascade_factor() -> None:
    f = etas.cascade_factor(TRUE, 1.0, 2.0, 1.0)
    assert 1.0 < f < 2.0
    weak = etas.cascade_factor(etas.EtasParams(1, 1e-4, 1, 0.01, 1.2, 1, 1, 1.5), 1.0, 2.0, 1.0)
    assert weak == pytest.approx(1.0, abs=1e-3)


def test_daily_forecast_is_calibrated_where_etas_is_true(
    planet: etas.Catalog, fitted: etas.EtasFit
) -> None:
    days = np.arange(2000, 2400, dtype=float)
    fc = etas.forecast_days(fitted, planet, days, GRID, PROJ)
    assert fc.rates.shape == (400, GRID.n_cells)
    observed = np.array([np.sum((planet.t >= d) & (planet.t < d + 1)) for d in days])
    expected = fc.rates.sum(axis=1)
    assert expected.sum() == pytest.approx(observed.sum(), rel=0.15)
    misses = np.mean(
        [csep.n_test(e, int(o)).delta_1 < 0.025 for e, o in zip(expected, observed, strict=True)]
    )
    assert misses < 0.1


def test_projection_scale() -> None:
    x, y = PROJ.to_km(np.array([43.5]), np.array([14.25]))
    assert y[0] == pytest.approx(110.57)
    assert x[0] == pytest.approx(111.32 * math.cos(math.radians(42.5)))
