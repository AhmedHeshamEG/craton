"""The P7 locks: calibration, the meter, physics, attacks, the guard, OOD, the referee, seals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from craton.bench import csep, etas
from craton.bench.data import DAY, Events
from craton.bench.locks import false_alarms, neighbourhood
from craton.bench.tasks import TASKS, GriddedForecast
from craton.sim import SIMULATORS, SimSpec
from craton.trust import calibration, guard, meter, ood, physics
from craton.trust.attacks import ATTACKS, site
from craton.trust.predictive import POISSON_K, Predictive, nb_cdf, nb_logpmf, nb_zero

SPEC = TASKS["CB-02"].spec  # type: ignore[attr-defined]


def _events(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, mag: np.ndarray) -> Events:
    n = len(t)
    return Events(
        np.asarray(t, float),
        np.asarray(lat, float),
        np.asarray(lon, float),
        np.full(n, 10.0),
        np.asarray(mag, float),
        np.asarray(["ML"] * n, dtype=np.str_),
        np.asarray([f"q{i}" for i in range(n)], dtype=np.str_),
    )


# ---------------------------------------------------------------- the negative binomial


@pytest.mark.parametrize("k", [0.5, 3.0, POISSON_K])
def test_negative_binomial_pmf_and_cdf_agree(k: float) -> None:
    n = np.arange(0, 200)
    pmf = np.exp(nb_logpmf(n, 4.0, k))
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.cumsum(pmf), nb_cdf(n, 4.0, k), atol=1e-9)
    assert nb_zero(np.array([4.0]), k)[0] == pytest.approx(pmf[0])
    mean = float((n * pmf).sum())
    assert mean == pytest.approx(4.0, rel=1e-4)


def test_poisson_limit_matches_scipy() -> None:
    n = np.arange(0, 30)
    assert np.allclose(np.exp(nb_logpmf(n, 3.3, POISSON_K)), stats.poisson.pmf(n, 3.3))
    assert np.allclose(nb_cdf(n, 3.3, POISSON_K), stats.poisson.cdf(n, 3.3))


# ---------------------------------------------------------------- calibration


def test_reliability_of_an_honest_forecast_is_good() -> None:
    rng = np.random.default_rng(0)
    p = 10 ** rng.uniform(-3, -0.5, 400_000)
    hit = rng.random(len(p)) < p
    rel = calibration.reliability(p, hit)
    assert rel.log_error() < 0.1  # sampling noise: ~130 hits in the lowest group
    assert rel.ece() < 0.005
    biased = calibration.reliability(p / 3, hit)
    assert biased.log_error() == pytest.approx(math.log2(3), abs=0.15)


def test_pit_is_uniform_for_the_right_model_and_u_shaped_for_poisson_on_clusters() -> None:
    rng = np.random.default_rng(1)
    mean = rng.uniform(0.5, 20, 4000)
    pois = rng.poisson(mean)
    u = calibration.randomized_pit(pois, mean[None], POISSON_K, 0)
    assert calibration.ks_uniform(u) < 0.03
    k = 0.8
    clustered = rng.negative_binomial(k, k / (k + mean))
    u2 = calibration.randomized_pit(clustered, mean[None], POISSON_K, 0)
    assert calibration.ks_uniform(u2) > 0.15
    assert np.mean(u2 < 0.05) + np.mean(u2 > 0.95) > 0.4
    k_hat = calibration.fit_dispersion(clustered.astype(float), mean)
    assert k_hat == pytest.approx(k, rel=0.15)
    u3 = calibration.randomized_pit(clustered, mean[None], k_hat, 0)
    assert calibration.ks_uniform(u3) < 0.04


def test_crps_matches_the_definition() -> None:
    mean = np.array([[2.0, 7.0]])
    obs = np.array([1.0, 9.0])
    got = calibration.crps_counts(obs, mean, POISSON_K)
    j = np.arange(0, 200)
    for d in range(2):
        f = stats.poisson.cdf(j, mean[0, d])
        want = np.sum((f - (obs[d] <= j)) ** 2)
        assert got[d] == pytest.approx(want, rel=1e-6)


def test_predictive_reads_members_and_dispersion() -> None:
    g = SPEC.grid
    shares = csep.gutenberg_richter_bins(1.0, 1.5, g)
    shares = shares / shares.sum()
    members = np.full((3, 2, g.n_cells), 1e-3)
    members[0] *= 2
    pred = Predictive(GriddedForecast(members.mean(axis=0), shares), members, 5.0, 2.0)
    first = int(g.mag_bin_of(np.array([3.0]))[0])
    assert pred.member_expected(first).shape == (3, 2, g.n_cells)
    p = pred.occurrence(first)
    assert np.all((p > 0) & (p < 1))
    poisson = Predictive(GriddedForecast(members.mean(axis=0), shares))
    assert np.all(poisson.occurrence(first) >= p - 1e-12)  # clustering lowers P(at least one)


# ---------------------------------------------------------------- the meter


def test_a_sharp_forecast_watches_less() -> None:
    g = SPEC.grid
    days = 50
    rng = np.random.default_rng(2)
    cells = rng.integers(0, g.n_cells, 40)
    day = np.arange(40)  # one quake a day
    flat = np.full((days, g.n_cells), 1e-8)
    sharp = flat.copy()
    sharp[day, cells] = 0.5
    area = meter.cell_areas(g)
    parts_flat = [meter.MeterPart(flat, day, cells, np.ones(g.n_mags) / g.n_mags, 25, area)]
    parts_sharp = [meter.MeterPart(sharp, day, cells, np.ones(g.n_mags) / g.n_mags, 25, area)]
    assert meter.space_time_share(parts_sharp) < 0.002
    assert meter.space_time_share(parts_flat) == 1.0  # all tied: watch everything
    r_sharp, nominal = meter.space_radius(parts_sharp)
    r_flat, _ = meter.space_radius(parts_flat)
    assert r_sharp < 10 < r_flat
    assert 0 <= nominal <= 1


def test_magnitude_width_of_gutenberg_richter() -> None:
    g = SPEC.grid
    shares = csep.gutenberg_richter_bins(1.0, 1.5, g)
    part = meter.MeterPart(
        np.ones((1, g.n_cells)), np.array([0]), np.array([5]), shares, 25, meter.cell_areas(g)
    )
    # b = 1 above M4: the 5th to 95th percentile spans log10(19) ≈ 1.28 magnitude units
    assert meter.magnitude_width([part], g) == pytest.approx(1.3, abs=0.15)


# ---------------------------------------------------------------- physics


def test_truncated_gutenberg_richter_keeps_the_budget_and_unbounded_breaks_it() -> None:
    g = SPEC.grid
    days = 60
    starts = SPEC.window.t0 + DAY * np.arange(days)
    calm = _events(
        np.array([starts[0] - 400 * DAY]), np.array([42.0]), np.array([13.0]), np.array([2.0])
    )
    gr = csep.gutenberg_richter_bins(1.0, 1.5, g)
    rates = np.full((days, g.n_cells), 1.5 / g.n_cells * 10 ** (1.5 - 1.5))  # 1.5 M1.5+/day
    open_top = GriddedForecast(rates, gr / gr.sum())
    cut = np.where(g.magnitudes <= 7.3 + 1e-9, gr, 0.0)
    capped = GriddedForecast(rates, cut / cut.sum())
    a = physics.check_forecast(open_top, g, calm, starts)
    b = physics.check_forecast(capped, g, calm, starts)
    assert b.moment_rate < a.moment_rate
    assert b.over_budget == 0
    assert b.violations == 0
    assert a.gr_rising == 0  # the open last bin is not counted as a rise


def test_negative_rates_and_rising_shares_are_violations() -> None:
    g = SPEC.grid
    starts = SPEC.window.t0 + DAY * np.arange(3)
    rates = np.full((3, g.n_cells), 1e-3)
    rates[1, 4] = -1.0
    rising = np.linspace(0.1, 1.0, g.n_mags)
    c = physics.check_forecast(
        GriddedForecast(rates, rising / rising.sum()), g, _events(*(np.zeros(0),) * 4), starts
    )
    assert c.bad_rates == 1
    assert c.gr_rising == 1
    assert c.violations >= 2


def test_omori_probe_counts_rises() -> None:
    g = SPEC.grid
    shares = csep.gutenberg_richter_bins(1.0, 1.5, g)
    falling = np.exp(-np.arange(10))[:, None] * np.ones((10, g.n_cells))
    assert physics.omori_violations(GriddedForecast(falling, shares / shares.sum()), 15)[0] == 0
    bumpy = falling.copy()
    bumpy[5] *= 10
    assert physics.omori_violations(GriddedForecast(bumpy, shares / shares.sum()), 15)[0] == 1


# ---------------------------------------------------------------- attacks and the guard


def _calm_catalogue(seed: int = 0) -> Events:
    return SIMULATORS["etas"].simulate(SimSpec((41.0, 44.0, 11.5, 15.0), 3 * 365, t0=1.2e9), seed)


def test_guard_drops_duplicates_and_nothing_else() -> None:
    ev = _calm_catalogue()
    out, rep = guard.clean(ev, ev.time[0])
    assert rep.duplicates == 0
    assert len(out) == len(ev)
    dup = _events(ev.time[:50] + 1.0, ev.lat[:50] + 0.001, ev.lon[:50], ev.mag[:50])
    both = Events.concat([ev, dup])
    out2, rep2 = guard.clean(both, ev.time[0])
    assert rep2.duplicates == 50
    assert len(out2) == len(ev)


def test_guard_takes_off_a_network_wide_magnitude_shift() -> None:
    rng = np.random.default_rng(4)
    n = 30_000
    t = np.sort(rng.uniform(0, 1000 * DAY, n)) + 1.2e9
    m = np.round(0.5 + rng.exponential(1 / np.log(10), n), 1)
    ev = _events(t, rng.uniform(41, 44, n), rng.uniform(11.5, 15, n), m)
    _, clean_rep = guard.clean(ev, t[0] + 400 * DAY)
    assert clean_rep.shifted_quakes == 0
    hit = (t >= t[0] + 600 * DAY) & (t < t[0] + 630 * DAY)
    shifted = Events(
        ev.time,
        ev.lat,
        ev.lon,
        ev.depth_km,
        np.round(ev.mag + 0.4 * hit, 1),
        ev.mag_type,
        ev.token_hash,
    )
    out, rep = guard.clean(shifted, t[0] + 400 * DAY)
    assert rep.shifted_stretches >= 4
    assert np.mean(np.abs(out.mag[hit] - ev.mag[hit]) < 0.15) > 0.6


def test_attacks_are_registered_and_placed_away_from_laquila() -> None:
    assert {"gps_step", "duplicates", "magnitude_shift", "b_drop", "fake_swarm"} <= set(ATTACKS)
    assert {a.name for a in ATTACKS.values() if a.counted} == {
        "gps_step",
        "duplicates",
        "magnitude_shift",
    }
    for seed in range(5):
        s = site(seed)
        assert s.t1 - s.t0 == pytest.approx(30 * DAY)
        from craton.trust.attacks import LAQUILA, km

        assert km(LAQUILA[0], LAQUILA[1], s.lat, s.lon) >= 60


def test_swarm_and_duplicates_add_quakes_shift_moves_magnitudes() -> None:
    ev = _calm_catalogue()
    s = site(0)
    s = type(s)(s.lat, s.lon, ev.time[0] + 100 * DAY, ev.time[0] + 130 * DAY)
    rng = np.random.default_rng(0)
    from craton.bench.data import Gnss

    g = Gnss(
        np.zeros(0),
        np.zeros(0, np.str_),
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros(0, np.str_),
    )
    swarm, _ = ATTACKS["fake_swarm"].apply(ev, g, s, rng)
    assert len(swarm) == len(ev) + 150
    dup, _ = ATTACKS["duplicates"].apply(ev, g, s, rng)
    inside = int(((ev.time >= s.t0) & (ev.time < s.t1)).sum())
    assert len(dup) == len(ev) + inside
    shift, _ = ATTACKS["magnitude_shift"].apply(ev, g, s, rng)
    assert np.sum(np.abs(shift.mag - ev.mag) > 0.3) == inside


def test_alarms_need_a_neighbourhood() -> None:
    g = SPEC.grid
    x = np.zeros((1, g.n_cells))
    x[0, g.n_lon * 5 + 5] = 1.0
    nb = neighbourhood(x, g.n_lat, g.n_lon)
    assert nb.sum() == pytest.approx(9.0)
    assert nb.max() == 1.0
    shares = csep.gutenberg_richter_bins(1.0, 1.5, g)
    rates = np.zeros((len(SPEC.day_starts), g.n_cells))
    rates[3, g.n_lon * 5 + 5] = 50.0
    fc = GriddedForecast(rates, shares / shares.sum())
    lat, lon = g.cell_centres()
    real = _events(
        np.array([SPEC.day_starts[3] + 100]),
        lat[[g.n_lon * 5 + 6]],
        lon[[g.n_lon * 5 + 6]],
        np.array([4.2]),
    )
    fa, alarms = false_alarms(fc, real, SPEC)
    # the M4.2 sits one cell east: the three alarmed cells two columns away can't see it
    assert alarms == 9
    assert fa == 3


# ---------------------------------------------------------------- OOD


def test_auroc_and_knn() -> None:
    assert ood.auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert ood.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(500, 4))
    near = ood.knn_distance(bank, rng.normal(size=(100, 4)))
    far = ood.knn_distance(bank, rng.normal(size=(100, 4)) + 6)
    assert ood.auroc(near, far) > 0.99
    self_d = ood.knn_distance(bank, bank, self_bank=True)
    assert np.all(self_d > 0)


def test_translation_keeps_distances() -> None:
    ev = _events(
        np.arange(3.0), np.array([37.0, 37.2, 37.1]), np.array([15.0, 15.1, 15.3]), np.ones(3)
    )
    moved = ood.translate(ev)
    assert float(np.median(moved.lat)) == pytest.approx(ood.CENTRE[0])

    def d(e: Events, i: int, j: int) -> float:
        dy = (e.lat[i] - e.lat[j]) * 111.195
        dx = (e.lon[i] - e.lon[j]) * 111.195 * math.cos(math.radians((e.lat[i] + e.lat[j]) / 2))
        return math.hypot(dx, dy)

    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert d(moved, i, j) == pytest.approx(d(ev, i, j), rel=0.02)
    assert len(ood.fingerprint(_calm_catalogue())) == 12


def test_no_detector_is_a_coin_flip() -> None:
    q = ood.OodQueries(np.zeros(10), np.zeros(5), np.zeros(7), _events(*(np.zeros(0),) * 4))
    s = ood.NoDetector().ood_scores(q, None, 0)
    assert s.summary()["auroc"] == 0.5


def test_etas_planet_is_marked_synthetic() -> None:
    ev = _calm_catalogue()
    assert set(ev.mag_type.tolist()) == {"sim"}
    assert ev.token_hash[0].startswith("sim-etas-")
    assert np.all((ev.lat >= 41) & (ev.lat < 44))
    assert isinstance(SIMULATORS["etas"].params, etas.EtasParams)  # type: ignore[attr-defined]


def test_day_dispersion_recovers_a_mean_dependent_size() -> None:
    rng = np.random.default_rng(5)
    mean = 10 ** rng.uniform(-1, 2, 6000)
    k = 1.5 * mean**0.7
    n = rng.negative_binomial(k, k / (k + mean)).astype(float)
    kappa, gamma = calibration.fit_day_dispersion(n, mean)
    assert gamma == pytest.approx(0.7, abs=0.1)
    assert kappa == pytest.approx(1.5, rel=0.3)
    pois = rng.poisson(mean).astype(float)
    assert calibration.fit_day_dispersion(pois, mean)[0] > 50
