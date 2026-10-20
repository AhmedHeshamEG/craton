"""Reference CSEP numbers from pyCSEP itself, for tests/test_bench_csep.py.

Craton grades forecasts with its own NumPy code (craton/bench/csep.py). This script feeds a
seeded random forecast and catalogue to pyCSEP 0.8.0's Poisson routines and writes what they
say, so the test can compare. pyCSEP pulls cartopy, rasterio and obspy, so it runs in a
throwaway environment, never in Craton's. From the repo root:

    uv run --no-project --python 3.12 --with pycsep==0.8.0 \
        python scripts/reference/csep_reference.py

It writes tests/fixtures/csep/reference.json.
"""

import json

import numpy as np
from csep.core import poisson_evaluations as pe
from csep.utils.stats import poisson_joint_log_likelihood_ndarray

OUT = "tests/fixtures/csep/reference.json"
N_SIMS = 4000


def case(seed: int, cells: int, mags: int, scale: float, bias: float) -> dict:
    rng = np.random.default_rng(seed)
    rates = rng.gamma(0.7, scale, size=(cells, mags))
    # The catalogue comes from a slightly different forecast, so the tests have work to do.
    truth = rates * rng.lognormal(0.0, bias, size=rates.shape)
    counts = rng.poisson(truth).astype(float)
    idx = np.nonzero(counts.ravel())
    ll = poisson_joint_log_likelihood_ndarray(
        np.log(rates.ravel()[idx]) * counts.ravel()[idx], counts.ravel()[idx], rates.sum()
    )
    d1, d2 = pe._number_test_ndarray(rates.sum(), counts.sum())
    s_q, s_obs, s_sims = pe._poisson_likelihood_test(
        rates.sum(axis=1),
        counts.sum(axis=1),
        num_simulations=N_SIMS,
        seed=seed,
        use_observed_counts=True,
        normalize_likelihood=True,
    )
    m_q, m_obs, m_sims = pe._poisson_likelihood_test(
        rates.sum(axis=0),
        counts.sum(axis=0),
        num_simulations=N_SIMS,
        seed=seed,
        use_observed_counts=True,
        normalize_likelihood=True,
    )
    return {
        "seed": seed,
        "rates": rates.tolist(),
        "counts": counts.tolist(),
        "joint_ll": float(ll),
        "n_test": [float(d1), float(d2)],
        "s_test": {
            "quantile": float(s_q),
            "observed_ll": float(s_obs),
            "mean": float(np.mean(s_sims)),
        },
        "m_test": {
            "quantile": float(m_q),
            "observed_ll": float(m_obs),
            "mean": float(np.mean(m_sims)),
        },
    }


def main() -> None:
    import csep

    cases = [
        case(1, 40, 8, 0.4, 0.2),
        case(2, 60, 10, 0.15, 0.8),
        case(3, 25, 6, 1.5, 0.05),
    ]
    out = {"pycsep": csep.__version__, "n_sims": N_SIMS, "cases": cases}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
