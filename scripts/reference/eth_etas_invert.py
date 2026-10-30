"""ETH Zurich's ETAS (lmizrahi/etas, MIT) fitted on CB-02's and CB-03's training data.

The second ETAS referee (``scripts/ablations/eth_etas_referee.py`` scores it). ETH's package pins
its own pandas and geopandas stack, so it runs in a throwaway environment, never in Craton's
(D-033). First write the catalogue from Craton's environment, then fit, from the repo root:

    uv run python scripts/ablations/eth_etas_referee.py --export
    uv run --no-project --python 3.12 --with "git+https://github.com/lmizrahi/etas" \
        --with "pandas<3" python scripts/reference/eth_etas_invert.py

The fit copies Craton's ETAS (``craton.bench.etas``): INGV M2+ in central Italy's box, targets
from 2005 to each task's cut, every quake since 1990 a possible parent. It writes
``results/reference/eth_etas-CB-02.json`` and ``-CB-03.json``.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from etas.inversion import ETASParameterCalculation

logging.basicConfig(level=logging.INFO)
OUT = Path("results/reference")
CUTS = {"CB-02": "2016-08-24", "CB-03": "2009-01-01"}
BOX = [[41.0, 11.5], [44.0, 11.5], [44.0, 15.0], [41.0, 15.0]]
MC = 2.0

catalog = pd.read_csv(OUT / "central_italy_m2.csv")
catalog["time"] = pd.to_datetime(catalog["time"], format="ISO8601")

for task, cut in CUTS.items():
    calc = ETASParameterCalculation(
        {
            "catalog": catalog[catalog["time"] < pd.Timestamp(cut)],
            "auxiliary_start": "1990-01-01 00:00:00",
            "timewindow_start": "2005-01-01 00:00:00",
            "timewindow_end": f"{cut} 00:00:00",
            "theta_0": {
                "log10_mu": -6.0,
                "log10_k0": -2.5,
                "a": 1.8,
                "log10_c": -2.5,
                "omega": -0.02,
                "log10_tau": 3.5,
                "log10_d": -0.85,
                "gamma": 1.3,
                "rho": 0.66,
            },
            "mc": MC,
            "delta_m": 0.1,
            "coppersmith_multiplier": 25,
            "shape_coords": BOX,
            "name": f"eth-etas-{task}",
        }
    )
    calc.prepare()
    calc.invert()
    body = {
        "task": task,
        "cut": cut,
        "package": "lmizrahi/etas",
        "theta": {k: float(v) for k, v in calc.theta.items() if v is not None},
        "beta": float(calc.beta),
        "mc": MC,
        "n_hat": float(calc.n_hat),
        "iterations": int(calc.i),
    }
    (OUT / f"eth_etas-{task}.json").write_text(json.dumps(body, indent=2) + "\n")
    print(task, json.dumps(body["theta"]))
