"""Locked forecasts: written and hashed before the day they forecast, graded after it.

The last of the eight locks, and the only one nobody can argue with. A forecast scored on data
it has already seen can always be made to look good. A forecast published *before* the day,
with a hash that proves it was never touched, can only be right or wrong.

    forecasts/<region>/<yyyy>/<yyyy-mm-dd>.json   one per day, sealed (sha256 of its content)
    forecasts/<region>/ledger.jsonl               one line per seal, append-only
    forecasts/<region>/grades/<yyyy-mm-dd>.json   written once the day's catalogue settles
    forecasts/README.md                           the scoreboard of the live forecasts

A forecast is pushed to the public repository before its day begins (UTC), so GitHub's own
record of the push is the timestamp. Missed days stay missed: nothing is ever backfilled.
Graded with the same Poisson log-likelihood as CB-02, at M3+ per 0.1° cell, Craton and the
ETAS floor locked side by side in the same file.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

from craton.bench import csep
from craton.bench.data import DAY, Events
from craton.spec.timeaxis import to_seconds
from craton.trust.predictive import nb_logpmf

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

SCHEMA = "craton-locked-forecast/1"
#: Days left for the catalogue to settle (INGV revises locations and magnitudes) before grading.
GRADE_AFTER_DAYS = 3


def forecasts_dir(root: Path, region: str) -> Path:
    return root / "forecasts" / region


def day_path(root: Path, region: str, day: str) -> Path:
    return forecasts_dir(root, region) / day[:4] / f"{day}.json"


def grade_path(root: Path, region: str, day: str) -> Path:
    return forecasts_dir(root, region) / "grades" / f"{day}.json"


def canonical(body: dict[str, Any]) -> str:
    """The exact text a seal covers: every field except the seal, keys sorted, no spaces."""
    return json.dumps(
        {k: v for k, v in body.items() if k != "sha256"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}


def intact(body: dict[str, Any]) -> bool:
    return bool(body.get("sha256")) and seal(body)["sha256"] == body["sha256"]


def sig(x: Any, digits: int = 6) -> Any:
    """Round to ``digits`` significant digits, so the stored number is the graded number."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.vectorize(lambda v: float(f"{v:.{digits - 1}e}") if v else 0.0)(arr)
    return out.tolist()


def write_locked(root: Path, region: str, body: dict[str, Any]) -> Path:
    """Seal and write a forecast, and append it to the ledger. Refuses to overwrite."""
    path = day_path(root, region, body["day"])
    if path.exists():
        raise FileExistsError(f"{path} is already locked; a locked forecast is never rewritten")
    sealed = seal(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sealed, indent=1, sort_keys=True) + "\n", "utf-8", newline="\n")
    ledger = forecasts_dir(root, region) / "ledger.jsonl"
    line = {"day": body["day"], "issued": body["issued"], "sha256": sealed["sha256"]}
    with ledger.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return path


def read_locked(path: Path) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if not intact(body):
        raise ValueError(f"{path}: the seal does not match the content")
    return body


def poisson_ll(rates: FloatArray, counts: FloatArray) -> float:
    lam = np.maximum(rates, 1e-300)
    return float(np.sum(counts * np.log(lam) - rates - gammaln(counts + 1)))


def grade(body: dict[str, Any], events: Events, grid: csep.Grid) -> dict[str, Any]:
    """Score a locked forecast against the quakes of its day (M3+, per cell)."""
    t0 = to_seconds(body["day"] + "T00:00:00Z")
    day = events.window(t0, t0 + DAY).above(body["magnitude"])
    cell = grid.cell_of(day.lat, day.lon)
    counts = np.bincount(cell[cell >= 0], minlength=grid.n_cells).astype(np.float64)
    n = float(counts.sum())
    craton = np.asarray(body["craton"]["rates"], dtype=np.float64)
    etas = np.asarray(body["etas"]["rates"], dtype=np.float64)
    ll_c, ll_e = poisson_ll(craton, counts), poisson_ll(etas, counts)
    total = float(craton.sum())
    k = float(body["craton"]["day_k"]) * total ** float(body["craton"].get("day_gamma", 0.0))
    return {
        "day": body["day"],
        "sha256": body["sha256"],
        "graded": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quakes": n,
        "largest": float(day.mag.max()) if len(day) else None,
        "craton_ll": ll_c,
        "etas_ll": ll_e,
        "gain_bits": (ll_c - ll_e) / math.log(2),
        "craton_expected": float(craton.sum()),
        "etas_expected": float(etas.sum()),
        "craton_day_log_score": float(nb_logpmf(n, total, k)),
        "abstained": bool(body["craton"]["abstain"]),
    }


def page(root: Path, region: str) -> Path:
    """forecasts/README.md: every locked day, graded or waiting, and the running total."""
    folder = forecasts_dir(root, region)
    locked = sorted(folder.glob("[0-9][0-9][0-9][0-9]/*.json"))
    grades = {p.stem: json.loads(p.read_text("utf-8")) for p in (folder / "grades").glob("*.json")}
    rows, total, quakes, graded = [], 0.0, 0.0, 0
    for p in reversed(locked):
        body = json.loads(p.read_text("utf-8"))
        g = grades.get(body["day"])
        exp_c, exp_e = sum(body["craton"]["rates"]), sum(body["etas"]["rates"])
        flag = " · out of familiar territory" if body["craton"]["abstain"] else ""
        sealed = f"`{body['sha256'][:12]}`{flag}"
        if g is None:
            rows.append(f"| {body['day']} | {exp_c:.2f} | {exp_e:.2f} | · | · | {sealed} |")
        else:
            rows.append(
                f"| {body['day']} | {exp_c:.2f} | {exp_e:.2f} | {g['quakes']:.0f} | "
                f"{g['gain_bits']:+.2f} | {sealed} |"
            )
    for g in grades.values():
        total += g["gain_bits"]
        quakes += g["quakes"]
        graded += 1
    per = f"{total / quakes:+.3f} bits per quake" if quakes else "no M3+ quake yet"
    text = f"""<!-- Generated by `craton live page`. Do not edit by hand. -->

# Live forecasts · {region.replace("_", " ")}

Every day before midnight UTC, Craton writes tomorrow's forecast for central Italy, seals it
with a SHA-256 hash and pushes it here. Nobody can change it afterwards: `craton check` refuses
any edit to a locked file, and GitHub keeps the time of the push. Three days later, once INGV
has settled the catalogue, the day is graded: Craton against the ETAS floor locked in the same
file, Poisson log-likelihood of the M3+ quakes per 0.1° cell. A gain above zero means Craton
gave the real quakes more probability than ETAS did.

These are forecasts: probabilities over the next day. Never a prediction of a quake.

**{len(locked)} days locked · {graded} graded · {quakes:.0f} M3+ quakes · Craton vs ETAS:
{total:+.2f} bits in total, {per}.**

| Day (UTC) | Craton expects M3+ | ETAS expects | Came | Gain (bits) | Seal |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Check a seal yourself: `craton live verify`. How it works: `lessons/P7.md`.
"""
    out = root / "forecasts" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, "utf-8", newline="\n")
    return out
