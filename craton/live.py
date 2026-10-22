"""The live loop: one locked forecast a day for central Italy, from a frozen model.

The **bundle** is everything the daily forecast needs, frozen at a release: the Field members
and ETAS fits of CB-02's three seeds (the exact models on the scoreboard), their calibration,
and the out-of-distribution bank. It is built once on a machine with the data
(``craton live bundle``), packed as ``craton-live-<region>-v<version>.tar.gz`` and attached to
the GitHub release, so the forecasts are made by a model anyone can download.

Every night the **live job** (``.github/workflows/live.yml``, ``craton live run``) fetches the
INGV catalogue (cached month by month), runs the bundle for the next UTC day, seals the file
(``craton.trust.lock``), grades every day old enough to grade, regenerates the page, and
pushes. It reads the catalogue up to the moment it runs (about 22:30 UTC), so the last hour
and a half before midnight is unseen: the price of being early.
"""

from __future__ import annotations

import io
import json
import math
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

from craton import __version__
from craton.adapters.http import Fetcher
from craton.adapters.ingv import EVENT_TYPES, _body_lines, fetch_window
from craton.adapters.region import Region, load_region
from craton.bench import etas
from craton.bench.data import DAY, Events
from craton.bench.tasks import TASKS, BenchContext, forecast_grid
from craton.field.config import FieldConfig, field_hash
from craton.field.forecaster import (
    M_SCORED,
    FieldForecaster,
    build_days,
    magnitude_shares,
    predict_members,
)
from craton.spec.instruments import MAGNITUDE_TYPES
from craton.spec.timeaxis import to_iso, to_seconds
from craton.trust import guard, lock, ood

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

LIVE_VERSION = "live-1"
REGION = "central_italy"
BUNDLE_TASK = "CB-02"
M_ALERT = 4.0
#: Quakes in a context, as in CB-06.
CONTEXT = 1024
#: Months this recent are fetched again on every run (INGV keeps revising them).
REFRESH_MONTHS = 2


def bundle_name(region: str = REGION) -> str:
    return f"craton-live-{region}-v{__version__}.tar.gz"


# ---------------------------------------------------------------- the catalogue


def _month_starts(start: float, until: float) -> list[float]:
    first = datetime.fromtimestamp(start, UTC).replace(day=1, hour=0, minute=0, second=0)
    out: list[float] = []
    y, m = first.year, first.month
    while True:
        t = to_seconds(datetime(y, m, 1, tzinfo=UTC))
        if t >= until:
            return out
        out.append(t)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def parse_lines(lines: list[str], region: Region) -> Events:
    """Quakes from INGV's FDSN text lines, by the same rules as the ``ingv_catalog`` adapter."""
    rows = []
    for line in lines:
        f = line.split("|")
        if len(f) != 14 or not f[10].strip() or f[9] not in MAGNITUDE_TYPES:
            continue
        if f[13] not in EVENT_TYPES:
            continue
        try:
            t = to_seconds(f[1] + "Z")
            lat, lon, depth, mag = float(f[2]), float(f[3]), float(f[4]), float(f[10])
        except ValueError:
            continue
        if region.box.contains(lat, lon):
            rows.append((t, lat, lon, depth, mag, f[9], f"ingv-{f[0]}"))
    cols = list(zip(*rows, strict=True)) if rows else [()] * 7
    ev = Events(
        np.asarray(cols[0], float),
        np.asarray(cols[1], float),
        np.asarray(cols[2], float),
        np.asarray(cols[3], float),
        np.asarray(cols[4], float),
        np.asarray(cols[5], dtype=np.str_),
        np.asarray(cols[6], dtype=np.str_),
    )
    return ev.select(np.argsort(ev.time, kind="stable"))


def fetch_catalogue(cache: Path, region: Region, http: Fetcher, now: float, log: Any) -> Events:
    """Every quake of the region from its start to ``now``: one cached file per month,
    the last ``REFRESH_MONTHS`` fetched again every time."""
    cache.mkdir(parents=True, exist_ok=True)
    starts = _month_starts(region.t_start, now)
    lines: list[str] = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else now
        path = cache / f"{(to_iso(a) or '')[:7]}.txt"
        if not path.is_file() or i >= len(starts) - REFRESH_MONTHS:
            data = fetch_window(http, region.box, a, b)
            path.write_bytes(data)
            log(f"fetched {path.name}")
        lines += _body_lines(path.read_bytes())
    ev = parse_lines(lines, region)
    _, first = np.unique(ev.token_hash, return_index=True)  # a month boundary can repeat one
    return ev.select(np.sort(first)).window(-math.inf, now)


# ---------------------------------------------------------------- the bundle


@dataclass
class Bundle:
    config: FieldConfig
    fits: list[etas.EtasFit]
    states: list[list[dict[str, torch.Tensor]]]
    calibration: list[dict[str, Any]]
    ood_bank: FloatArray  # standardised fingerprints of the training contexts
    ood_mu: FloatArray
    ood_sd: FloatArray
    ood_threshold: float
    meta: dict[str, Any]


def _fit_record(fit: etas.EtasFit) -> dict[str, Any]:
    bg = fit.background
    return {
        "params": fit.params.as_dict(),
        "mc": fit.mc,
        "b": fit.b,
        "t_ref": fit.t_ref,
        "window": [fit.window.t_start, fit.window.t_end],
        "n_targets": fit.n_targets,
        "log_likelihood": fit.log_likelihood,
        "converged": fit.converged,
        "background": {"box_km": list(bg.box_km), "h": bg.h, "floor": bg.floor},
    }


def _fit_from(rec: dict[str, Any], arrays: Any) -> etas.EtasFit:
    bgr = rec["background"]
    bg = etas.Background(
        np.asarray(arrays["x"]),
        np.asarray(arrays["y"]),
        np.asarray(arrays["w"]),
        (bgr["box_km"][0], bgr["box_km"][1], bgr["box_km"][2], bgr["box_km"][3]),
        bgr["h"],
        bgr["floor"],
    )
    return etas.EtasFit(
        etas.EtasParams(**rec["params"]),
        bg,
        rec["mc"],
        rec["b"],
        rec["t_ref"],
        etas.FitWindow(*rec["window"]),
        rec["n_targets"],
        rec["log_likelihood"],
        rec["converged"],
    )


def build_bundle(ctx: BenchContext, out: Path, log: Any, field_config: str = "mini") -> Path:
    """Freeze CB-02's Field and ETAS (every seed of the config), calibration and the OOD bank
    into ``out``, then pack it next to it. Needs the data and trains nothing new if the
    checkpoints exist."""
    from craton.bench.baselines import Etas
    from craton.bench.locks import reference_midnights

    ff = FieldForecaster(field_config, log=log)
    cfg = ff.config
    spec = TASKS[BUNDLE_TASK].spec  # type: ignore[attr-defined]
    out.mkdir(parents=True, exist_ok=True)
    fits, cals = [], []
    for seed in cfg.train.seeds:
        fit = Etas().fit(spec, ctx, seed)
        states, _ = ff.states(spec, ctx, seed, fit)
        cals.append(ff.calibration(spec, ctx, seed, fit, states))
        fits.append(_fit_record(fit))
        bg = fit.background
        np.savez_compressed(out / f"etas-s{seed}.npz", x=bg.x, y=bg.y, w=bg.w)
        torch.save({"members": states}, out / f"field-s{seed}.pt")
    pre = ctx.train_events("pretrain")
    mids = reference_midnights(ctx)
    prints = np.array([ood.fingerprint(c) for c in ood.last_quakes(pre, mids, CONTEXT)])
    mu, sd = prints.mean(axis=0), prints.std(axis=0) + 1e-9
    bank = (prints - mu) / sd
    self_d = ood.knn_distance(bank, bank, self_bank=True)
    np.savez_compressed(out / "ood.npz", bank=bank, mu=mu, sd=sd)
    manifest = json.loads((ctx.split_dir(spec.fit_split) / "manifest.json").read_text("utf-8"))
    meta = {
        "version": LIVE_VERSION,
        "craton": __version__,
        "region": REGION,
        "task": BUNDLE_TASK,
        "split": spec.fit_split,
        "split_id": manifest["id"],
        "snapshot": ctx.snapshot_id,
        "seeds": list(cfg.train.seeds),
        "members_per_seed": cfg.train.members,
        "field_config": cfg.model_dump(mode="json"),
        "field_hash": field_hash(cfg),
        "etas": fits,
        "calibration": cals,
        "ood_threshold": float(np.quantile(self_d, ood.ABSTAIN_QUANTILE)),
        "ood_contexts": len(bank),
        "made": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / "bundle.json").write_text(json.dumps(meta, indent=1) + "\n", "utf-8")
    tar = out.parent / bundle_name()
    with tarfile.open(tar, "w:gz") as tf:
        for p in sorted(out.iterdir()):
            tf.add(p, arcname=p.name)
    return tar


def load_bundle(path: Path) -> Bundle:
    """From a bundle folder or its .tar.gz."""
    files: dict[str, bytes] = {}
    if path.is_dir():
        files = {p.name: p.read_bytes() for p in path.iterdir()}
    else:
        with tarfile.open(path, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and "/" not in m.name and ".." not in m.name:
                    f = tf.extractfile(m)
                    assert f is not None
                    files[m.name] = f.read()
    meta = json.loads(files["bundle.json"])
    cfg = FieldConfig.model_validate(meta["field_config"])
    fits, states = [], []
    for seed, rec in zip(meta["seeds"], meta["etas"], strict=True):
        fits.append(_fit_from(rec, np.load(io.BytesIO(files[f"etas-s{seed}.npz"]))))
        body = torch.load(
            io.BytesIO(files[f"field-s{seed}.pt"]), map_location="cpu", weights_only=True
        )
        states.append(list(body["members"]))
    o = np.load(io.BytesIO(files["ood.npz"]))
    return Bundle(
        cfg,
        fits,
        states,
        meta["calibration"],
        o["bank"],
        o["mu"],
        o["sd"],
        meta["ood_threshold"],
        meta,
    )


# ---------------------------------------------------------------- the forecast


def forecast_day(bundle: Bundle, events: Events, day: str, issued: float) -> dict[str, Any]:
    """The unsealed forecast for the UTC day ``day`` (YYYY-MM-DD), reading ``events`` before
    ``issued``."""
    grid = forecast_grid()
    midnight = to_seconds(day + "T00:00:00Z")
    seen = events.window(-math.inf, min(issued, midnight))
    clean, report = guard.clean(seen, midnight - guard.LIVE_EDGE_DAYS * DAY)
    members, etas_m3, m4_share = [], [], []
    for fit, states in zip(bundle.fits, bundle.states, strict=True):
        days = build_days(
            fit, clean, np.array([midnight], dtype=np.float64), grid, bundle.config, False
        )
        members.append(predict_members(states, days, grid, bundle.config)[:, 0])
        etas_m3.append(np.exp(days.log_etas[0, :, 1].astype(np.float64)))
        shares = magnitude_shares(fit, grid)
        first3 = int(grid.mag_bin_of(np.array([M_SCORED]))[0])
        first4 = int(grid.mag_bin_of(np.array([M_ALERT]))[0])
        m4_share.append(shares[first4:].sum() / shares[first3:].sum())
    rates = np.concatenate(members)  # (members, cells)
    mean = rates.mean(axis=0)
    etas_rates = np.mean(etas_m3, axis=0)
    context = clean.select(np.arange(max(len(clean) - CONTEXT, 0), len(clean)))
    fp = (ood.fingerprint(context) - bundle.ood_mu) / bundle.ood_sd
    dist = float(ood.knn_distance(bundle.ood_bank, fp[None])[0])
    spread = np.log(np.maximum(rates, 1e-12)).std(axis=0)
    top = np.argsort(-mean)[:5]
    lat, lon = grid.cell_centres()
    day_k = float(np.median([c["day_k"] for c in bundle.calibration]))
    day_gamma = float(np.median([c["day_gamma"] for c in bundle.calibration]))
    return {
        "schema": lock.SCHEMA,
        "region": bundle.meta["region"],
        "day": day,
        "issued": datetime.fromtimestamp(issued, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reads_until": to_iso(float(seen.time[-1])) if len(seen) else None,
        "magnitude": M_SCORED,
        "grid": {k: v for k, v in grid.__dict__.items()},
        "model": {
            "craton": bundle.meta["craton"],
            "bundle": LIVE_VERSION,
            "field_hash": bundle.meta["field_hash"],
            "members": int(rates.shape[0]),
            "trained_until": bundle.meta["split"],
        },
        "inputs": {
            "source": "INGV FDSN event service, read as published at issue time",
            "quakes": len(seen),
            "guard": {"duplicates": report.duplicates, "shifted_quakes": report.shifted_quakes},
        },
        "craton": {
            "rates": lock.sig(mean),
            "expected": float(lock.sig(mean.sum())),
            "expected_m4": float(lock.sig(mean.sum() * float(np.mean(m4_share)))),
            "member_spread": float(lock.sig(spread[mean >= np.quantile(mean, 0.99)].mean())),
            "day_k": float(lock.sig(day_k)),
            "day_gamma": float(lock.sig(day_gamma)),
            "abstain": dist > bundle.ood_threshold,
            "ood_distance": float(lock.sig(dist)),
            "ood_threshold": float(lock.sig(bundle.ood_threshold)),
            "top_cells": [
                {
                    "lat": round(float(lat[c]), 2),
                    "lon": round(float(lon[c]), 2),
                    "rate": float(lock.sig(mean[c])),
                }
                for c in top
            ],
        },
        "etas": {"rates": lock.sig(etas_rates), "expected": float(lock.sig(etas_rates.sum()))},
    }


def tomorrow(now: float) -> str:
    return (datetime.fromtimestamp(now, UTC) + timedelta(days=1)).strftime("%Y-%m-%d")


def run(root: Path, bundle_path: Path, http: Fetcher, now: float, log: Any) -> dict[str, Any]:
    """The nightly job: lock tomorrow, grade what can be graded, regenerate the page."""
    region = load_region(REGION, root)
    bundle = load_bundle(bundle_path)
    events = fetch_catalogue(root / "data" / "live" / REGION / "ingv", region, http, now, log)
    day = tomorrow(now)
    done: dict[str, Any] = {"locked": None, "graded": []}
    if not lock.day_path(root, REGION, day).exists():
        body = forecast_day(bundle, events, day, now)
        done["locked"] = str(lock.write_locked(root, REGION, body).relative_to(root).as_posix())
    grid = forecast_grid()
    cutoff = now - lock.GRADE_AFTER_DAYS * DAY
    for p in sorted(lock.forecasts_dir(root, REGION).glob("[0-9][0-9][0-9][0-9]/*.json")):
        if to_seconds(p.stem + "T00:00:00Z") + DAY > cutoff:
            continue
        gp = lock.grade_path(root, REGION, p.stem)
        if gp.exists():
            continue
        g = lock.grade(lock.read_locked(p), events, grid)
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(g, indent=1, sort_keys=True) + "\n", "utf-8", newline="\n")
        done["graded"].append(p.stem)
    lock.page(root, REGION)
    return done


def verify(root: Path) -> list[str]:
    """Problems with any locked forecast: a broken seal or a ledger that disagrees."""
    problems = []
    for region_dir in sorted((root / "forecasts").glob("*/")):
        if not region_dir.is_dir():
            continue
        ledger = {}
        lp = region_dir / "ledger.jsonl"
        if lp.is_file():
            for line in lp.read_text("utf-8").splitlines():
                rec = json.loads(line)
                ledger[rec["day"]] = rec["sha256"]
        for p in sorted(region_dir.glob("[0-9][0-9][0-9][0-9]/*.json")):
            body = json.loads(p.read_text("utf-8"))
            if not lock.intact(body):
                problems.append(f"{p.relative_to(root).as_posix()}: seal broken")
            elif ledger.get(body["day"]) != body["sha256"]:
                problems.append(f"{p.relative_to(root).as_posix()}: not in the ledger as sealed")
    return problems
