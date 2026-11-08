"""``seismic_velocity_change``: dv/v between every nearby pair of stations, one token a day.

The slow half (daily correlations) lives in :mod:`craton.adapters.interferometry` and is cached;
the math in :mod:`craton.adapters.velocity`. This adapter turns them into an instrument:

1. per pair, each of the nine component pairs is stretched against its own reference (the
   region's ``reference`` period), on trailing stacks of 10 and 30 days, and the nine are
   averaged with weights 1/σ² (``dvv_raw``);
2. the slow drifts that are not the crust are fitted **on the reference days only** and
   removed (``dvv``): the network's secondary-microseism level and the tilt of its spectrum
   (both from ``seismic_noise_power`` tokens), plus an annual sine when the reference spans a
   year. Which of those terms are used is chosen on the reference too: fit on its first three
   quarters, keep the set with the smallest scatter on its last quarter (D-140);
3. one token per pair, day and stack, at the pair's midpoint.

Region YAML:

    adapters: [seismic_noise_power, seismic_velocity_change]
    sources:
      fdsn_waveforms:
        velocity:
          reference: ["2015-06-01", "2016-08-20"]   # before the test, never after
          max_pair_km: 80
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.interferometry import Correlations, correlations
from craton.adapters.region import Region
from craton.adapters.velocity import coda_window, design, fit_correction, pair_series
from craton.adapters.waveforms import fetch_waveforms, options
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawStore

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
BoolArray = np.ndarray[Any, np.dtype[np.bool_]]

SPEC = "1.5.0"
DAY_S = 86_400
STACKS = (10, 30)
"""Days stacked. Ten for a next-day forecaster, thirty for a cleaner series (D-140)."""
CHOICE_STACK = 10
"""The stack the correction terms are chosen on."""
TERM_SETS: tuple[tuple[str, ...], ...] = (
    (),
    ("microseism",),
    ("microseism", "balance"),
    ("annual",),
    ("annual", "microseism", "balance"),
)
ANNUAL_MIN_DAYS = 365
"""An annual sine is only fitted on a reference that holds a whole year."""


def reference_of(region: Region) -> tuple[float, float]:
    vel = options(region).get("velocity", {}) or {}
    ref = vel.get("reference")
    if not ref or len(ref) != 2:
        raise ValueError(f"region {region.name}: sources.fdsn_waveforms.velocity.reference")
    r0, r1 = (to_seconds(str(x) if "T" in str(x) else f"{x}T00:00:00Z") for x in ref)
    if not region.t_start <= r0 < r1 <= region.t_end:
        raise ValueError(f"region {region.name}: the reference must sit inside the region")
    return r0, r1


def network_noise(paths: list[Path], stations: set[str], days: FloatArray) -> dict[str, FloatArray]:
    """The network's median secondary-microseism level and short-minus-secondary tilt per day,
    from ``seismic_noise_power`` tokens of the stations in ``stations``."""
    sec: dict[float, list[float]] = {}
    tilt: dict[float, list[float]] = {}
    for p in paths:
        t = pq.read_table(p, columns=["time", "value", "qualifiers"]).to_pydict()
        for time, value, q in zip(t["time"], t["value"], t["qualifiers"], strict=True):
            if json.loads(q)["station"] in stations:
                sec.setdefault(time, []).append(value[1])
                tilt.setdefault(time, []).append(value[2] - value[1])
    return {
        "microseism": np.array([np.median(sec[d]) if d in sec else np.nan for d in days]),
        "balance": np.array([np.median(tilt[d]) if d in tilt else np.nan for d in days]),
    }


def _weighted_net(res: list[FloatArray], sig: list[FloatArray]) -> FloatArray:
    r = np.array(res)
    w = 1.0 / np.array(sig) ** 2
    w = np.where(np.isfinite(r) & np.isfinite(w), w, 0.0)
    tot = w.sum(axis=0)
    out: FloatArray = np.where(
        tot > 0, (np.nan_to_num(r) * w).sum(axis=0) / np.maximum(tot, 1e-30), np.nan
    )
    return out


def choose_terms(
    corr: Correlations, noise: dict[str, FloatArray], r0: float, r1: float
) -> tuple[tuple[str, ...], dict[str, float]]:
    """The correction terms, picked on the reference alone: build the reference and fit the
    terms on its first three quarters, keep the set whose network dv/v scatters least on the
    last quarter. Returns the set and every candidate's scatter."""
    days = corr.days
    cut = r0 + 0.75 * (r1 - r0)
    fit: BoolArray = (days >= r0) & (days < cut)
    check: BoolArray = (days >= cut) & (days < r1)
    series = [
        pair_series(corr.cc[p], corr.hours[p], fit, CHOICE_STACK, coda_window(corr.distance_km(p)))
        for p in range(len(corr.pairs))
    ]
    sets = [s for s in TERM_SETS if "annual" not in s or (r1 - r0) / DAY_S >= ANNUAL_MIN_DAYS]
    scatter: dict[str, float] = {}
    for terms in sets:
        x = design(days, terms, noise, CHOICE_STACK)
        res = []
        for dvv, sig, _ in series:
            b = fit_correction(dvv, sig, x, fit)
            res.append(dvv - x @ b if b is not None else np.full(len(days), np.nan))
        net = _weighted_net(res, [s[1] for s in series])
        scatter["+".join(terms) or "none"] = float(np.nanstd(net[check]))
    best = min(scatter, key=lambda k: scatter[k])
    return (tuple(best.split("+")) if best != "none" else ()), scatter


@register
class SeismicVelocityChange(Adapter):
    name = "seismic_velocity_change"
    version = "1.0.0"
    instruments = ("seismic_velocity_change",)
    requires = ("seismic_noise_power",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Daily seismic velocity change (dv/v) between station pairs, from ambient noise",
            provider="Computed by Craton from the waveforms the region names (central Italy: INGV)",
            homepage="https://www.fdsn.org/webservices/",
            instruments=self.instruments,
            license="CC-BY-4.0",
            terms=(
                "Derived from the region's waveforms; tokens carry the licence the region "
                "states (INGV's Italian networks: CC BY 4.0, cite the network DOI)."
            ),
            citations=(
                "INGV Seismological Data Centre (2006). Rete Sismica Nazionale (RSN). "
                "doi:10.13127/SD/X0FXNH7QFY",
                "Shapiro, N. M., Campillo, M. (2004). Emergence of broadband Rayleigh waves "
                "from correlations of the ambient seismic noise. GRL 31, L07614.",
                "Sens-Schönfelder, C., Wegler, U. (2006). Passive image interferometry and "
                "seasonal variations of seismic velocities at Merapi Volcano. GRL 33, L21302.",
                "Hobiger, M., Wegler, U., Shiomi, K., Nakahara, H. (2012). Coseismic and "
                "postseismic velocity changes detected by passive image interferometry. JGR "
                "117, B09313.",
                "Weaver, R. L., Hadziioannou, C., Larose, E., Campillo, M. (2011). On the "
                "precision of noise correlation interferometry. GJI 185, 1393-1404.",
                "Zhan, Z., Tsai, V. C., Clayton, R. W. (2013). Spurious velocity changes caused "
                "by temporal variations in ambient noise frequency content. GJI 194, 1574-1581.",
            ),
            access=(
                "The same cached 1 Hz LH waveforms as seismic_noise_power (nothing new is "
                "downloaded). Daily correlations are computed on the GPU and cached per month "
                "under data/cache/velocity/, keyed by the raw files and the settings."
            ),
            mapping=(
                ("kind", "`observation`"),
                ("lat, lon", "the midpoint of the two stations"),
                ("depth_km", "minus the two stations' mean elevation / 1000"),
                ("time", "midnight UTC starting the last day of the stack"),
                ("value", "dvv (corrected), dvv_raw, coherence"),
                ("uncertainty", "Weaver et al. (2011) σ, combined over the component pairs"),
                ("qualifiers", "station_a, station_b, stack_days, reference"),
                ("resolution", "half the station distance, and the stack's days"),
            ),
            uncertainty=(
                "From the coherence of each component's stretch (Weaver et al. 2011), combined "
                "as 1/sqrt(Σ 1/σ²). It assumes the nine components are independent, which they "
                "are not quite: real day-to-day scatter is larger (in the lesson)."
            ),
            known_issues=(
                "1 Hz data: the band is 0.1-0.45 Hz, below the 0.5-1 Hz where published "
                "studies saw the clearest coseismic drops in central Italy.",
                "The reference is the region's quiet period; dv/v is relative to it and means "
                "nothing across regions.",
                "The correction is fitted on the reference period only; a drift that starts "
                "after it (a new sensor, a new site) stays in dvv.",
                "Stacks read only past days, so a step shows up spread over the stack.",
                "Pairs farther apart than max_pair_km (80 km) are not correlated.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        return fetch_waveforms(region, raw, http, refresh, log)

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        r0, r1 = reference_of(ctx.region)
        corr = correlations(ctx.region, ctx.raw)
        for f in corr.used:
            ctx.raw_used[f.sha256] = f
        if not corr.pairs or len(corr.days) == 0:
            ctx.reject("no_pairs")
            return
        ids = {s.station_id for s in corr.stations}
        noise = network_noise(ctx.inputs.get("seismic_noise_power", []), ids, corr.days)
        terms, _ = choose_terms(corr, noise, r0, r1)
        ref: BoolArray = (corr.days >= r0) & (corr.days < r1)
        opts = options(ctx.region)
        ref_iso = f"{(to_iso(r0) or '')[:10]}/{(to_iso(r1) or '')[:10]}"
        by_year: dict[int, list[Token]] = {}
        for stack in STACKS:
            x = design(corr.days, terms, noise, stack)
            for p, (a, b) in enumerate(corr.pairs):
                sa, sb = corr.stations[a], corr.stations[b]
                dist = corr.distance_km(p)
                dvv, sig, coh = pair_series(
                    corr.cc[p], corr.hours[p], ref, stack, coda_window(dist)
                )
                beta = fit_correction(dvv, sig, x, ref)
                if beta is None:
                    ctx.reject("reference_too_short", int(np.isfinite(dvv).sum()))
                    continue
                fixed = dvv - x @ beta
                for i, day in enumerate(corr.days):
                    if not (np.isfinite(fixed[i]) and np.isfinite(sig[i])):
                        continue
                    tok = {
                        "spec": SPEC,
                        "kind": "observation",
                        "instrument": "seismic_velocity_change",
                        "lat": round((sa.lat + sb.lat) / 2, 5),
                        "lon": round((sa.lon + sb.lon) / 2, 5),
                        "depth_km": round(-(sa.elevation_m + sb.elevation_m) / 2000.0, 4),
                        "time": float(day),
                        "time_iso": to_iso(float(day)),
                        "value": [
                            round(float(fixed[i]), 7),
                            round(float(dvv[i]), 7),
                            round(float(coh[i]), 4),
                        ],
                        "uncertainty": [
                            max(round(float(sig[i]), 7), 1e-7),
                            max(round(float(sig[i]), 7), 1e-7),
                            0.01,
                        ],
                        "uncertainty_basis": "derived",
                        "qualifiers": {
                            "station_a": sa.station_id,
                            "station_b": sb.station_id,
                            "stack_days": str(stack),
                            "reference": ref_iso,
                        },
                        "provenance": {
                            "source": str(opts["provider"]),
                            "dataset_version": f"{sa.station_id}|{sb.station_id}"[:64],
                            "license": str(opts["license"]),
                            "adapter_version": self.version,
                            "content_hash": "sha256:"
                            + hashlib.sha256(
                                f"{sa.station_id}|{sb.station_id}|{day:.0f}|{stack}|"
                                f"{'+'.join(terms)}|{ref_iso}".encode()
                            ).hexdigest(),
                            "record": f"{dist:.1f} km, coda from {coda_window(dist)[0]:.0f} s, "
                            f"terms {'+'.join(terms) or 'none'}",
                            "synthetic": False,
                        },
                        "resolution": {
                            "spatial_km": max(dist / 2, 0.001),
                            "temporal_s": float(stack * DAY_S),
                        },
                    }
                    try:
                        t = validate_token(tok)
                    except (TokenError, ValueError) as exc:
                        codes = (
                            sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                        )
                        ctx.reject("invalid:" + ",".join(codes))
                        continue
                    by_year.setdefault(year_of(t.time), []).append(t)
        for yr in sorted(by_year):
            yield sorted(
                by_year[yr],
                key=lambda tk: (
                    tk.time,
                    tk.qualifiers["station_a"],
                    tk.qualifiers["station_b"],
                    int(tk.qualifiers["stack_days"]),
                ),
            )
