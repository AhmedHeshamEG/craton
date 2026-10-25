"""EarthquakeNPP's datasets (Stockman, Lawson & Werner, TMLR 2026), byte for byte.

EarthquakeNPP is the outside benchmark CB-29 scores against: five neural point processes
lost to ETAS on it. Its datasets are already cut in space, time and magnitude by its
authors, and CB-29 must read exactly what they read, so this adapter downloads their CSV
from a pinned commit of their repository and changes nothing but the format. Which dataset
comes from the region's YAML:

    sources:
      earthquakenpp: {dataset: ComCat/ComCat_catalog.csv}

Their CSVs keep time, place and magnitude only: no depth, no magnitude type. Tokens say so
honestly: depth 0 km with a vertical 1σ of ``DEPTH_SIGMA_KM`` (the crust's seismogenic
thickness), and the magnitude type ``M`` (unspecified, spec 1.1.0).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.spec.instruments import spec_for_magnitude
from craton.spec.timeaxis import to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

REPOSITORY = "https://github.com/ss15859/EarthquakeNPP"
#: The commit every download comes from (main on 2026-07-14).
COMMIT = "26d18048e1ca8ff2b02c7016b993de48ed0760f5"
RAW = f"https://raw.githubusercontent.com/ss15859/EarthquakeNPP/{COMMIT}/Datasets/"
LICENSE = "LicenseRef-USGS-public-domain"
MAGNITUDE_TYPE = "M"
MAGNITUDE_SIGMA = 0.2
DEPTH_SIGMA_KM = 15.0
SPATIAL_KM = 1.0
TEMPORAL_S = 0.01


def dataset(region: Region) -> str:
    name = region.options(EarthquakeNpp.name).get("dataset")
    if not name:
        raise ValueError(f"region {region.name}: sources.earthquakenpp needs 'dataset'")
    return str(name)


def parse_time(text: str) -> float:
    """Their pandas timestamps, ``1971-01-01 20:36:17.720`` (up to nanoseconds), as UTC."""
    head, _, frac = text.strip().partition(".")
    base = datetime.strptime(head, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return to_seconds(base) + (float("0." + frac) if frac else 0.0)


@register
class EarthquakeNpp(Adapter):
    name = "earthquakenpp"
    version = "1.0.0"
    instruments = ("seismic_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="EarthquakeNPP benchmark datasets",
            provider="Samuel Stockman, Daniel Lawson, Maximilian Werner (University of Bristol)",
            homepage=REPOSITORY,
            instruments=self.instruments,
            license=LICENSE,
            terms=(
                "The repository is MIT-licensed; the ComCat dataset is USGS ANSS ComCat, U.S. "
                "public domain. Cite the benchmark and the catalogue."
            ),
            citations=(
                "Stockman, S., Lawson, D. J., Werner, M. J. (2026). EarthquakeNPP: A Benchmark "
                "for Earthquake Forecasting with Neural Point Processes. Transactions on Machine "
                "Learning Research. https://arxiv.org/abs/2410.08226",
                "U.S. Geological Survey (2017). ANSS Comprehensive Catalog (ComCat). "
                "https://doi.org/10.5066/F7MS3QZH",
            ),
            access=(
                f"One CSV per dataset from `{RAW}` (commit `{COMMIT[:12]}`), downloaded once and "
                "cached by hash."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon", "latitude, longitude"),
                ("depth_km", f"not in the dataset: 0 with a vertical 1σ of {DEPTH_SIGMA_KM:g} km"),
                ("time", "time (UTC)"),
                ("value", "magnitude"),
                ("qualifiers", "magnitude_type `M` (the dataset gives none), event_id ← id"),
            ),
            uncertainty=(
                f"Magnitude 1σ an assumed {MAGNITUDE_SIGMA:g}; depth unknown (vertical 1σ "
                f"{DEPTH_SIGMA_KM:g} km); `uncertainty_basis: assumed`."
            ),
            known_issues=(
                "The datasets are already cut by their authors (ComCat_25: California's polygon, "
                "M ≥ 2.5, 1971 to 2020-01-17). That is the point: CB-29 reads what they read.",
                "No depth and no magnitude type survive in the CSV.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        name = dataset(region)
        if name in raw.current(self.name) and not refresh:
            report.cached += 1
            return report
        data = http.get(RAW + name)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        raw.put(self.name, name, RAW + name, data, today, ".csv")
        report.downloaded += 1
        report.bytes += len(data)
        log(f"{name}  {len(data) / 2**20:.1f} MiB")
        return report

    def _token(self, row: dict[str, str], raw: RawFile) -> dict[str, Any]:
        return {
            "spec": spec_for_magnitude(MAGNITUDE_TYPE),
            "kind": "event",
            "instrument": "seismic_catalog",
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "depth_km": 0.0,
            "time": parse_time(row["time"]),
            "value": [float(row["magnitude"])],
            "uncertainty": [MAGNITUDE_SIGMA],
            "uncertainty_basis": "assumed",
            "location_uncertainty": {"vertical_km": DEPTH_SIGMA_KM},
            "qualifiers": {"magnitude_type": MAGNITUDE_TYPE, "event_id": f"npp:{row['id']}"},
            "provenance": {
                "source": "EarthquakeNPP datasets",
                "dataset_version": f"{raw.label} at {COMMIT[:12]}, retrieved {raw.retrieved}",
                "license": LICENSE,
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"id={row['id']}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": TEMPORAL_S},
        }

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        raw = ctx.raw.current(self.name).get(dataset(ctx.region))
        if raw is None:
            ctx.reject("dataset_not_fetched")
            return
        rows = csv.DictReader(io.StringIO(ctx.use(raw).decode("utf-8")))
        batch: list[Token] = []
        year: int | None = None
        for row in rows:
            try:
                tok = validate_token(self._token(row, raw))
            except (TokenError, ValueError, KeyError) as exc:
                codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                ctx.reject("invalid:" + ",".join(codes))
                continue
            if not ctx.region.holds(tok.lat, tok.lon, tok.time):
                ctx.reject("outside_region")
                continue
            y = year_of(tok.time)
            if year is not None and y != year and batch:
                yield sorted(batch, key=lambda t: t.time)
                batch = []
            year = y
            batch.append(tok)
        if batch:
            yield sorted(batch, key=lambda t: t.time)
