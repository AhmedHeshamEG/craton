"""Earthquake catalogues published as one CSV file in a Zenodo record.

Research catalogues (the machine-learning ones of P8c: Tan et al. 2021 for central Italy
2016-17, Fonzetti et al. 2025 for L'Aquila 2009, Poggiali et al. 2025 for Alto Tiberina) are
usually a single CSV on Zenodo, each with its own column names. The format is the same, so the
adapter is one; the region's YAML says which record, which file and which column is which:

    sources:
      zenodo_catalog:
        record: 4736089
        file: Amatrice_CAT5.v20210504.csv
        md5: 381ed815227c5aeecf849b49bdcf4dc2
        provider: Tan, Waldhauser & Ellsworth (2021)
        license: CC-BY-4.0
        citation: "..."
        magnitude_type: ML
        columns: {lat: lat, lon: lon, depth: dep, magnitude: ml_median, id: id}
        time_parts: [year, month, day, hour, minute, second]   # or columns.time (ISO 8601, UTC)
        keep: {split: "0"}                                     # rows kept only if these match

The download is checked against the record's MD5, so a record that changes under us fails
loudly instead of quietly becoming a different catalogue.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.spec.instruments import spec_for_magnitude
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

API = "https://zenodo.org/api/records/{record}/files/{file}/content"
#: Magnitude 1σ when the catalogue gives none.
MAGNITUDE_SIGMA = 0.2
SPATIAL_KM = 0.1
TEMPORAL_S = 0.01
REQUIRED = ("record", "file", "md5", "provider", "license", "citation", "magnitude_type", "columns")
TIME_PARTS = ("year", "month", "day", "hour", "minute", "second")


def options(region: Region) -> dict[str, Any]:
    opts = region.options(ZenodoCatalog.name)
    for key in REQUIRED:
        if key not in opts:
            raise ValueError(f"region {region.name}: sources.zenodo_catalog needs {key!r}")
    cols = opts["columns"]
    for key in ("lat", "lon", "depth", "magnitude", "id"):
        if key not in cols:
            raise ValueError(f"region {region.name}: zenodo_catalog.columns needs {key!r}")
    if "time" not in cols and "time_parts" not in opts:
        raise ValueError(f"region {region.name}: zenodo_catalog needs columns.time or time_parts")
    return opts


def url(opts: dict[str, Any]) -> str:
    return API.format(record=opts["record"], file=opts["file"])


def parse_iso(text: str) -> float:
    """``2009-01-07T04:41:22.63`` (no zone: UTC, as the catalogues state) to seconds."""
    head, _, frac = text.strip().replace(" ", "T").partition(".")
    base = datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    return to_seconds(base) + (float("0." + frac) if frac else 0.0)


def parse_parts(row: dict[str, str], names: list[str]) -> float:
    """Year, month, day, hour, minute and (fractional) second columns to seconds."""
    y, mo, d, h, mi = (int(float(row[n])) for n in names[:5])
    base = datetime(y, mo, d, h, mi, tzinfo=UTC)
    return to_seconds(base) + float(row[names[5]])


@register
class ZenodoCatalog(Adapter):
    name = "zenodo_catalog"
    version = "1.0.0"
    instruments = ("seismic_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Research earthquake catalogues on Zenodo (one CSV per record)",
            provider="The authors the region names (Tan et al., Fonzetti et al., Poggiali et al.)",
            homepage="https://zenodo.org",
            instruments=self.instruments,
            license="LicenseRef-per-region",
            terms=(
                "Each region's YAML names the record, its authors, licence and citation; the "
                "three catalogues of P8c are CC BY 4.0. Cite the catalogue and its paper."
            ),
            citations=(
                "Tan, Y. J., Waldhauser, F., Ellsworth, W. L., et al. (2021). Machine-Learning-"
                "Based High-Resolution Earthquake Catalog Reveals How Complex Fault Structures "
                "Were Activated during the 2016-2017 Central Italy Sequence. The Seismic Record "
                "1(1), 11-19. Data: https://doi.org/10.5281/zenodo.4736089",
                "Fonzetti, R., Govoni, A., De Gori, P., Valoroso, L., Chiarabba, C. (2025). "
                "Machine learning-based high-resolution data set for the 2009 L'Aquila "
                "earthquake sequence. Geophysical Journal International. "
                "Data: https://doi.org/10.5281/zenodo.16535092",
                "Poggiali, G., et al. (2025). Fault Geometry and Source Mechanics of the "
                "Altotiberina Fault System from a High-Resolution Machine-Learning Earthquake "
                "Catalog. Bulletin of the Seismological Society of America 115(5), 2181-2201. "
                "Data: https://doi.org/10.5281/zenodo.15622508",
            ),
            access=(
                "One CSV per record through Zenodo's files API, checked against the record's "
                "MD5, downloaded once and cached by hash."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon, depth_km", "the columns the region names"),
                ("time", "one ISO 8601 column (UTC) or year … second columns"),
                ("value", "the magnitude column the region names"),
                (
                    "qualifiers",
                    "magnitude_type from the region (the catalogue's own scale), "
                    "event_id ← record:id",
                ),
            ),
            uncertainty=(
                f"Magnitude 1σ an assumed {MAGNITUDE_SIGMA:g} unless the region names a column; "
                "location 1σ from the catalogue's error columns where the region names them."
            ),
            known_issues=(
                "These catalogues were relocated with hindsight (double-difference over the "
                "whole sequence), so they only feed tasks of their own (CB-20, CB-21).",
                "Each computes its own local magnitude: Tan's runs ~0.26 below INGV's, "
                "Alto Tiberina's ~0.5 below, L'Aquila's ~0.1 below (median over matched quakes). "
                "The magnitude_type says ML but the scale is the catalogue's own.",
                "Fonzetti's relocated file (`catalog_148k_DD.csv`) has its latitude and longitude "
                "headers swapped; regions map the columns by content, not by header.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        opts = options(region)
        label = f"{opts['record']}/{opts['file']}"
        if label in raw.current(self.name) and not refresh:
            report.cached += 1
            return report
        data = http.get(url(opts))
        got = hashlib.md5(data, usedforsecurity=False).hexdigest()
        if got != opts["md5"]:
            raise ValueError(f"{label}: md5 {got} is not the record's {opts['md5']}")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        raw.put(self.name, label, url(opts), data, today, ".csv")
        report.downloaded += 1
        report.bytes += len(data)
        log(f"{label}  {len(data) / 2**20:.1f} MiB")
        return report

    def _token(self, row: dict[str, str], raw: RawFile, opts: dict[str, Any]) -> dict[str, Any]:
        cols = opts["columns"]
        mag = float(row[cols["magnitude"]])
        if not math.isfinite(mag):
            raise ValueError("no magnitude")
        t = (
            parse_iso(row[cols["time"]])
            if "time" in cols
            else parse_parts(row, list(opts["time_parts"]))
        )
        loc: dict[str, float] = {}
        if "horizontal_err" in cols:
            loc["horizontal_km"] = max(float(row[c]) for c in cols["horizontal_err"]) or 0.001
        if "vertical_err" in cols:
            loc["vertical_km"] = float(row[cols["vertical_err"]]) or 0.001
        mtype = str(opts["magnitude_type"])
        tok: dict[str, Any] = {
            "spec": spec_for_magnitude(mtype),
            "kind": "event",
            "instrument": "seismic_catalog",
            "lat": float(row[cols["lat"]]),
            "lon": float(row[cols["lon"]]),
            "depth_km": float(row[cols["depth"]]),
            "time": t,
            "time_iso": to_iso(t),
            "value": [mag],
            "uncertainty": [MAGNITUDE_SIGMA],
            "uncertainty_basis": "assumed",
            "qualifiers": {
                "magnitude_type": mtype,
                "event_id": f"zenodo{opts['record']}:{row[cols['id']]}",
            },
            "provenance": {
                "source": str(opts["provider"]),
                "dataset_version": f"zenodo {raw.label}, retrieved {raw.retrieved}",
                "license": str(opts["license"]),
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"id={row[cols['id']]}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": TEMPORAL_S},
        }
        if loc:
            tok["location_uncertainty"] = loc
        return tok

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        opts = options(ctx.region)
        raw = ctx.raw.current(self.name).get(f"{opts['record']}/{opts['file']}")
        if raw is None:
            ctx.reject("file_not_fetched")
            return
        keep = {str(k): str(v) for k, v in dict(opts.get("keep", {})).items()}
        rows = csv.DictReader(io.StringIO(ctx.use(raw).decode("utf-8")))
        by_year: dict[int, list[Token]] = {}
        for row in rows:
            if any(row.get(k, "").strip() != v for k, v in keep.items()):
                ctx.reject("not_kept")
                continue
            try:
                tok = validate_token(self._token(row, raw, opts))
            except (TokenError, ValueError, KeyError) as exc:
                codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                ctx.reject("invalid:" + ",".join(codes))
                continue
            if not ctx.region.holds(tok.lat, tok.lon, tok.time):
                ctx.reject("outside_region")
                continue
            by_year.setdefault(year_of(tok.time), []).append(tok)
        for y in sorted(by_year):
            yield sorted(by_year[y], key=lambda t: t.time)
