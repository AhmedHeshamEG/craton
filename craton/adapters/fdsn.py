"""Any earthquake catalogue behind an FDSN event web service: USGS ComCat, GeoNet, EMSC, …

The FDSN event service is the one standard every seismological agency speaks, so one adapter
reads them all; the region's YAML says which service, which magnitudes and under which terms
(``sources.fdsn_catalog``). Nothing here knows a place.

    sources:
      fdsn_catalog:
        service: https://earthquake.usgs.gov/fdsnws/event/1/query
        provider: USGS ANSS ComCat
        license: LicenseRef-USGS-public-domain
        min_magnitude: 2.0
        params: {eventtype: earthquake}

Requests go month by month in the FDSN text format, 13 columns (ComCat, EMSC) or 14 with the
event type (GeoNet, INGV). A service refuses a request that holds too many events (ComCat:
HTTP 400 over 20,000; others 413), so such a month is fetched in halves and joined into the
month's one raw file, as ``ingv_catalog`` does.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher, HttpError
from craton.adapters.ingv import (
    EVENT_TYPES,
    OTHER_SIGMA,
    SPATIAL_KM,
    TEMPORAL_S,
    _short,
    box_key,
    month_windows,
)
from craton.adapters.region import Box, Region
from craton.spec.instruments import MAGNITUDE_TYPES, spec_for_magnitude
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

HEADER = (
    "#EventID|Time|Latitude|Longitude|Depth/Km|Author|Catalog|Contributor|ContributorID|"
    "MagType|Magnitude|MagAuthor|EventLocationName"
)
#: Status codes with which services say "too many events in one request".
TOO_BIG = (400, 413)
MAX_SPLITS = 12

#: Services write magnitude types their own way (``mww``, ``ML``, ``Mlv``). Each spelling maps to
#: one word of the spec's vocabulary; the service's own spelling stays in the provenance record.
#: A type missing here is refused and counted, never guessed.
MAGNITUDE_ALIASES = {
    "ml": "ML",
    "mlr": "ML",
    "mlg": "ML",
    "mlv": "MLv",
    "md": "Md",
    "mc": "Mc",
    "mw": "Mw",
    "mww": "Mw",
    "mwc": "Mw",
    "mwr": "Mw",
    "mwb": "Mw",
    "mwp": "Mwp",
    "mwl": "MwL",
    "mwpd": "Mwpd",
    "mb": "mb",
    "mb_lg": "mb_Lg",
    "mblg": "mb_Lg",
    "lg": "mb_Lg",
    "ms": "Ms",
    "ms_20": "Ms",
    "ms_vx": "Ms",
    "msbb": "Ms",
    "mh": "Mh",
    "mj": "Mj",
    "m": "M",
}
MAGNITUDE_SIGMA = {"Mw": 0.1, "ML": 0.2}
"""Assumed 1σ by magnitude type, as for INGV; ``OTHER_SIGMA`` otherwise."""
#: Event types a service writes that say nothing: the qualifier is left out, not guessed.
SILENT_EVENT_TYPES = ("", "unknown", "not reported", "not existing")


def magnitude_type(raw: str) -> str | None:
    """The spec's word for a service's magnitude type, or None if there isn't one."""
    word = MAGNITUDE_ALIASES.get(raw.strip().lower())
    if word is None and raw.strip() in MAGNITUDE_TYPES:
        word = raw.strip()
    return word if word in MAGNITUDE_TYPES else None


def options(region: Region) -> dict[str, Any]:
    opts = region.options(FdsnCatalog.name)
    for key in ("service", "provider", "license"):
        if not opts.get(key):
            raise ValueError(f"region {region.name}: sources.fdsn_catalog needs {key!r}")
    return opts


def query_url(opts: dict[str, Any], box: Box, start: float, end: float) -> str:
    params: dict[str, str] = {
        "starttime": (to_iso(start) or "")[:-1],
        "endtime": (to_iso(end) or "")[:-1],
        "minlatitude": f"{box.lat_min:g}",
        "maxlatitude": f"{box.lat_max:g}",
        "minlongitude": f"{box.lon_min:g}",
        "maxlongitude": f"{box.lon_max:g}",
        "format": "text",
        "orderby": "time-asc",
    }
    if opts.get("min_magnitude") is not None:
        params["minmagnitude"] = f"{float(opts['min_magnitude']):g}"
    params.update({str(k): str(v) for k, v in (opts.get("params") or {}).items()})
    return f"{opts['service']}?{urllib.parse.urlencode(params, safe=':')}"


def body_lines(data: bytes) -> list[str]:
    text = data.decode("utf-8", "replace")
    return [line for line in text.splitlines() if line.strip() and not line.startswith("#")]


def fetch_window(
    http: Fetcher, opts: dict[str, Any], box: Box, start: float, end: float, depth: int = 0
) -> bytes:
    """One window's events as FDSN text; split in halves while the service says too many.
    A service with nothing to say answers 204 (an empty body here)."""
    try:
        return http.get(query_url(opts, box, start, end))
    except HttpError as exc:
        if exc.status == 204:
            return (HEADER + "\n").encode("utf-8")
        if exc.status not in TOO_BIG or depth >= MAX_SPLITS:
            raise
    mid = start + round((end - start) / 2)
    lines = body_lines(fetch_window(http, opts, box, start, mid, depth + 1))
    lines += body_lines(fetch_window(http, opts, box, mid, end, depth + 1))
    return ("\n".join([HEADER, *lines]) + "\n").encode("utf-8")


@register
class FdsnCatalog(Adapter):
    name = "fdsn_catalog"
    version = "1.0.0"
    instruments = ("seismic_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Earthquake catalogues through any FDSN event web service",
            provider="The agency the region names (USGS ANSS ComCat, GeoNet, EMSC, …)",
            homepage="https://www.fdsn.org/webservices/",
            instruments=self.instruments,
            license="LicenseRef-per-region",
            terms=(
                "Each region's YAML names its service, its provider and the provider's licence; "
                "every token carries that licence. USGS ComCat is public domain; GeoNet and "
                "EMSC publish under CC BY 4.0 (cite them)."
            ),
            citations=(
                "U.S. Geological Survey, Earthquake Hazards Program (2017). Advanced National "
                "Seismic System (ANSS) Comprehensive Catalog of Earthquake Events and Products. "
                "https://doi.org/10.5066/F7MS3QZH",
                "GeoNet: GNS Science (1970). New Zealand Earthquake Catalogue. "
                "https://doi.org/10.21420/0S8P-TZ38",
                "Euro-Mediterranean Seismological Centre (EMSC), https://www.seismicportal.eu",
            ),
            access=(
                "FDSN event service named in `sources.fdsn_catalog.service`, `format=text`, "
                "one request per calendar month inside the region's box, from the region's "
                "`min_magnitude`. A month the service calls too big (HTTP 400 or 413) is fetched "
                "in halves and joined under one header. At most one request per second; every "
                "month is downloaded once and cached by hash."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon", "Latitude, Longitude"),
                ("depth_km", "Depth/km (below sea level; under 0.11 km from the ellipsoid)"),
                ("time", "Time, UTC"),
                ("value", "Magnitude"),
                (
                    "qualifiers",
                    "magnitude_type ← MagType through `MAGNITUDE_ALIASES`, event_id ← EventID, "
                    "event_type ← EventType when the service gives one",
                ),
                ("provenance.record", "`eventid=<EventID>;author=<Author>;magtype=<MagType>`"),
                ("provenance.content_hash", "sha256 of the month's raw file"),
            ),
            uncertainty=(
                "The text format carries no errors. Magnitude 1σ is assumed: 0.1 for Mw, 0.2 "
                f"for ML, {OTHER_SIGMA:g} otherwise (`uncertainty_basis: assumed`); resolution "
                f"an assumed {SPATIAL_KM:g} km and {TEMPORAL_S:g} s."
            ),
            known_issues=(
                "Magnitude types are mixed and differ by agency and decade (ML, Md, mb, Mw "
                "flavours). Every token carries its type; nothing converts between them.",
                "ComCat's text format has no event type column: the region asks the service for "
                "`eventtype=earthquake` instead.",
                "Completeness differs by region and decade; each region states the magnitude it "
                "trusts (`completeness`) and the health report estimates it per year.",
                "Agencies revise recent months. A refresh brings new bytes, a new hash and new "
                "tokens; the old version stays in the raw store.",
                "An event on a month boundary can come back twice; the second is dropped.",
            ),
            adapter_version=self.version,
        )

    def label(self, box: Box, start: float, end: float) -> str:
        return f"{box_key(box)}/{_short(start)}/{_short(end)}"

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        opts = options(region)
        report = FetchReport()
        have = raw.current(self.name)
        windows = month_windows(region)
        for i, (a, b) in enumerate(windows, 1):
            label = self.label(region.box, a, b)
            if label in have and not refresh:
                report.cached += 1
                continue
            data = fetch_window(http, opts, region.box, a, b)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            raw.put(self.name, label, query_url(opts, region.box, a, b), data, today, ".txt")
            report.downloaded += 1
            report.bytes += len(data)
            log(f"{label}  {len(body_lines(data)):,} events  ({i}/{len(windows)})")
        return report

    def _token(self, f: list[str], mag_type: str, raw: RawFile, opts: dict[str, Any]) -> Any:
        when = f[1].rstrip("Z") + "Z"
        quals = {"magnitude_type": mag_type, "event_id": f[0].strip()}
        if len(f) == 14 and f[13].strip() not in SILENT_EVENT_TYPES:
            quals["event_type"] = f[13].strip()
        return {
            "spec": spec_for_magnitude(mag_type),
            "kind": "event",
            "instrument": "seismic_catalog",
            "lat": float(f[2]),
            "lon": float(f[3]),
            "depth_km": float(f[4]),
            "time": to_seconds(when),
            "time_iso": when,
            "value": [float(f[10])],
            "uncertainty": [MAGNITUDE_SIGMA.get(mag_type, OTHER_SIGMA)],
            "uncertainty_basis": "assumed",
            "qualifiers": quals,
            "provenance": {
                "source": f"{opts['provider']} FDSN event service",
                "dataset_version": f"FDSN text, retrieved {raw.retrieved}",
                "license": opts["license"],
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"eventid={f[0].strip()};author={f[5].strip()};magtype={f[9].strip()}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": TEMPORAL_S},
        }

    def parse(self, data: bytes, raw: RawFile, ctx: BuildContext) -> Iterator[Token]:
        """Tokens from one raw file, counting every line that can't become one."""
        opts = options(ctx.region)
        for line in body_lines(data):
            f = line.split("|")
            if len(f) not in (13, 14):
                ctx.reject("field_count")
                continue
            if not f[10].strip() or not f[4].strip():
                ctx.reject("no_magnitude" if not f[10].strip() else "no_depth")
                continue
            mag_type = magnitude_type(f[9])
            if mag_type is None:
                ctx.reject(f"magnitude_type:{f[9].strip() or 'blank'}")
                continue
            if len(f) == 14:
                kind = f[13].strip()
                if kind not in SILENT_EVENT_TYPES and kind not in EVENT_TYPES:
                    ctx.reject(f"event_type:{kind}")
                    continue
            try:
                tok = validate_token(self._token(f, mag_type, raw, opts))
            except (TokenError, ValueError) as exc:
                codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                ctx.reject("invalid:" + ",".join(codes))
                continue
            if not ctx.region.holds(tok.lat, tok.lon, tok.time):
                ctx.reject("outside_region")
                continue
            yield tok

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        have = ctx.raw.current(self.name)
        seen: set[str] = set()
        batch: list[Token] = []
        year: int | None = None
        for a, b in month_windows(ctx.region):
            raw = have.get(self.label(ctx.region.box, a, b))
            if raw is None:
                ctx.reject("month_not_fetched")
                continue
            for tok in self.parse(ctx.use(raw), raw, ctx):
                event_id = tok.qualifiers["event_id"]
                if event_id in seen:
                    ctx.reject("duplicate")
                    continue
                seen.add(event_id)
                y = year_of(tok.time)
                if year is not None and y != year and batch:
                    yield batch
                    batch = []
                year = y
                batch.append(tok)
        if batch:
            yield batch
