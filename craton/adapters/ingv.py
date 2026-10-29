"""INGV earthquake catalogue (ISIDe) through the FDSN event web service.

Every located earthquake in the region since 1985, all magnitudes, one ``seismic_catalog`` token
each. Requests go month by month in the FDSN text format; the service refuses any request over
20,000 events (HTTP 413), so a month that busy is fetched in halves and the pieces are joined into
the month's one raw file. Central Italy's busiest month, November 2016, has 13,209; a bigger
region will need the split.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher, HttpError
from craton.adapters.region import Box, Region
from craton.spec.instruments import MAGNITUDE_TYPES, SEISMIC_CATALOG
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

ENDPOINT = "https://webservices.ingv.it/fdsnws/event/1/query"
HEADER = (
    "#EventID|Time|Latitude|Longitude|Depth/Km|Author|Catalog|Contributor|ContributorID|"
    "MagType|Magnitude|MagAuthor|EventLocationName|EventType"
)
EVENT_TYPES = next(q for q in SEISMIC_CATALOG.qualifiers if q.name == "event_type").vocabulary or ()

MAGNITUDE_SIGMA = {"Mw": 0.1, "ML": 0.2}
"""Assumed 1σ by magnitude type (the text service publishes none); 0.3 for every other type."""
OTHER_SIGMA = 0.3
SPATIAL_KM = 1.0
"""Assumed size of a hypocentre's support, km. The text service gives no location errors."""
TEMPORAL_S = 0.1


def box_key(box: Box) -> str:
    return f"{box.lat_min:g}_{box.lat_max:g}_{box.lon_min:g}_{box.lon_max:g}"


def _short(t: float) -> str:
    iso = to_iso(t) or ""
    return iso[:10] if iso.endswith("T00:00:00Z") else iso


def month_windows(region: Region) -> list[tuple[float, float]]:
    """[start, end) windows, one per calendar month, clipped to the region's window."""
    out = []
    # Arithmetic from the epoch: Windows' fromtimestamp refuses moments before 1970.
    first = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=region.t_start)
    y, m = first.year, first.month
    while True:
        lo = to_seconds(datetime(y, m, 1, tzinfo=UTC))
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        hi = to_seconds(datetime(y2, m2, 1, tzinfo=UTC))
        a, b = max(lo, region.t_start), min(hi, region.t_end)
        if a < b:
            out.append((a, b))
        if hi >= region.t_end:
            return out
        y, m = y2, m2


def query_url(box: Box, start: float, end: float) -> str:
    params = {
        "starttime": (to_iso(start) or "")[:-1],
        "endtime": (to_iso(end) or "")[:-1],
        "minlat": f"{box.lat_min:g}",
        "maxlat": f"{box.lat_max:g}",
        "minlon": f"{box.lon_min:g}",
        "maxlon": f"{box.lon_max:g}",
        "format": "text",
        "orderby": "time-asc",
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params, safe=':')}"


def _body_lines(data: bytes) -> list[str]:
    text = data.decode("utf-8")
    return [line for line in text.splitlines() if line.strip() and not line.startswith("#")]


def fetch_window(http: Fetcher, box: Box, start: float, end: float, depth: int = 0) -> bytes:
    """One window's events as FDSN text, split in halves while the service says it's too big."""
    try:
        return http.get(query_url(box, start, end))
    except HttpError as exc:
        if exc.status != 413 or depth >= 12:
            raise
    mid = start + round((end - start) / 2)
    lines = _body_lines(fetch_window(http, box, start, mid, depth + 1))
    lines += _body_lines(fetch_window(http, box, mid, end, depth + 1))
    return ("\n".join([HEADER, *lines]) + "\n").encode("utf-8")


@register
class IngvCatalog(Adapter):
    name = "ingv_catalog"
    version = "1.0.0"
    instruments = ("seismic_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="INGV earthquake catalogue (ISIDe)",
            provider="Istituto Nazionale di Geofisica e Vulcanologia (INGV), Rome",
            homepage="https://terremoti.ingv.it/en/iside",
            instruments=self.instruments,
            license="CC-BY-4.0",
            terms=(
                "ISIDe data are licensed under Creative Commons Attribution 4.0 International. "
                "Use them freely and cite the database."
            ),
            citations=(
                "ISIDe Working Group (2007). Italian Seismological Instrumental and Parametric "
                "Database (ISIDe). Istituto Nazionale di Geofisica e Vulcanologia (INGV). "
                "https://doi.org/10.13127/ISIDE",
            ),
            access=(
                f"FDSN event service `{ENDPOINT}`, `format=text`, one request per calendar "
                "month inside the region's box. The service refuses requests over 20,000 events "
                "(HTTP 413); such months are fetched in halves and joined under one header. "
                "At most one request per second; every month is downloaded once and cached by "
                "hash, and only `craton data fetch --refresh` asks again."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon", "Latitude, Longitude"),
                ("depth_km", "Depth/Km (below sea level; under 0.11 km from the ellipsoid, D-012)"),
                ("time", "Time, UTC"),
                ("value", "Magnitude"),
                ("qualifiers", "magnitude_type ← MagType, event_id ← EventID, event_type"),
                ("provenance.record", "`eventid=<EventID>;author=<Author>`"),
                ("provenance.content_hash", "sha256 of the month's raw file"),
            ),
            uncertainty=(
                "The text service publishes no errors. Magnitude 1σ is assumed: 0.1 for Mw, "
                "0.2 for ML, 0.3 for every other type (`uncertainty_basis: assumed`). Location "
                f"errors are absent, so `resolution` is an assumed {SPATIAL_KM:g} km and "
                f"{TEMPORAL_S:g} s, and `location_uncertainty` is left out."
            ),
            known_issues=(
                "Magnitude types are mixed: mostly ML, some Md (older years) and Mw. Every token "
                "carries its type; nothing converts between them.",
                "The magnitude of completeness changes with time and with the network; it drops "
                "sharply after the 2000s and rises right after big shocks. The health report "
                "estimates it per year.",
                "Since 2005 the bulletin reviews only M ≥ 1.5; smaller events stay as the "
                "automatic or survey locations (`author` in the record says which).",
                "Events from 1985 to April 2005 were added to the service years after it went "
                "online; their locations come from older procedures and are less precise.",
                "INGV revises recent months. A refresh brings new bytes, a new hash and new "
                "tokens; the old version stays in the raw store.",
                "Event types other than earthquake, induced event, quarry blast and explosion are "
                "refused and counted, never guessed. In central Italy that is five controlled "
                "explosions and one anthropogenic event.",
                "120 events carry the generic magnitude type `M` (114 from 2009-2012 with author "
                "`UNKNOWN`, six from 1986-1990). Spec 1.0.0 has no `M`, so they are refused and "
                "counted; a later minor version can add it to the vocabulary.",
                "The catalogue reports duration magnitude Md until 2004 and local magnitude ML "
                "from 2005. Completeness is estimated for each type separately.",
            ),
            adapter_version=self.version,
        )

    # ------------------------------------------------------------------ fetch

    def label(self, box: Box, start: float, end: float) -> str:
        return f"{box_key(box)}/{_short(start)}/{_short(end)}"

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        have = raw.current(self.name)
        windows = month_windows(region)
        for i, (a, b) in enumerate(windows, 1):
            label = self.label(region.box, a, b)
            if label in have and not refresh:
                report.cached += 1
                continue
            data = fetch_window(http, region.box, a, b)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            raw.put(self.name, label, query_url(region.box, a, b), data, today, ".txt")
            report.downloaded += 1
            report.bytes += len(data)
            log(f"{label}  {len(_body_lines(data)):,} events  ({i}/{len(windows)})")
        return report

    # ------------------------------------------------------------------ tokens

    def _token(self, f: list[str], raw: RawFile) -> dict[str, Any]:
        mag_type = f[9]
        when = f[1] + "Z"
        return {
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
            "qualifiers": {"magnitude_type": mag_type, "event_id": f[0], "event_type": f[13]},
            "provenance": {
                "source": "INGV FDSN event service (webservices.ingv.it)",
                "dataset_version": f"ISIDe text, retrieved {raw.retrieved}",
                "license": "CC-BY-4.0",
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"eventid={f[0]};author={f[5]}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": TEMPORAL_S},
        }

    def parse(self, data: bytes, raw: RawFile, ctx: BuildContext) -> Iterator[Token]:
        """Tokens from one raw file, counting every line that can't become one."""
        for line in _body_lines(data):
            f = line.split("|")
            if len(f) != 14:
                ctx.reject("field_count")
                continue
            if not f[10].strip():
                ctx.reject("no_magnitude")
                continue
            if f[9] not in MAGNITUDE_TYPES:
                ctx.reject(f"magnitude_type:{f[9] or 'blank'}")
                continue
            if f[13] not in EVENT_TYPES:
                ctx.reject(f"event_type:{f[13] or 'blank'}")
                continue
            try:
                tok = validate_token(self._token(f, raw))
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
            label = self.label(ctx.region.box, a, b)
            raw = have.get(label)
            if raw is None:
                ctx.reject("month_not_fetched")
                continue
            for tok in self.parse(ctx.use(raw), raw, ctx):
                event_id = tok.qualifiers["event_id"]
                if event_id in seen:  # an event on a month boundary comes back twice
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
