"""GNSS daily positions from the Nevada Geodetic Laboratory (NGL), IGS20 frame.

NGL processes the raw data of ~20,000 GNSS stations worldwide into one position per station per
day. This adapter reads NGL's station list (``DataHoldings.txt``), keeps the stations inside the
region's box that were running during its window, downloads each one's ``tenv3`` time series,
and makes one ``gnss_daily`` token per station-day.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher, HttpError
from craton.adapters.region import Region
from craton.spec.timeaxis import to_iso, to_seconds
from craton.spec.token import Token, TokenError, validate_token, wrap_lon
from craton.store.raw import RawFile, RawStore

HOLDINGS_URL = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
TENV3_URL = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{station}.tenv3"
FRAME = "IGS20"
LICENSE = "LicenseRef-NGL-as-is"
"""NGL states no licence: data are supplied as-is and users are asked to cite (see the card)."""
HOLDINGS_LABEL = "stations"


@dataclass(frozen=True)
class Station:
    code: str
    lat: float
    lon: float
    first_day: str
    last_day: str


def parse_holdings(data: bytes) -> list[Station]:
    """Stations from ``DataHoldings.txt``. Malformed lines are skipped."""
    out = []
    for line in data.decode("utf-8", "replace").splitlines()[1:]:
        f = line.split()
        if len(f) < 10:
            continue
        try:
            out.append(Station(f[0], float(f[1]), wrap_lon(float(f[2])), f[7], f[8]))
        except ValueError:
            continue
    return out


def stations_in(region: Region, stations: list[Station]) -> list[Station]:
    """Stations inside the box whose data overlap the region's window."""
    start, end = (to_iso(t) or "" for t in (region.t_start, region.t_end))
    return sorted(
        (
            s
            for s in stations
            if region.box.contains(s.lat, s.lon)
            and s.first_day < end[:10]
            and s.last_day >= start[:10]
        ),
        key=lambda s: s.code,
    )


def station_label(code: str) -> str:
    return f"{FRAME}/{code}"


@register
class NglGnss(Adapter):
    name = "ngl_gnss"
    version = "1.0.0"
    instruments = ("gnss_daily",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="GNSS daily positions, Nevada Geodetic Laboratory (IGS20)",
            provider="Nevada Geodetic Laboratory, University of Nevada, Reno",
            homepage="https://geodesy.unr.edu/",
            instruments=self.instruments,
            license=LICENSE,
            terms=(
                "NGL publishes no licence. Its disclaimer says the data are supplied as-is with no "
                "warranty, and its acknowledgements page asks every user to cite Blewitt et al. "
                "(2018) and, where one exists, the original data citation of each station. "
                "Craton keeps the raw files in its local cache and redistributes none of them."
            ),
            citations=(
                "Blewitt, G., W. C. Hammond, and C. Kreemer (2018). Harnessing the GPS data "
                "explosion for interdisciplinary science. Eos, 99. "
                "https://doi.org/10.1029/2018EO104623",
                "Most central-Italian stations belong to INGV's RING network and other Italian "
                "networks; their own data citations are listed on each NGL station page.",
            ),
            access=(
                f"Station list `{HOLDINGS_URL}`, then one `tenv3` file per station from "
                f"`{TENV3_URL}`. At most one request per second; each file is downloaded once "
                "and cached by hash, and only `craton data fetch --refresh` asks again."
            ),
            mapping=(
                ("kind", "`observation`"),
                ("lat, lon", "the day's latitude and longitude (longitude wrapped, D-013)"),
                ("depth_km", "minus the day's ellipsoidal height / 1000"),
                ("time", "noon UTC of the solution's day"),
                (
                    "value",
                    "east = _e0 + __east, north = ____n0 + _north, up = u0 + ____up (metres)",
                ),
                ("uncertainty", "sig_e, sig_n, sig_u (reported)"),
                ("qualifiers", "station ← site, reference_frame = IGS20"),
                ("provenance.record", "`<site>.tenv3 <YYMMMDD> reflon=<reflon>`"),
                ("provenance.content_hash", "sha256 of the station's raw tenv3 file"),
            ),
            uncertainty=(
                "NGL's formal 1σ per component (`uncertainty_basis: reported`). Formal GNSS "
                "errors are known to be optimistic; correlated noise is not in them."
            ),
            known_issues=(
                "East is measured from a reference meridian (`reflon`, written in the record), "
                "north from the equator and up from the ellipsoid. Compare a station with itself "
                "over time; positions of two stations are not on one local grid.",
                "Daily solutions have the solid-Earth tide modelled out already. The tide tokens "
                "(`iers_tides`) are a separate forcing, not a correction to apply again.",
                "Antenna changes and earthquakes put steps in the series; NGL lists them on each "
                "station page. Nothing here removes them: the 2016 steps are signal.",
                "Some stations have long gaps; the health report counts gaps over 30 days.",
                "NGL reprocesses its archive from time to time; a refresh brings new bytes, a "
                "new hash and new tokens.",
            ),
            adapter_version=self.version,
        )

    # ------------------------------------------------------------------ fetch

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        have = raw.current(self.name)
        holdings = have.get(HOLDINGS_LABEL)
        if holdings is None or refresh:
            data = http.get(HOLDINGS_URL)
            holdings = raw.put(self.name, HOLDINGS_LABEL, HOLDINGS_URL, data, today, ".txt")
            report.downloaded += 1
            report.bytes += len(data)
        else:
            report.cached += 1
        chosen = stations_in(region, parse_holdings(raw.read(holdings)))
        for i, s in enumerate(chosen, 1):
            label = station_label(s.code)
            if label in have and not refresh:
                report.cached += 1
                continue
            url = TENV3_URL.format(station=s.code)
            try:
                data = http.get(url)
            except HttpError as exc:
                if exc.status != 404:
                    raise
                report.notes.append(f"{s.code}: no IGS20 tenv3 file (404)")
                continue
            raw.put(self.name, label, url, data, today, ".tenv3")
            report.downloaded += 1
            report.bytes += len(data)
            days = len(data.splitlines()) - 1
            log(f"{s.code}  {days:,} days  ({i}/{len(chosen)})")
        return report

    # ------------------------------------------------------------------ tokens

    def _token(self, f: list[str], t: float, raw: RawFile) -> dict[str, Any]:
        return {
            "kind": "observation",
            "instrument": "gnss_daily",
            "lat": float(f[20]),
            "lon": wrap_lon(float(f[21])),
            "depth_km": -float(f[22]) / 1000,
            "time": t,
            "time_iso": to_iso(t),
            "value": [
                float(f[7]) + float(f[8]),
                float(f[9]) + float(f[10]),
                float(f[11]) + float(f[12]),
            ],
            "uncertainty": [float(f[14]), float(f[15]), float(f[16])],
            "uncertainty_basis": "reported",
            "qualifiers": {"station": f[0], "reference_frame": FRAME},
            "provenance": {
                "source": "Nevada Geodetic Laboratory (geodesy.unr.edu)",
                "dataset_version": f"IGS20 tenv3, retrieved {raw.retrieved}",
                "license": LICENSE,
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"{f[0]}.tenv3 {f[1]} reflon={f[6]}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": 0.001, "temporal_s": 86_400.0},
        }

    def day_lines(self, data: bytes, ctx: BuildContext) -> Iterator[tuple[float, str]]:
        """(noon of the day, line) for every usable day inside the region's window."""
        days: set[str] = set()
        for line in data.decode("utf-8").splitlines()[1:]:
            f = line.split()
            if not f:
                continue
            if len(f) != 23:
                ctx.reject("field_count")
                continue
            try:
                day = datetime.strptime(f[1], "%y%b%d").replace(hour=12, tzinfo=UTC)
            except ValueError:
                ctx.reject("bad_date")
                continue
            t = to_seconds(day)
            if not ctx.region.t_start <= t < ctx.region.t_end:
                continue  # outside the window: not a rejection, just not this region's
            if f[1] in days:
                ctx.reject("duplicate")
                continue
            days.add(f[1])
            yield t, line

    def make(self, t: float, line: str, raw: RawFile, ctx: BuildContext) -> Token | None:
        try:
            return validate_token(self._token(line.split(), t, raw))
        except (TokenError, ValueError) as exc:
            codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
            ctx.reject("invalid:" + ",".join(codes))
            return None

    def parse(self, data: bytes, raw: RawFile, ctx: BuildContext) -> Iterator[Token]:
        """Tokens from one station's tenv3 file, inside the region's window."""
        for t, line in self.day_lines(data, ctx):
            tok = self.make(t, line, raw, ctx)
            if tok is not None:
                yield tok

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        have = ctx.raw.current(self.name)
        holdings = have.get(HOLDINGS_LABEL)
        if holdings is None:
            ctx.reject("station_list_not_fetched")
            return
        # Lines wait as text, bucketed by year; a year becomes tokens only when it is written.
        by_year: dict[int, list[tuple[float, str, RawFile]]] = defaultdict(list)
        for s in stations_in(ctx.region, parse_holdings(ctx.use(holdings))):
            raw = have.get(station_label(s.code))
            if raw is None:
                ctx.reject("station_not_fetched")
                continue
            for t, line in self.day_lines(ctx.use(raw), ctx):
                by_year[datetime.fromtimestamp(t, UTC).year].append((t, line, raw))
        for year in sorted(by_year):
            made = (self.make(t, line, raw, ctx) for t, line, raw in by_year.pop(year))
            yield [tok for tok in made if tok is not None]
