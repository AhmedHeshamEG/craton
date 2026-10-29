"""Fluid injection volumes from a regulator's yearly well workbook (US Class II, form 1012A).

US injection wells report to a regulator once a year, a row per well with twelve monthly
volumes. Oklahoma's Corporation Commission publishes the whole state as one Excel workbook a
year; other regulators use the same layout under other column names. So the format is the
adapter, and the region's YAML says where the workbooks are and which column is which:

    sources:
      uic_workbook:
        url: https://example.gov/{year}-uic-injection-volumes.xlsx
        years: [2011, 2024]
        provider: ...
        license: LicenseRef-...
        unit: bbl                          # bbl or m3
        columns:
          well: API
          lat: LAT
          lon: LON
          top: InjTopDepth                 # feet below the surface
          bottom: InjBotDepth
          total_depth: TotalDepth          # used when the interval is missing
          well_class: WellType
          fluid: FluidType
          formation: [FormationName, PermitFormationName]   # the first one present
          measurement: MeasurementType
        months: "{mon} Vol"                # Jan Vol … Dec Vol

One token per well and month with a volume above zero; a month without a token injected
nothing (or reported nothing). A row that repeats another row exactly is a double report and
is dropped; two different rows for the same well and year (a well changing hands mid-year,
a second formation) are both kept, so their volumes add up.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.adapters.xlsx import Cell, records
from craton.spec.timeaxis import to_iso, to_seconds
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawFile, RawStore

#: One US oil barrel in cubic metres (42 US gallons, exactly).
BARREL_M3 = 0.158987294928
FOOT_KM = 0.0003048
UNITS = {"bbl": BARREL_M3, "m3": 1.0}
MONTHS = tuple(calendar.month_abbr[1:])
#: Volume 1σ, as a share of the volume: a meter, a calculation from pump strokes, or other.
SIGMA_SHARE = {"METERED": 0.05, "CALCULATED": 0.2}
OTHER_SHARE = 0.3
#: A well's spatial support (km): it injects at one place.
SPATIAL_KM = 0.1
SPEC = "1.2.0"
REQUIRED = ("url", "years", "provider", "license", "unit", "columns", "months")
COLUMNS = ("well", "lat", "lon", "top", "bottom", "total_depth")
#: The longest text each qualifier takes (the spec's patterns); longer text is cut.
QUALIFIER_WIDTH = {"well_class": 32, "fluid": 64, "formation": 128}


def options(region: Region) -> dict[str, Any]:
    opts = region.options(UicWorkbook.name)
    for key in REQUIRED:
        if key not in opts:
            raise ValueError(f"region {region.name}: sources.uic_workbook needs {key!r}")
    for key in COLUMNS:
        if key not in opts["columns"]:
            raise ValueError(f"region {region.name}: uic_workbook.columns needs {key!r}")
    if opts["unit"] not in UNITS:
        raise ValueError(f"uic_workbook.unit must be one of {sorted(UNITS)}")
    return opts


def years(opts: dict[str, Any]) -> range:
    first, last = (int(y) for y in opts["years"])
    return range(first, last + 1)


def _num(v: Cell) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _text(v: Cell) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _column(row: dict[str, Cell], spec: str | list[str]) -> Cell:
    for name in [spec] if isinstance(spec, str) else spec:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def month_start(year: int, month: int) -> float:
    return to_seconds(datetime(year, month, 1, tzinfo=UTC))


@register
class UicWorkbook(Adapter):
    name = "uic_workbook"
    version = "1.0.0"
    instruments = ("fluid_injection",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Injection well volumes, one regulator workbook a year (form 1012A layout)",
            provider="The regulator the region names (Oklahoma Corporation Commission)",
            homepage="https://oklahoma.gov/occ/divisions/oil-gas/oil-gas-data.html",
            instruments=self.instruments,
            license="LicenseRef-per-region",
            terms=(
                "Oklahoma's UIC volumes are public records of the Oklahoma Corporation "
                "Commission, published for download with no licence attached; cite the "
                "Commission and the year's file."
            ),
            citations=(
                "Oklahoma Corporation Commission, Oil and Gas Conservation Division: UIC "
                "Injection Volumes by Year (form 1012A), https://oklahoma.gov/occ/divisions/"
                "oil-gas/oil-gas-data.html",
            ),
            access=(
                "One .xlsx workbook per year from the URL pattern the region gives, downloaded "
                "once and cached by hash; read with the standard library (`craton.adapters."
                "xlsx`)."
            ),
            mapping=(
                ("kind", "`forcing`"),
                ("lat, lon", "the well's surface location"),
                ("depth_km", "middle of the injection interval (feet → km), else total depth"),
                ("time", "first day of the month, 00:00 UTC; resolution one month"),
                ("value", "the month's volume, in m³ (barrels × 0.158987)"),
                ("qualifiers", "well ← API number, well_class, fluid, formation"),
            ),
            uncertainty=(
                "Assumed: 5% of the volume when metered, 20% when calculated, 30% otherwise."
            ),
            known_issues=(
                "Before 2011 the layout differs (one .xls for 2006-2010); not read.",
                "The newest years are revised weekly, so the cached bytes are one version of "
                "them; the raw store keeps which.",
                "Reports are yearly: a forecast that reads last month's volume assumes it is "
                "known by then, which only the weekly Arbuckle reports (since 2016) make true.",
                "Wells of every class are kept; enhanced-recovery wells inject about as much as "
                "disposal wells but rarely shake anything, so readers choose by well_class.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        opts = options(region)
        have = raw.current(self.name)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        for y in years(opts):
            label = f"{y}"
            if label in have and not refresh:
                report.cached += 1
                continue
            link = str(opts["url"]).format(year=y)
            data = http.get(link)
            raw.put(self.name, label, link, data, today, ".xlsx")
            report.downloaded += 1
            report.bytes += len(data)
            log(f"{y}  {len(data) / 2**20:.1f} MiB")
        return report

    def _tokens_of(
        self, row: dict[str, Cell], year: int, raw: RawFile, opts: dict[str, Any], n: int
    ) -> Iterator[dict[str, Any]]:
        cols = opts["columns"]
        well = _text(_column(row, cols["well"]))
        lat, lon = _num(_column(row, cols["lat"])), _num(_column(row, cols["lon"]))
        if not well or lat is None or lon is None or (lat == 0 and lon == 0):
            raise ValueError("no well or location")
        top, bottom = _num(_column(row, cols["top"])), _num(_column(row, cols["bottom"]))
        if top and bottom and bottom >= top:
            depth_ft = (top + bottom) / 2
        else:
            depth_ft = _num(_column(row, cols["total_depth"])) or 0.0
        if depth_ft <= 0:
            raise ValueError("no depth")
        quals = {"well": well}
        for key, width in QUALIFIER_WIDTH.items():
            if key in cols:
                text = _text(_column(row, cols[key]))[:width]
                if text:
                    quals[key] = text
        how = _text(_column(row, cols.get("measurement", ""))).upper()
        share = SIGMA_SHARE.get(how, OTHER_SHARE)
        scale = UNITS[str(opts["unit"])]
        for m, mon in enumerate(MONTHS, start=1):
            vol = _num(row.get(str(opts["months"]).format(mon=mon)))
            if vol is None or vol <= 0:
                continue
            t = month_start(year, m)
            days = calendar.monthrange(year, m)[1]
            v = vol * scale
            yield {
                "spec": SPEC,
                "kind": "forcing",
                "instrument": "fluid_injection",
                "lat": lat,
                "lon": lon,
                "depth_km": depth_ft * FOOT_KM,
                "time": t,
                "time_iso": to_iso(t),
                "value": [v],
                "uncertainty": [max(v * share, 1e-3)],
                "uncertainty_basis": "assumed",
                "qualifiers": quals,
                "provenance": {
                    "source": str(opts["provider"]),
                    "dataset_version": f"{year} workbook, retrieved {raw.retrieved}",
                    "license": str(opts["license"]),
                    "adapter_version": self.version,
                    "content_hash": raw.content_hash,
                    "record": f"row={n};month={m}",
                    "synthetic": False,
                },
                "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": days * 86_400.0},
            }

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        opts = options(ctx.region)
        have = ctx.raw.current(self.name)
        months = [str(opts["months"]).format(mon=mon) for mon in MONTHS]
        for y in years(opts):
            raw = have.get(f"{y}")
            if raw is None:
                ctx.reject("file_not_fetched")
                continue
            seen: set[tuple[Any, ...]] = set()
            batch: list[Token] = []
            for n, row in enumerate(records(ctx.use(raw)), start=2):
                well = _text(_column(row, opts["columns"]["well"]))
                key = (well, *(_num(row.get(c)) for c in months))
                if key in seen:
                    ctx.reject("duplicate_row")
                    continue
                seen.add(key)
                try:
                    found = [validate_token(t) for t in self._tokens_of(row, y, raw, opts, n)]
                except (TokenError, ValueError) as exc:
                    codes = sorted(exc.codes) if isinstance(exc, TokenError) else [str(exc)]
                    ctx.reject("invalid:" + ",".join(codes).replace(" ", "_"))
                    continue
                for tok in found:
                    if ctx.region.holds(tok.lat, tok.lon, tok.time):
                        batch.append(tok)
                    else:
                        ctx.reject("outside_region")
            yield sorted(batch, key=lambda t: (t.time, t.qualifiers["well"]))
