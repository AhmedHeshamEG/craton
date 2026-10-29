"""A reservoir model's pressure on a regular grid, published as one CSV in a Zenodo record.

Operators model their reservoirs and history-match the model to the pressures measured in
their wells; the model's pressure field is the best map of what pumping did to the rock.
Published, it is a CSV with the grid point's projected coordinates and one column per date:

    X,Y,P1957_10_01,P1958_10_01,...

The region's YAML says which record and file, the grid's coordinate system, how the date
columns are written and the pressure's unit:

    sources:
      pressure_grid:
        record: 10245813
        file: ReservoirModel_pressures_V1_2023.csv
        md5: 06cbc151d3cffd1ab1fa00b3302bad7f
        provider: ...
        license: LicenseRef-...
        citation: "..."
        crs: EPSG:28992             # craton.adapters.crs
        columns: {x: X, y: Y}
        dates: "P%Y_%m_%d"          # strptime pattern of a date column's header
        unit: bar                   # bar, MPa or Pa
        model: ...
        reservoir: ...
        depth_km: 3.0               # the reservoir's depth, when the grid gives none
        cell_km: 0.5

One token per grid point and date inside the region's box and window. A point whose pressure
is missing on a date makes no token for that date.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.crs import to_wgs84
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.adapters.zenodo import API
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.raw import RawStore

TO_MPA = {"bar": 0.1, "MPa": 1.0, "Pa": 1e-6}
#: Assumed 1σ of a modelled pressure (MPa): the spread between history-matched models.
PRESSURE_SIGMA_MPA = 0.5
#: A model's value stands for its date; one day of support.
TEMPORAL_S = 86_400.0
SPEC = "1.2.0"
REQUIRED = (
    "record",
    "file",
    "md5",
    "provider",
    "license",
    "citation",
    "crs",
    "columns",
    "dates",
    "unit",
    "model",
    "depth_km",
    "cell_km",
)


def options(region: Region) -> dict[str, Any]:
    opts = region.options(PressureGrid.name)
    for key in REQUIRED:
        if key not in opts:
            raise ValueError(f"region {region.name}: sources.pressure_grid needs {key!r}")
    if opts["unit"] not in TO_MPA:
        raise ValueError(f"pressure_grid.unit must be one of {sorted(TO_MPA)}")
    return opts


def date_columns(header: list[str], pattern: str) -> dict[str, float]:
    """The header's date columns and the moment (seconds, 00:00 UTC) each one names."""
    out = {}
    for name in header:
        try:
            out[name] = to_seconds(datetime.strptime(name, pattern).replace(tzinfo=UTC))
        except ValueError:
            continue
    return out


@register
class PressureGrid(Adapter):
    name = "pressure_grid"
    version = "1.0.0"
    instruments = ("reservoir_pressure",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Reservoir pressure from a history-matched model, on a grid (one CSV)",
            provider="The operator the region names (NAM, through TNO, for Groningen)",
            homepage="https://zenodo.org/records/10245813",
            instruments=self.instruments,
            license="LicenseRef-per-region",
            terms=(
                "Groningen's pressure model was provided by NAM and published by TNO on Zenodo "
                "(record 10245813) as open access with no licence stated. Craton reads it and "
                "never redistributes it; cite NAM, TNO and the record."
            ),
            citations=(
                "TNO (2023). Data for TNO Modelchain for Groningen [Data set]. Zenodo. "
                "https://doi.org/10.5281/zenodo.10245813 (files provided by NAM and KNMI)",
            ),
            access=(
                "One CSV through Zenodo's files API, checked against the record's MD5, "
                "downloaded once and cached by hash."
            ),
            mapping=(
                ("kind", "`forcing`"),
                ("lat, lon", "the grid point, converted from the region's `crs`"),
                ("depth_km", "the reservoir depth the region states (the grid gives none)"),
                ("time", "the date in the column's header, 00:00 UTC"),
                ("value", "pressure, converted to MPa"),
                ("qualifiers", "model, reservoir"),
            ),
            uncertainty=(
                f"Assumed {PRESSURE_SIGMA_MPA:g} MPa: the model is fitted to measured "
                "pressures, and its error is not published."
            ),
            known_issues=(
                "A model, matched with hindsight: the pressure of 2016 was refitted with the "
                "measurements up to 2023. Readers should say so.",
                "Dates after the model was made (Groningen's grid runs to 2054) are the "
                "operator's forecast of its own plans.",
                "One value per grid point for the whole reservoir thickness; depth is assumed.",
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
        link = API.format(record=opts["record"], file=opts["file"])
        data = http.get(link)
        got = hashlib.md5(data, usedforsecurity=False).hexdigest()
        if got != opts["md5"]:
            raise ValueError(f"{label}: md5 {got} is not the record's {opts['md5']}")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        raw.put(self.name, label, link, data, today, ".csv")
        report.downloaded += 1
        report.bytes += len(data)
        log(f"{label}  {len(data) / 2**20:.1f} MiB")
        return report

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        opts = options(ctx.region)
        raw = ctx.raw.current(self.name).get(f"{opts['record']}/{opts['file']}")
        if raw is None:
            ctx.reject("file_not_fetched")
            return
        reader = csv.reader(io.StringIO(ctx.use(raw).decode("utf-8")))
        header = [h.strip() for h in next(reader)]
        rows = list(reader)
        cols = opts["columns"]
        ix, iy = header.index(cols["x"]), header.index(cols["y"])
        dates = date_columns(header, str(opts["dates"]))
        x = np.array([float(r[ix]) for r in rows])
        y = np.array([float(r[iy]) for r in rows])
        lat, lon = to_wgs84(str(opts["crs"]), x, y)
        scale = TO_MPA[str(opts["unit"])]
        quals = {"model": str(opts["model"])}
        if opts.get("reservoir"):
            quals["reservoir"] = str(opts["reservoir"])
        version = f"zenodo {opts['record']}, retrieved {raw.retrieved}"
        by_year: dict[int, list[Token]] = {}
        for name, t in sorted(dates.items(), key=lambda kv: kv[1]):
            if not ctx.region.t_start <= t < ctx.region.t_end:
                ctx.reject("outside_window", len(rows))
                continue
            j = header.index(name)
            for n, r in enumerate(rows):
                if not ctx.region.box.contains(float(lat[n]), float(lon[n])):
                    ctx.reject("outside_box")
                    continue
                try:
                    p = float(r[j]) * scale
                    if not math.isfinite(p):
                        raise ValueError("not finite")
                    tok = validate_token(
                        {
                            "spec": SPEC,
                            "kind": "forcing",
                            "instrument": "reservoir_pressure",
                            "lat": round(float(lat[n]), 6),
                            "lon": round(float(lon[n]), 6),
                            "depth_km": float(opts["depth_km"]),
                            "time": t,
                            "time_iso": to_iso(t),
                            "value": [p],
                            "uncertainty": [PRESSURE_SIGMA_MPA],
                            "uncertainty_basis": "assumed",
                            "qualifiers": quals,
                            "provenance": {
                                "source": str(opts["provider"]),
                                "dataset_version": version,
                                "license": str(opts["license"]),
                                "adapter_version": self.version,
                                "content_hash": raw.content_hash,
                                "record": f"{r[ix]},{r[iy]};{name}",
                                "synthetic": False,
                            },
                            "resolution": {
                                "spatial_km": float(opts["cell_km"]),
                                "temporal_s": TEMPORAL_S,
                            },
                        }
                    )
                except (TokenError, ValueError) as exc:
                    codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                    ctx.reject("invalid:" + ",".join(codes))
                    continue
                by_year.setdefault(year_of(t), []).append(tok)
        for yr in sorted(by_year):
            yield sorted(by_year[yr], key=lambda tk: (tk.time, tk.lat, tk.lon))
