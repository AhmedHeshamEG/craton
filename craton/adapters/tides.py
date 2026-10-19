"""Solid-Earth tides, computed, not downloaded.

Two sets of ``solid_earth_tide`` forcing tokens:

* **At every earthquake**: the body tide at the ground above each catalogue event, at the moment
  it started. This is the forcing a model needs to ask "do quakes prefer some tidal phases?"
  Each token's ``content_hash`` is the hash of the event token it belongs to, so the chain runs
  tide → event → INGV raw file.
* **An hourly series at the region's centre**: the tide as a continuous clock, so a model sees
  the forcing on quiet days too. Across a region a few hundred km wide the tide changes smoothly,
  so one series stands for the whole box.

Values are rounded to the micrometre, far below the 1 mm uncertainty, so every machine writes
the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.geo.solid_tide import MODEL, solid_tide_enu
from craton.spec.timeaxis import to_iso, year_of
from craton.spec.token import Token, TokenError, validate_token
from craton.store.parquet import read_columns
from craton.store.raw import RawStore

SIGMA_M = 0.001
"""1σ per channel, metres: the code matches the JPL DE440 ephemeris to under 0.71 mm and the
IERS model itself is good to about a millimetre."""
STEP_S = 3600.0
DECIMALS = 6
CHUNK = 20_000


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@register
class IersTides(Adapter):
    name = "iers_tides"
    version = "1.0.0"
    instruments = ("solid_earth_tide",)
    requires = ("ingv_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Solid-Earth body tide (IERS 2010), computed by Craton",
            provider="Computed locally by `craton.geo.solid_tide`",
            homepage="https://iers-conventions.obspm.fr/chapter7.php",
            instruments=self.instruments,
            license="CC-BY-4.0",
            terms=(
                "Computed values. Tokens at earthquakes inherit the catalogue's CC-BY-4.0 "
                "(their positions and times are INGV's); the hourly series is computed from "
                "nothing but the clock and is offered under the same licence for simplicity. "
                "The code is a derived work of the IERS Conventions software and carries its "
                "notice (`craton/geo/IERS_NOTICE.txt`)."
            ),
            citations=(
                "Petit, G. and Luzum, B. (eds.) (2010). IERS Conventions (2010). IERS Technical "
                "Note 36, chapter 7.1.1. Verlag des Bundesamts für Kartographie und Geodäsie.",
                "Mathews, P. M., Dehant, V. and Gipson, J. M. (1997). Tidal station "
                "displacements. J. Geophys. Res. 102(B9), 20469-20477.",
                "Montenbruck, O. and Gill, E. (2000). Satellite Orbits: Models, Methods and "
                "Applications. Springer. Section 3.3.2 (Sun and Moon).",
            ),
            access=(
                "Nothing is downloaded. Event positions come from the `ingv_catalog` tokens of "
                "the same build; the Sun and the Moon come from analytic series."
            ),
            mapping=(
                ("kind", "`forcing`"),
                ("lat, lon", "the event's epicentre, or the centre of the region's box"),
                ("depth_km", "0: the displacement of the ellipsoid surface at that point"),
                ("time", "the event's origin time, or every hour on the hour"),
                ("value", "east, north, up displacement, metres, rounded to 1 µm"),
                ("qualifiers", "model = IERS2010"),
                ("provenance.record", "`event=<EventID>` or `hourly=centre`"),
                (
                    "provenance.content_hash",
                    "the event token's hash, or the hash of the request (place, time, model)",
                ),
            ),
            uncertainty=(
                f"{SIGMA_M * 1000:g} mm per channel (`derived`): the largest difference from the "
                "same model driven by the JPL DE440 ephemeris is 0.71 mm, and from PySolid "
                "0.4 mm, over six places from 1990 to 2026 (`tests/test_solid_tide.py`)."
            ),
            known_issues=(
                "Displacement, not stress. Whether a tide pushes a fault towards failure depends "
                "on the fault's orientation; a tidal-stress instrument can come later.",
                "Body tide only: ocean loading (a few mm to cm inland, more on coasts) and pole "
                "tide are not included.",
                "Earth rotation is approximated by mean sidereal time with UT1 = UTC; the error "
                "is far below a millimetre.",
                "The permanent tide is kept (the IERS 'conventional tide free' convention).",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        return FetchReport(notes=["computed from the catalogue at build time; nothing to fetch"])

    # ------------------------------------------------------------------ tokens

    def _tokens(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        t: np.ndarray,
        records: list[str],
        hashes: list[str],
        ctx: BuildContext,
    ) -> list[Token]:
        out = []
        for i in range(0, len(t), CHUNK):
            sl = slice(i, i + CHUNK)
            enu = np.round(solid_tide_enu(lat[sl], lon[sl], t[sl]), DECIMALS)
            for j, k in enumerate(range(i, min(i + CHUNK, len(t)))):
                data: dict[str, Any] = {
                    "kind": "forcing",
                    "instrument": "solid_earth_tide",
                    "lat": float(lat[k]),
                    "lon": float(lon[k]),
                    "depth_km": 0.0,
                    "time": float(t[k]),
                    "time_iso": to_iso(float(t[k])),
                    "value": [float(x) + 0.0 for x in enu[j]],
                    "uncertainty": [SIGMA_M] * 3,
                    "uncertainty_basis": "derived",
                    "qualifiers": {"model": MODEL},
                    "provenance": {
                        "source": "Craton solid-Earth tide (craton.geo.solid_tide)",
                        "dataset_version": f"{MODEL}, adapter {self.version}",
                        "license": "CC-BY-4.0",
                        "adapter_version": self.version,
                        "content_hash": hashes[k],
                        "record": records[k],
                        "synthetic": False,
                    },
                    "resolution": {"spatial_km": 1.0, "temporal_s": 60.0},
                }
                try:
                    out.append(validate_token(data))
                except TokenError as exc:
                    ctx.reject("invalid:" + ",".join(sorted(exc.codes)))
        return out

    def _event_inputs(self, files: list[Path]) -> dict[int, dict[str, list[Any]]]:
        """Catalogue events by year: lat, lon, time, event id, token hash."""
        by_year: dict[int, dict[str, list[Any]]] = {}
        cols = ("lat", "lon", "time", "qualifiers", "token_hash")
        for path in sorted(files):
            c = read_columns(path, cols)
            for lat, lon, t, q, h in zip(*(c[k] for k in cols), strict=True):
                bucket = by_year.setdefault(year_of(t), {k: [] for k in cols})
                for key, v in zip(cols, (lat, lon, t, q, h), strict=True):
                    bucket[key].append(v)
        return by_year

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        events = self._event_inputs(ctx.inputs.get("ingv_catalog", []))
        clat, clon = ctx.region.box.centre
        first = np.ceil(ctx.region.t_start / STEP_S) * STEP_S
        hours = np.arange(first, ctx.region.t_end, STEP_S)
        hour_years = hours.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
        years = sorted(set(events) | set(hour_years.tolist()))
        for year in years:
            batch: list[Token] = []
            ev = events.get(year)
            if ev:
                ids = [json.loads(q)["event_id"] for q in ev["qualifiers"]]
                batch += self._tokens(
                    np.array(ev["lat"]),
                    np.array(ev["lon"]),
                    np.array(ev["time"]),
                    [f"event={i}" for i in ids],
                    list(ev["token_hash"]),
                    ctx,
                )
            hy = hours[hour_years == year]
            if len(hy):
                records = ["hourly=centre"] * len(hy)
                hashes = [_sha(f"{MODEL} lat={clat:g} lon={clon:g} time={h:.0f}") for h in hy]
                batch += self._tokens(
                    np.full(len(hy), clat), np.full(len(hy), clon), hy, records, hashes, ctx
                )
            if batch:
                yield batch
