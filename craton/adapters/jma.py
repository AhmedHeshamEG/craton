"""The JMA Unified Hypocenter Catalogue: Japan's earthquakes from the Seismological Bulletin.

JMA publishes one zip a year (1997 in two halves), each a text file of fixed-width 96-byte
records, one per earthquake. The format is JMA's own, so it gets its own adapter; which part
of the archipelago and which magnitudes come from the region's YAML (``sources.jma_hypocenter``):

    sources:
      jma_hypocenter: {min_magnitude: 2.0}

Only records located by JMA itself (record type ``J``) become tokens; artificial events,
eruption earthquakes and low-frequency earthquakes are refused and counted.
"""

from __future__ import annotations

import io
import zipfile
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

BASE = "https://www.data.jma.go.jp/eqev/data/bulletin/data/hypo/"
FIRST_YEAR = 1983  # the first one-year file; earlier ones span decades
LICENSE = "CC-BY-4.0"
RECORD = 96
#: The bulletin's times are Japan Standard Time, UTC+9.
JST_OFFSET_S = 9 * 3600.0
#: JMA's magnitude letters (column 55) → the spec's word. J, D, d, V, v are all JMA magnitude
#: (from displacement or velocity amplitudes); W is a moment magnitude; B, S come from USGS.
MAGNITUDE_LETTERS = {"J": "Mj", "D": "Mj", "d": "Mj", "V": "Mj", "v": "Mj", "W": "Mw"}
MAGNITUDE_LETTERS.update({"B": "mb", "S": "Ms"})
#: Column 61: 1 natural earthquake, 2 natural but too few JMA stations; 3 artificial event,
#: 4 eruption earthquake, 5 low-frequency earthquake.
NATURAL = ("1", "2")
SUBSIDIARY_REASONS = {"3": "artificial", "4": "eruption", "5": "low_frequency"}
MAGNITUDE_SIGMA = 0.2
SPATIAL_KM = 1.0
TEMPORAL_S = 0.1


def files_for(region: Region) -> list[str]:
    """The yearly zips that cover the region's window."""
    y0 = max(datetime.fromtimestamp(region.t_start, UTC).year, FIRST_YEAR)
    y1 = datetime.fromtimestamp(region.t_end - 1, UTC).year
    out = []
    for y in range(y0, y1 + 1):
        out += ["h199701.zip", "h199710.zip"] if y == 1997 else [f"h{y}.zip"]
    return out


def magnitude(code: str) -> float | None:
    """F2.1 with JMA's letters for negative values: -1 → -0.1, A5 → -1.5, B0 → -2.0."""
    code = code.strip()
    if not code:
        return None
    if code[0] == "-":
        return -int(code[1:]) / 10
    if code[0] in "ABC":
        return -(10 * ("ABC".index(code[0]) + 1) + int(code[1])) / 10
    return int(code) / 10


def depth_km(field: str) -> float | None:
    """F5.2 (depth solved for) or I3 + 2 blanks (depth slices of 1 km since 1983)."""
    if not field.strip():
        return None
    if field[3:].strip():
        return int(field.replace(" ", "0")) / 100
    return float(int(field[:3]))


def record(line: str) -> dict[str, Any] | str:
    """One record's fields, or the reason it can't be a token."""
    if len(line) < RECORD - 1:
        return "short_record"
    if line[0] != "J":
        return f"agency:{line[0]}"
    sub = line[60]
    if sub not in NATURAL:
        return SUBSIDIARY_REASONS.get(sub, f"subsidiary:{sub.strip() or 'blank'}")
    try:
        mag = magnitude(line[52:54])
        depth = depth_km(line[44:49])
        sec = int(line[13:17].replace(" ", "0")) / 100
        lat = int(line[21:24]) + int(line[24:28].replace(" ", "0")) / 6000
        lon = int(line[32:36]) + int(line[36:40].replace(" ", "0")) / 6000
        stamp = datetime(
            int(line[1:5]),
            int(line[5:7]),
            int(line[7:9]),
            int(line[9:11]),
            int(line[11:13]),
            tzinfo=UTC,
        )
    except ValueError:
        return "unparseable"
    if mag is None:
        return "no_magnitude"
    if depth is None:
        return "no_depth"
    letter = line[54]
    if letter not in MAGNITUDE_LETTERS:
        return f"magnitude_type:{letter.strip() or 'blank'}"
    t = to_seconds(stamp) + sec - JST_OFFSET_S
    return {
        "time": t,
        "lat": lat,
        "lon": lon,
        "depth_km": depth,
        "mag": mag,
        "type": MAGNITUDE_LETTERS[letter],
        "letter": letter,
        "id": "jma:"
        + line[1:17].replace(" ", "0")
        + ":"
        + line[21:28].replace(" ", "0")
        + ":"
        + line[32:40].replace(" ", "0"),
        "flag": line[95].strip() if len(line) >= RECORD else "",
    }


@register
class JmaHypocenter(Adapter):
    name = "jma_hypocenter"
    version = "1.0.0"
    instruments = ("seismic_catalog",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="JMA Unified Hypocenter Catalogue (the Seismological Bulletin of Japan)",
            provider="Japan Meteorological Agency (JMA), with MEXT",
            homepage="https://www.data.jma.go.jp/eqev/data/bulletin/hypo_e.html",
            instruments=self.instruments,
            license=LICENSE,
            terms=(
                "JMA's Terms of Use are compatible with Creative Commons Attribution 4.0; the "
                "data may be used under CC BY 4.0 with the source credited."
            ),
            citations=(
                "Japan Meteorological Agency, The Seismological Bulletin of Japan, hypocenter "
                "files. https://www.data.jma.go.jp/eqev/data/bulletin/hypo_e.html",
            ),
            access=(
                f"One zip per year under `{BASE}` (1997 in two parts), each one fixed-width "
                "text file. Downloaded once each, at most one request per second, cached by hash."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon", "columns 22-28 and 33-40, degrees + minutes (JGD2000/2011 ≈ WGS84)"),
                ("depth_km", "columns 45-49 (F5.2, or 1 km depth slices)"),
                ("time", "columns 2-17, Japan Standard Time, minus 9 hours to UTC"),
                ("value", "magnitude 1, columns 53-54 (letters for negative values)"),
                (
                    "qualifiers",
                    "magnitude_type ← column 55 (J/D/d/V/v → Mj, W → Mw, B → mb, S → Ms)",
                ),
                ("provenance.record", "`jma:<date time>:<lat>:<lon>;type=<letter>;flag=<col 96>`"),
            ),
            uncertainty=(
                "The record has origin-time and location errors, kept out of the token for now; "
                f"magnitude 1σ is an assumed {MAGNITUDE_SIGMA:g} (`uncertainty_basis: assumed`)."
            ),
            known_issues=(
                "From April 2016 an automatic process roughly doubled the number of small "
                "earthquakes located; completeness drops at that date.",
                "Offshore completeness is far worse than on land; a region states one "
                "magnitude it trusts.",
                "Records located by USGS or other agencies (types U, I) are left out.",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        have = raw.current(self.name)
        names = files_for(region)
        for i, name in enumerate(names, 1):
            if name in have and not refresh:
                report.cached += 1
                continue
            data = http.get(BASE + name)
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            raw.put(self.name, name, BASE + name, data, today, ".zip")
            report.downloaded += 1
            report.bytes += len(data)
            log(f"{name}  {len(data) / 2**20:.1f} MiB  ({i}/{len(names)})")
        return report

    def _token(self, r: dict[str, Any], raw: RawFile) -> dict[str, Any]:
        return {
            "spec": spec_for_magnitude(r["type"]),
            "kind": "event",
            "instrument": "seismic_catalog",
            "lat": r["lat"],
            "lon": r["lon"],
            "depth_km": r["depth_km"],
            "time": r["time"],
            "value": [r["mag"]],
            "uncertainty": [MAGNITUDE_SIGMA],
            "uncertainty_basis": "assumed",
            "qualifiers": {
                "magnitude_type": r["type"],
                "event_id": r["id"],
                "event_type": "earthquake",
            },
            "provenance": {
                "source": "JMA Seismological Bulletin, hypocenter files",
                "dataset_version": f"{raw.label}, retrieved {raw.retrieved}",
                "license": LICENSE,
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"{r['id']};type={r['letter']};flag={r['flag'] or '-'}",
                "synthetic": False,
            },
            "resolution": {"spatial_km": SPATIAL_KM, "temporal_s": TEMPORAL_S},
        }

    def parse(self, data: bytes, raw: RawFile, ctx: BuildContext) -> Iterator[Token]:
        floor = ctx.region.options(self.name).get("min_magnitude")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            text = b"".join(zf.read(n) for n in sorted(zf.namelist())).decode("ascii", "replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            r = record(line)
            if isinstance(r, str):
                ctx.reject(r)
                continue
            if floor is not None and r["mag"] < float(floor) - 1e-9:
                ctx.reject("below_min_magnitude")
                continue
            if not ctx.region.holds(r["lat"], r["lon"], r["time"]):
                ctx.reject("outside_region")
                continue
            try:
                yield validate_token(self._token(r, raw))
            except (TokenError, ValueError) as exc:
                codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                ctx.reject("invalid:" + ",".join(codes))

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        have = ctx.raw.current(self.name)
        seen: set[str] = set()
        by_year: dict[int, list[Token]] = {}
        for name in files_for(ctx.region):
            raw = have.get(name)
            if raw is None:
                ctx.reject("year_not_fetched")
                continue
            for tok in self.parse(ctx.use(raw), raw, ctx):
                key = tok.qualifiers["event_id"]
                if key in seen:
                    ctx.reject("duplicate")
                    continue
                seen.add(key)
                by_year.setdefault(year_of(tok.time), []).append(tok)
            # A year is complete once the next year's file is read: JST to UTC moves the first
            # nine hours of each file into the year before, never later.
            done = [y for y in by_year if y < year_of(ctx.region.t_end) and name[1:5] > str(y)]
            for y in sorted(done):
                yield sorted(by_year.pop(y), key=lambda t: t.time)
        for y in sorted(by_year):
            yield sorted(by_year[y], key=lambda t: t.time)
