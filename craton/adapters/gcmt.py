"""The Global CMT catalogue: how every large quake on Earth since 1976 slipped.

A catalogue says where, when and how big. A moment tensor says how: which way the rock on
each side moved, as six numbers (a symmetric 3×3 tensor). The Global Centroid-Moment-Tensor
project (Dziewonski et al. 1981; Ekström et al. 2012) computes one for nearly every quake of
about Mw 5 and up, worldwide, from long-period waves, and publishes them as "ndk" files: five
80-character lines per quake (``allorder.ndk_explained`` on their site).

One file serves the whole Earth, so the region says which release and checks its hash:

    sources:
      gcmt:
        url: https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk
        sha256: baed5ad2...
        license: LicenseRef-GCMT-cite

One ``moment_tensor`` token per solution, placed at the centroid (where and when the
inversion puts the middle of the slip), elements in N·m in the r (up), t (south), p (east)
frame GCMT uses. A token says nothing about which nodal plane slipped: the tensor doesn't know.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport, Log
from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.spec.timeaxis import to_iso, to_seconds, year_of
from craton.spec.token import Token, TokenError, validate_token, wrap_lon
from craton.store.raw import RawFile, RawStore

SPEC = "1.3.0"
#: dyne·cm to N·m.
DYNE_CM = 1e-7
KM_PER_DEG = 111.195
#: An element the inversion held fixed has a standard error of 0; it is given this share of
#: the scalar moment instead (the spec wants every 1σ positive).
CONSTRAINED_SHARE = 0.01
SOURCE_TYPES = {"0": "general", "1": "deviatoric", "2": "double_couple"}
DEPTH_TYPES = {"FREE": "free", "FIX": "fixed", "BDY": "waveform"}
REQUIRED = ("url", "sha256", "license")
#: Line 3's fields (0-based, end excluded): time shift, σ, lat, σ, lon, σ, depth, σ.
CENTROID_COLUMNS = ((9, 18), (18, 22), (22, 29), (29, 34), (34, 42), (42, 47), (47, 53), (53, 58))


def options(region: Region) -> dict[str, Any]:
    opts = region.options(GlobalCMT.name)
    for key in REQUIRED:
        if key not in opts:
            raise ValueError(f"region {region.name}: sources.gcmt needs {key!r}")
    return opts


def moment_magnitude(m0_nm: float) -> float:
    """Mw from the scalar moment in N·m (Hanks & Kanamori 1979, IASPEI's 9.1)."""
    return (2.0 / 3.0) * (math.log10(m0_nm) - 9.1)


def rupture_km(mw: float) -> float:
    """Subsurface rupture length, all slip types (Wells & Coppersmith 1994): the token's size."""
    return float(10 ** (-2.44 + 0.59 * mw))


@dataclass(frozen=True)
class Solution:
    """One ndk entry, read but not yet checked."""

    event_id: str
    time: float
    time_sigma_s: float
    lat: float
    lat_sigma_deg: float
    lon: float
    lon_sigma_deg: float
    depth_km: float
    depth_sigma_km: float
    depth_type: str
    source_type: str
    half_duration_s: float
    elements: tuple[float, ...]  # rr, tt, pp, rt, rp, tp in N·m
    sigmas: tuple[float, ...]
    moment: float  # N·m
    line: int  # first line of the entry in the file, from 1


def _reference_time(date: str, clock: str) -> float:
    """The hypocentre's time. ndk writes seconds up to 60.0, so add them as a number."""
    day = datetime.strptime(date.strip(), "%Y/%m/%d").replace(tzinfo=UTC)
    hh, mm, ss = clock.strip().split(":")
    return to_seconds(day) + int(hh) * 3600 + int(mm) * 60 + float(ss)


def parse_ndk(text: str) -> Iterator[Solution | str]:
    """Every entry of an ndk file; an entry that can't be read yields a short reason instead."""
    lines = text.splitlines()
    for start in range(0, len(lines) - len(lines) % 5, 5):
        a, b, c, d, e = (ln.ljust(80) for ln in lines[start : start + 5])
        try:
            t0 = _reference_time(a[5:15], a[16:26])
            # Fixed columns: two entries write "1.060.0" with no space between the numbers.
            cen = [c[i:j] for i, j in CENTROID_COLUMNS]
            exp = int(d[0:2])
            vals = [float(x) for x in d[2:80].split()]
            if len(vals) != 12:
                raise ValueError("short line")
            scale = 10.0**exp * DYNE_CM
            src = b[62:68].replace("CMT:", "").strip()
            yield Solution(
                event_id=b[0:16].strip(),
                time=t0 + float(cen[0]),
                time_sigma_s=float(cen[1]),
                lat=float(cen[2]),
                lat_sigma_deg=float(cen[3]),
                lon=wrap_lon(float(cen[4])),
                lon_sigma_deg=float(cen[5]),
                depth_km=float(cen[6]),
                depth_sigma_km=float(cen[7]),
                depth_type=DEPTH_TYPES.get(c[59:63].strip(), ""),
                source_type=SOURCE_TYPES.get(src, ""),
                half_duration_s=float(b[75:80] or 0.0),
                elements=tuple(v * scale for v in vals[0::2]),
                sigmas=tuple(v * scale for v in vals[1::2]),
                moment=float(e[49:56]) * scale,
                line=start + 1,
            )
        except (ValueError, IndexError):
            yield "unparseable"


@register
class GlobalCMT(Adapter):
    name = "gcmt"
    version = "1.0.0"
    instruments = ("moment_tensor",)

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="Global Centroid-Moment-Tensor catalogue (one ndk release)",
            provider="The Global CMT Project (G. Ekström, M. Nettles), Lamont-Doherty",
            homepage="https://www.globalcmt.org",
            instruments=self.instruments,
            license="LicenseRef-GCMT-cite",
            terms=(
                "Free to download and use; the project asks that both reference papers be "
                "cited. No standard licence is stated, so Craton reads the file and never "
                "redistributes it."
            ),
            citations=(
                "Dziewonski, A. M., Chou, T.-A., Woodhouse, J. H. (1981). Determination of "
                "earthquake source parameters from waveform data for studies of global and "
                "regional seismicity. JGR 86, 2825-2852. doi:10.1029/JB086iB04p02825",
                "Ekström, G., Nettles, M., Dziewonski, A. M. (2012). The global CMT project "
                "2004-2010: Centroid-moment tensors for 13,017 earthquakes. PEPI 200-201, 1-9. "
                "doi:10.1016/j.pepi.2012.04.002",
            ),
            access=(
                "One ndk file per release from the project's Lamont-Doherty server (the "
                "region names the URL and its sha256), downloaded once and cached by hash."
            ),
            mapping=(
                ("kind", "`event`"),
                ("lat, lon, depth_km", "the centroid (line 3), not the hypocentre"),
                ("time", "the hypocentre's time (line 1) plus the centroid time shift"),
                ("value", "Mrr, Mtt, Mpp, Mrt, Mrp, Mtp and M0 (lines 4-5), dyne·cm → N·m"),
                ("uncertainty", "each element's standard error (line 4); M0 its largest"),
                ("location_uncertainty", "the centroid's standard errors, where non-zero"),
                ("qualifiers", "event_id (the CMT name), source_type, depth_type"),
                (
                    "resolution",
                    "rupture length from Mw (Wells & Coppersmith 1994); 2× half duration",
                ),
            ),
            uncertainty=(
                "Standard errors as GCMT reports them (formal, from the inversion: smaller "
                "than the true error). An element the inversion held fixed reports 0 and gets "
                f"{CONSTRAINED_SHARE:.0%} of M0 instead."
            ),
            known_issues=(
                "Complete from about Mw 5.5 worldwide (smaller where stations are dense); "
                "thin before 1977.",
                "Shallow quakes: Mrt and Mrp are poorly resolved and often fixed to zero.",
                "Centroid depths are below sea level, read here as below the ellipsoid (a few "
                "tens of metres apart at most, far below the depth error).",
                "Many World Stress Map records are single GCMT mechanisms: a task graded "
                "against WSM must leave its quake-based records out (CB-34 does).",
            ),
            adapter_version=self.version,
        )

    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        report = FetchReport()
        opts = options(region)
        label = str(opts["url"]).rsplit("/", 1)[-1]
        have = raw.current(self.name).get(label)
        if have is not None and have.sha256 == opts["sha256"] and not refresh:
            report.cached += 1
            return report
        data = http.get(str(opts["url"]))
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        entry = raw.put(self.name, label, str(opts["url"]), data, today, ".ndk")
        if entry.sha256 != opts["sha256"]:
            raise ValueError(f"{label}: sha256 {entry.sha256[:16]} is not {opts['sha256'][:16]}")
        report.downloaded += 1
        report.bytes += len(data)
        log(f"{label}  {len(data) / 2**20:.1f} MiB")
        return report

    def _token(self, s: Solution, raw: RawFile, license: str) -> dict[str, Any]:
        sig = [x if x > 0 else CONSTRAINED_SHARE * s.moment for x in s.sigmas]
        loc: dict[str, float] = {}
        if s.lat_sigma_deg > 0 or s.lon_sigma_deg > 0:
            loc["horizontal_km"] = KM_PER_DEG * max(s.lat_sigma_deg, s.lon_sigma_deg)
        if s.depth_sigma_km > 0:
            loc["vertical_km"] = s.depth_sigma_km
        if s.time_sigma_s > 0:
            loc["time_s"] = s.time_sigma_s
        quals = {"event_id": s.event_id}
        if s.source_type:
            quals["source_type"] = s.source_type
        if s.depth_type:
            quals["depth_type"] = s.depth_type
        tok: dict[str, Any] = {
            "spec": SPEC,
            "kind": "event",
            "instrument": "moment_tensor",
            "lat": s.lat,
            "lon": s.lon,
            "depth_km": s.depth_km,
            "time": s.time,
            "time_iso": to_iso(s.time),
            "value": [*s.elements, s.moment],
            "uncertainty": [*sig, max(sig)],
            "uncertainty_basis": "reported",
            "qualifiers": quals,
            "provenance": {
                "source": "Global CMT Project",
                "dataset_version": f"{raw.label}, retrieved {raw.retrieved}",
                "license": license,
                "adapter_version": self.version,
                "content_hash": raw.content_hash,
                "record": f"line {s.line}",
                "synthetic": False,
            },
            "resolution": {
                "spatial_km": max(1.0, rupture_km(moment_magnitude(s.moment))),
                "temporal_s": max(1.0, 2.0 * s.half_duration_s),
            },
        }
        if loc:
            tok["location_uncertainty"] = loc
        return tok

    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        opts = options(ctx.region)
        raw = ctx.raw.current(self.name).get(str(opts["url"]).rsplit("/", 1)[-1])
        if raw is None:
            ctx.reject("file_not_fetched")
            return
        by_year: dict[int, list[Token]] = {}
        seen: set[str] = set()
        for s in parse_ndk(ctx.use(raw).decode("ascii", "replace")):
            if isinstance(s, str):
                ctx.reject(s)
                continue
            if s.event_id in seen:
                ctx.reject("duplicate")
                continue
            try:
                tok = validate_token(self._token(s, raw, str(opts["license"])))
            except (TokenError, ValueError) as exc:
                codes = sorted(exc.codes) if isinstance(exc, TokenError) else ["unparseable"]
                ctx.reject("invalid:" + ",".join(codes))
                continue
            if not ctx.region.holds(tok.lat, tok.lon, tok.time):
                ctx.reject("outside_region")
                continue
            seen.add(s.event_id)
            by_year.setdefault(year_of(tok.time), []).append(tok)
        for yr in sorted(by_year):
            yield sorted(by_year[yr], key=lambda tk: (tk.time, tk.qualifiers["event_id"]))
