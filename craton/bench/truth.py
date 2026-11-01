"""Truth sources: data a method is graded against and never reads.

Every other dataset in Craton becomes tokens a model can train on. A truth source doesn't: it
answers a question the model was never shown (which way the crust is being squeezed, CB-23), so
it is kept out of the token store altogether. Nothing in ``craton data build`` can read it, and
no region file can list it, which is the strongest leak check there is.

A truth source is downloaded once, checked against the hash of the release it was frozen
from, cached under ``data/raw/<name>/`` like any raw file, and carries a data card
(``docs/data/<name>.md``). Add one by subclassing :class:`TruthSource` and calling
:func:`register_truth`.
"""

from __future__ import annotations

import csv
import io
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from craton.adapters.base import DataCard
from craton.adapters.http import Fetcher, Http
from craton.paths import raw_dir
from craton.store.raw import RawStore

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


class TruthSource(ABC):
    name: str
    version: str = "1.0.0"
    #: The tasks that grade against it.
    graded: str = ""
    #: The one file of the release, and its sha256: a new release is a new source version.
    url: str
    sha256: str
    label: str
    suffix: str

    @abstractmethod
    def card(self) -> DataCard:
        """The data card, as for an adapter."""

    def raw_bytes(self, root: Path, http: Fetcher | None = None, fetch: bool = True) -> bytes:
        """The release file: from the cache, or downloaded once and checked."""
        store = RawStore(raw_dir(root))
        have = store.current(self.name).get(self.label)
        if have is not None and have.sha256 == self.sha256:
            return store.read(have)
        if not fetch:
            raise FileNotFoundError(
                f"{self.name} is not downloaded; `craton bench prepare` fetches it"
            )
        data = (http or Http()).get(self.url)
        entry = store.put(
            self.name,
            self.label,
            self.url,
            data,
            datetime.now(UTC).strftime("%Y-%m-%d"),
            self.suffix,
        )
        if entry.sha256 != self.sha256:
            raise ValueError(
                f"{self.name}: the file at {self.url} hashes to {entry.sha256[:16]}, not the "
                f"frozen release's {self.sha256[:16]}"
            )
        return data


TRUTH_SOURCES: dict[str, TruthSource] = {}


def register_truth(source: TruthSource) -> TruthSource:
    if source.name in TRUTH_SOURCES:
        raise ValueError(f"a truth source called {source.name!r} is already registered")
    TRUTH_SOURCES[source.name] = source
    return source


# ---------------------------------------------------------------- the World Stress Map


#: 1σ of an azimuth by WSM quality (Heidbach et al. 2010: A ±15°, B ±20°, C ±25°).
QUALITY_SIGMA = {"A": 15.0, "B": 20.0, "C": 25.0}


@dataclass(frozen=True)
class StressRecords:
    """WSM records with a usable maximum horizontal stress azimuth (SHmax)."""

    lat: FloatArray
    lon: FloatArray
    azimuth: FloatArray  # degrees east of north, axial: 0 and 180 are the same
    depth_km: FloatArray  # NaN where the record gives none
    quality: np.ndarray[Any, np.dtype[np.str_]]
    kind: np.ndarray[Any, np.dtype[np.str_]]  # FMS, BO, FMF, …
    regime: np.ndarray[Any, np.dtype[np.str_]]  # NF, SS, TF, NS, TS, U

    def __len__(self) -> int:
        return len(self.lat)

    def select(self, mask: np.ndarray[Any, Any]) -> StressRecords:
        return StressRecords(
            self.lat[mask],
            self.lon[mask],
            self.azimuth[mask],
            self.depth_km[mask],
            self.quality[mask],
            self.kind[mask],
            self.regime[mask],
        )


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_wsm(data: bytes) -> StressRecords:
    """Every record of the WSM CSV whose quality is A, B or C and whose azimuth is real.

    D and E records are dropped (WSM itself says they don't constrain the orientation), as are
    the X-qualities (records judged unreliable) and the azimuth 999 (no orientation).
    """
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    cols: dict[str, list[Any]] = {k: [] for k in ("lat", "lon", "azi", "dep", "q", "t", "r")}
    for row in rows:
        q = (row.get("QUALITY") or "").strip()
        azi = _float(row.get("AZI") or "")
        if q not in QUALITY_SIGMA or not (0.0 <= azi <= 180.0):
            continue
        cols["lat"].append(_float(row["LAT"]))
        cols["lon"].append(_float(row["LON"]))
        cols["azi"].append(azi % 180.0)
        cols["dep"].append(_float(row.get("DEPTH") or ""))
        cols["q"].append(q)
        cols["t"].append((row.get("TYPE") or "").strip())
        cols["r"].append((row.get("REGIME") or "").strip() or "U")
    return StressRecords(
        np.asarray(cols["lat"], dtype=np.float64),
        np.asarray(cols["lon"], dtype=np.float64),
        np.asarray(cols["azi"], dtype=np.float64),
        np.asarray(cols["dep"], dtype=np.float64),
        np.asarray(cols["q"], dtype=np.str_),
        np.asarray(cols["t"], dtype=np.str_),
        np.asarray(cols["r"], dtype=np.str_),
    )


class WorldStressMap(TruthSource):
    name = "world_stress_map"
    version = "2025.1"
    graded = "CB-23"
    url = "https://datapub.gfz.de/download/10.5880.WSM.2025.001-Scbwez/WSM_Database_2025.csv"
    sha256 = "3b24aa5b33b251b6c52ee25bcafe3d23d036deb1c38e1caae51f63b808605d1b"
    label = "WSM_Database_2025.csv"
    suffix = ".csv"

    def card(self) -> DataCard:
        return DataCard(
            name=self.name,
            title="World Stress Map Database Release 2025 (truth only, never tokens)",
            provider="World Stress Map project, GFZ Helmholtz Centre for Geosciences, Potsdam",
            homepage="https://www.world-stress-map.org",
            instruments=(),
            license="CC-BY-4.0",
            terms=(
                "Public and free to download and use (GFZ Data Services lists CC BY 4.0). "
                "Cite the database release."
            ),
            citations=(
                "Heidbach, O., Rajabi, M., Di Giacomo, D., Harris, J., Lammers, S., Morawietz, "
                "S., Pierdominici, S., Reiter, K., von Specht, S., Storchak, D., Ziegler, M. O. "
                "(2025). World Stress Map Database Release 2025. GFZ Data Services. "
                "https://doi.org/10.5880/WSM.2025.001",
                "Heidbach, O., Rajabi, M. (2026). Patterns of contemporary horizontal stress "
                "orientation in the Earth's crust derived from the World Stress Map Database "
                "2025. Solid Earth 17(5), 735-745. https://doi.org/10.5194/se-17-735-2026",
            ),
            access=(
                "One CSV (100,842 records, 14 MiB) from GFZ Data Services, downloaded once by "
                "`craton bench prepare`, checked against the release's sha256 and cached."
            ),
            mapping=(
                ("used by", "CB-23 only, as the answer; no model or floor reads it as input"),
                ("azimuth", "`AZI`: SHmax, degrees east of north, axial (0 = 180)"),
                ("quality", "`QUALITY` A, B or C only (1σ 15°, 20°, 25°)"),
                ("depth", "`DEPTH` (km); records deeper than 40 km are left out"),
                (
                    "kind",
                    "`TYPE`: FMS single focal mechanism, FMF formal inversion, BO borehole "
                    "breakouts, HF hydraulic fracturing, GFI fault-slip, …",
                ),
            ),
            uncertainty=(
                "Each record's quality stands for a 1σ of its azimuth: A ±15°, B ±20°, C ±25°. "
                "A single focal mechanism (most records) gives the stress axes only within the "
                "angle between its P axis and the true σ1, which is why WSM rates it C at best."
            ),
            known_issues=(
                "85% of the records are single focal mechanisms, from the same quakes Craton "
                "reads as positions, times and magnitudes. A method sees where quakes were, "
                "never how they slipped, so the mechanism is still unseen data.",
                "Coverage follows seismicity and drilling: dense in Japan and California, a "
                "few hundred records in Alaska or central Italy.",
                "Records within one cell can disagree by more than their quality allows where "
                "the stress turns over a few km (a fault bend, a volcano, a slab).",
            ),
            adapter_version=self.version,
        )

    def records(self, root: Path, http: Fetcher | None = None, fetch: bool = True) -> StressRecords:
        return parse_wsm(self.raw_bytes(root, http, fetch))


WSM = WorldStressMap()
register_truth(WSM)
