"""What every adapter is: fetch raw files, turn them into tokens, describe itself in a card.

An adapter is the only code that knows one provider's quirks. Everything after it sees tokens.

    fetch()   provider → raw files, cached by hash (craton.store.raw)
    tokens()  raw files (or other adapters' tokens) → validated tokens, grouped by year
    card()    the data card: who made the data, licence, citation, coverage, known issues

Adding a source means writing one subclass and registering it (``craton.adapters.register``);
nothing in the store, the spec or the model changes. The three built-ins are the pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from craton.adapters.http import Fetcher
from craton.adapters.region import Region
from craton.spec.token import Token
from craton.store.raw import RawFile, RawStore

Log = Callable[[str], None]


@dataclass(frozen=True)
class DataCard:
    """Everything a stranger needs before trusting, citing or redistributing a dataset."""

    name: str
    title: str
    provider: str
    homepage: str
    instruments: tuple[str, ...]
    license: str
    """SPDX identifier, or a ``LicenseRef-`` when the provider states no standard licence."""
    terms: str
    """The provider's own terms, in plain words."""
    citations: tuple[str, ...]
    access: str
    """Endpoint, format, limits, and how this adapter stays polite."""
    mapping: tuple[tuple[str, str], ...]
    """(token slot, where its value comes from)."""
    uncertainty: str
    known_issues: tuple[str, ...]
    adapter_version: str


@dataclass
class FetchReport:
    downloaded: int = 0
    cached: int = 0
    bytes: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class BuildContext:
    """What ``tokens()`` gets: the region, the raw store, upstream token files, a reject log."""

    region: Region
    raw: RawStore
    inputs: dict[str, list[Path]] = field(default_factory=dict)
    rejects: Counter[str] = field(default_factory=Counter)
    raw_used: dict[str, RawFile] = field(default_factory=dict)

    def reject(self, reason: str, n: int = 1) -> None:
        """Count a record that did not become a token, by a short stable reason."""
        self.rejects[reason] += n

    def use(self, f: RawFile) -> bytes:
        """Read a raw file and remember it went into this build."""
        self.raw_used[f.sha256] = f
        return self.raw.read(f)


class Adapter(ABC):
    """One data source."""

    name: ClassVar[str]
    version: ClassVar[str]
    """Semantic version of the adapter; written into every token it makes."""
    instruments: ClassVar[tuple[str, ...]]
    requires: ClassVar[tuple[str, ...]] = ()
    """Adapters whose tokens this one reads (built first)."""

    @abstractmethod
    def card(self) -> DataCard:
        """The data card."""

    @abstractmethod
    def fetch(
        self, region: Region, raw: RawStore, http: Fetcher, refresh: bool, log: Log
    ) -> FetchReport:
        """Download what the region needs into ``raw``. Cached labels are skipped unless
        ``refresh``. Adapters that compute rather than download return an empty report."""

    @abstractmethod
    def tokens(self, ctx: BuildContext) -> Iterator[list[Token]]:
        """Yield the region's tokens, one list per year, each list free of duplicates."""
