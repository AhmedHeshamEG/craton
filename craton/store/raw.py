"""Raw downloads, kept exactly as the provider sent them, named by their own hash.

    data/raw/ingv_catalog/3f/3f9a1c0b...e2d4.txt
    data/raw/ingv_catalog/manifest.jsonl

Every token's ``provenance.content_hash`` is the sha256 of the raw file it was read from, so any
token leads back to these bytes. The manifest is append-only: one line per *new* version of a
request ("the INGV catalogue for October 2016", "station ARQT"). Fetching the same bytes again
adds nothing and keeps the first retrieval date, so a rebuild from the same downloads gives the
same tokens, byte for byte. A provider revising its data gives new bytes, a new hash and a new
line; the old file stays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawFile:
    """One downloaded file and where it came from."""

    source: str
    """The adapter that fetched it."""
    label: str
    """What it answers, stable across re-fetches: ``2016-10``, ``ARQT``, ``stations``."""
    url: str
    sha256: str
    size: int
    retrieved: str
    """UTC date it was first fetched with these exact bytes, ``YYYY-MM-DD``."""
    suffix: str

    @property
    def content_hash(self) -> str:
        """The form tokens carry in ``provenance.content_hash``."""
        return "sha256:" + self.sha256

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class RawStore:
    """Content-addressed raw files plus an append-only manifest per source."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _folder(self, source: str) -> Path:
        return self.root / source

    def path_of(self, f: RawFile) -> Path:
        return self._folder(f.source) / f.sha256[:2] / f"{f.sha256}{f.suffix}"

    def manifest_path(self, source: str) -> Path:
        return self._folder(source) / "manifest.jsonl"

    def entries(self, source: str) -> list[RawFile]:
        """Every manifest line for ``source``, oldest first."""
        path = self.manifest_path(source)
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [RawFile(**json.loads(line)) for line in lines if line.strip()]

    def current(self, source: str) -> dict[str, RawFile]:
        """The newest version of every label."""
        latest: dict[str, RawFile] = {}
        for entry in self.entries(source):
            latest[entry.label] = entry
        return latest

    def put(
        self, source: str, label: str, url: str, data: bytes, retrieved: str, suffix: str
    ) -> RawFile:
        """Store ``data`` as the current version of ``label``. Same bytes again: a no-op."""
        digest = hashlib.sha256(data).hexdigest()
        now = self.current(source).get(label)
        if now is not None and now.sha256 == digest:
            return now
        known = next((e for e in self.entries(source) if e.sha256 == digest), None)
        entry = RawFile(
            source=source,
            label=label,
            url=url,
            sha256=digest,
            size=len(data),
            retrieved=known.retrieved if known else retrieved,
            suffix=suffix,
        )
        path = self.path_of(entry)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        manifest = self.manifest_path(source)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8", newline="\n") as f:
            f.write(entry.to_json() + "\n")
        return entry

    def read(self, f: RawFile) -> bytes:
        """The file's bytes, checked against its hash."""
        data = self.path_of(f).read_bytes()
        if hashlib.sha256(data).hexdigest() != f.sha256:
            raise ValueError(f"raw file {f.label} ({f.sha256[:12]}) no longer matches its hash")
        return data
