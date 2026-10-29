"""The instrument registry: Craton's vocabulary.

An LLM's vocabulary says which token IDs exist. Craton's registry says which *instruments*
exist and what each one measures: its channels (name, unit, allowed range), which token kinds
it may produce, and the qualifiers it carries (a magnitude type, a station code, an intensity
scale). A token's ``value`` vector means nothing without its instrument's entry.

Entries are add-only. Once an instrument ships in a spec release, its channels, units, ranges,
kinds and qualifiers never change; a better definition is a new instrument with a new name.
:func:`frozen_problems` checks that against the snapshot taken at each release.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["observation", "forcing", "property", "event", "derived"]
KINDS: tuple[Kind, ...] = ("observation", "forcing", "property", "event", "derived")

UNITS: Mapping[str, str] = {
    # UCUM codes (https://ucum.org), plus two log units UCUM has no code for.
    "1": "dimensionless (magnitudes, intensities, ratios, probabilities)",
    "m": "metre",
    "mm": "millimetre",
    "km": "kilometre",
    "s": "second",
    "Pa": "pascal",
    "kPa": "kilopascal",
    "MPa": "megapascal",
    "Cel": "degree Celsius",
    "K": "kelvin",
    "mm/a": "millimetres per year",
    "m3": "cubic metre",
    "log10(m2)": "base-10 logarithm of a permeability in square metres",
    "log10(mm)": "base-10 logarithm of a length in millimetres",
}

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")


@dataclass(frozen=True)
class Channel:
    """One number in a token's value vector."""

    name: str
    unit: str
    description: str
    minimum: float = -math.inf
    maximum: float = math.inf

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"channel name {self.name!r} must be snake_case, 2-48 chars")
        if self.unit not in UNITS:
            raise ValueError(
                f"channel {self.name!r}: unit {self.unit!r} is not in craton.spec.UNITS"
            )
        if not self.minimum < self.maximum:
            raise ValueError(f"channel {self.name!r}: minimum must be below maximum")


@dataclass(frozen=True)
class Qualifier:
    """A labelled piece of text a token carries next to its numbers.

    ``vocabulary`` fixes the allowed values (a closed list, add-only); ``pattern`` constrains
    free text instead. A qualifier with neither accepts any short string.
    """

    name: str
    description: str
    required: bool = False
    vocabulary: tuple[str, ...] | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"qualifier name {self.name!r} must be snake_case, 2-48 chars")
        if self.vocabulary is not None and self.pattern is not None:
            raise ValueError(f"qualifier {self.name!r}: give a vocabulary or a pattern, not both")
        if self.pattern is not None:
            re.compile(self.pattern)

    def problem(self, value: str) -> str | None:
        """Why ``value`` is not acceptable for this qualifier, or None."""
        if self.vocabulary is not None and value not in self.vocabulary:
            return f"{value!r} is not one of {list(self.vocabulary)}"
        if self.pattern is not None and not re.fullmatch(self.pattern, value):
            return f"{value!r} does not match {self.pattern}"
        return None


@dataclass(frozen=True)
class Instrument:
    """Everything needed to read a token's value vector."""

    name: str
    description: str
    kinds: tuple[Kind, ...]
    channels: tuple[Channel, ...]
    qualifiers: tuple[Qualifier, ...] = ()
    constraint: Literal["none", "simplex"] = "none"
    """``simplex``: the channels are probabilities over classes and sum to 1."""
    added_in: str = "1.0.0"
    references: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise ValueError(f"instrument name {self.name!r} must be snake_case, 2-48 chars")
        if not self.kinds or any(k not in KINDS for k in self.kinds):
            raise ValueError(f"instrument {self.name!r}: kinds must be a subset of {KINDS}")
        if not self.channels:
            raise ValueError(f"instrument {self.name!r} needs at least one channel")
        if self.constraint == "simplex" and any(
            c.unit != "1" or c.minimum != 0 or c.maximum != 1 for c in self.channels
        ):
            raise ValueError(
                f"instrument {self.name!r}: simplex channels must be unitless in [0, 1]"
            )
        for what, names in (
            ("channel", [c.name for c in self.channels]),
            ("qualifier", [q.name for q in self.qualifiers]),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"instrument {self.name!r} repeats a {what} name")

    def qualifier(self, name: str) -> Qualifier | None:
        return next((q for q in self.qualifiers if q.name == name), None)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe description (infinite ranges become null)."""
        d = asdict(self)
        for ch in d["channels"]:
            for key in ("minimum", "maximum"):
                if math.isinf(ch[key]):
                    ch[key] = None
        out: dict[str, Any] = _jsonable(d)
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


class Registry:
    """A set of instruments, looked up by name. Registering is the only way to add one."""

    def __init__(self, instruments: tuple[Instrument, ...] = ()) -> None:
        self._items: dict[str, Instrument] = {}
        for inst in instruments:
            self.register(inst)

    def register(self, instrument: Instrument) -> Instrument:
        """Add an instrument. The same entry twice is a no-op; a changed one is refused."""
        existing = self._items.get(instrument.name)
        if existing is not None and existing != instrument:
            raise ValueError(
                f"instrument {instrument.name!r} is already registered with a different "
                "definition. Entries are add-only: give the new definition a new name."
            )
        self._items[instrument.name] = instrument
        return instrument

    def get(self, name: str) -> Instrument | None:
        return self._items.get(name)

    def __getitem__(self, name: str) -> Instrument:
        return self._items[name]

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Instrument]:
        return iter(sorted(self._items.values(), key=lambda i: i.name))

    def __len__(self) -> int:
        return len(self._items)

    def snapshot(self) -> dict[str, Any]:
        """Every entry as plain JSON, sorted by name: what a spec release freezes."""
        return {inst.name: inst.as_dict() for inst in self}


def frozen_problems(registry: Registry, snapshot_path: Path) -> list[str]:
    """Everything in ``registry`` that breaks the add-only promise made by a released snapshot.

    A frozen instrument that disappeared or changed in any way is a problem. New instruments
    are fine: that is how the vocabulary grows.
    """
    frozen = json.loads(snapshot_path.read_text(encoding="utf-8"))
    current = registry.snapshot()
    problems = []
    for name, entry in frozen["instruments"].items():
        if name not in current:
            problems.append(f"instrument {name!r} was removed (frozen in spec {frozen['spec']})")
        elif _without_vocabulary_growth(current[name], entry) != entry:
            problems.append(f"instrument {name!r} changed after spec {frozen['spec']} froze it")
    return problems


def _without_vocabulary_growth(current: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """``current`` with each qualifier vocabulary trimmed to its frozen list, if it only grew.

    A closed vocabulary may gain words (a new magnitude type) in a minor release; losing or
    renaming one is still a change.
    """
    before = {q["name"]: q for q in frozen["qualifiers"]}
    qualifiers = []
    for q in current["qualifiers"]:
        old = before.get(q["name"])
        grew = (
            old is not None
            and old["vocabulary"] is not None
            and q["vocabulary"] is not None
            and q["vocabulary"][: len(old["vocabulary"])] == old["vocabulary"]
        )
        qualifiers.append({**q, "vocabulary": old["vocabulary"]} if grew and old else q)
    return {**current, "qualifiers": qualifiers}
