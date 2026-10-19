"""Data adapters: one module per source, registered by name.

    from craton.adapters import get_adapter
    get_adapter("ingv_catalog").card()

A new source is a subclass of :class:`Adapter` decorated with :func:`register`; see
``CONTRIBUTING.md`` and the three built-ins below.
"""

from __future__ import annotations

from typing import TypeVar

from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport

_A = TypeVar("_A", bound=type[Adapter])
ADAPTERS: dict[str, type[Adapter]] = {}


def register(cls: _A) -> _A:
    """Add an adapter class. A different class under a taken name is refused."""
    existing = ADAPTERS.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(f"adapter {cls.name!r} is already registered by {existing.__module__}")
    ADAPTERS[cls.name] = cls
    return cls


def get_adapter(name: str) -> Adapter:
    if name not in ADAPTERS:
        raise KeyError(f"no adapter {name!r}; registered: {', '.join(sorted(ADAPTERS))}")
    return ADAPTERS[name]()


# The built-ins register themselves on import.
from craton.adapters import ingv, ngl, tides  # noqa: E402

__all__ = [
    "ADAPTERS",
    "Adapter",
    "BuildContext",
    "DataCard",
    "FetchReport",
    "get_adapter",
    "ingv",
    "ngl",
    "register",
    "tides",
]
