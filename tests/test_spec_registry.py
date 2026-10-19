import dataclasses
import json
from pathlib import Path

import pytest

from craton.spec import REGISTRY, Channel, Instrument, Qualifier, Registry, frozen_problems
from craton.spec.instruments import MAGNITUDE_TYPES, SEISMIC_CATALOG


def _probe(name: str = "borehole_strain") -> Instrument:
    return Instrument(
        name=name,
        description="A strainmeter.",
        kinds=("observation",),
        channels=(Channel("strain", "1", "Strain."),),
    )


def test_every_kind_has_an_instrument() -> None:
    kinds = {k for inst in REGISTRY for k in inst.kinds}
    assert kinds == {"observation", "forcing", "property", "event", "derived"}


def test_forces_and_properties_are_both_representable() -> None:
    names = {inst.name for inst in REGISTRY}
    assert "solid_earth_tide" in names
    assert {
        "rock_porosity",
        "rock_permeability",
        "grain_size",
        "lithology",
        "rock_temperature",
        "friction_coefficient",
    } <= names


def test_registering_a_new_instrument_extends_without_touching_the_core() -> None:
    reg = Registry(tuple(REGISTRY))
    reg.register(_probe())
    assert "borehole_strain" in reg
    assert len(reg) == len(REGISTRY) + 1


def test_same_definition_twice_is_fine_a_different_one_is_refused() -> None:
    reg = Registry()
    reg.register(_probe())
    reg.register(_probe())
    changed = dataclasses.replace(_probe(), description="Something else.")
    with pytest.raises(ValueError, match="add-only"):
        reg.register(changed)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Channel("Strain", "1", "bad name"),
        lambda: Channel("strain", "furlong", "unknown unit"),
        lambda: Channel("strain", "1", "empty range", 1, 1),
        lambda: Qualifier("q", "too short a name"),
        lambda: Qualifier("qq", "both", vocabulary=("a",), pattern="a"),
        lambda: Instrument("x_", "no channels", ("event",), ()),
        lambda: Instrument("xx", "bad kind", ("rumour",), (Channel("aa", "1", "a"),)),  # type: ignore[arg-type]
        lambda: Instrument(
            "xx", "repeated", ("event",), (Channel("aa", "1", "a"), Channel("aa", "1", "a"))
        ),
        lambda: Instrument(
            "xx",
            "simplex in metres",
            ("property",),
            (Channel("aa", "m", "a", 0, 1),),
            (),
            "simplex",
        ),
    ],
)
def test_bad_definitions_are_refused(build: object) -> None:
    with pytest.raises(ValueError, match=r"."):
        build()  # type: ignore[operator]


def test_qualifier_checks_vocabulary_and_pattern() -> None:
    mt = SEISMIC_CATALOG.qualifier("magnitude_type")
    assert mt is not None
    assert mt.problem("Mw") is None
    assert mt.problem("M") is not None
    station = Qualifier("station", "code", pattern=r"[A-Z0-9]{4}")
    assert station.problem("ARQT") is None
    assert station.problem("ARQTX") is not None


def _freeze(tmp_path: Path, reg: Registry) -> Path:
    out = tmp_path / "1.0.0.json"
    out.write_text(json.dumps({"spec": "1.0.0", "instruments": reg.snapshot()}), "utf-8")
    return out


def test_frozen_registry_accepts_additions(tmp_path: Path) -> None:
    snap = _freeze(tmp_path, Registry(tuple(REGISTRY)))
    grown = Registry((*REGISTRY, _probe()))
    assert frozen_problems(grown, snap) == []


def test_frozen_registry_catches_removal_and_change(tmp_path: Path) -> None:
    snap = _freeze(tmp_path, Registry(tuple(REGISTRY)))
    others = tuple(i for i in REGISTRY if i.name != "seismic_catalog")
    assert any("removed" in p for p in frozen_problems(Registry(others), snap))
    wider = dataclasses.replace(
        SEISMIC_CATALOG, channels=(Channel("magnitude", "1", "Magnitude.", -5, 10),)
    )
    assert any("changed" in p for p in frozen_problems(Registry((*others, wider)), snap))


def test_a_vocabulary_may_grow_but_not_shrink(tmp_path: Path) -> None:
    snap = _freeze(tmp_path, Registry(tuple(REGISTRY)))
    others = tuple(i for i in REGISTRY if i.name != "seismic_catalog")

    def with_types(types: tuple[str, ...]) -> Registry:
        q = dataclasses.replace(SEISMIC_CATALOG.qualifiers[0], vocabulary=types)
        inst = dataclasses.replace(SEISMIC_CATALOG, qualifiers=(q, *SEISMIC_CATALOG.qualifiers[1:]))
        return Registry((*others, inst))

    assert frozen_problems(with_types((*MAGNITUDE_TYPES, "Mjma")), snap) == []
    assert frozen_problems(with_types(MAGNITUDE_TYPES[1:]), snap) != []
