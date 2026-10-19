"""The validator suite: every good token passes, every malformed one fails with the right code."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from typing import Any

import pytest

from craton.spec import Token, TokenError, validate_token
from craton.spec.examples import EXAMPLES, example

Mutation = Callable[[dict[str, Any]], None]


def setter(path: str, value: Any) -> Mutation:
    def apply(d: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        for p in parents:
            d = d[p]
        d[last] = value

    return apply


def dropper(path: str) -> Mutation:
    def apply(d: dict[str, Any]) -> None:
        *parents, last = path.split(".")
        for p in parents:
            d = d[p]
        del d[last]

    return apply


NAN, INF = math.nan, math.inf

# (base example, mutation, expected code, why it must fail)
CASES: list[tuple[str, Mutation, str, str]] = [
    # where
    ("norcia_2016", setter("lat", 91.0), "lat_range", "north of the pole"),
    ("norcia_2016", setter("lat", -90.0001), "lat_range", "south of the pole"),
    ("norcia_2016", setter("lat", NAN), "not_finite", "NaN latitude"),
    ("norcia_2016", setter("lon", 180.0), "lon_range", "180 must be written -180"),
    ("norcia_2016", setter("lon", -180.5), "lon_range", "west of the dateline"),
    ("norcia_2016", setter("lon", -346.8), "lon_range", "NGL-style unwrapped longitude"),
    ("norcia_2016", setter("lon", INF), "not_finite", "infinite longitude"),
    ("norcia_2016", setter("depth_km", -11.0), "depth_range", "higher than any mountain"),
    ("norcia_2016", setter("depth_km", 6372.0), "depth_range", "deeper than the centre"),
    ("norcia_2016", setter("depth_km", "10"), "wrong_type", "depth as text"),
    ("norcia_2016", dropper("lat"), "missing_field", "no latitude"),
    # when
    ("norcia_2016", setter("time", 1e18), "time_range", "older than the Earth"),
    ("norcia_2016", setter("time", NAN), "not_finite", "NaN time"),
    ("norcia_2016", setter("time", "2016-10-30"), "wrong_type", "time must be seconds"),
    ("norcia_2016", setter("time_iso", "2016-10-30T07:40:17Z"), "time_mismatch", "an hour off"),
    ("norcia_2016", setter("time_iso", "2016-10-30T06:40:17"), "time_format", "no timezone"),
    ("norcia_2016", setter("time_iso", "yesterday"), "time_format", "not ISO 8601"),
    # what
    ("norcia_2016", setter("kind", "quake"), "bad_choice", "not a kind"),
    ("norcia_2016", setter("kind", "forcing"), "kind_mismatch", "a catalogue makes events"),
    ("norcia_2016", setter("instrument", "seismometer_9000"), "unknown_instrument", "unknown"),
    ("norcia_2016", setter("instrument", 7), "wrong_type", "instrument as a number"),
    ("norcia_2016", dropper("kind"), "missing_field", "no kind"),
    # value
    ("norcia_2016", setter("value", []), "empty", "no value"),
    ("norcia_2016", setter("value", [6.5, 1.0]), "value_length", "one channel, two values"),
    ("norcia_2016", setter("value", [11.0]), "value_range", "magnitude 11"),
    ("norcia_2016", setter("value", [-3.5]), "value_range", "below the scale"),
    ("norcia_2016", setter("value", ["6.5"]), "wrong_type", "magnitude as text"),
    ("norcia_2016", setter("value", [True]), "wrong_type", "a bool is not a number here"),
    ("norcia_2016", setter("value", [NAN]), "not_finite", "NaN value"),
    ("norcia_2016", setter("value", 6.5), "wrong_type", "a scalar, not a list"),
    # uncertainty
    ("norcia_2016", setter("uncertainty", [0.0]), "uncertainty_nonpositive", "perfect"),
    ("norcia_2016", setter("uncertainty", [-0.1]), "uncertainty_nonpositive", "negative"),
    ("norcia_2016", setter("uncertainty", []), "empty", "no uncertainty"),
    ("norcia_2016", setter("uncertainty", [0.1, 0.1]), "uncertainty_length", "too many"),
    ("norcia_2016", setter("uncertainty", [INF]), "not_finite", "infinite sigma"),
    ("norcia_2016", dropper("uncertainty"), "missing_field", "uncertainty is required"),
    ("norcia_2016", setter("uncertainty_basis", "guess"), "bad_choice", "not a basis"),
    ("norcia_2016", dropper("uncertainty_basis"), "missing_field", "basis is required"),
    ("norcia_2016", setter("location_uncertainty", {}), "empty", "empty location uncertainty"),
    (
        "norcia_2016",
        setter("location_uncertainty.horizontal_km", 0.0),
        "uncertainty_nonpositive",
        "a perfectly located quake",
    ),
    ("norcia_2016", setter("location_uncertainty.depth", 1.0), "unknown_field", "typo'd key"),
    # qualifiers
    ("norcia_2016", setter("qualifiers", {}), "missing_qualifier", "magnitude type required"),
    ("norcia_2016", setter("qualifiers.magnitude_type", "M"), "qualifier_value", "which M?"),
    ("norcia_2016", setter("qualifiers.colour", "red"), "unknown_qualifier", "not declared"),
    ("norcia_2016", setter("qualifiers.event_id", ""), "qualifier_value", "empty label"),
    ("norcia_2016", setter("qualifiers.event_id", 8863681), "wrong_type", "label as number"),
    ("arquata_gnss", setter("qualifiers.station", "arqt"), "qualifier_value", "lowercase code"),
    ("arquata_gnss", dropper("qualifiers.reference_frame"), "missing_qualifier", "frame needed"),
    ("arquata_gnss", setter("value", [1.0, 2.0]), "value_length", "GNSS has three channels"),
    ("arquata_gnss", setter("uncertainty", [0.001]), "uncertainty_length", "one sigma for three"),
    # provenance
    ("norcia_2016", setter("provenance.source", ""), "empty", "anonymous data"),
    ("norcia_2016", setter("provenance.license", "Creative Commons BY"), "license", "not SPDX"),
    ("norcia_2016", setter("provenance.adapter_version", "1.0"), "semver", "not semver"),
    ("norcia_2016", setter("provenance.content_hash", "md5:abc"), "content_hash", "not sha256"),
    (
        "norcia_2016",
        setter("provenance.content_hash", "sha256:" + "A" * 64),
        "content_hash",
        "uppercase hex",
    ),
    ("norcia_2016", setter("provenance.synthetic", "no"), "wrong_type", "text for a flag"),
    ("norcia_2016", dropper("provenance.synthetic"), "missing_field", "real or simulated?"),
    ("norcia_2016", setter("provenance.author", "me"), "unknown_field", "undeclared key"),
    ("norcia_2016", dropper("provenance"), "missing_field", "no provenance"),
    # resolution
    ("norcia_2016", setter("resolution.spatial_km", 0.0), "resolution", "zero-size support"),
    ("norcia_2016", setter("resolution.spatial_km", 30_000.0), "resolution", "bigger than Earth"),
    ("norcia_2016", setter("resolution.temporal_s", -1.0), "resolution", "negative duration"),
    ("norcia_2016", dropper("resolution"), "missing_field", "no resolution"),
    # versions and structure
    ("norcia_2016", setter("spec", "2.0.0"), "spec_version", "a future major version"),
    ("norcia_2016", setter("spec", "1.99.0"), "spec_version", "a newer minor than this code"),
    ("norcia_2016", setter("spec", "one"), "spec_version", "not a version"),
    ("norcia_2016", setter("magnitude", 6.5), "unknown_field", "fields can't be invented"),
    # kind-specific
    ("limestone_lithology", setter("value", [0.5] + [0.0] * 12), "simplex", "sums to 0.5"),
    ("limestone_lithology", setter("value", [1.2] + [0.0] * 12), "value_range", "p > 1"),
    ("montecassino_1349", setter("value", [13.0]), "value_range", "intensity 13 doesn't exist"),
    ("montecassino_1349", setter("qualifiers.scale", "Richter"), "qualifier_value", "not a scale"),
    ("fault_friction", setter("kind", "event"), "kind_mismatch", "friction is a property"),
]


def _base(name: str) -> dict[str, Any]:
    return copy.deepcopy(EXAMPLES[name]())


def test_the_suite_is_big_enough() -> None:
    assert len(CASES) >= 50


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_is_valid(name: str) -> None:
    assert isinstance(example(name), Token)


@pytest.mark.parametrize(
    ("base", "mutate", "code", "why"), CASES, ids=[f"{c[2]}:{c[3]}" for c in CASES]
)
def test_malformed_token_fails_with_its_code(
    base: str, mutate: Mutation, code: str, why: str
) -> None:
    data = _base(base)
    mutate(data)
    with pytest.raises(TokenError) as exc:
        validate_token(data)
    assert code in exc.value.codes, f"{why}: got {exc.value}"


def test_json_text_and_dict_give_the_same_token() -> None:
    data = _base("norcia_2016")
    assert validate_token(json.dumps(data)) == validate_token(data)


def test_json_text_with_nan_is_refused() -> None:
    text = json.dumps(_base("norcia_2016")).replace('"lat": 42.8303', '"lat": NaN')
    with pytest.raises(TokenError):
        validate_token(text)


def test_tokens_are_immutable() -> None:
    tok = example("norcia_2016")
    with pytest.raises(ValueError, match="frozen"):
        tok.lat = 0.0  # type: ignore[misc]


def test_token_hash_is_stable_and_sensitive() -> None:
    a = example("norcia_2016")
    data = _base("norcia_2016")
    data["value"] = [6.6]
    b = validate_token(data)
    assert a.token_hash() == example("norcia_2016").token_hash()
    assert a.token_hash() != b.token_hash()
    assert a.token_hash().startswith("sha256:")


def test_the_readable_time_copy_does_not_change_identity() -> None:
    data = _base("norcia_2016")
    del data["time_iso"]
    assert validate_token(data).token_hash() == example("norcia_2016").token_hash()


def test_int_coordinates_are_numbers_too() -> None:
    data = _base("norcia_2016")
    data["depth_km"] = 10
    assert validate_token(data).depth_km == 10.0


def test_errors_list_every_problem_with_field_and_message() -> None:
    data = _base("norcia_2016")
    data["lat"] = 95.0
    data["lon"] = 200.0
    with pytest.raises(TokenError) as exc:
        validate_token(data)
    assert {"lat_range", "lon_range"} <= exc.value.codes
    assert {p.field for p in exc.value.problems} >= {"lat", "lon"}
