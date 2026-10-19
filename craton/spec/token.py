"""Token Spec 1.0.0: one shape for every measurement on Earth.

    kind · instrument · lat, lon, depth · time · value · uncertainty · provenance · resolution

A seismometer reading, a day of GPS, the pull of the Moon, a rock's porosity and a line in a
medieval chronicle all become this. The instrument's registry entry says what the numbers in
``value`` mean; everything else is the same for all of them.

Every ``Token`` that exists has passed validation: building one any other way is impossible,
because the model checks itself against the instrument registry when it is created. Failures
carry a stable error code (``lat_range``, ``uncertainty_nonpositive``, ...) so tests and data
reports can count them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from craton.spec.instruments import REGISTRY
from craton.spec.registry import Kind, Registry
from craton.spec.timeaxis import EARTH_AGE_S, to_seconds

SPEC_VERSION = "1.0.0"
EARTH_RADIUS_KM = 6371.0
HALF_CIRCUMFERENCE_KM = 20_038.0
MAX_ABOVE_ELLIPSOID_KM = 10.0
"""Everest's summit is under 9 km above the ellipsoid; nothing solid sits higher."""

SIMPLEX_TOLERANCE = 1e-6
TIME_ISO_TOLERANCE_S = 1e-3

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_LICENSE = re.compile(r"^(LicenseRef-)?[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")

UncertaintyBasis = Literal["reported", "derived", "assumed"]


def _fail(code: str, message: str, **ctx: Any) -> PydanticCustomError:
    return PydanticCustomError(code, message, ctx or None)


def _finite(v: float, what: str) -> float:
    if not math.isfinite(v):
        raise _fail("not_finite", "{what} must be a finite number", what=what)
    return v


class _Part(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Provenance(_Part):
    """Where a token came from, precisely enough to fetch the same bytes again."""

    source: str = Field(min_length=1, max_length=256, description="Who published the data.")
    dataset_version: str = Field(min_length=1, max_length=64)
    license: str = Field(description="SPDX identifier, e.g. CC-BY-4.0.")
    adapter_version: str = Field(description="Semantic version of the code that made the token.")
    content_hash: str = Field(description="sha256 of the raw file or record it was read from.")
    record: str | None = Field(default=None, min_length=1, max_length=256)
    synthetic: bool = Field(description="True for anything a simulator made. Never mixed silently.")

    @field_validator("license")
    @classmethod
    def _license(cls, v: str) -> str:
        if not _LICENSE.match(v):
            raise _fail("license", "license must be an SPDX identifier, got {v!r}", v=v)
        return v

    @field_validator("adapter_version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not _SEMVER.match(v):
            raise _fail("semver", "adapter_version must be MAJOR.MINOR.PATCH, got {v!r}", v=v)
        return v

    @field_validator("content_hash")
    @classmethod
    def _hash(cls, v: str) -> str:
        if not _SHA256.match(v):
            raise _fail("content_hash", "content_hash must be 'sha256:' + 64 lowercase hex")
        return v


class Resolution(_Part):
    """The support of the measurement: how big a patch of space and time one value describes."""

    spatial_km: float = Field(description="Size of the area the value stands for.")
    temporal_s: float = Field(description="Length of the time window the value stands for.")

    @field_validator("spatial_km")
    @classmethod
    def _space(cls, v: float) -> float:
        _finite(v, "spatial_km")
        if not 0 < v <= HALF_CIRCUMFERENCE_KM:
            raise _fail("resolution", "spatial_km must be in (0, 20038], got {v}", v=v)
        return v

    @field_validator("temporal_s")
    @classmethod
    def _time(cls, v: float) -> float:
        _finite(v, "temporal_s")
        if not 0 < v <= 2 * EARTH_AGE_S:
            raise _fail("resolution", "temporal_s must be positive, got {v}", v=v)
        return v


class LocationUncertainty(_Part):
    """1σ uncertainty of *where and when*; ``uncertainty`` is about the value."""

    horizontal_km: float | None = None
    vertical_km: float | None = None
    time_s: float | None = None

    @field_validator("horizontal_km", "vertical_km", "time_s")
    @classmethod
    def _positive(cls, v: float | None, info: ValidationInfo) -> float | None:
        if v is None:
            return v
        _finite(v, str(info.field_name))
        if v <= 0:
            raise _fail("uncertainty_nonpositive", "{f} must be > 0", f=info.field_name)
        return v

    @model_validator(mode="after")
    def _something(self) -> LocationUncertainty:
        if self.horizontal_km is None and self.vertical_km is None and self.time_s is None:
            raise _fail("empty", "location_uncertainty needs at least one entry; omit it instead")
        return self


class Token(BaseModel):
    """One measurement. See the module docstring and ``SPEC_POLICY.md``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    spec: str = Field(default=SPEC_VERSION, description="Token Spec version it was written under.")
    kind: Kind
    instrument: str
    lat: float = Field(description="WGS84 latitude, degrees, [-90, 90].")
    lon: float = Field(description="WGS84 longitude, degrees, [-180, 180).")
    depth_km: float = Field(description="Below the WGS84 ellipsoid, km; negative is above it.")
    time: float = Field(description="Seconds since 1970-01-01T00:00:00Z, proleptic Gregorian.")
    time_iso: str | None = Field(default=None, description="The same moment in ISO 8601, if any.")
    value: list[float] = Field(min_length=1)
    uncertainty: list[float] = Field(min_length=1, description="1σ per value channel, > 0.")
    uncertainty_basis: UncertaintyBasis
    location_uncertainty: LocationUncertainty | None = None
    qualifiers: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance
    resolution: Resolution

    # ---------------------------------------------------------- one field at a time

    @field_validator("spec")
    @classmethod
    def _spec(cls, v: str) -> str:
        m = _SEMVER.match(v)
        cur = tuple(int(x) for x in SPEC_VERSION.split("."))
        if not m or int(m[1]) != cur[0] or int(m[2]) > cur[1]:
            raise _fail(
                "spec_version",
                "token written under spec {v!r}; this code reads {cur}.x up to {cur_full}",
                v=v,
                cur=cur[0],
                cur_full=SPEC_VERSION,
            )
        return v

    @field_validator("lat")
    @classmethod
    def _lat(cls, v: float) -> float:
        _finite(v, "lat")
        if not -90 <= v <= 90:
            raise _fail("lat_range", "lat must be in [-90, 90], got {v}", v=v)
        return v

    @field_validator("lon")
    @classmethod
    def _lon(cls, v: float) -> float:
        _finite(v, "lon")
        if not -180 <= v < 180:
            raise _fail(
                "lon_range",
                "lon must be in [-180, 180), got {v}; wrap it first (craton.spec.wrap_lon)",
                v=v,
            )
        return v

    @field_validator("depth_km")
    @classmethod
    def _depth(cls, v: float) -> float:
        _finite(v, "depth_km")
        if not -MAX_ABOVE_ELLIPSOID_KM <= v <= EARTH_RADIUS_KM:
            raise _fail("depth_range", "depth_km must be in [-10, 6371], got {v}", v=v)
        return v

    @field_validator("time")
    @classmethod
    def _time(cls, v: float) -> float:
        _finite(v, "time")
        if abs(v) > EARTH_AGE_S:
            raise _fail("time_range", "time is more than 4.6 billion years from 1970: {v}", v=v)
        return v

    @field_validator("value", "uncertainty")
    @classmethod
    def _all_finite(cls, v: list[float], info: ValidationInfo) -> list[float]:
        for i, x in enumerate(v):
            _finite(x, f"{info.field_name}[{i}]")
        return v

    @field_validator("qualifiers")
    @classmethod
    def _qualifier_text(cls, v: dict[str, str]) -> dict[str, str]:
        for key, text in v.items():
            if not text or len(text) > 256:
                raise _fail("qualifier_value", "qualifier {k!r} must be 1-256 characters", k=key)
        return v

    # ---------------------------------------------------------- the whole token

    @model_validator(mode="after")
    def _consistent(self, info: ValidationInfo) -> Token:
        registry: Registry = (info.context or {}).get("registry", REGISTRY)
        inst = registry.get(self.instrument)
        if inst is None:
            raise _fail(
                "unknown_instrument", "instrument {n!r} is not registered", n=self.instrument
            )
        if self.kind not in inst.kinds:
            raise _fail(
                "kind_mismatch",
                "{n} produces {kinds} tokens, not {k!r}",
                n=inst.name,
                kinds=list(inst.kinds),
                k=self.kind,
            )
        n = len(inst.channels)
        if len(self.value) != n:
            raise _fail(
                "value_length",
                "{name} has {n} channel(s), value has {got}",
                name=inst.name,
                n=n,
                got=len(self.value),
            )
        if len(self.uncertainty) != n:
            raise _fail(
                "uncertainty_length",
                "uncertainty needs {n} entr(ies), one per channel, got {got}",
                n=n,
                got=len(self.uncertainty),
            )
        for ch, x, s in zip(inst.channels, self.value, self.uncertainty, strict=True):
            if not ch.minimum <= x <= ch.maximum:
                raise _fail(
                    "value_range",
                    "{c} = {x} {u} is outside [{lo}, {hi}]",
                    c=ch.name,
                    x=x,
                    u=ch.unit,
                    lo=ch.minimum,
                    hi=ch.maximum,
                )
            if s <= 0:
                raise _fail(
                    "uncertainty_nonpositive",
                    "uncertainty of {c} must be > 0; nothing is measured perfectly",
                    c=ch.name,
                )
        if inst.constraint == "simplex" and abs(sum(self.value) - 1) > SIMPLEX_TOLERANCE:
            raise _fail("simplex", "{n} probabilities must sum to 1", n=inst.name)
        for key, text in self.qualifiers.items():
            q = inst.qualifier(key)
            if q is None:
                raise _fail("unknown_qualifier", "{n} has no qualifier {k!r}", n=inst.name, k=key)
            why = q.problem(text)
            if why:
                raise _fail("qualifier_value", "qualifier {k}: {why}", k=key, why=why)
        for q in inst.qualifiers:
            if q.required and q.name not in self.qualifiers:
                raise _fail(
                    "missing_qualifier", "{n} tokens need qualifier {k!r}", n=inst.name, k=q.name
                )
        if self.time_iso is not None:
            try:
                t = to_seconds(self.time_iso)
            except ValueError as exc:
                raise _fail("time_format", "time_iso: {e}", e=str(exc)) from None
            if abs(t - self.time) > TIME_ISO_TOLERANCE_S:
                raise _fail(
                    "time_mismatch", "time_iso is {d:.3f} s away from time", d=t - self.time
                )
        return self

    # ---------------------------------------------------------- helpers

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready dict, optional fields left out when empty."""
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        """One exact text per measurement: sorted keys, no spaces.

        ``time_iso`` is left out: it only repeats ``time`` for humans, and the same measurement
        must have the same identity whether or not someone added the readable copy.
        """
        body = self.to_dict()
        body.pop("time_iso", None)
        return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def token_hash(self) -> str:
        """``sha256:`` of the canonical JSON: the token's identity for dedup and leak checks."""
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def wrap_lon(lon: float) -> float:
    """Any longitude into [-180, 180). Providers love 0-360 and even -347."""
    wrapped = (lon + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0 else wrapped


# ---------------------------------------------------------------- validation with codes


class Problem(BaseModel):
    """One reason a token is invalid."""

    model_config = ConfigDict(frozen=True)

    code: str
    field: str
    message: str


class TokenError(ValueError):
    """A token failed validation. ``problems`` says why, each with a stable code."""

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        super().__init__("; ".join(f"[{p.code}] {p.field}: {p.message}" for p in problems))

    @property
    def codes(self) -> set[str]:
        return {p.code for p in self.problems}


_BUILTIN_CODES = {
    "missing": "missing_field",
    "extra_forbidden": "unknown_field",
    "literal_error": "bad_choice",
    "too_short": "empty",
    "too_long": "too_long",
    "string_too_short": "empty",
    "string_too_long": "too_long",
}


def problems_from(error: ValidationError) -> list[Problem]:
    """Pydantic's errors as Craton problems with stable codes."""
    out = []
    for e in error.errors():
        kind = str(e["type"])
        code = _BUILTIN_CODES.get(kind)
        if code is None:
            code = "wrong_type" if kind.endswith("_type") or kind.endswith("_parsing") else kind
        where = ".".join(str(p) for p in e["loc"]) or "token"
        out.append(Problem(code=code, field=where, message=e["msg"]))
    return out


def validate_token(data: dict[str, Any] | str | bytes, registry: Registry | None = None) -> Token:
    """Build a token from a dict or a JSON string, or raise :class:`TokenError`."""
    ctx = {"registry": registry or REGISTRY}
    try:
        if isinstance(data, str | bytes):
            return Token.model_validate_json(data, context=ctx)
        return Token.model_validate(data, context=ctx)
    except ValidationError as exc:
        raise TokenError(problems_from(exc)) from None
