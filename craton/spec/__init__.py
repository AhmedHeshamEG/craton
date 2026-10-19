"""Token Spec: one format for every measurement on Earth.

>>> from craton.spec import validate_token, REGISTRY
>>> sorted(REGISTRY)[0].name
'friction_coefficient'

See ``SPEC_POLICY.md`` for what may change between versions (only additions).
"""

from craton.spec.instruments import REGISTRY
from craton.spec.registry import (
    KINDS,
    UNITS,
    Channel,
    Instrument,
    Kind,
    Qualifier,
    Registry,
    frozen_problems,
)
from craton.spec.sequence import sequence_problems
from craton.spec.timeaxis import julian_to_gregorian, to_iso, to_seconds, year_of
from craton.spec.token import (
    SPEC_VERSION,
    LocationUncertainty,
    Problem,
    Provenance,
    Resolution,
    Token,
    TokenError,
    validate_token,
    wrap_lon,
)

__all__ = [
    "KINDS",
    "REGISTRY",
    "SPEC_VERSION",
    "UNITS",
    "Channel",
    "Instrument",
    "Kind",
    "LocationUncertainty",
    "Problem",
    "Provenance",
    "Qualifier",
    "Registry",
    "Resolution",
    "Token",
    "TokenError",
    "frozen_problems",
    "julian_to_gregorian",
    "sequence_problems",
    "to_iso",
    "to_seconds",
    "validate_token",
    "wrap_lon",
    "year_of",
]
