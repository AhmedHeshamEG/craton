"""The instruments of Token Spec 1.0.0.

Every entry here is frozen with the release (``craton/spec/frozen/1.0.0.json``). New instruments
are welcome in any later minor version; these ones never change.
"""

from __future__ import annotations

from craton.spec.registry import Channel, Instrument, Qualifier, Registry

_ID = r"[A-Za-z0-9_.:/-]{1,128}"

MAGNITUDE_TYPES = ("ML", "Mw", "Md", "mb", "mB", "Ms", "Mwp", "MwL", "Mwpd", "Mc", "Mmacro")
"""Magnitude scales are not interchangeable: an ML 4.0 and an Mw 4.0 are different numbers.
Every magnitude carries its type, and nothing mixes them silently."""

SEISMIC_CATALOG = Instrument(
    name="seismic_catalog",
    description="An earthquake from a seismic catalogue: where and when it began, and its size.",
    kinds=("event",),
    channels=(
        Channel("magnitude", "1", "Magnitude on the scale named by magnitude_type.", -3, 10),
    ),
    qualifiers=(
        Qualifier("magnitude_type", "Which magnitude scale.", True, MAGNITUDE_TYPES),
        Qualifier("event_id", "The catalogue's own ID for the event.", pattern=_ID),
        Qualifier(
            "event_type",
            "QuakeML event type.",
            vocabulary=("earthquake", "induced or triggered event", "quarry blast", "explosion"),
        ),
    ),
    references=("QuakeML 1.2, https://quake.ethz.ch/quakeml",),
)

MACROSEISMIC_REPORT = Instrument(
    name="macroseismic_report",
    description=(
        "What an earthquake did at one place, read from written accounts: chronicles, letters, "
        "questionnaires, damage surveys. The monk's diary lives here."
    ),
    kinds=("observation",),
    channels=(Channel("intensity", "1", "Macroseismic intensity on the named scale.", 1, 12),),
    qualifiers=(
        Qualifier("scale", "Intensity scale.", True, ("MCS", "EMS-98", "MMI", "MSK-64")),
        Qualifier("place", "Locality name as the study gives it.", pattern=r".{1,128}"),
        Qualifier("event_id", "The archive's ID for the earthquake.", pattern=_ID),
        Qualifier("study", "The study that assessed the intensity.", pattern=_ID),
    ),
    references=("ASMI, https://emidius.mi.ingv.it/ASMI/", "Grünthal (1998), EMS-98"),
)

GNSS_DAILY = Instrument(
    name="gnss_daily",
    description=(
        "One day of a GNSS station's position: east, north and up in metres, from the "
        "provider's daily solution, relative to the series origin named in its provenance."
    ),
    kinds=("observation",),
    channels=(
        Channel("east", "m", "Eastward position.", -2.1e7, 2.1e7),
        Channel("north", "m", "Northward position.", -2.1e7, 2.1e7),
        Channel("up", "m", "Vertical position.", -1.2e4, 1.2e4),
    ),
    qualifiers=(
        Qualifier("station", "Four-character station code.", True, pattern=r"[A-Z0-9]{4}"),
        Qualifier(
            "reference_frame",
            "Frame the positions are expressed in.",
            True,
            ("IGS14", "IGS20", "ITRF2014", "ITRF2020", "ETRF2000", "ETRF2014", "ETRF2020"),
        ),
    ),
    references=("Blewitt, Hammond & Kreemer (2018), Eos 99, doi:10.1029/2018EO104623",),
)

SOLID_EARTH_TIDE = Instrument(
    name="solid_earth_tide",
    description=(
        "How far the Sun and the Moon stretch the solid Earth at a point: the body tide's "
        "displacement, computed from a tide model."
    ),
    kinds=("forcing",),
    channels=(
        Channel("east", "m", "Eastward tidal displacement.", -1, 1),
        Channel("north", "m", "Northward tidal displacement.", -1, 1),
        Channel("up", "m", "Vertical tidal displacement.", -1, 1),
    ),
    qualifiers=(Qualifier("model", "Tide model.", True, ("IERS2010",)),),
    references=("IERS Conventions (2010), Technical Note 36, chapter 7",),
)

ROCK_POROSITY = Instrument(
    name="rock_porosity",
    description="Fraction of a rock's volume that is empty space.",
    kinds=("property",),
    channels=(Channel("porosity", "1", "Pore volume / total volume.", 0, 1),),
    qualifiers=(Qualifier("method", "How it was measured or inferred.", pattern=r".{1,64}"),),
)

ROCK_PERMEABILITY = Instrument(
    name="rock_permeability",
    description=(
        "How easily fluid flows through a rock. It spans twenty orders of magnitude, so the "
        "channel is its base-10 logarithm."
    ),
    kinds=("property",),
    channels=(Channel("log10_permeability", "log10(m2)", "log10 of permeability in m².", -25, -5),),
    qualifiers=(Qualifier("method", "How it was measured or inferred.", pattern=r".{1,64}"),),
)

GRAIN_SIZE = Instrument(
    name="grain_size",
    description="Granularity: how coarse the grains of a rock or sediment are (median diameter).",
    kinds=("property",),
    channels=(
        Channel("log10_median_diameter", "log10(mm)", "log10 of median grain size in mm.", -5, 4),
    ),
    qualifiers=(Qualifier("method", "How it was measured or inferred.", pattern=r".{1,64}"),),
)

GLIM_CLASSES = (
    "unconsolidated_sediments",
    "siliciclastic_sedimentary",
    "mixed_sedimentary",
    "carbonate_sedimentary",
    "pyroclastics",
    "evaporites",
    "metamorphics",
    "acid_plutonic",
    "intermediate_plutonic",
    "basic_plutonic",
    "acid_volcanic",
    "intermediate_volcanic",
    "basic_volcanic",
)

LITHOLOGY = Instrument(
    name="lithology",
    description=(
        "What the rock is, as probabilities over the 13 rock classes of the Global Lithological "
        "Map. A certain answer is a one-hot vector; a geologist's 'probably limestone, maybe "
        "marl' is a spread."
    ),
    kinds=("property",),
    channels=tuple(
        Channel(c, "1", f"Probability the rock is {c.replace('_', ' ')}.", 0, 1)
        for c in GLIM_CLASSES
    ),
    constraint="simplex",
    references=("Hartmann & Moosdorf (2012), G-cubed 13, doi:10.1029/2012GC004370",),
)

ROCK_TEMPERATURE = Instrument(
    name="rock_temperature",
    description="Temperature of the rock at a point, from a borehole or a thermal model.",
    kinds=("property",),
    channels=(Channel("temperature", "Cel", "Temperature.", -90, 7000),),
    qualifiers=(Qualifier("method", "How it was measured or inferred.", pattern=r".{1,64}"),),
)

FRICTION = Instrument(
    name="friction_coefficient",
    description="How hard it is to slide one side of a fault over the other.",
    kinds=("property",),
    channels=(Channel("friction", "1", "Coefficient of friction.", 0, 2),),
    qualifiers=(
        Qualifier("friction_kind", "Which friction.", True, ("static", "steady_state", "dynamic")),
        Qualifier("method", "How it was measured or inferred.", pattern=r".{1,64}"),
    ),
)

GUTENBERG_RICHTER_B = Instrument(
    name="gutenberg_richter_b",
    description=(
        "The b-value of a set of earthquakes: how fast the number of quakes falls as the "
        "magnitude rises. Computed from other tokens, so it is derived."
    ),
    kinds=("derived",),
    channels=(Channel("b_value", "1", "Gutenberg-Richter b-value.", 0.1, 4),),
    qualifiers=(
        Qualifier("method", "Estimator.", True, ("aki_utsu_mle", "b_positive", "least_squares")),
        Qualifier("magnitude_type", "Magnitude scale of the input events.", True, MAGNITUDE_TYPES),
    ),
    references=("Aki (1965); Utsu (1966); van der Elst (2021), b-positive",),
)

V1_INSTRUMENTS = (
    SEISMIC_CATALOG,
    MACROSEISMIC_REPORT,
    GNSS_DAILY,
    SOLID_EARTH_TIDE,
    ROCK_POROSITY,
    ROCK_PERMEABILITY,
    GRAIN_SIZE,
    LITHOLOGY,
    ROCK_TEMPERATURE,
    FRICTION,
    GUTENBERG_RICHTER_B,
)

REGISTRY = Registry(V1_INSTRUMENTS)
"""The default registry. Adapters and plugins add to it with ``REGISTRY.register(...)``."""
