"""One example token for every kind, and the monk.

Three are built from real records, quoted byte for byte below so their ``content_hash`` can be
checked by anyone:

* ``norcia_2016``: the Mw 6.5 Norcia earthquake, 30 October 2016, the largest in Italy since
  1980 (INGV FDSN event service, event 8863681).
* ``arquata_gnss``: the GNSS station ARQT in Arquata del Tronto, the day after. In one day the
  ground under it dropped about 45 cm (Nevada Geodetic Laboratory, IGS20 daily solutions).
* ``montecassino_1349``: the monk's diary. The earthquake of 9 September 1349 (Julian calendar)
  devastated Montecassino Abbey; Galli & Naso (2009) assess intensity 9 MCS there (INGV ASMI).

The rest show what forcing, property and derived tokens look like. Their numbers are
**illustrative, not measurements**, and their provenance says so.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from craton.spec.timeaxis import julian_to_gregorian, to_iso, to_seconds
from craton.spec.token import Token, validate_token, wrap_lon

RAW_NORCIA = (
    "8863681|2016-10-30T06:40:17.320000|42.8303|13.1092|10.0|BULLETIN-INGV||||Mw|6.5|--|"
    "4 km NE Norcia (PG)|earthquake"
)
RAW_ARQT = (
    "ARQT 16OCT31 2016.8323 57692 1921 1   13.2   -104 -0.417242   4735498  0.950565  1659  "
    "0.446011  0.0083 0.000653 0.000840 0.002586 -0.014543  0.081816 -0.085904  42.7549658145 "
    "-346.8012754927  1659.44601"
)
RAW_MONTECASSINO = (
    "quakeml:it.ingv.asmi/event/13490909_0815_001|quakeml:it.ingv.asmi/mdpset/GALNA009/11278|"
    "1349-09-09|Molise|24|10-11|MCS|325726|IT_55053|Abbazia di Montecassino|41.491|13.814|9||"
)

ILLUSTRATIVE = "Craton spec examples (illustrative values, not measurements)"


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _illustrative(name: str) -> dict[str, Any]:
    return {
        "source": ILLUSTRATIVE,
        "dataset_version": "1.0.0",
        "license": "CC0-1.0",
        "adapter_version": "1.0.0",
        "content_hash": sha256_text(f"craton illustrative example: {name}"),
        "record": name,
        "synthetic": False,
    }


def _at(iso: str) -> dict[str, Any]:
    return {"time": to_seconds(iso), "time_iso": iso}


# Centre of the Amatrice-Norcia sequence; the illustrative tokens sit here.
_NORCIA_LAT, _NORCIA_LON = 42.79, 13.15


def norcia_2016() -> dict[str, Any]:
    f = RAW_NORCIA.split("|")
    return {
        "kind": "event",
        "instrument": "seismic_catalog",
        "lat": float(f[2]),
        "lon": float(f[3]),
        "depth_km": float(f[4]),
        **_at(f[1] + "Z"),
        "value": [float(f[10])],
        # INGV publishes no magnitude uncertainty for this event; 0.1 is a common working value.
        "uncertainty": [0.1],
        "uncertainty_basis": "assumed",
        # From the same event's QuakeML: horizontal 190 m, depth 200 m, origin time 0.02 s.
        "location_uncertainty": {"horizontal_km": 0.19, "vertical_km": 0.2, "time_s": 0.02},
        "qualifiers": {"magnitude_type": f[9], "event_id": f[0], "event_type": f[13]},
        "provenance": {
            "source": "INGV FDSN event service (webservices.ingv.it)",
            "dataset_version": "ISIDe, retrieved 2026-10-20",
            "license": "CC-BY-4.0",
            "adapter_version": "1.0.0",
            "content_hash": sha256_text(RAW_NORCIA),
            "record": f"eventid={f[0]}",
            "synthetic": False,
        },
        # A hypocentre is where the rupture started: a point known to about 200 m. The rupture
        # itself was ~30 km long; that would be a different instrument.
        "resolution": {"spatial_km": 0.2, "temporal_s": 0.02},
    }


def arquata_gnss() -> dict[str, Any]:
    f = RAW_ARQT.split()
    day = datetime.strptime(f[1], "%y%b%d").replace(hour=12, tzinfo=UTC)
    iso = day.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "kind": "observation",
        "instrument": "gnss_daily",
        "lat": float(f[20]),
        "lon": wrap_lon(float(f[21])),  # NGL writes -346.80; on the axis it is 13.20
        "depth_km": -float(f[22]) / 1000,  # ellipsoidal height, so depth is negative
        **_at(iso),
        "value": [
            float(f[7]) + float(f[8]),
            float(f[9]) + float(f[10]),
            float(f[11]) + float(f[12]),
        ],
        "uncertainty": [float(f[14]), float(f[15]), float(f[16])],
        "uncertainty_basis": "reported",
        "qualifiers": {"station": f[0], "reference_frame": "IGS20"},
        "provenance": {
            "source": "Nevada Geodetic Laboratory (geodesy.unr.edu)",
            "dataset_version": "IGS20 tenv3, retrieved 2026-10-20",
            "license": "CC-BY-4.0",
            "adapter_version": "1.0.0",
            "content_hash": sha256_text(RAW_ARQT),
            "record": f"{f[0]}.tenv3 {f[1]}",
            "synthetic": False,
        },
        # A daily solution: one antenna, the whole UTC day, labelled at noon.
        "resolution": {"spatial_km": 0.001, "temporal_s": 86_400.0},
    }


def montecassino_1349() -> dict[str, Any]:
    f = RAW_MONTECASSINO.split("|")
    y, m, d = (int(x) for x in f[2].split("-"))
    g = julian_to_gregorian(y, m, d)  # 9 Sep 1349 (Julian) is 17 Sep on the axis
    noon = to_seconds(datetime(g.year, g.month, g.day, 12, tzinfo=UTC))
    return {
        "kind": "observation",
        "instrument": "macroseismic_report",
        "lat": float(f[10]),
        "lon": float(f[11]),
        # The abbey stands ~519 m above sea level, and sea level here is ~48 m above the
        # ellipsoid, so the token sits ~0.57 km above it.
        "depth_km": -0.57,
        "time": noon,
        "time_iso": to_iso(noon),
        "value": [float(f[12])],
        "uncertainty": [0.5],  # intensities come in half-degree steps
        "uncertainty_basis": "assumed",
        "location_uncertainty": {"horizontal_km": 0.5, "vertical_km": 0.1},
        "qualifiers": {
            "scale": f[6],
            "place": f[9],
            "event_id": f[0].rsplit("/", 1)[1],
            "study": "GALNA009",
        },
        "provenance": {
            "source": "INGV ASMI macroseismic data service (emidius.mi.ingv.it)",
            "dataset_version": "ASMI, retrieved 2026-10-20",
            "license": "CC-BY-4.0",
            "adapter_version": "1.0.0",
            "content_hash": sha256_text(RAW_MONTECASSINO),
            "record": f"mdp={f[7]}",
            "synthetic": False,
        },
        # The study dates it to the day, not the hour: the time slot is a whole day wide.
        "resolution": {"spatial_km": 0.5, "temporal_s": 86_400.0},
    }


def tide_arquata() -> dict[str, Any]:
    return {
        "kind": "forcing",
        "instrument": "solid_earth_tide",
        "lat": 42.75497,
        "lon": 13.19872,
        "depth_km": -1.659,
        **_at("2016-10-30T06:40:17Z"),
        "value": [0.012, -0.031, 0.142],
        "uncertainty": [0.001, 0.001, 0.002],
        "uncertainty_basis": "assumed",
        "qualifiers": {"model": "IERS2010"},
        "provenance": _illustrative("tide_arquata"),
        "resolution": {"spatial_km": 1.0, "temporal_s": 60.0},
    }


def _rock(name: str, instrument: str, value: list[float], sigma: list[float]) -> dict[str, Any]:
    return {
        "kind": "property",
        "instrument": instrument,
        "lat": _NORCIA_LAT,
        "lon": _NORCIA_LON,
        "depth_km": 3.0,
        **_at("2016-01-01T00:00:00Z"),
        "value": value,
        "uncertainty": sigma,
        "uncertainty_basis": "assumed",
        "provenance": _illustrative(name),
        # Rock properties change over geological time: the support is a block and an era.
        "resolution": {"spatial_km": 5.0, "temporal_s": 3.15576e13},  # one million years
    }


def limestone_porosity() -> dict[str, Any]:
    return _rock("limestone_porosity", "rock_porosity", [0.05], [0.02])


def limestone_permeability() -> dict[str, Any]:
    return _rock("limestone_permeability", "rock_permeability", [-17.0], [1.0])


def limestone_grain_size() -> dict[str, Any]:
    return _rock("limestone_grain_size", "grain_size", [-1.7], [0.3])


def limestone_lithology() -> dict[str, Any]:
    probs = {"carbonate_sedimentary": 0.8, "mixed_sedimentary": 0.15, "evaporites": 0.05}
    from craton.spec.instruments import GLIM_CLASSES

    value = [probs.get(c, 0.0) for c in GLIM_CLASSES]
    return _rock("limestone_lithology", "lithology", value, [0.05] * len(value))


def crust_temperature() -> dict[str, Any]:
    tok = _rock("crust_temperature", "rock_temperature", [250.0], [40.0])
    tok["depth_km"] = 10.0
    return tok


def fault_friction() -> dict[str, Any]:
    tok = _rock("fault_friction", "friction_coefficient", [0.6], [0.1])
    tok["qualifiers"] = {"friction_kind": "steady_state"}
    return tok


def b_value_sequence() -> dict[str, Any]:
    # Computed over the whole sequence: 24 Aug 2016 to 31 Dec 2017, labelled at its midpoint.
    start = to_seconds("2016-08-24T00:00:00Z")
    end = to_seconds("2018-01-01T00:00:00Z")
    mid = (start + end) / 2
    return {
        "kind": "derived",
        "instrument": "gutenberg_richter_b",
        "lat": _NORCIA_LAT,
        "lon": _NORCIA_LON,
        "depth_km": 8.0,
        "time": mid,
        "time_iso": to_iso(mid),
        "value": [1.0],
        "uncertainty": [0.05],
        "uncertainty_basis": "assumed",
        "qualifiers": {"method": "b_positive", "magnitude_type": "ML"},
        "provenance": _illustrative("b_value_sequence"),
        "resolution": {"spatial_km": 60.0, "temporal_s": end - start},
    }


EXAMPLES: dict[str, Callable[[], dict[str, Any]]] = {
    "norcia_2016": norcia_2016,
    "arquata_gnss": arquata_gnss,
    "montecassino_1349": montecassino_1349,
    "tide_arquata": tide_arquata,
    "limestone_porosity": limestone_porosity,
    "limestone_permeability": limestone_permeability,
    "limestone_grain_size": limestone_grain_size,
    "limestone_lithology": limestone_lithology,
    "crust_temperature": crust_temperature,
    "fault_friction": fault_friction,
    "b_value_sequence": b_value_sequence,
}


def example(name: str) -> Token:
    """A validated example token by name (see ``EXAMPLES``)."""
    return validate_token(EXAMPLES[name]())


def all_examples() -> dict[str, Token]:
    return {name: example(name) for name in EXAMPLES}
