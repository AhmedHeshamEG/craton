# Logbook

Where the project stands, one entry per working session. **Append-only**: `craton check` fails if an old line
changes. Corrections go in a new entry.

---

## 2026-10-19 — P0 — Foundation

TLDR: the empty but bulletproof skeleton. One command (`craton check`) runs lint, format, types, tests,
scoreboard integrity, the append-only guard and the ratchet, the same on the laptop and on CI. The ratchet
passes on an empty scoreboard, as it should: there's nothing to protect yet.

Built:
- `craton` CLI: `doctor` (hardware probe, with `--smoke` for a real fp16 training step), `config`, `check`,
  `ratchet`, `scoreboard`
- configs for mini (4.74M trunk), regional (85.1M) and planet (1.61B), each validated against its claimed size
- seeds and deterministic mode in one call; stable derived seeds for ensembles and workers
- results format, the ratchet (D-007), the generated scoreboard (D-008), the append-only records guard
- 54 tests; CI on Linux (CPU) and Windows (CPU)
- README hero and diagram, drawn by `scripts/figures/` from 85,502 real USGS events
- `lessons/P0.md`

Scores: none yet. CratonBench tasks arrive in P3, baselines before any model.

Surprises / problems:
- The original name collided three ways in this field; renamed to Craton (D-001).
- `craton` on PyPI is an empty placeholder; the distribution is `craton-earth` (D-002).
- The first version of the append-only guard compared raw text, so editing `D-001` into `D-001 (edited)`
  passed as an "append". A test caught it; it now compares whole lines.
- On the RTX 2060: PyTorch 2.14.1+cu126 sees the GPU (sm_75 is in its arch list) and a 60-step fp16 run with
  gradient scaling drops the loss 0.522 → 0.001, peak 17 MiB.

Decisions: D-001 … D-010.

Next: P1, the universal token. Write Token Spec v1.0.0 and its validator, with an example token for every
`kind`, including the monk's diary from 1349.

## 2026-10-20 — P1 — The universal token

TLDR: Token Spec 1.0.0 shipped and frozen. Ten slots fit an earthquake, a GNSS day, a tidal forcing, six rock
properties, a derived b-value and a monk's account from 1349. The validator refuses 72 kinds of malformed
token, each with its own error code, and `craton check` now fails on any change that removes or redefines
part of the spec.

Built:
- `craton/spec/`: the token model, the instrument registry (11 instruments), the clock, sequence checks, the
  frozen snapshot `frozen/1.0.0.json` and the add-only check
- `craton/store/`: Parquet token files by instrument and year, named by their own hash
- `craton spec` (show, example, validate, docs); `docs/token-spec.md` generated from the code
- 11 example tokens; three from real records (INGV event 8863681, NGL station ARQT, ASMI Montecassino)
- `SPEC_POLICY.md`, the token figure (`scripts/figures/token_slots.py`), `lessons/P1.md`
- tests: 54 → 200

Scores: no CratonBench task moves in P1. Validator suite: 72 malformed cases refused with the expected code,
11 / 11 examples valid; two writes of the same tokens give byte-identical files.

Surprises / problems:
- The plan put depth relative to sea level; the spec uses the WGS84 ellipsoid instead (D-012).
- The Montecassino date is Julian. 9 September 1349 is 17 September on a Gregorian axis; the spec converts
  once, in one tested function.
- The Nevada Geodetic Laboratory writes ARQT's longitude as −346.80. The validator refuses it until it's
  wrapped to 13.20 (D-013).
- ARQT, 10 km from the epicentre, dropped 0.447 m between its 29 and 30 October 2016 daily solutions. The
  Norcia station NRCI has no daily solutions from mid-October 2016, so the example uses ARQT.
- INGV gives no uncertainty for the Norcia Mw 6.5 magnitude; the token says `assumed` (D-015).
- PySolid, the planned solid-Earth tide library, has no Windows wheels (Linux and macOS only). P2 needs
  another route to tides on the laptop.
- A figure script called `token.py` shadowed Python's own `token` module and broke matplotlib's import.
  Renamed to `token_slots.py`.

Decisions: D-011 … D-020.

Next: P2, the first adapters. Start with the INGV catalogue through FDSN, then choose how to compute
solid-Earth tides on Windows.
