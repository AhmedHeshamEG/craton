# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [0.2.0] — 2026-10-20 · P1 The universal token

### Added
- **Token Spec 1.0.0** (`craton.spec`): one shape for every measurement, `kind · instrument · lat, lon,
  depth · time · value · uncertainty · provenance · resolution`, plus qualifiers and location uncertainty.
- The instrument registry with 11 instruments: `seismic_catalog`, `macroseismic_report`, `gnss_daily`,
  `solid_earth_tide`, `rock_porosity`, `rock_permeability`, `grain_size`, `lithology`, `rock_temperature`,
  `friction_coefficient`, `gutenberg_richter_b`.
- A strict validator with a stable error code for every failure; time-order and duplicate checks for files.
- One clock (`craton.spec.timeaxis`): seconds since 1970 UTC, Julian dates converted, deep time included.
- Token files (`craton.store`): Parquet by instrument and year, named by their own hash, byte-reproducible.
- `craton spec`, `craton spec example NAME`, `craton spec validate FILE`, `craton spec docs`.
- `craton check` now enforces the add-only spec against `craton/spec/frozen/1.0.0.json` and keeps
  `docs/token-spec.md` generated.
- `SPEC_POLICY.md`, `docs/token-spec.md`, the token figure, `lessons/P1.md`.

## [0.1.0] — 2026-10-19 · P0 Foundation

### Added
- `craton doctor`: GPU, VRAM, compute capability, fp16 / bf16 support, the precision every config will
  actually run with on this machine; `--smoke` trains a tiny network in fp16 to prove it.
- `craton config <size>`: a model size with its parameter count (the 12·d² rule).
- `craton check`: lint, format, types, tests, scoreboard integrity, append-only records, ratchet.
- `craton ratchet [--accept]` and `craton scoreboard`.
- Configs: `mini` (active), `regional` and `planet` (planned).
- The CratonBench catalogue: 18 planned tasks, CB-01 … CB-18.
- README figures from real USGS data, `lessons/P0.md`, CI on Linux and Windows.
