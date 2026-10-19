# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [0.3.0] — 2026-10-20 · P2 First data

### Added
- **Adapters** (`craton.adapters`): `fetch` → `tokens` → `card`, registered by name, run in dependency
  order. Three built-ins:
  - `ingv_catalog`: every earthquake INGV located in a region since 1985, through the FDSN event service.
  - `ngl_gnss`: daily positions of every Nevada Geodetic Laboratory station in a region (IGS20).
  - `iers_tides`: the solid-Earth tide at every catalogue earthquake, plus an hourly series per region.
- **The body tide in NumPy** (`craton.geo`): a port of the IERS 2010 routines with an analytic Sun and Moon.
  Matches the IERS test cases to 1e-12 m, PySolid to 0.4 mm and the JPL DE440 ephemeris to 0.71 mm.
  Runs on Windows; no new dependency.
- **The raw store** (`craton.store.raw`): downloads kept exactly as served, named by sha256, with an
  append-only manifest per source. Every token's `content_hash` points at its raw file.
- **Regions** (`configs/regions/`) and **snapshots**: a build is named by the hash of its content, so the
  same downloads always give the same snapshot id.
- `craton data list | fetch | build | cards`.
- **Data health** on the scoreboard (`results/data_health.json`): tokens, rejects by reason, the catalogue's
  magnitude of completeness per year, GNSS gaps, tide peaks. `craton check` keeps the two in sync.
- **Data cards** in `docs/data/`, generated from the adapters and drift-checked.
- The coverage figure (`scripts/figures/coverage.py`), `lessons/P2.md`, an adapter guide in
  `CONTRIBUTING.md`, decisions D-021 … D-028.

### Fixed
- The GNSS example token said CC-BY-4.0; the Nevada Geodetic Laboratory states no licence. It now says
  `LicenseRef-NGL-as-is` (D-024).
- `.gitignore` ignored every folder named `data/`, including `docs/data/`; it now ignores only the root one.

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
