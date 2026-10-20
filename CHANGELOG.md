# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [0.4.0] — 2026-10-20 · P3 CratonBench v1

### Added
- **CratonBench v1** (`craton.bench`): four frozen tests, each fingerprinted by its definition, metric and
  test files. CB-01 geo-perplexity (bits per masked token), CB-02 the 2016-17 Apennines and CB-03 2009
  (daily CSEP-style forecasts, log-likelihood per M3+ quake), CB-04 GPS gap-fill (RMSE in mm).
- **Frozen splits**: eras A (2016-08-24 → 2018) and B (2009) and 29 GNSS stations held out forever; three
  training splits cut to hash-named files. The frozen record is `craton/bench/frozen/central_italy.json`.
- **Leak checks** (`craton bench leaks`, and in `craton check`): every training shard read back; no
  held-out era, station or benchmark answer may appear.
- **Baselines**: climatology, Poisson, space-time ETAS (own implementation, exact likelihood, tested on
  synthetic catalogues) and inverse-distance weighting + trend.
- **CSEP tests** in NumPy (joint Poisson log-likelihood, N, S and M tests), matched against pyCSEP 0.8.0
  (`scripts/reference/csep_reference.py`).
- `craton bench list | prepare | leaks | run | cards`; the benchmark card `docs/bench.md`, generated.
- **The floor** on the scoreboard: baselines have their own ratchet lane, shown next to Craton's.
- `register_task` and `register_method`; a guide to adding a method in `CONTRIBUTING.md`.
- The Apennines forecast figure, `lessons/P3.md`, decisions D-029 … D-038. SciPy is a dependency.

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
