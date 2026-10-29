# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [Unreleased]

### Added
- **The project page** (`site/`, `scripts/site.py`, `.github/workflows/pages.yml`): every score drawn
  against ETAS on GitHub Pages, rebuilt from `results/` on each push to main.
- `docs/assets/scorecard-*.png`, `scripts/figures/scorecard.py`: Craton against ETAS on every
  forecasting test, the climb by phase, and the fourteen CB-22 sequences.
- **How the Field works, in five clips** (`site/anim/`, `docs/assets/field-forecast.gif`): squares,
  memory, forecast, training and the world, drawn from real data and the trained checkpoints.
- **The whole Earth (P8d, D-086).** Regions are YAML files with their own sources and completeness
  (D-088): California, Japan, New Zealand, Turkey, Chile, Alaska and EarthquakeNPP's ComCat_25 join
  central Italy. Adapters `fdsn_catalog` (any FDSN event service: ComCat, GeoNet, EMSC),
  `jma_hypocenter` (JMA's yearly files, JST to UTC) and `earthquakenpp` (their CSVs, pinned commit).
- **Token Spec 1.1.0** (D-087): magnitude types `M`, `Mj`, `Mh`, `mb_Lg`, `MLv`. Tokens are written
  under the oldest spec that accepts them, so existing data keep their bytes.
- **The world Field** (`craton/field/world.py`, `configs/field/world.yaml`, D-089, D-090): one Field
  trained on central Italy, California, Japan and New Zealand up to 2007, in tiles of CB-02's size,
  every region in its own magnitudes.
- **CB-22, world sequences** (`craton/bench/world.py`, D-091): fourteen held-out sequences, six in
  regions trained on and eight in regions never seen, against ETAS fitted on each box's own past.
- **CB-29, EarthquakeNPP** (`craton/bench/npp.py`, D-092): their data, splits, ETAS and metric;
  their ETAS reproduced to 2·10⁻⁴ nats per quake; Craton as their ETAS times a daily factor per cell,
  fine-tuned from the world Field (`craton/field/npp.py`).
- `scripts/ablations/world_on_italy.py`, `docs/assets/world-*.png`, `scripts/figures/world.py`,
  `lessons/P8.md` (P8d).
- **CB-19, the 30-day forecast** (D-083): CB-02's sequence, split and score, one forecast per 30
  days, sixteen periods. `ForecastSpec.period_days`; ETAS over any span with the cascade stretched
  to it (CPU and GPU); the Field trained on 30-day answers with purged training starts (D-084);
  `long_periods` says which forecasters can answer it.
- **The GPS ablation at 30 days** (`scripts/ablations/gps_thirty_days.py`, D-085): +0.13 bits per
  quake on every seed, kept for P8f.
- `docs/assets/thirty-days-*.png`, `scripts/figures/thirty_days.py`, `lessons/P8.md` (P8b).
- **CB-09, the two-planet test** (`craton/bench/planets.py`, D-081): CB-02's calendar and scoring
  on three frozen ETAS planets and three physics planets. The number is bits per M3+ quake on the
  physics planet; the ETAS planet's bits, the truth's bits and their split are extras.
- **The physics planet** (`craton/sim/rate_state.py`, D-079): Dieterich (1994) rate-and-state
  seismicity on ~1,500 patches along 36 Apennine-shaped faults, Coulomb stress transfer with
  shadows, hidden fluid sources, an exact integrator checked against the closed form.
- **The ETAS planet** `etas_italy` (D-080): central Italy as the `etas` floor fits it.
- **The sim-to-real gap** (D-082): five plain catalogue statistics against the real Apennines,
  stored with each planet in the frozen record.
- `PlanetContext`: any forecaster runs on a planet through the calls it makes on a region.
- `docs/assets/planets-*.png`, `scripts/figures/planets.py`, `lessons/P8.md` (P8a).

### Changed
- The roadmap no longer says GPS strain is known to beat ETAS at 30 days: the paper it cited can't
  be found (D-083).
- **P8 is built in six parts** (D-078); the version stays 0.8.x until the last one.
- **The roadmap grows to 17 phases** (D-077): P8 Other planets, P9 The hidden state, P10 Listening,
  P11 The storm detector, then the ghost (P12), the translator (P13), the microscope and the law
  (P14), long horizons with the predictability horizon (P15) and scale (P16). Each phase now carries
  the numbers it is expected to reach.
- **Ten planned tasks, CB-19 to CB-28**, on the scoreboard; CB-09 becomes the two-planet test, and
  CB-10, 12, 13, 14, 16 and 18 move to their new phases.

## [0.8.0] — 2026-10-22 · P7 The 8 locks: knowing when it doesn't know

### Added
- **Five trust tasks** (`craton/bench/locks.py`, D-065): CB-05 Calibration (the calibration error as
  a factor, D-066), CB-06 Unseen region (Calabria + Sicily, D-068), CB-07 Fake-precursor attack
  (D-069), CB-08 Physics sanity (D-070), CB-11 the Predictability meter (D-071). CB-05, 07, 08 and 11
  read the forecasts of CB-02 and CB-03, so every forecaster and floor is scored on them.
- **The trust layer** (`craton/trust`): forecasts as distributions (`predictive`: ensemble members,
  negative binomial dispersion), calibration, the meter, physics checks, fake-precursor attacks, the
  input guard (D-072), the out-of-distribution detector.
- **`craton referee`** (D-073): a periodicity claim against declustering, fake clocks or shifted
  tides, time-of-day shuffles and a locked held-out prediction; `craton referee` with no arguments
  replays the 2026-10-21 tidal claim (rejected) and two fake planets (`results/referee/replay.json`).
- **Locked forecasts** (D-074): `craton live run|bundle|page|verify`, `forecasts/`, the nightly
  `.github/workflows/live.yml`; the frozen model ships as a release asset. `craton check` refuses any
  edit to a locked forecast or grade.
- **The simulator registry** (`craton/sim`, D-075) with the ETAS planet; Omori bursts and periodic
  thinning for the referee's replays.
- Region `southern_italy` (INGV, 61,086 quakes, snapshot `6dd8efd2fbc23ce6`), read only by CB-06.
- Floors `no_detector` and `fingerprint_knn` (CB-06).
- `docs/assets/meter-*.png`, `scripts/figures/meter.py`.
- `craton check` verifies every locked forecast's seal and refuses edits to locked files and grades.

### Changed
- The Field averages **six** members, P6's four plus two new ones: an ensemble grows and is never
  retrained (D-076). It forecasts no magnitude above 7.3 and reads the catalogue through the guard
  (D-067). Day totals carry a negative binomial size k = κ·mean^γ fitted on validation days (D-066).
- CB-02 and CB-03 compute each forecast once and share it with the trust tasks.

## [0.7.0] — 2026-10-22 · P6 Beat ETAS: the Field

### Added
- **The Field** (`craton/field`): CB-02 and CB-03 are forecast by the 1,050 forecast cells as
  tokens, each with an Omori memory (fading counters from 1 hour to 12 years, every quake down to
  M1.5), attention between cells, and ETAS as the starting line: a bounded correction, at most 4x
  either way, zero before training (D-061). Four Fields averaged per forecast (D-062). Its own
  config (`configs/field/mini.yaml`) and checkpoints, so it never retrains the trunk (D-063).
- **The bit split**: CB-02 and CB-03 extras split the score per quake into how many, where, how
  big and a shuffle term, plus the score without the M5+ days (D-058).
- **The building**: `craton bench ceiling` → `results/ceiling.json`, oracle floors above ETAS,
  drawn on the scoreboard and in `docs/assets/building-*.png` (D-059).
- **Floors** (D-060): `interpolation` (CB-01's GNSS half: 9.65 bits, ahead of Craton's 9.75),
  `etas_aniso` (stretched aftershock zones: no gain), `etas_incomplete` (blind time + productivity:
  CB-02 −7.059, CB-03 −9.295).
- ETAS daily rates on the GPU for the Field's training days (`craton/field/base.py`).

### Changed
- `craton-mini` forecasts through the Field; `field_config=""` keeps the trunk's own next-day head,
  which the ablation runner still uses.
- Task phases on the scoreboard follow the roadmap: P6 is now Beat ETAS, the locks move to P7 and
  every later phase moves down one.

## [0.6.0] — 2026-10-21 · P5 Fusion: quakes + GPS + the Moon

### Added
- **Fusion**: encoders for `gnss_daily` and `solid_earth_tide` beside the quake encoder, a learned
  modality vector each, and `model.inputs` in the config to choose what a model reads (D-048). The
  trunk takes an optional padding mask.
- **The context builder** (`craton.train.context`): one documented policy fills a window with tides,
  the nearest GNSS stations' days and quakes by recency, keeping the past year's big ones (D-050);
  a next-day window reads the day ahead's tide.
- **GNSS as distance from a line**: each station's long-term line fitted on the training split, days
  centred on their window, a mixture-of-logistics head scored on CB-01's 1 mm bins; unseen stations
  borrow their line from their neighbours (D-049).
- **Cross-modal masking** in both directions and whole-station masking (D-051), off by config for
  the general models after measurement (D-056).
- **CB-17 In-context adaptation** (D-052): a region the model has never seen, its first N days in
  context. New split `cb17_fit`; floors `climatology` and `context_kde`.
- **CB-04 for Craton**: the model rebuilds held-out GNSS stations.
- **Ablation runner**: `craton bench ablate` → `results/ablation/`, `docs/ablation.md`, the
  ablation figure.
- `craton train cb17_fit`; config fields `model.inputs`, `context.*`, `train.windows`,
  `train.split_windows`.

### Changed
- A frozen benchmark record grows by new splits as well as new tasks (D-052).
- Training batches are cut in a worker process, pinned and copied without blocking; fused AdamW
  (D-055).
- The next-day rate: the trunk may move it at most 2x from the measured-rate path (D-057).
- Model cards say what a checkpoint reads and are re-rendered from their metadata.

## [0.5.0] — 2026-10-20 · P4 Craton-mini v0

### Added
- **Craton-mini v0** (`craton.model`): an event encoder (registered by instrument), 4D rotary positional
  encoding over time, latitude, longitude and depth (sinusoidal fallback), a pre-norm transformer trunk on
  PyTorch's fused attention, and two probabilistic heads: binned place-and-size for a hidden quake, and a
  next-day Poisson rate × cell × magnitude forecast. Hidden coordinates are zeroed, never rotated.
- **Training** (`craton.train`): masked windows and next-day questions cut from a split, an fp16 loop with
  gradient scaling, warmup + cosine, validation blocks and early stopping; peak VRAM and tokens/s measured.
- **Checkpoints with model cards**, named by split, config hash, seed and training version;
  `docs/models/craton-mini.md` collects the cards behind the scoreboard.
- `craton train SPLIT [--seeds …]` and the bench method `craton-mini` (CB-01, CB-02, CB-03, CB-15).
- **CB-15 Efficiency**: model FLOPs utilisation of the pretraining run on the RTX 2060, and a hard 6 GiB
  VRAM gate.
- Config fields: `model.positional`, `train.max_steps`, `warmup_steps`, `eval_every`, `patience`,
  `mask_share`, `objectives`.
- The first-model figure, `lessons/P4.md`, decisions D-039 … D-047.

### Changed
- `craton bench prepare --freeze` on an existing record adds new tasks only, and only if every frozen
  task rebuilds identically (D-044). CB-15 was added this way.

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
