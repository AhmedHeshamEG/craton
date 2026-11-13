# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [Unreleased]

## [0.11.0] — 2026-11-12 · P10 Listening: the crust's hum as two instruments

### Added
- **Token Spec 1.4.0** (P10a, D-137): `seismic_noise_power`, daily median of hourly band powers in
  three bands at every broadband station; `craton/adapters/waveforms.py` (any FDSN data centre)
  and `craton/adapters/noise.py`; regions `laquila_noise`, `central_italy_noise`.
- **Token Spec 1.5.0** (P10b, D-140): `seismic_velocity_change`, dv/v per station pair and day
  (10- and 30-day stacks) from nine-component ambient-noise correlations, corrected for the hum's
  loudness, its spectral tilt and the season with terms fitted and chosen on the reference period
  only. `craton/adapters/interferometry.py` (GPU correlations, cached per month),
  `craton/adapters/velocity.py`, `craton/adapters/dvv.py`. 81,264 tokens in central Italy (41 pairs),
  10,146 at L'Aquila (9).
- **Listening in the Field** (P10c, D-143 to D-145): `craton/field/listen.py`, seven numbers per
  cell per midnight (loudness change in three bands against each station's own 60 days, two dv/v
  stacks, two coverage counts), appended to the Omori memory; `configs/field/listen.yaml`. Not
  adopted: CB-02 −0.047, CB-03 0.00, CB-19 +0.126 (count +0.17 on every seed, where inside noise).
- **A borrowed ear** (P10d, D-146, D-147): `listen.ear: seislm`, three principal components of
  frozen SeisLM-base embeddings of ten quiet minutes per station-day
  (`scripts/ablations/seislm_ear.py`). Not borrowed: CB-19 +0.013 against the band powers' +0.126.
- **The P10 figure** (`scripts/figures/velocity.py`, `docs/assets/velocity-*.png`): central Italy's
  network dv/v day by day under the 2016-17 sequence, the shocks marked. `lessons/P10.md` complete.
- **CB-28, velocity gap-fill** (P10b, D-141): hide a station, rebuild its pairs' dv/v. Floors
  `velocity_zero` 0.347% (best) and `velocity_neighbours` 0.378%.

### Fixed
- `scripts/site.py` and `scripts/figures/scorecard.py` import the task registry before
  `craton.bench.world` (an import cycle since P9g had stopped the project page from building).

### Changed
- **The plan for P11 to P16** (D-148 to D-151): the remaining phases are now Field 2, a transformer
  shaped for the Earth, judged on a world gate with a sealed vault. The ghost channel, the
  translator, the microscope and the million-year router move to Later; their planned tasks
  (CB-10, CB-12, CB-13, CB-14, CB-18) are listed as `later`. CB-35, CB-36 and CB-37 are reserved.
  No code or score changed.

## [0.10.0] — 2026-11-06 · P9 The hidden state: stress, pressure and the wiring

### Added
- **Token Spec 1.3.0** (D-132): the `moment_tensor` instrument (Mrr … Mtp and M0 in N·m, at the
  centroid) and the unit `N.m`. Add-only; every older token keeps its version and hash.
- **Global CMT adapter** (`gcmt`, `craton/adapters/gcmt.py`): one ndk release read into tokens,
  56,832 solutions 1976-2020; the whole Earth is one region file, `configs/regions/gcmt_world.yaml`.
- **CB-34, stress from mechanisms (P9i, D-132, D-133).** CB-23's regions and cells, with the
  mechanisms before the cut as input, graded only on World Stress Map records not read from quakes.
  craton-mini 31.3° ± 0.95 through the **tensor field** (`craton/field/tensors.py`,
  `configs/tensors/mini.yaml`: masked modelling of moment tensors, never a stress record); floors
  `stress_interpolation` 27.9°, `stress_regional` 29.8°, `mechanism_kostrov` (new) 33.4°,
  `quake_lineation` 47.9°.
- **The map referee (P9i, D-134)** (`craton/trust/maps.py`, `scripts/referee_maps.py`): coin,
  floor, 200 toroidally shifted copies, a locked half; verdicts in `results/referee/maps.json`.
- **Three checks on the Field (P9i, D-135)**: `scripts/ablations/cap_and_mainshock.py` (how often the
  ×4 bound binds; the scores without Norcia's and L'Aquila's days) and
  `scripts/ablations/linear_probe.py` (the hidden features against the planets' fault state).
- Figure `docs/assets/mechanisms-{dark,light}.png` (`scripts/figures/mechanisms.py`).
- Benchmark card renderer for CB-34; model cards for the tensor field's checkpoints.

### Changed
- `cells_of` (CB-23) takes the minimum records per cell as an argument (default unchanged).
- The README says where a public map passed the referee and where it didn't (CB-33: two planets of
  three).

### Added (P9a-P9h)
- **CB-23, which way the crust is squeezed (P9a, D-106 … D-108).** 983 half-degree cells in
  central Italy and CB-22's six regions, graded against the World Stress Map 2025: give the
  direction of largest horizontal stress from the quake catalogue alone. Floors
  `stress_interpolation` 16.6°, `stress_regional` 26.7°, `quake_lineation` 44.7° (a coin: 45°).
- **Truth sources** (`craton/bench/truth.py`): data a task grades against and no model reads,
  kept out of the token store, with data cards (`docs/data/world_stress_map.md`).
- Figure `docs/assets/squeeze-{dark,light}.png` (`scripts/figures/squeeze.py`); `lessons/P9.md`.
- **CB-24, where a sequence lights up next (P9b, D-109).** Sixteen sequences on 0.1° cells:
  before each cell's first quake at the scored size, rank the dark cells, seeing only the past
  and the cells lit so far. Floors `distance_only` 9.3% in the top 3, `quake_history` 1.4% (a
  coin: 0.3%).
- **The CRS floor (P9b, D-110).** Coulomb rate-and-state (Dieterich 1994) from the catalogue
  alone: rectangles from M5.4 through `cutde` (new dependency, MIT), 1/r³ sources below, ETAS's
  background, the state bank on the GPU (`craton/bench/crs.py`, `configs/floors/crs.yaml`).
  CB-02 −7.518 bits per quake (ETAS −7.118).
- Figure `docs/assets/migration-{dark,light}.png` (`scripts/figures/migration.py`).
- **The leak audit (P9e, D-119).** Canary quakes at and after the cut must leave every training
  array identical (`tests/test_audit.py`); Field configs take an `audit:` section (`cut`,
  `shuffle_answers`) for the two controls, run by `scripts/ablations/leak_audit.py`. The earlier
  cut keeps CB-02's gain over ETAS; answers dealt to the wrong days lose it.
- **CB-26, the distilled law (P9e, D-120, D-121).** Symbolic regression (PySR, new `distill`
  dependency group) on the Field's correction to ETAS; a formula of at most ten terms forecasts
  CB-02's and CB-03's test windows. craton-mini 0.836 ± 0.047 of the network's gain kept; the floor
  `law_constant` (one constant factor on ETAS, out of the same search) 0.615 ± 0.007. Laws in
  `results/laws/` (`scripts/laws.py`), figure `docs/assets/law-{dark,light}.png`.
- **CB-33, planted stress (P9f, D-122, D-123).** The physics planets record each patch's
  rate-and-state state (`StressRecord`); a forecast is graded on how it ranks the cells by it.
  craton-mini 0.448, floor `quake_density` 0.395. The stress meter (`craton/field/meter.py`) is
  built and off (D-124).
- **CB-31, Groningen pressure (P9g, D-125, D-128).** A quake-only forecast map against the
  reservoir's pressure drop over each gas year 2017-2023 (`craton/bench/pressure.py`). Graded
  methods answer through `BlindView`, which refuses the pressure tokens; craton-mini through its
  quake-only twin (`quake_only_forecaster`). craton-mini −0.42, floor `quake_density` −0.23; the
  real quakes, graded the same way, −0.21 (`scripts/ablations/pressure_reference.py`).
- **The local-alpha floor (P9g, D-126, D-127).** `etas_local_alpha`: ETAS whose K and α are refit
  every day on the last 30 days of quakes, after Lombardi & Marzocchi (2010). CB-02 +0.297 over
  ETAS (craton-mini +0.303), CB-03 +0.588 (+1.165). Ablation `scripts/ablations/local_alpha.py`.
- Figures `docs/assets/local-alpha-{dark,light}.png` and `docs/assets/pressure-{dark,light}.png`.
- **CB-32, regions it never read (P9h, D-129).** The world Field trained four times, each without
  one of central Italy, California, Japan and New Zealand, scored on that region's sequences
  against ETAS fitted there (`craton/bench/unseen.py`, `scripts/ablations/unseen_regions.py`).
  craton-mini +0.336 ± 0.041 bits per quake, best floor `etas_local_alpha` +0.065.
- **Dimensionless inputs (P9h, D-130, D-131).** `dimless:` in Field configs
  (`craton/field/dimless.py`): counters over ETAS's expectation, Omori-time lifetimes, a
  rupture-length spread (Wells & Coppersmith 1994) and, in v2, the box's rate over its long-run
  rate. `world_dimless` and `world_dimless2` are ablations, not adopted (CB-32 +0.304, CB-22
  +0.339 against +0.336 and +0.361; `scripts/ablations/dimless.py`).
- Figure `docs/assets/unseen-regions-{dark,light}.png` (`scripts/figures/unseen.py`).

### Changed (P9a-P9h)
- World Field configs can read another config's pool of days (`dimless.pool_of`).
- ETAS's fit is cached by its cut as well as its task and seed.
- Benchmark cards print list values in a task's frozen summary.

## [0.9.0] — 2026-10-30 · P8 Other planets: real, human-made and fake

### Fixed
- Anisotropic ETAS no longer fails on a box with no quake big enough to have a shape.
- Month windows before 1970 (Windows refuses `datetime.fromtimestamp` there).

### Added
- **The curriculum (P8f, D-102).** Field configs can start every member from an earlier stage's
  Field (`curriculum.start_from`, `field_stage`): `planets` (six physics planets, each with a
  blind twin that reads what a network blinded by big shocks would record) → `earth` → `curriculum`.
  CB-20 +0.09 over P6's Field on every seed; CB-02 −0.03, CB-03 −0.02. Measured, not adopted.
- **Pumped planets** (`craton/field/planets.py`, `configs/field/pumped.yaml`, D-103): physics
  planets whose fluid sources switch off, as forcing for the forced Field. CB-30 +0.695 ± 0.006,
  the pump still worth +0.00. Not adopted.
- Region `sumatra` (ComCat, M4.5; D-104) and the ablation `world_sumatra`: CB-22 unchanged.
- `scripts/ablations/gps_thirty_days.py --no-jumps` and any daily or 30-day task (D-105): GPS
  without coseismic offsets loses its consistency; GPS stays out of the Fields.
- **A second ETAS referee** (`scripts/reference/eth_etas_invert.py`,
  `scripts/ablations/eth_etas_referee.py`, D-101): ETH Zurich's ETAS fitted on CB-02's and CB-03's
  data scores −7.467 / −9.929, below Craton's ETAS (−7.118 / −9.396), which stays the floor.
- `scripts/ablations/curriculum.py`, `earth_on_world.py`; `forcing_blind.py --config`;
  `lessons/P8.md` (P8f and P8 in one page).
- **Where people pushed the crust (P8e, D-097 … D-100).** Token Spec 1.2.0 with two forcing
  instruments, `fluid_injection` and `reservoir_pressure`, and the unit `m3`. Adapters
  `uic_workbook` (regulator injection workbooks, read by a standard-library `.xlsx` reader) and
  `pressure_grid` (a reservoir model's grid CSV, RD New converted to WGS84). Regions `groningen`
  (KNMI quakes, NAM's pressure model) and `oklahoma` (ComCat quakes, every well's monthly volume).
- **CB-30, human-made quakes** (`craton/bench/induced.py`, `craton/bench/forcing.py`): 30-day
  forecasts of Groningen and Oklahoma from 2017, the forcing read only as far as it was known;
  the physics floor `etas_forced`; the forced Field (`craton/field/forced.py`,
  `configs/field/forced.yaml`) and its ablation (`scripts/ablations/forcing_blind.py`).
- `craton bench run --carry`: a run of new tasks copies the rest from the method's last
  accepted run when nothing they depend on changed, and says so (`carried_from`).
- `docs/assets/pump-*.png`, `scripts/figures/pump.py`, `lessons/P8.md` (P8e).
- **What the network missed (P8c, D-094, D-095).** Adapter `zenodo_catalog` (research catalogues
  published as one CSV on Zenodo, MD5-checked) and three machine-learning catalogues: Tan et al. 2021
  (central Italy 2016-17), Fonzetti et al. 2025 (L'Aquila 2009), Poggiali et al. 2025 (Alto Tiberina).
- **CB-20 and CB-21, enhanced inputs** (`craton/bench/enhanced.py`): CB-02's and CB-03's days and scored
  quakes with INGV plus every unmatched machine-learning quake as history, on INGV's magnitude scale.
  `ForecastSpec.train_task` lets a task reuse another task's trained models.
- `docs/assets/missed-*.png`, `scripts/figures/missed.py`, `lessons/P8.md` (P8c).
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
- **P8 is built in six parts** (D-078); the version stayed 0.8.x until the last one.
- The board doesn't change in P8f: every CB-02 to CB-30 score is P8e's.
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
