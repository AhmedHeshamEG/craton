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

## 2026-10-20 — P2 — First data: adapters and an immutable store

TLDR: Real data flows. Central Italy, 1985 to September 2026, is 1,758,860 valid tokens: 320,757
earthquakes from INGV, 751,394 station-days from 293 GNSS stations, and 686,709 solid-Earth tide values
computed by Craton's own port of the IERS 2010 model. Built twice from the same downloads, the snapshot is
the same, byte for byte.

Built:
- `craton/adapters/`: the adapter interface (`fetch` → `tokens` → `card`), a registry, regions, a polite
  HTTP client, the build pipeline, health collectors, generated data cards
- three adapters: `ingv_catalog` (FDSN event service, month by month), `ngl_gnss` (NGL IGS20 daily
  positions), `iers_tides` (body tide at every earthquake plus an hourly series)
- `craton/geo/`: frames, leap seconds and TT, the Sun and the Moon, the IERS 2010 body tide in NumPy
- `craton/store/raw.py`: raw downloads by sha256 with an append-only manifest; every token's
  `content_hash` points at its raw file
- `craton data list | fetch | build | cards`; data health on the scoreboard; `docs/data/`
- the coverage figure (`scripts/figures/coverage.py`), `lessons/P2.md`, an adapter guide in CONTRIBUTING
- tests: 200 → 231

Scores: no CratonBench task yet (P3). Data health, from `results/data_health.json`:
- INGV: 320,757 events, 126 refused (120 with the generic magnitude type `M`, 5 controlled explosions,
  1 anthropogenic event). The `M` events: 114 from 2009-2012 by author `UNKNOWN`, 6 from 1986-1990. Md 29,466, ML 290,892, Mw 399. Largest: Norcia Mw 6.5, 30 October 2016.
  Longest stretch with no event in the box: 10.9 days.
- NGL: 293 stations, the first from June 1995; median 1,927 days per station, median coverage 83%,
  964 gaps over 30 days. Nothing refused.
- Tides: 320,757 at earthquakes + 365,952 hourly; peak displacement 0.319 m up, 0.080 m north.
- Reproducibility: snapshot `96a163f798d8f9cc`. Every full build from the same 795 raw files gave the
  same 115 token files, byte for byte, and the same snapshot id.
- The catalogue jumps in 2005 (about 60 events a month in 2004, about 250 in 2006): INGV's modern
  processing starts, and so does ML. P3's splits have to respect it.
- Fetch: 501 INGV requests (35.9 MiB) and 294 NGL requests (148.9 MiB), one per second, each once.

Surprises / problems:
- PySolid is GPL-3 and Milbert's Fortran has no licence, so neither could be vendored. The tide is a
  port of the IERS routines instead (D-021). It matches the IERS test cases to 1e-12 m, PySolid to 0.4 mm
  and the JPL DE440 ephemeris to 0.71 mm. The fourth test case published in the IERS routine's header is
  broken (its Sun is 8.4 million km away; its expected output repeats the third case).
- pyTMD calls its northward component "N" but it points south (colatitude). Caught by comparing with
  PySolid; the reference script flips it.
- The Nevada Geodetic Laboratory states no licence. P1's GNSS example said CC-BY-4.0; corrected (D-024).
- INGV's FDSN service refuses requests over 20,000 events (HTTP 413). No central-Italy month reaches it
  (November 2016 is the busiest, 13,209), but the adapter splits such months anyway; a test covers it.
- INGV reports Md until 2004 and ML from 2005. The first health report estimated completeness for ML only
  and showed nothing before 2005. It now estimates each type separately, never across types (D-027).
- The first snapshot id hashed the health numbers too, so a better statistic would have renamed
  unchanged data. The id now hashes only region, files and raw inputs.
- `.gitignore` had `data/`, which also hid `docs/data/`; it is now `/data/`.

Decisions: D-021 … D-028.

Next: P3, CratonBench v1. Freeze the splits against this snapshot (Apennines 2016–17, L'Aquila 2009,
held-out GNSS stations), then score climatology, ETAS and IDW + trend before any model exists.

## 2026-10-20 — P3 — CratonBench v1: the test before the model

TLDR: Four tests are frozen and the simple methods are scored, before any model exists. ETAS forecasts
the 2016-17 Apennines at −7.12 bits per M3+ quake, 7.5 bits better than a flat Poisson rate. A hidden
GNSS station is rebuilt to 17.8 mm, and a masked measurement costs climatology 22.7 bits. The training
splits are written to disk and read back token by token: no held-out token in 2.5 million.

Built:
- `craton/bench/`: splits and leak checks, the four tasks with fingerprints and protocols, distributions,
  CSEP tests, ETAS, four baselines, the harness, the frozen record `frozen/central_italy.json`, the
  benchmark card `docs/bench.md`
- `craton bench list | prepare | leaks | run | cards`; a "benchmarks frozen, no leaks" step in `craton check`
- two ratchet lanes (Craton and the floor) and the floor on the scoreboard
- the Apennines forecast figure (`scripts/figures/apennines_forecast.py`), `lessons/P3.md`, a method guide
  in CONTRIBUTING, the pyCSEP reference script and fixture
- tests: 231 → 270

Scores (mean ± std over seeds 0, 1, 2; the floor, from `results/scoreboard.json`):
- CB-01 climatology 22.66 bits per masked token (quakes 22.5, GNSS days 22.8; 28,000 queries)
- CB-02 ETAS −7.118 bits per M3+ quake, Poisson −14.64. 1,065 M3+ quakes over 494 days; ETAS expected 583,
  Poisson 75. At M1.5+: ETAS −2.76, Poisson −9.69.
- CB-03 ETAS −9.396, Poisson −16.94. 315 M3+ quakes in 2009; ETAS expected 70.
- CB-04 IDW + trend 17.77 mm (east 24.2, north 15.2, up 11.4) over 29 stations and 86,373 station-days
- The three ETAS restarts land on the same optimum (spread 1e-8 bits). The other baselines have no
  randomness; their std is 0.

Surprises / problems:
- The first CB-01 design hid only values. A GNSS token's lat/lon are that day's position to 1e-10°, so
  the answer was still in the coordinates. Position is hidden now, and so is the tide token at each
  masked quake, which sits at its epicentre (D-031).
- The first CB-04 run gave 22 m. NGL reuses station codes for different sites: MIR2 jumps 595 km north
  in a day, PES4 2.3 km, PERU 1.2 km, VITE 0.9 km, SMRA 0.6 km, REGU 351 m, BLRA 116 m, TOD3 3.8 m.
  Eight codes jump more than 1 m between two days; the largest real jump is ARQT's 0.447 m (Norcia). A
  station is now one monument only below 1 m (D-030). None of the eight was held out; TOD3, 0.6 km from
  held-out RETO, had wrecked its rebuild through the neighbour weights.
- P2 said INGV reports Md until 2004 and ML from 2005. Md continues after 2005: 7,157 events from 2005 to
  August 2016, 291 of CB-02's 42,807 test quakes. The forecast tasks keep the catalogue's preferred
  magnitude of any type and count them (D-032).
- CB-03's ETAS fit (four mostly quiet years, 1,484 quakes) is supercritical: branching ratio 1.26. The
  daily forecasts stay finite, but the parameters are poorly pinned (D-034).
- ETAS underestimates the branching ratio by about 0.1 on synthetic catalogues, because aftershocks that
  land outside the box are never seen. Known and tested.
- pyCSEP installs on Windows, but brings cartopy, rasterio and obspy; it is the referee in a throwaway
  environment. Craton's numbers match it exactly for the likelihoods and within simulation noise for the
  S and M quantiles (D-033).
- Two heredoc edits turned a `"\n"` into a real line break and broke a file; caught by the tests.
- The plan's neural point process baseline (arXiv 2301.09948) waits for P4: it is a neural model, and
  its headline catalogue isn't public in Craton's form (D-038).

Decisions: D-029 … D-038.

Next: P4, Craton-mini v0. A transformer on the catalogue alone, implementing `Forecaster` and
`MaskedPredictor`, trained on the `pretrain` and `cb02_fit` splits, scored against this floor.

## 2026-10-20 — P4 — Craton-mini v0: the first model

TLDR: The first transformer trains on the laptop in 8 to 16 minutes a run and never passes 2.6 GB. It
beats climatology on CB-01 by a bit per measurement (two bits on the quakes alone) and beats a flat
Poisson rate on both forecasts, but loses to ETAS by 3.8 bits per M3+ quake on 2016-17 and 5.7 on 2009.
Fully scored, on the board, losses included.

Built:
- `craton/model/`: the event encoder (registered by instrument), 4D rotary positions over time,
  latitude, longitude and depth with a sinusoidal fallback, the pre-norm trunk on fused attention, the
  binned masked head and the next-day head (Poisson rate × cell × magnitude), both losses in bits
- `craton/train/`: masked windows and next-day questions, the fp16 loop with gradient scaling, warmup +
  cosine and early stopping, checkpoints named by split, config, seed and code version, model cards
- the bench method `craton-mini`, `craton train SPLIT`, `docs/models/craton-mini.md`
- CB-15 (MFU inside a 6 GiB gate), frozen into the record by the new add-only `--freeze`
- the first-model figure (`scripts/figures/first_model.py`), `lessons/P4.md`, an encoder guide in
  CONTRIBUTING
- tests: 270 → 297

Scores (mean ± std over seeds 0, 1, 2; run `20261020-105609-craton-mini`, 7,467 s for nine trainings and
scoring):
- CB-01 21.66 ± 0.026 bits (climatology 22.66). Quakes 20.49 (climatology 22.48): latitude 6.89 (7.42),
  longitude 6.22 (7.02), depth 3.33 (3.37), magnitude 4.06 (4.67). GNSS answered by climatology (D-042).
- CB-02 −10.88 ± 0.67 (ETAS −7.12, Poisson −14.64). Seeds −10.22, −10.85, −11.57. Expected 320 M3+ quakes,
  observed 1,065 (ETAS 583). M1.5+: −6.40 (ETAS −2.76, Poisson −9.69).
- CB-03 −15.13 ± 0.79 (ETAS −9.40, Poisson −16.94). Expected 495 M3+ quakes, observed 315: it overshoots.
- CB-15 15.6 ± 0.1% MFU: 85,000 tokens/s, 1,347 MiB peak on the CB-01 run (8.0 min). Forecast runs peak
  at 2.5-2.6 GB.

Surprises / problems:
- The first forecasts expected 56-63 M3+ quakes for CB-02 (Poisson territory). It reacted to L'Aquila 2009
  (a training day) but collapsed after Amatrice: only 13 training days have over 200 quakes, attention
  can't count, and the final layer norm squashes scale. A clock per token and a straight path from six
  measured recent rates into the log rate fixed the count (trial: 579 expected, ETAS 583) and moved the
  score from −14.26 to −10.16 on one seed (D-045). The final seeds land between −10.2 and −11.6: the
  forecast is not stable across seeds yet.
- CB-03's model has one big sequence to learn from (Colfiorito 1997, in Md) and over-forecasts 2009.
- The loss first used log N! for the day where CB-02 uses log n! per cell and bin. A constant for
  training, but the claim was "the loss is the score"; a test caught it and the loss now matches exactly.
- A hidden quake's coordinates were still in the input tensor (the rotary blocks zeroed them, so the
  model never saw them; a model-level test proved it). They are now zeroed where the batch is built.
- The figure's first draft titled itself "looks where the aftershocks are". Measured, the median head
  puts 54% of its attention within 10 km of the hidden quake, where 55% of the window is; the most local
  of 48 heads puts 74%. The figure now says exactly that.
- Most runs hit the 2,500-step limit still improving; early stopping fired once (cb02_fit seed 2, step
  1,900). More steps are cheap and the next phase should take them.
- The fused attention backward is not bitwise deterministic on the 2060 (D-046).
- Twice more a shell heredoc turned `"\n"` into a real line break in Python source; caught by the parser.
  Multi-line source edits now go through the editor.

Decisions: D-039 … D-047.

Next: P5, fusion. Register GNSS and tide encoders, build the context selector, add cross-modal masking,
and run the ablation that shows each modality's share; CB-17 arrives.

## 2026-10-21 — P5 — Fusion: quakes + GPS + the Moon

TLDR: Craton-mini reads the catalogue, daily GNSS and the solid-Earth tide in one trunk. CB-01 drops
from 21.66 to 15.12 bits: hidden GNSS days cost 9.75 bits instead of 22.8, hidden quakes tie P4. It
rebuilds unseen GNSS stations a little better than IDW (17.50 vs 17.77 mm) and adapts to a never-seen
region from context (CB-17, new), though counting the context still adapts better. The tide's share
is zero within noise, everywhere. Forecasts: CB-02 the same as P4, CB-03 better and steadier; ETAS
still ahead on both.

Built:
- encoders for `gnss_daily` and `solid_earth_tide`, dispatch by instrument id, modality vectors,
  `model.inputs`, a padding mask in the trunk (D-048)
- `craton/train/context.py`: station lines, the context policy, `write()`; GNSS heads in mm on CB-01's
  bins (D-049, D-050)
- window kinds: quake, GNSS-centred (random, whole station, all GNSS, all quakes; D-051), region
- CB-17 and the `cb17_fit` split; floors `climatology` and `context_kde`; freezing grows by splits
  (D-052)
- `craton bench ablate`, `docs/ablation.md`, the ablation figure
- a worker process for batches, pinned non-blocking copies, fused AdamW (D-055)
- the capped rate correction (D-057); model cards that say what a checkpoint reads
- tests: 297 → 319

Scores (mean ± std over seeds 0, 1, 2; run `20261021-122053-craton-mini`):
- CB-01 15.12 ± 0.37 (P4 21.66): quakes 20.48 (P4 20.49), GNSS 9.75 (climatology 22.83)
- CB-02 −10.98 ± 0.77 (P4 −10.88, ETAS −7.12): same within noise, P4 stays champion
- CB-03 −14.05 ± 0.31 (P4 −15.13 ± 0.79, ETAS −9.40): better mean, within noise; P4 stays champion
- CB-04 17.50 ± 0.04 mm (new; IDW + trend 17.77)
- CB-15 16.27 ± 0.04% MFU (P4 15.59): 88,750 tokens/s, 1.4 GB peak, 12.3 min
- CB-17 24.56 ± 0.56 bits (new): 32.4 with no context → 24.6 with a year; climatology 27.72,
  context_kde 19.03
- Ablation (docs/ablation.md): GNSS +13.3 bits on GNSS days, nothing on quakes or forecasts; tides
  nothing anywhere; quakes +0.25 mm on CB-04 (within noise)

Surprises / problems:
- The first fused model scored CB-01 16.05 and would have passed the ratchet, but its quake half had
  got worse (22.47 vs 20.49): GNSS gains hid a quake loss. The cause was the data mix: region and
  GNSS-centred windows. Reading GNSS helps quakes (20.17 vs 20.54 without); playing GNSS games costs
  them 1.3+ bits at every dose tried (D-053, D-056).
- The rebuild measured each token's age from the window's middle (later from a GNSS day a week
  earlier): before and after blurred. Back to P4's rule (D-056).
- CB-03 swung from −14 to −22 across seeds (1,412 quakes expected where 315 came) once the model
  trained longer: the trunk's free rate correction extrapolated wildly in a crisis bigger than any in
  training. Capped at 2x (D-057).
- MFU first fell to 8.9%: CPU batch-building fought the training loop for the GIL, and blocking copies
  left the GPU idle. Fixed (D-055).
- Two background runs were stopped by the system for low RAM (16 GB): main process + worker each hold
  the data. Rerun with heavy apps closed.
- The ablation was measured before the rate cap (D-057 note).
- CB-04's floor error is two-thirds one station, VCRA (jumps of up to 315 mm a day) (D-054).
- The GNSS half of CB-01 beats a weak floor (a station's mean). A same-station interpolation baseline
  would be the honest bar; added to IDEAS for P6.

Decisions: D-048 … D-057.

Next: P6, the 8 locks. Calibration, ensembles, the OOD detector, physics budgets, locked forecasts,
and an honest interpolation baseline for CB-01's GNSS half.

## 2026-10-22 — P6 — Beat ETAS: the Field

TLDR: Craton is above ETAS on both forecasts for the first time. The forecast is now the Field: the
1,050 forecast cells as tokens with a fading memory of their quakes, attention between them, and ETAS
as the starting line. CB-02 −10.98 → −6.82 (ETAS −7.12), CB-03 −14.05 → −8.23 (ETAS −9.40). Almost
all of it is how many; *where* beats ETAS on 2009 and misses by 0.02 on 2016-17.

Built:
- the bit split (how many / where / how big / shuffle) and the score without M5+ days, as extras of
  CB-02 and CB-03 (D-058)
- `craton bench ceiling` and the building on the scoreboard and README (D-059)
- floors `interpolation`, `etas_aniso`, `etas_incomplete` (D-060)
- `craton/field`: Omori memory, GPU ETAS base, `FieldNet`, the forecaster, its own config and
  checkpoints; `craton-mini` forecasts through it (D-061, D-062, D-063, D-064)
- the building figure; ROADMAP gains P6 and every later phase moves down one
- tests: 319 → 329

Scores (mean ± std over seeds 0, 1, 2; run `20261022-014206-craton-mini`):
- CB-02 −6.818 ± 0.016 (P5 −10.98, P4 champion −10.88; ETAS −7.118, ETAS-incomplete −7.059). How
  many +0.32 over ETAS, where −0.02, without M5+ days −0.09. 1,118 M3+ expected, 1,065 seen.
- CB-03 −8.233 ± 0.006 (P5 −14.05, champion −15.13; ETAS −9.396, ETAS-incomplete −9.295). How many
  +1.02, where +0.15, without M5+ days +0.70. 216 expected, 315 seen.
- CB-01 15.12 (quakes 20.48, GNSS 9.75), CB-04 17.50 mm, CB-15 16.27%, CB-17 24.56: unchanged, same
  trunk checkpoints.
- Floors: interpolation CB-01 16.07 (GNSS half 9.65, ahead of Craton's 9.75); etas_aniso −7.119 /
  −9.401; etas_incomplete −7.059 / −9.295 (blind time 0 s, productivity ×1.1).
- The building (`results/ceiling.json`): count +2.10 / +3.55, where to ~10 km +0.60 / +0.66, to the
  cell +2.11 / +2.45, roof +4.21 / +6.00.

Surprises / problems:
- The spike (memory + cell tokens + ETAS start, no place context) already beat ETAS: −7.02 / −8.72.
  The trunk's view of each cell was never needed.
- All of the gain was how many. Without attention between cells *where* got worse by 0.36 bits.
- One Field per seed swung CB-02 by ±0.13; four averaged, ±0.02 (D-062).
- Stretched aftershock zones: zero. The small aftershocks already draw the fault.
- The blind-time floor chose a blind time of zero: daily counts can't see minutes of blindness.
- ETAS with M1.5 parents ran out of RAM on CB-02 (pairs inside L'Aquila); dropped as a base.
- ETAS over ~4,700 training days took an hour on the CPU; the GPU version takes seconds and matches to
  5·10⁻¹².
- The *where* exit test is missed on CB-02 by 0.02, all on the Norcia and Campotosto days (D-064).
- On CB-02's calm days ETAS is still 0.09 ahead; the Field wins the busy ones. M1.5+ on CB-02 is worse
  than ETAS (−2.95 vs −2.76): small quakes are extended from the M3+ output by Gutenberg-Richter.
- A new attribute called `field` on the method shadowed `dataclasses.field` in its own class body; lint
  caught it before any run.

Decisions: D-058 … D-064.

Next: P7, the locks. Calibration, ensembles (the Field already averages four), the OOD detector, physics
budgets, locked forecasts and `craton referee`.

## 2026-10-22 — P7 — The 8 locks: knowing when it doesn't know

TLDR: Five new tests a forecaster can fail out loud (calibration, an unseen region, fake
precursors, physics, the predictability meter), the referee, and locked forecasts: from tonight
one forecast a day is sealed and published before its day, then graded in public. Craton makes
you watch 0.84% of space-time to catch 90% of the M4+ quakes, against ETAS's 1.32%, breaks no
physics (every ETAS floor breaks the slip budget), and is worse calibrated cell by cell than the
floors. The referee kills the project's own tidal "discovery" of the day before.

Built:
- `craton/trust`: predictive distributions, calibration, the meter, physics checks, fake-precursor
  attacks, the input guard, the out-of-distribution detector, the referee, the seal (D-065 … D-073)
- CB-05, CB-06, CB-07, CB-08, CB-11 in `craton/bench/locks.py`, frozen (growth only)
- region `southern_italy` (61,086 quakes), read only by CB-06; floors `no_detector`,
  `fingerprint_knn`
- the Field: six members grown from P6's four (D-076), no magnitude above 7.3, the guard in front
  (D-067); negative binomial dispersion fitted on validation days (D-066)
- `craton/sim` with the ETAS planet (D-075); `craton referee`; `craton live run|bundle|page|verify`,
  the nightly workflow, `forecasts/`, and a check that refuses edits to locked files (D-074)
- the meter figure; tests: 329 → 360

Scores (mean ± std over seeds 0, 1, 2; run `20261022-123525-craton-mini`):
- CB-05 calibration error 0.32 ± 0.08 doublings (new; ETAS 0.22, ETAS-incomplete 0.10, Poisson
  1.58). ECE on committed cell-days 0.012. Day totals: PIT distance 0.039 (best of all; ETAS
  0.061), 97% of days inside the 90% interval.
- CB-06 AUROC 0.997 ± 0.001 (new; fingerprint floor 0.998, no detector 0.5). Trunk alone 0.26
  average. Abstains on 100% of Calabria and Sicily, 72% of central Italy's sequence days.
- CB-07 1.002 ± 0.004 (new; ETAS 1.002). GNSS step and duplicates 1.000; magnitude shift 1.006.
- CB-08 0 violations (new; ETAS, ETAS-aniso, ETAS-incomplete 1 each: the slip budget).
- CB-11 0.84 ± 0.06% of space-time (new; ETAS 1.32%, Poisson 24%): 213 alert days a year
  against 282, radius 40 km against 38.
- CB-02 −6.815 ± 0.013, CB-03 −8.231 ± 0.006 (P6 −6.818, −8.233; champions stay, within noise).
  CB-01, CB-04, CB-15, CB-17 unchanged.

Surprises / problems:
- The trunk's own vectors fail the unseen-region test (AUROC 0.12-0.38): central Italy's sequences
  look stranger to it than Sicily, and it can't see depth. Twelve hand-made numbers do the job.
- The 2026-10-21 tidal claim, replayed with the computed tide: raw p = 10⁻⁴¹³, declustered
  10⁻³·³, and it "predicts" 2009 at the same phase (10⁻³·⁹). Shifted tides lock as hard 71% of
  the time: the tide's phase isn't evenly spread in time. Rejected.
- First CB-07 design had no teeth: at an alarm level of 1% a day no fake moved any alarm. Now 0.1%.
- The guard's first magnitude rule fired on 6,800 real quakes; two rounds later, never in a test
  window (24 stretches in 40 years when run over everything).
- A single negative binomial size per checkpoint was far too wide inside sequences; the mean-
  dependent size fixed the day totals.
- The first P7 run retrained all Fields and came in a few thousandths under P6. Pure GPU noise, but
  the wrong design: ensembles now grow (D-076).
- The live mechanics test caught a bug: counting quakes on non-consecutive days.
- Field training ran 5x slower while CPU-heavy scripts shared the machine.

Decisions: D-065 … D-076.

Next: P8, fake planets: simulated catalogues where the answer is known, plugged into the
simulator registry that P7 left ready.

## 2026-10-25 — P8a — Two fake planets

TLDR: P8 is split into six parts (D-078), and this is the first: two kinds of simulated planet,
where the truth is known, and CB-09 to score on them. On the ETAS planet (Italy's own fitted ETAS
is the whole truth) Craton ties: 0.02 bits per quake below fitted ETAS, 0.002 below the truth. On
the physics planet (rate-and-state faults, stress shadows, a hidden fluid source) it beats ETAS
by +0.19 bits per quake on all three planets, all of it in *where*. That's beyond noise, and a
fifth of the +1 the plan hoped for. Every older number is unchanged.

Built:
- `craton/sim/rate_state.py`: the physics planet. Dieterich (1994) seismicity on ~1,500 patches
  along 36 Apennine-shaped faults plus a diffuse grid, Coulomb stress steps from every M2.5+, one
  or two fluid sources, an exact integrator and sampler (D-079)
- `etas_italy`, the ETAS planet: the `etas` floor's CB-02 fit, rounded (D-080)
- `craton/bench/planets.py`: CB-09 on six frozen planets with CB-02's calendar, `PlanetContext`,
  the truth forecast, the sim-to-real gap (D-081, D-082)
- the planets figure; `lessons/P8.md` (P8a); tests: 360 → 378

Scores (mean ± std over seeds 0, 1, 2 = planets 0, 1, 2; run `20261025-132227-craton-mini`):
- CB-09 −14.38 ± 0.42 bits per M3+ quake on the physics planet (new; ETAS and
  ETAS-aniso −14.57, Poisson −14.73). Craton − ETAS per planet: +0.165, +0.220, +0.195.
- ETAS planet: Craton −14.146, fitted ETAS −14.124, the truth −14.143. |Δ| = 0.022 < 0.05.
- Bit split on the physics planet: *where* +0.19, *how many* +0.00, *how big* 0.00.
- Sim-to-real gap: ETAS planets 1.32 doublings off the real Apennines, physics planets 0.88.
- CB-01 … CB-17: every number identical to P7 (the Field checkpoints are reused).

Surprises / problems:
- The QDYN spike failed: no Fortran toolchain, and it simulates one fault's cycle, where a
  forecaster needs a region's catalogue. The population form of rate-and-state does that job.
- The first freeze gave the ETAS planet b = 1.02 against a truth of 1.16. Magnitudes were simulated
  from exactly 2.0 and rounded, which half-empties the first bin. Caught by the gap table before
  any commit; planets now start half a step lower and the record was re-frozen.
- The P7 ETAS planet was too weak to be a test (0.7 quakes a day, test windows of 5-18 M3+).
  The new one is Italy as ETAS sees it.
- Fitted ETAS scores slightly above the truth on all three ETAS planets (+0.02): over one
  16-month window the truth is only the best forecast on average.
- On the physics planet a map of the past (Poisson) is only 0.16 bits behind ETAS; on the ETAS
  planet it is 2.6 behind. Small quakes trigger almost nothing there, so clustering is weak and
  there is less for any forecaster to win.
- Both planets are far smoother than the Earth: the busiest 1% of days hold 31% of the real
  quakes and 5-10% of a planet's.
- The Field's validation on the ETAS planet barely moves off its ETAS start, as it should
  where there is nothing to learn.

Decisions: D-078 … D-082.

Next: P8b, CB-19: a 30-day forecast, where GPS strain is known to pay.

## 2026-10-25 — P8b — Thirty days ahead

TLDR: CB-19 is CB-02 a month at a time: sixteen 30-day forecasts of the 2016-17 sequence. ETAS wins
it. Craton is 0.22 bits per quake behind, on every seed, against an expected +0.2. Without the
months that hold an M5+ the two are level; the loss is Norcia's month and Campotosto's. GPS added to
the Field gains +0.13 on every seed, all of it in the count, most likely from coseismic jumps and
afterslip; it joins every Field in P8f. Every older number is unchanged.

Built:
- `ForecastSpec.period_days` and CB-19 (D-083): sixteen whole periods, the last 14 days left out
- ETAS over any span, cascade stretched to it, on the CPU and the GPU (tested equal)
- the Field at 30 days (D-084): ETAS's 30-day start, 30-day answers by a running sum, training
  starts that reach a validation block dropped; `long_periods` on the forecasters that can answer
- `scripts/ablations/gps_thirty_days.py` (D-085); the thirty-days figure; `lessons/P8.md` (P8b);
  tests: 378 → 383

Scores (mean ± std over seeds 0, 1, 2; run `20261025-195238-craton-mini`):
- CB-19 −6.226 ± 0.13 bits per M3+ quake (new; ETAS −6.002, Poisson −10.517). Craton − ETAS per
  seed: −0.232, −0.352, −0.088.
- Bit split against ETAS: *how many* −0.10, *where* −0.12. Months without an M5+: −4.267 vs −4.254.
- With GPS (ablation, not on the board): −6.096 ± 0.10; per seed +0.06, +0.22, +0.11 over the Field.
- CB-01 … CB-17: every number identical to P8a (the Field checkpoints are reused).

Surprises / problems:
- The plan's source for "GPS strain is known to pay at 30 days" can't be found. The real evidence
  is GNSS strain rate as an ETAS background at Campi Flegrei (Petrillo & Taroni 2025), no
  pre-seismic GNSS motion before Amatrice or Norcia, and a seismic-velocity change before Norcia
  (Chiarabba et al. 2020). The roadmap is corrected.
- Seed 0 month by month: the Field mapped Norcia's month far better (+263 bits on *where*) and
  still lost it by 103, because it expected 9 M3+ quakes where 598 came (ETAS 14). Campotosto's
  month cost 150 more, on *where*: the Field had narrowed its map to the active faults.
- In the quiet months of 2017 ETAS keeps expecting 10-20 quakes a month when 3-13 come; the Field
  is closer in most of them.

Decisions: D-083 … D-085.

Next: P8d, one model on every region: global ComCat, JMA, GeoNet, AFAD, CSN, and CB-22.

## 2026-10-26 — P8d — The whole Earth

TLDR: One Field, trained on central Italy, California, Japan and New Zealand up to 2007, forecasts
places it never read better than ETAS fitted there: +0.36 bits per quake on Turkey, Chile and
Alaska (CB-22's never-seen half), and every held-out sequence in the regions it trained on is above
ETAS (+0.35). On EarthquakeNPP (CB-29), the outside benchmark where five neural forecasters lost to
ETAS, it beats their own ETAS on their own data by 0.089 nats per quake, on every seed. One honest
catch: ETAS-incomplete, a better floor, is level with Craton on the never-seen half.

Built:
- regions as YAML with `sources` and `completeness`; California, Japan, New Zealand, Turkey,
  Chile, Alaska and EarthquakeNPP's ComCat_25 (D-086, D-088)
- adapters `fdsn_catalog` (ComCat, GeoNet, EMSC), `jma_hypocenter`, `earthquakenpp` (D-088)
- Token Spec 1.1.0: `M`, `Mj`, `Mh`, `mb_Lg`, `MLv`; tokens written under the oldest spec that
  accepts them (D-087)
- every region in its own magnitudes; forecast specs carry mc, cut-off and thresholds (D-089)
- the world Field (`craton/field/world.py`), 42 tiles of four regions, cut 2007-01-01 (D-090)
- CB-22 (`craton/bench/world.py`, D-091) and CB-29 (`craton/bench/npp.py`, `craton/field/npp.py`,
  D-092), frozen
- `scripts/ablations/world_on_italy.py`; the world figure; `lessons/P8.md` (P8d); tests 383 → 402

Scores (mean ± std over seeds 0, 1, 2; run `20261026-102634-craton-mini`):
- CB-22 +0.361 ± 0.083 bits per quake above ETAS, never-seen regions (new). Turkey +0.43, Chile
  +0.48, Alaska +0.17: all three regions above ETAS on every seed. Counted sequences above ETAS:
  6, 5 and 6 of 7 by seed (Chignik 2021 below on every seed, −0.21; Elazig level, +0.01). Seen
  regions: +0.354, all six sequences above ETAS on every seed (Kumamoto +0.54, Tohoku +0.46, El Mayor +0.42,
  Ridgecrest +0.39, Kaikoura +0.18, Darfield +0.12).
- CB-22 floors: ETAS-incomplete +0.336 (the floor), ETAS 0, ETAS-aniso −0.106, Poisson −5.26.
  Against ETAS-incomplete, Craton is +0.31 on the seen half, +0.22 in Turkey, −0.11 in Chile and
  −0.03 in Alaska.
- CB-29 7.167 ± 0.002 nats per quake (new; their ETAS 7.256 here and 7.2554 as they publish it,
  Poisson 13.26): +0.088, +0.091, +0.088 per seed.
- World Field on central Italy (ablation, not on the board): CB-02 +0.21 over ETAS (−6.90; the
  Italy Field −6.82), CB-03 +0.79 (−8.61; the Italy Field −8.23). It has seen Italy only to 2007.
- CB-01 … CB-19: every number identical to P8b.

Surprises / problems:
- Maule 2010 broke the first score: −17 bits per quake. Its box holds 75 ComCat quakes above M4.5
  in twenty years; ETAS's fit ran to its bounds and expected 295 M5.5+ quakes the day after Maule
  when one came, and the Field doubled that. A sequence now counts only if its ETAS fits on 200+
  quakes (the others: 297-9,500). The rule was added after seeing Maule; with Maule counted, the
  never-seen half is negative on every seed (−1.61, −0.80, −1.86).
- The never-seen sequences are small: 34-49 scored quakes each in Chile and Alaska, 54 for
  Elazig (377 for Kahramanmaras). One by one their gains are noisy; the region averages, and the
  same sign on every seed, are what the result rests on.
- ETAS-incomplete is the real floor in Chile: ComCat sees Chile through the global network, which
  goes blind for hours after a great quake, and a blind-time model captures exactly that (+0.81 on
  Iquique). The world Field never learned blindness from a global network, because none of its
  regions is one.
- Italy's learning rate peaked after one epoch on every member of the world Field (+0.05 nats per
  quake on validation); 3·10⁻⁴ with validation every 500 steps reaches about +0.13. Seed 0's first
  run was stopped and retrained.
- Two trainers on memory-mapped pools held 11 GB of RAM, leaving under 2 GB free; one was
  stopped and the seeds trained one after another.
- AFAD's API returned nothing; Turkey comes through EMSC, whose yearly counts swing by four times
  as its contributors change.
- Their CSV has no depth and no magnitude type, so CB-29's tokens say depth unknown and type `M`.
- Unlike P8b's missing paper, everything the plan cited exists: EarthquakeNPP, its repository,
  its fitted parameters and its scores.

Decisions: D-086 … D-092.

Next: talk the result through with Hesham (the stop rule), then P8c, the ML catalogues: CB-20 and
CB-21.

## 2026-10-29 — P8d — The verdict

TLDR: P8d passed its stop rule on both counts, and the plan goes on. It is the first win on a
benchmark Craton didn't design, and the first sign the Field carries to places it never read. It
isn't a breakthrough: on the never-seen regions ETAS-incomplete is level with it, and the rule
that leaves Maule out was written after seeing Maule (with Maule, −1.42 on average). From now on
never-seen regions are claimed against ETAS-incomplete (D-093).

Decisions: D-093.

Next: P8c, the ML catalogues: CB-20 and CB-21.

## 2026-10-29 — P8c — What the network missed

TLDR: machine-learning catalogues of the 2016-17 sequence and of L'Aquila 2009 are now inputs, through
two new tasks that score CB-02's and CB-03's quakes with a richer history. The better catalogue helps
ETAS. It hurts Craton on 2016, which falls below ETAS on CB-20, because the Field had learned INGV's
gaps as a constant and counts them twice once they're filled. On L'Aquila Craton stays 1.1 bits above
ETAS.

Built:
- `zenodo_catalog` adapter (one CSV per Zenodo record, columns named in the region's YAML, MD5-checked);
  regions `central_italy_ml_2016` (Tan et al. 2021), `laquila_ml_2009` (Fonzetti et al. 2025),
  `alto_tiberina_ml` (Poggiali et al. 2025, built, not read by any forecast) (D-095)
- `craton/bench/enhanced.py`: the union of INGV and the unmatched machine-learning quakes (3 s, 10 km),
  magnitudes moved onto INGV's scale by a quantile map fitted below M3 (D-094); CB-20 and CB-21, frozen
- `ForecastSpec.train_task`: CB-20 and CB-21 reuse CB-02's and CB-03's trained Fields
- data health reads a token's year from its time; the scoreboard rounds the largest magnitude
- `scripts/figures/missed.py` and its figure; `lessons/P8.md` (P8c); tests 402 → 412

Scores (mean ± std over seeds 0, 1, 2; run `20261029-105451-craton-mini`):
- CB-20 −7.028 ± 0.017 (new). ETAS −6.985, ETAS-incomplete −6.955 (the floor), ETAS-aniso −6.987,
  Poisson −14.64. Craton is 0.07 below the floor, on every seed (−7.048, −7.021, −7.016).
- CB-21 −8.224 ± 0.007 (new). ETAS −9.360, ETAS-incomplete −9.260, ETAS-aniso −9.364, Poisson −16.94.
  Craton +1.04 over the floor, +1.14 over ETAS.
- The catalogue's worth to each method (CB-20 − CB-02 / CB-21 − CB-03): ETAS +0.13 / +0.04,
  ETAS-incomplete +0.10 / +0.04, Craton −0.21 / +0.01.
- CB-01 … CB-29: every other number identical to P8d.

Surprises / problems:
- The machine-learning catalogues each use their own local magnitude: Tan's reads a median 0.26 below
  INGV's, Poggiali's 0.50, Fonzetti's 0.09, and not by a constant (Tan's M0 is INGV's 0.55, its M3
  INGV's 3.15). Without the map, the union would have added thousands of fake small quakes.
- Fonzetti's catalogue has *fewer* quakes than INGV in the six hours after L'Aquila (91 against 226):
  machine-learning pickers drown in overlapping waveforms too. That's why the input is a union.
- The Field's loss on CB-20 is all in *how many*: it expects 1,558 M3+ quakes where 1,065 came (on
  CB-02, 1,082). *Where* scores −4.20 bits on both. It multiplies ETAS by 1.86 on CB-02 and still by
  1.94 when the gaps are filled (D-096).
- Fonzetti's relocated file has its latitude and longitude headers swapped.
- Each full baseline run takes about an hour (ETAS-incomplete 2.6 h); craton-mini 1.7 h.

Decisions: D-094, D-095, D-096.

Next: P8e, Groningen and Oklahoma (pressure → quakes).

## 2026-10-29 — P8e — Where people pushed the crust

TLDR: two places where the push behind the quakes is measured, Groningen's gas field (the
reservoir's pressure, one map a year) and Oklahoma's wastewater (every Arbuckle disposal well,
month by month), make a new task, CB-30: eight years of 30-day forecasts from 2017, when both
pushes were being turned down. Craton beats ETAS by +0.71 bits per quake and a physics floor
that follows the pump by +0.25, on every seed. But the same Field trained without the pump
does almost as well (+0.67): the gain is from reading the quakes, and "pressure → quakes" was
not learned in a way the score can see.

Built:
- Token Spec 1.2.0 (D-097): instruments `fluid_injection`, `reservoir_pressure`, unit `m3`; a
  token must name a spec at least as new as its instrument
- adapters `uic_workbook` (regulator injection workbooks) and `pressure_grid` (a reservoir
  model's grid CSV), with a standard-library `.xlsx` reader and an RD New → WGS84 converter;
  `fdsn_catalog` takes a region's `magnitude_aliases` and a source `start` (D-098)
- regions `groningen` (1,866 KNMI quakes, 474,099 pressure tokens, snapshot b392ac5e4c87e6dc) and
  `oklahoma` (16,838 ComCat quakes, 1,029,067 well-month tokens, snapshot 1989705b223c5506)
- `craton/bench/forcing.py`: forcing tokens → pushes → per-cell features, nothing read before
  its period has ended; one reader per instrument (`PUSH_READERS`), the extension point
- `craton/bench/induced.py`: CB-30, frozen; the physics floor `etas_forced` (D-099)
- `craton/field/forced.py`, `configs/field/forced.yaml`: the Field with ten forcing features
  per cell, both cases at once; `craton-mini` answers CB-30 through it
- `scripts/ablations/forcing_blind.py`; `scripts/figures/pump.py` and its figure
- `craton bench run --carry` (D-100); data cards for both new sources; tests 412 → 429

Scores (mean ± std over seeds 0, 1, 2; run `20261029-164113-craton-mini`):
- CB-30 +0.707 ± 0.045 (new). Groningen −10.196 bits per M1.5+ quake (ETAS −10.553, floor
  −10.517), Oklahoma −9.791 per M3+ quake (ETAS −10.848, floor −9.972). Floor +0.456,
  ETAS-aniso −0.0004, Poisson −1.137.
- Without the forcing: Groningen −10.213, Oklahoma −9.847 (seeds −10.28, −9.37, −9.89), CB-30
  +0.67. The pump's worth: +0.02 and +0.06, inside the noise; it narrows Oklahoma's seed spread
  from 0.46 to 0.05 bits.
- Every other number identical to P8c (carried, D-100).

Surprises / problems:
- KNMI's FDSN service writes its magnitude `MLn`, and `MLnq` before 2015; the first build kept
  only the 842 quakes from 2015 on.
- The proportional law (rate ∝ push) fits neither place: its fitted share came out at zero.
  Oklahoma's quakes rose 30× while the volume doubled, and fell 97% while it fell by half. What
  fits is the cell's own background turned by the push to a power near 2 to 3.
- Oklahoma's floor first read every disposal well and lost to ETAS on the test; Arbuckle
  disposal fell 67%, the rest didn't fall. The case now reads the Arbuckle only. That choice was
  made after seeing the floor's test score, and is logged as such (D-099).
- Every forecaster still expects far too much of Oklahoma: 609 M3+ quakes came, ETAS expected
  2,292, Craton 1,583. The Field trained only on years when the push rose; it can move ETAS by
  at most 4×.
- ETAS-aniso crashed on Groningen, where no quake is big enough to get an ellipse.
- `datetime.fromtimestamp` refuses moments before 1970 on Windows; the pressure model starts in 1957.

Decisions: D-097, D-098, D-099, D-100.

Next: P8f, the curriculum (planets → regions → central Italy), with the GPS features (D-085),
the catalogue-completeness fix (D-096), the global-network region (D-093) and a training
history that includes a cut in the push.

## 2026-10-30 — P8f — Fake planets first

TLDR: the curriculum, planets → the world's regions → central Italy, trained and scored on
three seeds. It fixes the one thing P8c broke, CB-20 +0.09 on every seed (now just above
ETAS-incomplete), and costs a little at home, CB-02 −0.03 and CB-03 −0.02 on every seed. So the
board keeps P6's Field. The four other fixes queued for P8f were each measured against a target
set before the run, and none met it. P8 ships as v0.9.0.

Built:
- `craton/field/planets.py`: the planet stage (six physics planets, seeds 1000-1005, each with a
  blind twin read through Helmstetter's short-term incompleteness), pumped planets for the
  forced Field
- `curriculum.start_from` and `planets` config sections, old hashes unchanged; `field_stage`;
  every Field can start its members from an earlier stage's
- configs `planets`, `earth`, `curriculum`, `regions_first`, `world_sumatra`, `pumped`; region
  `sumatra` (19,263 ComCat quakes, snapshot 560a85bfd6186c39)
- `scripts/ablations/curriculum.py`, `earth_on_world.py`, `eth_etas_referee.py`;
  `forcing_blind.py --config`; `gps_thirty_days.py --no-jumps`;
  `scripts/reference/eth_etas_invert.py`; tests 429 → 436

Scores (mean ± std over seeds 0, 1, 2, paired against `mini`):
- curriculum: CB-02 −6.847 ± 0.016 (mini −6.815, −0.03), CB-03 −8.248 ± 0.005 (−8.231, −0.02),
  CB-20 −6.940 ± 0.012 (−7.028, +0.09; ETAS-incomplete −6.955), CB-21 −8.239 ± 0.004 (−8.224, −0.02)
- regions first, no planets (seed 0): CB-02 −0.064, CB-20 +0.173, CB-03 −0.008
- pumped planets: CB-30 +0.695 ± 0.006 (board +0.707 ± 0.045), the pump's worth +0.00
- world + Sumatra (seed 0): CB-22 never-seen +0.360 (board's seed +0.361); Chile +0.46, Alaska
  +0.19, both still below ETAS-incomplete; planets first: +0.254
- GPS without jumps, CB-19: −6.261 / −5.919 / −6.137 (plain −6.226 mean)
- ETH's ETAS: CB-02 −7.467, CB-03 −9.929, below ours (−7.118, −9.396)
- the board: unchanged, every number P8e's

Surprises / problems:
- On held-out planet days the Field beats ETAS by 0.13-0.18 nats per quake, and the planets
  still hurt transfer: started from them, the earth Field loses 0.1 on CB-22.
- ETH's inversion converged and then crashed writing its output (a `None` parameter); the
  parameters were read back from its log. It also needs pandas below 3.
- Two seeds run at once overwrote each other's lines in one results file; seed 1 was rescored
  from its saved checkpoints.
- Two trainers at once leave about 2 GB of RAM free on this machine and swap; the GPU sits at
  ~40%. Batches are built on one CPU core.

Decisions: D-101, D-102, D-103, D-104, D-105.

Next: P9, the hidden state.
