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
