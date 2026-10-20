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
