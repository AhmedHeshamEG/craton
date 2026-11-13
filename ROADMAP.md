# Roadmap

Seventeen phases, P0 → P16. Each one ends with hard exit tests, a lesson in [`lessons/`](lessons/), a figure,
and a release. A phase isn't done if any exit test fails. Live numbers are in [`SCOREBOARD.md`](SCOREBOARD.md);
the day-by-day story is in [`LOGBOOK.md`](LOGBOOK.md).

```
CORE       ██████████████████████████████   8/8    P0–P7   foundation → beat ETAS → a model that knows when it doesn't know
DISCOVERY  ██████████████████████████████   3/3    P8–P10  other planets, hidden state, listening
FIELD 2    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/6    P11–P16 a transformer shaped for the Earth: viable test, heart + fair judge, places, what pushes them, the equation, ship
ALL        ███████████████████░░░░░░░░░░░   11/17
```

Two versions share one codebase: **Craton-mini** (a laptop RTX 2060 with 6 GB, central Italy; every phase runs
here first) and **Craton** regional → planet (same code, bigger config files, unlocked in P16).

P11 to P16 were redrawn on 2026-11-12, after P10: see "Why the plan changed after P10".

Each phase lists its **Goal**, **Why**, the **LLM analogy** it borrows, what it builds, its **Benchmarks**, what
is deliberately **not** in it, the exit tests (**Done when**), the numbers it is expected to reach (**Expect**)
and what its lesson teaches (**Learn**).

---

## CratonBench: the progress meter

The project's spine, like lm-eval-harness is for LLMs. Tasks arrive phase by phase. Once a task exists, **it
never gets removed and its test data never changes**; a better version gets a new ID.

**Ratchet rule.** A change lands only if no task gets worse beyond noise. Noise = 2σ over **3 seeds**. Every
scoreboard number is mean ± std over 3 seeds.

**Headline numbers** (top of `SCOREBOARD.md`):
1. **Geo-perplexity** (CB-01): how surprised the model is by real held-out Earth data. Lower = it understands
   more. The direct twin of LLM perplexity.
2. **Predictability meter** (CB-11): width of the 90%-coverage window. Smaller = closer to "exact time and
   place". Every improvement has to shrink it.
3. **Information gain over ETAS** (CB-02): does it beat the standard method on a real sequence.

| ID | Name | What it measures | Baseline to beat | Added in |
|---|---|---|---|---|
| CB-01 | **Geo-perplexity** | Negative log-likelihood (bits/token) on masked tokens from held-out regions + eras | Per-token climatology | P3 |
| CB-02 | **Apennines 2016–17 forecast** | Daily forecasts for Amatrice / Visso / Norcia / Campotosto, CSEP-style (pyCSEP N, S, M tests, log-likelihood, information gain per earthquake) at M≥3 and M≥1.5 | ETAS; the neural point process of arXiv 2301.09948 | P3 |
| CB-03 | **L'Aquila 2009 era holdout** | As CB-02, training only on data ≤ 2008, testing on 2009 | ETAS | P3 |
| CB-04 | **GPS gap-fill** | Hide whole GNSS stations, rebuild daily positions; RMSE (mm) | Inverse-distance weighting + linear trend | P3 |
| CB-05 | **Calibration** | Reliability diagram, ECE, PIT uniformity, CRPS | Uncalibrated model | P7 |
| CB-06 | **Unseen region** | Train without Calabria + Sicily, test there; AUROC of the "never seen this" detector | Same model, no OOD detector | P7 |
| CB-07 | **Fake-precursor attack** | False-alarm rate with injected fake GPS jumps / fake swarms | ≤ the clean false-alarm rate | P7 |
| CB-08 | **Physics sanity** | Outputs breaking the slip budget, Gutenberg–Richter, Omori decay, non-negative rates | 0 hard violations | P7 |
| CB-09 | **Two planets + sim-to-real gap** | ETAS planet: Craton must tie ETAS; physics planet: must beat it; synthetic vs real gap per generator | ETAS on each planet | P8 |
| CB-10 | **Planted discovery** | Tides removed on purpose: does the ghost recover the 12.42 h / 24 h / 14.8 d periods? | Random residual | later |
| CB-11 | **Predictability meter** | 90%-coverage window width in time (days), space (km), magnitude, for M≥4 | ETAS window width | P7 |
| CB-12 | **Interpreter faithfulness** | ~200 fixed questions: numeric agreement with tool outputs, citation coverage, hallucination rate, correct abstention | Plain LLM, no tools | later |
| CB-13 | **Microscope** | SAE reconstruction + recovery of a planted hidden variable + share of features with validated labels | Random directions | later |
| CB-14 | **Deep time** | Plate positions vs published GPlates reconstructions at past ages; angular error | Constant-velocity extrapolation | later |
| CB-15 | **Efficiency** | Model FLOPs utilisation of the pretraining run on the RTX 2060; tokens/s, peak VRAM, wall-clock alongside | Must fit in 6 GB (a hard gate) | P4 |
| CB-16 | **Scaling-law fit** | Loss vs params and tokens over 4–5 sizes; predict the next size before training it | — | P12 |
| CB-17 | **In-context adaptation** | Only the first N days of a new region in context, no retraining: does CB-01 improve as N grows? | Empty context | P5 |
| CB-18 | **Ladder rung 3** | Timing of the next event on repeating-earthquake patches | Periodic / Poisson | later |
| CB-19 | **30-day forecast** | As CB-02, one forecast per 30 days: the horizon where GPS strain might pay | ETAS | P8 |
| CB-20 | **Apennines, enhanced inputs** | CB-02's scoring, with the 2016–17 ML catalogue as input (relocated with hindsight, so its own task) | ETAS on the same inputs | P8 |
| CB-21 | **L'Aquila, enhanced inputs** | CB-03's scoring, with the 2009 ML catalogue as input | ETAS on the same inputs | P8 |
| CB-22 | **World sequences** | Daily forecasts on 14 held-out sequences: six in regions trained on (California, Japan, New Zealand), eight in regions never seen (Turkey, Chile, Alaska); gain per quake, the never-seen half is the headline | ETAS fitted on each box's own past | P8 |
| CB-23 | **Hidden-state truth** | SHmax inferred from quakes alone vs the World Stress Map 2025, 0.5° cells in seven regions (degrees; pressure gets its own ID when open data exists, D-108) | Regional-average stress; interpolation of the other cells | P9 |
| CB-24 | **Migration** | Where a sequence lights up next: the true next 0.1° cell in the method's top 3, given only the cells lit so far and the past; sixteen sequences (D-109) | Distance only; the busiest cells | P9 |
| CB-25 | **Mainshock days** | Skill on held-out world M6+ days: Molchan curve, false alarms per hit | ETAS | P15 |
| CB-26 | **Distilled law** | A symbolic formula's held-out gain over ETAS vs the network's | the network itself | P9 |
| CB-27 | **Predictability horizon** | Skill over ETAS vs lead time (1 day → 1 year): where it reaches zero, with error bars | — | P15 |
| CB-28 | **Velocity gap-fill** | Hide one station's dv/v series, rebuild it | Neighbour average | P10 |
| CB-29 | **EarthquakeNPP** | The outside benchmark (Stockman et al., TMLR 2026): their California data, splits, ETAS and metric, untouched; nats per quake | Their ETAS | P8 |
| CB-31 | **Groningen pressure** | A quake-only forecast map vs the NAM reservoir pressure drop over each held-out gas year 2017-2023 (0.05° cells over the reservoir); rank correlation. The model never reads the pressure | Last year's quakes | P9 |
| CB-32 | **Unseen-region forecast** | CB-02's scoring on whole regions left out of training, one at a time, no fine-tuning | ETAS fitted on the region | P9 |
| CB-33 | **Planted stress** | The physics planet's hidden rate-and-state state per cell vs what a forecast implies; rank correlation | Quake density | P9 |
| CB-34 | **Stress from mechanisms** | As CB-23, with focal mechanisms as input, graded only on World Stress Map records that are not from quakes (boreholes, fractures, overcoring) | Regional average; interpolation | P9 |
| CB-35 | **The world gate** | Bits per quake above ETAS fitted on each region's own past: the sharp sequences, and hundreds at the global network's hearing; paired per quake, with a sealed vault | ETAS refitted daily; the Field | P12 |
| CB-36 | **A new region from its first year** | A never-seen region's first year as context, frozen weights, forecasts for the rest | ETAS fitted on the same year | P13 |
| CB-37 | **The world law** | A short formula read out of Field 2, scored alone on the world gate | the network itself | P15 |

CB-02 and CB-03 also report, from P6 on: how-many / where / how-big bits, the score without the M5+ days, and
the oracle ceilings (the building).

---

# CORE

## P0 · Foundation → v0.1.0
**Goal:** an empty but bulletproof skeleton.

**Why:** the building's foundation. Everything after plugs into it without touching it.

**LLM analogy:** the training-infrastructure repo every lab builds before the first run (configs, logging, eval
hooks, CI).

**Build**
1. **Identity:** the name (D-001), `craton-earth` on PyPI / `import craton` (the bare `craton` name is a squatted
   0.01 placeholder), Apache-2.0, `CITATION.cff`.
2. **Project:** Python ≥ 3.11 with uv, pinned lockfile, PyTorch through two exclusive extras (`cu126` for the
   RTX 2060, `cpu` for CI and laptops without NVIDIA). Ruff, mypy (strict on the package), pytest.
3. **Records:** `LOGBOOK.md` and `DECISIONS.md`, both append-only (the check refuses edits to old lines);
   `SCOREBOARD.md` generated from `results/scoreboard.json`, never by hand (the check re-renders it and fails on
   any difference).
4. **Configs:** one YAML per size (`mini.yaml` real, `regional.yaml` and `planet.yaml` declared with
   `status: planned`), strict schema (unknown keys rejected), a parameter estimator (the 12·d² rule) that keeps
   each config honest about its size.
5. **Seeds + determinism:** one call seeds Python, NumPy and torch, sets the deterministic flags, and a test
   proves two runs match.
6. **Hardware probe:** `craton doctor` prints GPU, VRAM, compute capability, fp16 and bf16 support, and turns a
   bf16 request on Turing into fp16 + gradient scaling with a warning.
7. **The ratchet:** `craton ratchet` compares a run's results with the scoreboard and fails on any regression
   beyond 2σ, a vanished task, fewer than 3 seeds, or a task whose test data fingerprint changed. Passes on an
   empty scoreboard.
8. **One check:** `craton check` = lint + format + types + tests + scoreboard integrity + append-only guard +
   ratchet. CI runs the same command.
9. **The front door:** README (a hero figure made from real earthquake data, the
   idea in one screen, the roadmap tracker, quickstart), `ROADMAP.md`, `CONTRIBUTING.md` (the short version; the
   full adapter guide is P12), issue templates, social preview image.
10. `lessons/P0.md`.

**Benchmarks:** none yet. The ratchet machinery exists and passes on an empty scoreboard.

**Not in this phase:** the token spec (P1), any data adapter (P2), any model code.

**Done when**
- `craton check` passes locally and on CI
- fresh clone → `uv sync --extra cu126` → `craton doctor` sees the RTX 2060 with fp16 on
- the ratchet's own tests cover pass, regression, noise, missing task, too few seeds, changed fingerprint
- the release `v0.1.0` exists
- `lessons/P0.md` written

**Learn:** what a repo, environment, lockfile, config, unit test and CI are; why seeds matter; what the ratchet
does and why it protects the foundation.

## P1 · The universal token (the frozen foundation) → v0.2.0
**Goal:** one data format for every measurement on Earth, plus a validator.

**Why:** the "Attention Is All You Need" move for geology: unification through one simple format.

**LLM analogy:** the tokenizer + vocabulary. Text → token IDs there; any measurement → a typed token here.

**Build**
1. `craton/spec/` with **Token Spec v1.0.0**:
   - `kind`: observation | forcing | property | event | derived
   - `instrument`: from a versioned registry (seismic_catalog, gnss_daily, solid_earth_tide, …)
   - `lat`, `lon` (WGS84 degrees), `depth_km` (positive down from the WGS84 ellipsoid, D-012), `time` (float
     seconds since 1970 UTC, proleptic Gregorian, D-011; optional ISO 8601 copy)
   - `value` (vector, declared units and channel names per instrument)
   - `uncertainty` (same shape as value, 1σ, required, never zero)
   - `provenance`: source, dataset version, license, adapter version, content hash, `synthetic` flag
   - `resolution`: spatial (km) and temporal (s) support
2. A strict validator (pydantic): units, ranges, monotonic time, coordinate sanity.
3. Storage: Parquet partitioned by instrument and year, every file fingerprinted.
4. `SPEC_POLICY.md`: semantic versioning; minor versions **only add** optional fields or registry entries;
   nothing removed or redefined; a major version needs migration code, a DECISIONS entry and every benchmark
   re-passing.
5. Examples for every `kind`, forces and properties both (granularity, porosity, permeability, lithology,
   temperature, friction, …), and the monk's diary from 1349 as a token.
6. Figure: one token, drawn as an exploded diagram of its slots.

**Benchmarks:** the validator suite (good tokens pass, 50+ malformed cases fail).

**Not in this phase:** real downloads (P2).

**Done when**
- Spec v1.0.0 frozen and documented with an example for every `kind`
- the validator rejects every malformed case
- forces and properties both representable, with examples
- `lessons/P1.md` written

**Learn:** tokenization in LLMs and why a unified format is the key move; units and uncertainty; semantic
versioning; why add-only keeps the foundation intact.

## P2 · First adapters + immutable data store → v0.3.0
**Goal:** real data flowing into valid tokens.

**Why:** the platform's first contributors, and the pattern every future adapter copies.

**LLM analogy:** the data pipeline (crawl → clean → deduplicate → filter → shard) before pretraining.

**Build**
1. Adapter interface: `fetch()` → raw files cached; `to_tokens()` → validated tokens; `card()` → data card
   (source, license, coverage, known issues). Registered by name.
2. Adapters:
   1. **INGV earthquake catalog** through the FDSN event service (ObsPy client or plain HTTP). Central Italy
      box, 1985 → today, all magnitudes. Keep magnitude type and catalog version.
   2. **GNSS daily positions** from the Nevada Geodetic Laboratory for stations in the box.
   3. **Solid-Earth tides** computed at each station / event (PySolid or a verified equivalent), `kind=forcing`.
3. Raw downloads cached with hashes; every token file traceable to its raw file.
4. Coverage report: map + timeline, generated. This is the phase's figure.
5. Each adapter tested against a tiny offline fixture.
6. Respect every provider: rate limits, caching, licenses and citations in the cards.

**Benchmarks:** data health (counts, duplicates, gaps, coverage) on the scoreboard.

**Not in this phase:** splits and baselines (P3).

**Done when**
- all three adapters produce tokens that pass the validator
- re-running gives byte-identical output (same hashes)
- data cards written, licenses and citations recorded
- `lessons/P2.md` written

**Learn:** earthquake catalogs, GNSS time series, solid-Earth tides; magnitude of completeness; data licenses;
reproducibility by hashing.

## P3 · Benchmarks and baselines (before any model) → v0.4.0
**Goal:** CratonBench v1, with the simple methods scored, so the first model has something real to beat.

**Why:** without a fixed test you can't tell progress from luck. LLM labs build evals first.

**LLM analogy:** the eval harness, held-out sets, contamination checks.

**Build**
1. `craton/bench/` with **CB-01 … CB-04**, each task registered with its fingerprint.
2. Frozen splits written to disk with hashes:
   - test era A: 2016-08-24 → 2017-12-31 (Central Apennines)
   - test era B: 2009 (L'Aquila), training ≤ 2008 for CB-03
   - held-out GNSS stations for CB-04
3. **Leak checks** in `craton check`: no test-period or test-station token in any training shard.
4. Baselines: ETAS (a verified open implementation, else a small clean one with tests), climatology / Poisson,
   IDW + trend.
5. pyCSEP for the forecast tests.
6. Figure: the Apennines sequence, observed vs ETAS daily rate.

**Benchmarks:** CB-01 (climatology), CB-02, CB-03, CB-04, all scored.

**Not in this phase:** any neural model (P4).

**Done when**
- baselines on the scoreboard
- leak checks pass
- the ratchet uses them as the floor
- `lessons/P3.md` written

**Learn:** train / validation / test splits and leakage; log-likelihood and information gain; ETAS (Omori decay
+ Gutenberg–Richter + triggering); CSEP-style testing.

## P4 · Craton-mini v0 (catalog only) → v0.5.0
**Goal:** the first transformer, trained on the earthquake catalog alone.

**Why:** the smallest version that runs the whole loop: tokens → model → benchmarks.

**LLM analogy:** GPT-2 small, the first pretraining run; masked modelling as in BERT / MAE.

**Build**
1. Encoder for `event` tokens (continuous values → small MLP → d-dim vector), registered by instrument.
2. **4D positional encoding:** RoPE generalised to (lat, lon, depth, time), sinusoidal fallback, tested.
3. Trunk: d = 192–256, 4–6 layers, pre-norm, PyTorch SDPA attention.
4. Objectives, chosen by config: masked token prediction; autoregressive next-event prediction (point-process
   output: rate + magnitude distribution).
5. Probabilistic heads only (mixture density / binned categorical). Never a single number.
6. Training on the 2060: fp16 autocast + GradScaler, gradient accumulation, checkpoints, early stopping, peak
   VRAM recorded.
7. A model card per checkpoint.
8. Figure: the loss curve and the attention of one head over a real aftershock sequence.

**Benchmarks:** CB-01, CB-02, CB-03, **CB-15 (efficiency, new)**.

**Not in this phase:** GPS and tides as inputs (P5).

**Done when**
- trains end-to-end on the laptop inside 6 GB
- fully scored, even if it loses to ETAS
- at least matches climatology on CB-01
- `lessons/P4.md` written

**Learn:** embeddings, attention, layers, parameters (12·d²), loss, optimiser, learning-rate schedules, fp16,
overfitting; why the output is a distribution.

## P5 · Fusion: catalog + GPS + tides → v0.6.0
**Goal:** one model reading several kinds of data at once.

**Why:** where "remove blocks, replace them with geology blocks" becomes real, and where the Moon enters.

**LLM analogy:** multimodal models (one encoder per modality into one space); mixture of experts; in-context
learning.

**Build**
1. Per-instrument encoders into the shared space, registered by name.
2. Optional modality experts (MoE-lite) behind a config flag.
3. Context builder: region × time window → token sequence under the budget (~1024), with a documented
   selection policy (recency, distance, magnitude).
4. Cross-modal masking (hide GPS, predict it from quakes + tides, and back).
5. Ablation runner: with / without each modality, scored automatically. The ablation table is the figure.

**Benchmarks:** CB-01 … CB-04, **CB-17 (in-context adaptation, new)**.

**Not in this phase:** calibration and the locks (P7).

**Done when**
- beats P4 on CB-01 and CB-04 beyond noise
- the ablation table shows each modality's share, tides included
- `lessons/P5.md` written

**Learn:** multimodal fusion, context windows, in-context learning, ablations.

**Where it landed (v0.6.0):** CB-01 21.66 → 15.12 (GNSS days 9.75 bits, quakes tied at 20.48), CB-04
17.50 mm against IDW's 17.77, CB-17 new. Step 2 (modality experts) was left for a bigger model: with
three instruments and a 4.7M trunk there is nothing to route yet (D-048). Cross-modal masking is built
and switched off for the general models after it measurably cost the quakes (D-056).

## P6 · Beat ETAS: the Field → v0.7.0
**Goal:** Craton above ETAS on CB-02 and CB-03, beyond noise.

**Why:** P4 and P5 lost to ETAS by four to five bits per quake. Split, the loss was mostly *where*, and
the cause was structural: the 1,024-quake window reached back less than three days inside a sequence,
and one vector had to hold a map of 1,050 cells. Every later phase builds on a forecaster at least as
good as the 1988 standard.

**LLM analogy:** state-space layers (long memory at constant cost), attention over a grid (vision
transformers), residual learning on a strong base, deep ensembles.

**Build**
1. The bit split: CB-02 and CB-03 report how-many / where / how-big bits, and the score without the
   M5+ days. Extras only; the frozen test data doesn't move.
2. The building: oracle floors above ETAS (the true count, the true map to ~10 km and to the cell, the
   roof), measured once and drawn on the scoreboard.
3. Stronger floors: same-station interpolation for CB-01's GNSS half; ETAS with stretched aftershock
   zones; ETAS with short-term incompleteness (blind time).
4. The Field: Omori memory (fading counters per cell, 1 hour → 12 years), the forecast cells as tokens
   with attention between them, ETAS as the starting line (a bounded learned correction, zero at
   the start), small quakes in, M3+ scored.
5. Figure: the building, with Craton P5 in the basement and the Field where it lands.

**Benchmarks:** CB-02 and CB-03 with the new extras; every earlier task through the ratchet, and no
number or CB-01 half below P5.

**Not in this phase:** calibration and the locks (P7), other regions (P8).

**Done when**
- CB-02 and CB-03 above ETAS beyond noise (3 seeds)
- Craton's *where* bits ≥ ETAS's on both
- above the stronger floors too, or the gap documented
- the building figure on the README
- `lessons/P6.md` written

**Learn:** point processes and ETAS from the inside, Omori as a sum of exponentials, state-space layers,
residual learning, reading a loss split.

**Where it landed (v0.7.0):** CB-02 −10.98 → **−6.82** (ETAS −7.12, ETAS-incomplete −7.06), CB-03
−14.05 → **−8.23** (ETAS −9.40): above ETAS on both, beyond noise. Almost all of it is how many. *Where*
beats ETAS on CB-03 (+0.15) and misses on CB-02 by 0.02, all of it on the Norcia and Campotosto days: the
one exit test not met, left to the mainshock days work. Stretched aftershock zones add nothing; ETAS with
a blind time picks a blind time of zero (D-060). The trunk's P5 place context was not needed (D-061).

## P7 · The 8 locks (trustworthiness) → v0.8.0
**Goal:** the model knows when it doesn't know.

**Why:** a confidently wrong earthquake model is worse than none.

**LLM analogy:** calibration (temperature scaling), deep ensembles, OOD detection, red-teaming, refusal when
unsure.

**Build**
1. Calibration: proper scoring rules in training, post-hoc calibration, reliability plots.
2. Ensembles: 5+ members; disagreement reported as epistemic uncertainty.
3. OOD detector: distance of the context from training data in latent space; past a threshold → mandatory
   abstention.
4. Physics budgets: slip budget, Gutenberg–Richter, Omori, non-negativity; violations flagged and counted.
5. A multi-simulator hook, ready for P8.
6. Region holdout (Calabria + Sicily) frozen.
7. Fake-precursor injector: synthetic GPS jumps, fake swarms, sensor glitches.
8. **Locked forecasts:** written with timestamp + hash **before** the evaluation period is read, graded after,
   CSEP-compatible so a real prospective submission is possible later.
9. Figure: the predictability meter, Craton vs ETAS, as one bar each.

**Benchmarks:** **CB-05, CB-06, CB-07, CB-08, CB-11 (new)**.

**Not in this phase:** synthetic pretraining (P8).

**Done when**
- ECE below a documented threshold; PIT roughly uniform
- OOD AUROC reported; abstains more on the held-out region than in-region
- fake-precursor false alarms ≤ clean false alarms
- zero hard physics violations
- the predictability meter on the scoreboard
- `lessons/P7.md` written

**Learn:** aleatoric vs epistemic uncertainty, calibration, ensembles, OOD, prospective testing; why an honest
"I don't know" is a feature.

**Shipped (v0.8.0, 2026-10-22).** CB-11: Craton watches 0.84% of space-time to catch 90% of the M4+ quakes,
ETAS 1.32%. CB-08: zero violations (every ETAS floor breaks the slip budget). CB-06: AUROC 0.997, abstains on
all of Calabria and Sicily. CB-05: ECE 0.012 and the best-calibrated day totals (PIT distance 0.039), but cell
by cell 0.32 doublings off, behind ETAS-incomplete's 0.10. **Missed:** CB-07 is 1.002, 0.2% over the clean
false-alarm rate (a magnitude shift the guard caught only in part on one seed). Added beyond the plan: `craton
referee` (it rejects the project's own tidal claim of 2026-10-21) and locked forecasts published every night
in [`forecasts/`](forecasts/). Step by step: [`lessons/P7.md`](lessons/P7.md).

---

## Why the plan grew after P7

P6 split the old four-bit loss to ETAS before fixing it: almost all of it was *where*, and the cause was the
model's short memory and blurry map. Measured the same day:

- **How tall the building is.** Above ETAS, knowing each day's count is worth +2.1 bits per quake (CB-02) and
  +3.55 (CB-03), and 70–78% of that sits on the three days with an M5+. Knowing where to the cell is worth
  +2.1 / +2.45; to ~10 km, only +0.6.
- **Where the literature agrees.** Start from ETAS, feed the small quakes in, think in a grid field. Nobody has
  a neural win on *where* yet.
- **A false discovery, caught.** Tidal phase-locking of aftershocks looked like p = 10⁻²¹; a 24-hour clock locks
  just as hard. That is why the referee exists.

So after the locks come the AlphaFold-shaped moves: many more planets to learn from (P8), the crust's
hidden state graded against data it never saw (P9), instruments that report every day (P10), and a detector
for the mainshock days, where most of the headroom is (P11). The ghost, the translator, the microscope and
scale follow, renumbered.

---

# DISCOVERY

## P8 · Other planets: real, human-made and fake → v0.9.0
**Goal:** many more sequences to learn from, and planets where the answer is known.

**Why:** central Italy has three big sequences in 40 years; a correction learned on three is a horoscope.

**LLM analogy:** more pretraining data, curriculum, synthetic data with verifiable answers, transfer.

**Build**
1. **ML catalogues** as a new instrument source: Tan et al. 2016–17 (~900k events), L'Aquila 2009
   (PhaseNet + GaMMA), Altotiberina 2010–23 (~420k). They were relocated with hindsight, so they feed **new
   tasks only** (CB-20, CB-21); CB-02 and CB-03 stay pure.
2. **The whole Earth, one model (P8d; D-086).** Every region is a YAML file. One Field trained on central
   Italy, California (ComCat), Japan (JMA) and New Zealand (GeoNet) together, in each region's own
   magnitudes. CB-22 scores it on held-out sequences there **and zero-shot** in regions it never read
   (Turkey via EMSC, Chile and Alaska via ComCat). **CB-29** scores it on EarthquakeNPP, an outside
   benchmark where five neural point processes lost to ETAS. Stop rule, set before any result: lose CB-29
   or CB-22's never-seen half, and the phases after P8 are redone. It passed on both (D-093), level with
   ETAS-incomplete on the never-seen half, so later phases claim never-seen regions against ETAS-incomplete.
3. **Human-made planets with a measured cause:** Groningen (gas production and reservoir pressure) and
   Oklahoma (wastewater injection volumes). Learn "pressure → quakes" where pressure is known.
4. **Fake planets, as a two-planet test:** an ETAS planet (Craton must tie ETAS: it can't beat the truth) and
   a physics planet (`SPIKE`: QDYN or a quasi-dynamic rate-and-state code on Windows / WSL; fallback a
   verified spring-block model in NumPy) with real fault geometry, stress transfer and a fluid source (Craton
   must beat ETAS). Then fake data pretrains the correction only.
5. **CB-19, a 30-day forecast:** the horizon where GPS strain has the best chance to pay (D-083).
6. Curriculum: planets → regions → central Italy fine-tune.
7. Figure: one fake planet's 10,000 years next to the real Apennines; and the world map of sequences Craton
   learned from.

**Benchmarks:** **CB-09 (two-planet test + sim-to-real gap, new), CB-19, CB-20, CB-21, CB-22, CB-29,
CB-30 (new)** + everything before.

**Built in six parts** (D-078), each on `main` when its tests pass; `v0.9.0` is tagged after the last:
P8a the two planets and CB-09 · P8b CB-19 · P8d the whole Earth, CB-22 and CB-29 · P8c the ML catalogues,
CB-20 and CB-21 · P8e Groningen and Oklahoma · P8f the curriculum (in that order since the 2026-10-25
world replan, D-086).

**P8a, done (2026-10-25):** the physics planet is Dieterich's rate-and-state seismicity on an
Apennine-shaped fault network (the QDYN spike failed, D-079). ETAS planet: Craton −0.02 bits from
fitted ETAS (tie ✓). Physics planet: **+0.19** over ETAS on all three planets (beaten beyond noise ✓,
short of the +1 hoped for). Sim-to-real gap: ETAS planets 1.32 doublings off, physics planets 0.88.

**P8b, done (2026-10-25):** CB-19, sixteen 30-day forecasts of the 2016-17 sequence (D-083, D-084).
ETAS wins: Craton −0.22 bits per quake (expected ≈ +0.2), level without the months that hold an M5+.
GPS adds +0.13 on every seed, all in the count, most likely from coseismic jumps and afterslip; it joins
the Fields in P8f (D-085).

**P8c, done (2026-10-29):** CB-20 and CB-21 rerun CB-02's and CB-03's days with INGV plus every quake
a machine-learning catalogue adds (Tan et al. 2021; Fonzetti et al. 2025), magnitudes mapped onto INGV's
scale (D-094, D-095). The catalogue helps ETAS (+0.13 on 2016). Craton: CB-20 −7.028, **below** ETAS on
the same input (−6.985, expected ≈ +1.1); CB-21 −8.224, +1.14 over ETAS (expected ≈ +1.8). The Field had
learned INGV's gaps as a constant and counts them twice once they're filled (D-096); the fix is P8f's.

**P8e, done (2026-10-29):** CB-30, two places where the push is measured: Groningen's gas field
(the reservoir's pressure) and Oklahoma's wastewater (Arbuckle disposal, well by well). Thirty-day
forecasts from 2017, when both pushes were turned down. Craton **+0.71** bits per quake over ETAS
and +0.25 over a physics floor whose background follows the pump, on every seed. The same Field
without the pump scores +0.67: the gain is from reading the quakes, and the pump only steadies it
(D-099).

**P8f, done (2026-10-30), P8 shipped as v0.9.0:** the curriculum, planets (with blind twins) →
the world's regions → central Italy (D-102). CB-20 **+0.09** over P6's Field on every seed, now just
above ETAS-incomplete; CB-02 −0.03 and CB-03 −0.02, on every seed. So the board keeps P6's Field.
Pumped planets leave the pump worth nothing on CB-30 (D-103); a global-network region (Sumatra)
leaves CB-22 unchanged (D-104); GPS without its jumps loses its consistency and stays out (D-105);
ETH's ETAS scores below ours on CB-02 and CB-03, so ours stays the floor (D-101).

**Not in this phase:** the hidden-state layer (P9).

**Done when**
- ETAS planet tied within noise; physics planet beaten beyond noise
- CB-02 / CB-03 improve on P6 beyond noise from the new data, or the negative result is documented
- CB-22: Craton above ETAS on most held-out world sequences, and on most never-seen regions (zero-shot)
- CB-29: EarthquakeNPP's ETAS reproduced, and Craton's score against it published either way
- model size raised only if the scaling evidence supports it
- `lessons/P8.md` written

**Expect:** CB-02 ≈ +0.6 over ETAS · CB-03 ≈ +1.4 · CB-20 ≈ +1.1 · CB-21 ≈ +1.8 · CB-19 ≈ +0.2 ·
CB-22 above ETAS on ~2 sequences in 3 (seen regions), ~1 region in 2 zero-shot · CB-29 above their ETAS by
≈ +0.1 to +0.3 · CB-09 ETAS planet |Δ| < 0.05, physics planet ≈ +1.

**Learn:** simulators, rate-and-state friction, induced seismicity, catalogue completeness, transfer.

## P9 · The hidden state: stress, pressure and the wiring → v0.10.0
**Goal:** Craton infers what nobody measures (stress, fluid pressure, closeness to failure per patch) and
which places talk to each other, and the inference is graded against data it never saw.

**Why:** the AlphaFold move. The forecast becomes the exam that proves the hidden structure.

**LLM analogy:** latent-variable models; AlphaFold's pair representation; probing.

**Build**
1. **A hidden-state layer:** per cell, a stress-and-pressure vector that turns into rates through a
   rate-and-state-shaped law (seismicity as a stress meter, Dieterich 1994); trained end to end.
2. **The pair representation:** which cells excite which, learned from 40 years of co-activation; one
   connection graph per region.
3. **Graded against independent truth:** stress orientation vs the World Stress Map 2025 (CB-23, D-107; the
   2016–17 ML focal mechanisms planned here are not public); pressure vs an open map when one exists (Vp/Vs
   tomography and the Apennine CO₂ map are published as figures only, D-108).
4. **The migration test:** the graph learned on data before 2016 vs the order the 2016–17 sequence lit up
   (Amatrice → Visso → Norcia → Campotosto).
5. A Coulomb rate-and-state (CRS) baseline on CB-02 (Mancini et al. 2019 beat ETAS after Visso).
6. Figure: the inferred stress (and pressure, if graded) next to the truth it was never shown.

**Benchmarks:** **CB-23 (hidden-state truth), CB-24 (migration), CB-31 (Groningen pressure), CB-32 (unseen region), CB-33 (planted stress), CB-34 (stress from mechanisms), new**; CB-26 starts here; CRS joins the floors.

**Done when**
- CB-23 beats a regional-average-stress baseline; CB-24 beats distance-only
- CB-31, CB-32, CB-33, CB-34 and CB-26 each reported against their floors, or the negative result
- CB-02 / CB-03 improve on P8 or the negative result is documented
- the referee passes every map shown in public
- `lessons/P9.md` written

**Expect:** CB-23 stress axes within ~20° of the World Stress Map · CB-24 the true next patch in the graph's top 3 ·
CB-02 ≈ +0.9 · CB-03 ≈ +1.6.

**Learn:** stress, pore pressure, Coulomb failure, rate-and-state as a stress meter, inverse problems.

Built in parts (D-106), the tests first: **P9a** CB-23 and its floors · **P9b** CB-24 and a CRS floor ·
**P9c** the hidden-state layer · **P9d** the wiring and the camera. Four blocks on the catalogue alone lost,
and the Field's CB-24 gain stops at Italy's border, so the rest of P9 aims at three tests of understanding,
each graded on a truth the model never sees (D-116):
- **P9e** ✓ the leak audit (cut moved back; a time-shuffled control must lose its gain) and the first
  formula: symbolic regression on today's Field (CB-26, moved here from P14). Done 2026-11-05: the
  earlier cut keeps nearly all the gain (+0.28 vs +0.30), shuffled days lose it (−1.43); formulas of
  1-4 terms keep 0.84 of the gain on held-out time, one constant factor 0.61 (D-119 … D-121)
- **P9f** ✓ the stress meter: forecasts only through Dieterich's rate-and-state law, so the hidden layer
  is a stress history; graded on the physics planet's planted stress (CB-33). Done 2026-11-05: CB-33 on
  the board (mini 0.45, the quake-density floor 0.39, ETAS 0.35); the meter loses everywhere (CB-02
  −0.29, CB-03 −0.74, CB-33 0.41) and is not kept; its shared stress alone sharpens *where* a little
  (D-122 … D-124)
- **P9g** ✓ the pressure truth (CB-31) and the local-alpha floor. Done 2026-11-05: every quake-only map,
  the real quakes included (−0.21), ranks Groningen's cells against the year's pressure drop (mini
  −0.42) and with the total drop since 1957 (+0.45): the quakes carry years of loading, not this year's.
  ETAS refit daily on the last month (Lombardi & Marzocchi 2010) ties mini on CB-02 (+0.30 both) and
  gets half of its CB-03 gain (+0.59 vs +1.16) (D-125 … D-128)
- **P9h** ✓ regions it never read (CB-32) and inputs without units. Done 2026-11-06: the world Field
  trained without each region beats ETAS fitted there in all four, on every sequence and seed (+0.34,
  best floor +0.07). Dimensionless inputs help central Italy and California, hurt Japan and New
  Zealand, lose on the mean (+0.30) and are not kept (D-129 … D-131)
- **P9i** ✓ direction from focal mechanisms (CB-34), the referee on every public map, v0.10.0. Done
  2026-11-06: Global CMT moment tensors as tokens (Token Spec 1.3.0); the tensor field (masked
  modelling of mechanisms) 31.3° off stress no quake measured, the textbook Kostrov sum 33.4°, copying
  neighbouring boreholes 27.9°. The map referee passes CB-23 and CB-34, CB-33 on two planets of three,
  and fails CB-31 as stated. CB-02's gain over ETAS is the Norcia day; the ×4 cap binds on L'Aquila
  (D-132 … D-136)

## P10 · Listening: continuous noise as instruments → v0.11.0
**Goal:** two instruments that report every day, quake or no quake: seismic noise power and seismic velocity
change (dv/v from ambient-noise correlation).

**Why:** quakes are rare and noise isn't. In the lab, a fault's continuous noise predicted time to failure;
in Cascadia, seismic power rose exponentially weeks to months before slow slip (Los Alamos, 2017–2020).

**LLM analogy:** a new modality with its own encoder (like adding audio to a text model).

**Build**
1. `SPIKE`: a handful of INGV broadband stations on the Apennine faults through EIDA FDSN; daily noise
   correlations (MSNoise or own code) fit the laptop's disk and time.
2. Adapters: `seismic_noise_power`, `seismic_velocity_change`, with seasonal and ocean-noise corrections
   (ocean-noise direction can fake dv/v; Hadziioannou 2011).
3. Their encoders join the fusion; ablation as in P5.
4. Figure: dv/v under the 2016 sequence, day by day.

**Benchmarks:** **CB-28 (velocity gap-fill: hide a station's dv/v, rebuild it), new**; their ablation share
on CB-19 and CB-25.

**Done when**
- both instruments flow through the spec with data cards
- their ablation share is measured (zero is a result)
- `lessons/P10.md` written

**Expect:** CB-28 beats a neighbour average · ablation share ≈ 0 next-day, positive at 30 days.

**Learn:** ambient-noise interferometry, why the Earth hums, new modalities.

Built in parts (D-137), the instruments before the model: **P10a** continuous waveforms from any FDSN
data centre and daily noise power · **P10b** dv/v from noise correlations, CB-28 and its neighbour floor ·
**P10c** both instruments in the Field, the ablation · **P10d** a borrowed ear: frozen SeisLM embeddings
against our band powers · **P10e** the figure, the lesson, v0.11.0. The ear listens at 1 Hz: a 100 Hz
year of sixteen stations would be ~300 GB.

*P10b done (2026-11-08, D-140 … D-142):* dv/v per station pair is on the spec (1.5.0), clean between
quakes (0.04% network scatter) but blind to the 2016 shocks at 1 Hz and at 20 Hz (Norcia reads the
wrong way). CB-28 is frozen and its best floor is "nothing changed" (0.347%); the neighbour average
loses (0.378%), so P10c's model has to beat zero, not the neighbours.

*P10 closed (2026-11-12, v0.11.0):* listening in the Field (P10c) and a borrowed ear (P10d) were not
adopted: next day both are worth nothing, a month ahead the hum's loudness helps the count (+0.17 bits
per quake on every seed) but not the whole forecast. The figure: `docs/assets/velocity-*.png`.

## Why the plan changed after P10

P8, P9 and P10 tested about fifteen ideas on top of the Field: a hidden state, attention wired like
aftershocks, a camera for missing quakes, a stress meter, dimensionless inputs, GPS, the crust's hum, a
borrowed waveform model, a curriculum of fake planets. Every one lost, by 0.02 to 0.06 bits per quake. The
record says why:

- **One day was the judge.** All of them were graded on CB-02, and CB-02's whole lead over ETAS is the Norcia
  day (D-136). An ETAS refitted every day ties it (+0.297 against +0.303, D-127). A test that narrow cannot
  see a small gain.
- **The new instruments were starved.** About 450 of CB-02's 9,367 training days had any of them.
- **What works was never the yardstick.** Trained without a region, the model still beats ETAS fitted there:
  +0.34 against the best simple method's +0.07, on 8 of 8 sequences and every seed (CB-32). It also beats
  ETAS on EarthquakeNPP (CB-29), where five neural models lose to it.
- **The transformer was never shaped for the Earth.** The Field is a stock two-layer transformer reading 54
  counters written by hand. It never reads an earthquake.

So P11 to P16 are a new model and a new judge (D-148 to D-151):

- **Field 2**, a transformer shaped for the Earth. Places are the words: each patch of crust carries a hidden
  state of many numbers, and earthquakes are messages that write into the places around them. No place knows
  its address, so what is learned in Japan applies in Chile. Every quake brings its own ruler (distance in
  rupture lengths) and time runs on a log clock. Its attention can count, and one counting head with fixed
  settings is ETAS, so the model is born equal to ETAS and learns only what ETAS misses.
- **Other instruments are homework, never inputs.** The model has to predict GPS motion and how quakes slip
  from the same hidden state. Feeding them in lost six times.
- **A world gate.** Ideas are adopted on hundreds of sequences across the planet, with a sealed vault nobody
  reads until adoption. The board shows the best score ever reached per task.
- **Run it for real first.** P11 is the smallest real Field 2, trained and scored, before anything big.
- **One equation.** No addresses and shared weights make the model one rule applied at every place and time.
  The plan ends by reading that rule out and testing the formula alone.

Moved to Later: the ghost channel and the atlas, the translator as its own phase, the sparse-autoencoder
microscope (readable curves replace it), the million-year router. The storm detector became a measurement
taken check-first, inside P15.

---

# FIELD 2

## P11 · The viable test → v0.12.0 (P10 closed as v0.11.0)
**Redrawn on 2026-11-13, after its first run.**

**What the first run said (kept: it is a result).** The heart as first built, seed 0, on CB-32: untrained,
0.000 on all eight sequences and 7.256 on CB-29 (born as ETAS, exactly); trained with the region's ETAS
numbers as context, **−0.25**; trained without them, **+0.007**. The Field has +0.34. The stop rule fired.
Read from the code and the numbers, four causes:

1. ETAS's miss on the test sequences is all in *how many* (Italy 2016: 583 expected, 1,065 came; L'Aquila
   70 and 315; Tohoku 1,129 and 3,478). *Where* never moved. The heart gave each quake a fixed weight by age,
   size and distance and added them up: its query was empty, so it could not say "this place is running hot".
2. Training hid the miss. Each tile's ETAS was fitted on the very days the heart learned from, so its totals
   were already right there.
3. The context was learned from three regions and extrapolated on the fourth (it cut Italy's count to 230).
4. It heard nothing below completeness, places never saw each other, 6,018 parameters, a learning rate the
   world Field had already found too hot, one member, early stopping on the regions it trained on, and hard
   time bins.

**Goal:** the smallest real Field 2 that fixes those, trained and scored for real.

**Why:** places have to see each other (quakes infect their neighbours); the model should be ETAS's equation
made bigger, with terms for what ETAS cannot see; hard time bins are a bottleneck; and the dimensional
analysis belongs early. A memory of ETAS's own mistakes fed in by hand is out: it would be the answer
smuggled in as an input. The state is learned.

**LLM analogy:** attention needs a query with content; a recurrent state; training on the distribution you
will be tested on.

**Steps**
1. **Record the first run** (`results/ablation/field2_heart.json`); it opens the lesson.
2. **The check before building** (`craton screen`, minutes, no training): ETAS's miss on the pool, held the
   way the test holds it, and how much of it each candidate piece of information explains in a region left
   out. The running came ÷ expected ratio appears only as a yardstick of what there is to win, never as an
   input. The expectation for step 6 is written down *before* training, from this check.
3. **The heart, rebuilt**, each part behind its own switch:
   - **a query with content**: every place carries a learned state, written by the quakes that reach it
     and fading on the log clock; a message's weight depends on its bin *and* on the state of the place
     receiving it; trained by the forecast loss only;
   - **neighbours**: places read each other's state for a round or two, by relative geometry only;
   - **rulers, not context**: time over the region's Omori c, distance over rupture length, rate over the
     background, size above completeness; the region's ETAS numbers scale the inputs and are never one;
   - **small quakes**: it hears from mc − 0.5;
   - **a smooth clock**: weights blended between neighbouring ages;
   - **a plus beside the times**: an additive head next to the reweighted ETAS one;
   - **how excited ETAS is** (its rate over the background) as a state input: a switch, off by default.
4. **Born as ETAS** stays a test in the suite (within 0.01 bits).
5. **Train as tested**: ETAS fitted on one period, the heart trained on the period after it, over several
   origins per tile; learning rate 3e-4, four members, each stopping on a region it does not learn from.
6. **A lean pass first**: one member, seed 0, on CB-32 and CB-29. Only if it is above ETAS do the three
   seeds, the four members and the switch-off runs follow (CB-32 every seed; the Field is a four-member
   ensemble, so a fair comparison needs four).
7. **The three answers**, the film ("How Field 2 works", from the run's data), the honest verdict.

**Benchmarks:** CB-32, CB-29 (existing). No new task.

**Not in this phase:** more regions, the hourly test, rupture shapes, hide-and-rebuild, any other instrument.

**Done when**
- the check's verdicts and the written expectation exist before the training run
- the born-as-ETAS test passes
- the three answers are written with their numbers, the first run's beside them
- the film renders from the run's data
- `lessons/P11.md` written

**Stop rule:** if the rebuilt heart is not above ETAS on every seed of CB-32, the plan stops before P12.

**Expect:** written after the check (DECISIONS, D-153). The first guess (+0.15 to +0.30) was wrong.

**Learn:** attention as a sum, a query with content, why a model must be trained on the errors it will meet,
dimensional analysis as a way to remove the address.

## P12 · The heart and the fair judge → v0.13.0
**Goal:** a judge that can see a small gain, the growth curve, more of the world, and the full heart graded
on it against the Field.

**Why:** one Italian day graded fifteen ideas. And nobody knows yet whether more data is the lever.

**LLM analogy:** building the eval before the model (lm-eval-harness); scaling laws (Kaplan, Chinchilla);
data augmentation.

**Steps**
1. **The world gate, CB-35**: tier one, the sharp sequences (CB-22's and CB-32's); tier two, hundreds of
   sequences at the global network's hearing (ComCat, M6+ mainshocks after the training cut). Scored per
   quake, paired, bootstrap error bars. An idea is adopted only if it wins on both tiers.
2. **The vault**: a share of tier two sealed by hash. Nobody reads its sequences, us included; only
   the score comes out, at adoption time.
3. **The roof on the world**: the oracles of `craton bench ceiling` on the gate.
4. **The small judge** (~15 min, a fixed subset); `craton screen` (from P11) rerun on every past loser.
   From here on, every training run decides when to stop on a region it did not train on.
5. **One command for a stranger**: `craton bench gate MODEL` grades anyone's forecaster.
6. **The hourly test** (a new task, numbered when frozen): forecasts refreshed every hour or after every
   large quake, ETAS given the same refresh. A midnight test cannot score the quakes of a mainshock's own day
   for anyone.
7. **Ask from any moment**: train on the next 24 hours from any hour, not only midnight.
8. **More of the world**: more regions in the pool (Japan is half of it today), mixed through the rulers.
9. **ETAS wrong on purpose**: its numbers jittered in training, so the model corrects from evidence.
10. **The dimensional analysis in full**: the dimensionless groups of the whole equation, on paper, then one
    small run on them.
11. **Everything together**: one Field with the small winners (GPS and noise at 30 days, the curriculum on
    CB-20), inputs allowed per horizon so next-day never sees them.
12. **Calibration**: the day totals recalibrated before the cut (CB-05).
13. **The growth curve, CB-16**: the gate score against the number of tiles and three model sizes; the next
    point predicted before it is trained.
14. **The full heart**: softmax heads beside the additive ones, the region's first year and lookalike
    sequences as context, trained on everything the curve says pays.
15. Head to head on the gate: Field 2 vs the Field vs `etas_local_alpha`. Figure: the growth curve with the
    predicted point and the real one.

**Benchmarks:** **CB-35 (the world gate), new**; the hourly test, new; CB-16; CB-05, CB-19, CB-20.

**Reds aimed at:** CB-05, CB-20, CB-19.

**Done when**
- CB-35 frozen, the vault sealed, the roof measured
- the hourly test frozen, ETAS scored on it
- the growth curve's predicted point is within its stated error
- every past loser has a signal-check verdict
- Field 2's heart is graded on the gate beside the Field
- `lessons/P12.md` written

**Stop rule:** if the full heart cannot pass `etas_local_alpha` on the gate, the plan stops here and is
redrawn before P13.

**Expect:** written at the end of P11, from its numbers.

**Learn:** statistical power, sealed test sets, scaling laws, augmentation.

## P13 · The places → v0.14.0
**Goal:** the Earth as places that carry a hidden state, with quakes writing into them.

**Why:** a quake is a symptom; the rock in a place is what persists. Each place has many hidden values, and
its earthquakes are only the visible part of the iceberg.

**LLM analogy:** latent state tokens (Perceiver), masked modelling, in-context learning.

**Steps**
1. Place tokens with a many-number state; quakes write into them through the heart; places talk to each
   other; queries read from them.
2. **The memory that ages**: the state relaxes on the log clock, the squeezing learned. Replaces P11's fixed
   bins. On/off test.
3. **A big quake is a shape**: M5+ quakes write along their rupture (length from size, direction from the
   mechanism, GCMT). On/off test.
4. **Hide and rebuild**: hidden chunks of the past filled in, never across a test cut (a leak test in the
   suite). Aims at the quakes the network misses after a big one. On/off test.
5. **The context**: a never-seen region's first year in, frozen weights, a forecast out.
6. Adoption on the gate. Figure: the hidden state of every place on the day before and the day after a
   large quake.

**Benchmarks:** CB-35; CB-17; CB-20; **CB-36 (a new region from its first year), new**.

**Reds aimed at:** CB-17, CB-20.

**Done when**
- every part has its on/off number
- Field 2 with places is adopted on the gate, or the lesson says which parts stayed off and why
- `lessons/P13.md` written

**Expect:** gate +0.45 to +0.55, most of it in where · CB-17 below `context_kde` (19.0).

**Learn:** state against events, masked modelling, why a memory should forget on a log clock.

## P14 · What pushes the places → v0.15.0
**Goal:** the charging, the identity of each place, and the faults that link them.

**Why:** in P13 a place changes only when a quake happens. The real Earth has an engine running all the
time: plates load every place slowly, and a quake is the discharge.

**LLM analogy:** conditioning (a static embedding per token), graph attention.

**Steps**
1. **GPS worldwide** (Nevada Geodetic Laboratory, open with citation): velocities per place. Charging is
   computed from them, never from GEM's map (share-alike; a comparison floor only).
2. **Identity**: static numbers per place, each its own on/off test (heat flow, fault distance and
   direction, slab depth). Licences checked per source before use.
3. **Depth and faults**: each quake carries its depth; links between places follow fault direction. Full 3D
   is LATER.
4. **The receipt**: every forecast split into what moved it (charging, yesterday's quakes, identity).
5. Adoption on the gate. Figure: one forecast with its receipt.

**Benchmarks:** CB-35 and its where share; CB-11.

**Done when**
- each of charging, identity and fault links has its on/off number
- the receipt adds up to the forecast, tested
- `lessons/P14.md` written

**Expect:** gate +0.5 to +0.6 · the charging small and in where, not when (catalogues span ~40 years and
charging takes centuries): the part trusted least.

**Learn:** the earthquake cycle, strain rate, why "when" can't be learned from forty years.

## P15 · Understanding and the equation → v0.16.0
**Goal:** the three tests of understanding, and the equation.

**Why:** the project's two conjectures: a model can understand the Earth, and an equation describes what it
learned.

**LLM analogy:** auxiliary objectives, fine-tuning after pretraining, test-time sampling, distillation.

**Steps**
1. **Homework**: from the place state, predict GPS motion and how quakes slip (CB-04, CB-23, CB-34).
2. **Experiments**: after the real data, train on planets where one thing is pushed and watched, and on the
   human-made cases (CB-30, CB-33, CB-09).
3. **Dream the month**: many sampled futures for long forecasts; skill against lead time (CB-19, CB-27).
4. **The equation**: dimensionless inputs, the update rule read from the curves, a short formula. The test:
   the formula alone on the world gate (**CB-37, new**). The reading tool is a series fitted in log scale and
   snapped to known functions.
5. **The big days, check first**: the signal check on the gate's mainshock days. Build CB-25's detector only
   if the check finds signal; otherwise publish the measured answer.
6. **The state map**: every place's hidden state, daily, on a world map.
7. **One model**: Field 2 answers CB-01 and CB-04 through hide-and-rebuild and the homework; the old trunk
   retires when Field 2 passes it.

**Benchmarks:** **CB-37 (the world law), new**; CB-27; CB-25 if the check says so; CB-04, CB-23, CB-34,
CB-30, CB-33, CB-19, CB-01.

**Reds aimed at:** CB-19, CB-34, CB-23, CB-33.

**Done when**
- the three tests of understanding each have a number
- an equation is reported with its world score, or the negative result
- the big-days check has a verdict
- `lessons/P15.md` written

**Expect:** gate +0.6 to +0.7 · CB-19 green · a formula of ten terms or fewer keeping ~80% of the aftershock
gain · the big days: a measured "mostly no".

**Learn:** symbolic regression, dimensional analysis, chaos and predictability, the history of failed
precursors.

## P16 · Grow and ship → v1.0.0
**Goal:** the model grown as far as the curve says, and in strangers' hands.

**Why:** "same project, once we have stronger GPUs or new techniques, it just grows".

**LLM analogy:** model hub and cards, leaderboards.

**Steps**
1. Grow along the P12 curve (tokens before parameters); `regional.yaml` passes a smoke run (a rented GPU
   is optional).
2. The nightly locked forecast moves to Field 2 once it has passed the Field on the gate and the vault.
3. Weights with a model card, the data store and the planets as a dataset, the gate as an open leaderboard.
   Weights licence decided here.
4. **The demo**: pick a place and a day; the forecast, what happened, the state map and the receipt. The
   translator lives here as a small layer that may only say numbers the model returned.
5. `CONTRIBUTING.md` in full: add an adapter, a region, a task, a model.
6. The global prospective experiment (floatCSEP lets anyone add a model) once the locked forecasts hold.
7. The final report: what works, what doesn't, the board's history, what is next.

**Benchmarks:** everything on the board; CB-12 only if the translator earns it.

**Done when**
- one model answers the board
- weights, dataset, leaderboard and demo are live, and the demo runs a forecast for a stranger
- a new region added by following CONTRIBUTING alone passes every check
- `lessons/P16.md` and the final report written

**Expect:** the next size's gate score predicted within ~5%.

**Learn:** how an open research platform runs.

---

## Expected benchmarks

Bits per quake above ETAS fitted on each region's own past, in regions the model never read (CB-32 today,
the world gate from P12).

| After | World, never-read regions | Where it comes from |
|---|---|---|
| the Field (measured, P9h) | **+0.34** | 54 hand-made counters, a stock transformer |
| best simple floor (measured) | +0.07 | `etas_local_alpha` |
| P11 first run (measured, seed 0) | −0.25 with context, +0.007 without | the heart alone: an empty query |
| P11 rebuilt, lean pass (measured, one member, seed 0) | **−0.044** (expected −0.05 to +0.15) | state, neighbours, rulers, trained as tested |
| P12 the heart, in full | not set: P11 stopped | more of the world, any moment, the hourly test |
| P13 the places | +0.45 to +0.55 | state, rupture shapes, the context |
| P14 what pushes them | +0.5 to +0.6 | charging and identity, in where |
| P15 understanding | +0.6 to +0.7 | experiments, the month dreamed |

In plain words: Field 2 should end near double the Field's lead, anywhere on Earth, with a readable law for
most of what it knows. It is not expected to see large earthquakes coming. The old CB-02 / CB-03 table
(P11 storm detector +1.3 / +2.2) is retired: it rested on three large days.

Negative results are results: they go on the scoreboard and in the logbook like everything else.

## Later (not phased yet)
Moved here on 2026-11-12, each with its tasks: the ghost channel and the atlas (CB-10), the tidal-sensitivity test,
precipitation and air-pressure adapters · the translator as its own phase (CB-12; a small layer in P16's demo
instead) · the sparse-autoencoder microscope (CB-13; readable curves replace it) · the million-year router
and plate reconstructions (CB-14) · ladder rung 3 (CB-18) · region adapters (only if the context falls
short) · full 3D places · a paper · the dv/v wrong sign at Norcia · SeisLM variants.

From before: Spec 1.1.0: add generic `M` to the magnitude vocabulary (120 INGV events refused) · an
`ingv_quakeml` adapter for location and magnitude errors · a tidal-stress instrument · ocean loading · Vp/Vs
as a time-lapse instrument · Craton's own ML catalogue from raw waveforms · ladder rungs 4–5 · DAS /
ocean-floor data adapters · a prior-fitted Craton (FIM-PP, arXiv 2509.24762).

---

## Tracker

- [x] P0 Foundation · [x] P1 Universal token · [x] P2 Adapters + store · [x] P3 Benchmarks + baselines
- [x] P4 Craton-mini v0 · [x] P5 Fusion · [x] P6 Beat ETAS · [x] P7 The 8 locks
- [x] P8 Other planets (v0.9.0) · [x] P9 Hidden state (v0.10.0) · [x] P10 Listening (v0.11.0)
- [ ] P11 The viable test (stopped twice by its stop rule, 2026-11-13) · [ ] P12 The heart + the fair judge · [ ] P13 The places
- [ ] P14 What pushes the places · [ ] P15 Understanding + the equation · [ ] P16 Grow and ship (v1.0.0)
