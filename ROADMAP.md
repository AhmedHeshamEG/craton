# Roadmap

Seventeen phases, P0 → P16. Each one ends with hard exit tests, a lesson in [`lessons/`](lessons/), a figure,
and a release. A phase isn't done if any exit test fails. Live numbers are in [`SCOREBOARD.md`](SCOREBOARD.md);
the day-by-day story is in [`LOGBOOK.md`](LOGBOOK.md).

```
CORE       ██████████████████████████████   8/8    P0–P7   foundation → beat ETAS → a model that knows when it doesn't know
DISCOVERY  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/7    P8–P14  other planets, hidden state, listening, storm detector, ghost, translator, microscope
SCALE      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/2    P15–P16 any time horizon + the predictability horizon, then mini → planet
ALL        ██████████████░░░░░░░░░░░░░░░░   8/17
```

Two versions share one codebase: **Craton-mini** (a laptop RTX 2060 with 6 GB, central Italy; every phase runs
here first) and **Craton** regional → planet (same code, bigger config files, unlocked in P16).

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
| CB-10 | **Planted discovery** | Tides removed on purpose: does the ghost recover the 12.42 h / 24 h / 14.8 d periods? | Random residual | P12 |
| CB-11 | **Predictability meter** | 90%-coverage window width in time (days), space (km), magnitude, for M≥4 | ETAS window width | P7 |
| CB-12 | **Interpreter faithfulness** | ~200 fixed questions: numeric agreement with tool outputs, citation coverage, hallucination rate, correct abstention | Plain LLM, no tools | P13 |
| CB-13 | **Microscope** | SAE reconstruction + recovery of a planted hidden variable + share of features with validated labels | Random directions | P14 |
| CB-14 | **Deep time** | Plate positions vs published GPlates reconstructions at past ages; angular error | Constant-velocity extrapolation | P15 |
| CB-15 | **Efficiency** | Model FLOPs utilisation of the pretraining run on the RTX 2060; tokens/s, peak VRAM, wall-clock alongside | Must fit in 6 GB (a hard gate) | P4 |
| CB-16 | **Scaling-law fit** | Loss vs params and tokens over 4–5 sizes; predict the next size before training it | — | P16 |
| CB-17 | **In-context adaptation** | Only the first N days of a new region in context, no retraining: does CB-01 improve as N grows? | Empty context | P5 |
| CB-18 | **Ladder rung 3** | Timing of the next event on repeating-earthquake patches | Periodic / Poisson | P16 (spec only) |
| CB-19 | **30-day forecast** | As CB-02, one forecast per 30 days: the horizon where GPS strain might pay | ETAS | P8 |
| CB-20 | **Apennines, enhanced inputs** | CB-02's scoring, with the 2016–17 ML catalogue as input (relocated with hindsight, so its own task) | ETAS on the same inputs | P8 |
| CB-21 | **L'Aquila, enhanced inputs** | CB-03's scoring, with the 2009 ML catalogue as input | ETAS on the same inputs | P8 |
| CB-22 | **World sequences** | Daily forecasts on 14 held-out sequences: six in regions trained on (California, Japan, New Zealand), eight in regions never seen (Turkey, Chile, Alaska); gain per quake, the never-seen half is the headline | ETAS fitted on each box's own past | P8 |
| CB-23 | **Hidden-state truth** | Inferred stress axes vs focal-mechanism stress inversions (angle); inferred pressure vs Vp/Vs and CO₂ maps | Regional-average stress | P9 |
| CB-24 | **Migration** | Connection graph learned before 2016 vs the order the 2016–17 sequence lit up | Distance only | P9 |
| CB-25 | **Mainshock days** | Skill on held-out world M6+ days: Molchan curve, false alarms per hit | ETAS | P11 |
| CB-26 | **Distilled law** | A symbolic formula's held-out gain over ETAS vs the network's | the network itself | P14 |
| CB-27 | **Predictability horizon** | Skill over ETAS vs lead time (1 day → 1 year): where it reaches zero, with error bars | — | P15 |
| CB-28 | **Velocity gap-fill** | Hide one station's dv/v series, rebuild it | Neighbour average | P10 |
| CB-29 | **EarthquakeNPP** | The outside benchmark (Stockman et al., TMLR 2026): their California data, splits, ETAS and metric, untouched; nats per quake | Their ETAS | P8 |

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

**Benchmarks:** **CB-09 (two-planet test + sim-to-real gap, new), CB-19, CB-20, CB-21, CB-22, CB-29
(new)** + everything before.

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
3. **Graded against independent truth:** stress orientation vs the 72,000 focal mechanisms of the 2016–17
   ML catalogue (stress inversions); pressure vs Vp/Vs tomography and the Apennine CO₂ outflow maps.
4. **The migration test:** the graph learned on data before 2016 vs the order the 2016–17 sequence lit up
   (Amatrice → Visso → Norcia → Campotosto).
5. A Coulomb rate-and-state (CRS) baseline on CB-02 (Mancini et al. 2019 beat ETAS after Visso).
6. Figure: Italy's inferred pressure map next to the CO₂ map it was never shown.

**Benchmarks:** **CB-23 (hidden-state truth), CB-24 (migration), new**; CRS joins the floors.

**Done when**
- CB-23 beats a regional-average-stress baseline; CB-24 beats distance-only
- CB-02 / CB-03 improve on P8 or the negative result is documented
- the referee passes every map shown in public
- `lessons/P9.md` written

**Expect:** CB-23 stress axes within ~20° of the inversions · CB-24 the true next patch in the graph's top 3 ·
CB-02 ≈ +0.9 · CB-03 ≈ +1.6.

**Learn:** stress, pore pressure, Coulomb failure, rate-and-state as a stress meter, inverse problems.

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

## P11 · The storm detector: the mainshock days → v0.12.0
**Goal:** raise tomorrow's rate on the days a big quake is coming, without crying wolf.

**Why:** 70–78% of the headroom above ETAS sits on the M5+ days. On L'Aquila one day at 30× ETAS is worth
+1.3 bits for the whole year; break-even is ~97 false-alarm days per hit.

**LLM analogy:** a classifier head with calibrated probabilities; anomaly detection; test-time sampling of
many futures.

**Build**
1. A head for "an M5+ in this cell tomorrow", folded into the rates.
2. Its inputs: b-value trends (L'Aquila: 0.68 in the last 10 days), foreshock acceleration and migration,
   GPS transients, the hidden state (P9), the noise instruments (P10).
3. Trained on thousands of mainshock sequences worldwide (QuakeGen used 2,126), never on Italy alone.
4. Every alarm passes the referee and the fake-precursor attack; forecasts locked.
5. Figure: a Molchan diagram, Craton vs ETAS, and the L'Aquila foreshock months day by day.

**Benchmarks:** **CB-25 (mainshock-day skill on held-out world mainshocks, with false alarms per hit), new**;
CB-02, CB-03.

**Done when**
- CB-25 above ETAS's Molchan curve on held-out mainshocks
- false alarms per hit below the break-even on CB-02 and CB-03
- `lessons/P11.md` written

**Expect:** CB-25 flags ~1 in 4 held-out M6+ days at ≤ 50 false alarms per hit · CB-02 ≈ +1.3 ·
CB-03 ≈ +2.2.

**Learn:** foreshocks, b-value, Molchan diagrams, the history of failed precursors and why they failed.

## P12 · Ghost channel + the atlas → v0.13.0
**Goal:** let the model point at forces we didn't give it.

**Why:** no list of forces is provably complete; the ghost turns "missing input" into a discovery signal.

**LLM analogy:** learned register / latent tokens; residual analysis.

**Build**
1. A learnable "unexplained forcing" latent per region / time, regularised so it only takes what the known
   inputs can't explain.
2. Residual toolkit: spectra, autocorrelation, correlation with candidate external series.
3. **Planted discovery:** train with tides removed, check the ghost recovers the tidal periods.
4. **Tidal sensitivity, done right:** does tidal phase-locking of small quakes rise before big ones?
   Declustered, fake clocks, through the referee.
5. **The Forcing & Property Atlas** (`craton/atlas/`): every force and property with a status
   (`available` / `adapter-planned` / `no-data` / `null-test`; solar activity is a null test).
6. Two more adapters from the atlas (precipitation reanalysis, air pressure).
7. Figure: the ghost's spectrum with the tidal peaks it found on its own.

**Benchmarks:** **CB-10 (new)** + everything before.

**Done when**
- CB-10 recovers the planted tide with a clear spectral peak
- the tidal-sensitivity test has a referee verdict
- every atlas entry has a status
- `lessons/P12.md` written

**Expect:** CB-10 finds M2, O1 and K1 · tidal sensitivity: a clean measured answer either way.

**Learn:** residuals, spectral analysis, the Neptune story; why null tests matter.

## P13 · Interpreter A: the translator → v0.14.0
**Goal:** ask Craton questions in plain language.

**Why:** nobody reads a 256-number vector.

**LLM analogy:** tool use / function calling, RAG, grounded answers with citations.

**Build**
1. A typed query API: `query(region, time_or_window, question_type) → distributions + sources`.
2. A small local open LLM (quantised, fits the laptop) or an API model by config, wired through tool calls.
3. Enforced in code: the answer may only contain numbers that came from tool results; a checker rejects the
   rest.
4. Every claim cites the sensors / tokens / outputs it used.
5. Abstention passes through: when Craton abstains, the interpreter says "I don't know" and why.
6. The fixed **200-question set** for CB-12, written before any answer is built.
7. Figure: one real question, its answer, and every number traced back to its tool call.

**Benchmarks:** **CB-12 (new)**.

**Done when**
- hallucinated-number rate = 0 on CB-12
- beats the plain-LLM baseline on faithfulness
- `lessons/P13.md` written

**Expect:** CB-12 zero invented numbers; abstains correctly on every OOD question.

**Learn:** tool calling, grounding, why the LLM never does the physics.

## P14 · Interpreter B: the microscope + the law → v0.15.0
**Goal:** break the model's internal vectors into named concepts, and distill what it learned into a formula.

**Why:** where the "theory" gets pulled out of the simulator. Cranmer et al. (2020) got a new dark-matter
formula out of a network this way, and it generalised better than the network.

**LLM analogy:** mechanistic interpretability with sparse autoencoders; symbolic distillation.

**Build**
1. Activation capture from chosen layers.
2. Sparse autoencoder training on the laptop.
3. Feature browser: per feature, top contexts, maps and timelines (static HTML).
4. **Planted-variable test** on synthetic data with a known hidden variable.
5. A label counts only with a quantitative test against an independent measurement.
6. **The law:** symbolic regression (PySR) on the Field's correction over ETAS. A formula counts only if it
   keeps most of the network's gain on held-out sequences.
7. Figure: the feature that lit up for the planted variable, on a map; the formula next to Omori's.

**Benchmarks:** **CB-13, CB-26 (distilled law: the formula's CB-02 / CB-22 gain vs the network's), new**.

**Done when**
- the planted variable recovered
- a handful of features validated with numbers, unknown ones listed for investigation
- a formula reported with its held-out score, or the negative result
- `lessons/P14.md` written

**Expect:** CB-13 planted variable found · CB-26 a formula of ≤ 10 terms keeping ~80% of the gain.

**Learn:** activations, superposition, sparse dictionaries, symbolic regression, how laws get written.

---

# SCALE

## P15 · Long horizons + the predictability horizon → v0.16.0
**Goal:** any (place, time) question answered at the right resolution, and the crust's predictability limit
measured.

**Why:** the prompt to Craton is a place, a time and some context, and the time can be next week, 1349, or a
million years from now. And the question nobody has answered: how far ahead is the crust predictable at all? Weather's is about two weeks
(Lorenz).

**LLM analogy:** long context + hierarchical models; test-time compute (sampling many futures).

**Build**
1. Resolution ladder: rupture (s) / cycle (yr) / plate (Myr) tokens, each with declared resolution.
2. Plate-scale adapter from published reconstructions (pyGPlates or exported rotation files).
3. Query router: picks the sharpest resolution the evidence supports and **states it**.
4. Ensemble sampling of pasts and futures.
5. **The predictability horizon:** skill over ETAS as a function of lead time (1 day → 1 year), per region;
   the lead time where it reaches zero, with error bars, and how it moves with data and model size.
6. Figure: one place, zoomed from +7 days to +1 Myr; the skill-vs-lead-time curve.

**Benchmarks:** **CB-14, CB-27 (predictability horizon), new** + everything before.

**Done when**
- T = +1 Myr returns plate-scale output with its resolution stated; T = +7 days returns event rates
- CB-14 beats constant-velocity extrapolation
- CB-27 reported with error bars
- `lessons/P15.md` written

**Expect:** CB-14 beats constant velocity · CB-27 a horizon measured in weeks to months, with its error bar.

**Learn:** multi-scale modelling, plate reconstructions, chaos and predictability limits.

## P16 · Scale-ready + open platform → v0.17.0
**Goal:** mini → regional → planet with a config switch, and strangers can contribute safely.

**Why:** "same project, once we have stronger GPUs or new techniques, it just grows".

**LLM analogy:** scaling laws (Kaplan, Chinchilla), distributed training, LoRA, model hub + cards.

**Build**
1. Scaling study on the laptop: 4–5 sizes, fit loss vs params / tokens, predict the next size (CB-16);
   context length (1,024 vs 2,048) measured with its cost.
2. `regional.yaml` (~50–150M) and `planet.yaml` (~1–10B) pass a smoke run on rented GPUs.
3. DDP / FSDP behind config flags; nothing in the core assumes one GPU.
4. Regional low-rank adapters for new regions on a frozen trunk.
5. The full `CONTRIBUTING.md`: add an adapter, a force, a property, a benchmark; all add-only,
   ratchet-protected. CratonBench open to outside submissions.
6. CB-18 spec + a data survey of repeating earthquakes.
7. The final report: what works, what doesn't, the scoreboard history, what's next.
8. Figure: the scaling law with the predicted point and the real one.

**Benchmarks:** **CB-16 (new)**; CB-18 spec only.

**Done when**
- the scaling prediction lands within a documented error
- the regional config passes a smoke run
- a new adapter added by following CONTRIBUTING alone passes every check without touching core code
- `lessons/P16.md` + the final report written

**Expect:** CB-16 next size predicted within ~5% of its loss.

**Learn:** scaling laws, distributed training, LoRA, how an open research platform runs.

---

## Expected benchmarks

Bits per M3+ quake **above ETAS** on the two headline forecasts (CB-02 ETAS −7.12, CB-03 ETAS −9.40). A score
of k bits gives the real quakes 2^k times ETAS's probability. Measured rows are measured; the rest is the plan.

| After | CB-02 Apennines | CB-03 L'Aquila | Where it comes from |
|---|---|---|---|
| P5 (measured) | −4.7 (1/26 of ETAS) | −4.4 (1/21) | the window forgets, the map is blurry |
| P6 the Field (measured) | **+0.30** (1.2×) | **+1.16** (2.2×) | memory, a grid of cells, ETAS as the start |
| P8 other planets | +0.6 (1.5×) | +1.4 (2.6×) | many sequences, small quakes, pressure → quakes |
| P9 hidden state | +0.9 (1.9×) | +1.6 (3×) | stress, fluids, the wiring |
| P11 storm detector | **+1.3** (2.5×) | **+2.2** (4.6×) | the mainshock days |
| the roof (all-knowing) | +4.2 (18×) | +6.0 (64×) | measured by the oracles |

The great version, where everything works: CB-02 ≈ +2 (4×), CB-03 ≈ +3.5 (11×), the enhanced-input tasks higher
still. Every other task's expectation is on its phase's **Expect** line. The exact time and place of a big quake
is not on this table: that is the ladder's job, over years.

Negative results are results: they go on the scoreboard and in the logbook like everything else.

---

## Tracker

- [x] P0 Foundation · [x] P1 Universal token · [x] P2 Adapters + store · [x] P3 Benchmarks + baselines
- [x] P4 Craton-mini v0 · [x] P5 Fusion · [x] P6 Beat ETAS · [x] P7 The 8 locks
- [ ] P8 Other planets · [ ] P9 Hidden state · [ ] P10 Listening · [ ] P11 Storm detector
- [ ] P12 Ghost + atlas · [ ] P13 Translator · [ ] P14 Microscope + law
- [ ] P15 Long horizons · [ ] P16 Scale + open platform
