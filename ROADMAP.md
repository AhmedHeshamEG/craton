# Roadmap

Fourteen phases, P0 → P13. Each one ends with hard exit tests, a lesson in [`lessons/`](lessons/), a figure,
and a release. A phase isn't done if any exit test fails. Live numbers are in [`SCOREBOARD.md`](SCOREBOARD.md);
the day-by-day story is in [`LOGBOOK.md`](LOGBOOK.md).

```
CORE       ██████████████████████████░░░░   7/8    P0–P7   foundation → beat ETAS → a model that knows when it doesn't know
DISCOVERY  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/4    P8–P11  fake planets, the ghost, the translator, the microscope
SCALE      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/2    P12–P13 any time horizon, then mini → regional → planet
ALL        ███████████████░░░░░░░░░░░░░░░   7/14
```

Two versions share one codebase: **Craton-mini** (a laptop RTX 2060 with 6 GB, central Italy; every phase runs
here first) and **Craton** regional → planet (same code, bigger config files, unlocked in P13).

Each phase lists its **Goal**, **Why**, the **LLM analogy** it borrows, what it builds, its **Benchmarks**, what
is deliberately **not** in it, the exit tests (**Done when**) and what its lesson teaches (**Learn**).

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
| CB-09 | **Sim-to-real gap** | Synthetic-test vs real-test score gap, per generator, tracked over versions | — | P8 |
| CB-10 | **Planted discovery** | Tides removed on purpose: does the ghost recover the 12.42 h / 24 h / 14.8 d periods? | Random residual | P9 |
| CB-11 | **Predictability meter** | 90%-coverage window width in time (days), space (km), magnitude, for M≥4 | ETAS window width | P7 |
| CB-12 | **Interpreter faithfulness** | ~200 fixed questions: numeric agreement with tool outputs, citation coverage, hallucination rate, correct abstention | Plain LLM, no tools | P10 |
| CB-13 | **Microscope** | SAE reconstruction + recovery of a planted hidden variable + share of features with validated labels | Random directions | P11 |
| CB-14 | **Deep time** | Plate positions vs published GPlates reconstructions at past ages; angular error | Constant-velocity extrapolation | P12 |
| CB-15 | **Efficiency** | Model FLOPs utilisation of the pretraining run on the RTX 2060; tokens/s, peak VRAM, wall-clock alongside | Must fit in 6 GB (a hard gate) | P4 |
| CB-16 | **Scaling-law fit** | Loss vs params and tokens over 4–5 sizes; predict the next size before training it | — | P13 |
| CB-17 | **In-context adaptation** | Only the first N days of a new region in context, no retraining: does CB-01 improve as N grows? | Empty context | P5 |
| CB-18 | **Ladder rung 3** | Timing of the next event on repeating-earthquake patches | Periodic / Poisson | P13 (spec only) |

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

---

# DISCOVERY

## P8 · Fake planets (synthetic pretraining) → v0.9.0
**Goal:** pretrain on millions of simulated years, then fine-tune on the real Earth.

**Why:** fixes rarity (many full fault cycles) and gives enough tokens to justify a bigger model.

**LLM analogy:** synthetic-data pretraining + curriculum; sim-to-real transfer.

**Build**
1. At least **two** independent generators:
   1. ETAS synthetic catalogs with randomised parameters
   2. a physics-based cycle simulator (`SPIKE`: QDYN or another quasi-dynamic rate-and-state code builds and
      runs on Windows / WSL; fallback: a verified spring-block model in NumPy)
2. Synthetic GPS from simulated slip (elastic half-space), synthetic tides.
3. Same spec, `provenance.synthetic = true`.
4. Curriculum: synthetic only → mixed → real fine-tune.
5. 50–100M synthetic tokens on the laptop (generation runs overnight).
6. Figure: a fake planet's 10,000-year catalog next to the real Apennines.

**Benchmarks:** everything before + **CB-09 (sim-to-real gap, new)**.

**Done when**
- synthetic pretraining improves real CB-01 / CB-02 beyond noise, or the negative result is documented with
  evidence
- model size raised to ~2–5M only if the scaling evidence supports it
- the sim-to-real gap tracked per generator
- `lessons/P8.md` written

**Learn:** simulators, rate-and-state friction (intuition), synthetic data, curriculum, transfer.

## P9 · Ghost channel + the atlas → v0.10.0
**Goal:** let the model point at forces we didn't give it.

**Why:** no list of forces is provably complete; the ghost turns "missing input" into a discovery signal.

**LLM analogy:** learned register / latent tokens; residual analysis.

**Build**
1. A learnable "unexplained forcing" latent per region / time, regularised so it only takes what the known
   inputs can't explain.
2. Residual toolkit: spectra, autocorrelation, correlation with candidate external series.
3. **Planted discovery:** train with tides removed, check the ghost recovers the tidal periods.
4. **The Forcing & Property Atlas** (`craton/atlas/`): a machine-readable registry of all forces (the 7 layers
   + the measurement layer) and properties, each `available` / `adapter-planned` / `no-data` / `null-test`
   (solar activity is a null test).
5. Two more adapters from the atlas where public data is reachable (precipitation reanalysis, air pressure).
6. Figure: the ghost's spectrum with the tidal peaks it found on its own.

**Benchmarks:** **CB-10 (new)** + everything before.

**Done when**
- CB-10 recovers the planted tide with a clear spectral peak
- every atlas entry has a status
- `lessons/P9.md` written

**Learn:** residuals, spectral analysis, the Neptune story; why null tests matter.

## P10 · Interpreter A: the translator → v0.11.0
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
- `lessons/P10.md` written

**Learn:** tool calling, grounding, why the LLM never does the physics.

## P11 · Interpreter B: the microscope → v0.12.0
**Goal:** break the model's internal vectors into named concepts.

**Why:** where the "theory" gets pulled out of the simulator.

**LLM analogy:** mechanistic interpretability with sparse autoencoders.

**Build**
1. Activation capture from chosen layers.
2. Sparse autoencoder training on the laptop.
3. Feature browser: per feature, top contexts, maps and timelines (static HTML).
4. **Planted-variable test** on synthetic data with a known hidden variable.
5. A label counts only with a quantitative test against an independent measurement.
6. Figure: the feature that lit up for the planted variable, on a map.

**Benchmarks:** **CB-13 (new)**.

**Done when**
- the planted variable recovered
- a handful of features validated with numbers, unknown ones listed for investigation
- `lessons/P11.md` written

**Learn:** activations, superposition, sparse dictionaries, how labs read LLM internals.

---

# SCALE

## P12 · Long horizons + multi-scale zoom → v0.13.0
**Goal:** any (place, time) question answered at the right resolution, from seconds to millions of years.

**Why:** the prompt to Craton is a place, a time and some context. The time can be next week, 1349, or a million years from now.

**LLM analogy:** long context + hierarchical models (coarse-to-fine, Perceiver-style bottlenecks); test-time
compute (sampling many futures).

**Build**
1. Resolution ladder: rupture (s) / cycle (yr) / plate (Myr) tokens, each with declared resolution.
2. Plate-scale adapter from published reconstructions (pyGPlates or exported rotation files).
3. Query router: picks the sharpest resolution the evidence supports and **states it** in the answer.
4. Ensemble sampling of pasts and futures.
5. Figure: one place, zoomed from +7 days to +1 Myr, resolution shrinking on screen.

**Benchmarks:** **CB-14 (new)** + everything before.

**Done when**
- T = +1 Myr returns plate-scale output with its resolution stated; T = +7 days returns event rates
- CB-14 beats constant-velocity extrapolation
- `lessons/P12.md` written

**Learn:** multi-scale modelling, plate reconstructions, why resolution shrinks with horizon.

## P13 · Scale-ready + open platform → v0.14.0
**Goal:** mini → regional → planet with a config switch, and strangers can contribute safely.

**Why:** same project; when stronger GPUs or new techniques arrive, it grows without a redesign.

**LLM analogy:** scaling laws (Kaplan, Chinchilla), distributed training, LoRA, model hub + cards.

**Build**
1. Scaling study on the laptop: 4–5 sizes, fit loss vs params / tokens, predict the next size (CB-16).
2. `regional.yaml` (~50–150M) and `planet.yaml` (~1–10B) pass a smoke run on rented GPUs.
3. DDP / FSDP behind config flags; nothing in the core assumes one GPU.
4. Regional low-rank adapters for new regions on a frozen trunk.
5. The full `CONTRIBUTING.md`: add an adapter, a force, a property, a benchmark; all add-only,
   ratchet-protected.
6. CB-18 spec + a data survey of repeating earthquakes.
7. The final report: what works, what doesn't, the scoreboard history, what's next.
8. Figure: the scaling law with the predicted point and the real one.

**Benchmarks:** **CB-16 (new)**; CB-18 spec only.

**Done when**
- the scaling prediction lands within a documented error
- the regional config passes a smoke run
- a new adapter added by following CONTRIBUTING alone passes every check without touching core code
- `lessons/P13.md` + the final report written

**Learn:** scaling laws, distributed training, LoRA, how an open research platform runs.

---

## Expectations (honest)

| Outcome | Chance |
|---|---|
| Working repo, spec, pipelines, CratonBench, mini model training | ~90% |
| Mini matches ETAS on Apennines M3+ | ~70% |
| GPS + tides + fake planets beat ETAS on small quakes → arXiv paper | ~40% |
| Ghost channel / microscope finds something real and new | ~10–15% |
| Exact time / place of a big quake | ~0% on a laptop; the ladder's job, over years |

Negative results are results: they go on the scoreboard and in the logbook like everything else.

---

## Tracker

- [x] P0 Foundation · [x] P1 Universal token · [x] P2 Adapters + store · [x] P3 Benchmarks + baselines
- [x] P4 Craton-mini v0 · [x] P5 Fusion · [x] P6 Beat ETAS · [ ] P7 The 8 locks
- [ ] P8 Fake planets · [ ] P9 Ghost + atlas · [ ] P10 Translator · [ ] P11 Microscope
- [ ] P12 Long horizons · [ ] P13 Scale + open platform
