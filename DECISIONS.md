# Decisions

Every choice that shapes the project, with why and what was rejected. **Append-only**: `craton check` fails if
an old line changes. A decision that turns out wrong gets a new decision that supersedes it, and says so.

Format: `D-nnn · date · phase` — the decision · why · rejected.

---

### D-001 · 2026-10-19 · P0 — The name is Craton

A craton is the ancient, stable core of a continent: rock that hasn't deformed in billions of years while new
crust docks onto its edges. That's the engineering rule of this repo, as a rock.

**Why.** The working name was Pangea. Both spellings are taken in exactly this field: PANGAEA is the Earth and
environmental science data publisher run by AWI and MARUM (a likely data source for us, with a Python client
called `pangaeapy`), PANGAEA is a 2024 benchmark for geospatial foundation models (arXiv 2412.04204), and
Pangea is a multilingual multimodal LLM from CMU. A project people search for needs a name it can own.

**Rejected.** Keeping Pangaea (search and package collisions). Terrane (read as "terrain"). Rodinia (a
well-known GPU benchmark suite).

### D-002 · 2026-10-19 · P0 — `pip install craton-earth`, `import craton`

**Why.** `craton` on PyPI is held by an empty 0.01 placeholder. The import name stays short.

**Rejected.** `cratonml` (taken), renaming the package away from the project name.

### D-003 · 2026-10-19 · P0 — Code under Apache-2.0

**Why.** Permissive, with an explicit patent grant, which matters for a research platform that companies and
observatories may build on. Datasets keep their own licences, recorded in their data cards; the licence for
trained weights is decided when the first weights exist.

**Rejected.** MIT (no patent grant). GPL (keeps observatories and companies away from contributing adapters).

### D-004 · 2026-10-19 · P0 — uv, one lockfile, PyTorch through two exclusive extras

`uv sync --extra cu126` on an NVIDIA machine, `uv sync --extra cpu` everywhere else.

**Why.** PyPI's PyTorch wheel for Windows is CPU-only, and the CUDA wheels for Linux pull ~3 GB that CI doesn't
need. Two conflicting extras, each pinned to its own PyTorch index, keep one lockfile honest for both.

**Rejected.** A setup script that detects the GPU (one more thing to break). Torch outside the lockfile
(unpinned, breaks reproducibility).

### D-005 · 2026-10-19 · P0 — PyTorch 2.14.1 with CUDA 12.6

**Why.** The laptop GPU is an RTX 2060 (Turing, sm_75). The cu126 build lists sm_75 in its architecture list
(checked with `torch.cuda.get_arch_list()`), and `craton doctor --smoke` trains in fp16 on it. cu126 has the
widest architecture support of the current builds.

**Rejected.** cu128 (no 2.14 builds for Windows at the time). cu130/cu132 (fine for newer GPUs; no gain on
Turing).

### D-006 · 2026-10-19 · P0 — Mixed precision is planned per machine, never assumed

A config asks for a precision; `plan_precision` decides what the machine actually runs. bf16 on anything older
than Ampere becomes fp16 with gradient scaling and a warning; fp16 always uses a gradient scaler; no CUDA
means fp32 on the CPU.

**Why.** Turing has fp16 tensor cores and no native bf16. The same config must run on the laptop and on an
A100 without edits.

### D-007 · 2026-10-19 · P0 — The ratchet compares with the champion, noise = 2 × the larger std

A task's score on the board is its best accepted score (the champion). A run is *regressed* if it is worse than
the champion by more than `2 × max(std_champion, std_run)`, *improved* if better by more than that, otherwise
*same*, and "same" never replaces the champion. Every score needs ≥ 3 distinct seeds. A task can never vanish
from a run, and a changed fingerprint (test data or metric) is invalid: a changed task needs a new ID.

**Why.** Comparing with the previous run lets a hundred within-noise losses stack into a big one. Taking the
noisier side keeps a lucky low-variance champion from making every honest rerun look like a regression.

**Rejected.** Pooled standard error (too lenient with 3 seeds). A fixed tolerance per task (has to be guessed).

### D-008 · 2026-10-19 · P0 — Generated means enforced

`SCOREBOARD.md` is re-rendered from `results/scoreboard.json` by the check, which fails on any difference.
`LOGBOOK.md` and `DECISIONS.md` are compared line by line with a git reference (`HEAD` locally, the base
branch on CI), and the check fails if a committed line changed.

**Why.** "Never typed by hand" and "append-only" only hold if a machine refuses the alternative.

### D-009 · 2026-10-19 · P0 — Phase Pn ships as v0.(n+1).0

P0 is v0.1.0, P12 is v0.13.0. 1.0 is reserved for when the token spec and the platform are proven by an
outside contributor.

### D-010 · 2026-10-19 · P0 — Figures come from real data and a script

Every figure in the repo is drawn by a script in `scripts/figures/` from real or clearly synthetic data, in
one style (near-black ground, one accent, Inter), with light and dark variants. No generated imagery.

**Why.** A figure you can't regenerate is a claim you can't check.

### D-011 · 2026-10-20 · P1 — One clock: seconds since 1970, proleptic Gregorian

Every token's `time` is one float: seconds since 1970-01-01T00:00:00 UTC, leap seconds not counted (POSIX).
Dates from before 15 October 1582 are converted from the Julian calendar first, so the Montecassino report of
9 September 1349 sits on 17 September. `time_iso` is an optional readable copy, checked against `time` and
left out of the token's identity hash.

**Why.** One float runs from a 2016 seismogram (sub-microsecond) to a billion years ago (about a minute), so
the same axis serves P4 and P11. A `datetime` stops at year 1.

**Rejected.** `datetime64` (no deep time), a separate field for geological time (two clocks to keep in sync),
TAI (leap seconds matter to nobody at Craton's resolution, and every data provider speaks UTC).

### D-012 · 2026-10-20 · P1 — Depth is measured from the WGS84 ellipsoid

`depth_km` is positive down from the WGS84 ellipsoid; negative is above it. The plan said "sea level".

**Why.** The ellipsoid is one exact surface everywhere, and it is what GNSS measures natively. Sea level (the
geoid) depends on which geoid model you pick. Catalogue depths given below sea level differ by under 0.11 km
anywhere on Earth, which is inside the depth uncertainty of every hypocentre, and adapters record it as such.

**Rejected.** Sea level (model-dependent), depth below the local surface (a moving target on a volcano).

### D-013 · 2026-10-20 · P1 — Longitude lives in [-180, 180), and only there

Providers write 0–360 and even −346.8 (the Nevada Geodetic Laboratory's files do). `wrap_lon` converts; the
validator refuses anything else.

**Why.** One spelling per place, or the same measurement gets two identities and dedup breaks.

### D-014 · 2026-10-20 · P1 — Units belong to the instrument, as UCUM codes

A token carries numbers; its instrument's registry entry says each channel's name, unit and allowed range.

**Why.** Units written into every token bloat the files and let two tokens of one instrument disagree. UCUM
codes are an existing standard. Permeability and grain size span many orders of magnitude, so their channels
are base-10 logs (`log10(m2)`, `log10(mm)`).

**Rejected.** Per-token units. A units library dependency (the registry only needs to name units, not convert
them).

### D-015 · 2026-10-20 · P1 — Uncertainty is required, positive, and says where it came from

`uncertainty` (1σ per channel, never zero) is required, and so is `uncertainty_basis`: `reported` by the
source, `derived` by a computation, or `assumed` by the adapter. Where and when are uncertain too, so an
optional `location_uncertainty` (horizontal, vertical, time) sits next to `resolution`, which is the size of
the patch a value describes, a different thing.

**Why.** A model that weighs a monk against a GPS receiver needs to know how far to trust each, and whether
anyone measured that trust. INGV publishes no magnitude uncertainty for the Norcia main shock: the token says
`assumed` instead of pretending.

**Rejected.** Allowing zero (nothing is measured perfectly). Folding location error into `resolution`.

### D-016 · 2026-10-20 · P1 — Text labels are qualifiers, declared per instrument

Magnitude type, station code, intensity scale, reference frame: each instrument declares its qualifiers,
which ones are required, and a closed vocabulary or a pattern for each.

**Why.** An ML 4.0 and an Mw 4.0 are different numbers; the magnitude type is required on every catalogue
token, so they can never mix silently.

**Rejected.** One channel per magnitude type (mostly empty vectors). Free-form metadata (unvalidated).

### D-017 · 2026-10-20 · P1 — Categories are probabilities over a published class list

`lithology` is 13 channels, one per class of the Global Lithological Map (Hartmann & Moosdorf 2012), and the
channels must sum to 1. A sure answer is one-hot; "probably limestone" is a spread.

**Why.** Keeps `value` numeric for every instrument, carries the geologist's doubt, and borrows a class list
the field already uses.

### D-018 · 2026-10-20 · P1 — Add-only is checked against frozen snapshots

Each spec release writes `craton/spec/frozen/<version>.json` (token schema + registry). `craton check` fails
if the code removes or redefines anything in any snapshot of the current major version. Closed vocabularies
may grow. Descriptions are frozen with everything else. Details in `SPEC_POLICY.md`.

**Why.** "Add-only" only holds if a machine refuses the alternative, same as D-008.

### D-019 · 2026-10-20 · P1 — Token files are Parquet, named by their own hash

One file per (instrument, year) per write, under `instrument=<name>/year=<year>/`, named by the first 16 hex
characters of its sha256. Tokens are sorted (time, then token hash) and duplicates refused; the writer settings
are fixed, so the same tokens always give the same bytes. An existing name is never rewritten.

**Why.** Content-addressed names make files immutable and self-checking for free. Parquet is columnar,
compressed and readable from any language.

**Rejected.** Appending to a file in place (breaks immutability). HDF5 / Zarr (P1's data is tables; gridded
data can arrive as its own store later).

### D-020 · 2026-10-20 · P1 — Examples are real where the data is reachable, and labelled where not

Three examples are built from raw records quoted in the code and hashed: the Norcia Mw 6.5 (INGV), the GNSS
station ARQT the day after (Nevada Geodetic Laboratory), and Montecassino Abbey in 1349 (INGV ASMI). The
forcing, property and derived examples carry `source: "Craton spec examples (illustrative values, not
measurements)"`.

**Why.** Examples teach, and teaching with made-up numbers that look real is how wrong facts spread.

### D-021 · 2026-10-20 · P2 — Solid-Earth tides come from Craton's own NumPy port of the IERS 2010 model

`craton/geo/solid_tide.py` ports the IERS Conventions routines (steps 1 and 2) to NumPy, with the Sun and the
Moon from Montenbruck & Gill's analytic series.

**Why.** The P1 spike found that PySolid ships no Windows wheels. It is also GPL-3, so its code can't be
vendored into an Apache-2.0 repo, and the Fortran it wraps (Milbert's `solid.for`) states no licence at all.
pyTMD (MIT) computes the same model on Windows but brings pyproj, SciPy, xarray, h5netcdf, lxml and pint for
one function. The IERS licence allows derived works if they are renamed, say how they differ and keep the
notice: all three are done (`craton/geo/IERS_NOTICE.txt`). Checked three ways: the published IERS test cases
to 1e-12 m, PySolid to 0.4 mm and the JPL DE440 ephemeris (through pyTMD) to 0.71 mm, at six places from 1990
to 2026. The fourth test case in the IERS header is broken (its Sun is 8.4 million km away and its expected
output repeats the third case), so the test suite uses the other three.

**Rejected.** PySolid in WSL or Docker (the laptop path would need a container to compute a tide). pyTMD as a
dependency (six heavy packages for one function). Vendoring GPL code.

### D-022 · 2026-10-20 · P2 — Tide tokens sit at every earthquake, plus one hourly series per region

Each catalogue event gets a `solid_earth_tide` token at its epicentre and origin time; the region gets one
hourly series at the centre of its box.

**Why.** The question tides answer for Craton is "do quakes prefer some tidal phases?", which needs the tide
at each quake's moment. A daily tide at each GNSS station would be useless: NGL's daily positions already
have the body tide modelled out, and the daily mean of a 12-hour wave is nearly zero. Across a box a few
hundred km wide the body tide changes smoothly, so one hourly series carries the clock for quiet days. Each
event's tide token carries the event token's hash, so the chain runs tide → event → INGV raw file.

**Rejected.** Hourly tides at every station (tens of millions of nearly identical numbers). Tides at stations
daily (meaningless, see above).

### D-023 · 2026-10-20 · P2 — Raw downloads are content-addressed, with an append-only manifest per source

`data/raw/<source>/<sha[:2]>/<sha256>.<ext>`, plus `manifest.jsonl` naming what each file answers (a month, a
station). Fetching the same bytes again adds nothing and keeps the first retrieval date; a provider's revision
gives new bytes, a new line, and the old file stays. Every token's `content_hash` is its raw file's sha256.
An INGV month over the service's 20,000-event cap is fetched in halves and joined under one header, so a month
is always one raw file.

**Why.** Reproducibility: the same raw files must give the same tokens, byte for byte, and they do (a test
holds it). Keeping the first retrieval date stops a re-fetch of unchanged bytes from changing every token's
`dataset_version`.

**Rejected.** One file per HTTP request (how a month is split would depend on server limits, and the set of
files would too).

### D-024 · 2026-10-20 · P2 — NGL data are `LicenseRef-NGL-as-is`, not CC-BY-4.0

The Nevada Geodetic Laboratory publishes no licence: its disclaimer says the data are supplied as-is, and its
acknowledgements ask users to cite Blewitt et al. (2018) and each station's own data citation. P1's GNSS
example said CC-BY-4.0; it now says `LicenseRef-NGL-as-is`. Craton keeps NGL files in the local cache and
redistributes none of them.

**Why.** A licence field that claims more than the provider granted is a false statement in every token.

### D-025 · 2026-10-20 · P2 — The INGV adapter reads the FDSN text format, and says what it assumes

The text format has no location or magnitude errors. Magnitude 1σ is assumed by type (0.1 for Mw, 0.2 for
ML, 0.3 for everything else), `resolution` is an assumed 1 km and 0.1 s, and `location_uncertainty` is left
out. The record keeps the event's `author`, which tells reviewed bulletin locations from survey ones.
Unknown magnitude types and event types are refused and counted, never mapped by guesswork.

**Why.** QuakeML carries the errors but is about twenty times bigger, and the 20,000-event cap would turn
~500 requests into thousands. The errors can arrive later as a second adapter (`ingv_quakeml`) with its own
tokens; nothing about this one has to change.

**Rejected.** Fabricating location errors from the network geometry. Converting ML and Md to Mw.

### D-026 · 2026-10-20 · P2 — Regions are config files with a fixed end; a build is a snapshot named by its hash

`configs/regions/central_italy.yaml`: 41-44 °N, 11.5-15 °E, 1985-01-01 → 2026-10-01, three adapters. A build
writes one snapshot JSON naming every token file (with its hash) and every raw file it read; the snapshot's
name is the hash of that content. Two builds from the same downloads get the same name.

**Why.** "Today" as an end date makes every build different. A fixed end and a content-named snapshot turn
"is this the same data?" into comparing two short strings. P3 freezes its splits against a snapshot.

**Rejected.** A rolling window. Timestamps in the snapshot (they would change its name for no reason).

### D-027 · 2026-10-20 · P2 — Data health goes on the scoreboard; completeness by maximum curvature

`craton data build` writes `results/data_health.json` (committed): per adapter, tokens, span, rejects by
reason, and per instrument a few telling numbers. For the catalogue, the magnitude of completeness of ML per
year by maximum curvature plus 0.2 (Woessner & Wiemer 2005), with at least 50 events a year. `SCOREBOARD.md`
renders it, and `craton check` fails if the two disagree.

**Why.** The constitution says the scoreboard carries data health from P2, and every number on it comes from
a script. Maximum curvature is fast, standard and honest about its bias when the +0.2 is applied.

**Rejected.** A separate data report page (one more thing to drift). Entire-magnitude-range methods for Mc
(better, slower, and P3's job when completeness enters the benchmarks).

### D-028 · 2026-10-20 · P2 — Data cards are generated from the adapters

Each adapter returns a `DataCard`; `craton data cards` renders `docs/data/`, and `craton check` fails if the
files drift from the code.

**Why.** Same reason as D-008: a card someone has to remember to update will be wrong within a month.

### D-029 · 2026-10-20 · P3 — Two eras and one station in ten are held out, forever

Era A is 2016-08-24 → 2018-01-01 (Amatrice, Visso, Norcia, Campotosto); era B is the whole of 2009
(L'Aquila); 29 of 293 GNSS stations are held out. Three training splits are cut from the snapshot and
written to disk: `pretrain` (everything except the eras and the held-out stations), `cb02_fit`
(everything before 2016-08-24) and `cb03_fit` (everything before 2009). The forecast tasks fit only on
the past of their test window, and their forecasts may read quakes inside the window day by day, as a
real forecaster would.

**Why.** A forecast fitted on data from after its test window has seen the answer's aftermath. Fitting
only on the past makes CB-02 and CB-03 pseudo-prospective, the strongest test possible without waiting
for the future. Writing the splits to disk, instead of filtering on the fly, gives the leak check real
files to read back.

**Rejected.** One split for everything (a model trained on 2018-2026 would know how 2016's aftershocks
ended). Excluding 2009 from the CB-02 fit (L'Aquila is the best lesson the past offers about 2016).

### D-030 · 2026-10-20 · P3 — A GNSS station is one monument only if it never jumps more than 1 m in a day

Held-out stations are chosen by a salted hash among stations with at least 730 days, three visible
neighbours within 50 km, and no jump over 1 m between consecutive daily solutions. The CB-04 baseline
uses only such stations as neighbours.

**Why.** The Nevada Geodetic Laboratory reuses a few station codes for sites metres to hundreds of km
apart: MIR2 moves 595 km north overnight, PES4 2.3 km, TOD3 3.8 m. The largest real jump in the region
is ARQT's 0.447 m in the Norcia earthquake, so 1 m separates the two cleanly. A span limit doesn't
work: plate motion alone moves a 30-year station about 0.75 m.

**Rejected.** A span limit of 5 m (let TOD3 through; it sits 0.6 km from held-out RETO and wrecked its
rebuild). Fixing the codes in the adapter (data is immutable; a corrected NGL adapter can come later as
a new version and a new snapshot).

### D-031 · 2026-10-20 · P3 — CB-01 hides where and how big, and scores bits on fixed bins

For a masked quake the model sees its time and magnitude type; latitude, longitude, depth and magnitude
are hidden. For a masked GNSS day it sees the station and its site to 0.01°; east, north and up are
hidden. Each channel's answer is a distribution, scored as −log2 of the probability it gives the true
value's bin (0.01°, 1 km, 0.1 magnitude, 1 mm), floored at 2^-64. The score is the mean over
instruments of the mean bits per token. 15% of the era tokens are masked, chosen by a salted hash. The
tide token computed at a masked quake is hidden as well. Tide tokens are never scored.

**Why.** A GNSS token's latitude and longitude are that day's own position to 1e-10°, so hiding only
the value would leave the answer in the coordinates. A tide token at a quake sits at the quake's
epicentre and origin time. Bins turn a continuous answer into bits, the same unit as an LLM's
perplexity, and fixed bins make every model comparable. Tides are computed from the Sun and the Moon;
being surprised by them says nothing about the Earth. The mean over instruments stops the more numerous
one from deciding the score.

**Rejected.** Masking values only (leaks). Differential entropy (depends on units; no floor). Scoring
tides.

### D-032 · 2026-10-20 · P3 — CB-02 and CB-03 are scored per M3+ quake, CSEP-style

Daily forecasts on 0.1° cells and 0.1 magnitude bins from M1.5 up. The score is the joint Poisson
log-likelihood of the observed M3+ quakes divided by their number, in bits: the difference between two
models is their information gain per earthquake (Rhoades et al. 2011). The same number at M1.5+, the
daily N-test, and the S- and M-tests on the summed forecast are kept as extras. CB-02's first day is
2016-08-25: Amatrice's first 22 hours are history. Magnitudes are the catalogue's preferred one, of
whatever type (ML, Md or Mw), and the task card says so.

**Why.** Log-likelihood is a proper score and the CSEP standard. M3 is where the catalogue stays
complete even in the days after a main shock; M1.5 is where small-quake information lives and where a
neural model may win (Stockman et al. 2023), so it is tracked, not ratcheted. Starting a day after
Amatrice keeps one unforecastable day from deciding the whole score. INGV still reports some Md after
2005 (291 of the 42,807 test quakes in CB-02); dropping them would punch holes in the catalogue, so they
are kept and counted.

**Rejected.** A fixed ETAS reference baked into the metric (it would tie the task to one baseline
forever). Two task IDs for the two thresholds.

### D-033 · 2026-10-20 · P3 — ETAS and the CSEP tests are Craton's own code, checked against pyCSEP

`craton/bench/etas.py` and `craton/bench/csep.py` are NumPy and SciPy. pyCSEP 0.8.0 (BSD-3) runs in a
throwaway environment in `scripts/reference/csep_reference.py`; the tests match its joint
log-likelihood and N-test to machine precision and its S- and M-test quantiles within simulation noise.
ETAS is tested on synthetic catalogues it generated itself: the fit recovers the planted parameters, and
on 500 synthetic days the daily forecast expected 936 quakes where 929 happened.

**Why.** pyCSEP installs on Windows but brings cartopy, rasterio and obspy, for four statistics. The
best-known Python ETAS (lmizrahi/etas, MIT) pins its own stack and pulls a model package from a GitLab
commit. Both are better as referees than as dependencies, the same choice as the tides (D-021).

### D-034 · 2026-10-20 · P3 — The ETAS baseline

Space-time ETAS with a power-law spatial kernel cut at 50 km and renormalised. Fitted on quakes at or
above M2.0 from 2005 onwards by exact maximum likelihood: PyTorch float64 on the CPU for the gradient,
SciPy's L-BFGS-B for the steps, three rounds of stochastic declustering for the background (Zhuang et
al. 2002). Forecasts integrate the intensity over each day and each cell; quakes triggered inside the
day are added by a mean-field cascade factor; magnitudes follow Gutenberg-Richter, extrapolated down to
M1.5. The three seeds are three restarts from jittered starting points.

**Why.** The standard for short-term forecasts in Italy is ETAS, and a baseline that is easy to beat
proves nothing. M2.0 keeps the fit above the catalogue's completeness, which rises after big shocks.
Simulating thousands of catalogues per day is the textbook forecast; the cascade factor gets the same
expected counts at a fraction of the cost and is checked on synthetic data.

**Known limits.** The spatial integral of each kernel is taken as 1 when fitting, so quakes near the
box edge look less productive; on synthetic data the branching ratio comes out about 0.1 low. CB-03's
fit has only four years, mostly quiet, and lands supercritical (branching ratio 1.26). Both are part of
the floor a model has to beat.

### D-035 · 2026-10-20 · P3 — CB-04 is an RMSE in millimetres, offset removed

Every day of every held-out station is rebuilt; the score is the RMSE over all station-days and the
three components, after removing each station's mean residual per component. The baseline weights the
six nearest stable stations within 100 km by 1/d²: their velocity, plus their de-trended position that
day.

**Why.** Each station's series starts from its own arbitrary origin; no method can know it, so it must
not count.

### D-036 · 2026-10-20 · P3 — The scoreboard has two lanes: Craton and the floor

A run comes from a `model` or a `baseline`. Craton's runs are ratcheted against Craton's champion; each
baseline is ratcheted against its own earlier score. The scoreboard shows the best baseline on each task
as the **floor** next to Craton's number, and the gap between them.

**Why.** With one lane, the first model that loses to ETAS could never be scored, and the constitution
says it must be, plainly. With two, the ratchet still stops any lane from getting worse, and the floor
is always in view.

**Rejected.** Letting baselines be Craton's champions (a losing model would be blocked forever).

### D-037 · 2026-10-20 · P3 — SciPy joins the dependencies

**Why.** The Poisson distribution, the normal CDF and L-BFGS-B, from a library with wheels for every
platform. Writing these by hand would be slower and less trustworthy.

### D-038 · 2026-10-20 · P3 — The neural point process baseline waits for P4

The plan names the neural point process of Stockman, Lawson & Werner (arXiv 2301.09948) as a CB-02
baseline. It is a neural model (P3 has none), it is temporal only, and its headline result uses an
enhanced catalogue of about 900,000 events (Tan et al. 2021) that Craton doesn't have. Its paper is the
reference P4 compares with.

### D-039 · 2026-10-20 · P4 — Craton-mini v0: one trunk, two heads, trained on the score itself

The mini config's shape (d = 256, 6 layers, 8 heads, 1,024 tokens, 4.74M trunk parameters), pre-norm
GPT-2 blocks on PyTorch's fused attention, one encoder for catalogue quakes, and two heads. The masked
head gives a probability for every CB-01 bin of latitude, longitude, depth and magnitude; the next-day
head gives a Poisson rate for tomorrow, a probability per 0.1° cell and a probability per 0.1 magnitude
bin. Each bin keeps 0.1% of the probability spread evenly (no answer costs more than about 20 bits).
The masked loss is CB-01's bits on those bins; the next-day loss is CB-02's joint Poisson log-likelihood,
log n! included. Objectives and their weights come from the config.

**Why.** The plan asks for probabilistic heads only (binned or mixture). Binned heads on the benchmark's
own bins turn the training loss into the benchmark's score, the same move as training an LLM on the
perplexity it is judged by.

**Rejected.** Mixture density heads (they would be scored on bins anyway, and a sum of Gaussians wastes
mass outside the region). Regression heads (a single number; banned).

### D-040 · 2026-10-20 · P4 — Next-event prediction is a next-day point process

At every midnight, a query token reads the 1,023 quakes before it and the head forecasts the next 24
hours: rate × where × how big. That is the point process discretised to the day and the 0.1° cell,
which is exactly the question CB-02 asks.

**Why.** A continuous-time next-event head would need thousands of simulated futures per forecast day to
produce daily expected counts, and some days in 2016 had over a thousand quakes. The discretised head
answers in one forward pass and trains on the score itself.

**Rejected.** A continuous-time neural point process with simulation (left for later; the
EarthquakeNPP benchmark, Stockman et al. 2026, found none of five such models beat ETAS on California).

### D-041 · 2026-10-20 · P4 — One model per split, and what each task lets the model read

Craton-mini trains one model per split and seed: on `pretrain` for CB-01 (masked objective only: the
split has holes where the eras were, so its midnights would ask about missing days), on `cb02_fit` and
`cb03_fit` for CB-02 and CB-03 (both objectives). For CB-01 every snapshot quake except the hidden ones is
visible; the masked quakes go back in at their times with place and size hidden, and windows of 1,024
are read around them. For CB-02 and CB-03, each midnight reads the quakes before it.

**Why.** D-029: a forecast model must never learn from after its test window, and `pretrain` contains
2018-2026, the tail of the 2016 sequence. CB-01's definition hides only the masked tokens and their tides;
the other 85% of the eras is context, the way BERT reads the words around a masked one.

**Rejected.** One model for everything (it would have seen how 2016's aftershocks ended).

### D-042 · 2026-10-20 · P4 — CB-01's GNSS queries get the climatology answer in v0

Craton-mini v0 reads the catalogue only (the plan keeps GNSS and tides for P5). Its CB-01 answer for a
GNSS day is the climatology baseline's, and the run's extras report the two instruments separately.

**Why.** CB-01 is the mean over instruments, and every query needs an answer. Copying the floor for the
half the model can't read yet keeps the headline honest: any gain comes from the quakes alone.

### D-043 · 2026-10-20 · P4 — CB-15 is model FLOPs utilisation, inside 6 GiB

CB-15's score is the MFU of the CB-01 model's training run, per seed: tokens per second × (6N +
12·L·H·Q·T) FLOPs per token (PaLM, appendix B) ÷ 25.8 TFLOP/s, the RTX 2060's fp16 tensor-core peak with
fp32 accumulation (240 tensor cores × 64 FMA × 2 × 1.68 GHz, halved for fp32 accumulation on GeForce
Turing). A run whose peak VRAM exceeds 6 GiB cannot be scored. Tokens per second, peak VRAM and minutes
are extras.

**Why.** The ratchet never lets a task get worse and never lets a model drop one. With peak VRAM or
tokens per second as the score, the first bigger model would be blocked forever. MFU doesn't fall just
because a model grows; the budget is a hard gate, as the plan says ("must fit in 6 GB").

**Rejected.** Peak VRAM, tokens per second, a pass/fail flag (says nothing).

### D-044 · 2026-10-20 · P4 — A frozen benchmark grows; it never changes

`craton bench prepare --freeze` on an existing record now adds the tasks it doesn't have yet, and only
if everything already frozen is rebuilt identically. CB-15 entered `frozen/central_italy.json` this way.

**Why.** Tasks arrive phase by phase. Re-freezing would let an old task change silently; refusing to
grow would stop every phase after P3.

### D-045 · 2026-10-20 · P4 — The model gets a clock, and the rate head a straight path from the recent rate

Every token carries two clock numbers, log seconds since the window's reference moment and log seconds
since the previous quake. The next-day head adds, to its trunk-driven log rate, a straight line from six
recent rates measured just before midnight: k over the age of the k-th last quake for k = 1, 10, 100,
1,000, and the same for M3+ quakes at k = 1 and 10. The line starts at the average of the four
all-sizes rates ("tomorrow looks like now") and is not weight-decayed.

**Why.** The first runs forecast 56 to 63 M3+ quakes for CB-02 where 1,065 happened, no better than the
Poisson baseline. Attention averages, so it can't count how crowded its window is, and the trunk's final
layer norm squashes any sense of scale; only 13 training days had more than 200 quakes, and 2016 was
busier than any of them. With the straight path the same training gave −11.03 bits per quake instead of
−14.26. A line in log space extrapolates like a power law, the shape of Omori's law. The M3+ rates carry
ETAS's lesson that big quakes trigger more.

**Rejected.** Oversampling busy days (biases the per-quake objective). Counting tokens inside attention
(it can't). Waiting for synthetic catalogues (P7) before the model could react to a sequence at all.

### D-046 · 2026-10-20 · P4 — Bitwise reproducibility gives way to the fast attention kernel

The memory-efficient attention kernel PyTorch picks on the RTX 2060 has a non-deterministic backward
pass. Craton keeps it: seeds fix initialisation, data order and masks, and two runs of one seed agree to
noise, not to the bit. Every score is a mean over three seeds, which is the noise the ratchet measures.

**Why.** The deterministic fallback (the plain math kernel) is several times slower and uses more memory.

### D-047 · 2026-10-20 · P4 — The next-day head learns from every midnight since 1986

Next-day training uses every midnight from 1 January 1986 to the end of the split with a full context,
all magnitude types, while the masked head learns from the whole split. 10% of 30-day blocks, picked by a
salted hash, are kept for validation and early stopping.

**Why.** The ETAS baseline fits from 2005 because it can't tell Md from ML. The model sees each quake's
magnitude type, and before 2005 lies the 1997 Colfiorito sequence: one of only three big sequences in the
data, and the only one CB-03's model can learn from.

### D-048 · 2026-10-20 · P5 — One model, an encoder per instrument, dispatched by id

Every token carries its instrument's registry id (0 for a query, -1 for padding), up to 12 numbers and a
label. Each instrument the config lists (`model.inputs`) has its own encoder and a learned modality
vector; all encoders run over every token and each token keeps its own instrument's answer. The trunk
takes an optional padding mask. An id is a place in an add-only registry, so it never moves.

**Why.** The plan's step 1 (per-instrument encoders into one space) without touching the trunk. Running
the small encoders densely costs a few percent of arithmetic and saves the GPU syncs that picking tokens
out by boolean masks needs; with them, a training step was 30 ms slower.

**Rejected.** One encoder for everything (a GNSS day and a quake share no channels). Mixture-of-experts
trunk blocks (the plan's optional step 2): with three instruments and a 4.7M trunk there is nothing to
route yet; left for later behind a flag.

### D-049 · 2026-10-20 · P5 — A GNSS day is read as its distance from the station's line

Each station gets a line, fitted on the training split: its position in 2015 and its velocity. A day
enters as its distance from the line, then centred on the mean of the station's visible days in the
window (that level is a channel of its own). A hidden day is answered as a distance from line + level,
with a mixture of four logistic distributions in millimetres sharing one straight ramp in time, scored
on CB-01's 1 mm bins (the training loss is the benchmark's score, D-039). A station with too little
history, or none (CB-04's held-out stations), borrows its velocity from its six nearest neighbours,
weighted by 1/d², and has no level.

**Why.** Positions are millions of millimetres from an arbitrary origin and move a few centimetres a
year; the network needs what changes, not where the origin is. Centring a series on its own window is
reversible instance normalisation (Kim et al. 2022), and it keeps a station that jumped (an earthquake,
a new antenna, a reused code, D-030) readable. The ramp lets the model correct a borrowed velocity.

**Rejected.** Raw positions (meaningless scale). A Gaussian head (thin tails: one jump costs hundreds
of bits). Fitting held-out stations' lines from their own days (those are CB-04's answers).

### D-050 · 2026-10-20 · P5 — The context policy

A window of 1,024 is filled in order: the hourly tide nearest the anchor (25 hours; a next-day window
reads 12 hours before midnight and 24 after), GNSS days of the 12 nearest stations within 100 km (±7
days; 160 days at most around a GNSS day, 48 around a quake; the 5 days before a midnight), then quakes
nearest in time, and in a next-day window the 32 largest M3.5+ of the past year even when older.

**Why.** The plan asks for a documented policy over recency, distance and magnitude. The tide of the
forecast day is computed from the Sun and the Moon and known in advance, so reading it is not peeking.
A window about a quake keeps ~1,000 quakes because a hidden quake is found from other quakes; see
D-053.

### D-051 · 2026-10-20 · P5 — Cross-modal masking, both ways

A GNSS-anchored window is played in one of four modes: 15% of everything hidden (CB-01's game), one
station hidden whole with its line borrowed (CB-04's game), every GNSS day hidden (predict the ground
from quakes and tides), every quake hidden (the other way).

**Why.** Step 4 of the plan. The whole-station mode makes CB-04 a game the model practises instead of
one it is asked cold.

### D-052 · 2026-10-20 · P5 — CB-17 is a region left out of one split, at every time

CB-17 holds out the Alto Tiberina box (43.0-43.6 N, 12.0-12.6 E) from a new split, `cb17_fit`
(pretrain minus every token inside the box). Its model is shown the box's first N days of quakes from 1
January 2022 (N = 0, 7, 30, 90, 365; 0 to 874 quakes) and asked for every quake of 2023 (1,307, the
Umbertide M4.5 among them), in bits on CB-01's bins. The score is the bits at N = 365; the curve is in
the extras. Floors: climatology from `cb17_fit` (ignores the context) and `context_kde`, a kernel
density of the context itself. A frozen record now grows by splits as well as tasks (D-044).

**Why.** "A new region" taken literally: a model that has never seen a place. The KDE floor asks the
honest question: does the transformer learn more from the context than counting does?

**Rejected.** A region outside the box (the heads' bins end at its edges). GNSS stations as "new
regions" (an offset nobody can know makes the N = 0 point meaningless).

### D-053 · 2026-10-20 · P5 — Each game is trained where it is tested; fusion may not hide a loss

The first fused model (seed 0) scored CB-01 16.05 against P4's 21.66, but its hidden-quake half got
worse: 22.47 bits against P4's 20.49. Two causes. The in-context region windows hide ~800 quakes each
and took half of all the quake practice; and the windows held 839 quakes instead of 1,023. Region
windows now train only the CB-17 model (`train.split_windows`), a window about a quake reads at most 48
GNSS days, and runs take up to 4,000 steps (P4's were still improving at 2,500). Without region
windows the quake half was already back to 21.11. P5 ships only if each half of CB-01 matches or beats
P4, not just the mean.

**Why.** Attention chooses among the tokens in a window; it can't use tokens that aren't there or
learn a game it barely plays. The data mix decides what a model is good at, as in every LLM.

**Rejected.** Letting the ratchet decide alone: it compares CB-01's mean, where 13 bits of GNSS gain
would hide a 2-bit quake loss.

### D-054 · 2026-10-20 · P5 — CB-04's bar is the floor

P5's exit asks to beat P4 on CB-04, but P4 answered no GNSS question (D-042). The bar is the floor, IDW +
trend, 17.77 mm. 67% of that error's square is one station, VCRA, whose series jumps by up to 315 mm in
a day (2017) and 231 mm in 2018: monument or processing changes no neighbour can see.

### D-055 · 2026-10-20 · P5 — Batches are cut in a worker process

Training batches are cut by one worker process (a seeded stream, same order as a single loop), pinned,
and copied without blocking; the optimiser is fused AdamW. Hidden tokens are picked by positions the CPU
computed, and the next-day loss takes log n! from the CPU.

**Why.** Fused windows take ~40 ms of Python to cut. On a thread they fought the training loop for the
interpreter lock, and blocking copies left the GPU idle while the CPU caught up: the first fused runs
reached 8.9% MFU against P4's 15.6% (CB-15 would have regressed). Now 16.6%.

### D-056 · 2026-10-21 · P5 — Read everything, practise the quake game; a negative result kept

Seed-0 tests on the `pretrain` model, CB-01's hidden-quake half (P4: 20.49 bits):

| Masked windows | hidden quake, bits |
|---|---|
| quakes only, quake windows (the rebuilt P4) | 20.54 |
| fused, 55% quake / 45% GNSS-centred windows | 21.68 |
| fused, 85% / 15% | 21.51 |
| fused, 85% / 15%, each window weighted as one example | 22.14 (stopped at step 700) |
| **fused, 100% quake windows** | **20.17** |

Reading GNSS and tides makes the model better at quakes (20.17 against 20.54 without them). Playing
GNSS-centred games (a station hidden whole, all GNSS hidden, all quakes hidden) costs the quake half
1.3 bits or more at this size, at every dose tried, and buys nothing measurable on CB-04 (17.52-17.67
mm with or without them). The general models train on quake windows only; those still hide 15% of
their GNSS days, which the model fills in from quakes, tides and the other stations. The CB-17 model
adds region windows (15%), the game it is tested on. The GNSS-centred modes stay in the code and the
tests, off by config (`train.windows`), to be tried again on a bigger model (P12).

Also fixed on the way: the rebuilt clock counted each token's age from the middle of its window, so
an hour before and an hour after looked the same, and in a fused window from a GNSS day a week
earlier, which squashed every quake's age together. It counts from the window's first quake again,
as in P4 (quakes only: 20.78 → 20.54). Each hidden token now weighs 1/(hidden tokens in its window).
Patience 8: one run stopped at step 700 on validation noise.

**Rejected.** Shipping the 55/45 model: its CB-01 (15.53) beats P4's 21.66 by six bits, and the ratchet
would have taken it, but only because the GNSS gain hid a quake loss (D-053).

### D-057 · 2026-10-21 · P5 — The trunk may nudge tomorrow's rate at most 2x

The next-day log rate is the measured-rate path (D-045) plus the trunk's correction, now bounded
smoothly: ln 2 · tanh(c / ln 2). The trunk can move the forecast at most 2x either way.

**Why.** CB-03's model learns from data up to 2008, which holds one big sequence. With a free
correction its three seeds expected 287, 1,412 and 872 M3+ quakes for 2009, where 315 happened
(−18.14 ± 4.01 bits; P4 −15.13 ± 0.79). The ratchet called it "same" only because the spread was so
wide. A crisis bigger than any in training is exactly where a learned correction has nothing to stand
on and the Omori-shaped path does. Capped at 4x: seed 1 −16.61 (876 expected). At 2x: −13.78, −14.39,
−13.98 (225, 273, 210 expected): −14.05 ± 0.31.

**Rejected.** Early stopping on the forecast alone (validation kept improving to step 4,000: the
validation blocks are quiet months, the test is a crisis). Fewer steps (blind to the cause).

**Note.** The ablation (`docs/ablation.md`) was measured with the uncapped head (training v0.2). Its
quake, GNSS and CB-04 rows don't touch the rate head; its CB-02 row compares variants that all had
the uncapped head, so the comparison holds, but its absolute CB-02 numbers are not the scoreboard's.

### D-058 · 2026-10-22 · P6 — Every forecast score comes split into how many, where and how big

CB-02 and CB-03 report, as extras, the score per quake split exactly into four parts: how many
(each day's Poisson total), where (the share of the day's rate in the quake's cell), how big (the
share of the cell's rate in its magnitude bin) and a shuffle term (log N! − Σ log n!, the same for
every forecast). They add up to the score to rounding. Also the score without the days that hold
an M5+ quake. Extras only: the tasks' test data, metric and fingerprints do not move.

**Why.** The P5 loss was 4.7 bits behind ETAS on CB-02, and only splitting it showed 3.6 of them
were *where*. A phase that says "beat ETAS" needs to say on which part.

**Rejected.** A new task ID for the split (nothing about the test changes).

### D-059 · 2026-10-22 · P6 — The building: oracle floors above ETAS, measured once

`craton bench ceiling` writes `results/ceiling.json`: ETAS's own forecast with one thing replaced
by the truth: each day's count (+2.10 bits on CB-02, +3.55 on CB-03), the day's map to ~10 km
(+0.60, +0.66) or to the exact cell (+2.11, +2.45), and the roof, count and cell together (+4.21,
+6.00). The scoreboard draws every forecast on that building.

**Why.** "Above ETAS" needs a scale. The oracles read the answers, so no method can reach them;
they say how much room there is and where it is (most of the count room sits on the three M5+
days).

### D-060 · 2026-10-22 · P6 — Three stronger floors, and two of them barely move

- `interpolation` (CB-01): a hidden GNSS day is the line between the same station's nearest
  visible days, two logistics wide (widths per station from leave-one-out on the training split).
  GNSS half: **9.65 bits**. Craton P5 scores 9.75 there: Craton's GNSS skill is no better than
  drawing a line between neighbouring days. Kept on the board as the honest bar.
- `etas_aniso` (CB-02/03): every M4+ quake's zone becomes an ellipse along its first 6 hours of
  aftershocks (principal axes, stretch up to 6x, centred on their centroid; rupture radius from
  Wells and Coppersmith 1994). 85 quakes get shapes, aligned NW-SE along the Apennines.
  CB-02 −7.119, CB-03 −9.401: **nothing**. ETAS's M6 kernel is ~3 km wide, and the thousands of
  small aftershocks, each a parent of its own, already draw the fault.
- `etas_incomplete` (CB-02/03): the blind-time model of Hainzl (2016, 2022), simplified: ETAS keeps
  its fit, and a blind time Tb plus a productivity factor κ are chosen by grid search on the
  training days' counts per magnitude bin. Detection inside each day is computed minute by minute.
  Both tasks pick Tb = 0 and κ = 1.1: CB-02 −7.059, CB-03 −9.295. Daily counts can't see the
  blindness; the gain is the productivity factor.

**Rejected.** A full Hainzl refit with Tb inside the likelihood (a much larger change to the ETAS
fit; the simplified floor already says what daily data can tell). Faults from DISS for the
stretched zones (an adapter and a data card for a floor that the data says adds nothing; the
catalogue draws its own faults).

### D-061 · 2026-10-22 · P6 — The Field: Omori memory, cell tokens, ETAS as the starting line

CB-02 and CB-03 are now forecast by the Field (`craton/field`), not the trunk's next-day head:

- **Omori memory.** Each 0.1° cell keeps fading counters of the quakes in it, lifetimes 1 hour →
  12 years (9, log-spaced), counting every quake down to M1.5, weighted by 1 or by 10^(m−2), seen
  at three spreads (the cell, ~10 km, ~30 km): 54 numbers per cell per midnight. A power law is a
  sum of exponentials, so ETAS's triggering is one fixed recipe over these counters: a diagonal
  state-space layer with fixed decays.
- **Cell tokens.** The 1,050 cells are the tokens; two attention layers let each read the others.
- **ETAS start.** Out of each cell come two bounded corrections to ETAS's log rate (M2-3, M3+), at
  most 4x either way; the output layer starts at zero, so the untrained Field is ETAS exactly.
- **Small quakes in, big quakes scored.** Memory counts M1.5+, the loss learns from M2+ (the
  benchmark's Poisson score), only M3+ is forecast.

The spike, seed 0 → 2: the flip alone (no place context) takes CB-02 from −10.98 to
−7.02 ± 0.13 and CB-03 from −14.05 to −8.72 ± 0.14, ETAS −7.12 and −9.40. All of the gain is how
many; where ties ETAS. Without attention between cells, where loses 0.36 bits on CB-03; a single
correction for the whole day gains less count than one per cell.

**Rejected.** The P5 trunk's vector in every cell for now (the spike beat ETAS without it; it
stays the next thing to try). An ETAS with M1.5 parents as the base (CB-03 −9.05, all count, its
fit hits a parameter bound, and it runs out of memory on CB-02).

### D-062 · 2026-10-22 · P6 — A forecast is the average of four Fields

Each seed's forecast is the mean rate of four Fields (member seeds 100·s + m). One Field per seed
swung CB-02 between −6.87 and −7.10 (566 to 877 quakes expected): the ratchet's noise would have
been ±0.26. Four members: −7.019, −7.014, −7.044 (± 0.016).

**Why.** A deep ensemble: the members disagree where the training data is thin, and averaging
their rates is a better forecast than any one of them.

### D-063 · 2026-10-22 · P6 — The Field has its own config and checkpoints; ETAS runs on the GPU

`configs/field/mini.yaml`, its own hash, its own checkpoints (`results/checkpoints/field-...`),
so a Field change never retrains the trunk's twelve. `craton-mini`'s CB-01, CB-04 and CB-17 come
from the same trunk checkpoints as P5. ETAS's daily rates for the Field's ~4,600 training days run
on the GPU (`craton/field/base.py`), equal to the CPU code to 5·10⁻¹², about a hundred times
faster. The `etas` baseline keeps its CPU path, so its frozen numbers cannot move.

### D-064 · 2026-10-22 · P6 — The Field learns from 1991; the *where* exit test is missed, and kept

The Field's training days start on 1 January 1991, not 2005: it sees Colfiorito 1997 (in the older Md
magnitudes) as well as L'Aquila 2009, with ETAS's 2005 fit as the starting line throughout. Seed 0,
four members: CB-02 −7.02 → −6.85, CB-03 −8.61 → −8.25, and CB-03's *where* went from a tie to
+0.12 over ETAS.

Final, 3 seeds: CB-02 −6.818 ± 0.016, CB-03 −8.233 ± 0.006. *Where* against ETAS: CB-03 +0.15,
CB-02 −0.018. The CB-02 shortfall is all on two days, 30 October 2016 (Norcia, −38.8 bits) and
18 January 2017 (Campotosto, −18.2 bits), when a new mainshock struck away from the swarm the Field
was leaning on; on every other day the Field's map is ahead (+0.02 bits per quake). P6 ships with
this exit test unmet, said here, in the logbook and in the lesson.

**Why.** More sequences are the cheapest teacher there is. The rest of the plan (above ETAS beyond
noise on both, above the stronger floors) is met, and blending ETAS's map back in only made CB-02's
*where* worse at every mix (checked, not tuned on the test). Seeing a mainshock coming is the
mainshock-day work later in the plan, not something a map of recent activity can do.

**Rejected.** Mixing ETAS's map into the Field's by a weight chosen on the test (that is fitting the
answer). Holding the release until *where* ties on CB-02 (no honest lever left inside this phase).

### D-065 · 2026-10-22 · P7 — The trust tasks read the forecasts CB-02 and CB-03 already make

CB-05 (calibration), CB-07 (fake precursors), CB-08 (physics) and CB-11 (the meter) are scored on
the same two test windows and the same frozen answers as CB-02 and CB-03; their fingerprints
include those files. A forecaster may say more than its rates through one optional protocol,
`predictive` (`craton.trust.predictive`): every ensemble member's rates, the negative binomial
dispersion of a day's total and of one cell's count, and days it abstains on. A plain forecaster is
read as Poisson with one member, so every floor gets a score on every trust task with no extra
code. Each (method, task, seed) forecast is computed once and shared (`predictive_of`).

**Rejected.** New test windows for the trust tasks (they would split the evidence; the two
sequences are the hardest days the region has). A separate calibration model beside the Field (the
dispersion fitted post hoc on validation days does the job without a second network).

### D-066 · 2026-10-22 · P7 — CB-05 scores calibration as a factor, not a gap

For rare events the expected calibration error is small even when a forecast is badly off: a
forecast of 0.16% for cell-days that came 0.54% of the time is a gap of 0.004 and three times too
low. CB-05's number is the gap as a factor: |log₂(observed ÷ forecast)| per reliability group,
weighted by the group's size, over cell-days forecast at least 0.1% likely (observed share
smoothed as (hits + ½)/(n + 1)). 0 is perfect, 1 is "off by 2x". The ECE stays beside it, for
cell-days forecast at least 1% (the plan's exit test, threshold 0.05) and for all. The day totals
get a randomised PIT (KS distance from uniform), CRPS and 90%-interval hits.

Post-hoc calibration, fitted on the Field's validation days (one 30-day block in eight of the
training split, never the test), by maximum likelihood, stored as `calibration.json` beside each
checkpoint: the negative binomial size of a cell's count, and of a day's total as k = κ · mean^γ.
A single k for every day (γ = 0) came out at 0.56, which makes the spread grow like the mean
itself: far too wide inside a sequence (PIT distance 0.10 on CB-02, seed 0). The validation days
alone prefer γ ≈ 0.67 for CB-02's fits (log-likelihood −486.2 against −490.2), γ ≈ 0.1 for
CB-03's, whose validation years hold fewer busy days.

Result: Craton 0.32 ± 0.08 doublings off, behind ETAS (0.22) and ETAS-incomplete (0.10): Craton is
a little high on the middle groups (forecast 1.7%, observed 1.3%) and low on the top one (44%,
observed 51%). Its day totals are the best calibrated of any method (PIT distance 0.039; ETAS
0.061), its 90% intervals a little wide (97% of days inside). ECE on committed cell-days 0.012,
under the plan's 0.05.

**Rejected.** ECE alone as the number (it would rate a Poisson map that is 3x low as nearly
perfect). Isotonic recalibration of the cell rates (fitted on validation days that hold almost no
committed cell-days; the dispersion is one parameter and transfers).

### D-067 · 2026-10-22 · P7 — Six Fields, a magnitude ceiling, and the guard in front

- **Six members** per forecast (was four; the plan asks for five or more so the spread is an
  estimate, not an anecdote): P6's four kept, two new ones added (D-076).
- **No magnitude above 7.3.** The Field's Gutenberg-Richter shares stop at Mw 7.3: CPTI15's 7.1
  for Avezzano 1915, the largest quake on record in the box, plus 0.2. An open-ended law spreads
  rate up to M8, and on calm days that alone breaks the slip budget (D-070).
- **The guard** (D-072) cleans the test-time catalogue before the Field reads it.

Neither the ceiling nor the guard is in the checkpoints: they act at forecast time.

### D-068 · 2026-10-22 · P7 — CB-06: Calabria and Sicily, judged by kind, not by place

New region `southern_italy` (36.5-40 N, 12-17.5 E, INGV only): 61,086 quakes, snapshot
`6dd8efd2fbc23ce6`, never read by any training split. A context is the 1,024 quakes before a
midnight; three sets: training contexts (the bank, a midnight every 10 days of `pretrain`),
central Italy's held-out eras (every 3 days) and Calabria + Sicily (every 15 days). Score: the
AUROC of the out-of-distribution score, unseen vs in-region; the abstention threshold is the 95th
percentile of the training contexts' own scores.

Every context is moved so its median epicentre sits at the centre of central Italy before
anything reads it. The trunk sees absolute coordinates, so without this any detector scores a
meaningless 1.0 for noticing "elsewhere".

Craton's detector is the trunk's query vector (the one the forecast reads) and the fingerprint
of the very quakes it read (time span, depth and magnitude quantiles, spread), side by side,
nearest-neighbour distance (Sun et al. 2022). The trunk alone scores **0.38, 0.12, 0.28** over
the seeds, below a coin flip: it finds central Italy's own sequences stranger than Sicily, and it
cannot see depth (Calabria's slab quakes go to 500 km). Together: 0.997 ± 0.001, against the
fingerprint floor's 0.998: the trunk adds nothing yet. Abstains on 100% of Calabria and Sicily,
72% of central Italy's sequence days, 5% of training contexts. Floors: `no_detector` (0.5, the
plan's baseline) and `fingerprint_knn`.

**Rejected.** The trunk alone (it fails). Raw coordinates (a test of geography). Calibrating the
threshold on the unseen region (that would be reading the test).

### D-069 · 2026-10-22 · P7 — CB-07: five fakes before L'Aquila, three of them counted

Each seed plants five fakes into the inputs of the CB-03 window, inside a 30-day stretch of
January-February 2009, at a site at least 60 km from L'Aquila: a 5 cm step at the five nearest
GNSS stations; every quake listed twice; every magnitude 0.4 too high; small quakes grown so the
b-value falls to ~0.7; a 150-quake swarm. The first three carry no information about the Earth
and are **counted**; the b-value drop and the swarm are real-looking seismicity, where reacting
can be physics, so they are reported, not counted. An alarm is an expected M4+ rate of at least
0.001 in a cell and its eight neighbours (~30 km) on a day, about ten times the usual level; a
false alarm is an alarmed cell-day with no real M4+ in the same nine cells. Score: (false alarms
under attack + 1) ÷ (clean + 1), averaged over the counted three.

Measured first on ETAS: at 0.01 (the first choice) no fake moved any alarm, even though the fakes
raised ETAS's local rates up to 3x (duplicates), 114x (magnitude shift) and 2,800x (swarm):
outside a real sequence nothing reaches 1% a day, so the test had no teeth. The glitches are
network-wide (every quake in the box), because that is how they happen: two catalogues merged,
or a calibration change (INGV's own move from Md to ML in 2005 shifted the whole catalogue).

Result: Craton 1.002 ± 0.004 (the GNSS step and the duplicates exactly 1.000; the magnitude shift
1.006, all of it one seed the guard caught only in part), ETAS 1.002. The exit test (at or below
the clean rate) is missed by 0.2%. The swarm raises Craton's false alarms 2.6%, ETAS's 2%; the
b-value drop does nothing to either.

### D-070 · 2026-10-22 · P7 — CB-08's hard checks, and the slip budget in numbers

Hard violations, summed: non-finite or negative rates and shares; magnitude shares that grow with
magnitude (the open last bin excluded) or imply b outside 0.5-2 at M3; the **Omori probe** (the
real catalogue up to the end of Amatrice's first day, 25 Aug 2016, and of L'Aquila's, 7 Apr 2009,
then nothing: on each of the next 30 days the box's M3+ rate may not rise by more than 0.5%); and
the **slip budget**: on calm days (no M5+ in the box in the previous 30), the long-run seismic
moment the forecast implies may not exceed Kostrov's moment rate with every number generous:
2 · 33 GPa · 15 km · 400 km · 4 mm/yr = 1.58 × 10¹⁸ N·m/yr (extension 2-4 mm/yr from geodesy and
geology; dip 45°).

Result: Craton zero violations (calm-day moment 8.4 × 10¹⁷ on CB-02); every ETAS floor breaks the
budget on CB-02's calm days (1.66-1.81 × 10¹⁸), because its Gutenberg-Richter law runs to M8.
Nobody fails the Omori probe.

### D-071 · 2026-10-22 · P7 — CB-11: the meter is the share of space-time you watch

Rank every (day, 0.1° cell) of both test windows by the forecast's M4+ rate and watch from the
top until 90% of the real M4+ quakes (89 of them) are inside: the share watched is CB-11's number,
in %. Ties are broken against the forecast (a tied block is watched whole). Beside it: the days on
alert (ranking days by their total), the radius of the daily area holding 90% (coverage-matched,
the conformal-prediction trick: coverage guaranteed, only size compared), and the magnitude
width (central 90% given M4+).

Result: Craton **0.84 ± 0.06%**, ETAS 1.32% (a window 36% smaller), a map of the past 24%. The
gain is in time: 213 alert days a year against ETAS's 282; in space the two are level (radius
40 km against 38). Magnitude width 1.14 for everyone (all use Gutenberg-Richter).

**Rejected.** Each width at the forecast's own nominal 90% (it would reward an overconfident
forecast with a small window it doesn't cover; the nominal coverage is reported instead).

### D-072 · 2026-10-22 · P7 — The guard: two glitch rules, tested on 41 years of real data

On the live edge only (the forecast window and the 60 days before it): drop a quake if an earlier
one lies within 5 s, 3 km and 0.3 magnitude units; and per 5-day stretch, if the 25th, 50th and
75th magnitude percentiles all jumped by 0.2 or more against the previous year (median by 0.25 or
more), with 20+ quakes and no more than twice the usual count, and the jump is shared across the
map (two-thirds of the 0.5° blocks with 3+ quakes, at least three), take the median jump off the
stretch.

Real catalogue: 7 duplicates dropped before CB-02, 3 before CB-03, no shift in either window. Run
over the whole of 1987-2026 the shift rule fires on 24 stretches (once every two years or so);
each test alone fired about twice as often. The injected duplicates are caught completely; the
injected 0.4 shift on 39-100% of its quakes depending on the seed.

### D-073 · 2026-10-22 · P7 — `craton referee`, and the 2026-10-21 tidal claim replayed

Five tests (module docstring of `craton/trust/referee.py`): raw Schuster, declustered (Gardner-
Knopoff) p < 0.01, beat 99% of 200 fake clocks, beat 99% of 200 time-of-day shuffles, and a locked
prediction (effect hashed before a held-out catalogue is read; p < 0.05 there within 45°).

The replay, with the computed tide (phase of the vertical solid-Earth tide's analytic signal) on
the 2016-17 sequence: raw p = 10⁻⁴¹³. Declustered, 397 mainshocks, p = 10⁻³·³: passes. Locked on
2016-17 and tested on 2009: p = 10⁻³·⁹ at the same phase: passes. It is still false: slid by random
lags of 30 days to 10 years, the same tide "locks" as hard 71% of the time. The tide's phase is not
spread evenly over time, so any catalogue looks locked to it; the Schuster test assumes it is. That
control, shifted tides, is what the referee uses whenever the cycle is a computed series instead
of a clock. Fake planets: a clustered ETAS catalogue with day-night detection (raw p = 10⁻⁹³⁴) is
rejected on every test; a planted 12.42-hour signal passes all five. `results/referee/replay.json`.

### D-074 · 2026-10-22 · P7 — Locked forecasts, every night, from a frozen bundle

`craton live run` (GitHub Actions, `.github/workflows/live.yml`, 22:30 UTC): fetch the INGV
catalogue (cached per month, the last two refetched), run the frozen bundle for the next UTC day,
seal the file with SHA-256 and append to the ledger, grade every locked day at least three days
old, regenerate `forecasts/README.md`, push (~33 KB a day). The bundle (`craton live bundle`): CB-02's Field
members and ETAS fits for all three seeds (18 members), their calibration and the fingerprint bank,
attached to the release as `craton-live-central_italy-v0.8.0.tar.gz`. Graded per 0.1° cell at M3+,
Craton against the ETAS floor locked in the same file. The forecast reads the catalogue as
published at issue time, so the last ~1.5 h before midnight are unseen. Missed days stay missed. A
locked file or grade can never be edited (`craton check`, D-008's spirit).

Live abstention uses the fingerprint half of the detector only: the trunk adds nothing measurable
on CB-06 and would add three 20 MB checkpoints to the bundle.

**Rejected.** Running the live job on the laptop (it isn't always on). Training on CI (the Field
needs the GPU and the full data). Backfilling a missed day (a backfilled forecast is not a locked
one).

### D-075 · 2026-10-22 · P7 — A simulator registry, with ETAS as the first planet

`craton/sim`: `SIMULATORS`, a `Simulator` protocol (`simulate(SimSpec, seed) -> Events`) and the
ETAS planet. Synthetic quakes carry the magnitude type `sim` and token hashes naming the simulator
and seed, so nothing synthetic can pass for real data. The referee's replays use it; P8 adds
physics-based planets beside it.

### D-076 · 2026-10-22 · P7 — An ensemble grows; it is never retrained

The first P7 run retrained all six Fields per seed from scratch. CB-02 came out −6.825 ± 0.006 and
CB-03 −8.236 ± 0.011, a few thousandths of a bit under P6 (−6.818, −8.233): within noise, but
under the bar that no phase scores below the last. Attributed seed by seed: the guard changes
nothing (±0.0006 bits) and neither does the magnitude ceiling; retraining the same four member
seeds on the GPU alone moves a seed by up to 0.03 bits. Throwing away the shipped Fields was the
mistake: it is not constructive.

So the Field grows: when a config asks for more members than an existing checkpoint of the same
config has, `FieldForecaster.states` loads the smaller ensemble and trains only the new members
(each member is trained alone from its own seed, so this is the same job). P7's ensembles are
P6's four members plus members four and five of the full run (`grown_from` in each checkpoint's
meta). Decided on the principle before scoring it. Result: CB-02 −6.815 ± 0.013, CB-03
−8.231 ± 0.006, both at or above P6.

**Rejected.** Keeping the full retrain (it breaks the constructive rule and the bar). Choosing
between the two by their test scores (that is fitting the answer; the principle decided).

### D-077 · 2026-10-22 · after P7 — Seventeen phases, and expected numbers in place of odds

P6's split of the loss to ETAS, and the oracle ceilings measured the same day, showed where the room
above ETAS is: 70–78% of what a perfect daily count would add sits on the three days with an M5+, and
a sharp *where* is worth more than any simple small-quake map. Three sequences in forty years cannot
teach either. So four phases go in before the discovery work: more planets to learn from, including
regions where the cause of the quakes is measured (P8); a hidden stress-and-pressure state graded
against focal mechanisms and CO₂ maps it never saw (P9); continuous noise as instruments (P10); and a
mainshock-day detector trained on the world's sequences (P11). The ghost, the translator, the
microscope and scale follow as P12–P16. P14 adds a symbolic law distilled from the Field, and P15 the
predictability horizon. Ten new tasks: CB-19 to CB-28.

The roadmap's table of odds is replaced by expected numbers per phase. A chance is not something a
phase can be checked against; a number is, and a miss gets written down.

**Rejected.** Starting the ghost and the interpreters right after P7: they would read a forecaster
that wins only on *how many*. Feeding the ML catalogues into CB-02 and CB-03: they were relocated
with hindsight, so they get their own tasks (CB-20, CB-21) and the frozen tasks stay pure.
