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
