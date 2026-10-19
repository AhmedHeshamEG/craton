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
