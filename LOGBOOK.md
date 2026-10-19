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
