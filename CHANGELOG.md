# Changelog

One release per phase. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow D-009 (phase Pn ships as v0.(n+1).0).

## [0.1.0] — 2026-10-19 · P0 Foundation

### Added
- `craton doctor`: GPU, VRAM, compute capability, fp16 / bf16 support, the precision every config will
  actually run with on this machine; `--smoke` trains a tiny network in fp16 to prove it.
- `craton config <size>`: a model size with its parameter count (the 12·d² rule).
- `craton check`: lint, format, types, tests, scoreboard integrity, append-only records, ratchet.
- `craton ratchet [--accept]` and `craton scoreboard`.
- Configs: `mini` (active), `regional` and `planet` (planned).
- The CratonBench catalogue: 18 planned tasks, CB-01 … CB-18.
- README figures from real USGS data, `lessons/P0.md`, CI on Linux and Windows.
