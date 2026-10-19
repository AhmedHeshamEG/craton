# Contributing

Craton is built like a building: the foundation is finished before the floors, and every floor adds without
touching what's underneath. That makes contributing safe. You can't break what's there, because the checks
won't let anyone.

## The four rules

1. **Add, don't edit.** New instruments, forces, properties, benchmarks and model heads plug in through
   registries. If your change needs an edit to the core, open an issue first; the extension point gets
   redesigned in its own change.
2. **Nothing gets worse.** A change lands only if no CratonBench task regresses beyond noise (2σ over 3 seeds).
   `craton ratchet` decides, not a reviewer's mood.
3. **Every number comes from a script.** No hand-typed scores, no mocked results, no skipped tests marked as
   passing. Negative results are welcome; they go on the scoreboard like everything else.
4. **Forecasts, never predictions.** Craton outputs probabilities and uncertainty windows. Please keep that
   language in code, docs and issues.

## Setup

```console
uv sync --extra cu126        # or --extra cpu
uv run craton doctor --smoke
uv run craton check          # must be green before you open a PR
```

Plain `uv sync` without an extra removes PyTorch; always pass one.

## What's useful right now

The project is at phase P1 of P12 ([`ROADMAP.md`](ROADMAP.md)). The most helpful things today:

- **Data you know.** A public catalogue, GNSS network, borehole record or historical archive that should become
  tokens. Open an issue with the *new data source* template: what it is, its licence, how to fetch it.
- **Science checks.** If something in the roadmap is wrong or outdated (a method, a dataset, a claim), open an
  issue. The real world wins over the plan.
- **Benchmarks.** A test Craton should face, with frozen data and a metric.

- **Instruments.** The token spec only grows: a new instrument is a few lines and a new example. See
  [`SPEC_POLICY.md`](SPEC_POLICY.md) for how, and what can never change.

The full guide to writing an adapter, a force, a property or a benchmark arrives with the extension points
themselves (P1, P2, P3, P8), and the complete version in P12.

## Pull requests

- One concern per PR, conventional commit messages (`feat:`, `fix:`, `docs:`, `test:` …).
- `craton check` green locally; CI runs the same command.
- New data comes with a data card (source, licence, coverage, known issues). New tasks come with their
  fingerprint and baseline.
- `LOGBOOK.md` and `DECISIONS.md` are append-only: add entries, never edit old ones.
