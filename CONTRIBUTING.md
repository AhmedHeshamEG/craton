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

The project is at phase P3 of P12 ([`ROADMAP.md`](ROADMAP.md)). The most helpful things today:

- **Data you know.** A public catalogue, GNSS network, borehole record or historical archive that should become
  tokens. Open an issue with the *new data source* template: what it is, its licence, how to fetch it.
- **Science checks.** If something in the roadmap is wrong or outdated (a method, a dataset, a claim), open an
  issue. The real world wins over the plan.
- **Benchmarks.** A test Craton should face, with frozen data and a metric.

- **Instruments.** The token spec only grows: a new instrument is a few lines and a new example. See
  [`SPEC_POLICY.md`](SPEC_POLICY.md) for how, and what can never change.

Adding a method to CratonBench is below; the guides to a force and a property arrive with their extension
point (P8); the complete version lands in P12.

## Add a data source

An adapter is one module in `craton/adapters/` and touches nothing else. The three built-ins
(`ingv.py`, `ngl.py`, `tides.py`) are the pattern; copy the one closest to your source.

```python
from craton.adapters import register
from craton.adapters.base import Adapter, BuildContext, DataCard, FetchReport


@register
class MyNetwork(Adapter):
    name = "my_network"  # snake_case, unique
    version = "1.0.0"  # written into every token it makes
    instruments = ("gnss_daily",)  # registered instruments only (add one first if needed)

    def card(self) -> DataCard: ...  # provider, licence, citation, mapping, known issues
    def fetch(self, region, raw, http, refresh, log) -> FetchReport: ...  # download into `raw`
    def tokens(self, ctx: BuildContext): ...  # yield one list of tokens per year
```

The rules that keep the data trustworthy:

- **Fetch politely, once.** Use the `http` you're given (it waits between requests and backs off) and store
  every response with `raw.put(...)`; skip labels already in `raw.current(...)` unless `refresh`.
- **Build offline.** `tokens()` reads only `ctx.use(raw_file)`, never the network, so a rebuild from the same
  downloads gives the same bytes.
- **Refuse, count, never guess.** A record that can't become a valid token goes to `ctx.reject("reason")`.
  Unknown categories are refused, not mapped to the nearest known one.
- **Say what you assumed.** `uncertainty_basis` is `reported`, `derived` or `assumed`, and the card explains
  every assumption.
- **The licence is what the provider wrote.** If they state none, use a `LicenseRef-` and quote their terms in
  the card.
- **Test against a tiny real excerpt** in `tests/fixtures/` (see `tests/test_adapters.py`), with a fake
  network.

Then add it to a region in `configs/regions/`, run `craton data cards` (the cards are generated) and
`craton data build <region>`. `craton check` refuses a card that drifted from its adapter.

## Pull requests

- One concern per PR, conventional commit messages (`feat:`, `fix:`, `docs:`, `test:` …).
- `craton check` green locally; CI runs the same command.
- New data comes with a data card (source, licence, coverage, known issues). New tasks come with their
  fingerprint and baseline.
- `LOGBOOK.md` and `DECISIONS.md` are append-only: add entries, never edit old ones.

## Add a method to CratonBench

A method is any object with a `name` that implements a task's protocol from `craton/bench/tasks.py`:

| Protocol | Tasks | Returns |
|---|---|---|
| `MaskedPredictor.predict_masked` | CB-01 | a distribution per hidden channel (anything with `mass(lo, hi)`) |
| `Forecaster.forecast` | CB-02, CB-03 | expected quakes per (day, cell), and how they split over magnitudes |
| `GapFiller.fill` | CB-04 | daily east, north, up of each hidden station |
| `EfficiencyReport.efficiency` | CB-15 | measurements of the method's own training run |

```python
from craton.bench.baselines import register_method


class MyForecaster:
    name = "my_forecaster"
    kind = "baseline"  # or "model" for a Craton model

    def forecast(self, spec, ctx, seed):
        train = ctx.train_events(spec.fit_split)  # only the split; never the snapshot
        ...


register_method(MyForecaster())
```

Fit only on `ctx.train_*` (the leak-checked split shards). A forecast for a day may read quakes before
that day from `ctx.events`, nothing later. Then `uv run craton bench run my_forecaster` writes a run to
`results/pending/`, and `uv run craton ratchet --accept` puts it on the board if nothing got worse. A new
*task* is `register_task(...)` with a new ID; existing IDs never change ([`docs/bench.md`](docs/bench.md)).

## Teach the model a new instrument

The model reads an instrument through an encoder registered under the instrument's name in
`craton/model/encoders.py`: a module that maps that instrument's channels to a `d_model` vector.

```python
from craton.model.encoders import EncoderSpec, register_encoder

register_encoder(EncoderSpec("gnss_daily", ("east", "north", "up"), MyGnssEncoder))
```

Names are unique and never redefined. Time and place enter through the shared 4D positional encoding,
so an encoder only has to understand its own measurements.
