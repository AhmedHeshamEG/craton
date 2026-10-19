# Spec policy

The token spec is the foundation every dataset, model and benchmark in Craton stands on. Data tokenised in
2026 has to stay valid in 2036. So the spec can only grow.

The spec itself: [`docs/token-spec.md`](docs/token-spec.md) (generated from `craton/spec/`), or
`craton spec` in a terminal.

## Versions

Token Spec uses [semantic versioning](https://semver.org), separately from the package version.

| Change | Version | Allowed? |
|---|---|---|
| A new instrument in the registry | minor (1.0 → 1.1) | yes |
| A new word in a closed vocabulary (a magnitude type, an intensity scale) | minor | yes |
| A new **optional** token field | minor | yes |
| A new unit code | minor | yes |
| Fixing a bug in validator code that never accepted a valid token | patch | yes |
| Removing or renaming anything | — | no |
| Changing a channel's unit, range, order or meaning | — | no: register a new instrument |
| Making an optional field required | — | no |
| Changing a frozen description | — | no: wording is part of the meaning |
| Anything else that would reject a token that used to pass | major (2.0) | only with the steps below |

## How "add-only" is enforced

1. Every spec release writes a snapshot, `craton/spec/frozen/<version>.json`: the token's field schema and
   every registered instrument. A snapshot is never edited or deleted, and `freeze()` refuses to overwrite one.
2. `craton check` compares the live code with **every** snapshot of the current major version. Anything frozen
   that was removed or redefined fails the check, so it can't be merged.
3. `docs/token-spec.md` is rendered from the code, and the check fails if it drifts.
4. Every token says which spec it was written under (`spec`). Code reads any token of its own major version
   up to its own minor version, and refuses newer ones with a clear message instead of guessing.

## A major version

Avoid it. If one is ever unavoidable, it needs, in this order:

1. a `DECISIONS.md` entry explaining why an addition can't do the job;
2. migration code that converts every 1.x token to 2.0, with tests on every frozen example;
3. every CratonBench task re-scored and passing the ratchet on migrated data;
4. the 1.x snapshots kept, so the old promise stays checkable.

## Adding an instrument

```python
from craton.spec import REGISTRY, Channel, Instrument, Qualifier

REGISTRY.register(
    Instrument(
        name="borehole_strain",
        description="Areal strain from a borehole strainmeter.",
        kinds=("observation",),
        channels=(Channel("areal_strain", "1", "Areal strain, extension positive."),),
        qualifiers=(Qualifier("station", "Station code.", required=True, pattern=r"[A-Z0-9]{4}"),),
        added_in="1.1.0",
    )
)
```

Units come from `craton.spec.UNITS` ([UCUM](https://ucum.org) codes). Then add an example token, regenerate the
document with `craton spec docs`, and freeze the next minor version when it ships.
