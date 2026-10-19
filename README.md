<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/hero-light.png">
    <img alt="Every magnitude 5+ earthquake since 1976 on a world map with no coastlines. The plate boundaries appear on their own." src="docs/assets/hero-dark.png" width="100%">
  </picture>
</p>

<p align="center">
  <b>A foundation model for the solid Earth, built the way LLMs are built.</b><br>
  One token for every measurement. One objective. One operation.<br>
  And a benchmark suite that only lets it get better.
</p>

<p align="center">
  <a href="https://github.com/AhmedHeshamEG/craton/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/AhmedHeshamEG/craton/actions/workflows/ci.yml/badge.svg"></a>
  <a href="ROADMAP.md"><img alt="Phase" src="https://img.shields.io/badge/phase-P0%20of%20P12-ff6b35"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-1d1d1f"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-1d1d1f">
  <img alt="Runs on" src="https://img.shields.io/badge/runs%20on-a%206%20GB%20laptop%20GPU-1d1d1f">
</p>

---

## The idea

Language models got good when someone decided that all text, every book and forum and recipe, could be
cut into one kind of token, and one model could learn by filling in the blanks.

The solid Earth never had that moment. Seismometers have their models, GPS has its models, tide gauges and
rock labs and catalogue historians have theirs, and none of them read each other. Nothing reads the
seismometer, the GPS antenna, the pull of the Moon and a monk's diary from 1349 at the same time.

Craton is that one model.

- **One token.** Every measurement on Earth becomes the same shape:
  `kind · instrument · lat, lon, depth · time · value · uncertainty · provenance · resolution`.
  A seismogram window, a day of GPS, a tidal stress, a rock's porosity, a line in a medieval chronicle. The
  monk is just a sensor with a very large uncertainty slot.
- **One objective.** Hide any token, predict it from the rest. The past and the future are the same game; the
  holes are just in different places.
- **One operation.** Attention across space and time, with positions in four dimensions.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-it-works-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/how-it-works-light.svg">
    <img alt="Anything measured becomes one token, each instrument gets an encoder, attention runs across latitude, longitude, depth and time, and the output is a probability distribution." src="docs/assets/how-it-works-dark.svg" width="100%">
  </picture>
</p>

## What it will never tell you

The date of the next big earthquake.

Big earthquakes are rare, the stress that triggers them sits 15 km down where nothing can measure it, and
Parkfield taught seismology humility: a fault that was "due" around 1988 broke in 2004, under a forest of
instruments, with no clear warning. So every Craton output is a **distribution**: a probability, an
ensemble, a window with a stated resolution. The project tracks how wide that window is, and calls it the
predictability meter. Every version has to shrink it, or it doesn't count as progress.

## Three numbers

| | What it asks | Twin in the LLM world |
|---|---|---|
| **Geo-perplexity** | How surprised is the model by real Earth data it has never seen? | Perplexity |
| **Predictability meter** | How wide is the 90% window, in days, kilometres and magnitude? | Calibrated uncertainty |
| **Information gain over ETAS** | Does it beat the standard statistical model on a real sequence (Amatrice–Norcia, 2016–17)? | Beating the strong baseline |

Today all three read *not measured yet*, on purpose. The simple baselines get scored first (phase P3), so the
first model has something honest to beat. Live numbers: [`SCOREBOARD.md`](SCOREBOARD.md).

## Built like a building

You don't redesign the foundation every time you add a floor. Craton is constructive-only:

- **The token spec is frozen once released.** New versions may only *add* fields and instruments. Data written
  in 2026 stays valid forever.
- **Data is immutable.** Every download and token file is content-hashed; corrections arrive as new versions.
- **Benchmarks are immutable.** A test's data and metric never change. A better test gets a new ID.
- **The ratchet.** A change lands only if no benchmark gets worse beyond noise (2σ over 3 seeds), compared
  with the best score ever recorded, so small losses can't quietly stack up.
- **Every number comes from a script.** `SCOREBOARD.md` is generated; the check fails if a human edits it.
- **The logbook and decisions are append-only.** Mistakes get corrected below, never erased.

All of it is one command, the same on a laptop and on CI:

```console
$ uv run craton check
✓ lint · ✓ format · ✓ types · ✓ tests · ✓ scoreboard is generated · ✓ records append-only · ✓ ratchet
all green. The foundation holds.
```

## One codebase, three sizes

The only difference between them is a YAML file in [`configs/`](configs/).

| | Trunk | Shape | Context | Where |
|---|---|---|---|---|
| **Craton-mini** | 4.74M | d=256 × 6 layers | 1,024 tokens | a laptop RTX 2060, 6 GB, central Italy |
| **Craton-regional** | 85.1M | d=768 × 12 layers | 8,192 tokens | one rented A100 |
| **Craton-planet** | 1.61B | d=2048 × 32 layers | 131,072 tokens | a GPU cluster, once the scaling law says so |

Parameter counts come from `craton config <size>`, and a config that lies about its own size fails to load.

## Roadmap

```
CORE       ████░░░░░░░░░░░░░░░░░░░░░░░░░░   1/7    foundation → a model that knows when it doesn't know
DISCOVERY  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/4    fake planets, the ghost channel, the translator, the microscope
SCALE      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0/2    any time horizon, then mini → regional → planet
```

| | Phase | Borrowed from the LLM world |
|---|---|---|
| ✅ | **P0** Foundation: repo, checks, the ratchet | the training-infra repo every lab builds first |
| ◻️ | **P1** The universal token, frozen | the tokenizer |
| ◻️ | **P2** First data: INGV catalogue, GNSS, tides | the data pipeline |
| ◻️ | **P3** CratonBench + baselines, before any model | the eval harness |
| ◻️ | **P4** Craton-mini v0 | the GPT-2 small run |
| ◻️ | **P5** Fusion: quakes + GPS + the Moon | multimodal models |
| ◻️ | **P6** The 8 locks: knowing when it doesn't know | calibration, ensembles, red-teaming |
| ◻️ | **P7** Fake planets: pretraining on simulated millennia | synthetic data |
| ◻️ | **P8** The ghost channel: forces we forgot | register tokens |
| ◻️ | **P9** The translator: plain-language answers, numbers only from the model | tool use |
| ◻️ | **P10** The microscope: what the model learned | sparse autoencoders |
| ◻️ | **P11** Any horizon, from next week to a million years | long context |
| ◻️ | **P12** Scale-ready, open to every dataset | scaling laws, LoRA |

The full plan, with every benchmark and exit test: [`ROADMAP.md`](ROADMAP.md).

## Quickstart

```console
git clone https://github.com/AhmedHeshamEG/craton && cd craton
uv sync --extra cu126      # NVIDIA GPU   (no NVIDIA: uv sync --extra cpu)
uv run craton doctor       # what this machine can run, and how
uv run craton check        # everything that has to pass before a change lands
```

On the laptop it was built on, `craton doctor --smoke` says:

```console
machine  Windows 11 · Python 3.11.9 · 15.8 GiB RAM
pytorch  2.14.1+cu126 (CUDA 12.6)
gpu      NVIDIA GeForce RTX 2060 · 6 GiB · Turing sm_75
fp16     ✓ tensor cores · trains with gradient scaling
bf16     ✗ not on Turing · a bf16 request becomes fp16

 size       status    trunk   d × layers   context   precision here
 mini       active    4.74M   256 × 6      1,024     fp16 + scaler
 regional   planned   85.1M   768 × 12     8,192     bf16 → fp16 + scaler
 planet     planned   1.61B   2048 × 32    131,072   bf16 → fp16 + scaler

✓ smoke: 60 fp16 steps on cuda · loss 0.522 → 0.001 · peak 17.4 MiB
ready: Craton-mini trains on this GPU.
```

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

## Learn along

Every phase ships a lesson in [`lessons/`](lessons/), written for a smart friend who has never trained a model:
the idea step by step, the LLM analogy and where it breaks, the code in the order data flows, and a few
commands to run it yourself. Follow the phases and you've built a foundation model from an empty folder.

## Contributing

Adapters for new data, new forces, new benchmarks: everything plugs in through registries, nothing edits the
core, and the ratchet guards the rest. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Credits

Hero data: [USGS ComCat](https://earthquake.usgs.gov/earthquakes/search/) (public domain), drawn by
[`scripts/figures/banner.py`](scripts/figures/banner.py). Type: [Inter](https://rsms.me/inter/) (SIL OFL).
Craton stands on SeisLM, ETAS, CSEP and pyCSEP, GraphCast's lesson about reanalysis, and decades of open
catalogues from INGV and the Nevada Geodetic Laboratory. Full sources per phase in the lessons.

## Citing

See [`CITATION.cff`](CITATION.cff), or GitHub's "Cite this repository" button.

## License

Code: [Apache-2.0](LICENSE). Every dataset keeps its own licence, recorded in its data card.

<sub>A craton is the ancient core of a continent: rock that hasn't deformed in billions of years, while new crust
keeps docking onto its edges. That's the rule for this repo.</sub>
