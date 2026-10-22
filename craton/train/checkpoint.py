"""Checkpoints and their model cards.

A checkpoint is named by everything that made it: the split it learned from (by manifest id),
the config (by hash), the seed and the training code's version. Same name, same model; a
change to any of them trains a new one. Next to every ``model.pt`` sits ``card.md``, the
model card: what it was trained on, how big it is, what it cost, how it scored.

Checkpoints live in ``results/checkpoints/`` and stay out of git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from craton import __version__
from craton.config import CratonConfig, config_hash
from craton.paths import results_dir

#: Bumped whenever the training code changes what a checkpoint would contain.
TRAIN_VERSION = "v0.4"


def checkpoints_dir(root: Path) -> Path:
    return results_dir(root) / "checkpoints"


@dataclass(frozen=True)
class CheckpointKey:
    split: str
    split_id: str
    config: CratonConfig
    seed: int

    @property
    def name(self) -> str:
        return (
            f"{self.config.name}-{TRAIN_VERSION}-{self.split}-{self.split_id[:12]}-"
            f"{config_hash(self.config)[:8]}-s{self.seed}"
        )

    def folder(self, root: Path) -> Path:
        return checkpoints_dir(root) / self.name


def save(
    root: Path,
    key: CheckpointKey,
    state: dict[str, torch.Tensor],
    history: list[dict[str, float]],
    stats: dict[str, Any],
    data: dict[str, Any],
) -> Path:
    folder = key.folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": key.name,
        "craton_version": __version__,
        "train_version": TRAIN_VERSION,
        "config": key.config.model_dump(mode="json"),
        "config_hash": config_hash(key.config),
        "seed": key.seed,
        "split": key.split,
        "split_id": key.split_id,
        "data": data,
        "stats": stats,
        "history": history,
    }
    torch.save({"state": state, "meta": meta}, folder / "model.pt")
    (folder / "card.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (folder / "card.md").write_text(render_card(meta), encoding="utf-8", newline="\n")
    return folder


def load(root: Path, key: CheckpointKey) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    path = key.folder(root) / "model.pt"
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["state"], blob["meta"]


def _mib(n: float) -> str:
    return f"{n / 2**20:,.0f} MiB"


def render_card(meta: dict[str, Any]) -> str:
    """The model card: a page a stranger can read to know what this checkpoint is."""
    s, cfg, data = meta["stats"], meta["config"], meta["data"]
    p = s["params"]
    objectives = ", ".join(f"{k} x{v:g}" for k, v in s["objectives"].items() if v > 0)
    inputs = cfg["model"].get("inputs", ["seismic_catalog"])
    reads = ", ".join(READS.get(i, i) for i in inputs)
    heads = p["masked_head"] + p["next_day_head"] + p.get("gnss_head", 0)
    seen = f"{data['quakes']:,} quakes"
    if data.get("gnss_days"):
        seen += f", {data['gnss_days']:,} GNSS station-days"
    if data.get("tide_hours"):
        seen += f", {data['tide_hours']:,} hourly tide samples"
    lines = [
        f"# Model card · {meta['name']}",
        "",
        f"Craton-mini: a transformer over central Italy that reads {reads}. It answers with "
        "probability distributions only: bins of place and size for a hidden quake, a "
        "millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes "
        "per cell and magnitude bin for the next day. It forecasts; it does not predict "
        "earthquakes.",
        "",
        "| | |",
        "|---|---|",
        f"| Trained on | split `{meta['split']}` (manifest `{meta['split_id'][:16]}`), "
        f"{seen}, {data['first']} → {data['last']} |",
        f"| Reads | {', '.join(f'`{i}`' for i in inputs)} |",
        f"| Snapshot | `{data['snapshot']}` |",
        f"| Objectives | {objectives} |",
        f"| Shape | d = {cfg['model']['d_model']}, {cfg['model']['n_layers']} layers, "
        f"{cfg['model']['n_heads']} heads, context {cfg['model']['context_tokens']} tokens, "
        f"{cfg['model']['positional']} |",
        f"| Parameters | {p['total']:,} (trunk {p['trunk']:,}, encoders {p['encoder']:,}, "
        f"heads {heads:,}) |",
        f"| Seed | {meta['seed']} |",
        f"| Training | {s['steps']} steps, best at {s['best_step']}, {s['tokens_seen']:,} tokens, "
        f"{s['train_seconds'] / 60:.1f} min on {s['device']} ({s['precision']}) |",
        f"| Peak VRAM | {_mib(s['peak_vram_bytes'])} |",
        f"| Throughput | {s['tokens_per_s']:,.0f} tokens/s |",
        f"| Best validation | {s['best_val']:.3f} bits (weighted sum of the objectives) |",
        f"| Config hash | `{meta['config_hash']}` |",
        f"| Code | craton {meta['craton_version']}, training {meta['train_version']} |",
        "",
        "**Intended use.** Research and teaching: benchmark runs on CratonBench, and a "
        "first point on Craton's learning curve.",
        "",
        "**Not for.** Any decision about safety. The model has seen one region, and it has "
        "not been calibrated (that is P7).",
        "",
        "**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md "
        "before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's "
        "daily solutions, which already have the solid-Earth tide modelled out. The tide it "
        "reads is one hourly series at the region's centre.",
        "",
    ]
    return "\n".join(lines)


READS = {
    "seismic_catalog": "the INGV earthquake catalogue",
    "gnss_daily": "daily GNSS positions (NGL)",
    "solid_earth_tide": "the solid-Earth tide (IERS 2010)",
}


def write_index(root: Path, folders: list[Path], out: Path, extra: list[str] | None = None) -> Path:
    """One public page holding the model card of every checkpoint behind a benchmark run.

    ``extra`` cards (Markdown, each starting with a level-1 title) are appended as they are."""
    cards = []
    for f in sorted(folders):  # re-rendered from the metadata, so no card is ever stale
        meta = json.loads((f / "card.json").read_text(encoding="utf-8"))
        cards.append(render_card(meta).strip())
    head = [
        "<!-- Written by `craton bench run craton-mini` from the model cards next to each "
        "checkpoint. -->",
        "",
        "# Model cards",
        "",
        "One card per checkpoint: three seeds for each split a CratonBench task trains on, "
        "then the Fields that forecast CB-02 and CB-03. "
        "The checkpoints themselves stay out of git; `craton train SPLIT` rebuilds any of them.",
        "",
    ]
    body = [c.replace("# Model card · ", "## ", 1) + "\n" for c in cards]
    body += [c.strip().replace("# ", "## ", 1) + "\n" for c in extra or []]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + body), encoding="utf-8", newline="\n")
    return out
