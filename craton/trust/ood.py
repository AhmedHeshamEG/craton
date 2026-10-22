"""The "I have never seen this" detector.

A forecaster trained on central Italy will happily forecast anything you show it. The detector
asks first whether the context looks like anything it learned from. Following Sun et al.
(ICML 2022, "Out-of-distribution detection with deep nearest neighbors"): take the vector the
trunk builds for a context (the query token's last hidden state, the one the forecast reads),
normalise it, and measure how far it is from its k-th nearest neighbour among the vectors of
the training contexts. Far = never seen; past a threshold set on the training contexts alone
(their own 95th percentile), the forecast must abstain.

**Kind, not place.** The trunk reads every quake's absolute latitude and longitude, so a context
from Sicily is trivially "elsewhere", and a detector that only noticed that would score a
perfect and useless 1.0. Every context, training ones included, is therefore moved so its
median epicentre sits at the centre of central Italy (longitudes rescaled for the change of
latitude, so distances stay true) before the trunk reads it. What is left to tell apart is
the seismicity itself: how deep, how big, how fast, how clustered.

The floor is the same nearest-neighbour rule on a hand-made fingerprint of the same quakes
(time span, depth and magnitude quantiles, spatial spread). If the trunk's vectors can't beat
twelve numbers a person would write down, its "understanding" is not doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from craton.bench.data import DAY, Events
from craton.train.context import Catalog, Sources, write
from craton.train.data import midnight_window

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

CENTRE = (42.5, 13.25)
K_NEIGHBOURS = 10
ABSTAIN_QUANTILE = 0.95
EMBED_BATCH = 16


def translate(ev: Events, centre: tuple[float, float] = CENTRE) -> Events:
    """Move the quakes so their median epicentre sits at ``centre``, keeping distances."""
    if not len(ev):
        return ev
    lat0, lon0 = float(np.median(ev.lat)), float(np.median(ev.lon))
    lat = ev.lat - lat0 + centre[0]
    scale = np.cos(np.radians(ev.lat)) / np.cos(np.radians(lat))
    lon = centre[1] + (ev.lon - lon0) * scale
    return replace(ev, lat=lat, lon=lon)


def usable(events: Events, midnights: FloatArray, need: int) -> FloatArray:
    """The midnights with at least ``need`` quakes before them."""
    first = np.searchsorted(events.time, midnights, "left")
    return np.asarray(midnights[first >= need], dtype=np.float64)


def embed(
    run: Any, src: Sources, events: Events, midnights: FloatArray
) -> tuple[FloatArray, list[Events]]:
    """(n, d) normalised query-token vectors of the translated contexts, and the contexts.

    ``run(batch) -> (B, T, d)`` runs the trunk; ``src`` was built from ``events`` (untranslated)
    and decides which quakes each window holds.
    """
    vecs: list[FloatArray] = []
    ctxs: list[Events] = []
    for i in range(0, len(midnights), EMBED_BATCH):
        windows = []
        for day in midnights[i : i + EMBED_BATCH]:
            w = midnight_window(src, float(day))
            w.gnss = np.zeros(0, np.int64)
            w.gnss_hidden = np.zeros(0, bool)
            rows = np.sort(w.quakes)
            moved = translate(events.select(rows))
            ctxs.append(moved)
            w.cat = Catalog.of(moved)
            w.quakes = np.arange(len(rows), dtype=np.int64)
            w.quake_hidden = np.zeros(len(rows), bool)
            windows.append(w)
        batch = write(src, windows)
        h = run(batch)[:, -1].float()
        h = torch.nn.functional.normalize(h, dim=-1)
        vecs.append(h.cpu().numpy().astype(np.float64))
    return np.concatenate(vecs), ctxs


def fingerprint(ev: Events) -> FloatArray:
    """Twelve numbers a seismologist would use to describe a context of quakes."""
    span = max(float(ev.time[-1] - ev.time[0]) / DAY, 1e-3)
    dq = np.quantile(ev.depth_km, (0.1, 0.5, 0.9))
    mq = np.quantile(ev.mag, (0.1, 0.5, 0.9))
    km_lat = (ev.lat - np.median(ev.lat)) * 111.195
    km_lon = (ev.lon - np.median(ev.lon)) * 111.195 * np.cos(np.radians(np.median(ev.lat)))
    gaps = np.diff(ev.time) / DAY
    return np.array(
        [
            np.log10(span),
            *dq,
            float(np.mean(ev.depth_km > 30)),
            *mq,
            np.log10(1 + float(np.std(km_lat))),
            np.log10(1 + float(np.std(km_lon))),
            np.log10(1e-4 + float(np.median(gaps))) if len(gaps) else 0.0,
            float(np.mean(ev.mag >= 3.0)),
        ]
    )


def standardise(bank: FloatArray, *others: FloatArray) -> list[FloatArray]:
    mu, sd = bank.mean(axis=0), bank.std(axis=0) + 1e-9
    return [(x - mu) / sd for x in (bank, *others)]


def knn_distance(
    bank: FloatArray, queries: FloatArray, k: int = K_NEIGHBOURS, self_bank: bool = False
) -> FloatArray:
    """Distance from each query to its k-th nearest neighbour in ``bank`` (Euclidean; on
    normalised vectors this ranks like cosine). ``self_bank``: queries are the bank itself,
    so each skips its own entry."""
    b = torch.tensor(bank, dtype=torch.float64)
    out = []
    for i in range(0, len(queries), 512):
        q = torch.tensor(queries[i : i + 512], dtype=torch.float64)
        d = torch.cdist(q, b)
        kk = k + 1 if self_bank else k
        out.append(torch.topk(d, kk, largest=False).values[:, -1].numpy())
    return np.concatenate(out)


def auroc(negative: FloatArray, positive: FloatArray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    scores = np.concatenate([negative, positive])
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="stable")
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_neg, n_pos = len(negative), len(positive)
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_neg * n_pos))


@dataclass(frozen=True)
class OodScores:
    """Distances of the three sets of contexts; the threshold is set on ``reference`` alone."""

    reference: FloatArray  # training contexts, each scored without itself
    in_region: FloatArray  # central Italy, test eras: never trained on, same kind of place
    unseen: FloatArray  # Calabria and Sicily
    #: Anything else a detector wants on the record (its parts scored alone, say).
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return float(np.quantile(self.reference, ABSTAIN_QUANTILE))

    def summary(self) -> dict[str, float]:
        thr = self.threshold
        return {
            "auroc": auroc(self.in_region, self.unseen),
            "auroc_reference_vs_unseen": auroc(self.reference, self.unseen),
            "auroc_reference_vs_in_region": auroc(self.reference, self.in_region),
            "abstain_reference": float(np.mean(self.reference > thr)),
            "abstain_in_region": float(np.mean(self.in_region > thr)),
            "abstain_unseen": float(np.mean(self.unseen > thr)),
            "contexts_reference": float(len(self.reference)),
            "contexts_in_region": float(len(self.in_region)),
            "contexts_unseen": float(len(self.unseen)),
            **self.extra,
        }


@dataclass(frozen=True)
class OodQueries:
    """CB-06's three sets of midnights. ``reference`` contexts come from the ``pretrain``
    split (what the model learned from), ``in_region`` from the snapshot inside the held-out
    eras, and ``unseen`` from ``unseen_events``, the Calabria and Sicily catalogue."""

    reference: FloatArray
    in_region: FloatArray
    unseen: FloatArray
    unseen_events: Events


def last_quakes(ev: Events, midnights: FloatArray, n: int) -> list[Events]:
    """The ``n`` quakes before each midnight."""
    first = np.searchsorted(ev.time, midnights, "left")
    return [ev.select(np.arange(f - n, f)) for f in first]


def scores_of(bank: FloatArray, in_region: FloatArray, unseen: FloatArray) -> OodScores:
    return OodScores(
        knn_distance(bank, bank, self_bank=True),
        knn_distance(bank, in_region),
        knn_distance(bank, unseen),
    )


@dataclass
class NoDetector:
    """The floor CB-06 starts from: the same forecaster with no detector (every score 0)."""

    name: str = "no_detector"
    kind: str = "baseline"

    def ood_scores(self, queries: OodQueries, ctx: Any, seed: int) -> OodScores:
        return OodScores(
            np.zeros(len(queries.reference)),
            np.zeros(len(queries.in_region)),
            np.zeros(len(queries.unseen)),
        )


@dataclass
class FingerprintKnn:
    """The person's detector: twelve hand-made numbers per context (the last 1,024 quakes),
    standardised on the training contexts, then the same nearest-neighbour rule."""

    name: str = "fingerprint_knn"
    kind: str = "baseline"
    quakes: int = 1024

    def ood_scores(self, queries: OodQueries, ctx: Any, seed: int) -> OodScores:
        sets = (
            (ctx.train_events("pretrain"), queries.reference),
            (ctx.events, queries.in_region),
            (queries.unseen_events, queries.unseen),
        )
        f = [np.array([fingerprint(c) for c in last_quakes(ev, m, self.quakes)]) for ev, m in sets]
        bank, a, b = standardise(*f)
        return scores_of(bank, a, b)


def craton_vectors(run: Any, src: Sources, events: Events, midnights: FloatArray) -> FloatArray:
    """Craton's view of each context: the trunk's normalised query vector, and the
    fingerprint of the very quakes it read (standardised later, against the bank)."""
    vecs, ctxs = embed(run, src, events, midnights)
    prints = np.array([fingerprint(c) for c in ctxs])
    return np.concatenate([vecs, prints], axis=1)


def combine(bank: FloatArray, *others: FloatArray, d_trunk: int) -> list[FloatArray]:
    """Trunk vector (unit length) next to the standardised fingerprint scaled to unit length
    on average, so both halves weigh the same in the distance."""
    parts = standardise(bank[:, d_trunk:], *(o[:, d_trunk:] for o in others))
    n_fp = bank.shape[1] - d_trunk
    out = []
    for full, fp in zip((bank, *others), parts, strict=True):
        out.append(np.concatenate([full[:, :d_trunk], fp / np.sqrt(n_fp)], axis=1))
    return out
