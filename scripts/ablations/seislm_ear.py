"""P10d: a borrowed ear. SeisLM-base (Liu, Münchmeyer et al. 2024) heard on ten quiet minutes a day.

For every station-day our band powers exist on (``seismic_noise_power``, P10a), ten minutes of
100 Hz ``HH?`` are fetched, 02:00-02:10 UTC (the quietest hour of the Apennine night), cut into
twenty 30 s windows of 3001 samples, normalised the way SeisLM was trained (per component:
demean, divide by the standard deviation) and passed through the frozen encoder. Windows far
louder than the day's median (a quake, a truck) are dropped; the rest are averaged into one
240-number embedding per station-day, with its spread. Stored in ``data/cache/seislm/<region>.npz``
and read by ``craton.field.listen`` when a config sets ``listen.ear: seislm``.

    uv run --no-sync python scripts/ablations/seislm_ear.py fetch      # resumable
    uv run --no-sync python scripts/ablations/seislm_ear.py embed

SeisLM is a spike dependency, installed beside the lockfile (``uv pip install`` of the wheel on
its Hugging Face page); nothing in ``craton/`` imports it.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from craton.bench.data import DAY, read_table
from craton.bench.tasks import BenchContext

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ("central_italy_noise", "laquila_noise")
SERVICE = "https://webservices.ingv.it/fdsnws/dataselect/1/query"
RAW = ROOT / "data" / "raw" / "seislm_windows"
CACHE = ROOT / "data" / "cache" / "seislm"
START_S, LENGTH_S = 2 * 3600, 600
RATE = 100.0
WIN = 3001
N_WIN = 19  # 600 s / 30.01 s, whole windows only
LOUD = 3.0  # a window this many times the day's median rms is dropped
MODEL = "ltlltl2006/seislm-base"
REVISION = "13a3dcd6ae56c39f5b0750b817c8788a7fc3c99b"


def station_days(region: str) -> dict[str, tuple[float, float, list[float]]]:
    """station -> (lat, lon, midnights its band powers exist on)."""
    view = BenchContext(root=ROOT, region=region)
    t = read_table(
        view._snapshot_files("seismic_noise_power"), ("lat", "lon", "time", "qualifiers")
    )
    out: dict[str, tuple[float, float, set[float]]] = {}
    for lat, lon, tt, q in zip(
        *(t.column(c).to_pylist() for c in ("lat", "lon", "time", "qualifiers")), strict=True
    ):
        sta = json.loads(q)["station"]
        out.setdefault(sta, (lat, lon, set()))[2].add(tt)
    return {s: (a, b, sorted(d)) for s, (a, b, d) in out.items()}


def _iso(t: float) -> str:
    return np.datetime64(int(t), "s").astype(str)


def _month(t: float) -> str:
    return _iso(t)[:7]


def fetch(workers: int = 4) -> None:
    import urllib.request

    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for region in REGIONS:
        for sta, (_, _, days) in station_days(region).items():
            by_month: dict[str, list[float]] = defaultdict(list)
            for d in days:
                by_month[_month(d)].append(d)
            for month, ds in by_month.items():
                path = RAW / f"{sta}.{month}.mseed"
                if not path.exists():
                    jobs.append((path, sta, ds))
    print(f"{len(jobs)} station-months to fetch", flush=True)

    def one(path: Path, sta: str, ds: list[float]) -> int:
        net, code = sta.split(".")
        body = "".join(
            f"{net} {code} * HH? {_iso(d + START_S)} {_iso(d + START_S + LENGTH_S)}\n" for d in ds
        ).encode()
        for attempt in range(4):
            try:
                req = urllib.request.Request(SERVICE, data=body, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read() if r.status == 200 else b""
                break
            except urllib.error.HTTPError as e:
                if e.code in (204, 404):
                    data = b""
                    break
                time.sleep(10 * (attempt + 1))
            except OSError:
                time.sleep(10 * (attempt + 1))
        else:
            return -1
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return len(data)

    done, total, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, *j): j[0].name for j in jobs}
        for f in as_completed(futs):
            n = f.result()
            done += 1
            total += max(n, 0)
            if n < 0:
                print(f"failed {futs[f]}", flush=True)
            if done % 20 == 0 or done == len(jobs):
                print(
                    f"{done}/{len(jobs)}  {total / 1e9:.2f} GB  {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )


def windows(data: bytes, days: list[float]) -> dict[float, np.ndarray]:
    """midnight -> (N_WIN, 3, WIN) ZNE windows for the days whose three components are whole."""
    from craton.adapters.noise import decode

    segs, _ = decode(data)
    by_loc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in segs:
        if abs(s.rate - RATE) < 1e-6:
            by_loc[s.location][s.channel[-1]].append(s)
    n = round(LENGTH_S * RATE)
    out: dict[float, np.ndarray] = {}
    for d in days:
        a = d + START_S
        best = None
        for comps in by_loc.values():
            grids = []
            for c in ("Z", "N", "E"):
                g = np.full(n, np.nan)
                for s in comps.get(c, []) or comps.get({"N": "1", "E": "2"}.get(c, c), []):
                    first = round((s.start - a) * RATE)
                    lo, hi = max(first, 0), min(first + len(s.samples), n)
                    if lo < hi:
                        g[lo:hi] = s.samples[lo - first : hi - first]
                grids.append(g)
            x = np.stack(grids)
            if np.isfinite(x).all():
                best = x
                break
        if best is not None:
            out[d] = best[:, : N_WIN * WIN].reshape(3, N_WIN, WIN).transpose(1, 0, 2)
    return out


def embed(batch: int = 128) -> None:
    import torch
    from seisLM import encode, load_pretrained

    model = load_pretrained(MODEL, revision=REVISION).cuda().eval()
    CACHE.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        sd = station_days(region)
        names = sorted(sd)
        all_days = sorted({d for s in names for d in sd[s][2]})
        day0 = all_days[0]
        n_days = round((all_days[-1] - day0) / DAY) + 1
        emb = np.full((len(names), n_days, 240), np.nan, np.float32)
        spread = np.full_like(emb, np.nan)
        kept = np.zeros((len(names), n_days), np.int16)
        t0 = time.time()
        for i, sta in enumerate(names):
            by_month: dict[str, list[float]] = defaultdict(list)
            for d in sd[sta][2]:
                by_month[_month(d)].append(d)
            for month, ds in sorted(by_month.items()):
                path = RAW / f"{sta}.{month}.mseed"
                if not path.exists():
                    continue
                w = windows(path.read_bytes(), ds)
                if not w:
                    continue
                keys = sorted(w)
                x = np.concatenate([w[k] for k in keys])  # (days*N_WIN, 3, WIN)
                x = x - x.mean(-1, keepdims=True)
                std = x.std(-1, keepdims=True)
                rms = std.mean(axis=(1, 2)).reshape(len(keys), N_WIN)
                x = x / (np.where(std == 0, 1.0, std) + 1e-10)
                feats = []
                with torch.no_grad():
                    for b in range(0, len(x), batch):
                        xb = torch.from_numpy(x[b : b + batch]).float().cuda()
                        feats.append(encode(model, xb).float().mean(1).cpu().numpy())
                f = np.concatenate(feats).reshape(len(keys), N_WIN, 240)
                quiet = rms <= LOUD * np.median(rms, axis=1, keepdims=True)
                for j, k in enumerate(keys):
                    q = f[j][quiet[j]]
                    di = round((k - day0) / DAY)
                    emb[i, di] = q.mean(0)
                    spread[i, di] = q.std(0) / np.sqrt(len(q)) + 1e-6
                    kept[i, di] = len(q)
            print(
                f"{region} {sta} {int((kept[i] > 0).sum())} days  {time.time() - t0:.0f} s",
                flush=True,
            )
        np.savez_compressed(
            CACHE / f"{region}.npz",
            stations=np.array(names),
            lat=np.array([sd[s][0] for s in names]),
            lon=np.array([sd[s][1] for s in names]),
            days=day0 + DAY * np.arange(n_days),
            embedding=emb,
            spread=spread,
            kept=kept,
            model=np.array(f"{MODEL}@{REVISION}"),
        )


if __name__ == "__main__":
    {"fetch": fetch, "embed": embed}[sys.argv[1]]()
