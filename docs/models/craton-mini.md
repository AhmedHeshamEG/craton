<!-- Written by `craton bench run craton-mini` from the model cards next to each checkpoint. -->

# Model cards

One card per checkpoint: three seeds for each split a CratonBench task trains on. The checkpoints themselves stay out of git; `craton train SPLIT` rebuilds any of them.

## mini-v0.1-cb02_fit-bcf9ffb66b6f-87a98083-s0

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 1985-01-02 → 2016-08-23 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 0 |
| Training | 2500 steps, best at 2500, 81,920,000 tokens, 15.8 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,500 MiB |
| Throughput | 86,453 tokens/s |
| Best validation | 29.367 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-cb02_fit-bcf9ffb66b6f-87a98083-s1

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 1985-01-02 → 2016-08-23 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 1 |
| Training | 2500 steps, best at 2300, 81,920,000 tokens, 15.8 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,519 MiB |
| Throughput | 86,382 tokens/s |
| Best validation | 29.464 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-cb02_fit-bcf9ffb66b6f-87a98083-s2

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 1985-01-02 → 2016-08-23 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 2 |
| Training | 1900 steps, best at 1600, 62,259,200 tokens, 12.0 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,536 MiB |
| Throughput | 86,375 tokens/s |
| Best validation | 29.775 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-cb03_fit-7f4e735423f1-87a98083-s0

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 1985-01-02 → 2008-12-31 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 0 |
| Training | 2500 steps, best at 2500, 81,920,000 tokens, 15.6 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,561 MiB |
| Throughput | 87,327 tokens/s |
| Best validation | 29.893 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-cb03_fit-7f4e735423f1-87a98083-s1

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 1985-01-02 → 2008-12-31 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 1 |
| Training | 2500 steps, best at 2500, 81,920,000 tokens, 15.7 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,583 MiB |
| Throughput | 87,232 tokens/s |
| Best validation | 29.573 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-cb03_fit-7f4e735423f1-87a98083-s2

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 1985-01-02 → 2008-12-31 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 2 |
| Training | 2500 steps, best at 2500, 81,920,000 tokens, 15.7 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,599 MiB |
| Throughput | 87,229 tokens/s |
| Best validation | 29.793 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-pretrain-348835f47082-87a98083-s0

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 1985-01-02 → 2026-09-30 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 0 |
| Training | 2500 steps, best at 2400, 40,960,000 tokens, 8.0 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,325 MiB |
| Throughput | 85,695 tokens/s |
| Best validation | 21.274 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-pretrain-348835f47082-87a98083-s1

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 1985-01-02 → 2026-09-30 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 1 |
| Training | 2500 steps, best at 2400, 40,960,000 tokens, 8.0 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,346 MiB |
| Throughput | 84,862 tokens/s |
| Best validation | 21.395 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.

## mini-v0.1-pretrain-348835f47082-87a98083-s2

Craton-mini v0: a transformer over the INGV earthquake catalogue of central Italy. It answers with probability distributions only: bins of place and size for a hidden quake, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 1985-01-02 → 2026-09-30 |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,534,731 (trunk 4,739,072, encoder 136,704, heads 658,955) |
| Seed | 2 |
| Training | 2500 steps, best at 2400, 40,960,000 tokens, 8.1 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,371 MiB |
| Throughput | 84,512 tokens/s |
| Best validation | 21.115 bits (weighted sum of the objectives) |
| Config hash | `87a98083c88de3cd` |
| Code | craton 0.4.0, training v0.1 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region and one instrument, and it has not been calibrated (that is P6).

**Known limits.** Catalogue only (no GNSS, no tides). Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid.
