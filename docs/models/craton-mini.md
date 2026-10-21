<!-- Written by `craton bench run craton-mini` from the model cards next to each checkpoint. -->

# Model cards

One card per checkpoint: three seeds for each split a CratonBench task trains on. The checkpoints themselves stay out of git; `craton train SPLIT` rebuilds any of them.

## mini-v0.4-cb02_fit-bcf9ffb66b6f-fbecfe06-s0

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 323,746 GNSS station-days, 277,392 hourly tide samples, 1985-01-02 → 2016-08-23 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 0 |
| Training | 4000 steps, best at 4000, 131,072,000 tokens, 24.1 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,562 MiB |
| Throughput | 90,510 tokens/s |
| Best validation | 39.183 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb02_fit-bcf9ffb66b6f-fbecfe06-s1

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 323,746 GNSS station-days, 277,392 hourly tide samples, 1985-01-02 → 2016-08-23 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 1 |
| Training | 4000 steps, best at 3800, 131,072,000 tokens, 24.1 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,583 MiB |
| Throughput | 90,574 tokens/s |
| Best validation | 39.497 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb02_fit-bcf9ffb66b6f-fbecfe06-s2

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb02_fit` (manifest `bcf9ffb66b6f8951`), 132,281 quakes, 323,746 GNSS station-days, 277,392 hourly tide samples, 1985-01-02 → 2016-08-23 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 2 |
| Training | 4000 steps, best at 3700, 131,072,000 tokens, 24.1 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,605 MiB |
| Throughput | 90,596 tokens/s |
| Best validation | 39.448 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb03_fit-7f4e735423f1-fbecfe06-s0

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 60,879 GNSS station-days, 210,384 hourly tide samples, 1985-01-02 → 2008-12-31 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 0 |
| Training | 3000 steps, best at 2200, 98,304,000 tokens, 18.0 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,498 MiB |
| Throughput | 91,000 tokens/s |
| Best validation | 41.441 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb03_fit-7f4e735423f1-fbecfe06-s1

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 60,879 GNSS station-days, 210,384 hourly tide samples, 1985-01-02 → 2008-12-31 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 1 |
| Training | 4000 steps, best at 3500, 131,072,000 tokens, 24.0 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,521 MiB |
| Throughput | 90,871 tokens/s |
| Best validation | 41.073 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb03_fit-7f4e735423f1-fbecfe06-s2

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb03_fit` (manifest `7f4e735423f10c53`), 30,920 quakes, 60,879 GNSS station-days, 210,384 hourly tide samples, 1985-01-02 → 2008-12-31 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1, next_day x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 2 |
| Training | 4000 steps, best at 3600, 131,072,000 tokens, 24.1 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 2,542 MiB |
| Throughput | 90,756 tokens/s |
| Best validation | 41.047 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb17_fit-96c3880ee23e-fbecfe06-s0

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb17_fit` (manifest `96c3880ee23e0549`), 166,221 quakes, 526,722 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 0 |
| Training | 4000 steps, best at 3900, 65,536,000 tokens, 12.4 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,766 MiB |
| Throughput | 87,785 tokens/s |
| Best validation | 30.612 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb17_fit-96c3880ee23e-fbecfe06-s1

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb17_fit` (manifest `96c3880ee23e0549`), 166,221 quakes, 526,722 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 1 |
| Training | 4000 steps, best at 3700, 65,536,000 tokens, 12.4 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,756 MiB |
| Throughput | 87,822 tokens/s |
| Best validation | 30.857 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-cb17_fit-96c3880ee23e-fbecfe06-s2

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `cb17_fit` (manifest `96c3880ee23e0549`), 166,221 quakes, 526,722 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 2 |
| Training | 4000 steps, best at 3900, 65,536,000 tokens, 12.5 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,794 MiB |
| Throughput | 87,624 tokens/s |
| Best validation | 30.571 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-pretrain-348835f47082-fbecfe06-s0

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 593,823 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 0 |
| Training | 4000 steps, best at 3900, 65,536,000 tokens, 12.3 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,373 MiB |
| Throughput | 88,998 tokens/s |
| Best validation | 31.291 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-pretrain-348835f47082-fbecfe06-s1

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 593,823 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 1 |
| Training | 4000 steps, best at 3800, 65,536,000 tokens, 12.3 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,394 MiB |
| Throughput | 88,725 tokens/s |
| Best validation | 31.416 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## mini-v0.4-pretrain-348835f47082-fbecfe06-s2

Craton-mini: a transformer over central Italy that reads the INGV earthquake catalogue, daily GNSS positions (NGL), the solid-Earth tide (IERS 2010). It answers with probability distributions only: bins of place and size for a hidden quake, a millimetre-scale distribution for a hidden GNSS day, and an expected number of quakes per cell and magnitude bin for the next day. It forecasts; it does not predict earthquakes.

| | |
|---|---|
| Trained on | split `pretrain` (manifest `348835f470820aac`), 205,553 quakes, 593,823 GNSS station-days, 345,312 hourly tide samples, 1985-01-02 → 2026-09-30 |
| Reads | `seismic_catalog`, `gnss_daily`, `solid_earth_tide` |
| Snapshot | `96a163f798d8f9cc` |
| Objectives | masked x1, gnss x1 |
| Shape | d = 256, 6 layers, 8 heads, context 1024 tokens, rope4d |
| Parameters | 5,685,298 (trunk 4,739,072, encoders 277,248, heads 668,978) |
| Seed | 2 |
| Training | 4000 steps, best at 3800, 65,536,000 tokens, 12.3 min on NVIDIA GeForce RTX 2060 (fp16 + GradScaler) |
| Peak VRAM | 1,415 MiB |
| Throughput | 88,526 tokens/s |
| Best validation | 31.641 bits (weighted sum of the objectives) |
| Config hash | `fbecfe0609dac734` |
| Code | craton 0.5.0, training v0.4 |

**Intended use.** Research and teaching: benchmark runs on CratonBench, and a first point on Craton's learning curve.

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P6).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.
