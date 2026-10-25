<!-- Written by `craton bench run craton-mini` from the model cards next to each checkpoint. -->

# Model cards

One card per checkpoint: three seeds for each split a CratonBench task trains on, then the Fields that forecast CB-02 and CB-03. The checkpoints themselves stay out of git; `craton train SPLIT` rebuilds any of them.

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

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

**Not for.** Any decision about safety. The model has seen one region, and it has not been calibrated (that is P7).

**Known limits.** Magnitudes are the catalogue's preferred value of any type (Md before 2005, ML after). Forecasts are daily, on the CB-02 grid. GNSS days are NGL's daily solutions, which already have the solid-Earth tide modelled out. The tide it reads is one hourly series at the region's centre.

## Field · CB-02 · seed 0

- **What it does:** daily forecasts for CB-02, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 15, 8, 19, 22, 19, 10; validation gain over ETAS (nats per quake) 0.1099, 0.0790, 0.1041, 0.1151, 0.1181, 0.0776; 44.2 min.
- **Starting line:** ETAS with mu 0.5746, k 0.4767, a 1.225, c 0.01796, p 1.186, d 0.5407, gamma 0.6945, q 2.035, b 1.163, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-02 · seed 1

- **What it does:** daily forecasts for CB-02, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 13, 16, 13, 17, 13, 15; validation gain over ETAS (nats per quake) 0.0966, 0.1020, 0.1060, 0.1190, 0.0900, 0.0970; 52.5 min.
- **Starting line:** ETAS with mu 0.5746, k 0.4767, a 1.225, c 0.01796, p 1.186, d 0.5407, gamma 0.6945, q 2.035, b 1.163, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-02 · seed 2

- **What it does:** daily forecasts for CB-02, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 18, 10, 14, 18, 16, 13; validation gain over ETAS (nats per quake) 0.1178, 0.0901, 0.1081, 0.1096, 0.1002, 0.0998; 45.2 min.
- **Starting line:** ETAS with mu 0.5746, k 0.4767, a 1.225, c 0.01796, p 1.186, d 0.5407, gamma 0.6945, q 2.035, b 1.163, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-03 · seed 0

- **What it does:** daily forecasts for CB-03, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb03_fit`, 6,575 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 12, 23, 15, 9, 19, 8; validation gain over ETAS (nats per quake) 0.1788, 0.1802, 0.1595, 0.1512, 0.1948, 0.1521; 36.3 min.
- **Starting line:** ETAS with mu 0.604, k 0.8269, a 1.035, c 0.00592, p 1.039, d 1.822, gamma 0.1932, q 2.209, b 1.311, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-03 · seed 1

- **What it does:** daily forecasts for CB-03, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb03_fit`, 6,575 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 9, 20, 8, 8, 13, 15; validation gain over ETAS (nats per quake) 0.1576, 0.1772, 0.1415, 0.1677, 0.1657, 0.1801; 31.5 min.
- **Starting line:** ETAS with mu 0.604, k 0.8269, a 1.035, c 0.00592, p 1.039, d 1.822, gamma 0.1932, q 2.209, b 1.311, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-03 · seed 2

- **What it does:** daily forecasts for CB-03, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb03_fit`, 6,575 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 7, 9, 21, 11, 19, 11; validation gain over ETAS (nats per quake) 0.1556, 0.1467, 0.1999, 0.1582, 0.1615, 0.1700; 30.9 min.
- **Starting line:** ETAS with mu 0.604, k 0.8269, a 1.035, c 0.00592, p 1.039, d 1.822, gamma 0.1932, q 2.209, b 1.311, mc 2.
- **Code:** Craton 0.7.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-etas_planet-0 · seed 0

- **What it does:** daily forecasts for CB-09-etas_planet-0, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 15, 6, 12, 17, 11, 15; validation gain over ETAS (nats per quake) 0.0135, 0.0082, 0.0130, 0.0116, 0.0129, 0.0119; 58.8 min.
- **Starting line:** ETAS with mu 0.6721, k 0.4863, a 1.256, c 0.01911, p 1.199, d 0.5928, gamma 0.6964, q 2.075, b 1.151, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-etas_planet-1 · seed 1

- **What it does:** daily forecasts for CB-09-etas_planet-1, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 9, 9, 8, 12, 10, 8; validation gain over ETAS (nats per quake) 0.0084, 0.0110, 0.0106, 0.0101, 0.0131, 0.0083; 46.0 min.
- **Starting line:** ETAS with mu 0.679, k 0.4874, a 1.222, c 0.01892, p 1.201, d 0.5878, gamma 0.7298, q 2.096, b 1.156, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-etas_planet-2 · seed 2

- **What it does:** daily forecasts for CB-09-etas_planet-2, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 11, 5, 7, 9, 5, 5; validation gain over ETAS (nats per quake) 0.0010, 0.0000, 0.0015, 0.0021, 0.0000, 0.0000; 23.4 min.
- **Starting line:** ETAS with mu 0.6962, k 0.4718, a 1.252, c 0.02062, p 1.201, d 0.5348, gamma 0.7271, q 2.031, b 1.161, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-physics_planet-0 · seed 0

- **What it does:** daily forecasts for CB-09-physics_planet-0, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 13, 16, 13, 15, 12, 15; validation gain over ETAS (nats per quake) 0.1207, 0.1272, 0.1338, 0.1432, 0.1261, 0.1346; 70.1 min.
- **Starting line:** ETAS with mu 0.3304, k 10, a 0.1072, c 1, p 1.012, d 1.569, gamma 0.1814, q 4, b 1.001, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-physics_planet-1 · seed 1

- **What it does:** daily forecasts for CB-09-physics_planet-1, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 15, 16, 23, 9, 32, 35; validation gain over ETAS (nats per quake) 0.1403, 0.1401, 0.1449, 0.1215, 0.1537, 0.1515; 93.2 min.
- **Starting line:** ETAS with mu 0.4063, k 10, a 0.1311, c 1, p 1.011, d 1.771, gamma 0.009391, q 4, b 0.9952, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.

## Field · CB-09-physics_planet-2 · seed 2

- **What it does:** daily forecasts for CB-09-physics_planet-2, starting from ETAS and correcting it cell by cell, at most 4x either way.
- **Learned from:** split `cb02_fit`, 9,367 days from 1991-01-01; every midnight reads only the quakes before it.
- **Size:** 143,618 parameters per member, 6 members averaged.
- **Training:** epochs per member 16, 17, 10, 20, 27, 17; validation gain over ETAS (nats per quake) 0.1649, 0.1741, 0.1576, 0.1775, 0.1818, 0.1652; 58.6 min.
- **Starting line:** ETAS with mu 0.3827, k 10, a 0.1591, c 1, p 1.011, d 1.94, gamma 0.004524, q 4, b 1.001, mc 2.
- **Code:** Craton 0.8.0, Field f0.1, config hash `b99e49ae90f8c7f6`.
