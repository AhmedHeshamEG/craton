# Test fixtures

Small excerpts of real data, so every adapter test runs offline. Each file is quoted exactly as
the provider served it (only whole lines were cut).

| File | What | Source | Terms |
|---|---|---|---|
| `ingv/norcia.txt` | Every event INGV located in the central-Italy box, 30 Oct 2016, 06:35-06:50 UTC: the Mw 6.5 Norcia earthquake and the first minutes after it | INGV FDSN event service, ISIDe (doi:10.13127/ISIDE), retrieved 2026-10-20 | CC-BY-4.0 |
| `ngl/DataHoldings.txt` | Header and four lines of NGL's station list (three central-Italy stations, one in Australia) | Nevada Geodetic Laboratory, retrieved 2026-10-20 | as-is; cite Blewitt et al. (2018) |
| `ngl/ARQT.tenv3`, `ngl/CAT0.tenv3` | A few days of two IGS20 daily series around 30 Oct 2016. ARQT drops 0.447 m overnight; CAT0, 40 km south, barely moves | Nevada Geodetic Laboratory, retrieved 2026-10-20 | as-is; cite Blewitt et al. (2018) |
| `tides/reference.csv` | Body-tide displacement at six places, hourly for a day each, from PySolid 0.3.4 and pyTMD 3.0.9 (JPL DE440) | made by `scripts/reference/tides_reference.py` | computed |
