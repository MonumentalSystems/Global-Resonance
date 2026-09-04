# INTERMAGNET global vector network — 2024 data audit

## Outcome

The global-data gate is feasible and the first complete annual artifact has
been acquired. The official
[INTERMAGNET HAPI service](https://imag-data.bgs.ac.uk/GIN_V1/hapi) returned
all 527,040 minutes of leap-year 2024 for a 16-station, globally distributed
XYZ network without an account or source error.

The compact local artifact is
`data/operator/geomagnetic_global_2024_minute.npz` (51.2 MB). Its 192 raw,
resumable CSV chunks occupy 427.6 MB under
`data/operator/cache/intermagnet/`. Data are ignored by git; the acquisition
code and audit are tracked.

## Source and access audit

The HAPI catalog contained 3,074 products, including 154 observatories with a
`best-avail/PT1M/xyzf` product. Of these, 118 had catalog coverage spanning the
entire 2024 calendar year. A six-day availability probe around the 8--14 May
2024 geomagnetic storm found 107 stations with at least 95% complete XYZ
vectors.

HAPI supplies:

- station latitude, longitude, elevation, and coverage in `/info`;
- minute XYZ vectors in nT and the independent scalar F measurement;
- definitive, quasi-definitive, reported, and `best-avail` products;
- CSV, JSON, and binary output, with a maximum request duration of 366 days.

The annual artifact uses `best-avail`, which can mix publication qualities.
Stable follow-up results should either freeze this exact hash-audited artifact
or repeat on a fully definitive year. INTERMAGNET's default license is
CC BY-NC 4.0, with institute-specific exceptions; attribution and use must
follow the official [data conditions](https://intermagnet.org/data_conditions.html).

SuperMAG remains a useful independent replication source because it provides
baseline-subtracted perturbations and denser high-latitude coverage. Its
official [download/API page](https://supermag.jhuapl.edu/mag/?tab=api) requires
a registered user ID and acceptance of its Rules of the Road. No SuperMAG
credential or archive was present locally, so it was not used for this slice.

## Geometry-aware station selection

Candidate stations first had to pass the May-storm availability threshold.
Stations were then added greedily to maximize the log determinant of the
grouped vector-spherical-harmonic design. This selection used only coordinates
and source availability, never forecast targets.

At `lmax=2`, the design contains 25 coefficients: 9 radial, 8 poloidal, and 8
toroidal. The selected network is:

| Code | Region | Latitude | Longitude | Annual complete XYZ |
| --- | --- | ---: | ---: | ---: |
| TTB | Brazil | -1.205 | -48.513 | 98.72% |
| CKI | Cocos-Keeling Islands | -12.188 | 96.834 | 99.58% |
| TSU | Namibia | -19.202 | 17.584 | 92.38% |
| TUC | United States | 32.170 | -110.730 | 99.79% |
| NUR | Finland | 60.510 | 24.660 | 94.84% |
| KAK | Japan | 36.232 | 140.186 | 100.00% |
| AIA | Argentine Islands / Antarctica | -65.245 | -64.258 | 59.29% |
| PPT | Tahiti | -17.567 | -149.574 | 98.36% |
| CNB | Australia | -35.320 | 149.360 | 99.98% |
| JAI | India | 26.920 | 75.800 | 98.98% |
| MAW | Antarctica | -67.600 | 62.880 | 99.96% |
| SHU | Alaska / Aleutians | 55.350 | -160.460 | 99.26% |
| IPM | Easter Island | -27.171 | -109.420 | 77.27% |
| GUI | Canary Islands | 28.321 | -16.441 | 90.84% |
| GUA | Guam | 13.590 | 144.870 | 99.20% |
| HBK | South Africa | -25.880 | 27.710 | 99.71% |

AIA and IPM had substantial outages outside the May selection window. They are
retained because the remaining simultaneous stations preserve full rank; their
missing values are masks, never zeros or interpolated measurements. A future
annual-availability optimization can add redundancy rather than selecting on
the later forecast split.

## Annual coverage and conditioning

| Quantity | Result |
| --- | ---: |
| Minute steps | 527,040 |
| Station-component coverage | 94.27% |
| Complete XYZ coverage | 94.26% |
| Full-rank VSH minutes | 100.00% |
| Complete-network condition number | 1.459 |
| Observed condition, median | 1.729 |
| Observed condition, 95th percentile | 2.873 |
| Observed condition, maximum | 26.787 |

For comparison, the earlier eleven-station USGS network had complete-network
condition number 674.71 and reached 1,915.62 during an outage. The global
network therefore resolves the main spatial-identifiability weakness of the
January pilot. It does not by itself establish that degree-2 coefficients are
an unbiased reconstruction of unobserved fine-scale fields.

## Reproduction

```bash
python backend/intermagnet_hapi.py \
  --start 2024-01-01T00:00:00Z \
  --end 2025-01-01T00:00:00Z \
  --output data/operator/geomagnetic_global_2024_minute.npz
```

The downloader:

- requests only `Field_Vector` from the XYZF product;
- stores 31-day raw CSV chunks atomically;
- records SHA-256, byte count, source interval, and station metadata;
- converts HAPI fill values to explicit component masks;
- resumes from cached chunks without repeating network requests;
- reports VSH conditioning over the actual unique outage patterns.

## Next experiment gate

The next controlled experiment was completed without neural training first:

1. reduce minute vectors to robust hourly values and forecast hourly changes;
2. align the already cached 2024 NASA OMNI upstream drivers;
3. fit masked radial, poloidal, and toroidal coefficients at each hour;
4. use storm-stratified forward validation/test blocks;
5. compare parameter-matched Markov, one-pole, and recurrent linear controls;
6. proceed to the required three-seed RotationalAdamW neural experiment only
   if the tangential memory effect repeats across held-out storms.

See `ANNUAL_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`. The tangential
effect did not repeat: the validation-selected pole was 0.094% worse overall
and 0.198% worse during held-out Q4 storm hours, with uncertainty intervals
including zero. The one-hour nonlinear pole gate is therefore closed.

The subsequent 3/6/12-hour test is recorded in
`MULTIHORIZON_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`. It finds no
reliable aggregate pole gain. Its largest sector clue is quiet-time rather than
storm-time, motivating a training-only diurnal/seasonal residual test. That
test is now complete: the large 12-hour poloidal and 3-hour radial clues
disappear after leakage-safe calendar residualization. A much smaller 6-hour
poloidal effect remains for cross-year replication; see
[`RESIDUALIZED_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`](RESIDUALIZED_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md).
