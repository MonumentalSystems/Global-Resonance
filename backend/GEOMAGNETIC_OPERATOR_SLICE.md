# Phase B: solar-wind to geomagnetic-field data slice

## Scope

This slice establishes the real-data contract for a future spherical operator.
It does not train or publish a geomagnetic forecast.

`geomagnetic_operator_dataset.py` aligns:

- active-spacecraft NOAA real-time solar-wind magnetic and plasma measurements;
- the corrected GOES 0.1-0.8 nm X-ray channel;
- USGS X/Y/Z ground-magnetometer observations;
- explicit masks for every source, station, component, and spherical coefficient.

The old SWPC `products/solar-wind/*` endpoints used by the repository were
retired in 2026 and now return 404. `operator_data_sources.py` uses the current
RTSW endpoints, selects only `active=true` and `overall_quality=0` records, and
is shared with `/api/solar_wind` so research and dashboard parsing agree.

## Target construction

Static station baselines are removed by forecasting hourly X/Y/Z differences.
For each hour and component, available stations are fitted to a real spherical
harmonic basis through `lmax=2`. A coefficient vector is valid only when at
least nine stations are present and the masked design matrix has full rank.

These coefficients are **network least-squares descriptors**. The USGS network
is geographically uneven and cannot support a claim of full global-field
reconstruction. A later INTERMAGNET/SuperMAG expansion or a masked local graph
operator is required for that.

## Leakage and quality rules

- Missing values remain zero-valued only in storage and always have `mask=false`.
- Only the training block determines climatology and VAR normalization.
- Train, validation, and test blocks are chronological and separated by gaps.
- Dst, AE, and Kp are not upstream driver features because they already encode
  geomagnetic response.
- Electron-contaminated GOES measurements are excluded.
- No interpolation across station outages is performed.

## Controls

The saved metadata reports persistence and training-climatology errors. A ridge
VAR control runs only when enough fully observed training and test rows exist;
otherwise it reports why it is unavailable rather than silently imputing data.

## Current limitation

The replacement NOAA RTSW feeds expose roughly one rolling day. The builder is
therefore suitable now for contract validation and scheduled accumulation, not
for a credible model comparison. Historical Phase B training should pair NASA
OMNI hourly drivers with an audited historical magnetometer archive.

## Run

```bash
python backend/geomagnetic_operator_dataset.py
```

The default 24-hour artifact is written to the git-ignored path
`data/operator/geomagnetic_live_hourly.npz`.

## Live contract validation (2026-08-31 UTC)

The default builder completed against the public services and produced 24
hourly rows. All six driver channels were present. Twelve of thirteen requested
USGS stations returned X/Y/Z data; BOU returned HTTP 404 and remained explicitly
masked. The resulting coverage was:

- drivers: 100%;
- station components: 92.31%;
- fitted `lmax=2` coefficients: 95.83% (23/24 hours; the first temporal
  difference is intentionally unavailable).

The 24-hour control errors were persistence MSE 36,398.32 and training-only
climatology MSE 29,338.79. These values validate execution and masking only;
they are not model-comparison evidence. The ridge VAR correctly reported itself
unavailable because the tiny forward training split had 12 complete rows for
33 features.
