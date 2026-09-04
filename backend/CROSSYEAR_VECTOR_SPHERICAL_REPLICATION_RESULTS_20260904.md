# Cross-year vector spherical replication — 2026-09-04

## Outcome

The frozen 2024 residual clue repeats in direction and magnitude on independent
2023 data, but it does **not** achieve confirmatory interval support.

The hypothesis was fixed to a 24-hour exponential driver state predicting the
6-hour poloidal VSH residual. On untouched 2023 Q4, it improves MSE by
**0.0850%**, compared with **0.0705%** in the 2024 discovery year. Its 2023 Q3
validation sign also agrees. However, the 2023 paired 95% interval crosses
zero with both daily and weekly block resampling. The result is therefore
classified as **directional, not interval-confirmed replication**.

This strengthens the case that a very small linear memory signal may exist,
but it still does not pass the gate for a neural pole bank.

## Frozen confirmatory test

Only the discovery hypothesis was tested:

- VSH sector: poloidal;
- forecast lead: 6 hours;
- exponential pole half-life: 24 hours;
- calendar residualizer: the same fixed 15-term daily/annual basis with five
  robust IRLS iterations;
- calendar fitting: H1 coefficients only, independently within each year;
- ridge fitting: H1, with selection on Q3 only;
- final evaluation: one untouched Q4 per year;
- station panel: the same 16 observatories in the same order;
- VSH basis: the same degree-2 labels and coordinate convention;
- Kp and Dst: evaluation-only storm labels.

The 2023 station network is not a copied target artifact. Its minute data and
OMNI drivers were independently downloaded, masked, projected, and aligned.

## Confirmatory result

| Year | Role | Markov MSE | Fixed 24 h pole MSE | Change | Daily 95% interval | Weekly 95% interval |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | discovery | 415.181 | **414.889** | **+0.0705%** | [+0.0446%, +0.1034%] | [+0.0400%, +0.1118%] |
| 2023 | replication | 263.995 | **263.771** | **+0.0850%** | [-0.0521%, +0.2044%] | [-0.0579%, +0.1963%] |

The replication/discovery point-effect ratio is 1.21. The 2023 validation
comparison also favors the pole: 262.559 versus 262.777 MSE, a +0.0828%
improvement. Thus the direction did not arise only after opening Q4.

The exact scale is less stable under exploratory reselection. The full 2023
family selects a 12-hour rather than 24-hour pole for this target, with a
+0.1107% Q4 point improvement. Its 99.58% daily interval is
`[-0.0759%, +0.2800%]`, so it does not supply a multiplicity-corrected sector
confirmation either.

## Storm stratification

| 2023 Q4 stratum | Markov MSE | Fixed pole MSE | Change | Support | Daily 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quiet | 183.105 | **183.008** | +0.0531% | 1,962 h / 7 episodes | [-0.0884%, +0.1894%] |
| Storm | 1243.660 | **1241.897** | +0.1418% | 162 h / 12 episodes | [-0.1266%, +0.3950%] |
| Severe | 2259.355 | **2250.490** | +0.3924% | 23 h / 2 episodes | [+0.1768%, +0.9708%] |

The storm point signs are encouraging, but the storm interval crosses zero and
the severe result comes from only two episodes. Resampling 23 correlated hours
cannot turn that subset into independent-event evidence.

## Data audit

The fixed 16-station network covers all of 2023. The minute artifact contains
525,600 timestamps and has 92.01% complete-vector coverage. SHU and IPM have
substantial gaps (51.88% and 58.50% complete vectors), but masked VSH fitting
remains full-rank throughout:

- full-network condition number: 1.459;
- median observed minute condition number: 1.726;
- 95th percentile: 2.991;
- maximum condition number among fitted hourly targets: 5.032;
- VSH coefficient availability: 99.989%;
- complete OMNI driver availability: 97.637%.

There were no station fetch errors. Raw INTERMAGNET chunks and the OMNI2 file
are cached with SHA-256 provenance in the ignored data directory.

## Calendar behavior across years

The H1-only calendar procedure lowers 2023 Markov MSE at 1, 3, and 6 hours by
1.94%, 2.95%, and 2.10%. At 12 hours it worsens MSE by 0.44%, unlike its 2024
improvement. This is a useful warning: fitting an annual harmonic from only the
first half-cycle does not guarantee stable Q4 extrapolation. It does not affect
the frozen 6-hour conclusion, but a multi-year baseline should use explicitly
causal seasonal estimation rather than quietly pooling future seasons.

## Decision

The effect passes a direction-and-magnitude check but fails the independent
year's interval criterion. No neural experiment is warranted yet. A second
independent year would distinguish a repeatable low-amplitude effect from a
two-year coincidence; a station-panel perturbation would test whether one
observatory or outage pattern carries it.

## Reproduction

```bash
python backend/intermagnet_hapi.py \
  --start 2023-01-01T00:00:00Z \
  --end 2024-01-01T00:00:00Z \
  --output data/operator/geomagnetic_global_2023_minute.npz

python backend/annual_vector_spherical_operator.py \
  --source data/operator/geomagnetic_global_2023_minute.npz \
  --output data/operator/geomagnetic_vector_global_2023_hourly.npz

python backend/residualized_multihorizon_operator.py \
  --source data/operator/geomagnetic_vector_global_2023_hourly.npz \
  --output data/operator/geomagnetic_vector_global_2023_residualized_multihorizon.npz

python backend/crossyear_vector_spherical_replication.py
```

The compact comparison artifact is
`data/operator/geomagnetic_vector_crossyear_replication_2023_2024.npz`. All
data and result artifacts remain ignored because of size and source licensing.
