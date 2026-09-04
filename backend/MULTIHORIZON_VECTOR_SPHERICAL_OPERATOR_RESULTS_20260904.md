# Multi-horizon vector spherical controls — 2026-09-04

## Outcome

Extending the annual control from one hour to fixed 3-, 6-, and 12-hour leads
does not reveal a reliable aggregate fixed-pole advantage. The largest sector
point effect is a 1.69% improvement for a 24-hour pole on the 12-hour poloidal
target, but it does not survive correction across all nine sector-by-horizon
comparisons. Crucially, that effect reverses during storms: it is a 3.04%
improvement in quiet hours and a 1.15% degradation in storm hours.

The result points toward predictable solar-quiet/diurnal poloidal structure,
not the hypothesized multiscale storm-response memory. It does not justify a
nonlinear storm pole model.

## Frozen follow-up

The experiment changes only the forecast lead from the annual one-hour gate:

- leads were fixed in advance at 3, 6, and 12 hours;
- H1 remains training, Q3 validation, and Q4 test, with 24-hour gaps;
- current-driver Markov, single-pole, and signed orthogonal recurrent controls
  retain the same five forcing features and exact parameter matching;
- decay scale and ridge are selected on Q3 aggregate error only;
- every comparison uses identical causal rows;
- Kp and Dst remain evaluation-only labels;
- paired uncertainty uses both UTC-day and UTC-week resampling;
- 98.33% intervals correct the three aggregate horizons;
- 99.44% intervals correct the nine sector-by-horizon follow-ups.

The widened ridge grid spans `1e-4` through `1e6` by full decades. This was
necessary because the original pilot grid ended at a selected boundary. All
results below use the widened grid; no reported model remains at its upper
boundary.

## Aggregate results

| Lead | Markov MSE | Pole MSE / scale | Pole change | 98.33% daily interval | Recurrent change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 h | 335.80 | 335.45 / 24 h | +0.102% | [-0.268%, +0.426%] | +0.074% |
| 6 h | 335.51 | 335.35 / 24 h | +0.050% | [-0.054%, +0.138%] | +0.071% |
| 12 h | **342.12** | 342.24 / 12 h | -0.035% | [-0.155%, +0.039%] | -0.034% |

No pole interval excludes zero. The 6-hour orthogonal recurrence has a positive
weekly interval but a zero-crossing daily interval; this uncertainty-method
sensitivity and its 0.071% magnitude do not support an architecture claim.

The Markov controls continue to beat training climatology:

| Lead | Markov | Climatology | Persistence |
| ---: | ---: | ---: | ---: |
| 3 h | **335.80** | 347.11 | 715.21 |
| 6 h | **335.51** | 346.61 | 743.24 |
| 12 h | **342.12** | 346.38 | 703.31 |

## Sector results

| Lead | Sector | Markov | Pole / scale | Change | 99.44% daily interval |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 h | radial | 329.66 | 324.14 / 24 h | +1.675% | [-0.332%, +3.466%] |
| 3 h | poloidal | 428.53 | 427.33 / 12 h | +0.280% | [-0.543%, +1.019%] |
| 3 h | toroidal | **256.43** | 256.45 / 1 h | -0.007% | [-0.128%, +0.142%] |
| 6 h | radial | 322.84 | 322.84 / 24 h | +0.001% | [-0.149%, +0.123%] |
| 6 h | poloidal | 427.21 | 426.50 / 12 h | +0.166% | [-0.063%, +0.405%] |
| 6 h | toroidal | 260.59 | 260.57 / 24 h | +0.004% | [-0.006%, +0.017%] |
| 12 h | radial | **322.84** | 322.93 / 24 h | -0.028% | [-0.332%, +0.182%] |
| 12 h | poloidal | 440.31 | **432.86 / 24 h** | **+1.691%** | [-0.235%, +3.495%] |
| 12 h | toroidal | **260.60** | 260.61 / 12 h | -0.005% | [-0.021%, +0.004%] |

The 12-hour poloidal result has a positive 95% interval and its sign repeats
between validation and Q4. After correcting across all nine sector/horizon
looks, its daily interval is `[-0.235%, +3.495%]` and its weekly interval is
`[-0.146%, +4.770%]`. It is therefore a clue, not a confirmed effect. It is
also not a storm improvement:

| 12-hour poloidal stratum | Markov MSE | Pole MSE | Pole change |
| --- | ---: | ---: | ---: |
| Quiet | 321.45 | **311.69** | **+3.035%** |
| Storm | **2010.50** | 2033.59 | -1.148% |
| Severe storm | **4921.18** | 5006.65 | -1.737% |

The 3-hour radial point gain has the same split: +3.11% in quiet hours but
-1.59% during storms. Therefore selecting either longer-horizon effect as a
storm-memory success would reverse the actual stratified result.

## Interpretation

The low-degree poloidal field contains strong regular quiet-time structure,
including solar-quiet ionospheric currents and daily geometry. A long
exponential state can act as a crude phase/trend feature for that structure.
During disturbed intervals, the same smoothing suppresses rapid forcing and
degrades the response forecast.

The next clean test was consequently not a larger pole bank. We fit a
training-only diurnal/seasonal quiet-field baseline, subtracted it from the VSH
targets, and reran the same controls on residuals. The 12-hour poloidal and
3-hour radial gains disappeared, identifying calendar structure as their
source. A much smaller 6-hour poloidal residual remained and now requires
cross-year replication; see
[`RESIDUALIZED_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`](RESIDUALIZED_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md).

## Reproduction

```bash
python backend/multihorizon_vector_spherical_operator.py
```

The ignored result artifact is
`data/operator/geomagnetic_vector_global_2024_multihorizon.npz`. It contains
the selected controls and daily/weekly paired uncertainty for every horizon,
sector, and storm stratum.
