# Annual vector spherical operator controls — 2026-09-04

## Outcome

The annual held-out test does **not** replicate the January tangential
fixed-pole signal at a one-hour forecast horizon. A current-driver linear ARX
model remains the best point estimate overall and during storms. A radial pole
has a quiet-time point gain, but its multiplicity-adjusted interval includes
zero and it degrades storm forecasts. This is evidence of no demonstrated
storm-memory benefit rather than proof that causal cavities can never help.

No neural training was launched. The predeclared gate required a repeating
tangential memory effect before spending a three-seed nonlinear experiment.

## Data and target

The source is the 16-station INTERMAGNET artifact described in
`INTERMAGNET_GLOBAL_DATA_2024.md`, aligned with NASA OMNI2 hourly upstream
drivers.

- 527,040 minute samples cover all of leap-year 2024.
- Each station/component hour is the median of at least 30 real minute samples.
- The target is the next-hour change in 25 degree-2 VSH coefficients: 9 radial,
  8 poloidal, and 8 toroidal.
- Missing station values remain masks. No interpolation or zero-valued fill is
  treated as a measurement.
- Kp and Dst are evaluation-only storm labels. They are never features,
  targets, split boundaries, or hyperparameter selectors.

Hourly station-component coverage is 94.39%; hourly-difference coverage is
94.28%. The coefficient fit succeeds for 99.99% of hours. Its complete-network
condition number is 1.459, with observed median 1.729, 95th percentile 2.873,
and maximum 5.357.

## Frozen evaluation design

The chronological calendar split is:

| Block | Interval | Role |
| --- | --- | --- |
| Train | 1 January--30 June | fit candidate readouts |
| Gap | 1 July | excluded |
| Validation | 2 July--30 September | select ridge and recurrence scale |
| Gap | 1 October | excluded |
| Test | 2 October--31 December | one final evaluation |

The upstream OMNI driver set is `|B|`, `By GSM`, `Bz GSM`, proton speed, and
proton density. Every model uses the current 25-dimensional VSH state plus one
five-dimensional forcing representation and a bias: exactly 775 fitted
parameters. All models use the same rows, including a shared 24-hour causal
warm-up after driver gaps.

The compared forcing representations are:

1. **Markov:** the current five OMNI drivers;
2. **single pole:** a diagonal exponential state with validation-selected
   half-life from 1, 3, 6, 12, or 24 hours;
3. **orthogonal recurrent:** a stable signed-cycle linear state with the same
   five dimensions and validation-selected decay scale.

Ridge regularization is selected on validation. After selection, the model is
refit on train plus validation; Q4 is not evaluated for rejected candidates.

## Aggregate result

| Control | Selected scale | Test MSE | Change vs Markov |
| --- | ---: | ---: | ---: |
| Persistence | — | 648.93 | — |
| Training climatology | — | 347.30 | — |
| Current-driver Markov | — | **330.55** | — |
| Single exponential pole | 1 h | 330.87 | -0.094% |
| Orthogonal recurrent state | 1 h | 330.62 | -0.019% |

The pole was already worse during validation: 510.43 versus 509.25 for Markov.
The orthogonal state was also slightly worse on validation (509.53).

Paired 5,000-sample UTC-day block bootstraps give:

| Comparison | Stratum | Point change | 95% interval | P(improvement) |
| --- | --- | ---: | ---: | ---: |
| Pole vs Markov | all Q4 | -0.094% | [-0.342%, +0.127%] | 24.2% |
| Pole vs Markov | storm | -0.198% | [-0.869%, +0.405%] | 33.6% |
| Pole vs Markov | severe | -0.445% | [-1.429%, +0.435%] | 27.3% |
| Recurrent vs Markov | all Q4 | -0.019% | [-0.345%, +0.255%] | 48.9% |
| Recurrent vs Markov | storm | -0.379% | [-1.237%, +0.427%] | 26.5% |

Storm means use evaluation-only `Kp >= 5` or `Dst <= -50 nT`; severe means
`Kp >= 7` or `Dst <= -100 nT`. The common Q4 rows include 154 storm hours in
five episodes and 40 severe hours in three episodes.

## Sector-specific result

Each sector independently selects its ridge and temporal scale on validation,
while retaining exact Markov/pole/recurrent parameter matching within that
sector.

| Sector | Markov MSE | Pole MSE / scale | Pole change | Recurrent MSE / scale |
| --- | ---: | ---: | ---: | ---: |
| Radial | 311.60 | **309.03 / 3 h** | +0.824% | 309.36 / 24 h |
| Poloidal | **428.43** | 429.46 / 1 h | -0.241% | 428.48 / 1 h |
| Toroidal | **258.33** | 258.43 / 24 h | -0.035% | 258.36 / 24 h |

The pole's multiplicity-adjusted 98.33% daily intervals are:

- radial: [-0.078%, +1.879%];
- poloidal: [-0.984%, +0.440%];
- toroidal: [-0.270%, +0.147%].

The radial point gain comes entirely from quiet hours (+1.63%) and reverses
during storm hours (-1.00%). It is not evidence for storm-response memory.

The short January pilot reported 3-hour pole point gains of 2.10% for poloidal
and 4.43% for toroidal coefficients. Neither the scale nor the magnitude
repeats in annual validation/Q4. The old result was therefore selection or
period variance, not sufficient evidence for a sector-specific pole bank.

## Interpretation and next gate

At one-hour lead, the current VSH state already carries most short geomagnetic
memory. Replacing current upstream forcing with a five-dimensional cavity does
not add reproducible skill. The generic orthogonal recurrence converging to an
almost-Markov 1-hour scale reinforces that conclusion.

The regularization grid in the executable artifact spans `1e-4` through `1e6`
by decades. These numbers supersede the initial report produced with a grid
that ended at a selected boundary.

This result closes the proposed one-hour nonlinear pole experiment. Useful
next tests must change the scientific question rather than increase model
capacity around a failed signal:

1. evaluate 3-, 6-, and 12-hour lead times, where current-state ARX may no
   longer absorb driver history;
2. repeat on multiple fully definitive years with event-blocked validation;
3. test whether removing predictable solar-quiet/diurnal structure exposes a
   storm-driven residual response;
4. retain Markov forcing as the baseline instead of replacing it with a cavity.

The first item is now complete; see
`MULTIHORIZON_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`. A 12-hour
poloidal pole has a nominal quiet-time improvement but worsens held-out storm
hours and does not survive the full nine-comparison correction. This makes
training-only quiet-field residualization the next diagnostic.

## Reproduction

```bash
python backend/annual_vector_spherical_operator.py
```

The ignored artifact is
`data/operator/geomagnetic_vector_global_2024_hourly.npz` (4.1 MB). It contains
hourly station values and masks, VSH coefficients, upstream drivers,
evaluation-only indices, all source provenance, selected controls, and paired
uncertainty summaries.
