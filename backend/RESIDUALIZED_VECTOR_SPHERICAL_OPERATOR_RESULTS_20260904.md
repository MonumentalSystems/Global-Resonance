# Calendar-residualized vector spherical controls — 2026-09-04

## Outcome

The training-only calendar baseline explains the earlier large quiet-time pole
clues. The raw 12-hour poloidal pole improvement falls from **+1.691% overall**
and **+3.035% in quiet hours** to **-0.009% overall** and **+0.002% in quiet
hours** after residualization. The raw 3-hour radial improvement similarly
falls from +1.675% to an exact tie at the validation-selected null limit.

One much smaller residual effect remains. At a 6-hour lead, a 24-hour pole
improves aggregate reconstructed MSE by **0.0053%** and poloidal MSE by
**0.0705%**. Both daily and weekly block intervals remain positive after
correcting for the four aggregate or twelve sector-by-horizon comparisons.
This is a precise but tiny diagnostic result. It is not evidence that a larger
or nonlinear neural operator would improve useful storm prediction.

## Leakage-safe baseline

The baseline was fixed before evaluation:

- 24-hour and 12-hour sine/cosine pairs;
- one annual sine/cosine pair;
- the eight daily-by-annual interaction terms;
- an intercept, for 15 basis functions in total;
- one robust multivariate fit to the 25 VSH coefficients using only the H1
  training block (4,367 complete rows);
- five fixed IRLS iterations with Huber threshold 2.5 and ridge `1e-6`;
- no Kp, Dst, validation coefficients, or test coefficients in the fit.

The robust fit downweighted 6.92% of training rows; the median row weight was
1.0. Activity indices remain evaluation-only labels. The held-out leakage test
also replaces every validation/test coefficient by a value shifted by one
million and verifies that the fitted baseline is bitwise unchanged.

At each forecast target, the known deterministic calendar value can be added
back to the predicted residual. This addition cancels in the error, so
residual-space MSE is exactly the implied reconstructed original-coefficient
MSE reported below.

## Calendar contribution

Compared with the raw Markov control, calendar residualization lowers Q4
reconstructed Markov MSE at every lead:

| Lead | Raw Markov | Residualized Markov | Change |
| ---: | ---: | ---: | ---: |
| 1 h | 330.55 | 323.08 | -2.260% |
| 3 h | 335.80 | 326.59 | -2.743% |
| 6 h | 335.51 | 326.54 | -2.674% |
| 12 h | 342.12 | 326.27 | -4.632% |

For some residual targets, validation selects the exact infinite-ridge/null
limit. This is encoded explicitly rather than approximated with an arbitrary
large ridge. It means the fitted residual predictor should abstain and return
zero; it remains the zero-weight member of the same parameter-matched model.

## Residualized aggregate pole results

| Lead | Markov MSE | Pole MSE / scale | Pole change | Multiplicity-adjusted daily interval |
| ---: | ---: | ---: | ---: | ---: |
| 1 h | **323.08** | 323.56 / 1 h | -0.1482% | [-0.2494%, -0.0488%] |
| 3 h | **326.59** | 327.04 / 1 h | -0.1408% | [-0.2247%, -0.0692%] |
| 6 h | 326.540 / null | **326.522 / 24 h** | **+0.0053%** | **[+0.0005%, +0.0099%]** |
| 12 h | **326.27** | 326.29 / 12 h | -0.0066% | [-0.0256%, +0.0060%] |

The interval is 98.75%, controlling four aggregate leads by Bonferroni. The
6-hour weekly interval is also positive: `[+0.0025%, +0.0091%]`.

## Residualized sector results

| Lead | Sector | Markov | Pole / scale | Change | 99.58% daily interval |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 h | radial | **307.64** | 307.75 / 1 h | -0.0378% | [-0.1087%, +0.0476%] |
| 1 h | poloidal | **413.47** | 414.16 / 1 h | -0.1680% | [-0.9005%, +0.4866%] |
| 1 h | toroidal | 254.50 | **254.40 / 24 h** | +0.0373% | [-0.2537%, +0.2743%] |
| 3 h | radial | 312.11 / null | 312.11 / null | 0.0000% | [0.0000%, 0.0000%] |
| 3 h | poloidal | **415.76** | 415.85 / null | -0.0204% | **[-0.0354%, -0.0070%]** |
| 3 h | toroidal | **251.20** | 251.25 / 1 h | -0.0201% | [-0.1546%, +0.1440%] |
| 6 h | radial | 311.536 | **311.527 / 6 h** | +0.0030% | [-0.0080%, +0.0133%] |
| 6 h | poloidal | 415.181 / null | **414.889 / 24 h** | **+0.0705%** | **[+0.0347%, +0.1192%]** |
| 6 h | toroidal | 254.726 / null | 254.726 / null | 0.0000% | [0.0000%, 0.0000%] |
| 12 h | radial | **311.00** | 311.00 / 3 h | -0.0027% | [-0.0141%, +0.0049%] |
| 12 h | poloidal | **415.00** | 415.03 / 12 h | -0.0093% | [-0.0405%, +0.0142%] |
| 12 h | toroidal | **254.730** | 254.743 / 12 h | -0.0050% | [-0.0210%, +0.0050%] |

The 99.58% interval controls twelve sector-by-horizon comparisons. The 6-hour
poloidal weekly interval is `[+0.0280%, +0.1274%]`. Its validation MSE also has
the same sign: 754.566 for the null Markov solution versus 754.544 for the
selected pole.

## Storm stratification of the remaining effect

| 6-hour poloidal stratum | Markov MSE | Pole MSE | Pole change | Held-out support |
| --- | ---: | ---: | ---: | ---: |
| Quiet | 295.85 | **295.58** | +0.0915% | 2,006 h / 6 episodes |
| Storm | 1969.62 | **1969.04** | +0.0292% | 154 h / 5 episodes |
| Severe storm | 4922.51 | **4921.68** | +0.0168% | 40 h / 3 episodes |

Unlike the original 12-hour clue, the point estimate does not reverse during
storms. But it shrinks with activity and corresponds to less than one MSE unit
against storm errors near 2,000--5,000. The small number of independent events
also prevents a strong storm-response claim.

## Decision

The original large pole effects were predictable quiet-field calendar
structure, not evidence for a vector-spherical storm-memory architecture. The
remaining 6-hour poloidal effect is worth testing on additional years and a
different station panel because it is directionally consistent and survives
the present multiplicity correction. Its magnitude is too small, and its event
support too limited, to pass the gate for a neural pole bank.

The next clean slice is therefore **replication before complexity**: freeze the
same 15-term H1-only baseline and linear controls, then evaluate other years
and/or a leave-station-panel-out network. A nonlinear VSH model should only be
reconsidered if a materially useful storm-stratum gain repeats.

## Reproduction

```bash
python backend/residualized_multihorizon_operator.py
```

The ignored result artifact is
`data/operator/geomagnetic_vector_global_2024_residualized_multihorizon.npz`.
It stores the calendar design, fitted baseline, residual coefficients, raw and
residualized controls, exact null-limit selections, and daily/weekly bootstrap
intervals.
