# Historical spherical-operator pilot — 2026-08-31

## Outcome

The historical data path is feasible and reproducible, but this first linear
probe does **not** show an aggregate fixed-pole advantage.

`historical_geomagnetic_operator.py` built 672 hourly examples from 2024-01-01
through 2024-01-29 by pairing:

- NASA OMNI2 hourly upstream solar-wind `|B|`, `By GSM`, `Bz GSM`, proton speed,
  and proton density;
- one-minute USGS X/Y/Z variation measurements reduced to hourly medians;
- an irregular eleven-station network projected independently into nine real
  spherical-harmonic modes through `lmax=2`.

Dst, AE, Kp, and ap are intentionally excluded because they encode the
geomagnetic response rather than an upstream cause. Raw NASA and USGS responses
are cached atomically and recorded with SHA-256 hashes in artifact metadata.

## Source audit

The USGS service accepts `sampling_period=3600` for historical definitive data,
but the 2024 probe returned timestamps at `:29:30` with null X/Y/Z values. The
usable contract is one-minute `variation` data followed by an explicit hourly
median. This is appropriate only for short-timescale field changes: USGS notes
that variation data have arbitrary, slowly drifting baselines. The target is
therefore the hourly temporal difference, never the absolute station field.

The default historical network excludes:

- DED: the January response contained no usable X/Y/Z samples;
- SHU: the January request returned HTTP 404 after bounded retries.

Both observations came from the source-availability audit, not target values.
Users can override the station list for other periods.

## Coverage and split

| Quantity | Result |
| --- | ---: |
| OMNI driver coverage | 100.00% |
| Eleven-station component coverage | 96.39% |
| Valid `lmax=2` coefficient coverage | 95.93% |
| Fully complete hours | 95.83% |

The chronological split is train `[0, 403)`, validation `[404, 537)`, and test
`[538, 672)`, with one-hour gaps. Normalization and fitting never use test data.
Ridge penalties and the single-pole half-life are selected only on validation.

## Controls

All MSE values are on hourly changes of the fitted spherical coefficients.

| Control | Test MSE | Parameters | Interpretation |
| --- | ---: | ---: | --- |
| Persistence | 74,692.21 | 0 | Weak for an hourly-difference target |
| Training climatology | 37,520.05 | 27 | Beats persistence |
| Training-only ridge VAR | 34,989.79 | — | Untuned legacy control |
| Validation-tuned Markov ridge | **31,328.03** | 891 | Best aggregate result |
| Parameter-matched single pole | 31,526.93 | 891 | 0.63% worse; validation selected 3 h |
| Five-pole bank | 31,731.72 | 1,431 | 1.29% worse; diagnostic only |

The multi-pole bank is not parameter-matched, so it cannot support an
architecture-level claim. The single-pole comparison is parameter-matched, but
one 28-day period is still too small for a decisive negative result.

Excluding the constant `l=0` mode, as the current nonlinear spherical operator
does, leaves the conclusion unchanged: Markov MSE is 17,841.01, the selected
single pole is 17,944.83 (0.58% worse), and the pole bank is 17,946.67 (0.59%
worse).

## What the aggregate hides

Memory is strongly component-dependent:

- the parameter-matched 3-hour pole improves X by 3.62% and Y by 0.92%, but
  worsens Z by 4.27%;
- the five-pole bank improves X by 7.77% and Y by 1.01%, but worsens Z by 8.60%;
- the bank improves modes `(l=1,m=1)`, `(2,-1)`, and `(2,0)` by 0.38%, 3.12%,
  and 8.79%, respectively, while degrading the other modes.

This pattern suggests that sharing one scalar-harmonic response model across
X/Y/Z is the wrong inductive bias. X and Y are local horizontal components and Z
is radial; a vector-spherical-harmonic decomposition into radial, poloidal, and
toroidal sectors is better aligned with the physics. Component/sector-specific
pole banks are a more justified next experiment than simply adding more scalar
poles.

## What transfers in each direction

From conventional neural-operator practice into Harmonic GPT / Global
Resonance:

- preserve source masks and provenance through the spectral transform;
- select decay scale and regularization on forward validation only;
- use vector spherical harmonics for vector fields;
- report per-sector and per-degree errors, not only a global mean.

From the harmonic models into conventional operators:

- fixed causal cavities provide interpretable, bounded temporal memory without
  a learned recurrent transition;
- degree- or sector-specific decay gates can expose where response timescales
  differ;
- oscillator coherence can be used as a storm-regime diagnostic rather than as
  an unqualified replacement for the spatial operator.

## Next gate

Do not train the nonlinear spherical operator on this single month yet. First:

1. accumulate at least one full year spanning quiet and storm intervals, with
   an eventual goal of solar-cycle coverage;
2. replace independent scalar X/Y/Z fits with vector spherical harmonics;
3. compare component/sector-specific fixed poles against parameter-matched
   Markov and recurrent controls over storm-stratified forward test blocks;
4. only then run the required three-seed RotationalAdamW neural experiment with
   W&B logging and the results registry.

## Reproduction

```bash
python backend/historical_geomagnetic_operator.py
```

The generated data and raw cache live under `data/operator/` and are ignored by
git. A wider range can be resumed with `--start`, `--end`, and `--output`.
