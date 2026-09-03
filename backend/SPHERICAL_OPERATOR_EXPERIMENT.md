# Spherical operator + harmonic temporal closure

## Question

After representing a global field in the correct spherical harmonic basis,
does a causal multiscale cavity state improve forecasts over an equally sized
instantaneous spherical operator?

This is Phase A of a proposed Global Resonance operator stack. It is not an
operational solar, seismic, weather, or hazard forecast.

## Controlled benchmark

`spherical_operator_experiment.py` generates fields on `S^2` from eight
autocorrelated driver streams. Every `(l,m)` coefficient is the observable sum
of four hidden damped cavities. The current field therefore does not uniquely
identify the hidden timescale decomposition.

The A/B comparison is:

1. `SphericalMarkovOperator`: all candidate pole branches are mixed at the
   current step, without persistent state.
2. `SphericalPoleOperator`: the same branches are retained as causal fixed-pole
   state and reconciled against each observed field.

Both models:

- use a real spherical harmonic transform rather than a flat latitude-longitude
  FFT;
- share scalar transition parameters across all orientations `m` of degree
  `l`;
- preserve the `l=0` global mean exactly;
- use no pointwise MLP, softmax attention, dropout, normalization, auxiliary
  loss, data augmentation, or gradient clipping;
- train with Harmonic GPT's canonical `RotationalAdamW` parameter split;
- use chronological validation for checkpoint selection followed by a separate
  untouched test suffix, and report both teacher-forced one-step error and
  closed-loop rollout error;
- run seeds 42, 123, and 456 and log to W&B project `symbiogenesis`.

## Acceptance criteria

The fixed-pole closure is worth taking to real data only if it:

- improves mean validation rollout MSE over the parameter-matched Markov model
  across three seeds;
- does not trade that gain for unstable amplitude growth;
- preserves the mean mode to numerical precision;
- retains its coefficient prediction when synthesized at a higher grid
  resolution.

## Mapping to Global Resonance streams

The first real-data version should keep each source separate until after
timestamp normalization and source-specific quality control.

| Operator role | Candidate streams | Geometry / cadence |
|---|---|---|
| Solar boundary field | HMI/SDO magnetograms, sunspots, flare locations | `S^2`, minutes-hours |
| Upstream forcing | GOES X-ray/protons/electrons, DSCOVR/ACE solar wind | scalar/vector time series, minutes |
| Geomagnetic response | global magnetometers, Kp, Dst, Swarm | irregular stations on `S^2`, minutes-hours |
| Atmospheric response | fair-weather potential gradient, Schumann, lightning, wind | stations/grids on `S^2`, seconds-hours |
| Surface/interior response | seismic, telluric, ocean fields | events/stations/grids, seconds-days |
| Exogenous deterministic forcing | subsolar point, lunar and orbital phase | analytic, continuous |

For irregular station networks, do not interpolate blindly to a raster. Use a
weighted spherical analysis operator with station masks and uncertainty, or a
local DISCO/graph branch, before global harmonic mixing.

## Required real-data controls

- Strict forward-chaining splits, including a final untouched time block.
- Persistence, climatology, linear autoregression, and spherical Markov
  baselines.
- Source ablations and shuffled-time/null-driver controls.
- Missingness masks; no zero-filling that converts outages into physical
  events.
- Metrics by lead time and event prevalence, with calibration for alert tasks.
- No causal or hazard claim from correlation alone.

## Run

```bash
# Terminal 1: launch with log
/home/ms/harmonic-gpt/.venv/bin/python backend/spherical_operator_experiment.py > /tmp/global_s2_operator.log 2>&1 &

# Terminal 2: tail live output
tail -f /tmp/global_s2_operator.log
```

Artifacts are written under
`backend/output/spherical_operator_experiment/` and are ignored by git.

The completed three-seed result is recorded in
`SPHERICAL_OPERATOR_RESULTS_20260830.md`.
