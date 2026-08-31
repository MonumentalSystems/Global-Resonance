# Spherical fixed-pole closure results — 2026-08-30

## Result

The causal fixed-pole closure passed the Phase A synthetic benchmark. On the
untouched 64-step chronological test suffix it reduced mean closed-loop rollout
MSE by **51.33%** relative to the parameter-matched instantaneous spherical
operator.

| Model | Parameters | One-step MSE | 64-step rollout MSE | Rollout NMSE | Final-step MSE |
|---|---:|---:|---:|---:|---:|
| Spherical Markov | 1,206 | 0.11127 +/- 0.01876 | 5.3180 +/- 0.8758 | 0.7667 +/- 0.0042 | 6.5570 +/- 1.5388 |
| Spherical fixed-pole | 1,200 | **0.02528 +/- 0.00412** | **2.5882 +/- 0.4359** | **0.3730 +/- 0.0139** | **4.2109 +/- 0.9145** |

Seeds were 42, 123, and 456. The fixed-pole arm improved rollout MSE in every
seed: 5.1569 -> 2.5998, 6.2632 -> 3.0181, and 4.5339 -> 2.1466.

Both arms preserved the `l=0` mean with exactly zero measured drift. Maximum
predicted/true RMS was 0.771 +/- 0.040 for the fixed-pole arm versus
0.734 +/- 0.038 for the Markov arm. Neither exploded, although both remain
somewhat over-damped.

## Interpretation

This supports a narrow architectural claim: when an observed spherical field
is the collapsed output of several hidden response timescales, retaining those
timescales as causal state is materially better than spending the same
parameters on instantaneous spectral mixing.

It does **not** establish skill on solar, geomagnetic, atmospheric, oceanic, or
seismic data. The generator was intentionally constructed from multiple damped
cavities, so this experiment validates the implementation and makes a real-data
test worthwhile; it does not validate the physical hypotheses in the broader
repository.

The largest absolute errors remain at low degrees `l=1` and `l=2`. Those are
also where the recurrent model delivered its largest gains, consistent with
slow global modes benefiting most from retained state.

## Important implementation finding

The first convergence smoke failed because RotationalAdamW correctly preserved
the row norms of the driver projections, but the architecture gave the
fixed-pole model no independent way to learn physical input amplitude. Adding a
non-decayed scale per pole and spherical degree changed the result without
altering the fixed cavity decay. This amplitude/direction separation is likely
important for real physical fields too.

## Reproducibility

- Base repository commit: `07281f5f920dfc4fb8e901eb1449a51d9df13846`
- Experiment source SHA-256:
  `279b3e56949db80d3aaead8e55f5b01a4f16b12ab3e80c40467095424b4cbce9`
- Grid: 12 x 24, `lmax=5`
- Train / selection validation / untouched test: 96 / 32 / 64 steps
- Candidate half-lives: 1.5, 6, 24, 96 steps
- Optimizer: canonical Harmonic GPT RotationalAdamW parameter split
- Gradient clipping: none
- W&B project: `symbiogenesis`

Official runs:

- Seed 42: [Markov](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/f8zx9jmx), [fixed-pole](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/d4u6g8vh)
- Seed 123: [Markov](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/azvnjufh), [fixed-pole](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/6dm1nmq3)
- Seed 456: [Markov](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/1g12gb2e), [fixed-pole](https://wandb.ai/lisamegawatts-decentralized-intelligence-agency/symbiogenesis/runs/l5ybos2s)

The full machine-readable report and best checkpoints are under the ignored
directory `backend/output/spherical_operator_experiment/`.

## Phase B

The first real-data test should forecast the global geomagnetic response from
upstream solar forcing:

1. Inputs: GOES X-ray/proton/electron, DSCOVR/ACE solar wind, deterministic
   subsolar/lunar geometry, and source-quality masks.
2. Target: degree-truncated global magnetometer field, retaining station masks
   and uncertainty rather than zero-filling outages.
3. Controls: persistence, climatology, vector autoregression, spherical Markov,
   shuffled-time drivers, and source ablations.
4. Split: strict forward chaining by storm, with the most recent block untouched.
5. Metrics: coefficient and station-space error by lead time, spectrum by
   degree, storm-onset calibration, conservation/drift, and missingness stress.

Only after that passes should fair-weather, Schumann, radio, seismic, ocean,
telluric, and wind streams be added. Their distinct cadences should enter as
separate innovations into shared pole state, not as a single resampled feature
table.
