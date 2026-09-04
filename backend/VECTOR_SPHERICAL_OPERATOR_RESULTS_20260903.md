# Vector spherical operator pilot — 2026-09-03

## Result

Replacing three independent scalar-harmonic fits with a joint vector spherical
harmonic (VSH) basis changes the memory result, but does not yet justify a
sector-specific nonlinear model.

The basis uses the
[USGS geomagnetic convention](https://www.usgs.gov/programs/geomagnetism/introduction-geomagnetism)
X=north, Y=east, Z=down and maps it to spherical components:

- outward radial: `-Z`;
- southward colatitude (`theta`): `-X`;
- eastward azimuth (`phi`): `Y`.

For every real scalar `Y_lm`, the retained vector sectors are:

- radial: `Y_lm e_r`;
- poloidal: `grad_s(Y_lm) / sqrt(l(l+1))`;
- toroidal: `e_r cross grad_s(Y_lm) / sqrt(l(l+1))`.

At `lmax=2`, this gives 9 radial, 8 poloidal, and 8 toroidal coefficients. The
basis is orthonormal in the dense-sphere test and exactly recovers synthetic
band-limited vector fields.

## Real-data projection

The source is the 672-hour 2024 January OMNI2/USGS artifact from the historical
pilot. A joint masked least-squares fit produced valid 25-mode coefficient
vectors for 96.13% of hours.

The complete eleven-station design matrix has condition number 674.71. Across
the 646 fitted hours, the median and 95th percentile remain 674.71, while the
worst outage geometry reaches 1,915.62. The system is full rank but its US-heavy
geometry is ill-conditioned; low-degree coefficients remain network
descriptors, not a complete global magnetic-field reconstruction.

## Shared temporal controls

All learned linear controls have validation-selected ridge penalties. The
Markov and single-pole controls each have 775 fitted parameters.

| Temporal closure | Test MSE | Change vs Markov |
| --- | ---: | ---: |
| Direct-driver Markov | 59,925.40 | — |
| Shared single pole, 3 h | **59,243.02** | **+1.14%** |
| Shared five-pole bank | 66,548.24 | -11.05% |

The 1.14% test gain is exploratory: the same single pole is slightly worse on
validation (99,327.72 versus 98,539.46). One short forward split cannot
establish robust superiority.

The single pole's test effect is sector-dependent:

| Sector | Change vs shared Markov |
| --- | ---: |
| Radial | -4.74% |
| Poloidal | +2.10% |
| Toroidal | +4.43% |

The vector decomposition therefore localizes the apparent memory benefit to
the tangential field. This signal was obscured when X/Y/Z were treated as three
interchangeable scalar fields.

## Sector-specific controls

Separate sector models were kept parameter-matched by giving every output one
five-channel forcing representation. Validation independently selected decay,
direct/pole mixing, and ridge strength.

| Control | Test MSE | Change vs sector Markov |
| --- | ---: | ---: |
| Sector-specific Markov | **63,629.33** | — |
| Sector-specific single poles | 64,288.55 | -1.04% |
| Sector-specific gated poles | 64,280.20 | -1.02% |

All sectors selected a 3-hour half-life. The gated model selected:

- radial: 50% direct driver / 50% pole state;
- poloidal: 75% direct / 25% pole;
- toroidal: 75% direct / 25% pole.

Each gated choice improved its validation error, but the aggregate advantage
reversed on test. This is evidence of temporal nonstationarity or selection
variance, not a stable sector-specific win.

## Interpretation

The VSH representation itself is the strongest result. It turns the shared
single-pole comparison from a small aggregate loss in scalar XYZ space into a
small test gain and reveals that the benefit is tangential. The larger pole
bank still fails decisively, so adding cavities indiscriminately is not useful.

The next architecture should preserve three distinctions:

1. radial forcing can remain primarily Markovian;
2. poloidal and toroidal sectors may receive a small causal cavity;
3. decay/gating should be regularized across spherical degree rather than tuned
   independently from one short validation block.

Before neural training, expand the station geometry and time span. A global
INTERMAGNET or SuperMAG network would reduce the current conditioning problem,
and at least one year with multiple storms is needed to test whether the
tangential 3-hour response repeats. A later nonlinear experiment must use
three seeds, RotationalAdamW, W&B, checkpoints, and the results registry.

The data-acquisition portion of this gate is now complete. See
`INTERMAGNET_GLOBAL_DATA_2024.md`: the 16-station INTERMAGNET network covers all
527,040 minutes of 2024, remains full-rank at every minute, and reduces the
complete-network VSH condition number from 674.71 to 1.459. The annual temporal
control is now complete; see
`ANNUAL_VECTOR_SPHERICAL_OPERATOR_RESULTS_20260904.md`. It does not replicate
the January pole effect, including in the poloidal and toroidal sectors, so the
one-hour nonlinear pole experiment should not proceed from this evidence.

## Reproduction

```bash
python backend/vector_spherical_operator_pilot.py
```

The generated artifact is
`data/operator/geomagnetic_vector_historical_2024_january.npz` and is ignored by
git.
