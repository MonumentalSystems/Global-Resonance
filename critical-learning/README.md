# critical-learning

Independent crate for the near-critical learning lineage.

This crate depends on `harmonic-core` for the local Clifford algebra primitives,
but it is intentionally separate from the main simulator-facing model stack.

## Thesis

Self-organizing systems with feedback generically evolve toward the
Kosterlitz-Thouless critical stiffness `J_c = 2/pi`, where coherence and
adaptability are best balanced.

This crate implements the first model scaffold based on that thesis.

## Phase Sweep

The crate includes a small experiment runner that sweeps:

- `J`
- inhibition threshold
- discharge gain
- standing-wave amplitude
- standing-wave cycle / harmonic mode
- temporal harmonic amplitude / frequency
- coupling topology (`ring` vs `complete`)

and reports:

- boundary crossing rate
- suppression fraction
- synchronization
- adaptation
- balance
- explicit `best balance` and `best critical release` summaries

Example:

```bash
cargo run -p critical-learning --bin phase_sweep -- \
  --output-json critical-learning/results/phase_sweep.json \
  --output-md critical-learning/results/phase_sweep.md \
  --topologies ring,complete \
  --standing-wave-amplitudes 0.0,0.6
```

To compare integer cavity modes against a non-canceling `phi`-style mode:

```bash
cargo run -p critical-learning --bin phase_sweep -- \
  --output-json critical-learning/results/phase_sweep_modes.json \
  --output-md critical-learning/results/phase_sweep_modes.md \
  --topologies ring,complete \
  --standing-wave-amplitudes 0.6 \
  --wave-presets theta,phi
```

To compare temporal theta-like drive against a `phi`-style temporal modulation:

```bash
cargo run -p critical-learning --bin phase_sweep -- \
  --output-json critical-learning/results/phase_sweep_temporal.json \
  --output-md critical-learning/results/phase_sweep_temporal.md \
  --topologies ring,complete \
  --standing-wave-amplitudes 0.6 \
  --standing-wave-cycles 1.618034 \
  --temporal-harmonic-amplitudes 0.0,0.6 \
  --temporal-presets theta,phi
```

To compare critical control against brainwave-style bands and a zeta probe:

```bash
cargo run -p critical-learning --bin phase_sweep -- \
  --output-json critical-learning/results/phase_sweep_temporal_controls.json \
  --output-md critical-learning/results/phase_sweep_temporal_controls.md \
  --dt-values 0.03,0.04,0.05 \
  --j-values 0.58,0.63661975,0.66 \
  --standing-wave-amplitudes 0.6 \
  --standing-wave-cycles 1.618034 \
  --temporal-harmonic-amplitudes 0.6 \
  --temporal-presets critical,brainwaves,phi,zeta
```

In the current narrow sweeps, temporal `critical = 2/pi` is the strongest controller for staying near the critical manifold. `phi` remains competitive as a rotational companion constant, while brainwave-style bands and the exploratory `zeta` preset are useful comparators rather than current winners.
