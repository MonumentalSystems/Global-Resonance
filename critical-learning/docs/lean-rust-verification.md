# Lean ↔ Rust Verification Plan

## Purpose

The critical-learning lineage should not merely borrow intuition from Lean proofs.
It should expose an explicit proof-to-code contract:

- which formal statements are treated as design constraints
- which Rust modules implement those constraints
- which executable checks are expected to witness alignment

This document defines that bridge.

## Scope

The goal is not to "prove the Rust code in Lean" all at once.
The goal is to verify that the Rust modules preserve the same lawful structure
that the Lean development treats as foundational.

We split this into four layers:

1. algebraic invariants
2. dynamical invariants
3. critical-regime invariants
4. numerical / implementation invariants

## Canonical Lean Sources

Primary Lean files for the first bridge:

- `/home/ubuntu/lean/LeanProofs/LeanProofs/Foundations/UniversalAlgebra.lean`
- `/home/ubuntu/lean/LeanProofs/LeanProofs/Bivectors/CommutatorProperties.lean`
- `/home/ubuntu/lean/LeanProofs/LeanProofs/Bivectors/BivectorSubspace.lean`
- `/home/ubuntu/lean/LeanProofs/LeanProofs/KeyTheorems/MaxwellBivector.lean`
- `/home/ubuntu/lean/LeanProofs/LeanProofs/LieStructure/LoheModel.lean`
- `/home/ubuntu/lean/LeanProofs/LeanProofs/KeyTheorems/RGFlow.lean`

These are the first files that should constrain Rust behavior.

## Canonical Rust Sources

Primary Rust modules for the first bridge:

- [/home/ubuntu/harmonic rust/HarmonicRust/harmonic-core/src/clifford_cl3.rs](/home/ubuntu/harmonic%20rust/HarmonicRust/harmonic-core/src/clifford_cl3.rs)
- [/home/ubuntu/harmonic rust/HarmonicRust/harmonic-core/src/clifford_geodesic.rs](/home/ubuntu/harmonic%20rust/HarmonicRust/harmonic-core/src/clifford_geodesic.rs)
- [/home/ubuntu/harmonic rust/HarmonicRust/harmonic-core/src/gauge.rs](/home/ubuntu/harmonic%20rust/HarmonicRust/harmonic-core/src/gauge.rs)
- [/home/ubuntu/harmonic rust/HarmonicRust/critical-learning/src/model.rs](/home/ubuntu/harmonic%20rust/HarmonicRust/critical-learning/src/model.rs)

## Verification Buckets

### 1. Algebraic invariants

These should hold independent of training or task.

Examples:

- commutator antisymmetry
- bivector closure under commutator, when the theorem requires it
- normalization and state-space constraints
- fixed coupling coefficient assumptions encoded in the lawful update

Verification style:

- deterministic unit tests
- property tests over sampled multivectors
- direct correspondence notes to Lean theorem names

### 2. Dynamical invariants

These check that the implemented update respects the intended law.

Examples:

- lawful update reduces to the expected commutator-driven step
- damping acts on the correct grade component
- state normalization or conservation is preserved within numerical tolerance
- local coupling only uses permitted neighbors / topology

Verification style:

- step-level golden tests
- perturbation-response tests
- regression tests against hand-constructed states

### 3. Critical-regime invariants

These are model-level and are not purely algebraic.

Examples:

- `J_c = 2/pi` is represented exactly as the designated critical target
- `critical_gap = |J - J_c|`
- low-`J`, near-critical, and high-`J` runs produce measurably distinct regimes
- coherence/adaptability diagnostics vary sensibly across the sweep

Verification style:

- integration tests
- sweep tests
- JSONL metric snapshots

### 4. Numerical invariants

These check whether the implementation stays faithful under discretization.

Examples:

- step-size sensitivity
- float32 vs float64 stability where available
- bounded drift under repeated rollout
- no NaN / Inf under lawful parameter ranges

Verification style:

- numerical regression tests
- long-horizon finite checks
- tolerance-based invariance tests

## First Required Contracts

These are the minimum contracts we should encode first.

1. `Commutator antisymmetry`
Lean source:
- `CommutatorProperties.lean`

Rust obligation:
- sampled multivectors satisfy `[a,b] + [b,a] ~= 0`

2. `Bivector closure / grade discipline`
Lean source:
- `BivectorSubspace.lean`
- `UniversalAlgebra.lean`

Rust obligation:
- the parts of the update assumed to live in the bivector sector are explicitly extracted and applied only there

3. `Lawful damping placement`
Lean source:
- dynamical chain derived from the lawful update, plus model design assumptions

Rust obligation:
- damping acts on the intended grade component, not on an accidental transformed term

4. `Critical target constant`
Lean source:
- KT / Lohe reduction side of the theory

Rust obligation:
- `J_c = 2/pi` is a named constant in the implementation and used consistently in diagnostics and losses

5. `Near-critical regime separation`
Lean source:
- `RGFlow.lean`
- `LoheModel.lean`

Rust obligation:
- sweep tests show distinct low / near / high regimes using the same architecture

6. `Lohe to KT critical boundary`
Lean source:
- `LoheModel.lean`
- `RGFlow.lean`

Rust obligation:
- the model treats the commutator-driven Lohe coupling as the lawful source of the KT critical split
- `J_c = 2/pi` appears as a theorem-backed boundary, not a tuned hyperparameter

7. `Inhibition / thresholded discharge`
Lean source:
- `CoulombGas/Fugacity.lean`
- `CoulombGas/SingleVortexEnergy.lean`
- `CoulombGas/VortexRG.lean`
- `DarkEnergy.lean`

Rust obligation:
- inhibition and discharge are made explicit in the update law
- tests verify that raising the threshold suppresses discharge and increases inhibition
- the Rust parameterization is treated as a biological re-expression of proved suppression/proliferation and cutoff behavior
- any remaining gap is about exact parameter mapping, not about the existence of the threshold mechanism itself

## Recommended Workflow

1. Add a machine-readable theorem manifest.
2. Attach each Rust invariant test to one manifest entry.
3. Record whether the test is:
   - algebraic
   - dynamical
   - critical
   - numerical
4. Treat failing invariants as law violations, not just model regressions.

## Practical Rule

When a theorem is too abstract to map directly, do not claim verification yet.
Instead:

- name the Lean theorem
- state the Rust approximation
- state the gap

That keeps the bridge honest.
