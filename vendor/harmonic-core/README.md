# Vendored `harmonic-core` (minimal subset)

This is a **trimmed local copy** of the `harmonic-core` crate from the private
[`MonumentalSystems/HarmonicRust`](https://github.com/MonumentalSystems/HarmonicRust)
repository, vendored so the `solar-monitor` detector service builds in a hermetic
Docker container without access to that private repo or its `commutator` dependency.

## What's included

- `src/clifford_cl3.rs` — Clifford Cl(3) algebra primitives (geometric product,
  commutator, grade projections, bivector ops). Copied **verbatim** from upstream.
  This file is fully self-contained (only depends on `std`).

## What's omitted (and why it's safe)

Everything else in upstream `harmonic-core`: GPU backends (Metal/CUDA), the
`commutator-field` dependency, optimizers, and the ML model code. None of it is
reachable from the detector code path:

- `critical-learning` → uses only `harmonic_core::clifford_cl3`.
- `solar-monitor` detectors → use `critical-learning` + their own logic; the one
  ML import (`models::solar_flare::SolarFlareModel`) is an **optional** field that
  defaults to `None` and is feature-gated off in this deployment.

## Updating

If upstream `clifford_cl3.rs` changes, re-vendor with:

```bash
gh api repos/MonumentalSystems/HarmonicRust/contents/harmonic-core/src/clifford_cl3.rs \
  -q '.content' | base64 -d > vendor/harmonic-core/src/clifford_cl3.rs
```
