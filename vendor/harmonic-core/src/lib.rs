//! Minimal vendored subset of `harmonic-core`.
//!
//! Upstream: MonumentalSystems/HarmonicRust (private). The full crate carries
//! GPU backends, the `commutator-field` dependency, and ML model code — none of
//! which is reachable from the solar-monitor detector path. The detectors only
//! need the self-contained Clifford Cl(3) primitives, so that single module is
//! vendored here verbatim to keep the Docker build hermetic.
//!
//! If you need the full engine, depend on the upstream crate instead.

pub mod clifford_cl3;
