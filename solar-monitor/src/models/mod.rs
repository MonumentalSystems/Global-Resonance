//! Solar flare prediction models.
//!
//! Moved from harmonic-core to solar-monitor for consolidated solar research.

pub mod solar_flare;
pub mod solar_flare_v2;
pub mod solar_flare_v2_backward;

pub use solar_flare::{SolarFlareConfig, SolarFlareModel};
pub use solar_flare_v2::{SolarFlareV2, SolarFlareV2Config};
