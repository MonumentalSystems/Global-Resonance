//! Unified solar state model.
//!
//! Aggregates all solar data streams into a single coherent picture
//! of the Sun's current state and its Earth-directed effects.
//!
//! Layers:
//! 1. **Cycle** — where are we in the 11-year solar cycle? (F10.7, SSN)
//! 2. **Disk** — what's on the visible solar disk? (active regions, spots)
//! 3. **Activity** — what's happening right now? (flares, CMEs, escalation)
//! 4. **Heliosphere** — what's propagating toward Earth? (CME fronts, HSS, shocks)
//! 5. **Geospace** — what's hitting Earth's magnetosphere? (Dst, Kp, solar wind)

pub mod activity;
pub mod cycle;
pub mod disk;
pub mod donki;
pub mod geospace;
pub mod heliosphere;
pub mod physics;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Complete solar state snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct SolarState {
    pub timestamp: DateTime<Utc>,
    pub cycle: cycle::CycleState,
    pub disk: disk::DiskState,
    pub activity: activity::ActivityState,
    pub heliosphere: heliosphere::HeliosphereState,
    pub geospace: geospace::GeospaceState,
    /// Overall threat level (0..1), computed from all layers.
    pub threat_level: f64,
    /// Human-readable summary.
    pub summary: String,
}

impl SolarState {
    /// Compute the overall threat level from all layers.
    pub fn compute_threat(&self) -> f64 {
        // Weighted combination of layer-level threats
        let cycle_factor = self.cycle.activity_level(); // 0..1
        let disk_factor = self.disk.flare_potential(); // 0..1
        let activity_factor = self.activity.current_intensity(); // 0..1
        let helio_factor = self.heliosphere.earth_threat(); // 0..1
        let geo_factor = self.geospace.disturbance_level(); // 0..1

        // Weights: immediate activity and incoming heliosphere matter most
        let threat = 0.05 * cycle_factor
            + 0.20 * disk_factor
            + 0.30 * activity_factor
            + 0.30 * helio_factor
            + 0.15 * geo_factor;

        threat.min(1.0).max(0.0)
    }

    /// Generate a human-readable summary of the current state.
    pub fn generate_summary(&self) -> String {
        let mut parts = Vec::new();

        // Cycle context
        parts.push(format!(
            "Solar Cycle 25: SSN {:.0}, F10.7 {:.0} ({})",
            self.cycle.ssn,
            self.cycle.f10_7,
            self.cycle.phase_label()
        ));

        // Disk
        let n_regions = self.disk.active_regions.len();
        let max_flare_prob = self
            .disk
            .active_regions
            .iter()
            .map(|ar| ar.x_flare_probability)
            .fold(0.0f64, f64::max);
        if n_regions > 0 {
            parts.push(format!(
                "{} active regions, max X-class prob {:.0}%",
                n_regions, max_flare_prob
            ));
        } else {
            parts.push("No active regions".into());
        }

        // Activity
        if let Some(ref flare) = self.activity.latest_flare {
            parts.push(format!(
                "Latest flare: {} at {}",
                flare.class_type,
                flare.peak_time.format("%m-%d %H:%M")
            ));
        }
        if self.activity.cme_count_24h > 0 {
            parts.push(format!("{} CMEs in last 24h", self.activity.cme_count_24h));
        }

        // Heliosphere
        if !self.heliosphere.earth_directed_cmes.is_empty() {
            parts.push(format!(
                "{} Earth-directed CME(s) in transit",
                self.heliosphere.earth_directed_cmes.len()
            ));
        }
        if !self.heliosphere.active_hss.is_empty() {
            parts.push(format!(
                "{} high-speed stream(s)",
                self.heliosphere.active_hss.len()
            ));
        }

        // Geospace
        parts.push(format!(
            "Dst {:.0} nT, Kp {:.1}",
            self.geospace.dst, self.geospace.kp
        ));

        parts.join(". ")
    }
}
