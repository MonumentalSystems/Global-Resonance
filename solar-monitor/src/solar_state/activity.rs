//! Current solar activity — what's happening right now?

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Current activity state.
#[derive(Debug, Clone, Serialize)]
pub struct ActivityState {
    /// Latest significant flare (if any in last 24h).
    pub latest_flare: Option<FlareEvent>,
    /// All flares in last 24h.
    pub flares_24h: Vec<FlareEvent>,
    /// CME count in last 24h.
    pub cme_count_24h: usize,
    /// CME count in last 7 days.
    pub cme_count_7d: usize,
    /// Current X-ray flux (W/m^2, 0.1-0.8nm).
    pub xray_flux: f64,
    /// Current X-ray background level.
    pub xray_background: XrayBackground,
    /// Our escalation level.
    pub escalation_level: String,
}

/// A flare event from DONKI.
#[derive(Debug, Clone, Serialize)]
pub struct FlareEvent {
    pub begin_time: DateTime<Utc>,
    pub peak_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub class_type: String,
    /// Source location on disk (e.g., "N14W23").
    pub source_location: String,
    /// Associated active region number.
    pub active_region: Option<u32>,
    /// Linked CME IDs (from DONKI).
    pub linked_cme_ids: Vec<String>,
    /// Linked SEP IDs.
    pub linked_sep_ids: Vec<String>,
}

/// Background X-ray level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum XrayBackground {
    /// A-class (<1e-7 W/m^2) — very quiet.
    A,
    /// B-class (1e-7 to 1e-6) — quiet.
    B,
    /// C-class (1e-6 to 1e-5) — active.
    C,
    /// M-class (1e-5 to 1e-4) — very active.
    M,
}

impl ActivityState {
    pub fn new() -> Self {
        Self {
            latest_flare: None,
            flares_24h: Vec::new(),
            cme_count_24h: 0,
            cme_count_7d: 0,
            xray_flux: 0.0,
            xray_background: XrayBackground::B,
            escalation_level: "QUIET".into(),
        }
    }

    /// Current intensity (0..1).
    pub fn current_intensity(&self) -> f64 {
        let flux_factor = if self.xray_flux >= 1e-4 {
            1.0 // X-class
        } else if self.xray_flux >= 1e-5 {
            0.7 + 0.3 * (self.xray_flux / 1e-4).log10().min(1.0).max(0.0)
        } else if self.xray_flux >= 1e-6 {
            0.3 + 0.4 * (self.xray_flux / 1e-5).log10().min(1.0).max(0.0)
        } else {
            0.1
        };

        let flare_factor = match self.flares_24h.len() {
            0 => 0.0,
            1..=2 => 0.3,
            3..=5 => 0.6,
            _ => 0.9,
        };

        (flux_factor * 0.6 + flare_factor * 0.4).min(1.0)
    }
}
