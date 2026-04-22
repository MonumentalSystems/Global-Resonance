//! Solar disk state — what's on the visible Sun right now?

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// State of the visible solar disk.
#[derive(Debug, Clone, Serialize)]
pub struct DiskState {
    /// Currently tracked active regions.
    pub active_regions: Vec<ActiveRegion>,
    /// Total sunspot area (millionths of hemisphere).
    pub total_spot_area: u32,
    /// NOAA's daily flare probabilities.
    pub flare_probabilities: FlareProbs,
    pub last_update: Option<DateTime<Utc>>,
}

/// A single active region on the solar disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveRegion {
    /// NOAA region number.
    pub region: u32,
    /// Heliographic latitude (degrees, + = North).
    pub latitude: i32,
    /// Heliographic longitude (degrees, + = West).
    pub longitude: i32,
    /// Location string (e.g., "N14W23").
    pub location: String,
    /// Sunspot area (millionths of hemisphere).
    pub area: u32,
    /// McIntosh sunspot classification (e.g., "Dri", "Ekc").
    pub spot_class: String,
    /// Number of individual spots.
    pub number_spots: u32,
    /// Mount Wilson magnetic classification (alpha, beta, beta-gamma, beta-gamma-delta).
    pub mag_class: String,
    /// C-class flare probability (0-100%).
    pub c_flare_probability: f64,
    /// M-class flare probability (0-100%).
    pub m_flare_probability: f64,
    /// X-class flare probability (0-100%).
    pub x_flare_probability: f64,
    /// Proton event probability (0-100%).
    pub proton_probability: f64,
    /// Number of C/M/X-class flares produced.
    pub c_xray_events: u32,
    pub m_xray_events: u32,
    pub x_xray_events: u32,
}

/// NOAA daily flare probabilities (full-disk).
#[derive(Debug, Clone, Serialize)]
pub struct FlareProbs {
    pub c_class_1_day: f64,
    pub m_class_1_day: f64,
    pub x_class_1_day: f64,
    pub proton_1_day: f64,
}

impl DiskState {
    pub fn new() -> Self {
        Self {
            active_regions: Vec::new(),
            total_spot_area: 0,
            flare_probabilities: FlareProbs {
                c_class_1_day: 0.0,
                m_class_1_day: 0.0,
                x_class_1_day: 0.0,
                proton_1_day: 0.0,
            },
            last_update: None,
        }
    }

    /// Overall flare potential of the current disk (0..1).
    pub fn flare_potential(&self) -> f64 {
        // Max per-AR X-class probability, normalized
        let max_x = self
            .active_regions
            .iter()
            .map(|ar| ar.x_flare_probability)
            .fold(0.0f64, f64::max);
        let max_m = self
            .active_regions
            .iter()
            .map(|ar| ar.m_flare_probability)
            .fold(0.0f64, f64::max);

        // Combine: X-class probability dominates
        let prob_factor = (max_x / 30.0).min(1.0) * 0.7 + (max_m / 50.0).min(1.0) * 0.3;

        // Boost for complex magnetic configurations
        let has_delta = self
            .active_regions
            .iter()
            .any(|ar| ar.mag_class.contains("D") || ar.mag_class.contains("delta"));
        let mag_boost = if has_delta { 0.2 } else { 0.0 };

        (prob_factor + mag_boost).min(1.0)
    }

    /// Most dangerous active region (highest X-class probability).
    pub fn most_dangerous_region(&self) -> Option<&ActiveRegion> {
        self.active_regions.iter().max_by(|a, b| {
            a.x_flare_probability
                .partial_cmp(&b.x_flare_probability)
                .unwrap()
        })
    }
}
