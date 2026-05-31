//! Geospace state — what's hitting Earth's magnetosphere?

use serde::Serialize;

/// Earth's geospace environment.
#[derive(Debug, Clone, Serialize)]
pub struct GeospaceState {
    /// Dst index (nT). <-50 = storm, <-100 = severe.
    pub dst: f64,
    /// Kp index (0-9). >=5 = storm.
    pub kp: f64,
    /// Current geomagnetic storm level.
    pub storm_level: StormLevel,
    /// Active geomagnetic storms from DONKI.
    pub active_storms: Vec<GeomagStorm>,
}

/// NOAA geomagnetic storm scale (G1-G5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum StormLevel {
    /// Kp < 5, Dst > -30. Quiet.
    Quiet,
    /// Kp 5, Dst -30 to -50. G1 Minor.
    G1Minor,
    /// Kp 6, Dst -50 to -100. G2 Moderate.
    G2Moderate,
    /// Kp 7, Dst -100 to -200. G3 Strong.
    G3Strong,
    /// Kp 8, Dst -200 to -350. G4 Severe.
    G4Severe,
    /// Kp 9, Dst < -350. G5 Extreme.
    G5Extreme,
}

/// An active geomagnetic storm from DONKI.
#[derive(Debug, Clone, Serialize)]
pub struct GeomagStorm {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub kp_indices: Vec<f64>,
    pub peak_kp: f64,
}

impl GeospaceState {
    pub fn new() -> Self {
        Self {
            dst: 0.0,
            kp: 0.0,
            storm_level: StormLevel::Quiet,
            active_storms: Vec::new(),
        }
    }

    /// Disturbance level (0..1).
    pub fn disturbance_level(&self) -> f64 {
        let kp_factor = (self.kp / 9.0).min(1.0);
        let dst_factor = if self.dst < -30.0 {
            ((-self.dst - 30.0) / 300.0).min(1.0)
        } else {
            0.0
        };
        kp_factor.max(dst_factor)
    }

    /// Update storm level from Kp and Dst.
    pub fn update_storm_level(&mut self) {
        self.storm_level = if self.kp >= 9.0 || self.dst < -350.0 {
            StormLevel::G5Extreme
        } else if self.kp >= 8.0 || self.dst < -200.0 {
            StormLevel::G4Severe
        } else if self.kp >= 7.0 || self.dst < -100.0 {
            StormLevel::G3Strong
        } else if self.kp >= 6.0 || self.dst < -50.0 {
            StormLevel::G2Moderate
        } else if self.kp >= 5.0 || self.dst < -30.0 {
            StormLevel::G1Minor
        } else {
            StormLevel::Quiet
        };
    }
}

impl StormLevel {
    pub fn label(&self) -> &'static str {
        match self {
            StormLevel::Quiet => "Quiet",
            StormLevel::G1Minor => "G1 Minor",
            StormLevel::G2Moderate => "G2 Moderate",
            StormLevel::G3Strong => "G3 Strong",
            StormLevel::G4Severe => "G4 Severe",
            StormLevel::G5Extreme => "G5 Extreme",
        }
    }
}
