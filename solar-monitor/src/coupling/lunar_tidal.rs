use chrono::{Datelike, Utc};

use super::{CouplingEffect, PathwayStatus};

/// Lunar tidal pathway (deterministic — no feed data needed).
///
/// Enhanced tidal forcing near new/full moon, especially during
/// April-May-June (AMJ) severe weather season.
///
/// Based on Paper XXVI lunar-tornado correlations.
/// Effect: Enhancement.
pub struct LunarTidalPathway {
    score: f64,
    lunar_phase: f64,
    phase_name: String,
    in_season: bool,
}

impl LunarTidalPathway {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            lunar_phase: 0.0,
            phase_name: String::new(),
            in_season: false,
        }
    }

    pub fn update(&mut self) {
        let now = Utc::now();

        // Lunar phase (0 = new moon, 0.5 = full moon)
        self.lunar_phase = lunar_phase_fraction(now.timestamp() as f64);

        // Phase proximity to new (0.0) or full (0.5)
        // Distance to nearest syzygy (new or full)
        let dist_new = self.lunar_phase.min(1.0 - self.lunar_phase);
        let dist_full = (self.lunar_phase - 0.5).abs();
        let syzygy_distance = dist_new.min(dist_full);

        // Tidal enhancement when within ~3 days of syzygy
        // Lunation = 29.53 days, so 3 days = ~0.1 phase fraction
        let tidal_score = if syzygy_distance < 0.1 {
            1.0 - (syzygy_distance / 0.1)
        } else {
            0.0
        };

        // Season factor: AMJ (April-May-June) is peak tornado season
        let month = now.month();
        self.in_season = (4..=6).contains(&month);
        let season_factor = match month {
            4 => 0.8,
            5 => 1.0,
            6 => 0.8,
            3 | 7 => 0.4,
            _ => 0.1,
        };

        self.score = (tidal_score * season_factor).min(1.0);

        // Phase name
        self.phase_name = if self.lunar_phase < 0.05 || self.lunar_phase > 0.95 {
            "New Moon".into()
        } else if self.lunar_phase < 0.2 {
            "Waxing Crescent".into()
        } else if self.lunar_phase < 0.3 {
            "First Quarter".into()
        } else if (self.lunar_phase - 0.5).abs() < 0.05 {
            "Full Moon".into()
        } else if self.lunar_phase < 0.5 {
            "Waxing Gibbous".into()
        } else if self.lunar_phase < 0.7 {
            "Waning Gibbous".into()
        } else if self.lunar_phase < 0.8 {
            "Last Quarter".into()
        } else {
            "Waning Crescent".into()
        };
    }

    pub fn status(&self) -> PathwayStatus {
        PathwayStatus {
            name: "Lunar Tidal".into(),
            active: self.score > 0.1,
            score: self.score,
            effect: CouplingEffect::Enhancement,
            lag_hours: (0.0, 0.0), // Deterministic, no lag
            details: format!(
                "{} (phase {:.2}), {}",
                self.phase_name,
                self.lunar_phase,
                if self.in_season {
                    "AMJ season active"
                } else {
                    "off-season"
                }
            ),
        }
    }
}

/// Compute lunar phase fraction (0.0 = new moon, 0.5 = full moon).
///
/// Uses Meeus algorithm (simplified). Reference new moon:
/// J2000.0 = 2000-01-06 18:14 UTC (known new moon).
fn lunar_phase_fraction(unix_timestamp: f64) -> f64 {
    // Reference new moon: 2000-01-06 18:14:00 UTC
    const REF_NEW_MOON: f64 = 947181240.0; // Unix timestamp
    const SYNODIC_MONTH: f64 = 29.530588853 * 86400.0; // seconds

    let elapsed = unix_timestamp - REF_NEW_MOON;
    let phase = (elapsed % SYNODIC_MONTH) / SYNODIC_MONTH;
    if phase < 0.0 {
        phase + 1.0
    } else {
        phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lunar_phase_range() {
        let phase = lunar_phase_fraction(1700000000.0);
        assert!(phase >= 0.0 && phase < 1.0);
    }

    #[test]
    fn test_known_new_moon() {
        // Reference new moon should give phase ~0
        let phase = lunar_phase_fraction(947181240.0);
        assert!(phase.abs() < 0.01);
    }

    #[test]
    fn test_pathway_status() {
        let mut p = LunarTidalPathway::new();
        p.update();
        let status = p.status();
        assert_eq!(status.effect, CouplingEffect::Enhancement);
        assert!(status.score >= 0.0 && status.score <= 1.0);
    }
}
