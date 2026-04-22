use chrono::{DateTime, Utc};

use super::{CouplingEffect, PathwayStatus};
use crate::detection::FlareOnset;
use crate::feeds::xray::FlareClass;
use crate::feeds::FeedState;

/// Forbush decrease pathway: flare/CME -> cosmic ray flux decrease -> reduced
/// ionization -> suppressed convective initiation.
///
/// Lag: +3 to +8 days from flare onset.
/// Effect: Suppression of severe weather.
///
/// Based on Mironova, Tinsley & Zhou 2011; Tinsley & Zhou 2006.
pub struct ForbushPathway {
    /// Most recent significant flare detection.
    last_flare: Option<FlareRecord>,
    /// Current predicted CR flux decrease (%).
    predicted_cr_decrease: f64,
    /// CME arrival estimate (hours from flare).
    cme_arrival_hours: f64,
    score: f64,
}

struct FlareRecord {
    timestamp: DateTime<Utc>,
    class: FlareClass,
    peak_flux: f64,
    solar_wind_speed: f64,
}

impl ForbushPathway {
    pub fn new() -> Self {
        Self {
            last_flare: None,
            predicted_cr_decrease: 0.0,
            cme_arrival_hours: 0.0,
            score: 0.0,
        }
    }

    pub fn update(&mut self, feeds: &FeedState, flare: Option<&FlareOnset>) {
        // Get current solar wind speed for CME arrival estimate
        let sw_speed = feeds.solar_wind.back().map(|s| s.speed).unwrap_or(400.0); // quiet solar wind default

        // Register new flare if significant (M-class or above)
        if let Some(f) = flare {
            if matches!(f.class, FlareClass::M | FlareClass::X) {
                self.last_flare = Some(FlareRecord {
                    timestamp: f.timestamp,
                    class: f.class,
                    peak_flux: f.peak_flux,
                    solar_wind_speed: sw_speed,
                });
            }
        }

        // Compute score based on active Forbush prediction
        if let Some(ref flare) = self.last_flare {
            let hours_since = (Utc::now() - flare.timestamp).num_seconds() as f64 / 3600.0;

            // CME arrival time estimate (empirical: ~80h at 500km/s, scales inversely)
            self.cme_arrival_hours = 80.0 * (500.0 / flare.solar_wind_speed.max(300.0));

            // Forbush decrease magnitude estimate from flare class
            // X-class: ~5-15% CR decrease, M-class: ~1-5%
            let base_decrease = match flare.class {
                FlareClass::X => 10.0 * (flare.peak_flux / 1e-4).min(3.0),
                FlareClass::M => 3.0 * (flare.peak_flux / 1e-5).min(2.0),
                _ => 0.0,
            };

            // Time window: effect peaks around CME arrival, decays over ~5 days
            let effect_start = self.cme_arrival_hours;
            let effect_peak = effect_start + 24.0;
            let effect_end = effect_start + 192.0; // +8 days total

            if hours_since < effect_start {
                // CME hasn't arrived yet — score based on proximity
                self.predicted_cr_decrease = base_decrease;
                self.score = 0.3 * (hours_since / effect_start).min(1.0);
            } else if hours_since < effect_peak {
                // Rising phase
                let phase = (hours_since - effect_start) / (effect_peak - effect_start);
                self.predicted_cr_decrease = base_decrease * phase;
                self.score = 0.3 + 0.7 * phase;
            } else if hours_since < effect_end {
                // Decay phase
                let phase = 1.0 - (hours_since - effect_peak) / (effect_end - effect_peak);
                self.predicted_cr_decrease = base_decrease * phase.max(0.0);
                self.score = phase.max(0.0);
            } else {
                // Effect has passed
                self.score = 0.0;
                self.predicted_cr_decrease = 0.0;
                self.last_flare = None;
            }
        } else {
            self.score = 0.0;
            self.predicted_cr_decrease = 0.0;
        }
    }

    pub fn status(&self) -> PathwayStatus {
        let details = if let Some(ref flare) = self.last_flare {
            let hours = (Utc::now() - flare.timestamp).num_seconds() as f64 / 3600.0;
            format!(
                "{}-class flare {:.0}h ago, CME arrival ~{:.0}h, predicted CR decrease {:.1}%",
                flare.class.label(),
                hours,
                self.cme_arrival_hours,
                self.predicted_cr_decrease,
            )
        } else {
            "No significant flare activity".into()
        };

        PathwayStatus {
            name: "Forbush Chain".into(),
            active: self.score > 0.1,
            score: self.score,
            effect: CouplingEffect::Suppression,
            lag_hours: (72.0, 192.0),
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_flare_inactive() {
        let p = ForbushPathway::new();
        let status = p.status();
        assert!(!status.active);
        assert_eq!(status.score, 0.0);
        assert_eq!(status.effect, CouplingEffect::Suppression);
    }
}
