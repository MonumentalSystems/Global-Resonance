use super::{CouplingEffect, PathwayStatus};
use crate::feeds::FeedState;

/// Mansurov effect pathway.
///
/// IMF By polarity -> sector-dependent fair-weather potential gradient ->
/// Jz modulation -> cloud microphysics.
///
/// "Away" sector (By > 0 in GSM at Earth) enhances Jz at high latitudes.
/// Lag: ~4-7 days for atmospheric coupling response.
/// Effect: Enhancement when "away" sector dominates.
pub struct MansurovPathway {
    score: f64,
    /// Current IMF By sector classification.
    sector: ImfSector,
    /// Duration of current sector in hours.
    sector_duration_hours: f64,
    /// Mean By over recent window.
    mean_by: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ImfSector {
    /// By > 0 (away from Sun in Parker spiral convention at Earth).
    Away,
    /// By < 0 (toward Sun).
    Toward,
    /// Indeterminate (|By| < threshold).
    Neutral,
}

impl MansurovPathway {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            sector: ImfSector::Neutral,
            sector_duration_hours: 0.0,
            mean_by: 0.0,
        }
    }

    pub fn update(&mut self, feeds: &FeedState) {
        if feeds.solar_wind.len() < 10 {
            self.score = 0.0;
            self.sector = ImfSector::Neutral;
            self.sector_duration_hours = 0.0;
            self.mean_by = 0.0;
            return;
        }

        // Compute mean By over last 6 hours (360 samples at 1-min)
        let window = 360.min(feeds.solar_wind.len());
        let start = feeds.solar_wind.len() - window;
        let by_values: Vec<f64> = feeds.solar_wind.iter().skip(start).map(|s| s.by).collect();
        self.mean_by = by_values.iter().sum::<f64>() / by_values.len() as f64;

        // Classify sector (threshold: |By| > 2 nT for clear sector)
        let threshold = 2.0;
        self.sector = if self.mean_by > threshold {
            ImfSector::Away
        } else if self.mean_by < -threshold {
            ImfSector::Toward
        } else {
            ImfSector::Neutral
        };

        // Estimate sector duration by scanning backward
        self.sector_duration_hours = 0.0;
        let current_sign = self.mean_by.signum();
        for sample in feeds.solar_wind.iter().rev() {
            if sample.by.signum() == current_sign && sample.by.abs() > 1.0 {
                self.sector_duration_hours += 1.0 / 60.0; // 1-min cadence
            } else {
                break;
            }
        }

        // Score: away sector with sufficient duration
        match self.sector {
            ImfSector::Away => {
                // Stronger By and longer duration = higher score
                let magnitude_factor = ((self.mean_by.abs() - threshold) / 5.0).min(1.0).max(0.0);
                let duration_factor = (self.sector_duration_hours / 12.0).min(1.0);
                self.score = (magnitude_factor * 0.6 + duration_factor * 0.4).min(1.0);
            }
            ImfSector::Toward => {
                // Toward sector has weaker but opposite effect
                self.score = 0.0;
            }
            ImfSector::Neutral => {
                self.score = 0.0;
            }
        }
    }

    pub fn status(&self) -> PathwayStatus {
        let details = format!(
            "IMF By: {:.1} nT ({:?}), sector duration: {:.1}h",
            self.mean_by, self.sector, self.sector_duration_hours,
        );

        PathwayStatus {
            name: "Mansurov".into(),
            active: self.score > 0.1,
            score: self.score,
            effect: CouplingEffect::Enhancement,
            lag_hours: (96.0, 168.0), // 4-7 days
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_inactive() {
        let p = MansurovPathway::new();
        let status = p.status();
        assert!(!status.active);
        assert!(!status.active);
    }
}
