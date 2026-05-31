use super::{CouplingEffect, PathwayStatus};
use crate::feeds::FeedState;

/// High-Energy Electron Precipitation (HEEP) pathway.
///
/// >2 MeV electron flux increase -> precipitation into mesosphere ->
/// enhanced Jz current -> cloud microphysics effects.
///
/// Lag: +3.7-4 days (Li et al. 2016, CJSS 36:40-48).
/// Effect: Enhancement of severe weather.
pub struct HeepPathway {
    score: f64,
    /// Current electron flux (pfu).
    current_flux: f64,
    /// 6-hour trend (positive = rising).
    flux_trend: f64,
    /// Threshold for significant electron event (pfu).
    threshold: f64,
}

impl HeepPathway {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            current_flux: 0.0,
            flux_trend: 0.0,
            threshold: 1000.0, // 1000 pfu threshold for HEEP events
        }
    }

    pub fn update(&mut self, feeds: &FeedState) {
        if feeds.electrons.is_empty() {
            self.score = 0.0;
            return;
        }

        // Current flux
        self.current_flux = feeds.electrons.back().map(|s| s.flux).unwrap_or(0.0);

        // 6-hour trend: compare current to mean of samples from ~6h ago
        // At 1-min cadence, 6h = 360 samples
        let trend_window = 360.min(feeds.electrons.len());
        if feeds.electrons.len() > trend_window {
            let old_idx = feeds.electrons.len() - trend_window;
            let old_mean: f64 = feeds
                .electrons
                .iter()
                .skip(old_idx)
                .take(30)
                .map(|s| s.flux)
                .sum::<f64>()
                / 30.0_f64.min(feeds.electrons.len() as f64);
            if old_mean > 0.0 {
                self.flux_trend = (self.current_flux - old_mean) / old_mean;
            }
        }

        // Score based on flux level and trend
        if self.current_flux >= self.threshold {
            // Above threshold: score scales with log of excess
            let excess = self.current_flux / self.threshold;
            self.score = (excess.log10() / 2.0).min(1.0).max(0.0);

            // Boost score if trend is rising
            if self.flux_trend > 0.5 {
                self.score = (self.score * 1.3).min(1.0);
            }
        } else if self.current_flux >= self.threshold * 0.5 {
            // Approaching threshold
            self.score = 0.2 * (self.current_flux / self.threshold);
        } else {
            self.score = 0.0;
        }
    }

    pub fn status(&self) -> PathwayStatus {
        let details = format!(
            "Electron flux: {:.0} pfu (threshold: {:.0}), trend: {:+.1}%",
            self.current_flux,
            self.threshold,
            self.flux_trend * 100.0,
        );

        PathwayStatus {
            name: "HEEP".into(),
            active: self.score > 0.1,
            score: self.score,
            effect: CouplingEffect::Enhancement,
            lag_hours: (88.8, 96.0), // 3.7-4 days
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiet_electrons() {
        let p = HeepPathway::new();
        let status = p.status();
        assert!(!status.active);
        assert_eq!(status.effect, CouplingEffect::Enhancement);
    }
}
