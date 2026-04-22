use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Z-score based anomaly detector on a sliding window.
///
/// Operates on log10(flux) since X-ray flux spans orders of magnitude.
/// Triggers when current sample exceeds mean + threshold * sigma.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Sliding window of log10(flux) values.
    window: VecDeque<f64>,
    /// Window size (number of samples).
    window_size: usize,
    /// Z-score threshold for anomaly (default: 3.0).
    threshold: f64,
    /// Current state.
    current_zscore: f64,
    current_flux: f64,
    current_time: Option<DateTime<Utc>>,
    /// Whether we're currently in an anomalous state.
    in_flare: bool,
    /// Minimum flux to consider (avoids triggering on noise during quiet sun).
    min_flux_threshold: f64,
}

impl ZScoreDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            threshold,
            current_zscore: 0.0,
            current_flux: 0.0,
            current_time: None,
            in_flare: false,
            // C1.0 = 1e-6 W/m^2 minimum to trigger
            min_flux_threshold: 1e-6,
        }
    }

    /// Default detector: 60-sample window, 3-sigma threshold.
    pub fn default_detector() -> Self {
        Self::new(60, 3.0)
    }

    /// Ingest a new flux sample.
    pub fn ingest(&mut self, flux: f64, timestamp: DateTime<Utc>) {
        self.current_flux = flux;
        self.current_time = Some(timestamp);

        let log_flux = if flux > 0.0 { flux.log10() } else { -10.0 };

        self.window.push_back(log_flux);
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }

        // Need at least 10 samples for meaningful statistics
        if self.window.len() < 10 {
            self.current_zscore = 0.0;
            return;
        }

        let (mean, sigma) = self.mean_sigma();
        if sigma > 1e-12 {
            self.current_zscore = (log_flux - mean) / sigma;
        } else {
            self.current_zscore = 0.0;
        }

        // State transition: enter flare if above threshold, exit if below 1.0
        if self.current_zscore > self.threshold && flux >= self.min_flux_threshold {
            self.in_flare = true;
        } else if self.current_zscore < 1.0 {
            self.in_flare = false;
        }
    }

    /// Is the detector currently in an anomalous (flare) state?
    pub fn is_anomalous(&self) -> bool {
        self.in_flare
    }

    /// Current z-score.
    pub fn zscore(&self) -> f64 {
        self.current_zscore
    }

    /// Anomaly score normalized to 0..1 (sigmoid of z-score).
    pub fn score(&self) -> f64 {
        // Sigmoid centered at threshold
        1.0 / (1.0 + (-2.0 * (self.current_zscore - self.threshold)).exp())
    }

    /// If currently anomalous, return a FlareOnset event.
    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.in_flare {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.current_flux),
                peak_flux: self.current_flux,
                anomaly_score: self.score(),
            })
        } else {
            None
        }
    }

    fn mean_sigma(&self) -> (f64, f64) {
        let n = self.window.len() as f64;
        let mean: f64 = self.window.iter().sum::<f64>() / n;
        let variance: f64 = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1700000000 + secs * 60, 0).unwrap()
    }

    #[test]
    fn test_quiet_sun_no_anomaly() {
        let mut det = ZScoreDetector::default_detector();
        // Feed 60 quiet B-class samples
        for i in 0..60 {
            det.ingest(5e-7, ts(i));
        }
        assert!(!det.is_anomalous());
        assert!(det.score() < 0.5);
    }

    #[test]
    fn test_flare_detected() {
        let mut det = ZScoreDetector::default_detector();
        // Feed 60 quiet samples
        for i in 0..60 {
            det.ingest(5e-7, ts(i));
        }
        // Sudden M-class flare
        det.ingest(3e-5, ts(61));
        assert!(det.is_anomalous());
        assert!(det.score() > 0.5);
        let onset = det.onset_event().unwrap();
        assert_eq!(onset.class, FlareClass::M);
    }

    #[test]
    fn test_score_normalized() {
        let mut det = ZScoreDetector::default_detector();
        for i in 0..60 {
            det.ingest(5e-7, ts(i));
        }
        let score = det.score();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
