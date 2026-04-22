use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Rate-of-change (derivative) detector.
///
/// Monitors dF/dt of X-ray flux. Flare onsets have characteristic
/// rapid rise times (minutes for impulsive flares). This detector
/// catches the rising edge even before the flux reaches anomalous
/// absolute levels.
///
/// Uses finite difference on log10(flux) with a smoothing window
/// to reduce noise.
#[derive(Debug, Clone)]
pub struct RateOfChangeDetector {
    /// Recent log10(flux) values for derivative computation.
    values: VecDeque<f64>,
    /// Timestamps corresponding to values.
    times: VecDeque<DateTime<Utc>>,
    /// Window for derivative smoothing.
    smooth_window: usize,
    /// Threshold for rate anomaly (log10 units per minute).
    threshold: f64,
    /// Current state.
    current_rate: f64,
    current_flux: f64,
    current_time: Option<DateTime<Utc>>,
    /// Running statistics on rate for adaptive thresholding.
    rate_window: VecDeque<f64>,
    rate_window_size: usize,
}

impl RateOfChangeDetector {
    pub fn new(smooth_window: usize, threshold: f64) -> Self {
        Self {
            values: VecDeque::with_capacity(smooth_window + 1),
            times: VecDeque::with_capacity(smooth_window + 1),
            smooth_window,
            threshold,
            current_rate: 0.0,
            current_flux: 0.0,
            current_time: None,
            rate_window: VecDeque::with_capacity(120),
            rate_window_size: 120,
        }
    }

    /// Default: 8-sample smoothing, threshold 0.08 log10-units/min.
    /// Tuned against real GOES 7-day data (X1.5 event, 2026-03-30).
    /// X1.5 peak rate was 0.105 — threshold at 0.08 gives clean detection
    /// during the rise phase with margin.
    pub fn default_detector() -> Self {
        Self::new(8, 0.08)
    }

    pub fn ingest(&mut self, flux: f64, timestamp: DateTime<Utc>) {
        self.current_flux = flux;
        self.current_time = Some(timestamp);

        let log_flux = if flux > 0.0 { flux.log10() } else { -10.0 };

        self.values.push_back(log_flux);
        self.times.push_back(timestamp);

        while self.values.len() > self.smooth_window + 1 {
            self.values.pop_front();
            self.times.pop_front();
        }

        if self.values.len() < 2 {
            self.current_rate = 0.0;
            return;
        }

        // Smoothed derivative: (mean of last N/2 - mean of first N/2) / dt
        let n = self.values.len();
        let half = n / 2;
        let first_half: f64 = self.values.iter().take(half).sum::<f64>() / half as f64;
        let second_half: f64 = self.values.iter().skip(n - half).sum::<f64>() / half as f64;

        let dt_secs = (self.times[n - 1] - self.times[0]).num_seconds() as f64;
        let dt_min = if dt_secs > 0.0 { dt_secs / 60.0 } else { 1.0 };

        self.current_rate = (second_half - first_half) / dt_min;

        // Track rate statistics
        self.rate_window.push_back(self.current_rate);
        while self.rate_window.len() > self.rate_window_size {
            self.rate_window.pop_front();
        }
    }

    pub fn is_anomalous(&self) -> bool {
        self.current_rate > self.threshold
    }

    /// Anomaly score (0..1).
    pub fn score(&self) -> f64 {
        if self.current_rate <= 0.0 {
            return 0.0;
        }
        // Sigmoid centered at threshold
        1.0 / (1.0 + (-4.0 * (self.current_rate / self.threshold - 1.0)).exp())
    }

    pub fn rate(&self) -> f64 {
        self.current_rate
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1700000000 + secs * 60, 0).unwrap()
    }

    #[test]
    fn test_steady_no_rate() {
        let mut det = RateOfChangeDetector::default_detector();
        for i in 0..10 {
            det.ingest(5e-7, ts(i));
        }
        assert!(!det.is_anomalous());
        assert!(det.rate().abs() < 0.01);
    }

    #[test]
    fn test_rapid_rise_detected() {
        let mut det = RateOfChangeDetector::default_detector();
        // Steady baseline
        for i in 0..5 {
            det.ingest(5e-7, ts(i));
        }
        // Rapid rise over 3 minutes
        det.ingest(5e-6, ts(6));
        det.ingest(5e-5, ts(7));
        det.ingest(5e-4, ts(8));
        assert!(det.rate() > 0.0);
    }

    #[test]
    fn test_score_bounded() {
        let det = RateOfChangeDetector::default_detector();
        assert!(det.score() >= 0.0 && det.score() <= 1.0);
    }
}
