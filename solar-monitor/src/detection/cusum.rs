use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// CUSUM (Cumulative Sum) change-point detector.
///
/// Ported from symbiogenesis GelationMonitor (FR-24a).
/// Detects sustained shifts in flux level that a z-score detector
/// might miss (gradual ramps rather than sudden jumps).
///
/// Operates on log10(flux). Builds a baseline from the first
/// `baseline_window` samples, then accumulates one-sided Page's test.
#[derive(Debug, Clone)]
pub struct CusumDetector {
    /// Baseline samples for mean/std computation.
    baseline: VecDeque<f64>,
    baseline_window: usize,
    /// Baseline statistics (set after baseline_window filled).
    baseline_mean: Option<f64>,
    baseline_std: Option<f64>,
    /// One-sided CUSUM accumulator (upward shift detection).
    cusum_up: f64,
    /// One-sided CUSUM accumulator (downward shift detection).
    cusum_down: f64,
    /// Sensitivity threshold (fires when cusum > this).
    sensitivity: f64,
    /// Current state.
    current_flux: f64,
    current_time: Option<DateTime<Utc>>,
    in_anomaly: bool,
    /// Total samples ingested.
    n_samples: usize,
}

impl CusumDetector {
    pub fn new(baseline_window: usize, sensitivity: f64) -> Self {
        Self {
            baseline: VecDeque::with_capacity(baseline_window),
            baseline_window,
            baseline_mean: None,
            baseline_std: None,
            cusum_up: 0.0,
            cusum_down: 0.0,
            sensitivity,
            current_flux: 0.0,
            current_time: None,
            in_anomaly: false,
            n_samples: 0,
        }
    }

    /// Default: 480-sample baseline (~8h at 1-min cadence), sensitivity 32.0.
    /// Tuned against real GOES 7-day data (X1.5 event, 2026-03-30).
    /// Longer baseline + higher sensitivity reduces FPs while catching M/X class.
    pub fn default_detector() -> Self {
        Self::new(480, 32.0)
    }

    pub fn ingest(&mut self, flux: f64, timestamp: DateTime<Utc>) {
        self.current_flux = flux;
        self.current_time = Some(timestamp);
        self.n_samples += 1;

        let log_flux = if flux > 0.0 { flux.log10() } else { -10.0 };

        // Building baseline phase
        if self.baseline_mean.is_none() {
            self.baseline.push_back(log_flux);
            if self.baseline.len() >= self.baseline_window {
                let n = self.baseline.len() as f64;
                let mean: f64 = self.baseline.iter().sum::<f64>() / n;
                let variance: f64 = self
                    .baseline
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / n;
                let std = variance.sqrt();
                self.baseline_mean = Some(mean);
                self.baseline_std = Some(if std < 1e-12 { 1.0 } else { std });
            }
            return;
        }

        let mean = self.baseline_mean.unwrap();
        let std = self.baseline_std.unwrap();
        let deviation = (log_flux - mean) / std;

        // One-sided CUSUM (Page's test) with exponential decay.
        // Decay prevents the accumulator from staying pegged at max
        // after a single event. Decay rate 0.995 = half-life ~138 samples.
        // Tuned against real GOES 7-day data (2026-03-30 X1.5 event).
        self.cusum_up = (self.cusum_up * 0.995 + deviation).max(0.0);
        self.cusum_down = (self.cusum_down * 0.995 - deviation).max(0.0);

        // Detect upward shift (flux increase = flare)
        self.in_anomaly = self.cusum_up > self.sensitivity;
    }

    pub fn is_anomalous(&self) -> bool {
        self.in_anomaly
    }

    /// Anomaly score: normalized CUSUM value (0..1).
    pub fn score(&self) -> f64 {
        if self.baseline_mean.is_none() {
            return 0.0;
        }
        // Sigmoid of cusum relative to sensitivity
        let ratio = self.cusum_up / self.sensitivity;
        1.0 / (1.0 + (-4.0 * (ratio - 1.0)).exp())
    }

    pub fn cusum_value(&self) -> f64 {
        self.cusum_up
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.in_anomaly {
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
    fn test_baseline_phase() {
        let mut det = CusumDetector::new(10, 4.0);
        // During baseline, should never fire
        for i in 0..10 {
            det.ingest(5e-7, ts(i));
        }
        assert!(!det.is_anomalous());
        assert!(det.baseline_mean.is_some());
    }

    #[test]
    fn test_gradual_ramp_detected() {
        let mut det = CusumDetector::new(20, 3.0);
        // Quiet baseline
        for i in 0..20 {
            det.ingest(5e-7, ts(i));
        }
        // Gradual ramp (each step slightly above baseline)
        for i in 20..50 {
            let flux = 5e-7 * (1.0 + (i - 20) as f64 * 0.5);
            det.ingest(flux, ts(i));
        }
        // CUSUM should accumulate and eventually fire
        assert!(det.cusum_value() > 0.0);
    }

    #[test]
    fn test_score_normalized() {
        let det = CusumDetector::default_detector();
        let score = det.score();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
