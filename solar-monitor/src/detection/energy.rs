use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Energy-style detector (logit magnitude analog).
///
/// Uses X-ray flux only — electron flux at 5-min cadence creates step
/// artifacts that dominate the multi-channel energy and trigger on
/// data arrival rather than physics. The X-ray channel at 1-min cadence
/// is clean and sufficient.
///
/// Computes windowed z-score on log10(flux). This is similar to the
/// z-score detector but operates on a shorter window (30 samples)
/// and uses absolute deviation (fires on both increases and decreases).
///
/// Tuned against real GOES 7-day data (X1.5 event, 2026-03-30).
#[derive(Debug, Clone)]
pub struct EnergyDetector {
    /// Sliding window of log10(flux) values.
    window: VecDeque<f64>,
    window_size: usize,
    /// Current state.
    xray_flux: f64,
    current_time: Option<DateTime<Utc>>,
    n_samples: usize,
}

impl EnergyDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            xray_flux: 0.0,
            current_time: None,
            n_samples: 0,
        }
    }

    /// Default: 30-sample window.
    pub fn default_detector() -> Self {
        Self::new(30)
    }

    /// Ingest data. electron_flux accepted for API compatibility but not used.
    pub fn ingest(&mut self, xray_flux: f64, _electron_flux: f64, timestamp: DateTime<Utc>) {
        self.xray_flux = xray_flux;
        self.current_time = Some(timestamp);
        self.n_samples += 1;

        let log_flux = if xray_flux > 0.0 {
            xray_flux.log10()
        } else {
            -10.0
        };

        self.window.push_back(log_flux);
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Anomaly score (0..1). Higher = more anomalous.
    /// Windowed absolute z-score on log10(flux).
    pub fn score(&self) -> f64 {
        if self.window.len() < 10 {
            return 0.0;
        }

        let n = self.window.len() as f64;
        let mean: f64 = self.window.iter().sum::<f64>() / n;
        let variance: f64 = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();
        if std < 1e-12 {
            return 0.0;
        }

        let current = self.window.back().copied().unwrap_or(0.0);
        let z = ((current - mean) / std).abs();

        // Sigmoid: threshold at 2 sigma
        1.0 / (1.0 + (-2.0 * (z - 2.0)).exp())
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.xray_flux),
                peak_flux: self.xray_flux,
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
    fn test_quiet_no_anomaly() {
        let mut det = EnergyDetector::default_detector();
        for i in 0..30 {
            det.ingest(5e-7, 100.0, ts(i));
        }
        assert!(det.score() < 0.5);
    }

    #[test]
    fn test_flare_detected() {
        let mut det = EnergyDetector::default_detector();
        // Quiet baseline
        for i in 0..30 {
            det.ingest(5e-7, 100.0, ts(i));
        }
        // Sudden X-class
        det.ingest(1e-4, 100.0, ts(31));
        assert!(det.score() > 0.5);
    }

    #[test]
    fn test_score_bounded() {
        let det = EnergyDetector::default_detector();
        let score = det.score();
        assert!(score >= 0.0 && score <= 1.0);
    }
}
