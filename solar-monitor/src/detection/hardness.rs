use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// X-ray hardness ratio detector (short/long channel ratio).
///
/// The ratio of 0.05-0.4nm (short, hard) to 0.1-0.8nm (long, soft) X-ray
/// flux is the classic impulsive flare indicator. During quiet sun the ratio
/// is ~0.03-0.05. During impulsive flares it jumps to 0.2-0.3 because
/// the hot flare plasma emits more hard X-rays first.
///
/// This is what NOAA/SWPC operators watch for flare onset.
///
/// Both channels are at native 1-minute GOES cadence — no interpolation
/// artifacts like the electron data.
#[derive(Debug, Clone)]
pub struct HardnessRatioDetector {
    /// Sliding window of ratio values.
    window: VecDeque<f64>,
    window_size: usize,
    /// Current values.
    current_ratio: f64,
    current_long_flux: f64,
    current_time: Option<DateTime<Utc>>,
    n_samples: usize,
}

impl HardnessRatioDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            current_ratio: 0.0,
            current_long_flux: 0.0,
            current_time: None,
            n_samples: 0,
        }
    }

    /// Default: 60-sample window (1 hour at 1-min cadence).
    pub fn default_detector() -> Self {
        Self::new(60)
    }

    /// Ingest both X-ray channels.
    pub fn ingest(
        &mut self,
        short_flux: f64, // 0.05-0.4nm
        long_flux: f64,  // 0.1-0.8nm
        timestamp: DateTime<Utc>,
    ) {
        self.current_long_flux = long_flux;
        self.current_time = Some(timestamp);
        self.n_samples += 1;

        // Compute ratio (guard against division by zero)
        self.current_ratio = if long_flux > 1e-10 {
            short_flux / long_flux
        } else {
            0.0
        };

        self.window.push_back(self.current_ratio);
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Anomaly score (0..1). Based on windowed z-score of the ratio.
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

        // Only upward deviation matters (hardening = flare)
        let z = (self.current_ratio - mean) / std;
        if z <= 0.0 {
            return 0.0;
        }

        // Sigmoid at 2 sigma
        1.0 / (1.0 + (-2.0 * (z - 2.0)).exp())
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn ratio(&self) -> f64 {
        self.current_ratio
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.current_long_flux),
                peak_flux: self.current_long_flux,
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
    fn test_quiet_ratio() {
        let mut det = HardnessRatioDetector::default_detector();
        // Quiet sun: ratio ~0.04
        for i in 0..60 {
            det.ingest(2e-8, 5e-7, ts(i));
        }
        assert!(!det.is_anomalous());
        assert!((det.ratio() - 0.04).abs() < 0.01);
    }

    #[test]
    fn test_flare_hardening() {
        let mut det = HardnessRatioDetector::default_detector();
        // Quiet baseline
        for i in 0..60 {
            det.ingest(2e-8, 5e-7, ts(i));
        }
        // Flare: ratio jumps to 0.25
        det.ingest(2.5e-5, 1e-4, ts(61));
        assert!(det.score() > 0.5);
    }

    #[test]
    fn test_score_bounded() {
        let det = HardnessRatioDetector::default_detector();
        assert!(det.score() >= 0.0 && det.score() <= 1.0);
    }
}
