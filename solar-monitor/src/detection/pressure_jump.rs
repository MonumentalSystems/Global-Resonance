use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Solar wind dynamic pressure jump detector.
///
/// Sudden increases in solar wind dynamic pressure (Pdyn = n * m_p * v^2)
/// indicate CME or HSS arrival at Earth's magnetosphere. These are
/// storm sudden commencements (SSCs) that compress the magnetopause
/// and drive geomagnetic activity.
///
/// Detects pressure jumps by monitoring the rate of change of Pdyn.
/// A factor-of-2 increase in <30 minutes is a clear SSC signature.
#[derive(Debug, Clone)]
pub struct PressureJumpDetector {
    /// Recent Pdyn values (nPa).
    window: VecDeque<f64>,
    /// Timestamps for rate calculation.
    times: VecDeque<DateTime<Utc>>,
    window_size: usize,
    /// Current state.
    current_pdyn: f64,
    current_xray: f64,
    current_time: Option<DateTime<Utc>>,
}

impl PressureJumpDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            times: VecDeque::with_capacity(window_size),
            window_size,
            current_pdyn: 2.0,
            current_xray: 0.0,
            current_time: None,
        }
    }

    /// Default: 30-sample window.
    pub fn default_detector() -> Self {
        Self::new(30)
    }

    /// Ingest solar wind data. Computes Pdyn from speed and density.
    pub fn ingest_raw(
        &mut self,
        speed: f64,
        density: f64,
        xray_flux: f64,
        timestamp: DateTime<Utc>,
    ) {
        // Dynamic pressure in nPa: Pdyn = n * m_p * v^2
        // n in cm^-3, v in km/s → Pdyn (nPa) ≈ 1.672e-6 * n * v^2
        let pdyn = 1.672e-6 * density * speed * speed;
        self.ingest(pdyn, xray_flux, timestamp);
    }

    /// Ingest pre-computed dynamic pressure.
    pub fn ingest(&mut self, pdyn: f64, xray_flux: f64, timestamp: DateTime<Utc>) {
        self.current_pdyn = pdyn;
        self.current_xray = xray_flux;
        self.current_time = Some(timestamp);

        self.window.push_back(pdyn);
        self.times.push_back(timestamp);
        while self.window.len() > self.window_size {
            self.window.pop_front();
            self.times.pop_front();
        }
    }

    /// Anomaly score (0..1). Based on pressure ratio (current / baseline).
    pub fn score(&self) -> f64 {
        if self.window.len() < 10 {
            return 0.0;
        }

        // Baseline: mean of first half of window
        let half = self.window.len() / 2;
        let baseline: f64 = self.window.iter().take(half).sum::<f64>() / half as f64;

        if baseline < 0.1 {
            return 0.0;
        }

        // Current: mean of last 3 samples
        let current: f64 = self.window.iter().rev().take(3).sum::<f64>() / 3.0;

        // Ratio: how much has pressure increased?
        let ratio = current / baseline;

        if ratio <= 1.2 {
            return 0.0; // Less than 20% increase = nothing
        }

        // Score: sigmoid around ratio=2 (factor-of-2 jump = SSC)
        1.0 / (1.0 + (-3.0 * (ratio - 2.0)).exp())
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn pressure_ratio(&self) -> f64 {
        if self.window.len() < 10 {
            return 1.0;
        }
        let half = self.window.len() / 2;
        let baseline: f64 = self.window.iter().take(half).sum::<f64>() / half as f64;
        if baseline < 0.1 {
            1.0
        } else {
            self.current_pdyn / baseline
        }
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.current_xray),
                peak_flux: self.current_xray,
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

    fn ts(min: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1700000000 + min * 60, 0).unwrap()
    }

    #[test]
    fn test_quiet_wind() {
        let mut d = PressureJumpDetector::default_detector();
        for i in 0..30 {
            d.ingest(2.0, 5e-7, ts(i)); // steady 2 nPa
        }
        assert!(!d.is_anomalous());
    }

    #[test]
    fn test_pressure_jump() {
        let mut d = PressureJumpDetector::default_detector();
        // Quiet baseline
        for i in 0..15 {
            d.ingest(2.0, 5e-7, ts(i));
        }
        // Sudden jump to 6 nPa (3x increase = clear SSC)
        for i in 15..20 {
            d.ingest(6.0, 5e-7, ts(i));
        }
        assert!(d.score() > 0.3);
        assert!(d.pressure_ratio() > 2.0);
    }

    #[test]
    fn test_score_bounded() {
        let d = PressureJumpDetector::default_detector();
        assert!(d.score() >= 0.0 && d.score() <= 1.0);
    }
}
