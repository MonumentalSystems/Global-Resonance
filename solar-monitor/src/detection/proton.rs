use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Proton flux detector (>=1 MeV integral protons from GOES).
///
/// Solar energetic particle (SEP) events produce 10-100x proton flux
/// increases within minutes to hours of an impulsive flare. The >=1 MeV
/// channel showed a 76x increase during the March 30 X1.5.
///
/// Data arrives at 5-minute cadence. We apply EMA smoothing to reduce
/// step artifacts when interpolating to 1-minute detector cadence.
#[derive(Debug, Clone)]
pub struct ProtonDetector {
    /// EMA-smoothed log10(flux) values.
    window: VecDeque<f64>,
    window_size: usize,
    /// EMA state for smoothing 5-min data.
    ema_value: f64,
    ema_initialized: bool,
    /// EMA decay factor. 0.85 gives ~30-min half-life at 5-min cadence,
    /// or ~6-min half-life at 1-min cadence. Smooths the steps without
    /// killing the signal.
    ema_alpha: f64,
    /// Current state.
    current_flux: f64,
    current_xray_flux: f64,
    current_time: Option<DateTime<Utc>>,
    n_samples: usize,
}

impl ProtonDetector {
    pub fn new(window_size: usize, ema_alpha: f64) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            ema_value: 0.0,
            ema_initialized: false,
            ema_alpha,
            current_flux: 0.0,
            current_xray_flux: 0.0,
            current_time: None,
            n_samples: 0,
        }
    }

    /// Default: 60-sample window, EMA alpha 0.15 (smooth 5-min steps).
    pub fn default_detector() -> Self {
        Self::new(60, 0.15)
    }

    /// Ingest proton flux. Also takes xray_flux for flare class in onset events.
    pub fn ingest(&mut self, proton_flux: f64, xray_flux: f64, timestamp: DateTime<Utc>) {
        self.current_flux = proton_flux;
        self.current_xray_flux = xray_flux;
        self.current_time = Some(timestamp);
        self.n_samples += 1;

        let log_flux = if proton_flux > 0.0 {
            proton_flux.log10()
        } else {
            -2.0
        };

        // EMA smoothing
        if !self.ema_initialized {
            self.ema_value = log_flux;
            self.ema_initialized = true;
        } else {
            self.ema_value = self.ema_alpha * log_flux + (1.0 - self.ema_alpha) * self.ema_value;
        }

        self.window.push_back(self.ema_value);
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }
    }

    /// Anomaly score (0..1). Windowed z-score on EMA-smoothed log proton flux.
    /// Only upward deviations (flux increase = SEP event).
    pub fn score(&self) -> f64 {
        if self.window.len() < 15 {
            return 0.0;
        }

        let n = self.window.len() as f64;
        let mean: f64 = self.window.iter().sum::<f64>() / n;
        let variance: f64 = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();
        if std < 1e-12 {
            return 0.0;
        }

        let z = (self.ema_value - mean) / std;
        if z <= 0.0 {
            return 0.0;
        }

        // Sigmoid at 2 sigma
        1.0 / (1.0 + (-2.0 * (z - 2.0)).exp())
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn smoothed_flux(&self) -> f64 {
        10.0_f64.powf(self.ema_value)
    }

    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.current_xray_flux),
                peak_flux: self.current_xray_flux,
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
    fn test_quiet_protons() {
        let mut det = ProtonDetector::default_detector();
        for i in 0..60 {
            det.ingest(0.3, 5e-7, ts(i));
        }
        assert!(!det.is_anomalous());
    }

    #[test]
    fn test_sep_event() {
        let mut det = ProtonDetector::default_detector();
        // Quiet baseline
        for i in 0..60 {
            det.ingest(0.3, 5e-7, ts(i));
        }
        // SEP: 76x increase
        for i in 60..70 {
            det.ingest(15.0, 1e-4, ts(i));
        }
        assert!(det.score() > 0.3);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut det = ProtonDetector::new(30, 0.15);
        // Feed constant then step change
        for i in 0..30 {
            det.ingest(1.0, 5e-7, ts(i));
        }
        let before = det.smoothed_flux();
        det.ingest(100.0, 5e-7, ts(30));
        let after = det.smoothed_flux();
        // EMA should smooth: after should be between 1 and 100
        assert!(after > before);
        assert!(after < 100.0);
    }
}
