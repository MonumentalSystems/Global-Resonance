use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Multi-channel correlation detector.
///
/// Monitors cross-correlation between X-ray, electron, and IMF B-field
/// magnitude channels. During normal conditions, these channels have stable
/// relationships. Flare events break the correlation (B-field changes first
/// via IMF, X-ray rises next, electrons follow with a lag). A decorrelation
/// spike indicates an anomalous event.
///
/// Also monitors channel divergence (Mahalanobis-like distance in
/// 3D flux space when B-field is available, 2D otherwise).
#[derive(Debug, Clone)]
pub struct MultichannelDetector {
    /// Recent (log_xray, log_electron) pairs.
    pairs: VecDeque<(f64, f64)>,
    /// Recent B-field magnitudes (when available).
    b_magnitudes: VecDeque<f64>,
    window_size: usize,
    /// Running correlation coefficient (X-ray vs electron).
    correlation: f64,
    /// Running correlation (X-ray vs B-field magnitude).
    corr_xb: f64,
    /// Baseline correlations (computed from first window).
    baseline_corr: Option<f64>,
    baseline_corr_xb: Option<f64>,
    /// Current divergence from joint distribution center.
    divergence: f64,
    /// Running mean and covariance for Mahalanobis.
    mean_x: f64,
    mean_e: f64,
    mean_b: f64,
    var_x: f64,
    var_e: f64,
    var_b: f64,
    cov_xe: f64,
    cov_xb: f64,
    n_samples: usize,
    n_b_samples: usize,
    /// Current state.
    current_flux: f64,
    current_time: Option<DateTime<Utc>>,
}

impl MultichannelDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            pairs: VecDeque::with_capacity(window_size),
            b_magnitudes: VecDeque::with_capacity(window_size),
            window_size,
            correlation: 0.0,
            corr_xb: 0.0,
            baseline_corr: None,
            baseline_corr_xb: None,
            divergence: 0.0,
            mean_x: 0.0,
            mean_e: 0.0,
            mean_b: 0.0,
            var_x: 0.0,
            var_e: 0.0,
            var_b: 0.0,
            cov_xe: 0.0,
            cov_xb: 0.0,
            n_samples: 0,
            n_b_samples: 0,
            current_flux: 0.0,
            current_time: None,
        }
    }

    /// Default: 120-sample window.
    pub fn default_detector() -> Self {
        Self::new(120)
    }

    pub fn ingest(&mut self, xray_flux: f64, electron_flux: f64, timestamp: DateTime<Utc>) {
        self.current_flux = xray_flux;
        self.current_time = Some(timestamp);

        let log_x = if xray_flux > 0.0 {
            xray_flux.log10()
        } else {
            -10.0
        };
        let log_e = if electron_flux > 0.0 {
            electron_flux.log10()
        } else {
            -2.0
        };

        self.pairs.push_back((log_x, log_e));
        while self.pairs.len() > self.window_size {
            self.pairs.pop_front();
        }

        // Online mean/variance/covariance (Welford's)
        self.n_samples += 1;
        let n = self.n_samples as f64;
        let dx = log_x - self.mean_x;
        let de = log_e - self.mean_e;
        self.mean_x += dx / n;
        self.mean_e += de / n;
        // Update covariance (using old mean for x, new mean for e)
        self.cov_xe += dx * (log_e - self.mean_e) * (n - 1.0) / n;
        self.var_x += dx * (log_x - self.mean_x);
        self.var_e += de * (log_e - self.mean_e);

        // Compute windowed correlation
        if self.pairs.len() >= 20 {
            self.correlation = self.windowed_correlation();

            // Set baseline after first full window
            if self.baseline_corr.is_none() && self.pairs.len() >= self.window_size {
                self.baseline_corr = Some(self.correlation);
            }
        }

        // Mahalanobis-like divergence from running distribution
        if self.n_samples > 20 {
            let sx = (self.var_x / (n - 1.0)).sqrt();
            let se = (self.var_e / (n - 1.0)).sqrt();
            if sx > 1e-12 && se > 1e-12 {
                let zx = (log_x - self.mean_x) / sx;
                let ze = (log_e - self.mean_e) / se;
                self.divergence = (zx * zx + ze * ze).sqrt();
            }
        }
    }

    /// Ingest with B-field vector for 3-channel correlation monitoring.
    ///
    /// During flare onset, the X-ray/electron/B-field correlation structure
    /// breaks: B changes first (IMF disturbance), then X-ray (flare emission),
    /// then electrons (particle acceleration). The 3-channel Mahalanobis
    /// distance captures this decorrelation cascade.
    pub fn ingest_with_bfield(
        &mut self,
        xray_flux: f64,
        electron_flux: f64,
        bx: f64,
        by: f64,
        bz: f64,
        timestamp: DateTime<Utc>,
    ) {
        // Standard 2-channel ingest first.
        self.ingest(xray_flux, electron_flux, timestamp);

        // B-field magnitude channel.
        let bt = (bx * bx + by * by + bz * bz).sqrt();
        let log_b = if bt > 0.1 { bt.log10() } else { -1.0 };

        self.b_magnitudes.push_back(log_b);
        while self.b_magnitudes.len() > self.window_size {
            self.b_magnitudes.pop_front();
        }

        // Online stats for B channel.
        self.n_b_samples += 1;
        let n = self.n_b_samples as f64;
        let db = log_b - self.mean_b;
        self.mean_b += db / n;
        self.var_b += db * (log_b - self.mean_b);

        // X-B covariance.
        let log_x = if xray_flux > 0.0 {
            xray_flux.log10()
        } else {
            -10.0
        };
        let dx = log_x - self.mean_x;
        self.cov_xb += dx * (log_b - self.mean_b) * (n - 1.0) / n;

        // X-B windowed correlation.
        if self.b_magnitudes.len() >= 20 && self.pairs.len() >= 20 {
            self.corr_xb = self.windowed_correlation_xb();

            if self.baseline_corr_xb.is_none() && self.b_magnitudes.len() >= self.window_size {
                self.baseline_corr_xb = Some(self.corr_xb);
            }
        }

        // Upgrade divergence to 3D when B is available.
        if self.n_b_samples > 20 && self.n_samples > 20 {
            let sb = (self.var_b / (n - 1.0)).sqrt();
            let sx = (self.var_x / (self.n_samples as f64 - 1.0)).sqrt();
            let se = (self.var_e / (self.n_samples as f64 - 1.0)).sqrt();
            if sx > 1e-12 && se > 1e-12 && sb > 1e-12 {
                let zx = (log_x - self.mean_x) / sx;
                let log_e = self.pairs.back().map(|(_, e)| *e).unwrap_or(-2.0);
                let ze = (log_e - self.mean_e) / se;
                let zb = (log_b - self.mean_b) / sb;
                // 3D Mahalanobis (diagonal approximation).
                self.divergence = (zx * zx + ze * ze + zb * zb).sqrt();
            }
        }
    }

    fn windowed_correlation_xb(&self) -> f64 {
        let n = self.b_magnitudes.len().min(self.pairs.len());
        if n < 3 {
            return 0.0;
        }
        // Pair up the most recent n samples.
        let x_vals: Vec<f64> = self.pairs.iter().rev().take(n).map(|(x, _)| *x).collect();
        let b_vals: Vec<f64> = self.b_magnitudes.iter().rev().take(n).copied().collect();
        let mean_x: f64 = x_vals.iter().sum::<f64>() / n as f64;
        let mean_b: f64 = b_vals.iter().sum::<f64>() / n as f64;
        let cov: f64 = x_vals
            .iter()
            .zip(&b_vals)
            .map(|(x, b)| (x - mean_x) * (b - mean_b))
            .sum();
        let var_x: f64 = x_vals.iter().map(|x| (x - mean_x).powi(2)).sum();
        let var_b: f64 = b_vals.iter().map(|b| (b - mean_b).powi(2)).sum();
        let denom = (var_x * var_b).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            cov / denom
        }
    }

    fn windowed_correlation(&self) -> f64 {
        let n = self.pairs.len() as f64;
        if n < 3.0 {
            return 0.0;
        }
        let mean_x: f64 = self.pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_e: f64 = self.pairs.iter().map(|(_, e)| e).sum::<f64>() / n;
        let cov: f64 = self
            .pairs
            .iter()
            .map(|(x, e)| (x - mean_x) * (e - mean_e))
            .sum::<f64>();
        let var_x: f64 = self
            .pairs
            .iter()
            .map(|(x, _)| (x - mean_x).powi(2))
            .sum::<f64>();
        let var_e: f64 = self
            .pairs
            .iter()
            .map(|(_, e)| (e - mean_e).powi(2))
            .sum::<f64>();
        let denom = (var_x * var_e).sqrt();
        if denom < 1e-12 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Anomaly score (0..1). Combines decorrelation + divergence + B-field decorrelation.
    pub fn score(&self) -> f64 {
        if self.n_samples < 30 {
            return 0.0;
        }

        // X-ray/electron decorrelation score.
        let decorr_xe = if let Some(base) = self.baseline_corr {
            let drop = (base - self.correlation).max(0.0);
            (drop / 0.5).min(1.0)
        } else {
            0.0
        };

        // X-ray/B-field decorrelation score.
        // During flares, the X-ray/B-field correlation breaks because
        // B changes at L1 (solar wind) while X-ray comes direct from the Sun.
        let decorr_xb = if let Some(base) = self.baseline_corr_xb {
            let drop = (base - self.corr_xb).abs(); // abs because correlation can flip sign
            (drop / 0.5).min(1.0)
        } else {
            0.0
        };

        // Divergence score: Mahalanobis distance (2D or 3D).
        let div_score = 1.0 / (1.0 + (-1.0 * (self.divergence - 3.0)).exp());

        // Combine: max of all signals.
        // B-field decorrelation catches events earlier than X-ray/electron
        // because IMF changes propagate at solar wind speed (~400 km/s)
        // while X-rays arrive at light speed — the B-field at L1 reflects
        // solar magnetic topology changes that precede flare emission.
        decorr_xe.max(div_score).max(decorr_xb)
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn correlation(&self) -> f64 {
        self.correlation
    }

    pub fn divergence(&self) -> f64 {
        self.divergence
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
    fn test_correlated_channels_quiet() {
        let mut det = MultichannelDetector::default_detector();
        // Feed correlated data
        for i in 0..120 {
            let xray = 5e-7 * (1.0 + 0.1 * (i as f64 * 0.1).sin());
            let electron = 100.0 * (1.0 + 0.1 * (i as f64 * 0.1).sin());
            det.ingest(xray, electron, ts(i));
        }
        // Should be correlated and not anomalous
        assert!(det.correlation() > 0.5);
        assert!(det.score() < 0.5);
    }

    #[test]
    fn test_score_bounded() {
        let det = MultichannelDetector::default_detector();
        assert!(det.score() >= 0.0 && det.score() <= 1.0);
    }
}
