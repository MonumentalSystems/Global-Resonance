use chrono::{DateTime, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Sustained southward Bz detector.
///
/// Southward IMF (negative Bz in GSM) is the primary driver of
/// geomagnetic activity. A single negative reading means little,
/// but sustained Bz < -5 nT for 30+ minutes is highly geoeffective.
///
/// This detector tracks:
/// - Duration of sustained southward Bz
/// - Magnitude of the southward component
/// - Rate of Bz decrease (rapid southward turning)
///
/// Combined into a single score that predicts geomagnetic storm onset.
#[derive(Debug, Clone)]
pub struct BzSouthwardDetector {
    /// Recent Bz values for trend analysis.
    window: VecDeque<f64>,
    window_size: usize,
    /// Duration of current sustained southward period (minutes).
    sustained_minutes: f64,
    /// Timestamp when current southward period began.
    southward_since: Option<DateTime<Utc>>,
    /// Current Bz value.
    current_bz: f64,
    /// Current associated X-ray flux (for onset events).
    current_xray: f64,
    current_time: Option<DateTime<Utc>>,
    last_timestamp: Option<DateTime<Utc>>,
}

impl BzSouthwardDetector {
    pub fn new(window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            sustained_minutes: 0.0,
            southward_since: None,
            current_bz: 0.0,
            current_xray: 0.0,
            current_time: None,
            last_timestamp: None,
        }
    }

    /// Default: 60-sample window.
    pub fn default_detector() -> Self {
        Self::new(60)
    }

    /// Ingest a Bz reading (nT, GSM). Also takes xray for onset events.
    pub fn ingest(&mut self, bz: f64, xray_flux: f64, timestamp: DateTime<Utc>) {
        self.current_bz = bz;
        self.current_xray = xray_flux;
        self.current_time = Some(timestamp);

        self.window.push_back(bz);
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }

        // Track sustained southward duration
        if bz < -2.0 {
            if self.southward_since.is_none() {
                self.southward_since = Some(timestamp);
            }
            self.sustained_minutes = self
                .southward_since
                .map(|since| (timestamp - since).num_seconds() as f64 / 60.0)
                .unwrap_or(0.0);
        } else if bz > 0.0 {
            // Reset on clear northward
            self.southward_since = None;
            self.sustained_minutes = 0.0;
        }
        // If Bz is between -2 and 0, don't reset — could be fluctuating

        self.last_timestamp = Some(timestamp);
    }

    /// Anomaly score (0..1). Higher = more geoeffective conditions.
    pub fn score(&self) -> f64 {
        if self.window.len() < 5 {
            return 0.0;
        }

        // Magnitude factor: how strongly southward
        // -5 nT = threshold, -15 nT = strong, -25 nT = extreme
        let magnitude = if self.current_bz < -3.0 {
            ((-self.current_bz - 3.0) / 12.0).min(1.0)
        } else {
            0.0
        };

        // Duration factor: sustained southward is worse
        let duration = (self.sustained_minutes / 30.0).min(1.0); // 30min = full score

        // Rate factor: rapid southward turning
        let rate = if self.window.len() >= 10 {
            let recent: f64 = self.window.iter().rev().take(5).sum::<f64>() / 5.0;
            let earlier: f64 = self.window.iter().take(5).sum::<f64>() / 5.0;
            let dbz = earlier - recent; // positive if Bz is going more negative
            (dbz / 10.0).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Combined: magnitude dominates, boosted by duration and rate
        let score = magnitude * 0.5 + duration * 0.3 + rate * 0.2;
        score.min(1.0)
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn sustained_minutes(&self) -> f64 {
        self.sustained_minutes
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
    fn test_quiet_northward() {
        let mut d = BzSouthwardDetector::default_detector();
        for i in 0..30 {
            d.ingest(2.0, 5e-7, ts(i));
        }
        assert!(!d.is_anomalous());
        assert_eq!(d.sustained_minutes(), 0.0);
    }

    #[test]
    fn test_sustained_southward() {
        let mut d = BzSouthwardDetector::default_detector();
        // 60 minutes of Bz = -10 nT
        for i in 0..60 {
            d.ingest(-10.0, 5e-7, ts(i));
        }
        assert!(d.is_anomalous());
        assert!(d.sustained_minutes() > 55.0);
        assert!(d.score() > 0.5);
    }

    #[test]
    fn test_score_bounded() {
        let d = BzSouthwardDetector::default_detector();
        assert!(d.score() >= 0.0 && d.score() <= 1.0);
    }
}
