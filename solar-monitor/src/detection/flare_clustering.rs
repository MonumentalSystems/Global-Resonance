use chrono::{DateTime, Duration, Utc};
use std::collections::VecDeque;

use super::FlareOnset;
use crate::feeds::xray::FlareClass;

/// Flare clustering / event rate acceleration detector.
///
/// Major flares are more likely when an active region is already
/// producing elevated activity. This detector tracks the rate of
/// C+ class events and detects acceleration — increasing event
/// frequency indicates a region building toward a larger event.
///
/// Uses a dual time-window approach:
/// - Short window (1h): recent event count
/// - Long window (6h): baseline event count
/// - Score = ratio of short/long rates
///
/// Also tracks event magnitude escalation (are the events getting bigger?).
#[derive(Debug, Clone)]
pub struct FlareClusteringDetector {
    /// Timestamps of recent events (flux > threshold).
    events: VecDeque<(DateTime<Utc>, f64)>,
    /// Short window duration (minutes).
    short_window_min: i64,
    /// Long window duration (minutes).
    long_window_min: i64,
    /// Flux threshold for counting as an event (C1.0 = 1e-6).
    event_threshold: f64,
    /// Current state.
    current_xray: f64,
    current_time: Option<DateTime<Utc>>,
    /// Cooldown: don't count the same event twice.
    last_event_time: Option<DateTime<Utc>>,
    event_cooldown_min: i64,
}

impl FlareClusteringDetector {
    pub fn new(short_window_min: i64, long_window_min: i64) -> Self {
        Self {
            events: VecDeque::new(),
            short_window_min,
            long_window_min,
            event_threshold: 1e-6, // C1.0
            current_xray: 0.0,
            current_time: None,
            last_event_time: None,
            event_cooldown_min: 15, // Don't double-count within 15 min
        }
    }

    /// Default: 1h short window, 6h long window.
    pub fn default_detector() -> Self {
        Self::new(60, 360)
    }

    pub fn ingest(&mut self, xray_flux: f64, timestamp: DateTime<Utc>) {
        self.current_xray = xray_flux;
        self.current_time = Some(timestamp);

        // Register event if above threshold and cooled down
        if xray_flux >= self.event_threshold {
            let should_register = self
                .last_event_time
                .map(|last| (timestamp - last).num_minutes() >= self.event_cooldown_min)
                .unwrap_or(true);
            if should_register {
                self.events.push_back((timestamp, xray_flux));
                self.last_event_time = Some(timestamp);
            }
        }

        // Expire old events
        let cutoff = timestamp - Duration::minutes(self.long_window_min);
        while self.events.front().map_or(false, |(t, _)| *t < cutoff) {
            self.events.pop_front();
        }
    }

    /// Anomaly score (0..1).
    pub fn score(&self) -> f64 {
        let now = match self.current_time {
            Some(t) => t,
            None => return 0.0,
        };

        // Count events in each window
        let short_cutoff = now - Duration::minutes(self.short_window_min);
        let short_events: Vec<&(DateTime<Utc>, f64)> = self
            .events
            .iter()
            .filter(|(t, _)| *t >= short_cutoff)
            .collect();
        let long_events = self.events.len();

        if long_events < 2 {
            return 0.0;
        }

        // Rate acceleration: short window rate vs long window rate
        let short_rate = short_events.len() as f64 / (self.short_window_min as f64 / 60.0);
        let long_rate = long_events as f64 / (self.long_window_min as f64 / 60.0);

        let rate_ratio = if long_rate > 0.01 {
            short_rate / long_rate
        } else {
            0.0
        };

        // Magnitude escalation: are recent events bigger?
        let short_max = short_events.iter().map(|(_, f)| *f).fold(0.0f64, f64::max);
        let long_mean = self.events.iter().map(|(_, f)| *f).sum::<f64>() / long_events as f64;
        let mag_ratio = if long_mean > 1e-8 {
            short_max / long_mean
        } else {
            0.0
        };

        // Combined score
        let rate_score = ((rate_ratio - 1.0) / 2.0).max(0.0).min(1.0); // 3x rate = full score
        let mag_score = ((mag_ratio - 1.0) / 5.0).max(0.0).min(1.0); // 6x magnitude = full
        let count_score = (short_events.len() as f64 / 5.0).min(1.0); // 5+ events/hour = full

        (rate_score * 0.4 + mag_score * 0.3 + count_score * 0.3).min(1.0)
    }

    pub fn is_anomalous(&self) -> bool {
        self.score() > 0.5
    }

    pub fn events_in_short_window(&self) -> usize {
        let now = match self.current_time {
            Some(t) => t,
            None => return 0,
        };
        let cutoff = now - Duration::minutes(self.short_window_min);
        self.events.iter().filter(|(t, _)| *t >= cutoff).count()
    }

    pub fn events_in_long_window(&self) -> usize {
        self.events.len()
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
    fn test_quiet_no_events() {
        let mut d = FlareClusteringDetector::default_detector();
        for i in 0..60 {
            d.ingest(5e-7, ts(i)); // B-class, below threshold
        }
        assert_eq!(d.events_in_long_window(), 0);
        assert!(!d.is_anomalous());
    }

    #[test]
    fn test_clustering_detected() {
        let mut d = FlareClusteringDetector::default_detector();
        // Sparse background over 6 hours
        for i in 0..300 {
            d.ingest(5e-7, ts(i));
        }
        // Inject 2 C-class events in the long window (rare)
        d.ingest(3e-6, ts(100));
        d.ingest(2e-6, ts(200));
        // Now rapid cluster in last hour
        for i in 300..360 {
            if i % 20 == 0 {
                d.ingest(5e-6, ts(i)); // C-class every 20 min
            } else {
                d.ingest(5e-7, ts(i));
            }
        }
        assert!(d.events_in_short_window() >= 2);
    }

    #[test]
    fn test_score_bounded() {
        let d = FlareClusteringDetector::default_detector();
        assert!(d.score() >= 0.0 && d.score() <= 1.0);
    }
}
