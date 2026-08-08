pub mod electrons;
pub mod kp_dst;
pub mod protons;
pub mod sharp;
pub mod solar_wind;
pub mod xray;

use chrono::{DateTime, Utc};
use serde::Serialize;
use std::collections::VecDeque;

/// Maximum samples in each ring buffer (24h of 1-min data).
pub const MAX_RING_SIZE: usize = 1440;
/// Max for 5-min cadence feeds (24h = 288 samples).
pub const MAX_RING_SIZE_5MIN: usize = 288;

/// Errors from feed fetching.
#[derive(Debug, Clone, Serialize)]
pub struct FeedError {
    pub feed: String,
    pub message: String,
}

impl std::fmt::Display for FeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Feed '{}': {}", self.feed, self.message)
    }
}

impl std::error::Error for FeedError {}

/// Current state of all feeds.
#[derive(Debug, Clone, Serialize)]
pub struct FeedState {
    /// Long-wavelength X-ray (0.1-0.8nm), 1-min cadence.
    pub xray: VecDeque<xray::XraySample>,
    /// Short-wavelength X-ray (0.05-0.4nm), 1-min cadence.
    pub xray_short: VecDeque<xray::XrayShortSample>,
    /// >2 MeV electron flux, 5-min cadence.
    pub electrons: VecDeque<electrons::ElectronSample>,
    /// >=1 MeV proton flux, 5-min cadence.
    pub protons: VecDeque<protons::ProtonSample>,
    /// Solar wind plasma + magnetic field, 1-min cadence.
    pub solar_wind: VecDeque<solar_wind::SolarWindSample>,
    /// Kp geomagnetic index, 3-hour cadence.
    pub kp_dst: VecDeque<kp_dst::KpDstSample>,
    /// SHARP magnetogram parameters, 12-min cadence (~3h latency).
    /// Keyed by HARP number — stores latest for each active region.
    pub sharp: Vec<sharp::SharpRecord>,
    /// Last poll attempt, including attempts where every upstream failed.
    pub last_poll: Option<DateTime<Utc>>,
    /// Last poll where at least one upstream returned observations.
    pub last_update: Option<DateTime<Utc>>,
    pub errors: Vec<FeedError>,
}

impl FeedState {
    pub fn new() -> Self {
        Self {
            xray: VecDeque::with_capacity(MAX_RING_SIZE),
            xray_short: VecDeque::with_capacity(MAX_RING_SIZE),
            electrons: VecDeque::with_capacity(MAX_RING_SIZE_5MIN),
            protons: VecDeque::with_capacity(MAX_RING_SIZE_5MIN),
            solar_wind: VecDeque::with_capacity(MAX_RING_SIZE),
            kp_dst: VecDeque::with_capacity(MAX_RING_SIZE),
            sharp: Vec::new(),
            last_poll: None,
            last_update: None,
            errors: Vec::new(),
        }
    }

    pub fn append_xray(&mut self, samples: Vec<xray::XraySample>) {
        append_dedup(&mut self.xray, samples, MAX_RING_SIZE, |a, b| {
            a.time_tag == b.time_tag
        });
    }

    pub fn append_xray_short(&mut self, samples: Vec<xray::XrayShortSample>) {
        append_dedup(&mut self.xray_short, samples, MAX_RING_SIZE, |a, b| {
            a.time_tag == b.time_tag
        });
    }

    pub fn append_electrons(&mut self, samples: Vec<electrons::ElectronSample>) {
        append_dedup(&mut self.electrons, samples, MAX_RING_SIZE_5MIN, |a, b| {
            a.time_tag == b.time_tag
        });
    }

    pub fn append_protons(&mut self, samples: Vec<protons::ProtonSample>) {
        append_dedup(&mut self.protons, samples, MAX_RING_SIZE_5MIN, |a, b| {
            a.time_tag == b.time_tag
        });
    }

    pub fn append_solar_wind(&mut self, samples: Vec<solar_wind::SolarWindSample>) {
        append_dedup(&mut self.solar_wind, samples, MAX_RING_SIZE, |a, b| {
            a.time_tag == b.time_tag
        });
    }

    pub fn append_kp_dst(&mut self, samples: Vec<kp_dst::KpDstSample>) {
        append_dedup(&mut self.kp_dst, samples, MAX_RING_SIZE, |a, b| {
            a.time_tag == b.time_tag
        });
    }
}

/// Provenance and observation-age metadata for one upstream feed.
#[derive(Debug, Clone, Serialize)]
pub struct FeedFreshness {
    pub source: &'static str,
    pub sample_time: Option<DateTime<Utc>>,
    pub age_seconds: Option<i64>,
    pub max_age_seconds: i64,
    pub fresh: bool,
    pub sample_count: usize,
}

impl FeedFreshness {
    fn new(
        source: &'static str,
        sample_time: Option<DateTime<Utc>>,
        max_age_seconds: i64,
        sample_count: usize,
        now: DateTime<Utc>,
    ) -> Self {
        let age_seconds = sample_time.map(|time| (now - time).num_seconds().max(0));
        let fresh = age_seconds
            .map(|age| age <= max_age_seconds)
            .unwrap_or(false);
        Self {
            source,
            sample_time,
            age_seconds,
            max_age_seconds,
            fresh,
            sample_count,
        }
    }
}

/// Alert-pipeline readiness derived from source observation times, not poll time.
#[derive(Debug, Clone, Serialize)]
pub struct FeedQuality {
    pub status: &'static str,
    pub feed_ready: bool,
    pub alerting_ready: bool,
    pub detector_ready: bool,
    pub detector_samples: usize,
    pub detector_samples_required: usize,
    pub evaluated_at: DateTime<Utc>,
    pub last_poll: Option<DateTime<Utc>>,
    pub last_successful_fetch: Option<DateTime<Utc>>,
    pub xray: FeedFreshness,
    pub xray_short: FeedFreshness,
    pub electrons: FeedFreshness,
    pub protons: FeedFreshness,
    pub solar_wind: FeedFreshness,
    pub kp_dst: FeedFreshness,
    pub sharp: FeedFreshness,
}

impl FeedState {
    pub fn quality(&self, now: DateTime<Utc>) -> FeedQuality {
        // Limits reflect the source cadence plus a conservative outage allowance.
        let xray = FeedFreshness::new(
            "NOAA SWPC GOES XRS 1-day",
            self.xray.back().map(|sample| sample.time_tag),
            5 * 60,
            self.xray.len(),
            now,
        );
        let xray_short = FeedFreshness::new(
            "NOAA SWPC GOES XRS short channel",
            self.xray_short.back().map(|sample| sample.time_tag),
            5 * 60,
            self.xray_short.len(),
            now,
        );
        let electrons = FeedFreshness::new(
            "NOAA SWPC GOES integral electrons",
            self.electrons.back().map(|sample| sample.time_tag),
            15 * 60,
            self.electrons.len(),
            now,
        );
        let protons = FeedFreshness::new(
            "NOAA SWPC GOES integral protons",
            self.protons.back().map(|sample| sample.time_tag),
            15 * 60,
            self.protons.len(),
            now,
        );
        let solar_wind = FeedFreshness::new(
            "NOAA SWPC DSCOVR/ACE solar wind",
            self.solar_wind.back().map(|sample| sample.time_tag),
            10 * 60,
            self.solar_wind.len(),
            now,
        );
        let kp_dst = FeedFreshness::new(
            "NOAA SWPC planetary K-index (Dst estimated)",
            self.kp_dst.back().map(|sample| sample.time_tag),
            4 * 60 * 60,
            self.kp_dst.len(),
            now,
        );
        let sharp = FeedFreshness::new(
            "NASA/JSOC SHARP active-region parameters",
            self.sharp.iter().map(|sample| sample.time_tag).max(),
            6 * 60 * 60,
            self.sharp.len(),
            now,
        );

        // XRS is the primary channel for flare detection. Other channels improve
        // fidelity but their absence must be described as degraded, not fabricated.
        let feed_ready = xray.fresh;
        let optional_fresh = electrons.fresh && protons.fresh && solar_wind.fresh && kp_dst.fresh;
        let status = if feed_ready && optional_fresh {
            "ok"
        } else if feed_ready {
            "degraded"
        } else if xray.sample_count == 0 {
            "starting"
        } else {
            "stale"
        };

        FeedQuality {
            status,
            feed_ready,
            alerting_ready: false,
            detector_ready: false,
            detector_samples: 0,
            detector_samples_required: 0,
            evaluated_at: now,
            last_poll: self.last_poll,
            last_successful_fetch: self.last_update,
            xray,
            xray_short,
            electrons,
            protons,
            solar_wind,
            kp_dst,
            sharp,
        }
    }
}

impl FeedQuality {
    pub fn with_detector_samples(mut self, samples: usize, required: usize) -> Self {
        self.detector_samples = samples;
        self.detector_samples_required = required;
        self.detector_ready = samples >= required;
        self.alerting_ready = self.feed_ready && self.detector_ready;
        if self.feed_ready && !self.detector_ready {
            self.status = "warming_up";
        }
        self
    }
}

/// Generic append with deduplication and ring buffer capping.
fn append_dedup<T>(
    buf: &mut VecDeque<T>,
    samples: Vec<T>,
    max_size: usize,
    eq: impl Fn(&T, &T) -> bool,
) {
    for s in samples {
        if !buf.iter().any(|x| eq(x, &s)) {
            buf.push_back(s);
        }
    }
    while buf.len() > max_size {
        buf.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn quality_uses_observation_time_not_poll_time() {
        let now = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let mut feeds = FeedState::new();
        feeds.last_poll = Some(now);
        feeds.last_update = Some(now);
        feeds.xray.push_back(xray::XraySample {
            time_tag: now - chrono::Duration::minutes(30),
            satellite: 18,
            flux: 1e-7,
            current_class: None,
        });

        let quality = feeds.quality(now);
        assert_eq!(quality.status, "stale");
        assert!(!quality.feed_ready);
        assert_eq!(quality.xray.age_seconds, Some(30 * 60));
    }

    #[test]
    fn fresh_primary_feed_can_run_in_degraded_mode() {
        let now = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
        let mut feeds = FeedState::new();
        feeds.xray.push_back(xray::XraySample {
            time_tag: now - chrono::Duration::minutes(1),
            satellite: 18,
            flux: 1e-7,
            current_class: None,
        });

        let quality = feeds.quality(now);
        assert_eq!(quality.status, "degraded");
        assert!(quality.feed_ready);
        assert!(!quality.alerting_ready);
        let ready = quality.with_detector_samples(200, 200);
        assert!(ready.alerting_ready);
    }
}
