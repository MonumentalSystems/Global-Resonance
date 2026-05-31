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
