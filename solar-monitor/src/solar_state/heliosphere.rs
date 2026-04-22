//! Heliosphere state — what's propagating toward Earth?

use chrono::{DateTime, Utc};
use serde::Serialize;

/// State of the Sun-Earth heliosphere.
#[derive(Debug, Clone, Serialize)]
pub struct HeliosphereState {
    /// Earth-directed CMEs currently in transit.
    pub earth_directed_cmes: Vec<CmeInTransit>,
    /// Active high-speed streams.
    pub active_hss: Vec<HighSpeedStream>,
    /// Recent interplanetary shocks.
    pub recent_shocks: Vec<IpShock>,
    /// Current solar wind conditions at L1.
    pub solar_wind: SolarWindConditions,
}

/// A CME propagating toward Earth.
#[derive(Debug, Clone, Serialize)]
pub struct CmeInTransit {
    pub launch_time: DateTime<Utc>,
    /// Speed from cone model analysis (km/s).
    pub speed: f64,
    /// Half-angle (degrees) — wider = more likely to impact.
    pub half_angle: f64,
    /// Estimated arrival time at Earth.
    pub estimated_arrival: Option<DateTime<Utc>>,
    /// Hours until estimated arrival.
    pub hours_to_arrival: Option<f64>,
    /// Source active region.
    pub source_region: Option<u32>,
    /// Associated flare class.
    pub associated_flare: Option<String>,
    /// DONKI activity ID.
    pub activity_id: String,
}

/// A high-speed solar wind stream (from coronal hole).
#[derive(Debug, Clone, Serialize)]
pub struct HighSpeedStream {
    pub event_time: DateTime<Utc>,
    pub activity_id: String,
}

/// An interplanetary shock.
#[derive(Debug, Clone, Serialize)]
pub struct IpShock {
    pub event_time: DateTime<Utc>,
    /// Location: "Earth" or spacecraft name.
    pub location: String,
    pub activity_id: String,
}

/// Current solar wind at L1.
#[derive(Debug, Clone, Serialize)]
pub struct SolarWindConditions {
    /// Bulk speed (km/s).
    pub speed: f64,
    /// IMF Bz (nT, GSM). Negative = southward = geoeffective.
    pub bz: f64,
    /// IMF By (nT, GSM). For Mansurov effect.
    pub by: f64,
    /// Proton density (n/cm^3).
    pub density: f64,
    /// Dynamic pressure proxy (nPa).
    pub dynamic_pressure: f64,
}

impl HeliosphereState {
    pub fn new() -> Self {
        Self {
            earth_directed_cmes: Vec::new(),
            active_hss: Vec::new(),
            recent_shocks: Vec::new(),
            solar_wind: SolarWindConditions {
                speed: 400.0,
                bz: 0.0,
                by: 0.0,
                density: 5.0,
                dynamic_pressure: 2.0,
            },
        }
    }

    /// Earth threat level from heliospheric conditions (0..1).
    pub fn earth_threat(&self) -> f64 {
        let mut threat = 0.0;

        // Earth-directed CMEs in transit
        for cme in &self.earth_directed_cmes {
            let speed_factor = (cme.speed / 1500.0).min(1.0);
            let imminence = cme
                .hours_to_arrival
                .map(|h| (1.0 - h / 72.0).max(0.0))
                .unwrap_or(0.3);
            threat += speed_factor * imminence * 0.4;
        }

        // Southward Bz (geoeffective)
        if self.solar_wind.bz < -5.0 {
            threat += ((-self.solar_wind.bz - 5.0) / 20.0).min(0.3);
        }

        // High speed solar wind
        if self.solar_wind.speed > 600.0 {
            threat += ((self.solar_wind.speed - 600.0) / 400.0).min(0.2);
        }

        // Active HSS
        threat += (self.active_hss.len() as f64 * 0.05).min(0.1);

        threat.min(1.0)
    }
}
