pub mod bz_southward;
pub mod criticality;
pub mod cusum;
pub mod planetary_kan;
pub mod energy;
pub mod escalation;
pub mod flare_clustering;
pub mod hardness;
pub mod multichannel;
pub mod pressure_jump;
pub mod proton;
pub mod rank_fusion;
pub mod rate_of_change;
pub mod zscore;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::feeds::xray::FlareClass;

/// A detected flare onset event.
#[derive(Debug, Clone, Serialize)]
pub struct FlareOnset {
    pub timestamp: DateTime<Utc>,
    pub class: FlareClass,
    /// Peak flux in W/m^2.
    pub peak_flux: f64,
    /// Anomaly score (0..1, higher = more anomalous).
    pub anomaly_score: f64,
}
