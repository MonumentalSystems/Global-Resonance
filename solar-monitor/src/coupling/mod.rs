pub mod forbush;
pub mod heep;
pub mod lunar_tidal;
pub mod mansurov;
pub mod ssc;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::detection::FlareOnset;
use crate::feeds::FeedState;

/// Effect direction on severe weather.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CouplingEffect {
    /// Reduces severe weather likelihood (e.g., Forbush chain).
    Suppression,
    /// Increases severe weather likelihood.
    Enhancement,
}

/// Status of a single coupling pathway.
#[derive(Debug, Clone, Serialize)]
pub struct PathwayStatus {
    pub name: String,
    pub active: bool,
    /// Activation score 0..1.
    pub score: f64,
    pub effect: CouplingEffect,
    /// Lag range in hours from trigger to atmospheric effect.
    pub lag_hours: (f64, f64),
    /// Human-readable explanation of current state.
    pub details: String,
}

/// Combined stressor loading index across all pathways.
#[derive(Debug, Clone, Serialize)]
pub struct StressorScore {
    /// Weighted composite score.
    pub total: f64,
    /// Individual pathway statuses.
    pub pathways: Vec<PathwayStatus>,
    pub timestamp: DateTime<Utc>,
}

/// Manages all 5 coupling pathways and computes the combined stressor index.
pub struct StressorIndex {
    pub forbush: forbush::ForbushPathway,
    pub heep: heep::HeepPathway,
    pub ssc: ssc::SscPathway,
    pub mansurov: mansurov::MansurovPathway,
    pub lunar: lunar_tidal::LunarTidalPathway,
    /// Weights for each pathway (order: forbush, heep, ssc, mansurov, lunar).
    pub weights: [f64; 5],
}

impl StressorIndex {
    pub fn new() -> Self {
        Self {
            forbush: forbush::ForbushPathway::new(),
            heep: heep::HeepPathway::new(),
            ssc: ssc::SscPathway::new(),
            mansurov: mansurov::MansurovPathway::new(),
            lunar: lunar_tidal::LunarTidalPathway::new(),
            // Equal weighting by default
            weights: [1.0; 5],
        }
    }

    /// Update all pathways with current feed state and any flare onset.
    pub fn update(&mut self, feeds: &FeedState, flare: Option<&FlareOnset>) {
        let quality = feeds.quality(Utc::now());
        let mut fresh = feeds.clone();
        if !quality.electrons.fresh {
            fresh.electrons.clear();
        }
        if !quality.protons.fresh {
            fresh.protons.clear();
        }
        if !quality.solar_wind.fresh {
            fresh.solar_wind.clear();
        }
        if !quality.kp_dst.fresh {
            fresh.kp_dst.clear();
        }

        self.forbush.update(&fresh, flare);
        self.heep.update(&fresh);
        self.ssc.update(&fresh);
        self.mansurov.update(&fresh);
        self.lunar.update();
    }

    /// Compute the combined stressor score.
    pub fn compute(&self) -> StressorScore {
        let statuses = vec![
            self.forbush.status(),
            self.heep.status(),
            self.ssc.status(),
            self.mansurov.status(),
            self.lunar.status(),
        ];

        let total: f64 = statuses
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| {
                let sign = match s.effect {
                    CouplingEffect::Suppression => -1.0,
                    CouplingEffect::Enhancement => 1.0,
                };
                s.score * w * sign
            })
            .sum();

        StressorScore {
            total,
            pathways: statuses,
            timestamp: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stressor_index_default() {
        let idx = StressorIndex::new();
        let score = idx.compute();
        assert_eq!(score.pathways.len(), 5);
    }
}
