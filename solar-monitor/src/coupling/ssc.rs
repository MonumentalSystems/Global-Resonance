use super::{CouplingEffect, PathwayStatus};
use crate::feeds::FeedState;

/// Storm Sudden Commencement (SSC) / telluric current pathway.
///
/// CME impact -> magnetopause compression -> dB/dt pulse ->
/// telluric currents -> enhanced vertical electric field.
///
/// Lag: Hours (fastest pathway).
/// Effect: Enhancement (if strong).
///
/// We detect SSC precursors from solar wind data:
/// sudden speed jump + negative Bz turning.
pub struct SscPathway {
    score: f64,
    /// Estimated dB/dt proxy from solar wind dynamic pressure change.
    db_dt_proxy: f64,
    /// Whether negative Bz is present (southward IMF).
    southward_imf: bool,
    /// Solar wind speed jump in km/s over last hour.
    speed_jump: f64,
}

impl SscPathway {
    pub fn new() -> Self {
        Self {
            score: 0.0,
            db_dt_proxy: 0.0,
            southward_imf: false,
            speed_jump: 0.0,
        }
    }

    pub fn update(&mut self, feeds: &FeedState) {
        if feeds.solar_wind.len() < 2 {
            self.score = 0.0;
            return;
        }

        let current = feeds.solar_wind.back().unwrap();
        self.southward_imf = current.bz < -2.0; // Bz < -2 nT is southward

        // Speed jump: compare current to ~1h ago (60 samples at 1-min cadence)
        let lookback = 60.min(feeds.solar_wind.len() - 1);
        let old_idx = feeds.solar_wind.len() - 1 - lookback;
        let old_speed = feeds.solar_wind[old_idx].speed;
        self.speed_jump = current.speed - old_speed;

        // dB/dt proxy from dynamic pressure change
        // Pdyn ~ n * V^2, so sudden V increase = compression
        let old_density = feeds.solar_wind[old_idx].density.max(0.1);
        let pdyn_now = current.density * current.speed.powi(2);
        let pdyn_old = old_density * old_speed.powi(2);
        self.db_dt_proxy = if pdyn_old > 0.0 {
            (pdyn_now - pdyn_old) / pdyn_old
        } else {
            0.0
        };

        // Score: needs both speed jump AND southward Bz for effective coupling
        let speed_score = if self.speed_jump > 100.0 {
            ((self.speed_jump - 100.0) / 200.0).min(1.0)
        } else {
            0.0
        };

        let bz_score = if current.bz < -5.0 {
            ((-current.bz - 5.0) / 15.0).min(1.0)
        } else {
            0.0
        };

        // SSC requires both conditions
        self.score = if speed_score > 0.1 && bz_score > 0.1 {
            (speed_score * 0.5 + bz_score * 0.5).min(1.0)
        } else if speed_score > 0.3 {
            // Strong speed jump alone can cause SSC
            speed_score * 0.5
        } else {
            0.0
        };
    }

    pub fn status(&self) -> PathwayStatus {
        let details = format!(
            "Speed jump: {:+.0} km/s, Bz: {}, dP/P: {:+.1}%",
            self.speed_jump,
            if self.southward_imf {
                "southward"
            } else {
                "northward"
            },
            self.db_dt_proxy * 100.0,
        );

        PathwayStatus {
            name: "SSC Telluric".into(),
            active: self.score > 0.1,
            score: self.score,
            effect: CouplingEffect::Enhancement,
            lag_hours: (1.0, 6.0),
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiet_solar_wind() {
        let p = SscPathway::new();
        let status = p.status();
        assert!(!status.active);
        assert_eq!(status.effect, CouplingEffect::Enhancement);
    }
}
