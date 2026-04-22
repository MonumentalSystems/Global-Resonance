//! Orbital angle computation for planetary inputs to SolarFlareV2.
//!
//! All 8 planets + lunar nodal precession as sin/cos pairs = 18 inputs.
//! All angles are deterministic functions of time (Julian date).
//! Mean orbital elements — sufficient for the envelope modulation.
//!
//! From the Harmonic Cascade paper: all known solar/climate periodicities
//! form a subharmonic ladder of integer ratios rooted in orbital obliquity.
//! Bond/124 = Jupiter (0.1%), Bond/50 = Saturn (0.2%), Bond/9 = Neptune (0.9%).
//! Inner planets matter for individual flare timing (days-weeks).

use std::f64::consts::PI;

/// Orbital periods in Julian years.
const MERCURY_PERIOD: f64 = 0.2408;
const VENUS_PERIOD: f64 = 0.6152;
const EARTH_PERIOD: f64 = 1.0000;
const MARS_PERIOD: f64 = 1.8809;
const JUPITER_PERIOD: f64 = 11.862;
const SATURN_PERIOD: f64 = 29.457;
const URANUS_PERIOD: f64 = 84.011;
const NEPTUNE_PERIOD: f64 = 164.79;
const LUNAR_NODAL: f64 = 18.613;   // lunar nodal precession

/// J2000.0 epoch (Julian date 2451545.0)
const J2000: f64 = 2451545.0;

/// Mean ecliptic longitudes at J2000.0 (degrees).
const MERCURY_L0: f64 = 252.25;
const VENUS_L0: f64 = 181.98;
const EARTH_L0: f64 = 100.46;
const MARS_L0: f64 = 355.45;
const JUPITER_L0: f64 = 34.35;
const SATURN_L0: f64 = 49.94;
const URANUS_L0: f64 = 313.23;
const NEPTUNE_L0: f64 = 304.88;
const LUNAR_NODE_L0: f64 = 125.04; // ascending node at J2000

/// Number of orbital inputs: 8 planets × 2 (sin/cos) + lunar node × 2 = 18.
pub const N_ORBITAL_INPUTS: usize = 18;

/// Compute all 18 orbital inputs (sin/cos pairs) for a given Julian date.
///
/// Returns [sin_merc, cos_merc, sin_ven, cos_ven, sin_earth, cos_earth,
///          sin_mars, cos_mars, sin_jup, cos_jup, sin_sat, cos_sat,
///          sin_ura, cos_ura, sin_nep, cos_nep, sin_node, cos_node].
pub fn orbital_inputs(jd: f64) -> [f32; N_ORBITAL_INPUTS] {
    let t_yr = (jd - J2000) / 365.25; // years since J2000

    let periods = [
        MERCURY_PERIOD, VENUS_PERIOD, EARTH_PERIOD, MARS_PERIOD,
        JUPITER_PERIOD, SATURN_PERIOD, URANUS_PERIOD, NEPTUNE_PERIOD,
    ];
    let l0s = [
        MERCURY_L0, VENUS_L0, EARTH_L0, MARS_L0,
        JUPITER_L0, SATURN_L0, URANUS_L0, NEPTUNE_L0,
    ];

    let mut out = [0.0f32; N_ORBITAL_INPUTS];
    for (i, (&period, &lon0)) in periods.iter().zip(l0s.iter()).enumerate() {
        let lon = (lon0 + 360.0 * t_yr / period) * PI / 180.0;
        out[i * 2] = lon.sin() as f32;
        out[i * 2 + 1] = lon.cos() as f32;
    }
    // Lunar nodal precession (retrograde)
    let node_lon = (LUNAR_NODE_L0 - 360.0 * t_yr / LUNAR_NODAL) * PI / 180.0;
    out[16] = node_lon.sin() as f32;
    out[17] = node_lon.cos() as f32;

    out
}

/// Convert a calendar date to Julian date (approximate, sufficient for orbital angles).
pub fn date_to_jd(year: i32, month: u32, day: u32) -> f64 {
    // Meeus algorithm
    let y = if month <= 2 { year - 1 } else { year } as f64;
    let m = if month <= 2 { month + 12 } else { month } as f64;
    let a = (y / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y + 4716.0)).floor() + (30.6001 * (m + 1.0)).floor() + day as f64 + b - 1524.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_j2000_epoch() {
        let jd = date_to_jd(2000, 1, 1);
        assert!((jd - 2451544.5).abs() < 1.0, "J2000 should be near 2451545");
    }

    #[test]
    fn test_orbital_inputs_finite() {
        let inputs = orbital_inputs(date_to_jd(2026, 4, 5));
        for v in &inputs {
            assert!(v.is_finite());
            assert!(v.abs() <= 1.0, "sin/cos should be in [-1,1]");
        }
    }

    #[test]
    fn test_jupiter_period() {
        // Jupiter should return to same angle after ~11.86 years
        let jd0 = date_to_jd(2020, 1, 1);
        let jd1 = jd0 + JUPITER_PERIOD * 365.25;
        let i0 = orbital_inputs(jd0);
        let i1 = orbital_inputs(jd1);
        // sin/cos should be similar after one period
        assert!((i0[0] - i1[0]).abs() < 0.01, "Jupiter sin should repeat");
        assert!((i0[1] - i1[1]).abs() < 0.01, "Jupiter cos should repeat");
    }
}
