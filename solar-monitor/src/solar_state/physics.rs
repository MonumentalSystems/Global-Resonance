//! Physics-based models from the Geometric Resonance Papers.
//!
//! Implements the quantitative predictions from Papers XXV, XXX, XXXII:
//! - Alfvén velocity from solar wind observables
//! - Topological reconnection heating rate (Q ∝ v_A³/L)
//! - CME geometric hit/miss model (90% accuracy)
//! - Grade-based temporal coupling windows (0/2/4)
//! - Seismic risk zones (subsolar geometry)

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Alfvén velocity computed from solar wind conditions at L1.
///
/// v_A = B_t / sqrt(μ₀ρ)
/// where B_t = total IMF (nT), ρ = proton mass * number density
///
/// This is the fundamental energy transport speed — determines
/// reconnection rate, wave heating, and coupling timescales.
pub fn alfven_velocity(bt_nt: f64, density_per_cc: f64) -> f64 {
    if density_per_cc <= 0.0 || bt_nt <= 0.0 {
        return 0.0;
    }
    // B in Tesla: bt_nt * 1e-9
    // ρ in kg/m³: density * 1e6 (cm⁻³ to m⁻³) * 1.672e-27 (proton mass)
    let b_t = bt_nt * 1e-9;
    let rho = density_per_cc * 1e6 * 1.672e-27;
    let mu0 = 4.0 * std::f64::consts::PI * 1e-7;

    b_t / (mu0 * rho).sqrt() / 1e3 // Convert m/s to km/s
}

/// Topological reconnection heating rate.
///
/// Q ∝ v_A³ / L
/// From Paper XXV: this is independent of resistivity (unlike Sweet-Parker).
/// Higher v_A and shorter L = more intense heating = more flare potential.
pub fn reconnection_heating_rate(v_a_kms: f64, length_scale_km: f64) -> f64 {
    if length_scale_km <= 0.0 {
        return 0.0;
    }
    // Q in erg/cm³/s (typical coronal units)
    // Normalization: v_A=1000 km/s, L=10000 km → Q ~ 1e-4 erg/cm³/s
    let v_a_cgs = v_a_kms * 1e5; // km/s to cm/s
    let l_cgs = length_scale_km * 1e5; // km to cm
    let rho_cgs = 1e-16; // typical coronal density in g/cm³

    rho_cgs * v_a_cgs.powi(3) / l_cgs
}

/// CME geometric impact model from Paper XXXII.
///
/// Uses source location and half-angle to predict Earth impact.
/// 90% accuracy on backtest of 10 events (better than ENLIL for this sample).
///
/// Returns (impact_class, predicted_kp, confidence)
pub fn cme_impact_prediction(
    source_lat: f64, // degrees, heliographic
    source_lon: f64, // degrees, heliographic (+ = West)
    half_angle: f64, // degrees, from cone model
    speed: f64,      // km/s
) -> CmeImpactPrediction {
    // Angular offset from disk center
    let theta = (source_lat.powi(2) + source_lon.powi(2)).sqrt();

    let (impact_class, predicted_kp) = if theta < half_angle - 15.0 {
        // Direct hit
        let kp = (speed / 200.0).min(9.0).max(5.0);
        (CmeImpactClass::DirectHit, kp)
    } else if theta < half_angle + 5.0 {
        // Glancing blow
        let kp = (speed / 400.0).min(6.0).max(2.0);
        (CmeImpactClass::GlancingBlow, kp)
    } else if theta < half_angle + 10.0 {
        // Weak clip
        let kp = (speed / 600.0).min(3.0).max(1.0);
        (CmeImpactClass::WeakClip, kp)
    } else {
        // Miss
        (CmeImpactClass::Miss, 0.0)
    };

    // Transit time estimate with drag
    let drag_gamma = 2e-8; // /km drag coefficient
    let ambient_speed = 400.0; // km/s
    let effective_speed = if speed > ambient_speed {
        // Speed decays toward ambient during transit
        ambient_speed + (speed - ambient_speed) * (-drag_gamma * 1.5e8_f64).exp()
    } else {
        speed
    };
    let cos_theta = (theta.to_radians()).cos().max(0.1);
    let transit_hours = (1.5e8 / (effective_speed * cos_theta)) / 3600.0;

    CmeImpactPrediction {
        impact_class,
        predicted_kp,
        source_offset_deg: theta,
        transit_hours,
        confidence: if half_angle > 20.0 { 0.8 } else { 0.5 },
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CmeImpactPrediction {
    pub impact_class: CmeImpactClass,
    pub predicted_kp: f64,
    pub source_offset_deg: f64,
    pub transit_hours: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CmeImpactClass {
    DirectHit,
    GlancingBlow,
    WeakClip,
    Miss,
}

/// Grade-based temporal coupling windows from Papers XXV and XXX.
///
/// After a significant solar event (M/X flare + CME), the Earth response
/// follows three distinct timescales corresponding to the three grades
/// of the bivector field equation F∇F = ⟨F∇F⟩₀ + ½[F,∇F] + F∧∇F.
#[derive(Debug, Clone, Serialize)]
pub struct CouplingWindow {
    pub grade: u8,
    pub name: String,
    pub mechanism: String,
    pub lag_hours: (f64, f64),
    /// Enhancement factor (>1 = enhanced, <1 = suppressed).
    pub enhancement: f64,
    /// Whether this window is currently active.
    pub active: bool,
    /// Hours since the triggering event.
    pub hours_since_trigger: Option<f64>,
}

/// Compute active coupling windows for a given time after a flare/CME event.
pub fn coupling_windows(
    flare_time: DateTime<Utc>,
    cme_arrival_time: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Vec<CouplingWindow> {
    let hours_since_flare = (now - flare_time).num_minutes() as f64 / 60.0;
    let hours_since_cme = cme_arrival_time.map(|t| (now - t).num_minutes() as f64 / 60.0);

    vec![
        CouplingWindow {
            grade: 0,
            name: "Grade-0 (EM)".into(),
            mechanism: "Ionospheric SID compression, pushes J > J_c".into(),
            lag_hours: (0.0, 6.0),
            enhancement: 0.82, // Suppression
            active: hours_since_flare >= 0.0 && hours_since_flare <= 6.0,
            hours_since_trigger: Some(hours_since_flare),
        },
        CouplingWindow {
            grade: 4,
            name: "Grade-4 (Ionospheric)".into(),
            mechanism: "Ionospheric relaxation back through J_c".into(),
            lag_hours: (12.0, 24.0),
            enhancement: 1.13, // Enhancement
            active: hours_since_flare >= 12.0 && hours_since_flare <= 24.0,
            hours_since_trigger: Some(hours_since_flare),
        },
        CouplingWindow {
            grade: 2,
            name: "Grade-2 (CME mechanical)".into(),
            mechanism: "CME compression, peak at +41h for X-class".into(),
            lag_hours: (24.0, 72.0),
            enhancement: 1.36, // Peak enhancement (X-class)
            active: hours_since_cme.map_or(false, |h| h >= 0.0 && h <= 48.0),
            hours_since_trigger: hours_since_cme,
        },
    ]
}

/// Compute seismic risk zone from subsolar point geometry.
///
/// From Paper XXV: angular distance from subsolar point determines
/// the coupling strength. Wavefront zone (60-75°) has strongest
/// enhancement (1.36×), subsolar is suppressed (0.85×).
pub fn seismic_zone_factor(angular_distance_deg: f64) -> (f64, &'static str) {
    if angular_distance_deg < 15.0 {
        (0.85, "Subsolar (suppressed)")
    } else if angular_distance_deg < 30.0 {
        (1.0, "Inner zone (neutral)")
    } else if angular_distance_deg < 60.0 {
        (1.13, "Inner wavefront (enhanced)")
    } else if angular_distance_deg < 75.0 {
        (1.36, "Wavefront (peak enhancement)")
    } else if angular_distance_deg < 120.0 {
        (1.0, "Mid-zone (neutral)")
    } else if angular_distance_deg < 135.0 {
        (0.87, "Far side (suppressed)")
    } else if angular_distance_deg < 165.0 {
        (1.0, "Deep far side (neutral)")
    } else {
        (1.16, "Antipodal (reconvergence)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alfven_velocity() {
        // Typical solar wind: B=5nT, n=5/cc → v_A ~50 km/s
        let va = alfven_velocity(5.0, 5.0);
        assert!(va > 30.0 && va < 100.0, "v_A = {}", va);
    }

    #[test]
    fn test_alfven_velocity_fast_wind() {
        // Fast solar wind: B=10nT, n=3/cc → v_A ~130 km/s
        let va = alfven_velocity(10.0, 3.0);
        assert!(va > 80.0 && va < 200.0, "v_A = {}", va);
    }

    #[test]
    fn test_cme_direct_hit() {
        let pred = cme_impact_prediction(10.0, 5.0, 45.0, 1000.0);
        assert_eq!(pred.impact_class, CmeImpactClass::DirectHit);
        assert!(pred.predicted_kp >= 5.0);
    }

    #[test]
    fn test_cme_miss() {
        let pred = cme_impact_prediction(60.0, 30.0, 20.0, 500.0);
        assert_eq!(pred.impact_class, CmeImpactClass::Miss);
    }

    #[test]
    fn test_seismic_zones() {
        assert!(seismic_zone_factor(10.0).0 < 1.0); // suppressed
        assert!(seismic_zone_factor(65.0).0 > 1.2); // peak enhancement
        assert!(seismic_zone_factor(170.0).0 > 1.0); // antipodal enhancement
    }
}
