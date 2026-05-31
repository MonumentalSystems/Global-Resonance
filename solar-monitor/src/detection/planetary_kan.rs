//! Planetary KAN: learnable periodic modulation of criticality score.
//!
//! Each planet gets a CircularBSpline (KAN basis function) that maps
//! its ecliptic longitude to a modulation factor. The product of all
//! modulations adjusts the criticality score.
//!
//! The KAN learns which planets matter and what phase relationship
//! triggers flares. Initialized to flat (no modulation) so the
//! criticality detector works unchanged until the KAN is trained.
//!
//! Architecture:
//!   9 orbital angles (Mercury..Neptune + lunar node)
//!   → 9 CircularBSpline evaluations
//!   → learnable weights per planet
//!   → weighted sum + bias → sigmoid → modulation ∈ [0.5, 1.5]
//!
//! Total learnable params: 9 × n_knots + 9 weights + 1 bias
//! With n_knots=8: 9×8 + 10 = 82 params

use std::f32::consts::PI;
use serde::{Deserialize, Serialize};

/// Number of orbital bodies: 8 planets + lunar nodal precession.
pub const N_BODIES: usize = 9;

/// Orbital periods in Julian years (for computing angles from JD).
pub const ORBITAL_PERIODS: [f64; N_BODIES] = [
    0.2408,   // Mercury
    0.6152,   // Venus
    1.0000,   // Earth
    1.8809,   // Mars
    11.862,   // Jupiter
    29.457,   // Saturn
    84.011,   // Uranus
    164.79,   // Neptune
    -18.613,  // Lunar node (negative = retrograde)
];

/// Mean ecliptic longitudes at J2000.0 (degrees).
pub const ORBITAL_L0: [f64; N_BODIES] = [
    252.25,   // Mercury
    181.98,   // Venus
    100.46,   // Earth
    355.45,   // Mars
    34.35,    // Jupiter
    49.94,    // Saturn
    313.23,   // Uranus
    304.88,   // Neptune
    125.04,   // Lunar node
];

pub const BODY_NAMES: [&str; N_BODIES] = [
    "Mercury", "Venus", "Earth", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune",
    "LunarNode",
];

const J2000: f64 = 2451545.0;

/// Circular B-spline basis on S¹ for learned periodic coupling.
/// Quadratic periodic B-spline with n_knots equally spaced on [0, 2π).
#[derive(Clone, Serialize, Deserialize)]
pub struct CircularBSpline {
    pub coeffs: Vec<f32>,
    pub n_knots: usize,
}

impl CircularBSpline {
    /// Create initialized to small random values (not flat zero).
    /// The gradient is zero at flat init because sigmoid'(0) × 0 = 0.
    /// Small random values give the gradient something to grab.
    pub fn new_flat(n_knots: usize) -> Self {
        // Deterministic pseudo-random init from knot index
        let coeffs: Vec<f32> = (0..n_knots).map(|k| {
            let hash = (k as u64).wrapping_mul(2654435761).wrapping_add(42);
            ((hash % 10000) as f32 / 10000.0 - 0.5) * 0.1 // small random in [-0.05, 0.05]
        }).collect();
        CircularBSpline { coeffs, n_knots }
    }

    /// Evaluate the learned periodic function at angle theta (radians).
    pub fn forward(&self, theta: f32) -> f32 {
        let k = self.n_knots;
        let spacing = 2.0 * PI / k as f32;
        let t = ((theta % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
        let idx_f = t / spacing;
        let idx0 = idx_f.floor() as usize % k;
        let frac = idx_f - idx_f.floor();

        // Quadratic B-spline: weighted sum of 3 neighbors
        let i0 = (idx0 + k - 1) % k;
        let i1 = idx0;
        let i2 = (idx0 + 1) % k;

        let b0 = 0.5 * (1.0 - frac) * (1.0 - frac);
        let b1 = 0.5 + frac * (1.0 - frac);
        let b2 = 0.5 * frac * frac;

        self.coeffs[i0] * b0 + self.coeffs[i1] * b1 + self.coeffs[i2] * b2
    }

    /// Backward: gradient of output w.r.t. coefficients.
    pub fn backward(&self, theta: f32) -> Vec<f32> {
        let k = self.n_knots;
        let spacing = 2.0 * PI / k as f32;
        let t = ((theta % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
        let idx_f = t / spacing;
        let idx0 = idx_f.floor() as usize % k;
        let frac = idx_f - idx_f.floor();

        let i0 = (idx0 + k - 1) % k;
        let i1 = idx0;
        let i2 = (idx0 + 1) % k;

        let b0 = 0.5 * (1.0 - frac) * (1.0 - frac);
        let b1 = 0.5 + frac * (1.0 - frac);
        let b2 = 0.5 * frac * frac;

        let mut grad = vec![0.0f32; k];
        grad[i0] = b0;
        grad[i1] = b1;
        grad[i2] = b2;
        grad
    }
}

/// Planetary KAN: learnable periodic modulation from orbital geometry.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlanetaryKAN {
    /// One CircularBSpline per body (9 total).
    pub splines: Vec<CircularBSpline>,
    /// Per-body weight (how much this planet matters).
    pub weights: Vec<f32>,
    /// Bias term.
    pub bias: f32,
    /// Number of B-spline knots per body.
    pub n_knots: usize,
}

/// Gradients for PlanetaryKAN parameters.
#[derive(Clone)]
pub struct PlanetaryKANGrads {
    pub d_spline_coeffs: Vec<Vec<f32>>,  // N_BODIES × n_knots
    pub d_weights: Vec<f32>,             // N_BODIES
    pub d_bias: f32,
}

impl PlanetaryKAN {
    /// Create a new PlanetaryKAN with n_knots per body.
    /// Initialized to flat (no modulation): all spline coeffs = 0, weights small.
    pub fn new(n_knots: usize) -> Self {
        PlanetaryKAN {
            splines: (0..N_BODIES).map(|_| CircularBSpline::new_flat(n_knots)).collect(),
            weights: vec![0.01; N_BODIES], // small initial weights
            bias: 0.0,
            n_knots,
        }
    }

    /// Total learnable parameters.
    pub fn param_count(&self) -> usize {
        N_BODIES * self.n_knots + N_BODIES + 1
    }

    /// Compute orbital angles from Julian date.
    pub fn angles_from_jd(jd: f64) -> [f32; N_BODIES] {
        let t_yr = (jd - J2000) / 365.25;
        let mut angles = [0.0f32; N_BODIES];
        for i in 0..N_BODIES {
            let period = ORBITAL_PERIODS[i];
            let lon = (ORBITAL_L0[i] + 360.0 * t_yr / period) * std::f64::consts::PI / 180.0;
            angles[i] = lon as f32;
        }
        angles
    }

    /// Forward pass: compute modulation factor from orbital angles.
    ///
    /// Returns modulation ∈ [0.5, 1.5] centered at 1.0 (no effect).
    /// Each spline maps an angle to a value, weighted sum → sigmoid → scale.
    pub fn forward(&self, angles: &[f32; N_BODIES]) -> f32 {
        let mut z = self.bias;
        for i in 0..N_BODIES {
            z += self.weights[i] * self.splines[i].forward(angles[i]);
        }
        // sigmoid maps to [0,1], then scale to [0.5, 1.5]
        let sig = 1.0 / (1.0 + (-z).exp());
        0.5 + sig
    }

    /// Backward pass: compute gradients given d_output (gradient of loss w.r.t. modulation).
    pub fn backward(&self, angles: &[f32; N_BODIES], d_out: f32) -> PlanetaryKANGrads {
        // Forward recompute
        let mut z = self.bias;
        let mut spline_vals = [0.0f32; N_BODIES];
        for i in 0..N_BODIES {
            spline_vals[i] = self.splines[i].forward(angles[i]);
            z += self.weights[i] * spline_vals[i];
        }
        let sig = 1.0 / (1.0 + (-z).exp());
        let d_sig = d_out; // d_modulation/d_sig = 1.0 (linear scaling)
        let d_z = d_sig * sig * (1.0 - sig);

        let d_bias = d_z;
        let mut d_weights = vec![0.0f32; N_BODIES];
        let mut d_spline_coeffs = Vec::with_capacity(N_BODIES);

        for i in 0..N_BODIES {
            d_weights[i] = d_z * spline_vals[i];
            let d_spline_out = d_z * self.weights[i];
            let spline_grad = self.splines[i].backward(angles[i]);
            let d_coeffs: Vec<f32> = spline_grad.iter().map(|&g| g * d_spline_out).collect();
            d_spline_coeffs.push(d_coeffs);
        }

        PlanetaryKANGrads {
            d_spline_coeffs,
            d_weights,
            d_bias,
        }
    }

    /// SGD update step.
    pub fn sgd_step(&mut self, grads: &PlanetaryKANGrads, lr: f32) {
        self.bias -= lr * grads.d_bias;
        for i in 0..N_BODIES {
            self.weights[i] -= lr * grads.d_weights[i];
            for (c, &dc) in self.splines[i].coeffs.iter_mut().zip(grads.d_spline_coeffs[i].iter()) {
                *c -= lr * dc;
            }
        }
    }

    /// Print diagnostic: which planets have the largest learned weights.
    pub fn print_weights(&self) {
        let mut indexed: Vec<(usize, f32)> = self.weights.iter().enumerate()
            .map(|(i, &w)| (i, w.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Planetary KAN weights (by magnitude):");
        for (i, mag) in indexed {
            let spline_energy: f32 = self.splines[i].coeffs.iter().map(|c| c * c).sum::<f32>().sqrt();
            println!("    {:>10}: weight={:+.4}, spline_energy={:.4}",
                BODY_NAMES[i], self.weights[i], spline_energy);
        }
    }
}

/// Convert calendar date to Julian date.
pub fn date_to_jd(year: i32, month: u32, day: u32) -> f64 {
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
    fn test_flat_init_no_modulation() {
        let kan = PlanetaryKAN::new(8);
        let angles = PlanetaryKAN::angles_from_jd(date_to_jd(2026, 4, 5));
        let mod_factor = kan.forward(&angles);
        // Should be near 1.0 (sigmoid(0) = 0.5 → 0.5 + 0.5 = 1.0)
        assert!((mod_factor - 1.0).abs() < 0.01, "flat init should give ~1.0, got {mod_factor}");
    }

    #[test]
    fn test_param_count() {
        let kan = PlanetaryKAN::new(8);
        assert_eq!(kan.param_count(), 9 * 8 + 9 + 1); // 82
    }

    #[test]
    fn test_backward_finite() {
        let kan = PlanetaryKAN::new(8);
        let angles = PlanetaryKAN::angles_from_jd(date_to_jd(2024, 6, 15));
        let grads = kan.backward(&angles, 1.0);
        assert!(grads.d_bias.is_finite());
        for w in &grads.d_weights { assert!(w.is_finite()); }
        for coeffs in &grads.d_spline_coeffs {
            for c in coeffs { assert!(c.is_finite()); }
        }
    }

    #[test]
    fn test_jupiter_period() {
        let jd0 = date_to_jd(2020, 1, 1);
        let jd1 = jd0 + 11.862 * 365.25; // one Jupiter orbit
        let a0 = PlanetaryKAN::angles_from_jd(jd0);
        let a1 = PlanetaryKAN::angles_from_jd(jd1);
        let diff = (a0[4] - a1[4]).abs() % (2.0 * PI);
        let diff = diff.min(2.0 * PI - diff);
        assert!(diff < 0.02, "Jupiter should return after one period, diff={diff}");
    }
}
