//! Analytical BPTT for SolarFlareV2 Clifford lattice dynamics.
//!
//! Provides Jacobians for Cl(3,0) operations so that gradients flow
//! through the dynamics integration steps without finite-difference.
//!
//! The key insight: geometric_product(a, b) is bilinear, so its Jacobians
//! are just the left/right multiplication matrices from the GP table.
//! The commutator [a, b] = ab - ba has Jacobian that's the difference
//! of these matrices.

use harmonic_core::clifford_cl3::{
    clifford_norm, geometric_product_table, Multivector,
};

/// 8×8 matrix type for Cl(3,0) Jacobians.
pub type Mat88 = [[f32; 8]; 8];

/// Left multiplication matrix: geometric_product(a, b) = M_L(a) * b.
/// d(a*b)/db = M_L(a).
pub fn gp_left_matrix(a: &Multivector) -> Mat88 {
    let table = geometric_product_table();
    let mut m = [[0.0f32; 8]; 8];
    // c_k = Σ_i Σ_j a_i * b_j * T[i][j][k]
    // d c_k / d b_j = Σ_i a_i * T[i][j][k]
    // So M_L[k][j] = Σ_i a_i * T[i][j][k]
    for k in 0..8 {
        for j in 0..8 {
            let mut val = 0.0f32;
            for i in 0..8 {
                val += a[i] * table[i][j][k];
            }
            m[k][j] = val;
        }
    }
    m
}

/// Right multiplication matrix: geometric_product(a, b) = M_R(b) * a.
/// d(a*b)/da = M_R(b).
pub fn gp_right_matrix(b: &Multivector) -> Mat88 {
    let table = geometric_product_table();
    let mut m = [[0.0f32; 8]; 8];
    // c_k = Σ_i Σ_j a_i * b_j * T[i][j][k]
    // d c_k / d a_i = Σ_j b_j * T[i][j][k]
    // So M_R[k][i] = Σ_j b_j * T[i][j][k]
    for k in 0..8 {
        for i in 0..8 {
            let mut val = 0.0f32;
            for j in 0..8 {
                val += b[j] * table[i][j][k];
            }
            m[k][i] = val;
        }
    }
    m
}

/// Jacobian of clifford_commutator(a, b) = a*b - b*a w.r.t. a.
/// d[a,b]/da = M_R(b) - M_L(b)ᵀ  (since d(ba)/da = M_L_of_ba w.r.t. a)
/// Actually: d(ab)/da = M_R(b), d(ba)/da = transpose of gp_left_matrix applied differently.
/// Let's compute directly: [a,b] = ab - ba.
/// d(ab)/da = M_R(b) (right-multiply by b)
/// d(ba)/da: ba = geometric_product(b, a), so d(ba)/da_i = Σ_j b_j * T[j][i][k]
/// which is gp_right_matrix but with b in the first slot: M_L(b) transposed... no.
/// Actually d(ba)/da is just gp_left_matrix(b) since ba = M_L(b) * a.
pub fn commutator_jacobian_a(a: &Multivector, b: &Multivector) -> Mat88 {
    let mr_b = gp_right_matrix(b); // d(ab)/da
    let ml_b = gp_left_matrix(b);  // d(ba)/da
    let mut j = [[0.0f32; 8]; 8];
    for k in 0..8 {
        for i in 0..8 {
            j[k][i] = mr_b[k][i] - ml_b[k][i];
        }
    }
    j
}

/// Jacobian of clifford_commutator(a, b) = a*b - b*a w.r.t. b.
pub fn commutator_jacobian_b(a: &Multivector, b: &Multivector) -> Mat88 {
    let ml_a = gp_left_matrix(a);  // d(ab)/db
    let mr_a = gp_right_matrix(a); // d(ba)/db
    let mut j = [[0.0f32; 8]; 8];
    for k in 0..8 {
        for i in 0..8 {
            j[k][i] = ml_a[k][i] - mr_a[k][i];
        }
    }
    j
}

/// Jacobian of clifford_normalize(x) = x / ||x||.
/// d(x/||x||)/dx = (I - x̂ x̂ᵀ) / ||x||
pub fn normalize_jacobian(x: &Multivector) -> Mat88 {
    let norm = clifford_norm(x);
    let inv_norm = 1.0 / norm;
    let mut x_hat = [0.0f32; 8];
    for i in 0..8 {
        x_hat[i] = x[i] * inv_norm;
    }
    let mut j = [[0.0f32; 8]; 8];
    for k in 0..8 {
        for i in 0..8 {
            let kronecker = if k == i { 1.0 } else { 0.0 };
            j[k][i] = (kronecker - x_hat[k] * x_hat[i]) * inv_norm;
        }
    }
    j
}

/// Jacobian of bivector_multivector(x): extracts grade-2 components.
/// This is just a projection matrix: identity on indices 4,5,6, zero elsewhere.
pub fn bivector_projection_jacobian() -> Mat88 {
    let mut j = [[0.0f32; 8]; 8];
    j[4][4] = 1.0;
    j[5][5] = 1.0;
    j[6][6] = 1.0;
    j
}

/// Multiply 8×8 matrix by 8-vector: M * v.
#[inline]
pub fn mat88_mul_vec(m: &Mat88, v: &Multivector) -> Multivector {
    let mut r = [0.0f32; 8];
    for k in 0..8 {
        for i in 0..8 {
            r[k] += m[k][i] * v[i];
        }
    }
    r
}

/// Multiply transpose of 8×8 matrix by 8-vector: Mᵀ * v.
#[inline]
pub fn mat88t_mul_vec(m: &Mat88, v: &Multivector) -> Multivector {
    let mut r = [0.0f32; 8];
    for i in 0..8 {
        for k in 0..8 {
            r[i] += m[k][i] * v[k];
        }
    }
    r
}

/// Multiply two 8×8 matrices: C = A * B.
#[inline]
pub fn mat88_mul(a: &Mat88, b: &Mat88) -> Mat88 {
    let mut c = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Add scaled matrix: A += s * B.
#[inline]
pub fn mat88_add_scaled(a: &mut Mat88, b: &Mat88, s: f32) {
    for i in 0..8 {
        for j in 0..8 {
            a[i][j] += s * b[i][j];
        }
    }
}

/// Identity 8×8 matrix.
pub fn mat88_identity() -> Mat88 {
    let mut m = [[0.0f32; 8]; 8];
    for i in 0..8 {
        m[i][i] = 1.0;
    }
    m
}

// ============================================================================
// BPTT through one dynamics step
// ============================================================================

/// Analytical backward through one Euler dynamics step:
///   pre_norm = psi + dt * delta
///   psi_next = normalize(pre_norm)
///
/// where delta = j_model * (comm + coupling) - gamma * psi_biv + repulsion + forcing * kick
///
/// Returns: (d_psi, d_omega_accum, d_gamma_accum)
/// d_psi is the gradient w.r.t. the input state psi.
/// d_omega_accum and d_gamma_accum accumulate parameter gradients.
pub fn step_backward(
    // Forward quantities (saved during forward)
    psi: &Multivector,          // input state
    omega_mv: &Multivector,     // omega as multivector (bivector)
    kick: &Multivector,         // forcing for this timestep
    neighbor_states: &[Multivector], // states of all sites (for coupling)
    coupling_row: &[f32],       // K_ij for this site's row
    j_model: f32,
    gamma: f32,
    forcing_scale: f32,
    dt: f32,
    pre_norm: &Multivector,     // psi + dt * delta (before normalization)
    // Backward input
    d_psi_next: &Multivector,   // gradient from the next step
) -> (
    Multivector,    // d_psi (gradient w.r.t. input state)
    [f32; 3],       // d_omega (3 bivector components)
    f32,            // d_gamma
    Vec<Multivector>, // d_neighbor_states (one per neighbor)
    Vec<f32>,       // d_coupling_row
) {
    let n = neighbor_states.len();

    // --- Backward through normalization ---
    let j_norm = normalize_jacobian(pre_norm);
    let d_pre_norm = mat88t_mul_vec(&j_norm, d_psi_next);

    // --- Backward through Euler step: pre_norm = psi + dt * delta ---
    // d_psi += d_pre_norm (identity term)
    // d_delta = dt * d_pre_norm
    let mut d_psi = d_pre_norm;
    let mut d_delta = [0.0f32; 8];
    for i in 0..8 {
        d_delta[i] = dt * d_pre_norm[i];
    }

    // --- Backward through delta = j_model*(comm + coupling) - gamma*psi_biv + rep + forcing*kick ---

    // d(j_model * comm)/d_comm = j_model * I
    // d(comm)/d(psi): comm = [omega_mv, psi], so d_comm/d_psi = commutator_jacobian_b(omega_mv, psi)
    let j_comm_psi = commutator_jacobian_b(omega_mv, psi);
    // d_psi += j_model * d_delta ᵀ · J_comm_psi
    for i in 0..8 {
        for k in 0..8 {
            d_psi[i] += j_model * d_delta[k] * j_comm_psi[k][i];
        }
    }

    // d_omega: comm = [omega_mv, psi], d_comm/d_omega = commutator_jacobian_a(omega_mv, psi)
    let j_comm_omega = commutator_jacobian_a(omega_mv, psi);
    let mut d_omega_mv = [0.0f32; 8];
    for i in 0..8 {
        for k in 0..8 {
            d_omega_mv[i] += j_model * d_delta[k] * j_comm_omega[k][i];
        }
    }
    // omega_mv has nonzero only at indices 4,5,6 (bivector)
    let d_omega = [d_omega_mv[4], d_omega_mv[5], d_omega_mv[6]];

    // d_gamma: -gamma * psi_biv → d_gamma = -Σ_k d_delta[k] * psi_biv[k]
    let d_gamma = -(d_delta[4] * psi[4] + d_delta[5] * psi[5] + d_delta[6] * psi[6]);

    // d_psi from -gamma * psi_biv term: only indices 4,5,6
    d_psi[4] -= gamma * d_delta[4];
    d_psi[5] -= gamma * d_delta[5];
    d_psi[6] -= gamma * d_delta[6];

    // d_coupling and d_neighbor_states from coupling term
    let mut d_neighbors = vec![[0.0f32; 8]; n];
    let mut d_coupling_row = vec![0.0f32; n];
    let site_idx = 0; // the current site (we process one at a time)

    for fj in 0..n {
        let k_ij = coupling_row[fj];
        if k_ij.abs() < 1e-10 {
            continue;
        }
        // coupling_force += k_ij * [neighbor[fj], psi]
        // d(k_ij * [n, p])/d_k_ij = [n, p]
        let comm_np = {
            let mut c = [0.0f32; 8];
            let ab = crate::clifford_cl3::geometric_product(&neighbor_states[fj], psi);
            let ba = crate::clifford_cl3::geometric_product(psi, &neighbor_states[fj]);
            for i in 0..8 { c[i] = ab[i] - ba[i]; }
            c
        };
        // d_coupling_row[fj] = j_model * Σ_k d_delta[k] * comm_np[k]
        let mut dk = 0.0f32;
        for k in 0..8 {
            dk += d_delta[k] * comm_np[k];
        }
        d_coupling_row[fj] = j_model * dk;

        // d([n, p])/d_n and d([n, p])/d_p
        let j_np_n = commutator_jacobian_a(&neighbor_states[fj], psi);
        let j_np_p = commutator_jacobian_b(&neighbor_states[fj], psi);
        for i in 0..8 {
            for k in 0..8 {
                d_neighbors[fj][i] += j_model * k_ij * d_delta[k] * j_np_n[k][i];
                d_psi[i] += j_model * k_ij * d_delta[k] * j_np_p[k][i];
            }
        }
    }

    // forcing * kick term: no learnable params (kick is input, forcing_scale is from j_sun)
    // d_psi has no contribution from kick (kick is independent of psi)

    (d_psi, d_omega, d_gamma, d_neighbors, d_coupling_row)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use harmonic_core::clifford_cl3::*;

    #[test]
    fn test_gp_left_matrix() {
        let a: Multivector = [1.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05];
        let b: Multivector = [0.3, -0.2, 0.6, 0.1, -0.4, 0.2, -0.1, 0.3];
        let ab = geometric_product(&a, &b);
        let ml = gp_left_matrix(&a);
        let ab_via_mat = mat88_mul_vec(&ml, &b);
        for i in 0..8 {
            assert!(
                (ab[i] - ab_via_mat[i]).abs() < 1e-5,
                "M_L mismatch at {}: {} vs {}",
                i, ab[i], ab_via_mat[i]
            );
        }
    }

    #[test]
    fn test_gp_right_matrix() {
        let a: Multivector = [1.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05];
        let b: Multivector = [0.3, -0.2, 0.6, 0.1, -0.4, 0.2, -0.1, 0.3];
        let ab = geometric_product(&a, &b);
        let mr = gp_right_matrix(&b);
        let ab_via_mat = mat88_mul_vec(&mr, &a);
        for i in 0..8 {
            assert!(
                (ab[i] - ab_via_mat[i]).abs() < 1e-5,
                "M_R mismatch at {}: {} vs {}",
                i, ab[i], ab_via_mat[i]
            );
        }
    }

    #[test]
    fn test_commutator_jacobian_finite_diff() {
        let a: Multivector = [1.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05];
        let b: Multivector = [0.3, -0.2, 0.6, 0.1, -0.4, 0.2, -0.1, 0.3];
        let eps = 1e-4f32;

        let j_a = commutator_jacobian_a(&a, &b);
        let comm_0 = clifford_commutator(&a, &b);

        for i in 0..8 {
            let mut a_plus = a;
            a_plus[i] += eps;
            let comm_plus = clifford_commutator(&a_plus, &b);
            for k in 0..8 {
                let fd = (comm_plus[k] - comm_0[k]) / eps;
                assert!(
                    (fd - j_a[k][i]).abs() < 1e-2,
                    "d[a,b]/da[{}][{}]: analytical={:.4} fd={:.4}",
                    k, i, j_a[k][i], fd
                );
            }
        }
    }

    #[test]
    fn test_normalize_jacobian_finite_diff() {
        let x: Multivector = [1.0, 0.5, -0.3, 0.2, 0.1, -0.1, 0.4, 0.05];
        let eps = 1e-4f32;

        let j = normalize_jacobian(&x);
        let n0 = clifford_normalize(&x);

        for i in 0..8 {
            let mut x_plus = x;
            x_plus[i] += eps;
            let n_plus = clifford_normalize(&x_plus);
            for k in 0..8 {
                let fd = (n_plus[k] - n0[k]) / eps;
                assert!(
                    (fd - j[k][i]).abs() < 1e-2,
                    "d(norm)/dx[{}][{}]: analytical={:.4} fd={:.4}",
                    k, i, j[k][i], fd
                );
            }
        }
    }
}
