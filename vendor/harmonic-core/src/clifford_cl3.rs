//! Native Cl(3,0) multivector primitives and recurrent dynamics.
//!
//! Basis: \[1, e1, e2, e3, e12, e23, e13, e123\]
//!
//! Provides the geometric product, commutator, grade projections,
//! byte→Clifford encoding, and the `CliffordCl3Block` recurrent block
//! whose dynamics follow:  ∂ₜΨ = \[Ω, Ψ\] − γ⟨Ψ⟩₂ + kick
//!
//! The bivector defect cache stores rare bivector configurations
//! (||b|| > threshold) as the constellation — self-organizing
//! morphological signatures discovered at runtime.

use std::f32::consts::PI;

// ============================================================================
// Basis masks and geometric product table
// ============================================================================

/// Cl(3,0) basis element bitmasks: [1, e1, e2, e3, e12, e23, e13, e123]
const BASIS_MASKS: [u8; 8] = [0, 1, 2, 4, 3, 6, 5, 7];

/// Signs under grade reversal: grades 0,1 → +1, grades 2,3 → −1.
const REVERSE_SIGNS: [f32; 8] = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

/// Compute sign of geometric product e_A · e_B in Euclidean Cl(3,0).
fn blade_mul_sign(a_mask: u8, b_mask: u8) -> f32 {
    let mut sign = 1i32;
    for i in 0..3u8 {
        if (a_mask >> i) & 1 == 1 {
            let lower_bits = b_mask & ((1 << i) - 1);
            if lower_bits.count_ones() % 2 == 1 {
                sign = -sign;
            }
        }
    }
    sign as f32
}

/// Dense 8×8×8 geometric product table.
/// `table[i][j][k]` = coefficient of basis_k in basis_i * basis_j.
fn build_geometric_product_table() -> [[[f32; 8]; 8]; 8] {
    let mut mask_to_idx = [0u8; 8];
    for (idx, &mask) in BASIS_MASKS.iter().enumerate() {
        mask_to_idx[mask as usize] = idx as u8;
    }

    let mut table = [[[0.0f32; 8]; 8]; 8];
    for (i, &a_mask) in BASIS_MASKS.iter().enumerate() {
        for (j, &b_mask) in BASIS_MASKS.iter().enumerate() {
            let out_mask = a_mask ^ b_mask;
            let sign = blade_mul_sign(a_mask, b_mask);
            let k = mask_to_idx[out_mask as usize] as usize;
            table[i][j][k] = sign;
        }
    }
    table
}

/// Lazily-initialized static geometric product table.
pub fn geometric_product_table() -> &'static [[[f32; 8]; 8]; 8] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[[[f32; 8]; 8]; 8]> = OnceLock::new();
    TABLE.get_or_init(build_geometric_product_table)
}

// ============================================================================
// Multivector operations
// ============================================================================

/// 8-component Cl(3,0) multivector: [scalar, e1, e2, e3, e12, e23, e13, e123].
pub type Multivector = [f32; 8];

/// Cl(3,0) geometric product: c = a * b.
pub fn geometric_product(a: &Multivector, b: &Multivector) -> Multivector {
    let table = geometric_product_table();
    let mut c = [0.0f32; 8];
    for i in 0..8 {
        for j in 0..8 {
            let ai_bj = a[i] * b[j];
            if ai_bj != 0.0 {
                for k in 0..8 {
                    c[k] += ai_bj * table[i][j][k];
                }
            }
        }
    }
    c
}

/// Scalar component of the Cl(3,0) geometric product `a · b`.
///
/// Specialised fast path that computes only the index-0 output of
/// [`geometric_product`]. For any two Cl(3,0) multivectors, the scalar part
/// of their geometric product is a signature-weighted inner product:
///
/// ```text
///   ⟨a·b⟩₀ = a[0]·b[0] + a[1]·b[1] + a[2]·b[2] + a[3]·b[3]
///          − a[4]·b[4] − a[5]·b[5] − a[6]·b[6] − a[7]·b[7]
/// ```
///
/// The sign flip on indices 4..=7 comes from
/// `e12² = e23² = e13² = e123² = −1` in Euclidean Cl(3,0) (while
/// `e1² = e2² = e3² = +1`). See the Cayley table in
/// [`build_geometric_product_table`] for the full derivation.
///
/// Used by the Clifford binding memory read path, where only the grade-0
/// projection of `rev(R) · Q` is needed to score slot similarity — avoids
/// computing the other 7 components of the full product.
#[inline(always)]
pub fn scalar_part_of_product(a: &Multivector, b: &Multivector) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
        - a[4] * b[4]
        - a[5] * b[5]
        - a[6] * b[6]
        - a[7] * b[7]
}

/// Grade reversal: vectors unchanged, bivectors and pseudoscalar flip sign.
pub fn clifford_reverse(x: &Multivector) -> Multivector {
    let mut out = *x;
    for i in 0..8 {
        out[i] *= REVERSE_SIGNS[i];
    }
    out
}

/// Clifford commutator: [a, b] = a*b − b*a.
pub fn clifford_commutator(a: &Multivector, b: &Multivector) -> Multivector {
    let ab = geometric_product(a, b);
    let ba = geometric_product(b, a);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = ab[i] - ba[i];
    }
    out
}

/// Coefficient-space norm: sqrt(sum of squares).
pub fn clifford_norm(x: &Multivector) -> f32 {
    let mut s = 0.0f32;
    for &v in x.iter() {
        s += v * v;
    }
    s.max(1e-16).sqrt()
}

/// Normalize multivector by coefficient-space norm.
pub fn clifford_normalize(x: &Multivector) -> Multivector {
    let n = clifford_norm(x);
    let inv = 1.0 / n;
    let mut out = *x;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

/// Initialize Clifford state with scalar +/-1 and zero higher grades.
pub fn init_clifford_state(scalar_sign: f32) -> Multivector {
    let mut psi = [0.0f32; 8];
    psi[0] = scalar_sign;
    psi
}

// ============================================================================
// Grade projections
// ============================================================================

/// Grade-0 (scalar) part: index 0.
#[inline]
pub fn scalar_part(x: &Multivector) -> f32 {
    x[0]
}

/// Grade-1 (vector) part: indices 1,2,3  → [e1, e2, e3].
#[inline]
pub fn vector_part(x: &Multivector) -> [f32; 3] {
    [x[1], x[2], x[3]]
}

/// Grade-2 (bivector) part: indices 4,5,6  → [e12, e23, e13].
#[inline]
pub fn bivector_part(x: &Multivector) -> [f32; 3] {
    [x[4], x[5], x[6]]
}

/// Grade-3 (pseudoscalar) part: index 7.
#[inline]
pub fn pseudoscalar_part(x: &Multivector) -> f32 {
    x[7]
}

/// Norm of the bivector part.
pub fn bivector_norm(x: &Multivector) -> f32 {
    let b = bivector_part(x);
    (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).max(1e-16).sqrt()
}

/// Scalar-to-vector magnitude ratio.
pub fn scalar_vector_ratio(x: &Multivector) -> f32 {
    let v = vector_part(x);
    let v_mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).max(1e-16).sqrt();
    x[0].abs() / v_mag
}

// ============================================================================
// Similarity functions
// ============================================================================

/// Scalar channel of reverse(a) * b, normalized to [-1, 1].
pub fn clifford_scalar_similarity(a: &Multivector, b: &Multivector) -> f32 {
    let rev_a = clifford_reverse(a);
    let prod = geometric_product(&rev_a, b);
    let denom = (clifford_norm(a) * clifford_norm(b)).max(1e-8);
    (prod[0] / denom).clamp(-1.0, 1.0)
}

/// Positive Clifford similarity in [0, 1].
pub fn hermitian_similarity(a: &Multivector, b: &Multivector) -> f32 {
    ((clifford_scalar_similarity(a, b) + 1.0) * 0.5).clamp(1e-8, 1.0)
}

/// Bivector-space cosine similarity in [0, 1].
pub fn bivector_hermitian_similarity(a: &Multivector, b: &Multivector) -> f32 {
    let a_biv = bivector_part(a);
    let b_biv = bivector_part(b);
    let a_n = (a_biv[0] * a_biv[0] + a_biv[1] * a_biv[1] + a_biv[2] * a_biv[2])
        .max(1e-16)
        .sqrt();
    let b_n = (b_biv[0] * b_biv[0] + b_biv[1] * b_biv[1] + b_biv[2] * b_biv[2])
        .max(1e-16)
        .sqrt();
    let dot = a_biv[0] * b_biv[0] + a_biv[1] * b_biv[1] + a_biv[2] * b_biv[2];
    (dot / (a_n * b_n)).abs().clamp(0.0, 1.0)
}

// ============================================================================
// Bivector ↔ Multivector conversion
// ============================================================================

/// Embed 3 bivector coefficients into an 8-component multivector.
/// Fills indices 4,5,6 (e12, e23, e13); rest zero.
pub fn bivector_to_multivector(biv: &[f32; 3]) -> Multivector {
    let mut mv = [0.0f32; 8];
    mv[4] = biv[0];
    mv[5] = biv[1];
    mv[6] = biv[2];
    mv
}

/// Extract the bivector-only multivector from a full multivector.
/// Zeros all grades except grade-2.
pub fn bivector_multivector(x: &Multivector) -> Multivector {
    let mut out = [0.0f32; 8];
    out[4] = x[4];
    out[5] = x[5];
    out[6] = x[6];
    out
}

// ============================================================================
// Sandwich product and rotors (Cl(3,0) fast path)
// ============================================================================

/// Sandwich product: r * x * reverse(r).
///
/// The fundamental operation for rotations in geometric algebra.
/// When r is a unit rotor (even-grade, unit norm), this performs
/// a grade-preserving rotation. Zero heap allocation.
#[inline]
pub fn sandwich_cl3(r: &Multivector, x: &Multivector) -> Multivector {
    let rev_r = clifford_reverse(r);
    let rx = geometric_product(r, x);
    geometric_product(&rx, &rev_r)
}

/// Construct a unit rotor from a bivector axis and angle.
///
/// rotor = cos(theta/2) + sin(theta/2) * B_hat
///
/// where B_hat is the unit bivector in the plane of rotation.
/// `biv` is [e12, e23, e13] (the 3 bivector components).
/// Returns a unit multivector (rotor).
pub fn rotor_from_bivector_cl3(biv: &[f32; 3], theta: f32) -> Multivector {
    let biv_norm = (biv[0] * biv[0] + biv[1] * biv[1] + biv[2] * biv[2])
        .max(1e-16)
        .sqrt();
    let half = theta / 2.0;
    let cos_h = half.cos();
    let sin_h = half.sin();

    let mut rotor = [0.0f32; 8];
    rotor[0] = cos_h;
    if biv_norm > 1e-12 {
        let s = sin_h / biv_norm;
        rotor[4] = biv[0] * s; // e12
        rotor[5] = biv[1] * s; // e23
        rotor[6] = biv[2] * s; // e13
    }
    rotor
}

// ============================================================================
// Rotor SLERP and sign alignment (for Clifford binding memory)
// ============================================================================

/// Norm restricted to the rotor (grade 0+2) subspace.
///
/// A unit rotor in Cl(3,0) lives on S³ ⊂ ℝ⁴ spanned by {1, e12, e23, e13}
/// (indices 0, 4, 5, 6). This helper sums only those components so that
/// `rotor_norm_cl3(r) == 1` for any rotor produced by
/// [`rotor_from_bivector_cl3`] regardless of numerical noise on the
/// grade-1 / grade-3 slots.
pub fn rotor_norm_cl3(r: &Multivector) -> f32 {
    let s = r[0] * r[0] + r[4] * r[4] + r[5] * r[5] + r[6] * r[6];
    s.max(1e-16).sqrt()
}

/// Rotor-subspace inner product: ⟨r0 · reverse(r1)⟩₀ restricted to grade 0+2.
///
/// For two rotors this equals `cos(half-angle)` between them on S³ and is
/// used both for sign-alignment (double cover) and for SLERP.
#[inline]
fn rotor_dot_cl3(r0: &Multivector, r1: &Multivector) -> f32 {
    r0[0] * r1[0] + r0[4] * r1[4] + r0[5] * r1[5] + r0[6] * r1[6]
}

/// Spherical linear interpolation between two unit rotors on S³.
///
/// Operates on the grade-0+2 subspace. If the rotors are nearly parallel
/// (|dot| > 0.9995), falls back to a normalized lerp to avoid division
/// by `sin(θ) → 0`. Otherwise uses the standard SLERP formula
///
/// ```text
///   slerp(r0, r1, t) = (sin((1−t)·θ) · r0 + sin(t·θ) · r1) / sin(θ)
/// ```
///
/// Before interpolating, `r1` is sign-flipped if `⟨r0, r1⟩ < 0` so the
/// shortest great-circle arc is used (SO(3) double cover).
///
/// Output is a unit rotor: `rotor_norm_cl3(slerp(...)) ≈ 1`.
pub fn slerp_cl3(r0: &Multivector, r1: &Multivector, t: f32) -> Multivector {
    // Shortest-path sign flip on r1.
    let mut r1_use = *r1;
    let mut dot = rotor_dot_cl3(r0, &r1_use);
    if dot < 0.0 {
        for v in r1_use.iter_mut() {
            *v = -*v;
        }
        dot = -dot;
    }
    let dot = dot.clamp(-1.0, 1.0);

    // Near-parallel → normalized lerp (numerically stable).
    if dot > 0.9995 {
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = (1.0 - t) * r0[i] + t * r1_use[i];
        }
        // Re-normalize on the rotor (grade 0+2) subspace.
        let n = rotor_norm_cl3(&out);
        let inv = 1.0 / n;
        for i in 0..8 {
            out[i] *= inv;
        }
        return out;
    }

    let theta = dot.acos();
    let sin_theta = theta.sin().max(1e-12);
    let w0 = ((1.0 - t) * theta).sin() / sin_theta;
    let w1 = (t * theta).sin() / sin_theta;

    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = w0 * r0[i] + w1 * r1_use[i];
    }
    out
}

/// Analytical backward through [`slerp_cl3`].
///
/// Given `d_out` (gradient on the 8-component slerp output) and the same
/// `(r0, r1, t)` arguments used in the forward, returns
/// `(d_r0, d_r1, d_t)`.
///
/// The forward has two branches — standard SLERP and a near-parallel LERP+
/// re-normalise fallback (`|dot| > 0.9995`) — and the backward picks the
/// matching branch based on the forward-time `dot(r0, r1)`. In the non-
/// parallel branch,
///
/// ```text
///   r1_use   = sign · r1,   sign = sign(dot(r0, r1))
///   dot      = |dot(r0, r1)|   (clamped to [-1, 1])
///   θ        = acos(dot)
///   sinθ     = sin(θ)
///   w0       = sin((1−t)·θ) / sinθ
///   w1       = sin(t·θ) / sinθ
///   out[i]   = w0·r0[i] + w1·r1_use[i]
/// ```
///
/// `d_w0`, `d_w1` pull through the cos(θ), sin(θ) identities to `d_θ`, which
/// in turn flows into `d_dot = −d_θ / sinθ`. The (r0, r1_use) → dot dependency
/// is `dot = Σ_{k∈rotor slots} r0[k]·r1_use[k]`, giving the rotor-slot
/// contribution. We also pick up a direct term from `∂out/∂r0 = w0·I` and
/// `∂out/∂r1_use = w1·I`, which must be multiplied by `sign` to get
/// `∂out/∂r1`.
///
/// In the parallel branch,
///
/// ```text
///   raw[i]   = (1−t)·r0[i] + t·r1_use[i]
///   n        = sqrt(Σ_{k∈rotor slots} raw[k]^2)
///   out[i]   = raw[i] / n
/// ```
///
/// which is just a normalised LERP — the backward is a single-line
/// `d_raw = (d_out − out·(out·d_out)) / n` on the rotor slots (grade 0+2),
/// zeros elsewhere.
///
/// `t` is unclamped (the caller is responsible for keeping it in `[0, 1]`).
pub fn slerp_cl3_backward(
    r0: &Multivector,
    r1: &Multivector,
    t: f32,
    d_out: &Multivector,
) -> (Multivector, Multivector, f32) {
    // Reproduce the sign-alignment branch in lockstep with the forward.
    let dot_raw = rotor_dot_cl3(r0, r1);
    let sign = if dot_raw < 0.0 { -1.0 } else { 1.0 };
    let r1_use: Multivector = {
        let mut out = *r1;
        for v in out.iter_mut() {
            *v *= sign;
        }
        out
    };
    let dot = (sign * dot_raw).clamp(-1.0, 1.0);

    let mut d_r0 = [0.0f32; 8];
    let mut d_r1_use = [0.0f32; 8];
    let mut d_t = 0.0f32;

    if dot > 0.9995 {
        // Parallel branch: out = raw / n; only rotor slots (0,4,5,6) matter.
        // raw[i] = (1−t)·r0[i] + t·r1_use[i] on those slots; zero elsewhere.
        let mut raw = [0.0f32; 8];
        for &i in &[0usize, 4, 5, 6] {
            raw[i] = (1.0 - t) * r0[i] + t * r1_use[i];
        }
        let mut n2 = 0.0f32;
        for &i in &[0usize, 4, 5, 6] {
            n2 += raw[i] * raw[i];
        }
        let n = n2.max(1e-16).sqrt();
        let inv_n = 1.0 / n;
        let inv_n3 = inv_n * inv_n * inv_n;
        // out[i] = raw[i] / n on rotor slots. d_raw[i] = d_out[i]/n − raw[i] * (Σ_k raw[k] d_out[k]) / n^3.
        let mut raw_dot_dout = 0.0f32;
        for &i in &[0usize, 4, 5, 6] {
            raw_dot_dout += raw[i] * d_out[i];
        }
        let mut d_raw = [0.0f32; 8];
        for &i in &[0usize, 4, 5, 6] {
            d_raw[i] = d_out[i] * inv_n - raw[i] * raw_dot_dout * inv_n3;
        }
        // d_r0[i] = (1−t) · d_raw[i]; d_r1_use[i] = t · d_raw[i] on rotor slots.
        // d_t = Σ (r1_use[i] − r0[i]) · d_raw[i] on rotor slots.
        for &i in &[0usize, 4, 5, 6] {
            d_r0[i] = (1.0 - t) * d_raw[i];
            d_r1_use[i] = t * d_raw[i];
            d_t += (r1_use[i] - r0[i]) * d_raw[i];
        }
    } else {
        // Standard SLERP branch.
        let theta = dot.acos();
        let sin_theta = theta.sin().max(1e-12);
        let inv_sin = 1.0 / sin_theta;
        let a0 = (1.0 - t) * theta; // arg of sin for w0
        let a1 = t * theta;
        let sin_a0 = a0.sin();
        let sin_a1 = a1.sin();
        let cos_a0 = a0.cos();
        let cos_a1 = a1.cos();
        let w0 = sin_a0 * inv_sin;
        let w1 = sin_a1 * inv_sin;

        // --- d_w0, d_w1 from out[i] = w0·r0[i] + w1·r1_use[i].
        let mut d_w0 = 0.0f32;
        let mut d_w1 = 0.0f32;
        for i in 0..8 {
            d_w0 += r0[i] * d_out[i];
            d_w1 += r1_use[i] * d_out[i];
            // Direct contributions: ∂out/∂r0[i] = w0, ∂out/∂r1_use[i] = w1.
            d_r0[i] += w0 * d_out[i];
            d_r1_use[i] += w1 * d_out[i];
        }

        // --- d_theta from w0, w1.
        //   w0 = sin(a0)/sinθ, a0 = (1−t)·θ
        //   dw0/dθ = [ (1−t)·cos(a0)·sinθ − sin(a0)·cosθ ] / sin²θ
        //   dw0/dt = [ (−θ)·cos(a0) ] / sinθ
        //   Analogous for w1 with factor t.
        let cos_theta = theta.cos();
        let inv_sin2 = inv_sin * inv_sin;
        let dw0_dtheta = ((1.0 - t) * cos_a0 * sin_theta - sin_a0 * cos_theta) * inv_sin2;
        let dw1_dtheta = (t * cos_a1 * sin_theta - sin_a1 * cos_theta) * inv_sin2;
        let dw0_dt = -theta * cos_a0 * inv_sin;
        let dw1_dt = theta * cos_a1 * inv_sin;

        let mut d_theta = d_w0 * dw0_dtheta + d_w1 * dw1_dtheta;
        d_t += d_w0 * dw0_dt + d_w1 * dw1_dt;

        // --- d_theta → d_dot via θ = acos(dot), dθ/d_dot = −1/sinθ.
        // Guard against endpoints where sinθ → 0; the forward already uses
        // sin_theta.max(1e-12).
        let clamp_mask = dot_raw.abs() < 1.0;
        let d_dot = if clamp_mask {
            -d_theta * inv_sin
        } else {
            // dot is saturated at ±1; gradient is ill-defined but the
            // near-parallel branch should have fired already. Zero it out.
            d_theta = 0.0;
            let _ = d_theta;
            0.0
        };

        // --- d_dot flows into rotor slots of (r0, r1_use):
        //   dot = Σ_{k∈{0,4,5,6}} r0[k] · r1_use[k]
        for &k in &[0usize, 4, 5, 6] {
            d_r0[k] += d_dot * r1_use[k];
            d_r1_use[k] += d_dot * r0[k];
        }
    }

    // Un-apply the shortest-path sign to get d_r1.
    let mut d_r1 = [0.0f32; 8];
    for i in 0..8 {
        d_r1[i] = sign * d_r1_use[i];
    }

    // Numerical guards.
    for v in d_r0.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    for v in d_r1.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    if !d_t.is_finite() {
        d_t = 0.0;
    }

    (d_r0, d_r1, d_t)
}

/// Resolve the double-cover ambiguity: rotors R and −R represent the same
/// rotation, so EMA/slerp updates must pick a consistent branch.
///
/// If `⟨r_candidate, r_ref⟩₀ < 0` (rotor-subspace inner product on grade 0+2),
/// negates every component of `r_candidate` in place. Otherwise leaves it
/// untouched.
pub fn rotor_sign_align(r_candidate: &mut Multivector, r_ref: &Multivector) {
    if rotor_dot_cl3(r_candidate, r_ref) < 0.0 {
        for v in r_candidate.iter_mut() {
            *v = -*v;
        }
    }
}

// ============================================================================
// Byte → Clifford encoding
// ============================================================================

/// Fixed byte → Cl(3,0) map spanning all three vector + bivector directions.
///
/// scalar s ∈ [-1, 1] from byte value.
/// Vectors:   e1 = sin(πs/2),  e2 = cos(πs/3),  e3 = sin(πs/5)
/// Bivectors: e12 = cos(πs/2), e23 = sin(πs/7),  e13 = cos(πs/11)
///
/// The e3 vector component is essential: without it, the commutator [Ω, Ψ]
/// stays within the {e1, e2, e12} subalgebra and the full Cl(3,0) grade
/// structure is never explored. All three vector directions must be seeded
/// for the oscillator to access the complete algebra.
pub fn byte_to_clifford(byte_val: u8) -> Multivector {
    let s = (byte_val as f32 / 255.0) * 2.0 - 1.0;
    let mut out = [0.0f32; 8];
    out[0] = s; // scalar
    out[1] = (s * PI / 2.0).sin(); // e1
    out[2] = (s * PI / 3.0).cos(); // e2
    out[3] = (s * PI / 5.0).sin(); // e3  (NEW — unlocks full algebra)
    out[4] = (s * PI / 2.0).cos(); // e12
    out[5] = (s * PI / 7.0).sin(); // e23 (NEW — cross-plane bivector)
    out[6] = (s * PI / 11.0).cos(); // e13 (NEW — cross-plane bivector)
    clifford_normalize(&out)
}

// ============================================================================
// BivectorDefectCache — FIFO cache of rare bivector configurations
// ============================================================================

/// FIFO cache of rare bivector configurations (the Constellation).
///
/// Stores (bivector, byte) pairs. Max `max_size` entries, FIFO eviction.
/// Only caches bivectors with ||b|| > threshold (morphological signatures).
#[derive(Clone)]
pub struct BivectorDefectCache {
    pub max_size: usize,
    pub threshold: f32,
    pub bivectors: Vec<[f32; 3]>,
    pub bytes: Vec<u8>,
}

impl BivectorDefectCache {
    pub fn new(max_size: usize, threshold: f32) -> Self {
        BivectorDefectCache {
            max_size,
            threshold,
            bivectors: Vec::with_capacity(max_size),
            bytes: Vec::with_capacity(max_size),
        }
    }

    /// Cache bivector if rare (||b|| > threshold). FIFO eviction.
    /// Returns true if cached.
    pub fn update(&mut self, psi: &Multivector, byte_val: u8) -> bool {
        let biv = bivector_part(psi);
        let norm = (biv[0] * biv[0] + biv[1] * biv[1] + biv[2] * biv[2]).sqrt();
        if norm > self.threshold {
            self.bivectors.push(biv);
            self.bytes.push(byte_val);
            if self.bivectors.len() > self.max_size {
                self.bivectors.remove(0);
                self.bytes.remove(0);
            }
            true
        } else {
            false
        }
    }

    /// Bivector-space cosine similarity of `psi` to all cached entries.
    /// Returns Vec of (similarity, byte) pairs.
    pub fn similarity(&self, psi: &Multivector) -> Vec<(f32, u8)> {
        let biv = bivector_part(psi);
        let biv_norm = (biv[0] * biv[0] + biv[1] * biv[1] + biv[2] * biv[2])
            .max(1e-16)
            .sqrt();

        self.bivectors
            .iter()
            .zip(self.bytes.iter())
            .map(|(cached, &byte)| {
                let cn = (cached[0] * cached[0] + cached[1] * cached[1] + cached[2] * cached[2])
                    .max(1e-16)
                    .sqrt();
                let dot = biv[0] * cached[0] + biv[1] * cached[1] + biv[2] * cached[2];
                let sim = (dot / (biv_norm * cn)).abs().clamp(0.0, 1.0);
                (sim, byte)
            })
            .collect()
    }

    /// Compute logits (vocab_size) from bivector similarity to cache entries.
    /// For each byte, take MAX similarity across cache entries for that byte.
    /// Bytes with no entries get logit -1.  Output in [-1, 1].
    pub fn logits(&self, psi: &Multivector, vocab_size: usize) -> Vec<f32> {
        let mut out = vec![-1.0f32; vocab_size];

        if self.bivectors.is_empty() {
            return out;
        }

        let sims = self.similarity(psi);
        for (sim, byte) in sims {
            let idx = byte as usize;
            if idx < vocab_size {
                // MAX pooling: keep highest similarity per byte
                let logit = 2.0 * sim - 1.0; // map [0,1] → [-1,1]
                if logit > out[idx] {
                    out[idx] = logit;
                }
            }
        }
        out
    }

    pub fn fill(&self) -> usize {
        self.bivectors.len()
    }

    pub fn reset(&mut self) {
        self.bivectors.clear();
        self.bytes.clear();
    }
}

// ============================================================================
// CliffordCl3Block — Native Cl(3,0) recurrent dynamics block
// ============================================================================

/// Configuration for CliffordCl3Block.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CliffordCl3Config {
    pub n_groups: usize,
    pub dt: f32,
    pub gamma_init: f32,
}

impl Default for CliffordCl3Config {
    fn default() -> Self {
        CliffordCl3Config {
            n_groups: 1,
            dt: 0.2,
            gamma_init: 0.01,
        }
    }
}

/// Native Cl(3,0) recurrent block.
///
/// Update: dΨ/dt = [Ω, Ψ] − γ⟨Ψ⟩₂ + kick
///
/// Bivector connection Ω on S², positive damping γ,
/// isotropic bivector damping on Ψ's bivector (NOT commutator's).
pub struct CliffordCl3Block {
    pub config: CliffordCl3Config,
    /// Per-group rotation axes on S² (n_groups × 3).
    pub omega: Vec<[f32; 3]>,
    /// Per-group damping coefficients.
    pub gamma: Vec<f32>,
}

impl CliffordCl3Block {
    pub fn new(config: CliffordCl3Config) -> Self {
        let g = config.n_groups.max(1);

        // Initialize each group with a different rotation axis on S²
        let mut omega = Vec::with_capacity(g);
        for gi in 0..g {
            let angle = gi as f32 * PI / g as f32;
            let x = angle.cos();
            let y = angle.sin();
            let z = 0.0f32;
            let n = (x * x + y * y + z * z).max(1e-8).sqrt();
            omega.push([x / n, y / n, z / n]);
        }

        let gamma = vec![config.gamma_init; g];

        CliffordCl3Block {
            config,
            omega,
            gamma,
        }
    }

    /// Single dynamics step for one group.
    ///
    /// Returns (psi_next, bivector_energy, scalar_vector_ratio).
    pub fn step(
        &self,
        psi: &Multivector,
        kick: &Multivector,
        group: usize,
    ) -> (Multivector, f32, f32) {
        let omega_mv = bivector_to_multivector(&self.omega[group]);
        let gamma = self.gamma[group].max(1e-6);

        // [Ω, Ψ]
        let comm = clifford_commutator(&omega_mv, psi);

        // −γ⟨Ψ⟩₂ : damp Ψ's bivector (grade-2 of state)
        let psi_biv = bivector_multivector(psi);

        // δ = [Ω, Ψ] − γ⟨Ψ⟩₂ + kick
        let mut delta = [0.0f32; 8];
        for i in 0..8 {
            delta[i] = comm[i] - gamma * psi_biv[i] + kick[i];
        }

        // Euler step + normalize
        let mut psi_next = [0.0f32; 8];
        for i in 0..8 {
            psi_next[i] = psi[i] + self.config.dt * delta[i];
        }
        let psi_next = clifford_normalize(&psi_next);

        let bv_energy = bivector_norm(&psi_next);
        let sv_ratio = scalar_vector_ratio(&psi_next);

        (psi_next, bv_energy, sv_ratio)
    }

    /// Run causal recurrence over a token sequence (byte-level).
    ///
    /// Returns per-position states (n_groups × 8 flattened) and diagnostics.
    pub fn rollout(&self, input: &[u8]) -> (Vec<Vec<Multivector>>, CliffordDiagnostics) {
        let g = self.config.n_groups.max(1);

        // Per-group state, all start at scalar +1
        let mut psi: Vec<Multivector> = (0..g).map(|_| init_clifford_state(1.0)).collect();
        let mut all_states = Vec::with_capacity(input.len());
        let mut bv_energies = Vec::with_capacity(input.len());
        let mut sv_ratios = Vec::with_capacity(input.len());

        for &byte in input {
            let kick = byte_to_clifford(byte);

            // Per-group step
            for gi in 0..g {
                let (next, bv_e, sv_r) = self.step(&psi[gi], &kick, gi);
                psi[gi] = next;
                if gi == 0 {
                    bv_energies.push(bv_e);
                    sv_ratios.push(sv_r);
                }
            }

            all_states.push(psi.clone());
        }

        let diag = CliffordDiagnostics {
            bivector_energy: bv_energies,
            scalar_vector_ratio: sv_ratios,
            gamma: self.gamma.clone(),
            omega: self.omega.clone(),
        };
        (all_states, diag)
    }

    /// Total learnable parameters: 3 per group (omega) + 1 per group (gamma).
    pub fn n_params(&self) -> usize {
        self.config.n_groups.max(1) * 4
    }
}

/// Diagnostics from a CliffordCl3Block rollout.
pub struct CliffordDiagnostics {
    pub bivector_energy: Vec<f32>,
    pub scalar_vector_ratio: Vec<f32>,
    pub gamma: Vec<f32>,
    pub omega: Vec<[f32; 3]>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_product_identity() {
        // 1 * x = x for any x
        let one = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let x = [0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let result = geometric_product(&one, &x);
        for i in 0..8 {
            assert!(
                (result[i] - x[i]).abs() < 1e-6,
                "1*x != x at index {}: {} vs {}",
                i,
                result[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_geometric_product_e1_squared() {
        // e1 * e1 = +1 in Euclidean Cl(3,0)
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = geometric_product(&e1, &e1);
        assert!((result[0] - 1.0).abs() < 1e-6, "e1² should be +1");
        for i in 1..8 {
            assert!(result[i].abs() < 1e-6, "e1² should have zero non-scalar");
        }
    }

    #[test]
    fn test_geometric_product_e1_e2() {
        // e1 * e2 = e12
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = geometric_product(&e1, &e2);
        // e12 is at index 4
        assert!(
            (result[4] - 1.0).abs() < 1e-6,
            "e1*e2 should give e12=1, got {}",
            result[4]
        );
    }

    #[test]
    fn test_commutator_antisymmetric() {
        let a = clifford_normalize(&[0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05]);
        let b = clifford_normalize(&[-0.2, 0.6, 0.1, -0.4, 0.3, 0.5, -0.2, 0.1]);
        let comm_ab = clifford_commutator(&a, &b);
        let comm_ba = clifford_commutator(&b, &a);
        for i in 0..8 {
            assert!(
                (comm_ab[i] + comm_ba[i]).abs() < 1e-5,
                "[a,b] + [b,a] != 0 at index {}",
                i
            );
        }
    }

    #[test]
    fn test_normalize() {
        let x = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let n = clifford_normalize(&x);
        let norm = clifford_norm(&n);
        assert!((norm - 1.0).abs() < 1e-6, "Normalized should have norm 1");
    }

    #[test]
    fn test_reverse_involution() {
        // Applying reverse twice gives identity.
        let x = [0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let rr = clifford_reverse(&clifford_reverse(&x));
        for i in 0..8 {
            assert!((rr[i] - x[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_byte_to_clifford_normalized() {
        for byte in [0u8, 42, 127, 200, 255] {
            let mv = byte_to_clifford(byte);
            let norm = clifford_norm(&mv);
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "byte_to_clifford({}) should be normalized, got norm={}",
                byte,
                norm
            );
        }
    }

    #[test]
    fn test_byte_to_clifford_distinct() {
        let a = byte_to_clifford(0);
        let b = byte_to_clifford(255);
        // Scalars should differ
        assert!(
            (a[0] - b[0]).abs() > 0.1,
            "byte 0 and 255 should produce different Clifford elements"
        );
    }

    #[test]
    fn test_bivector_defect_cache_fifo() {
        let mut cache = BivectorDefectCache::new(3, 0.1);
        // Create multivectors with varying bivector norms
        for i in 0..5 {
            let mut mv = [0.0f32; 8];
            mv[4] = 0.5 + i as f32 * 0.1; // bivector norm > threshold
            cache.update(&mv, i as u8);
        }
        // Should have evicted oldest, keeping last 3
        assert_eq!(cache.fill(), 3);
        assert_eq!(cache.bytes, vec![2, 3, 4]);
    }

    #[test]
    fn test_bivector_defect_cache_threshold() {
        let mut cache = BivectorDefectCache::new(10, 0.7);
        // Small bivector — should NOT be cached
        let small = [1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]; // norm ~0.17
        assert!(!cache.update(&small, 42));
        assert_eq!(cache.fill(), 0);

        // Large bivector — should be cached
        let large = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]; // norm ~0.87
        assert!(cache.update(&large, 99));
        assert_eq!(cache.fill(), 1);
    }

    #[test]
    fn test_cache_logits_max_pooling() {
        let mut cache = BivectorDefectCache::new(100, 0.1);
        // Add two entries for the same byte with different bivectors
        let mut mv1 = [0.0f32; 8];
        mv1[4] = 1.0;
        cache.update(&mv1, 42);

        let mut mv2 = [0.0f32; 8];
        mv2[5] = 1.0;
        cache.update(&mv2, 42);

        // Query with mv1 — should get high similarity to byte 42
        let logits = cache.logits(&mv1, 256);
        assert!(logits[42] > logits[0], "Byte 42 should have higher logit");
    }

    #[test]
    fn test_cl3_block_creation() {
        let config = CliffordCl3Config {
            n_groups: 4,
            dt: 0.2,
            gamma_init: 0.01,
        };
        let block = CliffordCl3Block::new(config);
        assert_eq!(block.omega.len(), 4);
        assert_eq!(block.gamma.len(), 4);
        assert_eq!(block.n_params(), 16); // 4 groups × (3 omega + 1 gamma)
    }

    #[test]
    fn test_cl3_block_step_preserves_norm() {
        let block = CliffordCl3Block::new(CliffordCl3Config::default());
        let psi = init_clifford_state(1.0);
        let kick = byte_to_clifford(65); // 'A'
        let (next, _, _) = block.step(&psi, &kick, 0);
        let norm = clifford_norm(&next);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Step should preserve norm, got {}",
            norm
        );
    }

    #[test]
    fn test_cl3_block_rollout_finite() {
        let config = CliffordCl3Config {
            n_groups: 2,
            dt: 0.2,
            gamma_init: 0.05,
        };
        let block = CliffordCl3Block::new(config);
        let input = b"Hello, world!";
        let (states, diag) = block.rollout(input);
        assert_eq!(states.len(), input.len());
        assert_eq!(diag.bivector_energy.len(), input.len());

        // Check all values are finite
        for state in &states {
            for mv in state {
                for &v in mv.iter() {
                    assert!(v.is_finite(), "State contains non-finite value");
                }
            }
        }
    }

    #[test]
    fn test_bivector_hermitian_similarity_self() {
        let mv = clifford_normalize(&[0.5, 0.1, -0.3, 0.7, 0.5, -0.3, 0.4, 0.05]);
        let sim = bivector_hermitian_similarity(&mv, &mv);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_hermitian_similarity_range() {
        let a = clifford_normalize(&[0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05]);
        let b = clifford_normalize(&[-0.2, 0.6, 0.1, -0.4, 0.3, 0.5, -0.2, 0.1]);
        let sim = hermitian_similarity(&a, &b);
        assert!(sim >= 0.0 && sim <= 1.0, "Similarity out of range: {}", sim);
    }

    #[test]
    fn test_grade_projections_orthogonal() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(scalar_part(&x), 1.0);
        assert_eq!(vector_part(&x), [2.0, 3.0, 4.0]);
        assert_eq!(bivector_part(&x), [5.0, 6.0, 7.0]);
        assert_eq!(pseudoscalar_part(&x), 8.0);
    }

    #[test]
    fn test_sandwich_grade_preserving() {
        // Rotating a vector should produce a vector (no scalar/bivector/pseudoscalar)
        let rotor = rotor_from_bivector_cl3(&[1.0, 0.0, 0.0], PI / 4.0);
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rotated = sandwich_cl3(&rotor, &e1);

        // Scalar and pseudoscalar parts should be ~0
        assert!(rotated[0].abs() < 1e-5, "Scalar part: {}", rotated[0]);
        assert!(rotated[7].abs() < 1e-5, "Pseudoscalar part: {}", rotated[7]);
        // Bivector parts should be ~0
        assert!(rotated[4].abs() < 1e-5, "e12 part: {}", rotated[4]);
        assert!(rotated[5].abs() < 1e-5, "e23 part: {}", rotated[5]);
        assert!(rotated[6].abs() < 1e-5, "e13 part: {}", rotated[6]);
    }

    #[test]
    fn test_rotor_unit_norm() {
        let rotor = rotor_from_bivector_cl3(&[0.3, -0.5, 0.8], 1.23);
        let norm = clifford_norm(&rotor);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Rotor should have unit norm, got {}",
            norm
        );
    }

    #[test]
    fn slerp_cl3_endpoints() {
        let r0 = rotor_from_bivector_cl3(&[1.0, 0.0, 0.0], 0.3);
        let r1 = rotor_from_bivector_cl3(&[0.0, 1.0, 0.0], 1.1);

        let at0 = slerp_cl3(&r0, &r1, 0.0);
        // At t=0 we must recover r0 exactly (possibly up to a sign flip on r1,
        // which does not affect the t=0 endpoint).
        for i in 0..8 {
            assert!(
                (at0[i] - r0[i]).abs() < 1e-6,
                "slerp(r0,r1,0)[{}] = {} vs {}",
                i,
                at0[i],
                r0[i]
            );
        }

        let at1 = slerp_cl3(&r0, &r1, 1.0);
        // At t=1 we must recover r1 up to the shortest-path sign flip.
        let dot = rotor_dot_cl3(&r0, &r1);
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..8 {
            assert!(
                (at1[i] - sign * r1[i]).abs() < 1e-6,
                "slerp(r0,r1,1)[{}] = {} vs {}",
                i,
                at1[i],
                sign * r1[i]
            );
        }
    }

    #[test]
    fn slerp_cl3_midpoint_is_unit() {
        // Two arbitrary non-parallel rotors.
        let r0 = rotor_from_bivector_cl3(&[0.3, -0.5, 0.8], 0.9);
        let r1 = rotor_from_bivector_cl3(&[-0.6, 0.2, 0.4], 2.1);
        // Sanity: both inputs are unit rotors.
        assert!((rotor_norm_cl3(&r0) - 1.0).abs() < 1e-5);
        assert!((rotor_norm_cl3(&r1) - 1.0).abs() < 1e-5);

        let mid = slerp_cl3(&r0, &r1, 0.5);
        let n = rotor_norm_cl3(&mid);
        assert!(
            (n - 1.0).abs() < 1e-5,
            "slerp midpoint should be unit rotor, got norm {}",
            n
        );
    }

    #[test]
    fn rotor_sign_align_flips_when_opposite() {
        let r = rotor_from_bivector_cl3(&[0.2, -0.7, 0.4], 1.3);
        let mut r_neg = r;
        for v in r_neg.iter_mut() {
            *v = -*v;
        }
        // Pre-condition: scalar parts have opposite sign.
        assert!(
            r[0] * r_neg[0] < 0.0,
            "test setup: r and -r must disagree on scalar sign"
        );

        rotor_sign_align(&mut r_neg, &r);
        // Post-condition: grade-0 components now agree in sign and every
        // component matches `r` to within fp noise.
        assert!(
            r[0] * r_neg[0] >= 0.0,
            "after align, scalar parts should agree in sign"
        );
        for i in 0..8 {
            assert!(
                (r_neg[i] - r[i]).abs() < 1e-6,
                "after align, component {} = {} vs {}",
                i,
                r_neg[i],
                r[i]
            );
        }
    }

    #[test]
    fn rotor_sign_align_leaves_aligned_rotors_untouched() {
        let r_ref = rotor_from_bivector_cl3(&[1.0, 0.0, 0.0], 0.5);
        let before = rotor_from_bivector_cl3(&[0.0, 1.0, 0.0], 0.7);
        // Pick a candidate whose rotor-dot with r_ref is positive so no flip
        // should occur.
        assert!(rotor_dot_cl3(&before, &r_ref) >= 0.0);
        let mut after = before;
        rotor_sign_align(&mut after, &r_ref);
        for i in 0..8 {
            assert!((after[i] - before[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn rotor_norm_cl3_of_rotor_is_one() {
        // Several (bivector, angle) pairs — all must produce unit rotors.
        let cases: [([f32; 3], f32); 4] = [
            ([1.0, 0.0, 0.0], 0.0),
            ([0.3, -0.5, 0.8], 1.23),
            ([-0.7, 0.1, 0.4], -2.8),
            ([0.0, 0.0, 1.0], PI),
        ];
        for (biv, theta) in cases.iter() {
            let r = rotor_from_bivector_cl3(biv, *theta);
            let n = rotor_norm_cl3(&r);
            assert!(
                (n - 1.0).abs() < 1e-5,
                "rotor_norm_cl3 != 1 for (biv={:?}, theta={}): got {}",
                biv,
                theta,
                n
            );
        }
    }

    #[test]
    fn scalar_part_of_product_matches_full_geometric_product() {
        // For 10 random multivector pairs, the specialised fast path must
        // match scalar_part(geometric_product(a, b)) to f32 precision.
        use std::f32::consts::PI;
        for seed in 0..10u32 {
            // Simple reproducible pseudo-random sequence (no external RNG).
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            for i in 0..8 {
                let t = (seed as f32 * 0.37 + i as f32 * 0.91).sin();
                let u = (seed as f32 * 0.53 + i as f32 * 1.21 + PI / 3.0).cos();
                a[i] = t;
                b[i] = u;
            }
            let full = scalar_part(&geometric_product(&a, &b));
            let fast = scalar_part_of_product(&a, &b);
            let diff = (full - fast).abs();
            assert!(
                diff < 1e-5,
                "seed={seed}: full={full}, fast={fast}, |Δ|={diff}"
            );
        }
    }

    #[test]
    fn slerp_cl3_backward_matches_finite_diff() {
        // Non-parallel rotors (forces the standard SLERP branch).
        let r0 = rotor_from_bivector_cl3(&[0.3, -0.5, 0.8], 0.9);
        let r1 = rotor_from_bivector_cl3(&[-0.6, 0.2, 0.4], 2.1);
        let t = 0.37f32;
        // Random d_out.
        let mut d_out = [0.0f32; 8];
        for (i, v) in d_out.iter_mut().enumerate() {
            *v = ((i as f32 * 0.71 + 0.3).sin() * 0.8).cos();
        }
        let (d_r0, d_r1, d_t) = slerp_cl3_backward(&r0, &r1, t, &d_out);

        let fd_t = {
            let eps = 1e-3f32;
            let yp = slerp_cl3(&r0, &r1, t + eps);
            let ym = slerp_cl3(&r0, &r1, t - eps);
            let mut s = 0.0f32;
            for i in 0..8 {
                s += (yp[i] - ym[i]) / (2.0 * eps) * d_out[i];
            }
            s
        };
        let rel = (d_t - fd_t).abs() / fd_t.abs().max(d_t.abs()).max(1e-4);
        assert!(
            rel < 0.05 || (d_t - fd_t).abs() < 1e-3,
            "slerp d_t analytical={} fd={} rel={}",
            d_t,
            fd_t,
            rel
        );
        // Check d_r0[0] and d_r1[4] via FD.
        for (r_idx, j) in [(0usize, 0usize), (1, 4)] {
            let eps = 1e-3f32;
            let mut rp0 = r0;
            let mut rp1 = r1;
            if r_idx == 0 {
                rp0[j] += eps;
            } else {
                rp1[j] += eps;
            }
            let mut rm0 = r0;
            let mut rm1 = r1;
            if r_idx == 0 {
                rm0[j] -= eps;
            } else {
                rm1[j] -= eps;
            }
            let yp = slerp_cl3(&rp0, &rp1, t);
            let ym = slerp_cl3(&rm0, &rm1, t);
            let mut s = 0.0f32;
            for i in 0..8 {
                s += (yp[i] - ym[i]) / (2.0 * eps) * d_out[i];
            }
            let an = if r_idx == 0 { d_r0[j] } else { d_r1[j] };
            let rel = (an - s).abs() / s.abs().max(an.abs()).max(1e-4);
            assert!(
                rel < 0.1 || (an - s).abs() < 2e-3,
                "slerp d_r{}[{}] analytical={} fd={} rel={}",
                r_idx,
                j,
                an,
                s,
                rel
            );
        }
    }

    #[test]
    fn test_rotor_90deg_e12_sends_e1_to_e2() {
        // In the e12 plane, a 90° rotation sends e1 → e2.
        let rotor = rotor_from_bivector_cl3(&[1.0, 0.0, 0.0], PI / 2.0);
        let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rotated = sandwich_cl3(&rotor, &e1);

        // Should be approximately ±e2
        assert!(
            rotated[1].abs() < 1e-5,
            "e1 component should be ~0, got {}",
            rotated[1]
        );
        assert!(
            (rotated[2].abs() - 1.0).abs() < 1e-5,
            "|e2 component| should be ~1, got {}",
            rotated[2]
        );
        assert!(
            rotated[3].abs() < 1e-5,
            "e3 component should be ~0, got {}",
            rotated[3]
        );
    }
}
