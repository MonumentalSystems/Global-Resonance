//! SolarFlareModel — manifold-native solar flare prediction from SHARP time series.
//!
//! Physics-informed architecture replacing standard transformers/LSTMs with
//! harmonic oscillator dynamics on S^{d-1}:
//!
//! ```text
//! SHARP params [T, 9]
//!     → GeodesicPhaseEncoder: ψ(x) = cos(πx)·u + sin(πx)·v  (per field)
//!     → HelmholtzTemporal: γ-cumsum at 4 timescales             (multi-scale)
//!     → LoheSync: learned 9×9 coupling on S^{d-1}               (cross-field)
//!     → CliffordReadout: grade-0 + grade-2 projections           (physics features)
//!     → ClassificationHead: Linear→SiLU→Linear→Sigmoid           (P(flare))
//! ```
//!
//! Key physics mappings (from Paper 0: Algebra of Synchronization Failure):
//! - SHARP params on S^{d-1} = oscillator states on the manifold
//! - Helmholtz γ-cumsum = multi-scale temporal windows (τ = 48min to 6.4h)
//! - Lohe sync order r = natural criticality indicator (KT transition at J_c = 2/π)
//! - Grade-2 bivector readout = [F, ∇F] commutator (non-potential field complexity)
//!
//! Target: TSS ≥ 0.84 on SHARP time series (parity with SolarFlareNet/Doria Rosales).
//! ~1.5K params vs SOTA ~100K+ (physics-informed inductive bias replaces parameter count).

use harmonic_core::primitives::{silu, Linear};
use harmonic_core::sequence_ops::{gated_cumsum, gated_cumsum_backward};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Number of SHARP magnetogram parameters.
/// TOTUSJH, TOTUSJZ, USFLUX, MEANALP, R_VALUE, TOTPOT, SAVNCPP, AREA_ACR, ABSNJZH
pub const N_SHARP_FIELDS: usize = 9;

/// Number of Helmholtz temporal heads (timescales).
pub const N_HEADS: usize = 4;

/// KT critical coupling (from the synchronization failure paper).
pub const J_CRITICAL: f32 = 2.0 / PI;

// ============================================================================
// SolarFlareConfig
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarFlareConfig {
    /// Oscillator dimension per SHARP field (points on S^{d_osc-1}).
    pub d_osc: usize,
    /// Number of input timesteps (T samples at 12-min cadence).
    pub seq_len: usize,
    /// Number of Lohe sync integration steps per forward pass.
    pub sync_steps: usize,
    /// Lohe sync integration timestep.
    pub sync_dt: f32,
    /// Hidden dimension in classification head.
    pub hidden_dim: usize,
    /// Helmholtz decay timescales in samples (at 12-min cadence).
    /// Default: [4, 8, 16, 32] = [48min, 1.6h, 3.2h, 6.4h].
    pub helmholtz_taus: [f32; N_HEADS],
    /// Learning rate for manifold params (encoder weights on S^{d-1}).
    pub lr_manifold: f32,
    /// Learning rate for scalar params (classifier, gamma, coupling).
    pub lr_scalar: f32,
    /// Positive class weight for BCE loss (neg_count / pos_count).
    /// Set to 1.0 for unweighted, or e.g. 10-40 for imbalanced flare data.
    pub pos_weight: f32,
}

impl Default for SolarFlareConfig {
    fn default() -> Self {
        SolarFlareConfig {
            d_osc: 16,
            seq_len: 10,
            sync_steps: 3,
            sync_dt: 0.1,
            hidden_dim: 16,
            helmholtz_taus: [4.0, 8.0, 16.0, 32.0],
            lr_manifold: 0.01,
            lr_scalar: 0.01,
            pos_weight: 1.0,
        }
    }
}

// ============================================================================
// SolarFlareModel
// ============================================================================

/// Manifold-native solar flare prediction model.
///
/// Maps SHARP magnetogram time series through oscillator dynamics on S^{d-1}
/// to produce P(flare ≥ C5.0 within prediction window).
pub struct SolarFlareModel {
    pub config: SolarFlareConfig,

    // --- [1] Manifold Encoder: Geodesic Phase Encoding ---
    // Maps scalar SHARP values to points on S^{d_osc-1} via great circle arcs:
    //   ψ(x) = cos(π·x) · u + sin(π·x) · v
    // where (u, v) are orthonormal vectors per field. x=0→u, x=0.5→v, x=1→-u.
    // This preserves on-manifold structure by construction (no normalization needed)
    // and different input values map to genuinely different directions.
    pub encoder_u: Vec<f32>, // (N_SHARP_FIELDS, d_osc) row-major — base direction per field
    pub encoder_v: Vec<f32>, // (N_SHARP_FIELDS, d_osc) row-major — orthogonal direction per field

    // --- [2] Helmholtz Temporal ---
    // Decay rates γ for gated_cumsum, one per (head, d_osc).
    // γ = exp(-1/τ) where τ is in samples.
    // Shape: (N_HEADS * d_osc,) but we apply per field independently.
    pub helmholtz_gamma: Vec<f32>, // (N_HEADS * d_osc,)

    // --- [3] Lohe Sync ---
    // Learned coupling matrix K_{ij} between the 9 SHARP fields.
    // Antisymmetric: K_{ij} = -K_{ji}, so store upper triangle (36 params).
    // Coupling drives ψ_i toward mean-field of other fields.
    pub coupling_matrix: Vec<f32>, // (N_SHARP_FIELDS * N_SHARP_FIELDS,) full, enforced antisym

    // --- [5] Classification Head ---
    // Two-layer MLP: Linear(readout_dim → hidden) → SiLU → Linear(hidden → 1)
    pub head_w1: Linear,
    pub head_w2: Linear,
}

/// Gradients for all learnable parameters.
pub struct SolarFlareGrads {
    pub d_encoder_u: Vec<f32>,
    pub d_encoder_v: Vec<f32>,
    pub d_helmholtz_gamma: Vec<f32>,
    pub d_coupling_matrix: Vec<f32>,
    pub d_head_w1_weight: Vec<f32>,
    pub d_head_w1_bias: Vec<f32>,
    pub d_head_w2_weight: Vec<f32>,
    pub d_head_w2_bias: Vec<f32>,
}

/// Forward pass intermediate activations (saved for backward).
pub struct SolarFlareActivations {
    /// Raw SHARP input: (T, 9)
    pub input: Vec<f32>,
    /// After geodesic phase encoding: (T, 9, d_osc) — on S^{d-1} by construction.
    /// ψ(x) = cos(πx)·u + sin(πx)·v, so no separate `encoded_raw` is needed.
    pub encoded: Vec<f32>,
    /// After Helmholtz: (N_HEADS, T, 9, d_osc)
    pub helmholtz_out: Vec<Vec<f32>>,
    /// After Helmholtz mean: (T, 9, d_osc)
    pub temporal: Vec<f32>,
    /// After Lohe sync: (T, 9, d_osc)
    pub synced: Vec<f32>,
    /// Aggregated state: (9 * d_osc,)
    pub aggregated: Vec<f32>,
    /// Readout features: (readout_dim,)
    pub readout: Vec<f32>,
    /// Hidden layer output: (hidden_dim,)
    pub hidden: Vec<f32>,
    /// Pre-sigmoid logit
    pub logit: f32,
    /// Final probability
    pub prob: f32,
}

impl SolarFlareModel {
    pub fn new(config: SolarFlareConfig) -> Self {
        let d = config.d_osc;
        let n = N_SHARP_FIELDS;

        // Initialize encoder orthonormal frames (u, v) per field.
        // Each field gets a unique great circle on S^{d-1}.
        // u_fi: random unit vector, v_fi: random unit vector orthogonalized to u_fi.
        let mut encoder_u = vec![0.0f32; n * d];
        let mut encoder_v = vec![0.0f32; n * d];
        for field in 0..n {
            let base = field * d;
            // Generate pseudo-random u
            for j in 0..d {
                let hash = (((field * d + j) as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(0)
                    >> 16) as f32
                    / 65536.0
                    - 0.5;
                encoder_u[base + j] = hash;
            }
            // Normalize u to S^{d-1}
            let norm_u: f32 = (0..d)
                .map(|j| encoder_u[base + j] * encoder_u[base + j])
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for j in 0..d {
                encoder_u[base + j] /= norm_u;
            }
            // Generate pseudo-random v
            for j in 0..d {
                let hash = (((field * d + j) as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(7919)
                    >> 16) as f32
                    / 65536.0
                    - 0.5;
                encoder_v[base + j] = hash;
            }
            // Gram-Schmidt: v = v - <v,u>u, then normalize.
            // Two passes for numerical stability (classical GS can leave residual <u,v> ~ 1e-3 in f32).
            for _gs_pass in 0..2 {
                let dot: f32 = (0..d)
                    .map(|j| encoder_v[base + j] * encoder_u[base + j])
                    .sum();
                for j in 0..d {
                    encoder_v[base + j] -= dot * encoder_u[base + j];
                }
            }
            let norm_v: f32 = (0..d)
                .map(|j| encoder_v[base + j] * encoder_v[base + j])
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for j in 0..d {
                encoder_v[base + j] /= norm_v;
            }
        }

        // Helmholtz gamma: exp(-1/τ) for each head
        let mut helmholtz_gamma = Vec::with_capacity(N_HEADS * d);
        for head in 0..N_HEADS {
            let tau = config.helmholtz_taus[head];
            let gamma = (-1.0 / tau).exp();
            for _ in 0..d {
                helmholtz_gamma.push(gamma);
            }
        }

        // Coupling matrix: initialize near J_c with small perturbation
        let mut coupling_matrix = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                // Start near the KT critical point so the physics is active from init.
                // At J_c * 0.5, the system is in the subcritical regime but close enough
                // that gradient-driven coupling changes can push it through the transition.
                let val = J_CRITICAL * 0.5;
                coupling_matrix[i * n + j] = val;
                coupling_matrix[j * n + i] = -val; // antisymmetric
            }
        }

        // Readout dimension: grade-0 (3 per group) + grade-2 (cross-group pairs)
        // + sync_order + phase (2 per field) + raw input skip (9)
        let readout_dim = 3 * n + n * (n - 1) / 2 + 1 + 2 * n + n;

        let head_w1 = make_linear(readout_dim, config.hidden_dim);
        let mut head_w2 = make_linear(config.hidden_dim, 1);
        // Initialize output bias to class prior logit: log(p/(1-p)) where p ≈ 0.20.
        // This breaks the "predict all positive" symmetry from epoch 1.
        if let Some(ref mut bias) = head_w2.bias {
            bias[0] = -1.4; // log(0.20 / 0.80) ≈ -1.39
        }

        SolarFlareModel {
            config,
            encoder_u,
            encoder_v,
            helmholtz_gamma,
            coupling_matrix,
            head_w1,
            head_w2,
        }
    }

    /// Readout dimension: 3*9 (grade-0) + 36 (grade-2 pairs) + 1 (sync) + 2*9 (phase) + 9 (raw skip) = 91
    pub fn readout_dim(&self) -> usize {
        let n = N_SHARP_FIELDS;
        3 * n + n * (n - 1) / 2 + 1 + 2 * n + n
    }

    /// Total number of learnable parameters.
    pub fn param_count(&self) -> usize {
        let d = self.config.d_osc;
        let n = N_SHARP_FIELDS;
        let enc = 2 * n * d; // encoder u + v
        let helm = N_HEADS * d; // gamma per head
        let coup = n * n; // coupling matrix
        let rdim = self.readout_dim();
        let h1 = rdim * self.config.hidden_dim + self.config.hidden_dim; // w1 + b1
        let h2 = self.config.hidden_dim * 1 + 1; // w2 + b2
        enc + helm + coup + h1 + h2
    }

    // ========================================================================
    // Forward pass
    // ========================================================================

    /// Full forward pass: SHARP time series → P(flare).
    ///
    /// `input`: (T * 9) flat row-major, min-max normalized SHARP params.
    /// Returns (probability, activations for backward).
    pub fn forward(&self, input: &[f32]) -> (f32, SolarFlareActivations) {
        let t = self.config.seq_len;
        let n = N_SHARP_FIELDS;
        let d = self.config.d_osc;
        assert_eq!(input.len(), t * n, "Input must be T * 9");

        // [1] Geodesic Phase Encoder: ψ(x) = cos(π·x)·u + sin(π·x)·v
        // Maps scalar SHARP values to points on S^{d-1} via great circle arcs.
        // On-manifold by construction — no normalization needed.
        // Different input values map to genuinely different directions (unlike linear+normalize).
        let mut encoded = vec![0.0f32; t * n * d];

        for ti in 0..t {
            for fi in 0..n {
                let x = input[ti * n + fi];
                let theta = PI * x;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                let u_base = fi * d;
                let enc_base = ti * n * d + fi * d;
                for di in 0..d {
                    encoded[enc_base + di] = cos_theta * self.encoder_u[u_base + di]
                        + sin_theta * self.encoder_v[u_base + di];
                }
            }
        }

        // [2] Helmholtz Temporal: γ-cumsum per head, per field
        // For each field, apply gated_cumsum across time with head-specific gamma.
        // Then average across heads.
        let mut helmholtz_out: Vec<Vec<f32>> = Vec::with_capacity(N_HEADS);
        let mut temporal = vec![0.0f32; t * n * d];

        for head in 0..N_HEADS {
            let gamma_start = head * d;
            let gamma_slice = &self.helmholtz_gamma[gamma_start..gamma_start + d];

            // Process each field independently through gated_cumsum
            let mut head_out = vec![0.0f32; t * n * d];
            for fi in 0..n {
                // Extract field fi across all timesteps: (T, d)
                let mut field_seq = vec![0.0f32; t * d];
                for ti in 0..t {
                    for di in 0..d {
                        field_seq[ti * d + di] = encoded[ti * n * d + fi * d + di];
                    }
                }
                // Apply gated_cumsum: (T, d) → (T, d)
                let cumsum = gated_cumsum(&field_seq, t, d, gamma_slice);
                // Write back
                for ti in 0..t {
                    for di in 0..d {
                        head_out[ti * n * d + fi * d + di] = cumsum[ti * d + di];
                    }
                }
            }
            helmholtz_out.push(head_out);
        }

        // Average across heads
        for idx in 0..t * n * d {
            let mut sum = 0.0f32;
            for head in 0..N_HEADS {
                sum += helmholtz_out[head][idx];
            }
            temporal[idx] = sum / N_HEADS as f32;
        }

        // Re-normalize to S^{d-1} after head averaging.
        // The gated_cumsum + averaging breaks the unit-norm constraint.
        // Without this, Lohe sync operates on non-unit vectors, which changes
        // the effective coupling strength and invalidates the J_c = 2/π physics.
        for ti in 0..t {
            for fi in 0..n {
                let base = ti * n * d + fi * d;
                let norm: f32 = (0..d)
                    .map(|di| temporal[base + di] * temporal[base + di])
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                for di in 0..d {
                    temporal[base + di] /= norm;
                }
            }
        }

        // [3] Lohe Sync: cross-field coupling on S^{d-1}
        // For each timestep, run sync_steps of Lohe mean-field integration:
        //   dψ_i/dt = Σ_j K_{ij} * (ψ_j - ψ_i * <ψ_i, ψ_j>)
        // This is the tangent-space Lohe update on S^{d-1}.
        let mut synced = temporal.clone();
        let dt = self.config.sync_dt;

        for ti in 0..t {
            for _step in 0..self.config.sync_steps {
                let mut deltas = vec![0.0f32; n * d];

                for i in 0..n {
                    let psi_i_base = ti * n * d + i * d;
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let k_ij = self.coupling_matrix[i * n + j];
                        if k_ij.abs() < 1e-10 {
                            continue;
                        }
                        let psi_j_base = ti * n * d + j * d;

                        // Dot product <ψ_i, ψ_j>
                        let dot: f32 = (0..d)
                            .map(|di| synced[psi_i_base + di] * synced[psi_j_base + di])
                            .sum();

                        // Tangent vector: ψ_j - <ψ_i,ψ_j> * ψ_i (projected to tangent plane of S^{d-1} at ψ_i)
                        for di in 0..d {
                            deltas[i * d + di] +=
                                k_ij * (synced[psi_j_base + di] - dot * synced[psi_i_base + di]);
                        }
                    }
                }

                // Euler step + re-normalize
                for i in 0..n {
                    let base = ti * n * d + i * d;
                    for di in 0..d {
                        synced[base + di] += dt * deltas[i * d + di];
                    }
                    // Re-normalize to S^{d-1}
                    let norm: f32 = (0..d)
                        .map(|di| synced[base + di] * synced[base + di])
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-8);
                    for di in 0..d {
                        synced[base + di] /= norm;
                    }
                }
            }
        }

        // [4] Aggregate: take last timestep
        let last_t = t - 1;
        let aggregated: Vec<f32> = synced[last_t * n * d..(last_t + 1) * n * d].to_vec();

        // [5] Clifford Readout: grade-0 + grade-2 features + raw input skip
        let raw_input_last: Vec<f32> = (0..n).map(|fi| input[last_t * n + fi]).collect();
        let mut readout = self.compute_readout(&aggregated, &raw_input_last);
        // Guard: replace any NaN readout features with 0.0 to prevent propagation
        // (can occur with degenerate inputs where all fields are identical)
        for v in readout.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // [6] Classification head
        let h1_out = self.head_w1.forward(&readout);
        let mut hidden = vec![0.0f32; self.config.hidden_dim];
        for i in 0..self.config.hidden_dim {
            hidden[i] = silu(h1_out[i]);
        }
        let h2_out = self.head_w2.forward(&hidden);
        let logit = if h2_out[0].is_finite() {
            h2_out[0]
        } else {
            0.0
        };
        let prob = sigmoid(logit);

        let acts = SolarFlareActivations {
            input: input.to_vec(),
            encoded,
            helmholtz_out,
            temporal,
            synced,
            aggregated,
            readout,
            hidden,
            logit,
            prob,
        };

        (prob, acts)
    }

    /// Compute physics-grounded readout features from aggregated state.
    ///
    /// - Grade-0 (scalar): per-field alignment magnitude, cos, sin vs mean field
    /// - Grade-2 (bivector): pairwise cross products (non-potentiality = [F,∇F])
    /// - Sync order: global Kuramoto order parameter r
    /// - Raw input skip: last timestep's 9 normalized SHARP values (direct signal)
    fn compute_readout(&self, state: &[f32], raw_input_last: &[f32]) -> Vec<f32> {
        let n = N_SHARP_FIELDS;
        let d = self.config.d_osc;
        assert_eq!(state.len(), n * d);

        // Compute mean field across all 9 fields
        let mut mean = vec![0.0f32; d];
        for fi in 0..n {
            for di in 0..d {
                mean[di] += state[fi * d + di];
            }
        }
        for di in 0..d {
            mean[di] /= n as f32;
        }
        let mean_norm: f32 = mean.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        for di in 0..d {
            mean[di] /= mean_norm;
        }

        let mut features = Vec::with_capacity(self.readout_dim());

        // Per-field Clifford decomposition against mean field (3 per field):
        //   Geometric product: ψ_i * mean_hat = <ψ_i, mean_hat> + ψ_i ∧ mean_hat
        //   Grade-0: <ψ_i, mean_hat> = cos(θ)  — alignment (scalar part)
        //   Grade-2: ||ψ_i ∧ mean_hat|| = sin(θ) — non-potentiality (bivector part)
        //   Energy:  cos²(θ) = grade-0 squared — alignment energy invariant
        // These are the three independent Clifford grades of the geometric product.
        // On S^{d-1}, sin(θ) = ||wedge|| is NOT redundant with cos(θ) for the network
        // because sqrt is not a linear operation — providing both saves the MLP from
        // learning the nonlinearity. The cos² is the grade-0 Casimir invariant.
        for fi in 0..n {
            let base = fi * d;
            let dot: f32 = (0..d).map(|di| state[base + di] * mean[di]).sum();
            let cos_val = dot.clamp(-1.0, 1.0);
            let wedge_norm = (1.0 - cos_val * cos_val).max(0.0).sqrt();
            features.push(cos_val); // grade-0: scalar alignment
            features.push(wedge_norm); // grade-2: bivector misalignment ||ψ ∧ mean||
            features.push(cos_val * cos_val); // grade-0 energy: Casimir invariant
        }

        // Grade-2: pairwise bivector norms (non-potentiality)
        // ||ψ_i ∧ ψ_j|| = sqrt(1 - <ψ_i, ψ_j>²)
        // This IS the commutator strength |[F, ∇F]| from the sync failure paper.
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f32 = (0..d).map(|di| state[i * d + di] * state[j * d + di]).sum();
                let biv_norm = (1.0 - dot * dot).max(0.0).sqrt();
                features.push(biv_norm);
            }
        }

        // Sync order r: magnitude of mean field (Kuramoto order parameter)
        // r → 1 = all fields aligned (ordered phase, post-J_c)
        // r → 0 = fields disordered (below J_c)
        // r ≈ J_c = critical transition = flare precursor
        let r: f32 = {
            let mut sum = vec![0.0f32; d];
            for fi in 0..n {
                for di in 0..d {
                    sum[di] += state[fi * d + di];
                }
            }
            let mag: f32 = sum.iter().map(|&x| x * x).sum::<f32>().sqrt();
            mag / n as f32
        };
        features.push(r);

        // Per-field phase readout: project state back onto the (u, v) great circle.
        // phase_cos = <ψ_fi, u_fi>, phase_sin = <ψ_fi, v_fi>
        // These features directly encode the input-dependent angular position on
        // each field's great circle — critical for discrimination since the
        // geometric invariants above (inter-field angles) are initially dominated
        // by random encoder geometry and carry little input-dependent signal.
        for fi in 0..n {
            let psi_base = fi * d;
            let uv_base = fi * d;
            let phase_cos: f32 = (0..d)
                .map(|di| state[psi_base + di] * self.encoder_u[uv_base + di])
                .sum();
            let phase_sin: f32 = (0..d)
                .map(|di| state[psi_base + di] * self.encoder_v[uv_base + di])
                .sum();
            features.push(phase_cos);
            features.push(phase_sin);
        }

        // Raw input skip connection: last timestep's 9 normalized SHARP values.
        // This gives the classification head direct access to discriminative signal
        // while the manifold layers learn useful geometric representations.
        // Without this, the head is starved of input-dependent signal at init.
        for fi in 0..n {
            features.push(raw_input_last[fi]);
        }

        features
    }

    // ========================================================================
    // Backward pass
    // ========================================================================

    /// Backward pass: compute gradients of BCE loss w.r.t. all parameters.
    ///
    /// Returns (loss, gradients).
    pub fn backward(&self, acts: &SolarFlareActivations, target: f32) -> (f32, SolarFlareGrads) {
        let t = self.config.seq_len;
        let n = N_SHARP_FIELDS;
        let d = self.config.d_osc;
        let hdim = self.config.hidden_dim;

        // Weighted BCE loss and its gradient w.r.t. logit
        let (loss, d_logit) =
            binary_cross_entropy_weighted(acts.prob, target, self.config.pos_weight);

        // --- [6] Classification head backward ---
        // d_logit → head_w2 backward
        let d_h2_out = vec![d_logit];
        let (d_hidden_pre, mut d_head_w2_weight, d_head_w2_bias) =
            linear_backward(&self.head_w2, &acts.hidden, &d_h2_out);

        // SiLU backward: d/dx[x * σ(x)] = σ(x) + x * σ(x) * (1 - σ(x))
        let mut d_h1_out = vec![0.0f32; hdim];
        for i in 0..hdim {
            let x = acts.hidden[i]; // This is post-silu. We need pre-silu.
                                    // h1_out was pre-silu, hidden is post-silu. We need h1_out.
                                    // Reconstruct: hidden[i] = silu(h1_out[i])
                                    // But we don't store h1_out. Let's compute it from readout.
            let h1_val = {
                let mut v = 0.0f32;
                for j in 0..acts.readout.len() {
                    v += self.head_w1.weight[i * acts.readout.len() + j] * acts.readout[j];
                }
                if let Some(ref bias) = self.head_w1.bias {
                    v += bias[i];
                }
                v
            };
            let sig = sigmoid(h1_val);
            let dsilu = sig + h1_val * sig * (1.0 - sig);
            d_h1_out[i] = d_hidden_pre[i] * dsilu;
        }

        // head_w1 backward
        let (d_readout, mut d_head_w1_weight, d_head_w1_bias) =
            linear_backward(&self.head_w1, &acts.readout, &d_h1_out);

        // --- [5] Readout backward ---
        let last_t = t - 1;
        let raw_input_last: Vec<f32> = (0..n).map(|fi| acts.input[last_t * n + fi]).collect();
        let d_aggregated = self.readout_backward(&acts.aggregated, &d_readout, &raw_input_last);

        // --- [4] Aggregate backward: last timestep only ---
        let mut d_synced = vec![0.0f32; t * n * d];
        for idx in 0..n * d {
            d_synced[last_t * n * d + idx] = d_aggregated[idx];
        }

        // --- [3] Lohe Sync backward (BPTT through Euler steps) ---
        // Forward was: for each timestep, sync_steps Euler integrations:
        //   pre_norm_i = ψ_i + dt * Σ_j K_{ij} * (ψ_j - <ψ_i,ψ_j>ψ_i)
        //   ψ_i' = pre_norm_i / ||pre_norm_i||
        //
        // Backward: reverse through each step, accumulating d_coupling and d_temporal.
        // Re-run forward to get intermediate states at each step.
        let sync_steps = self.config.sync_steps;
        let dt = self.config.sync_dt;
        let mut d_coupling_matrix = vec![0.0f32; n * n];

        // Re-run forward sync to store states at each step (step 0 = input = temporal).
        // states[step] has shape (t * n * d): state BEFORE step `step` is applied.
        // states[0] = acts.temporal (post-normalization, pre-sync input).
        let mut sync_states: Vec<Vec<f32>> = Vec::with_capacity(sync_steps + 1);
        sync_states.push(acts.temporal.clone());

        for _step in 0..sync_steps {
            let prev = sync_states.last().unwrap();
            let mut next = prev.clone();

            for ti in 0..t {
                let mut deltas = vec![0.0f32; n * d];
                for i in 0..n {
                    let psi_i_base = ti * n * d + i * d;
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let k_ij = self.coupling_matrix[i * n + j];
                        if k_ij.abs() < 1e-10 {
                            continue;
                        }
                        let psi_j_base = ti * n * d + j * d;
                        let dot: f32 = (0..d)
                            .map(|di| prev[psi_i_base + di] * prev[psi_j_base + di])
                            .sum();
                        for di in 0..d {
                            deltas[i * d + di] +=
                                k_ij * (prev[psi_j_base + di] - dot * prev[psi_i_base + di]);
                        }
                    }
                }
                for i in 0..n {
                    let base = ti * n * d + i * d;
                    for di in 0..d {
                        next[base + di] = prev[base + di] + dt * deltas[i * d + di];
                    }
                    let norm: f32 = (0..d)
                        .map(|di| next[base + di] * next[base + di])
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-8);
                    for di in 0..d {
                        next[base + di] /= norm;
                    }
                }
            }
            sync_states.push(next);
        }
        // sync_states[sync_steps] should match acts.synced

        // Now backprop through steps in reverse.
        // d_synced holds gradient w.r.t. the final synced state = sync_states[sync_steps].
        // We propagate backward through each step to get d_temporal (= d w.r.t. sync_states[0]).
        let mut d_state = d_synced; // gradient w.r.t. current step's output

        for step in (0..sync_steps).rev() {
            let prev = &sync_states[step];
            let post = &sync_states[step + 1]; // output of this step (= normalized)
            let mut d_prev = vec![0.0f32; t * n * d];

            for ti in 0..t {
                // For each field i: post_i = normalize(prev_i + dt * delta_i)
                // where delta_i = Σ_j K_{ij} * (prev_j - <prev_i, prev_j> * prev_i)
                //
                // Step 1: Backward through normalization.
                // post = pre_norm / ||pre_norm||
                // d(pre_norm) = (I - post post^T) / ||pre_norm|| * d(post)
                // Since post is on S^{d-1}, ||pre_norm|| ≈ 1 + O(dt²).
                // We need pre_norm, which we recompute.
                for i in 0..n {
                    let base = ti * n * d + i * d;
                    let prev_i_base = i * d; // offset within this timestep's delta array

                    // Recompute pre_norm for this field
                    let mut delta_i = vec![0.0f32; d];
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let k_ij = self.coupling_matrix[i * n + j];
                        if k_ij.abs() < 1e-10 {
                            continue;
                        }
                        let psi_j_base = ti * n * d + j * d;
                        let dot: f32 = (0..d)
                            .map(|di| prev[base + di] * prev[psi_j_base + di])
                            .sum();
                        for di in 0..d {
                            delta_i[di] += k_ij * (prev[psi_j_base + di] - dot * prev[base + di]);
                        }
                    }

                    let mut pre_norm = vec![0.0f32; d];
                    for di in 0..d {
                        pre_norm[di] = prev[base + di] + dt * delta_i[di];
                    }
                    let pn_norm: f32 = pre_norm
                        .iter()
                        .map(|&x| x * x)
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-8);

                    // Normalization backward: d_pre_norm = (I - n̂ n̂ᵀ)/||pre_norm|| * d_post
                    // where n̂ = post[base..] (the normalized vector)
                    let mut d_pre_norm = vec![0.0f32; d];
                    let n_dot_dout: f32 =
                        (0..d).map(|di| post[base + di] * d_state[base + di]).sum();
                    for di in 0..d {
                        d_pre_norm[di] =
                            (d_state[base + di] - post[base + di] * n_dot_dout) / pn_norm;
                    }

                    // pre_norm = prev_i + dt * delta_i
                    // d_prev_i += d_pre_norm (direct)
                    // d_delta_i = dt * d_pre_norm
                    for di in 0..d {
                        d_prev[base + di] += d_pre_norm[di];
                    }

                    // Backward through delta_i = Σ_j K_{ij} * (prev_j - <prev_i, prev_j> * prev_i)
                    // Let tang_ij = prev_j - dot_ij * prev_i
                    // delta_i = Σ_j K_{ij} * tang_ij
                    //
                    // d_K_{ij} += dt * <d_pre_norm, tang_ij>
                    // d_prev_j += dt * K_{ij} * d_pre_norm  (through prev_j in tang)
                    // d_prev_i += dt * K_{ij} * (-dot_ij * d_pre_norm + <tang, d_pre_norm> stuff)
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let k_ij = self.coupling_matrix[i * n + j];
                        let psi_j_base = ti * n * d + j * d;
                        let dot_ij: f32 = (0..d)
                            .map(|di| prev[base + di] * prev[psi_j_base + di])
                            .sum();

                        // d_K_{ij} += dt * Σ_di d_pre_norm[di] * tang_ij[di]
                        let mut dk = 0.0f32;
                        for di in 0..d {
                            let tang = prev[psi_j_base + di] - dot_ij * prev[base + di];
                            dk += d_pre_norm[di] * tang;
                        }
                        d_coupling_matrix[i * n + j] += dt * dk;

                        if k_ij.abs() < 1e-10 {
                            continue;
                        }

                        // d(tang_ij)/d(prev_j) = I - prev_i prev_i^T (proj to tangent at prev_i)
                        // Wait — tang_ij = prev_j - <prev_i, prev_j> prev_i
                        // d(tang_ij)/d(prev_j)_kl = δ_kl - prev_i_k * prev_i_l
                        // d(tang_ij)/d(prev_i)_kl = -<prev_j>_l * prev_i_k - dot_ij * δ_kl  [more complex]
                        //
                        // Simplified: d_prev_j += dt * K_{ij} * tangent-projected d_pre_norm
                        //             d_prev_i += dt * K_{ij} * (-dot * d_pre_norm - <d_pre_norm, prev_j - dot*prev_i> * prev_i stuff)
                        //
                        // Full derivative of tang_ij w.r.t. prev_i and prev_j:
                        let d_delta_pre = dt * k_ij; // scalar factor

                        // Contribution to d_prev_j: d_delta_pre * (d_pre_norm - <d_pre_norm, prev_i> prev_i)
                        let dpn_dot_pi: f32 =
                            (0..d).map(|di| d_pre_norm[di] * prev[base + di]).sum();
                        for di in 0..d {
                            d_prev[psi_j_base + di] +=
                                d_delta_pre * (d_pre_norm[di] - dpn_dot_pi * prev[base + di]);
                        }

                        // Contribution to d_prev_i through tang_ij:
                        // d(tang_ij)/d(prev_i) applied to d_pre_norm:
                        //   = -<d_pre_norm, prev_j> prev_i  (from d(<prev_i, prev_j>)/d(prev_i) = prev_j)
                        //   + (-dot_ij) * d_pre_norm          (from -dot * d(prev_i)/d(prev_i) = -dot * I)
                        // Wait, more carefully:
                        //   tang = prev_j - dot * prev_i, where dot = <prev_i, prev_j>
                        //   d(tang)/d(prev_i) · v = -<v, prev_j> prev_i - dot * v + <v, prev_i> dot * prev_i
                        //   Hmm, let's use: d(dot)/d(prev_i) = prev_j, so
                        //   d(tang)/d(prev_i) · v = -(d(dot)/d(prev_i) · v) * prev_i - dot * v
                        //                         = -<v, prev_j> prev_i - dot * v
                        //   But prev_i is on S^{d-1}, and there's also the d(prev_i)/d(prev_i) = I term.
                        //   Full: d(-dot*prev_i)/d(prev_i) · v = -<prev_j, v> prev_i - dot * v
                        let dpn_dot_pj: f32 = (0..d)
                            .map(|di| d_pre_norm[di] * prev[psi_j_base + di])
                            .sum();
                        for di in 0..d {
                            d_prev[base + di] += d_delta_pre
                                * (-dpn_dot_pj * prev[base + di] - dot_ij * d_pre_norm[di]);
                        }
                    }
                }
            }
            // Per-field gradient clipping within BPTT to prevent exploding gradients.
            // Without this, the 3-step Lohe integration can amplify gradients
            // exponentially for strongly-coupled fields.
            let max_field_grad_norm = 10.0f32;
            for ti in 0..t {
                for fi in 0..n {
                    let base = ti * n * d + fi * d;
                    let gnorm: f32 = (0..d)
                        .map(|di| d_prev[base + di] * d_prev[base + di])
                        .sum::<f32>()
                        .sqrt();
                    if gnorm > max_field_grad_norm {
                        let scale = max_field_grad_norm / gnorm;
                        for di in 0..d {
                            d_prev[base + di] *= scale;
                        }
                    }
                    // Also sanitize any NaN
                    for di in 0..d {
                        if !d_prev[base + di].is_finite() {
                            d_prev[base + di] = 0.0;
                        }
                    }
                }
            }

            d_state = d_prev;
        }

        // Sanitize coupling gradients (NaN from degenerate inputs)
        for v in d_coupling_matrix.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // d_state is now d w.r.t. sync_states[0] = acts.temporal (post-normalization)
        let d_temporal = d_state;

        // --- [2b] Backward through post-Helmholtz normalization ---
        // Forward: temporal_raw = avg(helmholtz_heads), temporal = normalize(temporal_raw)
        // d_temporal currently holds gradients w.r.t. the normalized temporal.
        // We need d_temporal_raw via the normalization Jacobian:
        //   J_{ij} = (δ_{ij} - n_i * n_j) / ||raw||
        let mut d_temporal_raw = vec![0.0f32; t * n * d];
        for ti in 0..t {
            for fi in 0..n {
                let base = ti * n * d + fi * d;
                // Recompute raw norm (before normalization)
                // temporal_raw = acts.temporal before normalization — but we normalized in-place.
                // The normalized vectors are in acts.temporal (which became input to sync).
                // We stored the pre-sync state in acts.temporal. Actually, acts.temporal
                // IS the post-average, pre-sync value. But we now normalize before sync,
                // so acts.temporal stores the post-average values (pre-normalization).
                // Wait — acts.temporal was set in forward AFTER the averaging loop.
                // Then we normalize temporal[] in-place before sync.
                // So acts.temporal has the PRE-normalization values (it was saved before
                // the normalization code runs? No — let me check.)
                //
                // In forward: temporal[idx] = sum/N_HEADS, then we normalize temporal in-place,
                // then we clone temporal into synced, then acts.temporal = temporal (normalized).
                // So acts.temporal is POST-normalization. We need the pre-norm values.
                // We can reconstruct: pre_norm = normalized * norm. But we don't have norm.
                //
                // Alternative: the helmholtz_out heads are stored in acts. Recompute the average.
                let mut raw = vec![0.0f32; d];
                for di in 0..d {
                    let mut sum = 0.0f32;
                    for head in 0..N_HEADS {
                        sum += acts.helmholtz_out[head][base + di];
                    }
                    raw[di] = sum / N_HEADS as f32;
                }
                let raw_norm: f32 = raw.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
                let mut n_hat = vec![0.0f32; d];
                for di in 0..d {
                    n_hat[di] = raw[di] / raw_norm;
                }

                // J_{ij} = (δ_{ij} - n_hat_i * n_hat_j) / raw_norm
                for di in 0..d {
                    let mut grad = 0.0f32;
                    for dj in 0..d {
                        let kronecker = if di == dj { 1.0 } else { 0.0 };
                        let jac = (kronecker - n_hat[di] * n_hat[dj]) / raw_norm;
                        grad += jac * d_temporal[base + dj];
                    }
                    d_temporal_raw[base + di] = grad;
                }
            }
        }

        // --- [2] Helmholtz Temporal backward ---
        // d_temporal_raw → average across heads → per-head gated_cumsum_backward
        let mut d_helmholtz_gamma = vec![0.0f32; N_HEADS * d];
        let mut d_encoded = vec![0.0f32; t * n * d];

        for head in 0..N_HEADS {
            let gamma_start = head * d;
            let gamma_slice = &self.helmholtz_gamma[gamma_start..gamma_start + d];

            for fi in 0..n {
                // Extract grad for this field: (T, d)
                let mut grad_field = vec![0.0f32; t * d];
                for ti in 0..t {
                    for di in 0..d {
                        // d_temporal_raw is avg of heads, so grad to each head = d_temporal_raw / N_HEADS
                        grad_field[ti * d + di] =
                            d_temporal_raw[ti * n * d + fi * d + di] / N_HEADS as f32;
                    }
                }

                // Extract input for this field: (T, d)
                let mut inp_field = vec![0.0f32; t * d];
                for ti in 0..t {
                    for di in 0..d {
                        inp_field[ti * d + di] = acts.encoded[ti * n * d + fi * d + di];
                    }
                }

                // gated_cumsum_backward
                let (d_inp, d_gamma) =
                    gated_cumsum_backward(&grad_field, &inp_field, gamma_slice, t, d);

                // Accumulate
                for ti in 0..t {
                    for di in 0..d {
                        d_encoded[ti * n * d + fi * d + di] += d_inp[ti * d + di];
                    }
                }
                for di in 0..d {
                    d_helmholtz_gamma[gamma_start + di] += d_gamma[di];
                }
            }
        }

        // --- [1] Geodesic Phase Encoder backward ---
        // ψ(x) = cos(πx)·u + sin(πx)·v
        // dψ/du = cos(πx)·I → d_u += cos(πx) · d_encoded
        // dψ/dv = sin(πx)·I → d_v += sin(πx) · d_encoded
        // No normalization Jacobian needed — output is on S^{d-1} by construction.
        let mut d_encoder_u = vec![0.0f32; n * d];
        let mut d_encoder_v = vec![0.0f32; n * d];

        for ti in 0..t {
            for fi in 0..n {
                let x = acts.input[ti * n + fi];
                let theta = PI * x;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();
                let enc_base = ti * n * d + fi * d;
                let uv_base = fi * d;
                for di in 0..d {
                    d_encoder_u[uv_base + di] += cos_theta * d_encoded[enc_base + di];
                    d_encoder_v[uv_base + di] += sin_theta * d_encoded[enc_base + di];
                }
            }
        }

        let grads = SolarFlareGrads {
            d_encoder_u,
            d_encoder_v,
            d_helmholtz_gamma,
            d_coupling_matrix,
            d_head_w1_weight: d_head_w1_weight,
            d_head_w1_bias: d_head_w1_bias,
            d_head_w2_weight: d_head_w2_weight,
            d_head_w2_bias: d_head_w2_bias,
        };

        (loss, grads)
    }

    /// Backward through the readout function (aggregated → readout features).
    ///
    /// Analytical gradient through the Clifford grade decomposition.
    /// Per-field features: grade-0 (cos), grade-2 (||wedge||), energy (cos²).
    /// Pairwise features: ||ψ_i ∧ ψ_j|| = sqrt(1 - dot²).
    /// Global: sync order r (Kuramoto order parameter).
    ///
    /// Geometry notes:
    /// - Grade-0 gradient through mean_hat uses full Jacobian of unit-vector normalization
    ///   d(v/||v||)/dv = (I - v̂ v̂ᵀ)/||v||. This tangent-space projector removes only
    ///   the radial component (normalization artifact) while preserving the tangential
    ///   signal — it does NOT normalize away content-dependent alignment information.
    /// - Grade-2 gradient is exact on S^{d-1}: d(||ψ_i ∧ ψ_j||)/d(dot) = -dot/||∧||.
    ///   No additional normalization since states are already unit vectors.
    /// - Sync order gradient points each ψ_fi toward the mean field — the natural
    ///   tangent-space gradient of the order parameter on the product manifold.
    fn readout_backward(
        &self,
        aggregated: &[f32],
        d_readout: &[f32],
        _raw_input_last: &[f32],
    ) -> Vec<f32> {
        let n = N_SHARP_FIELDS;
        let d = self.config.d_osc;

        let mut d_agg = vec![0.0f32; n * d];

        // Recompute mean field (unnormalized and normalized)
        let mut mean = vec![0.0f32; d];
        for fi in 0..n {
            for di in 0..d {
                mean[di] += aggregated[fi * d + di];
            }
        }
        for di in 0..d {
            mean[di] /= n as f32;
        }
        let mean_norm: f32 = mean.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut mean_hat = vec![0.0f32; d];
        for di in 0..d {
            mean_hat[di] = mean[di] / mean_norm;
        }

        // --- Grade-0 features (3 per field): cos, sin, cos² ---
        // cos_fi = <ψ_fi, mean_hat> where mean_hat = (Σ ψ_j / n) / ||Σ ψ_j / n||
        //
        // Full gradient: d(cos_fi)/d(ψ_k) has two paths:
        //   (a) Direct: if k == fi, d(<ψ_fi, mean_hat>)/d(ψ_fi) includes mean_hat
        //   (b) Through mean: ψ_k contributes to mean_hat for all k
        //
        // d(mean_hat)/d(ψ_k) = (1/n) * (I - mean_hat mean_hatᵀ) / mean_norm
        // d(cos_fi)/d(ψ_k) = δ_{k,fi} * mean_hat + ψ_fi ᵀ * d(mean_hat)/d(ψ_k)
        //
        // The (I - mean_hat mean_hatᵀ)/||mean|| term is the tangent-space projector
        // on S^{d-1} at mean_hat — this is geometrically correct, not over-normalizing,
        // because it projects out only the radial component (which is the normalization
        // artifact) while preserving the tangential signal.

        // Accumulate d_agg from grade-0 features for all fields
        // First compute the total upstream gradient w.r.t. mean_hat from all fields
        let inv_n_norm = 1.0 / (n as f32 * mean_norm);
        let mut d_mean_hat_total = vec![0.0f32; d]; // total gradient through mean_hat

        for fi in 0..n {
            let base = fi * d;
            let dot: f32 = (0..d).map(|di| aggregated[base + di] * mean_hat[di]).sum();
            let cos_val = dot.clamp(-1.0, 1.0);
            let wedge_norm = (1.0 - cos_val * cos_val).max(0.0).sqrt().max(1e-8);

            let feat_idx = fi * 3;
            let d_cos = d_readout[feat_idx]; // d_loss/d(grade-0 scalar)
            let d_wedge = d_readout[feat_idx + 1]; // d_loss/d(grade-2 bivector norm)
            let d_energy = d_readout[feat_idx + 2]; // d_loss/d(cos² energy)

            // Chain rule: all three features are functions of cos_val
            // d(wedge_norm)/d(cos) = -cos/wedge_norm
            // d(cos²)/d(cos) = 2*cos
            let d_cos_total = d_cos + d_wedge * (-cos_val / wedge_norm) + d_energy * 2.0 * cos_val;

            // (a) Direct path: d(cos_fi)/d(ψ_fi) = mean_hat
            for di in 0..d {
                d_agg[base + di] += d_cos_total * mean_hat[di];
            }

            // (b) Through mean_hat: d(cos_fi)/d(mean_hat) = ψ_fi
            for di in 0..d {
                d_mean_hat_total[di] += d_cos_total * aggregated[base + di];
            }
        }

        // Now propagate d_mean_hat_total through the normalization:
        // d(mean_hat)/d(mean) = (I - mean_hat mean_hatᵀ) / ||mean||
        // d(mean)/d(ψ_k) = 1/n for all k
        // So d_agg[k] += (1/n) * (d_mean_hat_total - mean_hat * <mean_hat, d_mean_hat_total>) / ||mean||
        let mh_dot: f32 = (0..d).map(|di| mean_hat[di] * d_mean_hat_total[di]).sum();
        for fi in 0..n {
            let base = fi * d;
            for di in 0..d {
                // Tangent-space projection preserves geometric signal
                d_agg[base + di] += inv_n_norm * (d_mean_hat_total[di] - mean_hat[di] * mh_dot);
            }
        }

        // --- Grade-2 features: ||ψ_i ∧ ψ_j|| = sqrt(1 - dot_ij²) ---
        // These are exact on S^{d-1}: no normalization needed since states are unit vectors.
        // d(||∧||)/d(dot) = -dot / ||∧||, d(dot)/d(ψ_i) = ψ_j, d(dot)/d(ψ_j) = ψ_i
        let mut feat_idx = 3 * n;
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f32 = (0..d)
                    .map(|di| aggregated[i * d + di] * aggregated[j * d + di])
                    .sum();
                let biv_norm = (1.0 - dot * dot).max(0.0).sqrt().max(1e-8);

                let d_biv = d_readout[feat_idx];
                let d_dot = d_biv * (-dot / biv_norm);

                for di in 0..d {
                    d_agg[i * d + di] += d_dot * aggregated[j * d + di];
                    d_agg[j * d + di] += d_dot * aggregated[i * d + di];
                }
                feat_idx += 1;
            }
        }

        // --- Sync order r = ||Σ ψ_fi|| / n ---
        // r is the Kuramoto order parameter — its gradient naturally points each ψ_fi
        // toward the mean field direction, which is geometrically correct (it's the
        // tangent-space gradient of the order parameter on the product manifold).
        let d_r = d_readout[feat_idx];
        let mut sum = vec![0.0f32; d];
        for fi in 0..n {
            for di in 0..d {
                sum[di] += aggregated[fi * d + di];
            }
        }
        let sum_norm = sum.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        for fi in 0..n {
            for di in 0..d {
                d_agg[fi * d + di] += d_r * sum[di] / (n as f32 * sum_norm);
            }
        }
        feat_idx += 1;

        // --- Per-field phase features: phase_cos = <ψ, u>, phase_sin = <ψ, v> ---
        // d(phase_cos)/d(ψ[k]) = u[k], d(phase_sin)/d(ψ[k]) = v[k]
        // These are the direct gradients — no normalization (u, v are fixed unit vectors).
        for fi in 0..n {
            let psi_base = fi * d;
            let uv_base = fi * d;
            let d_pc = d_readout[feat_idx];
            let d_ps = d_readout[feat_idx + 1];
            for di in 0..d {
                d_agg[psi_base + di] +=
                    d_pc * self.encoder_u[uv_base + di] + d_ps * self.encoder_v[uv_base + di];
            }
            feat_idx += 2;
        }

        d_agg
    }

    // ========================================================================
    // Parameter update (ManifoldSGD)
    // ========================================================================

    /// Apply one optimization step using ManifoldSGD.
    ///
    /// Encoder weights use geodesic updates on S^{d-1} (per row = per field).
    /// All other params use standard SGD.
    pub fn sgd_step(&mut self, grads: &SolarFlareGrads) {
        let d = self.config.d_osc;
        let n = N_SHARP_FIELDS;

        // Global gradient norm clipping (max norm = 5.0)
        // Raised from 1.0: with ~1490 params, max_norm=1.0 squashes effective LR
        // to near-zero, preventing learning. 5.0 allows meaningful updates while
        // still preventing explosion.
        let grad_norm_sq: f32 = grads
            .d_encoder_u
            .iter()
            .chain(grads.d_encoder_v.iter())
            .chain(grads.d_helmholtz_gamma.iter())
            .chain(grads.d_coupling_matrix.iter())
            .chain(grads.d_head_w1_weight.iter())
            .chain(grads.d_head_w1_bias.iter())
            .chain(grads.d_head_w2_weight.iter())
            .chain(grads.d_head_w2_bias.iter())
            .map(|g| g * g)
            .sum();
        let grad_norm = grad_norm_sq.sqrt();
        let max_norm = 5.0f32;
        let clip_scale = if grad_norm > max_norm {
            max_norm / grad_norm
        } else {
            1.0
        };
        let lr_m = self.config.lr_manifold * clip_scale;
        let lr_s = self.config.lr_scalar * clip_scale;

        // Encoder u and v: geodesic SGD on S^{d-1} per field row, then Gram-Schmidt.
        // Both u and v live on S^{d-1} and must remain orthonormal (u ⊥ v).
        for fi in 0..n {
            let base = fi * d;

            // Update u: project gradient to tangent plane of S^{d-1} at u, step, normalize
            {
                let grad_u = &grads.d_encoder_u[base..base + d];
                let dot: f32 = (0..d)
                    .map(|di| grad_u[di] * self.encoder_u[base + di])
                    .sum();
                for di in 0..d {
                    let tan = grad_u[di] - dot * self.encoder_u[base + di];
                    self.encoder_u[base + di] -= lr_m * tan;
                }
                let norm: f32 = (0..d)
                    .map(|di| self.encoder_u[base + di].powi(2))
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                for di in 0..d {
                    self.encoder_u[base + di] /= norm;
                }
            }

            // Update v: project gradient to tangent plane of S^{d-1} at v, step, normalize
            {
                let grad_v = &grads.d_encoder_v[base..base + d];
                let dot: f32 = (0..d)
                    .map(|di| grad_v[di] * self.encoder_v[base + di])
                    .sum();
                for di in 0..d {
                    let tan = grad_v[di] - dot * self.encoder_v[base + di];
                    self.encoder_v[base + di] -= lr_m * tan;
                }
                let norm: f32 = (0..d)
                    .map(|di| self.encoder_v[base + di].powi(2))
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                for di in 0..d {
                    self.encoder_v[base + di] /= norm;
                }
            }

            // Gram-Schmidt: re-orthogonalize v to u (v = v - <v,u>u, normalize)
            // This preserves the great circle structure required for geodesic phase encoding.
            // Two passes for f32 numerical stability.
            for _gs in 0..2 {
                let dot_uv: f32 = (0..d)
                    .map(|di| self.encoder_v[base + di] * self.encoder_u[base + di])
                    .sum();
                for di in 0..d {
                    self.encoder_v[base + di] -= dot_uv * self.encoder_u[base + di];
                }
            }
            let norm_v: f32 = (0..d)
                .map(|di| self.encoder_v[base + di].powi(2))
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for di in 0..d {
                self.encoder_v[base + di] /= norm_v;
            }
        }

        // Helmholtz gamma: SGD with clamping to (0, 1)
        for i in 0..N_HEADS * d {
            self.helmholtz_gamma[i] -= lr_s * grads.d_helmholtz_gamma[i];
            self.helmholtz_gamma[i] = self.helmholtz_gamma[i].clamp(0.01, 0.999);
        }

        // Coupling matrix: SGD + enforce antisymmetry
        for i in 0..n * n {
            self.coupling_matrix[i] -= lr_s * grads.d_coupling_matrix[i];
        }
        // Re-enforce antisymmetry: K[i,j] = (K[i,j] - K[j,i]) / 2
        for i in 0..n {
            self.coupling_matrix[i * n + i] = 0.0;
            for j in (i + 1)..n {
                let avg = (self.coupling_matrix[i * n + j] - self.coupling_matrix[j * n + i]) / 2.0;
                self.coupling_matrix[i * n + j] = avg;
                self.coupling_matrix[j * n + i] = -avg;
            }
        }

        // Classification head: standard SGD
        for i in 0..self.head_w1.weight.len() {
            self.head_w1.weight[i] -= lr_s * grads.d_head_w1_weight[i];
        }
        if let Some(ref mut bias) = self.head_w1.bias {
            for i in 0..bias.len() {
                bias[i] -= lr_s * grads.d_head_w1_bias[i];
            }
        }
        for i in 0..self.head_w2.weight.len() {
            self.head_w2.weight[i] -= lr_s * grads.d_head_w2_weight[i];
        }
        if let Some(ref mut bias) = self.head_w2.bias {
            for i in 0..bias.len() {
                bias[i] -= lr_s * grads.d_head_w2_bias[i];
            }
        }
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Save model checkpoint to JSON.
    pub fn save_checkpoint(&self, path: &std::path::Path) -> std::io::Result<()> {
        let checkpoint = SolarFlareCheckpoint {
            config: self.config.clone(),
            encoder_u: self.encoder_u.clone(),
            encoder_v: self.encoder_v.clone(),
            helmholtz_gamma: self.helmholtz_gamma.clone(),
            coupling_matrix: self.coupling_matrix.clone(),
            head_w1_weight: self.head_w1.weight.clone(),
            head_w1_bias: self.head_w1.bias.clone().unwrap_or_default(),
            head_w2_weight: self.head_w2.weight.clone(),
            head_w2_bias: self.head_w2.bias.clone().unwrap_or_default(),
        };
        let json = serde_json::to_string_pretty(&checkpoint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load model checkpoint from JSON.
    pub fn load_checkpoint(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let cp: SolarFlareCheckpoint = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let readout_dim = {
            let n = N_SHARP_FIELDS;
            3 * n + n * (n - 1) / 2 + 1 + 2 * n + n
        };

        let head_w1 = Linear {
            weight: cp.head_w1_weight,
            bias: Some(cp.head_w1_bias),
            in_features: readout_dim,
            out_features: cp.config.hidden_dim,
        };
        let head_w2 = Linear {
            weight: cp.head_w2_weight,
            bias: Some(cp.head_w2_bias),
            in_features: cp.config.hidden_dim,
            out_features: 1,
        };

        Ok(SolarFlareModel {
            config: cp.config,
            encoder_u: cp.encoder_u,
            encoder_v: cp.encoder_v,
            helmholtz_gamma: cp.helmholtz_gamma,
            coupling_matrix: cp.coupling_matrix,
            head_w1,
            head_w2,
        })
    }

    /// Predict flare probability from a SHARP time series.
    ///
    /// Convenience wrapper around forward() that returns just the probability.
    pub fn predict(&self, sharp_series: &[f32]) -> f32 {
        let (prob, _) = self.forward(sharp_series);
        prob
    }

    /// Compute Lohe sync order parameter r from current field states.
    ///
    /// This is a physics diagnostic: r → J_c = KT transition = flare precursor.
    pub fn sync_order(&self, state: &[f32]) -> f32 {
        let n = N_SHARP_FIELDS;
        let d = self.config.d_osc;
        let mut sum = vec![0.0f32; d];
        for fi in 0..n {
            for di in 0..d {
                sum[di] += state[fi * d + di];
            }
        }
        let mag: f32 = sum.iter().map(|&x| x * x).sum::<f32>().sqrt();
        mag / n as f32
    }
}

// ============================================================================
// Checkpoint serialization
// ============================================================================

#[derive(Serialize, Deserialize)]
struct SolarFlareCheckpoint {
    config: SolarFlareConfig,
    encoder_u: Vec<f32>,
    encoder_v: Vec<f32>,
    helmholtz_gamma: Vec<f32>,
    coupling_matrix: Vec<f32>,
    head_w1_weight: Vec<f32>,
    head_w1_bias: Vec<f32>,
    head_w2_weight: Vec<f32>,
    head_w2_bias: Vec<f32>,
}

// ============================================================================
// Utility functions
// ============================================================================

/// Create a Linear layer with Xavier initialization.
fn make_linear(in_features: usize, out_features: usize) -> Linear {
    // Kaiming init: scale = sqrt(2/fan_in) for ReLU-family activations (SiLU).
    // This ensures hidden layer activations have unit variance at init,
    // keeping the classification head in the non-linear regime of SiLU
    // where gradients flow meaningfully.
    let scale = (2.0 / in_features as f32).sqrt();
    let mut weight = vec![0.0f32; out_features * in_features];
    for i in 0..weight.len() {
        // Wrap to 32 bits to get uniform distribution in [0, 2^32)
        let hash = (((i as u32).wrapping_mul(2654435761).wrapping_add(12345)) >> 16) as f32
            / 65536.0
            - 0.5;
        weight[i] = hash * scale;
    }
    let bias = vec![0.0f32; out_features];
    Linear {
        weight,
        bias: Some(bias),
        in_features,
        out_features,
    }
}

/// Sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Weighted binary cross-entropy loss: -[w_pos * y * ln(p) + (1-y) * ln(1-p)]
///
/// pos_weight > 1 increases gradient signal for positive samples,
/// critical for imbalanced data (e.g. 2% flare rate).
/// Returns (loss, d_loss/d_logit).
fn binary_cross_entropy_weighted(pred: f32, target: f32, pos_weight: f32) -> (f32, f32) {
    let p = pred.clamp(1e-7, 1.0 - 1e-7);
    let loss = -(pos_weight * target * p.ln() + (1.0 - target) * (1.0 - p).ln());
    // d_loss/d_logit = p - target, but with pos_weight on positive term:
    // d/d_logit[-w*y*log(σ(z)) - (1-y)*log(1-σ(z))]
    //   = -w*y*(1-σ(z)) + (1-y)*σ(z)
    //   = σ(z) - y*(w*σ(z) + w*(1-σ(z)))  ... simplify:
    //   = (1-y)*p - w*y*(1-p)
    // Wait, let me be more careful:
    //   = -w*y*(1-p) + (1-y)*p
    let d_logit = (1.0 - target) * pred - pos_weight * target * (1.0 - pred);
    (loss, d_logit)
}

/// Linear layer backward pass.
///
/// Given d_output, computes (d_input, d_weight, d_bias).
fn linear_backward(
    layer: &Linear,
    input: &[f32],
    d_output: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_f = layer.in_features;
    let out_f = layer.out_features;

    // d_input = W^T @ d_output
    let mut d_input = vec![0.0f32; in_f];
    for j in 0..in_f {
        for i in 0..out_f {
            d_input[j] += layer.weight[i * in_f + j] * d_output[i];
        }
    }

    // d_weight = d_output ⊗ input (outer product)
    let mut d_weight = vec![0.0f32; out_f * in_f];
    for i in 0..out_f {
        for j in 0..in_f {
            d_weight[i * in_f + j] = d_output[i] * input[j];
        }
    }

    // d_bias = d_output
    let d_bias = d_output.to_vec();

    (d_input, d_weight, d_bias)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_flare_forward_shape() {
        let config = SolarFlareConfig::default();
        let model = SolarFlareModel::new(config.clone());
        let t = config.seq_len;
        let input = vec![0.5f32; t * N_SHARP_FIELDS];

        let (prob, acts) = model.forward(&input);

        assert!(prob >= 0.0 && prob <= 1.0, "prob = {}", prob);
        assert_eq!(acts.encoded.len(), t * N_SHARP_FIELDS * config.d_osc);
        assert_eq!(acts.readout.len(), model.readout_dim());
        assert!(acts.logit.is_finite(), "logit = {}", acts.logit);
    }

    #[test]
    fn test_solar_flare_forward_finite() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        let t = model.config.seq_len;

        // Simulate realistic SHARP-like values (min-max normalized to [0, 1])
        let mut input = vec![0.0f32; t * N_SHARP_FIELDS];
        for ti in 0..t {
            for fi in 0..N_SHARP_FIELDS {
                let val = ((ti * N_SHARP_FIELDS + fi) as f32 * 0.1).sin().abs();
                input[ti * N_SHARP_FIELDS + fi] = val;
            }
        }

        let (prob, acts) = model.forward(&input);

        assert!(prob.is_finite(), "prob not finite: {}", prob);
        assert!(prob >= 0.0 && prob <= 1.0, "prob out of range: {}", prob);
        for (i, &v) in acts.encoded.iter().enumerate() {
            assert!(v.is_finite(), "encoded[{}] not finite: {}", i, v);
        }
        for (i, &v) in acts.readout.iter().enumerate() {
            assert!(v.is_finite(), "readout[{}] not finite: {}", i, v);
        }
    }

    #[test]
    fn test_solar_flare_backward_finite() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        let t = model.config.seq_len;
        let input = vec![0.5f32; t * N_SHARP_FIELDS];

        let (prob, acts) = model.forward(&input);
        let (loss, grads) = model.backward(&acts, 1.0);

        assert!(loss.is_finite(), "loss = {}", loss);
        assert!(loss >= 0.0, "loss should be non-negative: {}", loss);

        for (i, &v) in grads.d_encoder_u.iter().enumerate() {
            assert!(v.is_finite(), "d_encoder_u[{}] = {}", i, v);
        }
        for (i, &v) in grads.d_helmholtz_gamma.iter().enumerate() {
            assert!(v.is_finite(), "d_helmholtz_gamma[{}] = {}", i, v);
        }
        for (i, &v) in grads.d_head_w1_weight.iter().enumerate() {
            assert!(v.is_finite(), "d_head_w1_weight[{}] = {}", i, v);
        }
    }

    #[test]
    fn test_solar_flare_manifold_constraint() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        let t = model.config.seq_len;
        let d = model.config.d_osc;
        let input = vec![0.3f32; t * N_SHARP_FIELDS];

        let (_, acts) = model.forward(&input);

        // Check that encoded states are on S^{d-1}
        for ti in 0..t {
            for fi in 0..N_SHARP_FIELDS {
                let base = ti * N_SHARP_FIELDS * d + fi * d;
                let norm: f32 = (0..d)
                    .map(|di| acts.encoded[base + di] * acts.encoded[base + di])
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "Encoded field {} at t={} not on sphere: norm = {}",
                    fi,
                    ti,
                    norm
                );
            }
        }

        // Check that synced states are on S^{d-1} (after Lohe sync)
        for ti in 0..t {
            for fi in 0..N_SHARP_FIELDS {
                let base = ti * N_SHARP_FIELDS * d + fi * d;
                let norm: f32 = (0..d)
                    .map(|di| acts.synced[base + di] * acts.synced[base + di])
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-4,
                    "Synced field {} at t={} not on sphere: norm = {}",
                    fi,
                    ti,
                    norm
                );
            }
        }
    }

    #[test]
    fn test_solar_flare_param_count() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        let count = model.param_count();
        // Should be ~1.5K: 2*9*16=288 enc + 4*16=64 helm + 81 coup + 64*16+16=1040 h1 + 17 h2 = ~1490
        assert!(count > 1000 && count < 3000, "param_count = {}", count);
        println!("SolarFlareModel param count: {}", count);
    }

    #[test]
    fn test_solar_flare_sgd_step() {
        let mut config = SolarFlareConfig::default();
        config.lr_scalar = 0.1; // Larger LR for visible progress in one step
        config.lr_manifold = 0.1;
        let mut model = SolarFlareModel::new(config);
        let t = model.config.seq_len;

        // Use varied input to avoid degenerate initialization
        let mut input = vec![0.0f32; t * N_SHARP_FIELDS];
        for i in 0..input.len() {
            input[i] = (i as f32 * 0.37).sin().abs() * 0.8 + 0.1;
        }

        // Train toward target=0 (push prob down)
        let (prob0, acts) = model.forward(&input);
        let (loss0, grads) = model.backward(&acts, 0.0);

        // Run multiple SGD steps to see convergence
        for _ in 0..5 {
            let (_, acts_i) = model.forward(&input);
            let (_, grads_i) = model.backward(&acts_i, 0.0);
            model.sgd_step(&grads_i);
        }

        let (prob1, acts1) = model.forward(&input);
        let (loss1, _) = model.backward(&acts1, 0.0);

        println!(
            "SGD 5 steps toward target=0: loss {:.6} → {:.6}, prob {:.6} → {:.6}",
            loss0, loss1, prob0, prob1
        );
        // After 5 steps, loss should have decreased
        assert!(
            loss1 < loss0 + 0.001,
            "Loss did not decrease after 5 steps: {:.6} → {:.6}",
            loss0,
            loss1
        );
    }

    #[test]
    fn test_solar_flare_readout_dim() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        // 3*9 = 27 grade-0 + 9*8/2 = 36 grade-2 + 1 sync + 2*9 = 18 phase + 9 raw skip = 91
        assert_eq!(model.readout_dim(), 91);
    }

    #[test]
    fn test_solar_flare_checkpoint_roundtrip() {
        let model = SolarFlareModel::new(SolarFlareConfig::default());
        let t = model.config.seq_len;
        let input = vec![0.5f32; t * N_SHARP_FIELDS];
        let (prob_before, _) = model.forward(&input);

        let path = &std::env::temp_dir().join("solar_flare_test_checkpoint.json");
        model.save_checkpoint(path).unwrap();
        let loaded = SolarFlareModel::load_checkpoint(path).unwrap();
        let (prob_after, _) = loaded.forward(&input);

        assert!(
            (prob_before - prob_after).abs() < 1e-6,
            "Checkpoint roundtrip mismatch: {} vs {}",
            prob_before,
            prob_after
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_binary_cross_entropy() {
        // Test BCE loss values
        let (loss, grad) = binary_cross_entropy_weighted(0.9, 1.0, 1.0);
        assert!(
            loss < 0.2,
            "loss should be small for correct prediction: {}",
            loss
        );
        assert!(grad < 0.0, "grad should push prob up toward 1: {}", grad);

        let (loss2, grad2) = binary_cross_entropy_weighted(0.1, 0.0, 1.0);
        assert!(
            loss2 < 0.2,
            "loss should be small for correct prediction: {}",
            loss2
        );
        assert!(
            grad2 > 0.0,
            "grad should push prob down toward 0: {}",
            grad2
        );

        let (loss3, _) = binary_cross_entropy_weighted(0.5, 1.0, 1.0);
        assert!(loss3 > loss, "50/50 should have higher loss than 90/10");
    }

    #[test]
    fn test_sync_order() {
        let config = SolarFlareConfig::default();
        let model = SolarFlareModel::new(config.clone());
        let d = config.d_osc;

        // All fields aligned → r should be close to 1
        let mut aligned = vec![0.0f32; N_SHARP_FIELDS * d];
        for fi in 0..N_SHARP_FIELDS {
            aligned[fi * d] = 1.0; // All point along first axis
        }
        let r_aligned = model.sync_order(&aligned);
        assert!(r_aligned > 0.9, "Aligned r = {}", r_aligned);

        // Random-ish fields → r should be lower
        let mut scattered = vec![0.0f32; N_SHARP_FIELDS * d];
        for fi in 0..N_SHARP_FIELDS {
            let axis = fi % d;
            scattered[fi * d + axis] = 1.0;
        }
        let r_scattered = model.sync_order(&scattered);
        assert!(
            r_scattered < r_aligned,
            "Scattered r = {} should be < aligned r = {}",
            r_scattered,
            r_aligned
        );
    }

    #[test]
    fn test_solar_flare_nan_stress() {
        // Train 50 steps on varied data to catch NaN from BPTT / readout gradients.
        let mut config = SolarFlareConfig::default();
        config.lr_manifold = 0.01;
        config.lr_scalar = 0.01;
        let mut model = SolarFlareModel::new(config);
        let t = model.config.seq_len;

        for step in 0..50 {
            // Different input each step to stress different code paths
            let mut input = vec![0.0f32; t * N_SHARP_FIELDS];
            for i in 0..input.len() {
                let hash = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(step * 7919);
                input[i] = (hash % 10000) as f32 / 10000.0;
            }
            let target = if step % 3 == 0 { 1.0 } else { 0.0 };

            let (prob, acts) = model.forward(&input);
            assert!(prob.is_finite(), "NaN prob at step {step}");

            let (loss, grads) = model.backward(&acts, target);
            assert!(
                loss.is_finite(),
                "NaN loss at step {step}: prob={prob}, target={target}"
            );

            // Check all gradients are finite
            for &v in grads.d_encoder_u.iter() {
                assert!(v.is_finite(), "NaN in d_encoder_u at step {step}");
            }
            for &v in grads.d_helmholtz_gamma.iter() {
                assert!(v.is_finite(), "NaN in d_helmholtz_gamma at step {step}");
            }
            for &v in grads.d_coupling_matrix.iter() {
                assert!(v.is_finite(), "NaN in d_coupling_matrix at step {step}");
            }
            for &v in grads.d_head_w1_weight.iter() {
                assert!(v.is_finite(), "NaN in d_head_w1_weight at step {step}");
            }
            for &v in grads.d_head_w2_weight.iter() {
                assert!(v.is_finite(), "NaN in d_head_w2_weight at step {step}");
            }

            model.sgd_step(&grads);
        }
        println!("50-step NaN stress test passed");
    }
}
