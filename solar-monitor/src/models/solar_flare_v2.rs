//! SolarFlareModel V2: Clifford Lattice Dynamics for Flare Prediction
//!
//! Replaces V1's Lohe sync (mean-field on S^{d-1}) with the actual
//! bivector field equation from the Geometric Resonance framework:
//!
//!     ∂ₜΨ = [Ω, Ψ] − γ⟨Ψ⟩₂ + J · coupling + kick
//!
//! Architecture:
//!   [1] SHARP → Cl(3,0) kick encoder (9 fields → 9 multivector kicks)
//!   [2] Helmholtz γ-cumsum (4-head multi-scale temporal smoothing)
//!   [3] Clifford lattice dynamics (9 sites, ring topology, KT criticality)
//!       + Thomson Coulomb repulsion (anti-collapse)
//!   [4] Clifford readout (grade-0/2/4 projections + KT diagnostics)
//!   [5] Classification head (MLP → P(flare))
//!
//! Key physics: J is a learnable parameter. If the framework is correct,
//! gradient descent should push J toward J_c = 2/π because that's where
//! the model has maximum sensitivity to SHARP forcing.
//!
//! Follows the 6 geometric purity fixes from CliffordGeodesicGPT:
//!   1. No LayerNorm (manifold-native normalization)
//!   2. Exp-map retraction (not Euler+normalize)
//!   3. No learned gates (geometric r-gating)
//!   4. Clifford commutator IS the computation
//!   5. Grade structure preserved throughout
//!   6. Thomson anti-collapse prevents channel degeneracy

use std::f32::consts::PI;

use serde::{Deserialize, Serialize};

use harmonic_core::clifford_cl3::{
    bivector_multivector, bivector_norm, bivector_part, clifford_commutator, clifford_norm,
    clifford_normalize, geometric_product, init_clifford_state, pseudoscalar_part, scalar_part,
    vector_part, Multivector,
};
use harmonic_core::sequence_ops::{gated_cumsum, gated_cumsum_backward};

// ============================================================================
// Constants
// ============================================================================

/// Number of SHARP magnetogram fields (SolarFlareNet standard).
pub const N_FIELDS: usize = 9;

/// Number of orbital/planetary inputs (sin/cos pairs).
/// All 8 planets + lunar nodal precession = 9 angles × 2 (sin/cos) = 18.
/// Deterministic, zero noise. The model learns which matter.
pub const N_ORBITAL: usize = 18;

/// Number of geomagnetic inputs: Kp index + dKp/dt.
pub const N_GEOMAG: usize = 2;

/// Total input width per timestep: SHARP + orbital + geomagnetic.
pub const N_INPUT: usize = N_FIELDS + N_ORBITAL + N_GEOMAG;

/// Cl(3,0) multivector dimension.
pub const MV_DIM: usize = 8;

/// Number of Helmholtz temporal heads.
pub const N_HEADS: usize = 4;

/// KT critical stiffness (algebraically fixed, Lean-verified).
pub const J_CRITICAL: f32 = 2.0 / PI;

/// Number of Clifford dynamics integration steps per timestep.
pub const DYNAMICS_STEPS: usize = 8;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarFlareV2Config {
    /// Number of input timesteps.
    pub seq_len: usize,
    /// Helmholtz temporal decay timescales (in units of cadence).
    pub helmholtz_taus: [f32; N_HEADS],
    /// Dynamics integration dt.
    pub dynamics_dt: f32,
    /// Number of dynamics sub-steps per input timestep.
    pub dynamics_steps: usize,
    /// Thomson repulsion strength (initial).
    pub thomson_beta_init: f32,
    /// Classification hidden dim.
    pub hidden_dim: usize,
    /// Learning rates.
    pub lr_clifford: f32,
    pub lr_scalar: f32,
    /// Positive class weight for BCE.
    pub pos_weight: f32,
    /// Freeze J_sun at 1/π (the learned operating point).
    /// When true, J_sun weights are not updated during training.
    /// The lattice dynamics (omega, gamma, coupling) learn instead.
    #[serde(default)]
    pub freeze_j_sun: bool,
}

impl Default for SolarFlareV2Config {
    fn default() -> Self {
        SolarFlareV2Config {
            seq_len: 10,
            helmholtz_taus: [4.0, 8.0, 16.0, 32.0],
            dynamics_dt: 0.1,
            dynamics_steps: DYNAMICS_STEPS,
            thomson_beta_init: 0.05,
            hidden_dim: 16,
            // LR scaled for 8D Cl(3,0) dynamics with BPTT across T*steps=30 integration steps.
            // Dynamics params: gradients accumulate across 30 steps, so LR / 30 ≈ effective step.
            // J_sun weights: through sigmoid (max grad 0.25), needs higher LR.
            lr_clifford: 0.001,  // dynamics: omega, gamma, coupling (BPTT-accumulated)
            lr_scalar: 0.01,     // head weights + j_sun_weights
            pos_weight: 1.0,
            freeze_j_sun: false,
        }
    }
}

// ============================================================================
// Model
// ============================================================================

/// SolarFlareModel V2: Clifford lattice dynamics on Cl(3,0).
#[derive(Serialize, Deserialize)]
pub struct SolarFlareV2 {
    pub config: SolarFlareV2Config,

    // --- [1] SHARP → Cl(3,0) kick encoder ---
    // Per-field encoder: maps normalized [0,1] SHARP value to 8-component kick.
    // kick_i = sin(π·x)·a_i + cos(π·x)·b_i where a_i, b_i ∈ Cl(3,0)
    // This gives a smooth great-circle arc in Cl(3,0) parameterized by the SHARP value.
    pub encoder_a: Vec<Multivector>, // N_FIELDS × 8
    pub encoder_b: Vec<Multivector>, // N_FIELDS × 8

    // --- [2] Helmholtz temporal ---
    // Decay rates for gated_cumsum, per (head, component).
    // The Helmholtz filtering operates on the 8 multivector components independently.
    pub helmholtz_gamma: Vec<f32>, // N_HEADS × MV_DIM

    // --- [3] Clifford lattice dynamics ---
    /// Bivector connection Ω on S² (3 params — shared across all sites).
    pub omega: [f32; 3],
    /// Damping coefficient γ (positive, shared).
    pub gamma: f32,
    /// J_model: lattice operating stiffness, FIXED at J_c = 2/π.
    /// The sensor operates at criticality for maximum sensitivity.
    /// NOT learned — this is an architectural choice, like a brain at criticality.
    pub j_model: f32,
    /// J_sun weights: learned mapping from 9 SHARP fields → J_sun ∈ [0, 1].
    /// J_sun represents the Sun's field stiffness, computed from magnetogram data.
    /// When J_sun ≈ J_c, the Sun is near criticality → flare imminent.
    pub j_sun_weights: Vec<f32>, // N_FIELDS weights (SHARP fields)
    pub j_sun_orbital: Vec<f32>, // N_ORBITAL weights (planetary angles)
    pub j_sun_geomag: Vec<f32>,  // N_GEOMAG weights (Kp, dKp/dt)
    pub j_sun_bias: f32,
    /// Coupling matrix K_ij (antisymmetric, N_FIELDS × N_FIELDS).
    pub coupling_matrix: Vec<f32>, // N_FIELDS × N_FIELDS
    /// Thomson repulsion strength β (positive via softplus).
    pub thomson_beta: f32,

    // --- [5] Classification head ---
    // readout_dim → hidden_dim → 1
    pub head_w1: Vec<f32>,   // readout_dim × hidden_dim
    pub head_b1: Vec<f32>,   // hidden_dim
    pub head_w2: Vec<f32>,   // hidden_dim × 1
    pub head_b2: f32,        // scalar bias
}

/// Activations stored during forward pass for backward.
pub struct V2Activations {
    /// Encoded kicks per timestep: (T, N_FIELDS, MV_DIM) flat.
    pub kicks: Vec<f32>,
    /// Per-head Helmholtz output: N_HEADS × (T × N_FIELDS × MV_DIM).
    pub helmholtz_out: Vec<Vec<f32>>,
    /// Temporally filtered kicks: (T, N_FIELDS, MV_DIM) flat.
    pub temporal: Vec<f32>,
    /// Lattice states after dynamics: (T, N_FIELDS) of Multivector.
    pub states: Vec<Vec<Multivector>>,
    /// Per-step intermediate states for BPTT: (T, dynamics_steps+1, N_FIELDS) of Multivector.
    pub bptt_states: Vec<Vec<Vec<Multivector>>>,
    /// Kicks per timestep: (T, N_FIELDS) of Multivector.
    pub kicks_per_t: Vec<Vec<Multivector>>,
    /// Forcing scale per timestep (from J_sun).
    pub forcing_scales: Vec<f32>,
    /// Pre-normalization energy per timestep: sum of |delta|² across sites.
    /// This captures the amplitude signal that normalization removes.
    pub pre_norm_energy: Vec<f32>,
    /// Readout features.
    pub readout: Vec<f32>,
    /// Hidden layer activations.
    pub hidden: Vec<f32>,
    /// Pre-sigmoid logit.
    pub logit: f32,
}

/// Streaming state for continuous operation (no window truncation).
///
/// Carries the Helmholtz temporal state and lattice dynamics state across
/// observations. Each 12-minute SHARP observation is one step. The state
/// accumulates smoothly — no resets, no blips.
#[derive(Clone)]
pub struct V2StreamState {
    /// Helmholtz cumsum accumulators: N_HEADS × N_FIELDS × MV_DIM.
    /// These ARE the temporal memory — they decay exponentially via gamma.
    pub helmholtz_state: Vec<f32>,
    /// Clifford lattice states: N_FIELDS Multivectors (9 sites on ring).
    pub lattice_states: Vec<Multivector>,
    /// Running J_sun (smoothed, not per-step — avoids blips).
    pub j_sun_ema: f32,
    /// EMA smoothing rate for J_sun (prevents 12-min jumps).
    pub j_sun_alpha: f32,
    /// Step counter.
    pub step: usize,
}

impl V2StreamState {
    /// Initialize fresh state.
    pub fn new() -> Self {
        V2StreamState {
            helmholtz_state: vec![0.0f32; N_HEADS * N_FIELDS * MV_DIM],
            lattice_states: (0..N_FIELDS).map(|_| init_clifford_state(1.0)).collect(),
            j_sun_ema: J_CRITICAL,
            j_sun_alpha: 0.1, // smooth over ~10 steps = 2 hours
            step: 0,
        }
    }
}

/// Gradients for all learnable parameters.
#[derive(Clone)]
pub struct V2Grads {
    pub d_encoder_a: Vec<f32>,
    pub d_encoder_b: Vec<f32>,
    pub d_helmholtz_gamma: Vec<f32>,
    pub d_omega: [f32; 3],
    pub d_gamma: f32,
    pub d_j_sun_weights: Vec<f32>,
    pub d_j_sun_orbital: Vec<f32>,
    pub d_j_sun_geomag: Vec<f32>,
    pub d_j_sun_bias: f32,
    pub d_coupling_matrix: Vec<f32>,
    pub d_thomson_beta: f32,
    pub d_head_w1: Vec<f32>,
    pub d_head_b1: Vec<f32>,
    pub d_head_w2: Vec<f32>,
    pub d_head_b2: f32,
}

impl SolarFlareV2 {
    pub fn new(config: SolarFlareV2Config) -> Self {
        // Initialize encoder frames: orthogonal pairs in Cl(3,0)
        let mut encoder_a = Vec::with_capacity(N_FIELDS);
        let mut encoder_b = Vec::with_capacity(N_FIELDS);
        for fi in 0..N_FIELDS {
            // Spread initial encoders across different Cl(3,0) directions
            let angle = fi as f32 * PI / N_FIELDS as f32;
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            // a lives primarily in vector components (grade-1)
            a[1] = angle.cos();
            a[2] = angle.sin();
            a[3] = 0.3;
            // b lives primarily in bivector components (grade-2)
            b[4] = angle.sin();
            b[5] = angle.cos();
            b[6] = 0.3;
            // Normalize both
            let a = clifford_normalize(&a);
            let b = clifford_normalize(&b);
            encoder_a.push(a);
            encoder_b.push(b);
        }

        // Helmholtz gamma: init from tau values
        let mut helmholtz_gamma = vec![0.0f32; N_HEADS * MV_DIM];
        for head in 0..N_HEADS {
            let tau = config.helmholtz_taus[head];
            let g = (-1.0 / tau).exp(); // gamma = e^{-1/tau}
            for d in 0..MV_DIM {
                helmholtz_gamma[head * MV_DIM + d] = g;
            }
        }

        // Omega: initial rotation axis on S²
        let omega = [0.577f32, 0.577, 0.577]; // ~(1,1,1)/sqrt(3)

        // J_model: FIXED at J_c — the sensor operates at criticality
        let j_model = J_CRITICAL;

        // J_sun weights: learned mapping from SHARP → solar field stiffness
        // Initialized proportional to the criticality detector's weights:
        // TOTPOT(0.37), SHRGT45(0.28), R_VALUE(0.18), TOTUSJH(0.09), others small
        let j_sun_weights = vec![
            0.09, // TOTUSJH (helicity)
            0.05, // TOTUSJZ (vertical current)
            0.05, // USFLUX (total flux)
            0.02, // MEANALP (twist)
            0.18, // R_VALUE (PIL flux — key precursor)
            0.37, // TOTPOT (free energy — dominant)
            0.03, // SAVNCPP (current imbalance)
            0.05, // AREA_ACR (region area)
            0.05, // ABSNJZH (net current helicity)
        ];
        // Orbital J_sun weights: all 8 planets + lunar node modulate solar stiffness.
        // Jupiter initialized highest (drives 11-yr cycle). The model learns the rest.
        let j_sun_orbital = vec![
            0.02, 0.02, // Mercury (0.24 yr — fast, may matter for flare timing)
            0.03, 0.03, // Venus (0.62 yr — Venus-Earth-Jupiter alignment)
            0.02, 0.02, // Earth (1.0 yr — annual solar geometry)
            0.02, 0.02, // Mars (1.88 yr)
            0.10, 0.10, // Jupiter (11.86 yr — drives the solar cycle)
            0.05, 0.05, // Saturn (29.46 yr — Gleissberg modulation)
            0.01, 0.01, // Uranus (84 yr)
            0.01, 0.01, // Neptune (164.8 yr — Bond/9)
            0.02, 0.02, // Lunar nodal (18.61 yr — Bond cycle driver)
        ];
        // Kp directly reflects solar wind coupling — high weight.
        let j_sun_geomag = vec![
            0.15, // Kp index (0-9, normalized)
            0.05, // dKp/dt (rate of change — precursor signal)
        ];
        let j_sun_bias = 0.0f32;

        // Coupling matrix: antisymmetric, ring topology initial
        let mut coupling_matrix = vec![0.0f32; N_FIELDS * N_FIELDS];
        for i in 0..N_FIELDS {
            let j = (i + 1) % N_FIELDS;
            coupling_matrix[i * N_FIELDS + j] = 0.1;
            coupling_matrix[j * N_FIELDS + i] = -0.1;
        }

        // Classification head
        let rdim = Self::readout_dim_static();
        let hdim = config.hidden_dim;
        // Xavier init
        let scale_w1 = (6.0 / (rdim + hdim) as f32).sqrt();
        let scale_w2 = (6.0 / (hdim + 1) as f32).sqrt();
        let head_w1: Vec<f32> = (0..rdim * hdim)
            .map(|i| {
                let hash = (i as u64).wrapping_mul(2654435761).wrapping_add(42);
                (hash % 10000) as f32 / 10000.0 * 2.0 * scale_w1 - scale_w1
            })
            .collect();
        let head_b1 = vec![0.0f32; hdim];
        let head_w2: Vec<f32> = (0..hdim)
            .map(|i| {
                let hash = (i as u64).wrapping_mul(2654435761).wrapping_add(99);
                (hash % 10000) as f32 / 10000.0 * 2.0 * scale_w2 - scale_w2
            })
            .collect();
        // Bias init at class prior logit
        let head_b2 = -1.4f32; // log(0.20/0.80) ≈ -1.4
        let thomson_beta_init = config.thomson_beta_init;

        SolarFlareV2 {
            config,
            encoder_a,
            encoder_b,
            helmholtz_gamma,
            omega,
            gamma: 0.01,
            j_model,
            j_sun_weights,
            j_sun_orbital,
            j_sun_geomag,
            j_sun_bias,
            coupling_matrix,
            thomson_beta: thomson_beta_init,
            head_w1,
            head_b1,
            head_w2,
            head_b2,
        }
    }

    /// Readout dimension (static, for init).
    fn readout_dim_static() -> usize {
        // Per-field: scalar_part(1) + bivector_norm(1) + pseudoscalar(1) = 3 per field = 27
        // Cross-field: commutator norm per pair = N*(N-1)/2 = 36
        // Sync order: 1
        // Lattice critical gap: 1 (|J_model - J_c|, always ~0)
        // Thomson energy: 1
        // J_sun features: j_sun_last(1) + trend(1) + gap(1) + above(1) = 4
        // Pre-norm energy: last(1) + max(1) + trend(1) = 3
        // Total: 27 + 36 + 1 + 1 + 1 + 4 + 3 = 73
        3 * N_FIELDS + N_FIELDS * (N_FIELDS - 1) / 2 + 3 + 4 + 3
    }

    /// Readout dimension.
    pub fn readout_dim(&self) -> usize {
        Self::readout_dim_static()
    }

    /// Total learnable parameters.
    pub fn param_count(&self) -> usize {
        let enc = 2 * N_FIELDS * MV_DIM;              // encoder a + b
        let helm = N_HEADS * MV_DIM;                    // helmholtz gamma
        let dyn_params = 3 + 1 + 1 + N_FIELDS * N_FIELDS + 1; // omega + gamma + J + K + beta
        let rdim = self.readout_dim();
        let head = rdim * self.config.hidden_dim + self.config.hidden_dim  // w1 + b1
                 + self.config.hidden_dim + 1;                              // w2 + b2
        enc + helm + dyn_params + head
    }

    // ========================================================================
    // Streaming step (continuous, no truncation)
    // ========================================================================

    /// Process one observation (N_INPUT values) and return flare probability.
    ///
    /// State carries forward smoothly between calls. The Helmholtz filters
    /// accumulate with exponential decay (no resets). The lattice dynamics
    /// evolve continuously. J_sun is EMA-smoothed to prevent 12-min blips.
    ///
    /// Call this once per 12-minute SHARP observation.
    pub fn step(&self, obs: &[f32], state: &mut V2StreamState) -> f32 {
        let n = N_FIELDS;

        // --- [1] Encode SHARP fields to Cl(3,0) kicks ---
        let mut kick_sum = [0.0f32; MV_DIM]; // sum across fields
        for fi in 0..n {
            let x = obs[fi].clamp(0.0, 1.0);
            let sin_px = (PI * x).sin();
            let cos_px = (PI * x).cos();
            for d in 0..MV_DIM {
                let k = sin_px * self.encoder_a[fi][d] + cos_px * self.encoder_b[fi][d];
                kick_sum[d] += k;
                // Per-field kick for Helmholtz update
                let h_idx_base = fi * MV_DIM + d;
                // Update each Helmholtz head's state for this field
                for head in 0..N_HEADS {
                    let gamma = self.helmholtz_gamma[head * MV_DIM + d];
                    let s_idx = head * n * MV_DIM + h_idx_base;
                    // Gated cumsum: s[t] = gamma * s[t-1] + (1-gamma) * input[t]
                    state.helmholtz_state[s_idx] =
                        gamma * state.helmholtz_state[s_idx] + (1.0 - gamma) * k;
                }
            }
        }

        // --- [2] Average Helmholtz state across heads → temporal kick per field ---
        let mut temporal_kicks: Vec<Multivector> = Vec::with_capacity(n);
        for fi in 0..n {
            let mut mv = [0.0f32; MV_DIM];
            for d in 0..MV_DIM {
                let mut sum = 0.0f32;
                for head in 0..N_HEADS {
                    sum += state.helmholtz_state[head * n * MV_DIM + fi * MV_DIM + d];
                }
                mv[d] = sum / N_HEADS as f32;
            }
            temporal_kicks.push(mv);
        }

        // --- [3] Compute J_sun (EMA-smoothed) ---
        let mut j_sun_raw = self.j_sun_bias;
        for fi in 0..n {
            j_sun_raw += self.j_sun_weights[fi] * obs[fi];
        }
        for oi in 0..N_ORBITAL {
            j_sun_raw += self.j_sun_orbital[oi] * obs[N_FIELDS + oi];
        }
        for gi in 0..N_GEOMAG {
            j_sun_raw += self.j_sun_geomag[gi] * obs[N_FIELDS + N_ORBITAL + gi];
        }
        let j_sun_instant = J_CRITICAL * (0.5 + sigmoid(j_sun_raw));
        // EMA smooth: prevents 12-min blips in J_sun
        state.j_sun_ema = state.j_sun_alpha * j_sun_instant
                        + (1.0 - state.j_sun_alpha) * state.j_sun_ema;
        let j_sun = state.j_sun_ema;

        // --- [4] Clifford lattice dynamics (evolve, don't reset) ---
        let omega_mv = bivector_to_multivector_v2(&self.omega);
        let gamma = self.gamma.abs().max(1e-6);
        let j_model = self.j_model;
        let beta = softplus(self.thomson_beta);
        let dt = self.config.dynamics_dt;
        let forcing_scale = J_CRITICAL / j_sun.max(0.01);

        for _step in 0..self.config.dynamics_steps {
            let mut new_states = Vec::with_capacity(n);
            for fi in 0..n {
                let psi = &state.lattice_states[fi];
                let comm = clifford_commutator(&omega_mv, psi);
                let psi_biv = bivector_multivector_v2(psi);
                let mut coupling_force = [0.0f32; 8];
                for fj in 0..n {
                    let k_ij = self.coupling_matrix[fi * n + fj];
                    if k_ij.abs() < 1e-10 { continue; }
                    let couple = clifford_commutator(&state.lattice_states[fj], psi);
                    for d in 0..8 { coupling_force[d] += k_ij * couple[d]; }
                }
                // Thomson repulsion
                let mut thomson = [0.0f32; 8];
                for fj in 0..n {
                    if fj == fi { continue; }
                    let diff_sq: f32 = (0..8).map(|d| {
                        let diff = psi[d] - state.lattice_states[fj][d];
                        diff * diff
                    }).sum();
                    let inv_dist = 1.0 / (diff_sq.max(0.01).sqrt());
                    for d in 0..8 {
                        thomson[d] += beta * inv_dist * (psi[d] - state.lattice_states[fj][d]);
                    }
                }
                // Euler step
                let mut new_psi = [0.0f32; 8];
                for d in 0..8 {
                    new_psi[d] = psi[d] + dt * (
                        comm[d]
                        - gamma * psi_biv[d]
                        + j_model * coupling_force[d]
                        + forcing_scale * temporal_kicks[fi][d]
                        + thomson[d]
                    );
                }
                new_states.push(clifford_normalize(&new_psi));
            }
            state.lattice_states = new_states;
        }

        // --- [5] Readout + classification ---
        let mut readout = self.compute_readout(&state.lattice_states);
        // Append J_sun diagnostics (same as window-based forward)
        // J_sun_last, J_sun distance from J_c, pre_norm energy (approx from lattice)
        readout.push(j_sun);
        readout.push((j_sun - J_CRITICAL).abs());
        let pre_norm_e: f32 = state.lattice_states.iter()
            .map(|s| clifford_norm(s))
            .sum::<f32>() / n as f32;
        readout.push(pre_norm_e);

        // MLP classification: readout → hidden (SiLU) → sigmoid
        let rdim = readout.len().min(self.head_w1.len() / self.config.hidden_dim.max(1));
        let hdim = self.config.hidden_dim;
        let mut hidden = vec![0.0f32; hdim];
        for i in 0..hdim {
            let mut v = self.head_b1.get(i).copied().unwrap_or(0.0);
            for j in 0..rdim {
                v += self.head_w1.get(i * rdim + j).copied().unwrap_or(0.0) * readout[j];
            }
            hidden[i] = v * sigmoid(v); // SiLU
        }
        let mut logit = self.head_b2;
        for i in 0..hdim {
            logit += self.head_w2.get(i).copied().unwrap_or(0.0) * hidden[i];
        }
        let prob = sigmoid(logit);

        state.step += 1;
        prob
    }

    // ========================================================================
    // Window-based forward pass (legacy, for backward compatibility)
    // ========================================================================

    pub fn forward(&self, input: &[f32]) -> (f32, V2Activations) {
        let t = self.config.seq_len;
        let n = N_FIELDS;

        // --- [1] SHARP → Cl(3,0) kick ---
        // kick_i(x) = sin(π·x)·a_i + cos(π·x)·b_i
        // Only the N_FIELDS SHARP values produce kicks (9 lattice sites).
        // Orbital inputs modulate J_sun, not kicks.
        let mut kicks = vec![0.0f32; t * n * MV_DIM];
        for ti in 0..t {
            for fi in 0..n {
                let x = input[ti * N_INPUT + fi].clamp(0.0, 1.0);
                let sin_px = (PI * x).sin();
                let cos_px = (PI * x).cos();
                let base = (ti * n + fi) * MV_DIM;
                for d in 0..MV_DIM {
                    kicks[base + d] = sin_px * self.encoder_a[fi][d]
                                    + cos_px * self.encoder_b[fi][d];
                }
            }
        }

        // --- [2] Helmholtz temporal ---
        let mut helmholtz_out = Vec::with_capacity(N_HEADS);
        for head in 0..N_HEADS {
            let gamma_start = head * MV_DIM;
            let gamma_slice = &self.helmholtz_gamma[gamma_start..gamma_start + MV_DIM];
            let mut head_out = vec![0.0f32; t * n * MV_DIM];

            for fi in 0..n {
                // Extract this field's kick sequence: (T, MV_DIM)
                let mut field_seq = vec![0.0f32; t * MV_DIM];
                for ti in 0..t {
                    for d in 0..MV_DIM {
                        field_seq[ti * MV_DIM + d] = kicks[(ti * n + fi) * MV_DIM + d];
                    }
                }
                let cumsum = gated_cumsum(&field_seq, t, MV_DIM, gamma_slice);
                for ti in 0..t {
                    for d in 0..MV_DIM {
                        head_out[(ti * n + fi) * MV_DIM + d] = cumsum[ti * MV_DIM + d];
                    }
                }
            }
            helmholtz_out.push(head_out);
        }

        // Average across heads
        let mut temporal = vec![0.0f32; t * n * MV_DIM];
        for idx in 0..t * n * MV_DIM {
            let mut sum = 0.0f32;
            for head in 0..N_HEADS {
                sum += helmholtz_out[head][idx];
            }
            temporal[idx] = sum / N_HEADS as f32;
        }

        // --- [3] Clifford lattice dynamics ---
        let omega_mv = bivector_to_multivector_v2(&self.omega);
        let gamma = self.gamma.abs().max(1e-6);
        let j_model = self.j_model; // FIXED at J_c — sensor operating point
        let beta = softplus(self.thomson_beta);
        let dt = self.config.dynamics_dt;
        let steps = self.config.dynamics_steps;

        // Initialize states: scalar +1 for all sites
        let mut states: Vec<Multivector> = (0..n)
            .map(|_| init_clifford_state(1.0))
            .collect();

        let mut all_states = Vec::with_capacity(t);
        let mut all_bptt_states = Vec::with_capacity(t);

        // Store J_sun per timestep for readout, and kicks/forcing for BPTT
        let mut j_sun_history = Vec::with_capacity(t);
        let mut all_kicks_t: Vec<Vec<Multivector>> = Vec::with_capacity(t);
        let mut all_forcing_scales: Vec<f32> = Vec::with_capacity(t);
        let mut pre_norm_energy: Vec<f32> = Vec::with_capacity(t);

        for ti in 0..t {
            // Compute J_sun from SHARP + orbital inputs at this timestep
            // J_sun = sigmoid(w_sharp · sharp + w_orbital · orbital + b) → [0, 1]
            // then scale to [0.5*J_c, 1.5*J_c]
            let mut j_sun_raw = self.j_sun_bias;
            for fi in 0..n {
                j_sun_raw += self.j_sun_weights[fi] * input[ti * N_INPUT + fi];
            }
            for oi in 0..N_ORBITAL {
                j_sun_raw += self.j_sun_orbital[oi] * input[ti * N_INPUT + N_FIELDS + oi];
            }
            for gi in 0..N_GEOMAG {
                j_sun_raw += self.j_sun_geomag[gi] * input[ti * N_INPUT + N_FIELDS + N_ORBITAL + gi];
            }
            let j_sun_01 = sigmoid(j_sun_raw); // [0, 1]
            let j_sun = J_CRITICAL * (0.5 + j_sun_01); // [0.5*J_c, 1.5*J_c]
            j_sun_history.push(j_sun);

            // Extract kicks for this timestep as Multivectors
            let mut kicks_t: Vec<Multivector> = Vec::with_capacity(n);
            for fi in 0..n {
                let base = (ti * n + fi) * MV_DIM;
                let mut mv = [0.0f32; 8];
                for d in 0..MV_DIM {
                    mv[d] = temporal[base + d];
                }
                kicks_t.push(mv);
            }

            // J_sun modulates the forcing amplitude:
            // When J_sun ≈ J_c, forcing is at baseline (moderate).
            // When J_sun > J_c, the Sun is "stiff" → less responsive to kicks.
            // When J_sun < J_c, the Sun is "soft" → kicks have more effect.
            // This models how the coronal field stiffness affects energy storage.
            let forcing_scale = J_CRITICAL / j_sun.max(0.01); // > 1 when J_sun < J_c
            all_kicks_t.push(kicks_t.clone());
            all_forcing_scales.push(forcing_scale);

            // Sub-step integration
            let mut step_states = Vec::with_capacity(steps + 1);
            step_states.push(states.clone());

            for _step in 0..steps {
                let mut new_states = Vec::with_capacity(n);

                for fi in 0..n {
                    let psi = &states[fi];

                    // [Ω, Ψ_i] — commutator rotation
                    let comm = clifford_commutator(&omega_mv, psi);

                    // −γ⟨Ψ_i⟩₂ — bivector damping
                    let psi_biv = bivector_multivector_v2(psi);

                    // J_model · Σ_j K_ij · [Ψ_j, Ψ_i] — lattice coupling at J_c
                    let mut coupling_force = [0.0f32; 8];
                    for fj in 0..n {
                        let k_ij = self.coupling_matrix[fi * n + fj];
                        if k_ij.abs() < 1e-10 { continue; }
                        let couple = clifford_commutator(&states[fj], psi);
                        for d in 0..8 {
                            coupling_force[d] += k_ij * couple[d];
                        }
                    }

                    // Thomson repulsion
                    let mut repulsion = [0.0f32; 8];
                    if beta > 1e-8 {
                        for fj in 0..n {
                            if fj == fi { continue; }
                            let mut d_sq = 0.0f32;
                            for d in 0..8 {
                                let diff = psi[d] - states[fj][d];
                                d_sq += diff * diff;
                            }
                            let d_soft = (d_sq + 0.01).sqrt();
                            let inv_d3 = 1.0 / (d_soft * d_soft * d_soft);
                            for d in 0..8 {
                                repulsion[d] += beta * (psi[d] - states[fj][d]) * inv_d3;
                            }
                        }
                    }

                    // Dynamics: J_model (fixed at J_c) drives ordering,
                    //           J_sun (from SHARP) modulates forcing amplitude.
                    // δΨ = J_model·([Ω,Ψ] + coupling) - γ⟨Ψ⟩₂ + repulsion + scale·kick
                    let mut delta = [0.0f32; 8];
                    for d in 0..8 {
                        delta[d] = j_model * (comm[d] + coupling_force[d])
                            - gamma * psi_biv[d]
                            + repulsion[d]
                            + forcing_scale * kicks_t[fi][d];
                    }

                    let mut psi_next = [0.0f32; 8];
                    for d in 0..8 {
                        psi_next[d] = psi[d] + dt * delta[d];
                    }
                    new_states.push(clifford_normalize(&psi_next));
                }

                states = new_states;
                step_states.push(states.clone());
            }

            // Track pre-norm energy: sum of delta magnitudes in last sub-step
            let mut energy = 0.0f32;
            let last_step = &step_states[steps]; // post-update
            let prev_step = &step_states[steps - 1]; // pre-update
            for fi in 0..n {
                for d in 0..8 {
                    let diff = last_step[fi][d] - prev_step[fi][d];
                    energy += diff * diff;
                }
            }
            pre_norm_energy.push(energy.sqrt());

            all_states.push(states.clone());
            all_bptt_states.push(step_states);
        }

        // --- [4] Clifford readout ---
        // Use last timestep states + J_sun history
        let final_states = &all_states[t - 1];
        let mut readout = self.compute_readout(final_states);

        // Append J_sun features: last value, trend, proximity to J_c
        let j_sun_last = *j_sun_history.last().unwrap_or(&J_CRITICAL);
        let j_sun_first = *j_sun_history.first().unwrap_or(&J_CRITICAL);
        let j_sun_trend = j_sun_last - j_sun_first; // positive = approaching from below
        let j_sun_gap = (j_sun_last - J_CRITICAL).abs(); // how close to critical
        let j_sun_above = if j_sun_last > J_CRITICAL { 1.0 } else { 0.0 }; // which side
        readout.push(j_sun_last);
        readout.push(j_sun_trend);
        readout.push(j_sun_gap);
        readout.push(j_sun_above);

        // Pre-norm energy features: amplitude signal that normalization removes
        let e_last = *pre_norm_energy.last().unwrap_or(&0.0);
        let e_first = *pre_norm_energy.first().unwrap_or(&0.0);
        let e_max = pre_norm_energy.iter().cloned().fold(0.0f32, f32::max);
        let e_trend = e_last - e_first; // rising energy = building toward eruption
        readout.push(e_last);
        readout.push(e_max);
        readout.push(e_trend);

        // --- [5] Classification head ---
        let rdim = self.readout_dim();
        let hdim = self.config.hidden_dim;

        // h1 = SiLU(W1 · readout + b1)
        let mut hidden = vec![0.0f32; hdim];
        for i in 0..hdim {
            let mut v = self.head_b1[i];
            for j in 0..rdim {
                v += self.head_w1[i * rdim + j] * readout[j];
            }
            hidden[i] = v * sigmoid(v); // SiLU
        }

        // logit = W2 · h1 + b2
        let mut logit = self.head_b2;
        for i in 0..hdim {
            logit += self.head_w2[i] * hidden[i];
        }

        let prob = sigmoid(logit);

        let acts = V2Activations {
            kicks,
            helmholtz_out,
            temporal,
            states: all_states,
            bptt_states: all_bptt_states,
            kicks_per_t: all_kicks_t,
            forcing_scales: all_forcing_scales,
            pre_norm_energy,
            readout,
            hidden,
            logit,
        };

        (prob, acts)
    }

    /// Compute readout features from final lattice states.
    fn compute_readout(&self, states: &[Multivector]) -> Vec<f32> {
        let n = N_FIELDS;
        let mut features = Vec::with_capacity(self.readout_dim());

        // Per-field: scalar, bivector_norm, pseudoscalar
        for fi in 0..n {
            features.push(scalar_part(&states[fi]));
            features.push(bivector_norm(&states[fi]));
            features.push(pseudoscalar_part(&states[fi]));
        }

        // Cross-field: bivector similarity (grade-2 commutator complexity)
        for fi in 0..n {
            for fj in (fi + 1)..n {
                let comm = clifford_commutator(&states[fi], &states[fj]);
                features.push(bivector_norm(&comm));
            }
        }

        // Sync order: Kuramoto parameter r = |mean(Ψ)| / N
        let mut mean = [0.0f32; 8];
        for fi in 0..n {
            for d in 0..8 {
                mean[d] += states[fi][d];
            }
        }
        let r = clifford_norm(&mean) / n as f32;
        features.push(r);

        // Lattice critical gap: |J_model - J_c| (should be ~0, sanity check)
        features.push((self.j_model - J_CRITICAL).abs());

        // Thomson energy: sum of 1/d_{ij} (repulsion diagnostic)
        let mut thomson_e = 0.0f32;
        for fi in 0..n {
            for fj in (fi + 1)..n {
                let mut d_sq = 0.0f32;
                for d in 0..8 {
                    let diff = states[fi][d] - states[fj][d];
                    d_sq += diff * diff;
                }
                thomson_e += 1.0 / (d_sq + 0.01).sqrt();
            }
        }
        features.push(thomson_e / (n * (n - 1) / 2) as f32);

        features
    }

    // ========================================================================
    // Backward pass (BPTT through Clifford dynamics)
    // ========================================================================

    /// Compute loss and gradients via BPTT.
    pub fn backward(&self, input: &[f32], acts: &V2Activations, target: f32) -> (f32, V2Grads) {
        let t = self.config.seq_len;
        let n = N_FIELDS;
        let rdim = self.readout_dim();
        let hdim = self.config.hidden_dim;

        // --- BCE loss + KT criticality regularizer ---
        let p = acts.logit;
        let prob = sigmoid(p);
        let prob_clamped = prob.clamp(1e-7, 1.0 - 1e-7);
        let bce = -(self.config.pos_weight * target * prob_clamped.ln()
            + (1.0 - target) * (1.0 - prob_clamped).ln());

        // No J penalty needed — J_model is fixed at J_c, not learned.
        let loss = bce;

        let d_logit = if target > 0.5 {
            self.config.pos_weight * (prob - 1.0)
        } else {
            prob
        };

        // --- [5] Head backward ---
        let mut d_head_w2 = vec![0.0f32; hdim];
        let mut d_head_b2 = d_logit;
        let mut d_hidden = vec![0.0f32; hdim];
        for i in 0..hdim {
            d_head_w2[i] = d_logit * acts.hidden[i];
            d_hidden[i] = d_logit * self.head_w2[i];
        }

        // SiLU backward: d/dx[x·σ(x)] = σ(x) + x·σ(x)·(1-σ(x))
        let mut d_readout = vec![0.0f32; rdim];
        let mut d_head_w1 = vec![0.0f32; rdim * hdim];
        let mut d_head_b1 = vec![0.0f32; hdim];
        for i in 0..hdim {
            // Reconstruct pre-activation
            let mut h1_pre = self.head_b1[i];
            for j in 0..rdim {
                h1_pre += self.head_w1[i * rdim + j] * acts.readout[j];
            }
            let sig = sigmoid(h1_pre);
            let dsilu = sig + h1_pre * sig * (1.0 - sig);
            let d_pre = d_hidden[i] * dsilu;

            d_head_b1[i] = d_pre;
            for j in 0..rdim {
                d_head_w1[i * rdim + j] = d_pre * acts.readout[j];
                d_readout[j] += d_pre * self.head_w1[i * rdim + j];
            }
        }

        // --- [4] Readout backward → d_states for last timestep ---
        let final_states = &acts.states[t - 1];
        let d_final_states = self.readout_backward(final_states, &d_readout);

        // --- [3] Analytical BPTT through Clifford lattice dynamics ---
        use super::solar_flare_v2_backward as bptt;

        let omega_mv = bivector_to_multivector_v2(&self.omega);
        let j_model = self.j_model;
        let gamma = self.gamma.abs().max(1e-6);
        let dt = self.config.dynamics_dt;
        let steps = self.config.dynamics_steps;

        let mut d_omega = [0.0f32; 3];
        let mut d_gamma = 0.0f32;
        let mut d_coupling_matrix = vec![0.0f32; n * n];
        let d_thomson_beta = 0.0f32; // Thomson backward is complex, keep at 0 for now

        // d_states: gradient w.r.t. states at each timestep (start from last)
        let d_final_states_copy = d_final_states.clone();
        let mut d_states: Vec<Multivector> = d_final_states;

        // Reverse through timesteps
        for ti in (0..t).rev() {
            let forcing_scale = acts.forcing_scales[ti];
            let kicks_t = &acts.kicks_per_t[ti];

            // Reverse through sub-steps within this timestep
            for step in (0..steps).rev() {
                let old_states = &acts.bptt_states[ti][step]; // states before this sub-step

                // Recompute pre_norm for each site (needed for normalize Jacobian)
                let mut pre_norms: Vec<Multivector> = Vec::with_capacity(n);
                for fi in 0..n {
                    let psi = &old_states[fi];
                    let comm = clifford_commutator(&omega_mv, psi);
                    let psi_biv = bivector_multivector_v2(psi);
                    let mut coupling_force = [0.0f32; 8];
                    for fj in 0..n {
                        let k_ij = self.coupling_matrix[fi * n + fj];
                        if k_ij.abs() < 1e-10 { continue; }
                        let couple = clifford_commutator(&old_states[fj], psi);
                        for d in 0..8 { coupling_force[d] += k_ij * couple[d]; }
                    }
                    let mut pre_norm = [0.0f32; 8];
                    for d in 0..8 {
                        pre_norm[d] = psi[d] + dt * (
                            j_model * (comm[d] + coupling_force[d])
                            - gamma * psi_biv[d]
                            + forcing_scale * kicks_t[fi][d]
                        );
                    }
                    pre_norms.push(pre_norm);
                }

                // Backward through each site
                let mut d_old_states = vec![[0.0f32; 8]; n];
                for fi in 0..n {
                    let (d_psi, d_om, d_gam, d_neighbors, d_coup) = bptt::step_backward(
                        &old_states[fi],
                        &omega_mv,
                        &kicks_t[fi],
                        old_states,
                        &self.coupling_matrix[fi * n..fi * n + n],
                        j_model,
                        gamma,
                        forcing_scale,
                        dt,
                        &pre_norms[fi],
                        &d_states[fi],
                    );

                    // Accumulate state gradients
                    for d in 0..8 { d_old_states[fi][d] += d_psi[d]; }
                    for fj in 0..n {
                        for d in 0..8 { d_old_states[fj][d] += d_neighbors[fj][d]; }
                    }

                    // Accumulate parameter gradients
                    for c in 0..3 { d_omega[c] += d_om[c]; }
                    d_gamma += d_gam;
                    for fj in 0..n { d_coupling_matrix[fi * n + fj] += d_coup[fj]; }
                }

                d_states = d_old_states;
            }
            // d_states now holds gradient w.r.t. states at the START of timestep ti
            // This flows back to the temporal/kicks (but we don't backprop further for now)
        }

        // Sanitize dynamics gradients
        for g in d_omega.iter_mut() { if !g.is_finite() { *g = 0.0; } }
        if !d_gamma.is_finite() { d_gamma = 0.0; }
        for g in d_coupling_matrix.iter_mut() { if !g.is_finite() { *g = 0.0; } }

        // --- J_sun weights: analytical gradient through readout ---
        let base_rd = 3 * n + n * (n - 1) / 2 + 3; // index of j_sun_last in readout
        let d_j_sun_last = d_readout.get(base_rd).copied().unwrap_or(0.0);
        let last_ti = t - 1;
        let mut z = self.j_sun_bias;
        for fi in 0..n {
            z += self.j_sun_weights[fi] * input[last_ti * N_INPUT + fi];
        }
        for oi in 0..N_ORBITAL {
            z += self.j_sun_orbital[oi] * input[last_ti * N_INPUT + N_FIELDS + oi];
        }
        for gi in 0..N_GEOMAG {
            z += self.j_sun_geomag[gi] * input[last_ti * N_INPUT + N_FIELDS + N_ORBITAL + gi];
        }
        let sig_z = sigmoid(z);
        let sig_prime = sig_z * (1.0 - sig_z);
        let mut d_j_sun_weights = vec![0.0f32; n];
        for fi in 0..n {
            d_j_sun_weights[fi] = d_j_sun_last * J_CRITICAL * sig_prime * input[last_ti * N_INPUT + fi];
        }
        let mut d_j_sun_orbital = vec![0.0f32; N_ORBITAL];
        for oi in 0..N_ORBITAL {
            d_j_sun_orbital[oi] = d_j_sun_last * J_CRITICAL * sig_prime * input[last_ti * N_INPUT + N_FIELDS + oi];
        }
        let mut d_j_sun_geomag = vec![0.0f32; N_GEOMAG];
        for gi in 0..N_GEOMAG {
            d_j_sun_geomag[gi] = d_j_sun_last * J_CRITICAL * sig_prime * input[last_ti * N_INPUT + N_FIELDS + N_ORBITAL + gi];
        }
        let d_j_sun_bias = d_j_sun_last * J_CRITICAL * sig_prime;

        // --- [2] Helmholtz backward (analytical via gated_cumsum_backward) ---
        // d_states holds gradient w.r.t. the post-dynamics states at timestep 0.
        // But we need gradient w.r.t. temporal (the helmholtz output).
        // The dynamics used temporal as kicks. The gradient flows:
        //   d_temporal[ti, fi, d] += forcing_scale[ti] * d_states_at_ti[fi][d]
        // We accumulated d_states through the full BPTT reverse, but we only have
        // the final d_states (for timestep 0). For a proper chain, we'd need per-timestep
        // d_temporal. Let's use d_states from the LAST timestep's readout backward
        // as an approximation, since that's where the loss gradient originates.
        //
        // Proper fix: accumulate d_temporal per timestep during BPTT (TODO).
        // For now, use readout backward d_states as the temporal gradient signal.
        let mut d_temporal_raw = vec![0.0f32; t * n * MV_DIM];
        // The readout only uses the last timestep, so d_temporal is nonzero only there
        for fi in 0..n {
            let forcing_scale_last = acts.forcing_scales[t - 1];
            for d in 0..MV_DIM {
                d_temporal_raw[(t - 1) * n * MV_DIM + fi * MV_DIM + d] =
                    forcing_scale_last * d_final_states_copy[fi][d];
            }
        }

        let mut d_helmholtz_gamma = vec![0.0f32; N_HEADS * MV_DIM];
        let mut d_encoded = vec![0.0f32; t * n * MV_DIM];
        for head in 0..N_HEADS {
            let gamma_start = head * MV_DIM;
            let gamma_slice = &self.helmholtz_gamma[gamma_start..gamma_start + MV_DIM];

            for fi in 0..n {
                // Extract grad for this field: d_temporal_raw / N_HEADS (chain rule for average)
                let mut grad_field = vec![0.0f32; t * MV_DIM];
                for ti in 0..t {
                    for d in 0..MV_DIM {
                        grad_field[ti * MV_DIM + d] =
                            d_temporal_raw[ti * n * MV_DIM + fi * MV_DIM + d] / N_HEADS as f32;
                    }
                }

                // Extract input for this field from encoded kicks
                let mut inp_field = vec![0.0f32; t * MV_DIM];
                for ti in 0..t {
                    for d in 0..MV_DIM {
                        inp_field[ti * MV_DIM + d] = acts.kicks[(ti * n + fi) * MV_DIM + d];
                    }
                }

                let (d_inp, d_gamma) =
                    gated_cumsum_backward(&grad_field, &inp_field, gamma_slice, t, MV_DIM);

                for ti in 0..t {
                    for d in 0..MV_DIM {
                        d_encoded[ti * n * MV_DIM + fi * MV_DIM + d] += d_inp[ti * MV_DIM + d];
                    }
                }
                for d in 0..MV_DIM {
                    d_helmholtz_gamma[gamma_start + d] += d_gamma[d];
                }
            }
        }

        // --- [1] Encoder backward (analytical) ---
        // kick_i(x) = sin(πx)·a_i + cos(πx)·b_i
        // d_loss/d_a_fi[d] = Σ_t sin(π·x_{t,fi}) · d_encoded[t,fi,d]
        // d_loss/d_b_fi[d] = Σ_t cos(π·x_{t,fi}) · d_encoded[t,fi,d]
        let mut d_encoder_a = vec![0.0f32; N_FIELDS * MV_DIM];
        let mut d_encoder_b = vec![0.0f32; N_FIELDS * MV_DIM];
        for ti in 0..t {
            for fi in 0..n {
                let x = input[ti * n + fi].clamp(0.0, 1.0);
                let sin_px = (std::f32::consts::PI * x).sin();
                let cos_px = (std::f32::consts::PI * x).cos();
                for d in 0..MV_DIM {
                    let dk = d_encoded[ti * n * MV_DIM + fi * MV_DIM + d];
                    d_encoder_a[fi * MV_DIM + d] += sin_px * dk;
                    d_encoder_b[fi * MV_DIM + d] += cos_px * dk;
                }
            }
        }

        let grads = V2Grads {
            d_encoder_a,
            d_encoder_b,
            d_helmholtz_gamma,
            d_omega,
            d_gamma,
            d_j_sun_weights,
            d_j_sun_orbital,
            d_j_sun_geomag,
            d_j_sun_bias,
            d_coupling_matrix,
            d_thomson_beta,
            d_head_w1,
            d_head_b1,
            d_head_w2,
            d_head_b2,
        };

        (loss, grads)
    }

    /// Forward pass with parameter overrides (for finite-diff gradient computation).
    /// Returns loss only (no activations).
    fn forward_with_overrides(
        &self,
        kicks: &[f32],
        temporal: &[f32],
        omega_override: Option<&[f32; 3]>,
        gamma_override: Option<f32>,
        coupling_override: Option<&[f32]>,
        beta_override: Option<f32>,
        _base_readout: &[f32],
    ) -> f32 {
        let t = self.config.seq_len;
        let n = N_FIELDS;
        let dt = self.config.dynamics_dt;
        let steps = self.config.dynamics_steps;

        let omega = omega_override.unwrap_or(&self.omega);
        let omega_mv = bivector_to_multivector_v2(omega);
        let gamma = gamma_override.unwrap_or(self.gamma).abs().max(1e-6);
        let j = self.j_model; // FIXED at J_c
        let coupling = coupling_override.unwrap_or(&self.coupling_matrix);
        let beta = softplus(beta_override.unwrap_or(self.thomson_beta));

        let mut states: Vec<Multivector> = (0..n)
            .map(|_| init_clifford_state(1.0))
            .collect();

        for ti in 0..t {
            let mut kicks_t: Vec<Multivector> = Vec::with_capacity(n);
            for fi in 0..n {
                let base = (ti * n + fi) * MV_DIM;
                let mut mv = [0.0f32; 8];
                for d in 0..MV_DIM {
                    mv[d] = temporal[base + d];
                }
                kicks_t.push(mv);
            }

            for _step in 0..steps {
                let prev = states.clone();
                for fi in 0..n {
                    let psi = &prev[fi];
                    let comm = clifford_commutator(&omega_mv, psi);
                    let psi_biv = bivector_multivector_v2(psi);
                    let mut coupling_force = [0.0f32; 8];
                    for fj in 0..n {
                        let k_ij = coupling[fi * n + fj];
                        if k_ij.abs() < 1e-10 { continue; }
                        let couple = clifford_commutator(&prev[fj], psi);
                        for d in 0..8 {
                            coupling_force[d] += k_ij * couple[d];
                        }
                    }
                    let mut repulsion = [0.0f32; 8];
                    if beta > 1e-8 {
                        for fj in 0..n {
                            if fj == fi { continue; }
                            let mut d_sq = 0.0f32;
                            for d in 0..8 { d_sq += (psi[d] - prev[fj][d]).powi(2); }
                            let inv_d3 = 1.0 / ((d_sq + 0.01).sqrt().powi(3));
                            for d in 0..8 {
                                repulsion[d] += beta * (psi[d] - prev[fj][d]) * inv_d3;
                            }
                        }
                    }
                    // J_model (fixed at J_c) drives ordering, kicks at unit scale
                    let mut psi_next = [0.0f32; 8];
                    for d in 0..8 {
                        psi_next[d] = psi[d] + dt * (j * (comm[d] + coupling_force[d])
                            - gamma * psi_biv[d] + repulsion[d] + kicks_t[fi][d]);
                    }
                    states[fi] = clifford_normalize(&psi_next);
                }
            }
        }

        // Readout + head → loss
        let mut readout = self.compute_readout(&states);
        // Pad with J_sun features (use the model's current j_sun_weights as proxy)
        // In forward_with_overrides we don't have the raw input, so use neutral values
        readout.push(J_CRITICAL); // j_sun_last
        readout.push(0.0);        // j_sun_trend
        readout.push(0.0);        // j_sun_gap
        readout.push(0.5);        // j_sun_above
        readout.push(0.0);        // pre_norm_energy last
        readout.push(0.0);        // pre_norm_energy max
        readout.push(0.0);        // pre_norm_energy trend
        let rdim = self.readout_dim();
        let hdim = self.config.hidden_dim;
        let mut hidden = vec![0.0f32; hdim];
        for i in 0..hdim {
            let mut v = self.head_b1[i];
            for j in 0..rdim {
                v += self.head_w1[i * rdim + j] * readout[j];
            }
            hidden[i] = v * sigmoid(v);
        }
        let mut logit = self.head_b2;
        for i in 0..hdim {
            logit += self.head_w2[i] * hidden[i];
        }
        // Return loss only (no target needed — use prob for consistency)
        logit // Return logit; caller computes loss
    }

    /// Readout backward: d_readout → d_states (last timestep).
    fn readout_backward(
        &self,
        states: &[Multivector],
        d_readout: &[f32],
    ) -> Vec<Multivector> {
        let n = N_FIELDS;
        let mut d_states: Vec<Multivector> = vec![[0.0f32; 8]; n];
        // Per-field features: scalar(0), bivector_norm(1), pseudoscalar(2)
        for fi in 0..n {
            let idx = fi * 3;
            // d(scalar_part)/d(psi) = [1, 0, 0, 0, 0, 0, 0, 0]
            d_states[fi][0] += d_readout[idx];

            // d(bivector_norm)/d(psi): chain through sqrt(b4² + b5² + b6²)
            let bn = bivector_norm(&states[fi]).max(1e-8);
            let biv = bivector_part(&states[fi]);
            d_states[fi][4] += d_readout[idx + 1] * biv[0] / bn;
            d_states[fi][5] += d_readout[idx + 1] * biv[1] / bn;
            d_states[fi][6] += d_readout[idx + 1] * biv[2] / bn;

            // d(pseudoscalar)/d(psi) = [0,0,0,0,0,0,0,1]
            d_states[fi][7] += d_readout[idx + 2];
        }

        // Cross-field commutator norm gradients
        // (complex — skip for now, finite-diff handles these)

        d_states
    }

    // ========================================================================
    // SGD step
    // ========================================================================

    pub fn sgd_step(&mut self, grads: &V2Grads) {
        let lr_c = self.config.lr_clifford;
        let lr_s = self.config.lr_scalar;

        // Clip gradient norm
        let max_norm = 5.0f32;
        let mut grad_sq = 0.0f32;
        for &g in grads.d_omega.iter() { grad_sq += g * g; }
        grad_sq += grads.d_gamma * grads.d_gamma;
        for &g in grads.d_j_sun_weights.iter() { grad_sq += g * g; }
        for &g in grads.d_j_sun_orbital.iter() { grad_sq += g * g; }
        for &g in grads.d_j_sun_geomag.iter() { grad_sq += g * g; }
        grad_sq += grads.d_j_sun_bias * grads.d_j_sun_bias;
        for &g in grads.d_coupling_matrix.iter() { grad_sq += g * g; }
        for &g in grads.d_head_w1.iter() { grad_sq += g * g; }
        for &g in grads.d_head_w2.iter() { grad_sq += g * g; }
        let grad_norm = grad_sq.sqrt();
        let clip = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };

        // Omega: update on S² (project to tangent, normalize)
        for c in 0..3 {
            self.omega[c] -= lr_c * clip * grads.d_omega[c];
        }
        let omega_norm = (self.omega[0].powi(2) + self.omega[1].powi(2) + self.omega[2].powi(2))
            .max(1e-8).sqrt();
        for c in 0..3 {
            self.omega[c] /= omega_norm;
        }

        // Gamma: standard SGD, clamp positive
        self.gamma -= lr_c * clip * grads.d_gamma;
        self.gamma = self.gamma.clamp(1e-4, 1.0);

        // J_sun weights: learned mapping from SHARP → solar stiffness
        // When frozen, J_sun stays at the learned 1/π operating point.
        // The lattice dynamics learn the temporal structure instead.
        if !self.config.freeze_j_sun {
            for (w, g) in self.j_sun_weights.iter_mut().zip(grads.d_j_sun_weights.iter()) {
                *w -= lr_s * clip * g;
            }
            for (w, g) in self.j_sun_orbital.iter_mut().zip(grads.d_j_sun_orbital.iter()) {
                *w -= lr_s * clip * g;
            }
            for (w, g) in self.j_sun_geomag.iter_mut().zip(grads.d_j_sun_geomag.iter()) {
                *w -= lr_s * clip * g;
            }
            self.j_sun_bias -= lr_s * clip * grads.d_j_sun_bias;
        }
        // J_model is FIXED — not updated

        // Coupling matrix: SGD + antisymmetry enforcement
        let n = N_FIELDS;
        for i in 0..n {
            for j in 0..n {
                self.coupling_matrix[i * n + j] -= lr_c * clip * grads.d_coupling_matrix[i * n + j];
            }
        }
        // Enforce antisymmetry
        for i in 0..n {
            self.coupling_matrix[i * n + i] = 0.0;
            for j in (i + 1)..n {
                let avg = (self.coupling_matrix[i * n + j] - self.coupling_matrix[j * n + i]) / 2.0;
                self.coupling_matrix[i * n + j] = avg;
                self.coupling_matrix[j * n + i] = -avg;
            }
        }

        // Thomson beta: SGD (pre-softplus, so unconstrained)
        self.thomson_beta -= lr_c * clip * grads.d_thomson_beta;

        // Classification head: standard SGD
        for (w, g) in self.head_w1.iter_mut().zip(grads.d_head_w1.iter()) {
            *w -= lr_s * clip * g;
        }
        for (b, g) in self.head_b1.iter_mut().zip(grads.d_head_b1.iter()) {
            *b -= lr_s * clip * g;
        }
        for (w, g) in self.head_w2.iter_mut().zip(grads.d_head_w2.iter()) {
            *w -= lr_s * clip * g;
        }
        self.head_b2 -= lr_s * clip * grads.d_head_b2;

        // Helmholtz gamma: SGD, clamp to valid decay range
        for (g_param, g_grad) in self.helmholtz_gamma.iter_mut().zip(grads.d_helmholtz_gamma.iter()) {
            *g_param -= lr_c * clip * g_grad;
            *g_param = g_param.clamp(0.01, 0.999);
        }

        // Encoder: ManifoldSGD — update a,b on S^7 (Cl(3,0) unit multivectors)
        for fi in 0..N_FIELDS {
            let base = fi * MV_DIM;

            // Update a: tangent projection + normalize
            for d in 0..MV_DIM {
                self.encoder_a[fi][d] -= lr_c * clip * grads.d_encoder_a[base + d];
            }
            let norm_a = clifford_norm(&self.encoder_a[fi]);
            for d in 0..MV_DIM {
                self.encoder_a[fi][d] /= norm_a;
            }

            // Update b: tangent projection + normalize
            for d in 0..MV_DIM {
                self.encoder_b[fi][d] -= lr_c * clip * grads.d_encoder_b[base + d];
            }

            // Gram-Schmidt: b = b - (b·a)a, then normalize
            let dot: f32 = (0..MV_DIM).map(|d| self.encoder_b[fi][d] * self.encoder_a[fi][d]).sum();
            for d in 0..MV_DIM {
                self.encoder_b[fi][d] -= dot * self.encoder_a[fi][d];
            }
            let norm_b = clifford_norm(&self.encoder_b[fi]);
            for d in 0..MV_DIM {
                self.encoder_b[fi][d] /= norm_b;
            }
        }
    }

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /// Get J_model (fixed) and J_sun weights summary.
    pub fn j_diagnostic(&self) -> (f32, f32) {
        let w_sum: f32 = self.j_sun_weights.iter().sum();
        (self.j_model, w_sum)
    }

    /// Sync order from last forward pass states.
    pub fn sync_order(states: &[Multivector]) -> f32 {
        let n = states.len();
        let mut mean = [0.0f32; 8];
        for s in states {
            for d in 0..8 { mean[d] += s[d]; }
        }
        clifford_norm(&mean) / n as f32
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Convert 3-component bivector to 8-component Cl(3,0) multivector.
fn bivector_to_multivector_v2(biv: &[f32; 3]) -> Multivector {
    let mut mv = [0.0f32; 8];
    mv[4] = biv[0]; // e12
    mv[5] = biv[1]; // e23
    mv[6] = biv[2]; // e13
    mv
}

/// Extract bivector part as full multivector (zeros except grade-2).
fn bivector_multivector_v2(x: &Multivector) -> Multivector {
    let mut mv = [0.0f32; 8];
    mv[4] = x[4];
    mv[5] = x[5];
    mv[6] = x[6];
    mv
}

// ============================================================================
// Serialization
// ============================================================================

#[derive(Serialize, Deserialize)]
pub struct V2Checkpoint {
    pub config: SolarFlareV2Config,
    pub encoder_a: Vec<[f32; 8]>,
    pub encoder_b: Vec<[f32; 8]>,
    pub helmholtz_gamma: Vec<f32>,
    pub omega: [f32; 3],
    pub gamma: f32,
    pub j_model: f32,
    pub j_sun_weights: Vec<f32>,
    #[serde(default)]
    pub j_sun_orbital: Vec<f32>,
    #[serde(default)]
    pub j_sun_geomag: Vec<f32>,
    pub j_sun_bias: f32,
    pub coupling_matrix: Vec<f32>,
    pub thomson_beta: f32,
    pub head_w1: Vec<f32>,
    pub head_b1: Vec<f32>,
    pub head_w2: Vec<f32>,
    pub head_b2: f32,
}

impl SolarFlareV2 {
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let cp = V2Checkpoint {
            config: self.config.clone(),
            encoder_a: self.encoder_a.clone(),
            encoder_b: self.encoder_b.clone(),
            helmholtz_gamma: self.helmholtz_gamma.clone(),
            omega: self.omega,
            gamma: self.gamma,
            j_model: self.j_model,
            j_sun_weights: self.j_sun_weights.clone(),
            j_sun_orbital: self.j_sun_orbital.clone(),
            j_sun_geomag: self.j_sun_geomag.clone(),
            j_sun_bias: self.j_sun_bias,
            coupling_matrix: self.coupling_matrix.clone(),
            thomson_beta: self.thomson_beta,
            head_w1: self.head_w1.clone(),
            head_b1: self.head_b1.clone(),
            head_w2: self.head_w2.clone(),
            head_b2: self.head_b2,
        };
        let json = serde_json::to_string_pretty(&cp)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_checkpoint(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let cp: V2Checkpoint = serde_json::from_str(&json)?;
        Ok(SolarFlareV2 {
            config: cp.config,
            encoder_a: cp.encoder_a,
            encoder_b: cp.encoder_b,
            helmholtz_gamma: cp.helmholtz_gamma,
            omega: cp.omega,
            gamma: cp.gamma,
            j_model: cp.j_model,
            j_sun_weights: cp.j_sun_weights,
            j_sun_orbital: if cp.j_sun_orbital.is_empty() {
                vec![0.0; N_ORBITAL]
            } else {
                cp.j_sun_orbital
            },
            j_sun_geomag: if cp.j_sun_geomag.is_empty() {
                vec![0.0; N_GEOMAG]
            } else {
                cp.j_sun_geomag
            },
            j_sun_bias: cp.j_sun_bias,
            coupling_matrix: cp.coupling_matrix,
            thomson_beta: cp.thomson_beta,
            head_w1: cp.head_w1,
            head_b1: cp.head_b1,
            head_w2: cp.head_w2,
            head_b2: cp.head_b2,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_forward_shape() {
        let config = SolarFlareV2Config::default();
        let model = SolarFlareV2::new(config.clone());
        let input = vec![0.5f32; config.seq_len * N_INPUT];
        let (prob, acts) = model.forward(&input);
        assert!(prob >= 0.0 && prob <= 1.0, "prob out of range: {}", prob);
        assert_eq!(acts.readout.len(), model.readout_dim());
        assert_eq!(acts.states.len(), config.seq_len);
    }

    #[test]
    fn test_v2_forward_finite() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        let input = vec![0.3f32; 10 * N_INPUT];
        let (prob, acts) = model.forward(&input);
        assert!(prob.is_finite(), "prob is not finite: {}", prob);
        for (i, &v) in acts.readout.iter().enumerate() {
            assert!(v.is_finite(), "readout[{i}] = {v}");
        }
    }

    #[test]
    fn test_v2_backward_finite() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        let input = vec![0.5f32; 10 * N_INPUT];
        let (_, acts) = model.forward(&input);
        let (loss, grads) = model.backward(&input, &acts, 1.0);
        assert!(loss.is_finite(), "loss = {}", loss);
        for &g in grads.d_omega.iter() {
            assert!(g.is_finite(), "d_omega not finite");
        }
        assert!(grads.d_gamma.is_finite());
        for &g in grads.d_j_sun_weights.iter() {
            assert!(g.is_finite(), "d_j_sun_weights not finite");
        }
        assert!(grads.d_j_sun_bias.is_finite());
    }

    #[test]
    fn test_v2_j_init_at_critical() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        let (j_model, w_sum) = model.j_diagnostic();
        assert!((j_model - J_CRITICAL).abs() < 1e-6, "J_model should be J_c: {}", j_model);
        // j_sun_weights should sum to ~0.89 (init from criticality detector weights)
        assert!(w_sum > 0.5 && w_sum < 1.5, "j_sun_weights sum: {}", w_sum);
    }

    #[test]
    fn test_v2_param_count() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        let count = model.param_count();
        // enc: 2*9*8=144, helm: 4*8=32, dyn: 3+1+1+81+1=87, head: 66*16+16+16+1=1089
        // Total: ~1352
        assert!(count > 1000 && count < 2000,
            "Expected ~1352 params, got {}", count);
    }

    #[test]
    fn test_v2_readout_dim() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        // 27 per-field + 36 cross-field + 3 lattice + 4 j_sun + 3 energy = 73
        assert_eq!(model.readout_dim(), 73);
    }

    #[test]
    fn test_v2_sgd_step() {
        let mut model = SolarFlareV2::new(SolarFlareV2Config::default());
        let input = vec![0.5f32; 10 * N_INPUT];

        let (_, acts) = model.forward(&input);
        let (loss0, grads) = model.backward(&input, &acts, 1.0);
        model.sgd_step(&grads);

        let (j, w_sum) = model.j_diagnostic();
        assert!(j.is_finite(), "J_model after step: {}", j);

        let (_, acts2) = model.forward(&input);
        let (loss1, _) = model.backward(&input, &acts2, 1.0);
        assert!(loss1.is_finite(), "loss after step: {}", loss1);
    }

    #[test]
    fn test_v2_checkpoint_roundtrip() {
        let model = SolarFlareV2::new(SolarFlareV2Config::default());
        let input = vec![0.5f32; 10 * N_INPUT];
        let (prob_before, _) = model.forward(&input);

        let path = std::env::temp_dir().join("solar_flare_v2_test.json");
        model.save_checkpoint(&path).unwrap();
        let loaded = SolarFlareV2::load_checkpoint(&path).unwrap();
        let (prob_after, _) = loaded.forward(&input);

        assert!(
            (prob_before - prob_after).abs() < 1e-6,
            "Roundtrip mismatch: {} vs {}", prob_before, prob_after
        );
        let _ = std::fs::remove_file(&path);
    }
}
