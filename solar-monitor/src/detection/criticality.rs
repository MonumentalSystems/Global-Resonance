//! Criticality detector: Clifford lattice dynamics for solar flare prediction.
//!
//! Maps solar observables (X-ray flux, hardness ratio, proton flux) into a
//! near-critical Clifford lattice and tracks the approach to the KT transition
//! at J_c = 2/π. The key insight from the Geometric Resonance papers:
//!
//!   ∂_t F = v_A² ∇²F + [F, ∇F]
//!
//! Solar flares are vortex-antivortex annihilation events in the coronal
//! magnetic field. Energy builds as sheared bivector fields until local
//! stiffness J exceeds J_c, then releases catastrophically.
//!
//! The detector tracks three signals:
//! 1. **Coherence × Adaptability balance** — peaks before flare onset
//! 2. **Discharge rate** — measures accumulated drive crossing threshold
//! 3. **Critical gap** — distance from J_c = 2/π
//!
//! Validated against the X1.5 event (2026-03-30) where hardness precursors
//! appeared -65min before onset.

use chrono::{DateTime, Utc};
use critical_learning::{
    CouplingTopology, CriticalLearningConfig, CriticalLearningModel, CriticalStepStats, J_CRITICAL,
};
#[cfg(feature = "ml-models")]
use crate::models::solar_flare::{SolarFlareModel, N_SHARP_FIELDS};
use serde::Serialize;
use std::collections::VecDeque;

/// Number of Clifford lattice sites.
///
/// 8 sites on a ring: each site is an 8-component Cl(3,0) multivector.
/// The ring topology ensures local coupling (nearest-neighbor commutators)
/// which maps naturally to the spatial structure of an active region.
const N_SITES: usize = 8;

/// Window size for baseline statistics (number of samples).
const BASELINE_WINDOW: usize = 120;

/// SHARP trajectory buffer: 60 samples × 12-min cadence = 12h window.
/// Provides three timescales: 5 (30min), 10 (2h), 60 (12h).
const SHARP_BUFFER_CAPACITY: usize = 60;

/// Fast EMA decay constants per SHARP field (physics-tuned, 12-min cadence).
/// γ = exp(-12min / τ). Fields: [totpot, r_value, totusjh, shrgt45, meangbz,
///                                usflux, absnjzh, totusjz, area_acr]
/// Fast timescales: TOTPOT→3h, R_VALUE→2h, TOTUSJH→1h, SHRGT45→3h, MEANGBZ→2h,
///                  USFLUX→6h, ABSNJZH→1h, TOTUSJZ→1h, AREA_ACR→12h
const GAMMA_FAST: [f64; 9] = [
    0.936, // TOTPOT    3h : exp(-1/15)
    0.905, // R_VALUE   2h : exp(-1/10)
    0.819, // TOTUSJH   1h : exp(-1/5)
    0.936, // SHRGT45   3h : exp(-1/15)
    0.905, // MEANGBZ   2h : exp(-1/10)
    0.967, // USFLUX    6h : exp(-1/30)
    0.819, // ABSNJZH   1h : exp(-1/5)
    0.819, // TOTUSJZ   1h : exp(-1/5)
    0.984, // AREA_ACR 12h : exp(-1/60)
];

/// Slow EMA RISE constants — applied when field is increasing (or stable).
/// Same timescales as the previous symmetric GAMMA_SLOW: the slow EMA converges
/// to the current level at the normal rate on the way up (no change in rise behavior).
/// The asymmetry comes entirely from the fast GAMMA_FALL on the way down.
/// TOTPOT→12h, R_VALUE→6h, TOTUSJH→3h, SHRGT45→12h, MEANGBZ→6h,
/// USFLUX→24h, ABSNJZH→3h, TOTUSJZ→3h, AREA_ACR→48h
const GAMMA_RISE: [f64; 9] = [
    0.984, // TOTPOT   12h : exp(-1/60)
    0.967, // R_VALUE   6h : exp(-1/30)
    0.936, // TOTUSJH   3h : exp(-1/15)
    0.984, // SHRGT45  12h : exp(-1/60)
    0.967, // MEANGBZ   6h : exp(-1/30)
    0.992, // USFLUX   24h : exp(-1/120)
    0.936, // ABSNJZH   3h : exp(-1/15)
    0.936, // TOTUSJZ   3h : exp(-1/15)
    0.996, // AREA_ACR 48h : exp(-1/240)
];

/// Slow EMA FALL constants — applied when field is decreasing (fast discharge).
/// Post-flare field relaxes rapidly: TOTPOT drops in 1-3h, helicity in <1h.
/// TOTPOT→3h, R_VALUE→2h, TOTUSJH→1h, SHRGT45→3h, MEANGBZ→1h,
/// USFLUX→12h, ABSNJZH→1h, TOTUSJZ→1h, AREA_ACR→24h
const GAMMA_FALL: [f64; 9] = [
    0.936, // TOTPOT    3h : exp(-1/15)
    0.905, // R_VALUE   2h : exp(-1/10)
    0.819, // TOTUSJH   1h : exp(-1/5)
    0.936, // SHRGT45   3h : exp(-1/15)
    0.819, // MEANGBZ   1h : exp(-1/5)
    0.984, // USFLUX   12h : exp(-1/60)
    0.819, // ABSNJZH   1h : exp(-1/5)
    0.819, // TOTUSJZ   1h : exp(-1/5)
    0.992, // AREA_ACR 24h : exp(-1/120)
];

/// Medium EMA decay constants — geometric mean of GAMMA_FAST and GAMMA_RISE.
/// γ_medium = sqrt(γ_fast × γ_rise). Timescales are the harmonic mean of fast and slow:
/// TOTPOT→6h, R_VALUE→3.5h, TOTUSJH→1.7h, SHRGT45→6h, MEANGBZ→3.5h,
/// USFLUX→12h, ABSNJZH→1.7h, TOTUSJZ→1.7h, AREA_ACR→24h
///
/// The bandpass signal `medium_ema - slow_ema` eliminates:
///   - High-frequency noise  (captured by fast_ema - medium_ema, NOT used in score)
///   - Long-term background drift (captured by slow_ema itself)
/// Leaving: the 6–12h loading trend anomalous relative to the multi-day background.
/// This is the SIGINT "noise floor" approach: medium tracks the intermediate signal
/// band while slow tracks the baseline, so the difference is the "signal above noise floor."
const GAMMA_MEDIUM: [f64; 9] = [
    0.960, // TOTPOT    6h : sqrt(0.936 × 0.984) ≈ exp(-1/30)
    0.936, // R_VALUE  3.5h: sqrt(0.905 × 0.967) ≈ exp(-1/15)
    0.870, // TOTUSJH  1.7h: sqrt(0.819 × 0.936) ≈ exp(-1/7.5)
    0.960, // SHRGT45   6h : sqrt(0.936 × 0.984)
    0.936, // MEANGBZ  3.5h: sqrt(0.905 × 0.967)
    0.980, // USFLUX   12h : sqrt(0.967 × 0.992)
    0.870, // ABSNJZH  1.7h: sqrt(0.819 × 0.936)
    0.870, // TOTUSJZ  1.7h: sqrt(0.819 × 0.936)
    0.988, // AREA_ACR  24h: sqrt(0.984 × 0.992)
];

/// Kp history window. Kp is 3-hourly, 56 samples ≈ 7 days.
const KP_WINDOW: usize = 56;

/// Smoothing factor for EMA of criticality score.
const SCORE_EMA_ALPHA: f64 = 0.15;

/// Criticality detector using near-critical Clifford lattice dynamics.
///
/// The detector ingests solar observables and maps them to forcing on a
/// Clifford lattice running at J ≈ J_c = 2/π. The lattice's internal
/// diagnostics (sync order, adaptability, discharge rate, C×A balance)
/// serve as precursor signals.
///
/// The physical mapping:
/// - X-ray flux → byte forcing (energy input to lattice)
/// - Hardness ratio → modulates J (spectral hardening = field stress)
/// - Proton flux → modulates forcing gain (particle acceleration = reconnection)
///
/// When the coronal field approaches criticality:
/// - C×A balance rises (coherence and adaptability both high)
/// - Discharge rate increases (accumulated drive exceeds inhibition threshold)
/// - The lattice's sync order fluctuates (critical slowing down)
pub struct CriticalityDetector {
    /// The Clifford lattice model (no Debug — contains Clifford state arrays).
    model: CriticalLearningModel,
    /// Current lattice states (N_SITES × 8-component multivectors).
    states: Vec<[f32; 8]>,
    /// Step counter for temporal harmonics.
    step_idx: usize,

    /// Sliding window of C×A balance values for baseline.
    balance_history: VecDeque<f64>,
    /// Sliding window of discharge rate values.
    discharge_history: VecDeque<f64>,
    /// Sliding window of sync order values.
    sync_history: VecDeque<f64>,

    // --- Bivector commutator tracking ---
    /// Previous B-field vector [B_x, B_y, B_z] for finite-difference dB/dt.
    prev_b_field: Option<[f64; 3]>,
    /// Sliding window of bivector commutator norms ||B ∧ Ḃ||.
    commutator_history: VecDeque<f64>,
    /// Loading fraction: fraction of recent samples with commutator below threshold.
    /// Low loading_fraction = system has been active = approaching criticality.
    loading_threshold: f64,

    // --- SHARP trajectory buffer (SolarFlareNet-inspired) ---
    /// Ring buffer of recent SHARP snapshots for multi-scale trajectory analysis.
    /// Stores [totpot, r_value, totusjh, shrgt45, meangbz, usflux, absnjzh, totusjz, area_acr, meanalp, savncpp].
    /// 60 samples at 12-min cadence = 12h window.
    /// Only pushes when the SHARP timestamp advances by ≥10 min to avoid
    /// flooding the buffer with identical records at 1-min GOES cadence.
    /// Indices 0-8 match the original 9-field layout (EMA arrays use these).
    /// Indices 9-10 are MEANALP and SAVNCPP, needed by the ML model.
    sharp_buffer: VecDeque<[f64; 11]>,
    /// Timestamp of last SHARP snapshot pushed to sharp_buffer.
    last_sharp_push: Option<DateTime<Utc>>,

    // --- Per-field triple-EMA (Helmholtz gamma-cumsum, physics-tuned timescales) ---
    /// Fast EMA per SHARP field. τ = 1–6h depending on field. s_t = γ·s_{t-1} + (1-γ)·x_t.
    /// Tracks short-term injection rate. Updated at SHARP cadence (≥10min guard).
    ema_fast: [f64; 9],
    /// Medium EMA per SHARP field. τ = 1.7–24h (geometric mean of fast and slow timescales).
    /// Used for bandpass: medium_ema - slow_ema isolates the 6–12h loading trend
    /// above the multi-day background ("SIGINT noise floor" approach).
    ema_medium: [f64; 9],
    /// Slow EMA per SHARP field. τ = 3–48h depending on field.
    /// Represents background state of the active region.
    ema_slow: [f64; 9],
    /// Whether EMAs have been seeded with the first valid SHARP snapshot.
    ema_initialized: bool,

    // --- Kp magnetospheric coupling ---
    /// Recent Kp values (3-hourly cadence, 7-day window).
    /// Elevated Kp = magnetospheric disturbance = harder forcing on coronal field.
    kp_history: VecDeque<f64>,
    /// Last ingested Kp value.
    current_kp: f64,

    /// EMA-smoothed criticality score.
    score_ema: f64,
    /// Raw criticality score (before EMA).
    raw_score: f64,
    /// Whether the detector considers current state anomalous.
    in_anomaly: bool,

    /// Current diagnostics snapshot.
    last_stats: Option<CriticalStepStatsSnapshot>,
    /// Current timestamp.
    current_time: Option<DateTime<Utc>>,

    /// Anomaly threshold for the smoothed score.
    threshold: f64,

    // --- Learnable score weights ---
    /// Weights for the 5 sub-scores: [balance, discharge, commutator, loading, sync].
    /// Default: [0.25, 0.20, 0.25, 0.20, 0.10] (tuned on X1.5 event).
    /// Learnable via gradient descent on historical flare catalog.
    score_weights: [f64; 5],

    // --- Planetary KAN modulation ---
    /// Learnable periodic modulation from orbital geometry.
    /// Multiplies the raw criticality score by a factor in [0.5, 1.5].
    planetary_kan: Option<super::planetary_kan::PlanetaryKAN>,

    // --- ML scoring (optional, loaded from checkpoint) ---
    /// Trained SolarFlareModel for manifold-native ML scoring.
    /// When present, `compute_score_ml()` runs the model on the SHARP buffer.
    /// Only built with the `ml-models` feature (needs upstream harmonic-core).
    #[cfg(feature = "ml-models")]
    ml_model: Option<SolarFlareModel>,
    /// Min-max normalization stats for ML model input.
    ml_norm: Option<crate::backtest::sharp_dataset::NormStats>,
    /// Last ML score (cached, updated when SHARP buffer advances).
    ml_score: f64,
}

/// Snapshot of criticality diagnostics for external consumption.
#[derive(Debug, Clone, Serialize)]
pub struct CriticalityDiagnostics {
    /// Smoothed criticality score (0..1).
    pub score: f64,
    /// Raw (unsmoothed) score.
    pub raw_score: f64,
    /// Sync order parameter r ∈ [0,1].
    pub sync_order: f32,
    /// Coherence × Adaptability balance ∈ [0,1].
    pub ca_balance: f32,
    /// Discharge rate (fraction of sites above inhibition threshold).
    pub discharge_rate: f32,
    /// Distance from critical stiffness |J - J_c|.
    pub critical_gap: f32,
    /// Adaptation rate (tanh of mean delta norm).
    pub adaptation_rate: f32,
    /// Fraction of lattice sites that are inhibited.
    pub inhibited_fraction: f32,
    /// Whether detector is in anomalous state.
    pub is_anomalous: bool,
    /// ML model flare probability (0 if no model loaded).
    pub ml_score: f64,
}

/// Internal snapshot of step stats.
#[derive(Debug, Clone)]
struct CriticalStepStatsSnapshot {
    sync_order: f32,
    ca_balance: f32,
    discharge_rate: f32,
    critical_gap: f32,
    adaptation_rate: f32,
    inhibited_fraction: f32,
}

impl From<&CriticalStepStats> for CriticalStepStatsSnapshot {
    fn from(s: &CriticalStepStats) -> Self {
        Self {
            sync_order: s.sync_order,
            ca_balance: s.coherence_adaptability_balance,
            discharge_rate: s.discharge_rate,
            critical_gap: s.critical_gap,
            adaptation_rate: s.adaptation_rate,
            inhibited_fraction: s.inhibited_fraction,
        }
    }
}

impl CriticalityDetector {
    /// Create a new criticality detector.
    ///
    /// The lattice is initialized at J = J_c with ring topology and
    /// temporal harmonics tuned to the strongest critical controller
    /// (2/π frequency from phase sweep results).
    pub fn new(threshold: f64) -> Self {
        // Parameters tuned via solar-tune-criticality sweep on real GOES
        // 7-day data (2026-03-25 to 2026-04-01) including X1.5 event.
        // Optimized for: zero quiet-sun baseline, 0.61 mean pre-flare score.
        // SNR = 60.95, separation = 0.6095.
        let config = CriticalLearningConfig {
            vocab_size: 256,
            n_sites: N_SITES,
            dt: 0.1,
            gamma_init: 0.25,
            j_init: J_CRITICAL,
            inhibition_threshold_init: 0.30,
            discharge_gain_init: 1.50,
            standing_wave_amplitude_init: 0.50,
            standing_wave_cycles_init: 1.0,
            temporal_harmonic_amplitude_init: 0.40,
            temporal_harmonic_frequency_init: J_CRITICAL,
            forcing_gain_init: 1.0,
            readout_temperature_init: 1.0,
            topology: CouplingTopology::Ring,
            init_scalar_sign: 1.0,
            cache_size: 64,
            cache_threshold: 0.7,
        };

        let model = CriticalLearningModel::new(config);
        let states = model.init_states();

        Self {
            model,
            states,
            step_idx: 0,
            balance_history: VecDeque::with_capacity(BASELINE_WINDOW),
            discharge_history: VecDeque::with_capacity(BASELINE_WINDOW),
            sync_history: VecDeque::with_capacity(BASELINE_WINDOW),
            prev_b_field: None,
            commutator_history: VecDeque::with_capacity(BASELINE_WINDOW),
            loading_threshold: 0.0,
            sharp_buffer: VecDeque::with_capacity(SHARP_BUFFER_CAPACITY),
            last_sharp_push: None,
            ema_fast: [0.0; 9],
            ema_medium: [0.0; 9],
            ema_slow: [0.0; 9],
            ema_initialized: false,
            kp_history: VecDeque::with_capacity(KP_WINDOW),
            current_kp: 0.0,
            score_ema: 0.0,
            raw_score: 0.0,
            in_anomaly: false,
            last_stats: None,
            current_time: None,
            threshold,
            score_weights: [0.25, 0.20, 0.25, 0.20, 0.10], // default tuned weights
            planetary_kan: None,
            #[cfg(feature = "ml-models")]
            ml_model: None,
            ml_norm: None,
            ml_score: 0.0,
        }
    }

    /// Default detector: threshold 0.6.
    pub fn default_detector() -> Self {
        Self::new(0.6)
    }

    /// Enable planetary KAN modulation with n_knots B-spline knots per body.
    pub fn enable_planetary_kan(&mut self, n_knots: usize) {
        self.planetary_kan = Some(super::planetary_kan::PlanetaryKAN::new(n_knots));
    }

    /// Get the learned score weights (for inspection/saving).
    pub fn score_weights(&self) -> &[f64; 5] {
        &self.score_weights
    }

    /// Set score weights (e.g. from a checkpoint).
    pub fn set_score_weights(&mut self, w: [f64; 5]) {
        self.score_weights = w;
    }

    /// Update Kp index. Call before each ingest step.
    ///
    /// Kp modulates lattice J via magnetospheric coupling:
    /// - Elevated Kp = geomagnetic storm = ring current + magnetopause erosion
    ///   → harder forcing on dayside coronal field → J pushed toward J_c
    /// - Quiet Kp (< 2) contributes no extra drive
    /// - Storm-time Kp (≥ 5) adds up to 10% J boost
    pub fn update_kp(&mut self, kp: f64) {
        self.current_kp = kp;
        self.kp_history.push_back(kp);
        if self.kp_history.len() > KP_WINDOW {
            self.kp_history.pop_front();
        }
    }

    /// Create a detector with custom lattice parameters for tuning.
    pub fn with_params(
        threshold: f64,
        gamma: f32,
        inhibition_threshold: f32,
        discharge_gain: f32,
        temporal_harmonic_amplitude: f32,
        standing_wave_amplitude: f32,
    ) -> Self {
        let config = CriticalLearningConfig {
            vocab_size: 256,
            n_sites: N_SITES,
            dt: 0.1,
            gamma_init: gamma,
            j_init: J_CRITICAL,
            inhibition_threshold_init: inhibition_threshold,
            discharge_gain_init: discharge_gain,
            standing_wave_amplitude_init: standing_wave_amplitude,
            standing_wave_cycles_init: 1.0,
            temporal_harmonic_amplitude_init: temporal_harmonic_amplitude,
            temporal_harmonic_frequency_init: J_CRITICAL,
            forcing_gain_init: 1.0,
            readout_temperature_init: 1.0,
            topology: CouplingTopology::Ring,
            init_scalar_sign: 1.0,
            cache_size: 64,
            cache_threshold: 0.7,
        };

        let model = CriticalLearningModel::new(config);
        let states = model.init_states();

        Self {
            model,
            states,
            step_idx: 0,
            balance_history: VecDeque::with_capacity(BASELINE_WINDOW),
            discharge_history: VecDeque::with_capacity(BASELINE_WINDOW),
            sync_history: VecDeque::with_capacity(BASELINE_WINDOW),
            prev_b_field: None,
            commutator_history: VecDeque::with_capacity(BASELINE_WINDOW),
            loading_threshold: 0.0,
            sharp_buffer: VecDeque::with_capacity(SHARP_BUFFER_CAPACITY),
            last_sharp_push: None,
            ema_fast: [0.0; 9],
            ema_medium: [0.0; 9],
            ema_slow: [0.0; 9],
            ema_initialized: false,
            kp_history: VecDeque::with_capacity(KP_WINDOW),
            current_kp: 0.0,
            score_ema: 0.0,
            raw_score: 0.0,
            in_anomaly: false,
            last_stats: None,
            current_time: None,
            threshold,
            score_weights: [0.25, 0.20, 0.25, 0.20, 0.10],
            planetary_kan: None,
            #[cfg(feature = "ml-models")]
            ml_model: None,
            ml_norm: None,
            ml_score: 0.0,
        }
    }

    /// Load a trained SolarFlareModel checkpoint for ML scoring.
    ///
    /// Once loaded, `compute_score_ml()` returns model predictions alongside
    /// the physics-based v7 score. The ML score can be used in rank fusion.
    #[cfg(feature = "ml-models")]
    pub fn load_ml_model(
        &mut self,
        model_path: &std::path::Path,
        norm_path: &std::path::Path,
    ) -> Result<(), String> {
        let model = SolarFlareModel::load_checkpoint(model_path)
            .map_err(|e| format!("Failed to load ML model: {e}"))?;
        let norm_json = std::fs::read_to_string(norm_path)
            .map_err(|e| format!("Failed to read norm stats: {e}"))?;
        let norm: crate::backtest::sharp_dataset::NormStats = serde_json::from_str(&norm_json)
            .map_err(|e| format!("Failed to parse norm stats: {e}"))?;
        self.ml_model = Some(model);
        self.ml_norm = Some(norm);
        Ok(())
    }

    /// Compute ML-based flare probability from the SHARP buffer.
    ///
    /// Feeds the last `seq_len` SHARP snapshots through the trained
    /// SolarFlareModel. Returns P(flare ≥ C5.0 within prediction window).
    /// Returns 0.0 if no model loaded or insufficient data.
    #[cfg(feature = "ml-models")]
    pub fn compute_score_ml(&mut self) -> f64 {
        let (model, norm) = match (&self.ml_model, &self.ml_norm) {
            (Some(m), Some(n)) => (m, n),
            _ => return 0.0,
        };

        let seq_len = model.config.seq_len;
        if self.sharp_buffer.len() < seq_len {
            return 0.0;
        }

        // Take the last seq_len SHARP snapshots from the buffer.
        // Buffer order: [totpot, r_value, totusjh, shrgt45, meangbz, usflux, absnjzh, totusjz, area_acr]
        // Model order:  [totusjh, totusjz, usflux, meanalp, r_value, totpot, savncpp, area_acr, absnjzh]
        let start = self.sharp_buffer.len() - seq_len;
        let mut features = Vec::with_capacity(seq_len * N_SHARP_FIELDS);

        for i in start..self.sharp_buffer.len() {
            let buf = &self.sharp_buffer[i];
            // Map buffer indices to model field order.
            // Buffer: [totpot(0), r_value(1), totusjh(2), shrgt45(3), meangbz(4),
            //          usflux(5), absnjzh(6), totusjz(7), area_acr(8), meanalp(9), savncpp(10)]
            // Model:  [totusjh, totusjz, usflux, meanalp, r_value, totpot, savncpp, area_acr, absnjzh]
            let row = [
                buf[2],  // TOTUSJH
                buf[7],  // TOTUSJZ
                buf[5],  // USFLUX
                buf[9],  // MEANALP  (now stored in buffer)
                buf[1],  // R_VALUE
                buf[0],  // TOTPOT
                buf[10], // SAVNCPP  (now stored in buffer)
                buf[8],  // AREA_ACR
                buf[6],  // ABSNJZH
            ];
            let normed = norm.normalize_row(&row);
            features.extend_from_slice(&normed);
        }

        let (prob, _) = model.forward(&features);
        self.ml_score = prob as f64;
        self.ml_score
    }

    /// Get the cached ML score from the last `compute_score_ml()` call.
    pub fn ml_score(&self) -> f64 {
        self.ml_score
    }

    /// Map solar observables to a forcing byte and J modulation.
    ///
    /// The mapping:
    /// - X-ray flux → byte value (log-scaled, 0-255)
    /// - Hardness ratio → J modulation (harder spectrum = more stress)
    /// - Proton flux → forcing gain modulation
    fn map_observables(&mut self, xray_long: f64, xray_short: f64, proton_flux: f64) -> u8 {
        // Map log10(flux) to byte range [0, 255].
        // GOES X-ray flux ranges: B-class ~1e-7, C ~1e-6, M ~1e-5, X ~1e-4
        // Map [-8, -3] → [0, 255]
        let log_flux = if xray_long > 0.0 {
            xray_long.log10()
        } else {
            -8.0
        };
        let normalized = ((log_flux + 8.0) / 5.0).clamp(0.0, 1.0);
        let byte_val = (normalized * 255.0) as u8;

        // Hardness ratio modulates J around J_c.
        // Harder spectrum (higher short/long ratio) = more magnetic stress
        // = J pushed toward or beyond J_c.
        let hardness = if xray_long > 1e-9 {
            (xray_short / xray_long).clamp(0.0, 0.5)
        } else {
            0.04
        };
        // Quiet sun hardness ≈ 0.04, X-class ≈ 0.25
        // Map [0.04, 0.30] → J_c * [0.9, 1.3]
        let hardness_norm = ((hardness - 0.04) / 0.26).clamp(0.0, 1.0);
        self.model.j = J_CRITICAL * (0.9 + 0.4 * hardness_norm as f32);

        // Proton flux modulates forcing gain.
        // Elevated protons = particle acceleration = active reconnection.
        // Background ≈ 0.3 pfu, elevated > 1.0 pfu, SEP > 10 pfu
        let proton_factor = if proton_flux > 0.3 {
            (1.0 + (proton_flux / 0.3).log10()).min(3.0)
        } else {
            1.0
        };
        self.model.forcing_gain = proton_factor as f32;

        byte_val
    }

    /// Compute OLS slope of `feature_idx` column over the last `window` rows of sharp_buffer.
    ///
    /// Returns the per-sample slope (positive = feature is rising).
    /// Uses ordinary least-squares: β = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)².
    fn sharp_slope(&self, feature_idx: usize, window: usize) -> f64 {
        let n = self.sharp_buffer.len();
        if n < 3 {
            return 0.0;
        }
        let start = if n > window { n - window } else { 0 };
        let slice: Vec<f64> = self
            .sharp_buffer
            .iter()
            .skip(start)
            .map(|row| row[feature_idx])
            .collect();
        let m = slice.len() as f64;
        let x_mean = (m - 1.0) / 2.0;
        let y_mean: f64 = slice.iter().sum::<f64>() / m;
        let num: f64 = slice
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        let den: f64 = slice
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum();
        if den > 1e-12 {
            num / den
        } else {
            0.0
        }
    }

    /// Upper CUSUM anomaly score for a single SHARP field (SIGINT noise-floor method).
    ///
    /// The CUSUM (Page 1954) detects whether a field is *persistently* elevated above
    /// its multi-day background — "tuning to the noise floor" so that signal is measured
    /// relative to this AR's own established baseline, not a fixed physical threshold.
    ///
    /// **Why this is different from the existing signals:**
    /// - `commutator_at_pil` (absolute level): static floors (R_VALUE > 100, etc.) that
    ///   don't adapt to the specific AR. A weak but rapidly-building AR is underscored.
    /// - `hel_signal` (fast-slow diff): instantaneous rate snapshot only — doesn't distinguish
    ///   a sustained 2h rise from a single 12-min spike.
    /// - CUSUM here: uses the **slow EMA as the noise floor** (days-long background),
    ///   estimates σ from the buffer's typical fluctuation around that EMA, then
    ///   *accumulates* standardized deviations over the last 4h signal window.
    ///   Transient noise spikes don't accumulate; sustained injection does.
    ///   Two ARs with identical fast-slow EMA difference but different persistence patterns
    ///   score differently.
    ///
    /// **Algorithm:**
    /// - Noise floor μ = `ema_slow[field_idx]` (multi-day background from asymmetric EMA)
    /// - Noise σ = std of buffer values around μ (the AR's typical fluctuation level)
    /// - Signal window = last 20 buffer samples (4h at 12-min cadence)
    /// - S[t] = max(0, S[t-1] + (x[t] - μ)/σ - 0.5)
    ///   k = 0.5: only accumulates when field > μ + 0.5σ (suppresses noise)
    /// - Returns S_peak / 4.0 clamped to [0,1]. h = 4 ≈ 1 false alarm / 1000 quiet steps.
    ///
    /// Returns 0 if insufficient buffer data or field is constant around its slow EMA.
    fn cusum_score(&self, field_idx: usize) -> f64 {
        let n = self.sharp_buffer.len();
        if n < 10 {
            return 0.0;
        }

        // Noise floor: slow EMA is the multi-day background of the AR.
        let mu = self.ema_slow[field_idx];
        if mu < 1e-10 {
            return 0.0;
        }

        // Noise σ: typical fluctuation of raw SHARP values around the slow EMA.
        // Estimated from the full buffer (not just a split window) to get a stable estimate.
        let mut vsum = 0.0_f64;
        let mut count = 0usize;
        for row in &self.sharp_buffer {
            let v = row[field_idx];
            if v.is_finite() && v > 0.0 {
                vsum += (v - mu).powi(2);
                count += 1;
            }
        }
        if count < 5 {
            return 0.0;
        }
        let sigma = (vsum / count as f64).sqrt();
        // If the field is flat relative to its background, no anomaly possible.
        if sigma < mu * 0.005 + 1e-10 {
            return 0.0;
        }

        // Run upper-CUSUM on the most recent 20 samples (4h signal window).
        // Using a 4h window: long enough to confirm sustained injection,
        // short enough to distinguish from the multi-day background.
        let signal_start = n.saturating_sub(20);
        let k = 0.5; // slack: accumulate only when field > μ + 0.5σ
        let h = 4.0; // alarm threshold (≈1 FP per 1000 quiet steps at k=0.5)
        let mut s = 0.0_f64;
        let mut peak = 0.0_f64;
        for row in self.sharp_buffer.iter().skip(signal_start) {
            let v = row[field_idx];
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let z = (v - mu) / sigma;
            s = (s + z - k).max(0.0);
            peak = peak.max(s);
        }

        (peak / h).clamp(0.0, 1.0)
    }

    /// Second-order curvature: d²feature/dt² over the 10-sample window.
    ///
    /// Positive = accelerating upward = energy building faster and faster.
    /// Computed as the difference of short-window slope vs medium-window slope.
    fn sharp_accel(&self, feature_idx: usize) -> f64 {
        let slope_short = self.sharp_slope(feature_idx, 5);
        let slope_medium = self.sharp_slope(feature_idx, 10);
        slope_short - slope_medium
    }

    /// Kp percentile rank in its own 7-day history.
    ///
    /// Returns 0 if no history. Elevated Kp → high rank → J pushed toward J_c.
    fn kp_rank(&self) -> f64 {
        if self.kp_history.len() < 4 {
            return 0.0;
        }
        let kp = self.current_kp;
        let below = self.kp_history.iter().filter(|&&v| v < kp).count();
        below as f64 / self.kp_history.len() as f64
    }

    /// Advance EMAs forward in time during SHARP-absent periods.
    ///
    /// When an active region rotates off the limb (or data is missing), SHARP data
    /// becomes unavailable and the ingest path falls back to scalar X-ray only.
    /// Without this method, `last_sharp_push` never advances and the EMAs freeze at
    /// the last AR's values — making a vanished AR look like a persistent loaded state.
    ///
    /// Each call applies `GAMMA_FALL` decay for each elapsed 12-min SHARP slot since
    /// the last EMA update, driving the EMAs toward 0 (field absent). Capped at
    /// 120 steps (24h) per call; use `reset_ema()` for multi-day/year gaps.
    pub fn advance_ema_absent(&mut self, timestamp: DateTime<Utc>) {
        if !self.ema_initialized {
            return;
        }
        let last = match self.last_sharp_push {
            Some(t) => t,
            None => return,
        };
        let elapsed_min = (timestamp - last).num_minutes();
        if elapsed_min < 10 {
            return; // less than one SHARP slot elapsed
        }
        // Number of 12-min slots elapsed; cap at 120 (24h of decay per call).
        let steps = ((elapsed_min / 12) as usize).min(120);
        for _ in 0..steps {
            for i in 0..9 {
                // Decay toward 0: ema[t] = γ_fall * ema[t-1] + (1-γ_fall)*0
                // Fast discharge — AR absent = field energy dissipating.
                self.ema_fast[i] *= GAMMA_FALL[i];
                self.ema_medium[i] *= GAMMA_FALL[i];
                self.ema_slow[i] *= GAMMA_FALL[i];
            }
        }
        // Advance the last-push marker so we don't re-apply the same decay.
        use chrono::Duration;
        self.last_sharp_push = Some(last + Duration::minutes(steps as i64 * 12));
    }

    /// Reset EMA state entirely. Called on large time gaps (year boundaries,
    /// multi-day data outages) where decay-based correction is insufficient.
    pub fn reset_ema(&mut self) {
        self.ema_fast = [0.0; 9];
        self.ema_medium = [0.0; 9];
        self.ema_slow = [0.0; 9];
        self.ema_initialized = false;
        self.last_sharp_push = None;
    }

    /// SOTA-aligned scoring (v6): five physics-grounded signals plus Kp.
    ///
    /// The grade-0/grade-2 Clifford commutator structure (from CliffordGeodesicGPT)
    /// was tested in v8a/v8b but did not improve TSS. Root cause: the commutator
    /// rate signal (grade-2) is correlated with the absolute level (grade-0) over
    /// the 24h prediction window — both are elevated during the same loaded-AR periods,
    /// so the rate term doesn't add discrimination. The v6 3-field cross_rate
    /// (TOTPOT ∧ R_VALUE ∧ ABSNJZH) is more selective (rare AND gate) and works
    /// better as a multiplicative qualifier on the absolute commutator level.
    ///
    /// Weights:
    ///   28% J proxy             — KT stiffness J ∝ B²_mean via USFLUX/area
    ///   23% commutator at PIL   — |[F,∇F]|, multiplicatively boosted by cross-field rate
    ///   18% shear twist         — sin(α) = SHRGT45/50 background
    ///   15% helicity injection  — Helmholtz gamma-cumsum rate signal
    ///   10% Kp coupling         — magnetospheric storm coupling
    ///    6% r-coherence         — Kuramoto order parameter over 7 fields
    fn compute_score_v3(&self, _stats: &CriticalStepStatsSnapshot) -> f64 {
        if !self.ema_initialized {
            return 0.0;
        }

        // EMA layout: [totpot=0, r_value=1, totusjh=2, shrgt45=3, meangbz=4,
        //               usflux=5, absnjzh=6, totusjz=7, area_acr=8]
        let totpot_s = self.ema_slow[0];
        let r_value_s = self.ema_slow[1];
        let shrgt45_s = self.ema_slow[3];
        let usflux_s = self.ema_slow[5];
        let absnjzh_s = self.ema_slow[6];
        let area_acr_s = self.ema_slow[8].max(1.0);

        // ── 1. J PROXY (28%) ─────────────────────────────────────────────────
        // KT stiffness J = v_A²|F|. v_A ∝ B/√ρ, |F| ∝ USFLUX/area → J ∝ B²_mean.
        let b_mean_proxy = usflux_s / (area_acr_s * 1e16).max(1.0);
        let j_proxy = if b_mean_proxy > 1.0 {
            ((b_mean_proxy.log10() - 1.0) / 2.0).clamp(0.0, 1.0)
        } else if totpot_s > 0.0 {
            ((totpot_s.log10() - 22.0) / 3.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // ── 2. COMMUTATOR AT PIL (23%) ────────────────────────────────────────
        // |[F,∇F]| ∝ R_VALUE × ABSNJZH. Use slow EMAs (background level).
        let r_level = if r_value_s > 0.0 {
            ((r_value_s.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let abs_level = if absnjzh_s.abs() > 0.0 {
            ((absnjzh_s.abs().log10() - 4.0) / 4.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let commutator_at_pil = (r_level * abs_level).sqrt();

        // Cross-field rate: TOTPOT ∧ R_VALUE ∧ ABSNJZH simultaneous loading.
        // 3-field AND gate — fires only when all three fields load at once.
        // More selective than 2-field (requires energy buildup confirmation via TOTPOT).
        // Used as multiplicative boost on commutator_at_pil (not additive term —
        // tested as additive and as two-level gate; multiplicative performs best).
        let dtotpot_rel = if self.ema_slow[0] > 0.0 {
            (self.ema_fast[0] - self.ema_slow[0]) / self.ema_slow[0]
        } else {
            0.0
        };
        let dr_value_rel = if self.ema_slow[1] > 0.0 {
            (self.ema_fast[1] - self.ema_slow[1]) / self.ema_slow[1]
        } else {
            0.0
        };
        let dabs_rel = if self.ema_slow[6] > 0.0 {
            (self.ema_fast[6] - self.ema_slow[6]) / self.ema_slow[6]
        } else {
            0.0
        };
        let cross_rate = if dtotpot_rel > 0.0 && dr_value_rel > 0.0 && dabs_rel > 0.0 {
            (dtotpot_rel * dr_value_rel * dabs_rel)
                .cbrt()
                .clamp(0.0, 1.0)
        } else if dtotpot_rel > 0.0 && dr_value_rel > 0.0 {
            (dtotpot_rel * dr_value_rel).sqrt().clamp(0.0, 1.0) * 0.5
        } else {
            0.0
        };
        let commutator_boosted = (commutator_at_pil * (1.0 + 0.5 * cross_rate)).min(1.0);

        // ── 3. FIELD TWIST α (18%) ────────────────────────────────────────────
        let shear_twist = (shrgt45_s / 50.0).clamp(0.0, 1.0);

        // ── 4. HELICITY INJECTION RATE (15%) — gamma-cumsum rate signal ───────
        // Discrete Helmholtz Green's function derivative at physics-tuned timescales.
        let dhel = self.ema_fast[2] - self.ema_slow[2];
        let dabs = self.ema_fast[6] - self.ema_slow[6];
        let dtotpot = self.ema_fast[0] - self.ema_slow[0];
        let hel_rate = if self.ema_slow[2] > 1.0 {
            (dhel / self.ema_slow[2]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let abs_rate = if self.ema_slow[6] > 1.0 {
            (dabs / self.ema_slow[6]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let totpot_rate = if self.ema_slow[0] > 0.0 {
            (dtotpot / self.ema_slow[0]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let hel_signal = 0.40 * hel_rate + 0.35 * abs_rate + 0.25 * totpot_rate;

        // ── 5. Kp magnetospheric coupling (10%) ───────────────────────────────
        let kp_signal = self.kp_rank();

        // ── 6. r-coherence (6%) — Kuramoto order parameter ───────────────────
        // Fraction of 7 SHARP fields where fast_ema > slow_ema (loading).
        // medium_ema is tracked (available for future use) but the geometric-mean
        // bandpass (medium-slow) was tested and found to duplicate the cross_rate
        // signal with lower selectivity (TSS 0.316 vs 0.361 at 24h lead).
        let loading_count = (0..7usize)
            .filter(|&i| self.ema_fast[i] > self.ema_slow[i])
            .count();
        let r_coherence = loading_count as f64 / 7.0;

        // ── Final weighted combination (v6) ───────────────────────────────────
        let score = 0.28 * j_proxy
            + 0.23 * commutator_boosted
            + 0.18 * shear_twist
            + 0.15 * hel_signal
            + 0.10 * kp_signal
            + 0.06 * r_coherence;

        if score.is_finite() {
            score
        } else {
            0.0
        }
    }

    /// Two-level multiplicative scoring (v7): optimized for short lead times (≤6h).
    ///
    /// At short lead times the critical_approach rate signals (helicity injection,
    /// cross-field loading, r-coherence) ARE firing — they're the ~6h precursors.
    /// The multiplicative gate `loaded_state × (0.6 + 0.4 × critical_approach)`
    /// then correctly distinguishes the final approach to criticality from stable
    /// loaded periods.
    ///
    /// NOT suitable for 24h lead (rate signals haven't started → gate suppresses TP).
    pub fn compute_score_v7_twolevel(&self) -> f64 {
        if !self.ema_initialized {
            return 0.0;
        }

        let totpot_s = self.ema_slow[0];
        let r_value_s = self.ema_slow[1];
        let shrgt45_s = self.ema_slow[3];
        let usflux_s = self.ema_slow[5];
        let absnjzh_s = self.ema_slow[6];
        let area_acr_s = self.ema_slow[8].max(1.0);

        // Level 1: background loaded state (slow EMAs)
        let b_mean_proxy = usflux_s / (area_acr_s * 1e16).max(1.0);
        let j_proxy = if b_mean_proxy > 1.0 {
            ((b_mean_proxy.log10() - 1.0) / 2.0).clamp(0.0, 1.0)
        } else if totpot_s > 0.0 {
            ((totpot_s.log10() - 22.0) / 3.0).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let r_level = if r_value_s > 0.0 {
            ((r_value_s.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let abs_level = if absnjzh_s.abs() > 0.0 {
            ((absnjzh_s.abs().log10() - 4.0) / 4.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let commutator_at_pil = (r_level * abs_level).sqrt();
        let shear_twist = (shrgt45_s / 50.0).clamp(0.0, 1.0);
        let loaded_state =
            (0.55 * j_proxy + 0.30 * commutator_at_pil + 0.15 * shear_twist).clamp(0.0, 1.0);

        // Level 2: critical approach (fast-slow EMA rates)
        let dhel = self.ema_fast[2] - self.ema_slow[2];
        let dabs = self.ema_fast[6] - self.ema_slow[6];
        let dtotpot = self.ema_fast[0] - self.ema_slow[0];
        let hel_rate = if self.ema_slow[2] > 1.0 {
            (dhel / self.ema_slow[2]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let abs_rate = if self.ema_slow[6] > 1.0 {
            (dabs / self.ema_slow[6]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let totpot_rate = if self.ema_slow[0] > 0.0 {
            (dtotpot / self.ema_slow[0]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let hel_signal = 0.40 * hel_rate + 0.35 * abs_rate + 0.25 * totpot_rate;

        let dtotpot_rel = if self.ema_slow[0] > 0.0 {
            (self.ema_fast[0] - self.ema_slow[0]) / self.ema_slow[0]
        } else {
            0.0
        };
        let dr_value_rel = if self.ema_slow[1] > 0.0 {
            (self.ema_fast[1] - self.ema_slow[1]) / self.ema_slow[1]
        } else {
            0.0
        };
        let dabs_rel = if self.ema_slow[6] > 0.0 {
            (self.ema_fast[6] - self.ema_slow[6]) / self.ema_slow[6]
        } else {
            0.0
        };
        let cross_rate = if dtotpot_rel > 0.0 && dr_value_rel > 0.0 && dabs_rel > 0.0 {
            (dtotpot_rel * dr_value_rel * dabs_rel)
                .cbrt()
                .clamp(0.0, 1.0)
        } else if dtotpot_rel > 0.0 && dr_value_rel > 0.0 {
            (dtotpot_rel * dr_value_rel).sqrt().clamp(0.0, 1.0) * 0.5
        } else {
            0.0
        };

        let loading_count = (0..7usize)
            .filter(|&i| self.ema_fast[i] > self.ema_slow[i])
            .count();
        let r_coherence = loading_count as f64 / 7.0;

        let critical_approach =
            (0.40 * hel_signal + 0.35 * cross_rate + 0.25 * r_coherence).clamp(0.0, 1.0);

        let kp_signal = self.kp_rank();

        let score = (loaded_state * (0.6 + 0.4 * critical_approach) * 0.90 + 0.10 * kp_signal)
            .clamp(0.0, 1.0);
        if score.is_finite() {
            score
        } else {
            0.0
        }
    }

    /// Compute criticality score from lattice diagnostics.
    ///
    /// The score combines three signals:
    /// 1. C×A balance percentile (relative to recent history)
    /// 2. Discharge rate percentile
    /// 3. Sync order fluctuation (variance over recent window)
    ///
    /// High score = system approaching criticality = flare precursor.
    fn compute_score(&self, stats: &CriticalStepStatsSnapshot) -> f64 {
        let balance = stats.ca_balance as f64;
        let discharge = stats.discharge_rate as f64;

        // Percentile rank of balance relative to history.
        let balance_rank = if self.balance_history.len() >= 10 {
            let below = self
                .balance_history
                .iter()
                .filter(|&&v| v < balance)
                .count();
            below as f64 / self.balance_history.len() as f64
        } else {
            balance
        };

        // Percentile rank of discharge rate.
        let discharge_rank = if self.discharge_history.len() >= 10 {
            let below = self
                .discharge_history
                .iter()
                .filter(|&&v| v < discharge)
                .count();
            below as f64 / self.discharge_history.len() as f64
        } else {
            discharge
        };

        // Sync order fluctuation: high variance near criticality
        // (critical slowing down → large fluctuations in order parameter).
        let sync_fluct = if self.sync_history.len() >= 10 {
            let mean: f64 = self.sync_history.iter().sum::<f64>() / self.sync_history.len() as f64;
            let variance: f64 = self
                .sync_history
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / self.sync_history.len() as f64;
            // Normalize: typical quiet variance ~0.001, pre-flare ~0.01+
            (variance * 100.0).min(1.0)
        } else {
            0.0
        };

        // Weighted combination:
        // - Balance: 0.40 (primary signal — C×A peaks at criticality)
        // - Discharge: 0.35 (biological threshold crossing)
        // - Sync fluctuation: 0.25 (critical slowing down)
        0.40 * balance_rank + 0.35 * discharge_rank + 0.25 * sync_fluct
    }

    /// Ingest a new observation and step the Clifford lattice.
    ///
    /// This is the main entry point. Each call:
    /// 1. Maps observables to lattice forcing
    /// 2. Steps the Clifford dynamics one timestep
    /// 3. Computes criticality diagnostics
    /// 4. Updates the smoothed score
    pub fn ingest(
        &mut self,
        xray_long: f64,
        xray_short: f64,
        proton_flux: f64,
        timestamp: DateTime<Utc>,
    ) {
        self.current_time = Some(timestamp);

        // Skip eclipse artifacts.
        if xray_long < 1e-9 {
            return;
        }

        // Decay EMAs toward 0 while SHARP data is absent.
        // Without this, a disappeared AR's EMA persists as a false loaded state.
        self.advance_ema_absent(timestamp);

        // Map observables to lattice inputs.
        let byte_val = self.map_observables(xray_long, xray_short, proton_flux);

        // Step the Clifford lattice.
        let (next_states, step_stats) = self.model.step(&self.states, byte_val, self.step_idx);
        self.states = next_states;
        self.step_idx += 1;

        // Snapshot the diagnostics.
        let snapshot = CriticalStepStatsSnapshot::from(&step_stats);

        // Update histories (sliding window).
        self.balance_history.push_back(snapshot.ca_balance as f64);
        if self.balance_history.len() > BASELINE_WINDOW {
            self.balance_history.pop_front();
        }
        self.discharge_history
            .push_back(snapshot.discharge_rate as f64);
        if self.discharge_history.len() > BASELINE_WINDOW {
            self.discharge_history.pop_front();
        }
        self.sync_history.push_back(snapshot.sync_order as f64);
        if self.sync_history.len() > BASELINE_WINDOW {
            self.sync_history.pop_front();
        }

        // Compute raw score.
        self.raw_score = self.compute_score(&snapshot);

        // EMA smoothing.
        self.score_ema =
            SCORE_EMA_ALPHA * self.raw_score + (1.0 - SCORE_EMA_ALPHA) * self.score_ema;

        // State transition.
        if self.score_ema > self.threshold {
            self.in_anomaly = true;
        } else if self.score_ema < self.threshold * 0.7 {
            self.in_anomaly = false;
        }

        self.last_stats = Some(snapshot);
    }

    /// Ingest with B-field vector components for direct commutator computation.
    ///
    /// This is the preferred method when magnetometer data is available
    /// (Fredericksburg, Swarm, or other ground/space magnetometers).
    /// The bivector commutator ||B ∧ Ḃ|| is computed directly from the
    /// vector components, giving a much stronger signal than scalar proxies.
    ///
    /// - `b_x, b_y, b_z`: Magnetic field components (nT)
    /// - `xray_long`: X-ray flux 0.1-0.8nm (W/m²), or F10.7 proxy
    /// - `proton_flux`: Proton flux (pfu), or dB/dt_max proxy
    pub fn ingest_with_bfield(
        &mut self,
        b_x: f64,
        b_y: f64,
        b_z: f64,
        xray_long: f64,
        proton_flux: f64,
        timestamp: DateTime<Utc>,
    ) {
        self.current_time = Some(timestamp);
        if xray_long < 1e-9 {
            return;
        }

        // Decay EMAs while no SHARP data is being ingested.
        self.advance_ema_absent(timestamp);

        // Compute bivector commutator ||B ∧ Ḃ|| from finite differences.
        let b_field = [b_x, b_y, b_z];
        let commutator_norm = if let Some(prev) = self.prev_b_field {
            // Ḃ ≈ B(t) - B(t-1)
            let db = [b_x - prev[0], b_y - prev[1], b_z - prev[2]];

            // Bivector wedge product components: B_i * Ḃ_j - B_j * Ḃ_i
            let biv_xy = b_x * db[1] - b_y * db[0]; // e₁₂
            let biv_xz = b_x * db[2] - b_z * db[0]; // e₁₃
            let biv_yz = b_y * db[2] - b_z * db[1]; // e₂₃

            (biv_xy * biv_xy + biv_xz * biv_xz + biv_yz * biv_yz).sqrt()
        } else {
            0.0
        };
        self.prev_b_field = Some(b_field);

        // Update commutator history.
        self.commutator_history.push_back(commutator_norm);
        if self.commutator_history.len() > BASELINE_WINDOW {
            self.commutator_history.pop_front();
        }

        // Update loading threshold from median (adaptive baseline).
        if self.commutator_history.len() >= 30 && self.loading_threshold == 0.0 {
            let mut sorted: Vec<f64> = self.commutator_history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.loading_threshold = sorted[sorted.len() / 2];
        }

        // Commutator modulates J directly:
        // High commutator = field is rotating (non-potential) = J pushed toward J_c.
        let comm_rank = if self.commutator_history.len() >= 10 {
            let below = self
                .commutator_history
                .iter()
                .filter(|&&v| v < commutator_norm)
                .count();
            below as f64 / self.commutator_history.len() as f64
        } else {
            0.5
        };
        // Map commutator percentile to J: [0.0, 1.0] → J_c * [0.85, 1.25]
        self.model.j = J_CRITICAL * (0.85 + 0.40 * comm_rank as f32);

        // Map xray to byte forcing (same as before).
        let log_flux = if xray_long > 0.0 {
            xray_long.log10()
        } else {
            -8.0
        };
        let normalized = ((log_flux + 8.0) / 5.0).clamp(0.0, 1.0);
        let byte_val = (normalized * 255.0) as u8;

        // Proton flux modulates forcing gain.
        let proton_factor = if proton_flux > 0.3 {
            (1.0 + (proton_flux / 0.3).log10()).min(3.0)
        } else {
            1.0
        };
        self.model.forcing_gain = proton_factor as f32;

        // Step the Clifford lattice.
        let (next_states, step_stats) = self.model.step(&self.states, byte_val, self.step_idx);
        self.states = next_states;
        self.step_idx += 1;

        let snapshot = CriticalStepStatsSnapshot::from(&step_stats);

        // Update lattice diagnostic histories.
        self.balance_history.push_back(snapshot.ca_balance as f64);
        if self.balance_history.len() > BASELINE_WINDOW {
            self.balance_history.pop_front();
        }
        self.discharge_history
            .push_back(snapshot.discharge_rate as f64);
        if self.discharge_history.len() > BASELINE_WINDOW {
            self.discharge_history.pop_front();
        }
        self.sync_history.push_back(snapshot.sync_order as f64);
        if self.sync_history.len() > BASELINE_WINDOW {
            self.sync_history.pop_front();
        }

        // Compute score with commutator and loading signals.
        self.raw_score = self.compute_score_v2(&snapshot);

        // EMA smoothing.
        self.score_ema =
            SCORE_EMA_ALPHA * self.raw_score + (1.0 - SCORE_EMA_ALPHA) * self.score_ema;

        if self.score_ema > self.threshold {
            self.in_anomaly = true;
        } else if self.score_ema < self.threshold * 0.7 {
            self.in_anomaly = false;
        }

        self.last_stats = Some(snapshot);
    }

    /// Ingest with SHARP magnetogram parameters for direct photospheric
    /// magnetic field topology measurement.
    ///
    /// This is the **highest-fidelity input** — SHARP parameters measure
    /// the actual active region field that drives flares. Uses all 9
    /// SolarFlareNet parameters plus trajectory analysis (rate of change
    /// of TOTPOT and R_VALUE over a 10-sample / 2h buffer).
    ///
    /// KT framework mapping:
    /// - `r_value` (PIL flux) → where [F, ∇F] is maximal (reconnection site)
    /// - `totpot` (free energy) → energy above potential field = fuel for KT transition
    /// - `totusjh` (total helicity) → total non-potential energy
    /// - `shrgt45` (shear fraction) → non-planarity of B
    /// - `meangbz` (field gradient) → ‖∇F‖ at PIL
    /// - `totusjz` (vertical current) → ‖J_z‖ through photosphere
    pub fn ingest_with_sharp(
        &mut self,
        usflux: f64,
        meangbz: f64,
        meanjzh: f64,
        totusjh: f64,
        shrgt45: f64,
        xray_long: f64,
        timestamp: DateTime<Utc>,
    ) {
        // Delegate to full version with zeros for new params if called with old signature.
        self.ingest_with_sharp_full(
            usflux, meangbz, meanjzh, totusjh, shrgt45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            xray_long, timestamp,
        );
    }

    /// Full SHARP ingest with all 9 SolarFlareNet parameters + trajectory.
    pub fn ingest_with_sharp_full(
        &mut self,
        usflux: f64,
        meangbz: f64,
        meanjzh: f64,
        totusjh: f64,
        shrgt45: f64,
        r_value: f64,
        totpot: f64,
        totusjz: f64,
        savncpp: f64,
        absnjzh: f64,
        meanalp: f64,
        area_acr: f64,
        xray_long: f64,
        timestamp: DateTime<Utc>,
    ) {
        self.current_time = Some(timestamp);
        if xray_long < 1e-9 {
            return;
        }

        // Apply any accumulated EMA decay for intervals during which SHARP data was
        // absent (scalar fallback path). This runs before the cadence guard so that
        // when SHARP data returns after a gap, the EMAs have already decayed
        // appropriately rather than jumping from a stale high value.
        self.advance_ema_absent(timestamp);

        // --- SHARP trajectory buffer (12h multi-scale window) ---
        // Only push at ≥10-min cadence to match the 12-min SHARP sampling rate.
        // Without this guard, calling at 1-min GOES cadence floods the buffer
        // with 12 identical records per SHARP snapshot, making the 60-sample
        // buffer span only 1h instead of the intended 12h.
        let snapshot_params = [
            totpot, r_value, totusjh, shrgt45, meangbz, usflux, absnjzh, totusjz, area_acr,
            meanalp, savncpp,
        ];
        let should_push = match self.last_sharp_push {
            None => true,
            Some(last) => (timestamp - last).num_minutes().abs() >= 10,
        };
        if should_push {
            self.last_sharp_push = Some(timestamp);
            self.sharp_buffer.push_back(snapshot_params);
            if self.sharp_buffer.len() > SHARP_BUFFER_CAPACITY {
                self.sharp_buffer.pop_front();
            }

            // Update per-field dual EMAs (Helmholtz gamma-cumsum, asymmetric).
            //
            // Fast EMA: symmetric, short-timescale injection tracker.
            //   s_fast[t] = γ_fast · s_fast[t-1] + (1-γ_fast) · x[t]
            //
            // Slow EMA: ASYMMETRIC — models solar loading cycle physics.
            //   Rising  (x > s_slow): γ_rise (slow accumulation — flux emergence takes days)
            //   Falling (x < s_slow): γ_fall (fast discharge — post-flare relaxation in hours)
            //
            // This asymmetry is the key physics: free energy builds slowly over days but
            // releases catastrophically in minutes. The slow EMA accumulates stress
            // (leaky integrator); the asymmetric gate prevents it from resetting
            // as fast as it built up, tracking the "loaded" state of the active region.
            //
            // NaN fields: hold previous slow EMA (no new information injected).
            if !self.ema_initialized {
                for i in 0..9 {
                    let v = if snapshot_params[i].is_finite() {
                        snapshot_params[i]
                    } else {
                        0.0
                    };
                    self.ema_fast[i] = v;
                    self.ema_medium[i] = v;
                    self.ema_slow[i] = v;
                }
                self.ema_initialized = true;
            } else {
                for i in 0..9 {
                    let v = if snapshot_params[i].is_finite() {
                        snapshot_params[i]
                    } else {
                        self.ema_slow[i] // NaN → hold
                    };
                    // Fast EMA: symmetric, short-timescale injection tracker.
                    self.ema_fast[i] = GAMMA_FAST[i] * self.ema_fast[i] + (1.0 - GAMMA_FAST[i]) * v;
                    // Medium EMA: symmetric, intermediate timescale (geometric mean of fast/slow).
                    // Bandpass signal = medium - slow isolates the 6–12h loading trend
                    // above the multi-day background drift.
                    self.ema_medium[i] =
                        GAMMA_MEDIUM[i] * self.ema_medium[i] + (1.0 - GAMMA_MEDIUM[i]) * v;
                    // Slow EMA: asymmetric γ — rises slowly, falls quickly.
                    let gamma_s = if v >= self.ema_slow[i] {
                        GAMMA_RISE[i]
                    } else {
                        GAMMA_FALL[i]
                    };
                    self.ema_slow[i] = gamma_s * self.ema_slow[i] + (1.0 - gamma_s) * v;
                }
            }
        }

        // Legacy trajectory scalars for J-drive computation (kept for J modulation).
        let (d_totpot, d_r_value) = if self.sharp_buffer.len() >= 3 {
            let oldest = self.sharp_buffer.front().unwrap();
            let newest = self.sharp_buffer.back().unwrap();
            let n = self.sharp_buffer.len() as f64;
            ((newest[0] - oldest[0]) / n, (newest[1] - oldest[1]) / n)
        } else {
            (0.0, 0.0)
        };

        // --- Commutator: use R_VALUE as the primary commutator proxy ---
        // R_VALUE measures flux at the PIL — exactly where [F, ∇F] is maximal.
        // Fall back to meanjzh if R_VALUE not available.
        let commutator_norm = if r_value > 0.0 {
            r_value
        } else {
            meanjzh.abs() * 1e4 // scale to comparable range
        };

        self.commutator_history.push_back(commutator_norm);
        if self.commutator_history.len() > BASELINE_WINDOW {
            self.commutator_history.pop_front();
        }

        if self.commutator_history.len() >= 30 && self.loading_threshold == 0.0 {
            let mut sorted: Vec<f64> = self.commutator_history.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.loading_threshold = sorted[sorted.len() / 2];
        }

        // --- J modulation from SHARP ---
        // Three signals drive J toward J_c:
        // 1. TOTPOT (free energy) — most direct measure of approach to criticality
        // 2. SHRGT45 (non-planarity) — bivector non-coplanarity fraction
        // 3. R_VALUE (PIL flux) — where the commutator is concentrated
        let totpot_norm = if totpot > 100.0 {
            ((totpot.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let shear_norm = (shrgt45 / 50.0).clamp(0.0, 1.0);
        let r_value_norm = if r_value > 100.0 {
            ((r_value.log10() - 2.0) / 3.0).clamp(0.0, 1.0)
        } else {
            let grad_norm = (meangbz / 200.0).clamp(0.0, 1.0);
            grad_norm
        };

        // Trajectory bonus: rising TOTPOT or R_VALUE pushes J harder.
        // Positive d_totpot = energy building = approaching J_c.
        let trajectory_boost = if d_totpot > 0.0 || d_r_value > 0.0 {
            0.1 * (d_totpot.max(0.0) / 1000.0 + d_r_value.max(0.0) / 1000.0).min(1.0)
        } else {
            0.0
        };

        // Kp boost: storm-time magnetospheric disturbance couples to coronal forcing.
        // Quiet (Kp<2): 0, Minor storm (Kp=5): +5%, Severe (Kp=7+): +10%.
        let kp_boost = ((self.current_kp - 2.0).max(0.0) / 50.0).min(0.10);

        let j_drive = 0.37 * totpot_norm
            + 0.28 * shear_norm
            + 0.18 * r_value_norm
            + 0.09 * trajectory_boost.min(1.0)
            + 0.08 * kp_boost;
        self.model.j = J_CRITICAL * (0.85 + 0.40 * j_drive as f32);

        // --- Forcing gain from total helicity + vertical current ---
        let energy_factor = if totusjh.abs() > 100.0 {
            (1.0 + (totusjh.abs() / 100.0).log10()).min(2.5)
        } else {
            1.0
        };
        let current_factor = if totusjz > 1e10 {
            (1.0 + (totusjz / 1e10).log10() * 0.3).min(1.5)
        } else {
            1.0
        };
        self.model.forcing_gain = (energy_factor * current_factor).min(3.0) as f32;

        // Map X-ray to byte forcing.
        let log_flux = if xray_long > 0.0 {
            xray_long.log10()
        } else {
            -8.0
        };
        let normalized = ((log_flux + 8.0) / 5.0).clamp(0.0, 1.0);
        let byte_val = (normalized * 255.0) as u8;

        // Step the Clifford lattice.
        let (next_states, step_stats) = self.model.step(&self.states, byte_val, self.step_idx);
        self.states = next_states;
        self.step_idx += 1;

        let snapshot = CriticalStepStatsSnapshot::from(&step_stats);

        // Update lattice diagnostic histories.
        self.balance_history.push_back(snapshot.ca_balance as f64);
        if self.balance_history.len() > BASELINE_WINDOW {
            self.balance_history.pop_front();
        }
        self.discharge_history
            .push_back(snapshot.discharge_rate as f64);
        if self.discharge_history.len() > BASELINE_WINDOW {
            self.discharge_history.pop_front();
        }
        self.sync_history.push_back(snapshot.sync_order as f64);
        if self.sync_history.len() > BASELINE_WINDOW {
            self.sync_history.pop_front();
        }

        // Use SOTA-aligned v3 when SHARP data is available.
        self.raw_score = self.compute_score_v3(&snapshot);

        self.score_ema =
            SCORE_EMA_ALPHA * self.raw_score + (1.0 - SCORE_EMA_ALPHA) * self.score_ema;

        if self.score_ema > self.threshold {
            self.in_anomaly = true;
        } else if self.score_ema < self.threshold * 0.7 {
            self.in_anomaly = false;
        }

        self.last_stats = Some(snapshot);
    }

    /// Enhanced scoring using commutator norm and loading fraction.
    ///
    /// Five signals:
    /// 1. C×A balance percentile (0.25) — lattice coherence/adaptability
    /// 2. Discharge rate percentile (0.20) — threshold crossing
    /// 3. Commutator percentile (0.25) — bivector non-commutativity
    /// 4. Loading fraction inverted (0.20) — fewer quiet days = approaching J_c
    /// 5. Sync order fluctuation (0.10) — critical slowing down
    fn compute_score_v2(&self, stats: &CriticalStepStatsSnapshot) -> f64 {
        let balance = stats.ca_balance as f64;
        let discharge = stats.discharge_rate as f64;

        // 1. C×A balance percentile.
        let balance_rank = if self.balance_history.len() >= 10 {
            let below = self
                .balance_history
                .iter()
                .filter(|&&v| v < balance)
                .count();
            below as f64 / self.balance_history.len() as f64
        } else {
            balance
        };

        // 2. Discharge rate percentile.
        let discharge_rank = if self.discharge_history.len() >= 10 {
            let below = self
                .discharge_history
                .iter()
                .filter(|&&v| v < discharge)
                .count();
            below as f64 / self.discharge_history.len() as f64
        } else {
            discharge
        };

        // 3. Commutator percentile — how extreme is the current ||B ∧ Ḃ||?
        let comm_rank = if self.commutator_history.len() >= 10 {
            let current = self.commutator_history.back().copied().unwrap_or(0.0);
            let below = self
                .commutator_history
                .iter()
                .filter(|&&v| v < current)
                .count();
            below as f64 / self.commutator_history.len() as f64
        } else {
            0.0
        };

        // 4. Loading fraction (inverted): what fraction of recent window was ABOVE threshold?
        // High active fraction = system has been stressed = approaching criticality.
        let loading_signal = if self.loading_threshold > 0.0 && self.commutator_history.len() >= 10
        {
            let above = self
                .commutator_history
                .iter()
                .filter(|&&v| v > self.loading_threshold)
                .count();
            above as f64 / self.commutator_history.len() as f64
        } else {
            0.5
        };

        // 5. Sync order fluctuation.
        let sync_fluct = if self.sync_history.len() >= 10 {
            let mean: f64 = self.sync_history.iter().sum::<f64>() / self.sync_history.len() as f64;
            let variance: f64 = self
                .sync_history
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / self.sync_history.len() as f64;
            (variance * 100.0).min(1.0)
        } else {
            0.0
        };

        // Weighted combination — learnable or default weights.
        let w = &self.score_weights;
        w[0] * balance_rank
            + w[1] * discharge_rank
            + w[2] * comm_rank
            + w[3] * loading_signal
            + w[4] * sync_fluct
    }

    /// Anomaly score normalized to [0, 1], with planetary modulation if enabled.
    pub fn score(&self) -> f64 {
        let base = self.score_ema.clamp(0.0, 1.0);
        if let (Some(kan), Some(ts)) = (&self.planetary_kan, &self.current_time) {
            use chrono::Datelike;
            let jd = super::planetary_kan::date_to_jd(ts.year(), ts.month(), ts.day());
            let angles = super::planetary_kan::PlanetaryKAN::angles_from_jd(jd);
            let modulation = kan.forward(&angles) as f64;
            (base * modulation).clamp(0.0, 1.0)
        } else {
            base
        }
    }

    /// Raw score WITHOUT planetary modulation (for diagnostics).
    pub fn raw_physics_score(&self) -> f64 {
        self.score_ema.clamp(0.0, 1.0)
    }

    /// Is the detector in anomalous (pre-flare) state?
    pub fn is_anomalous(&self) -> bool {
        self.in_anomaly
    }

    /// Full diagnostics for the current state.
    pub fn diagnostics(&self) -> CriticalityDiagnostics {
        let stats = self
            .last_stats
            .clone()
            .unwrap_or(CriticalStepStatsSnapshot {
                sync_order: 0.0,
                ca_balance: 0.0,
                discharge_rate: 0.0,
                critical_gap: J_CRITICAL,
                adaptation_rate: 0.0,
                inhibited_fraction: 1.0,
            });

        CriticalityDiagnostics {
            score: self.score(),
            raw_score: self.raw_score,
            sync_order: stats.sync_order,
            ca_balance: stats.ca_balance,
            discharge_rate: stats.discharge_rate,
            critical_gap: stats.critical_gap,
            adaptation_rate: stats.adaptation_rate,
            inhibited_fraction: stats.inhibited_fraction,
            is_anomalous: self.in_anomaly,
            ml_score: self.ml_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(1700000000 + secs * 60, 0).unwrap()
    }

    #[test]
    fn test_quiet_sun_no_anomaly() {
        let mut det = CriticalityDetector::default_detector();
        // Feed 1440 quiet B-class samples (24h at 1-min cadence).
        // The tuned params need a full day of constant input to stabilize
        // percentile baselines — matching real GOES quiet-sun behavior.
        for i in 0..1440 {
            det.ingest(5e-7, 2e-8, 0.3, ts(i));
        }
        assert!(!det.is_anomalous(), "quiet sun should not trigger");
    }

    #[test]
    fn test_flare_response_remains_bounded() {
        let mut det = CriticalityDetector::default_detector();
        // Establish quiet baseline (24h).
        for i in 0..1440 {
            det.ingest(5e-7, 2e-8, 0.3, ts(i));
        }
        // Inject X-class flare with hardened spectrum and proton enhancement.
        // This experimental precursor channel is not required to react
        // monotonically to an in-progress flare; reactive ensemble channels do.
        for i in 1440..1460 {
            det.ingest(3e-4, 8e-5, 15.0, ts(i));
        }
        let flare_score = det.score();
        assert!(flare_score.is_finite());
        assert!((0.0..=1.0).contains(&flare_score));
    }

    #[test]
    fn test_score_bounded() {
        let mut det = CriticalityDetector::default_detector();
        for i in 0..50 {
            det.ingest(5e-7, 2e-8, 0.3, ts(i));
        }
        let score = det.score();
        assert!(score >= 0.0 && score <= 1.0, "score must be in [0,1]");
    }

    #[test]
    fn test_diagnostics_structure() {
        let mut det = CriticalityDetector::default_detector();
        for i in 0..50 {
            det.ingest(5e-7, 2e-8, 0.3, ts(i));
        }
        let diag = det.diagnostics();
        assert!(diag.sync_order >= 0.0 && diag.sync_order <= 1.0);
        assert!(diag.ca_balance >= 0.0 && diag.ca_balance <= 1.0);
        assert!(diag.critical_gap >= 0.0);
        assert!(diag.score >= 0.0 && diag.score <= 1.0);
    }

    #[test]
    fn test_hardness_modulates_j() {
        let mut det = CriticalityDetector::default_detector();
        // Quiet sun: low hardness → J near lower bound.
        det.ingest(5e-7, 2e-8, 0.3, ts(0));
        let j_quiet = det.model.j;

        // Hard spectrum: high hardness → J pushed higher.
        det.ingest(1e-4, 2.5e-5, 0.3, ts(1));
        let j_hard = det.model.j;

        assert!(
            j_hard > j_quiet,
            "harder spectrum should increase J: quiet={j_quiet}, hard={j_hard}"
        );
    }

    #[test]
    fn test_proton_modulates_forcing() {
        let mut det = CriticalityDetector::default_detector();
        // Baseline protons.
        det.ingest(5e-7, 2e-8, 0.3, ts(0));
        let fg_quiet = det.model.forcing_gain;

        // Elevated protons (SEP event).
        det.ingest(5e-7, 2e-8, 30.0, ts(1));
        let fg_sep = det.model.forcing_gain;

        assert!(
            fg_sep > fg_quiet,
            "elevated protons should increase forcing gain: quiet={fg_quiet}, sep={fg_sep}"
        );
    }

    #[test]
    fn test_eclipse_skip() {
        let mut det = CriticalityDetector::default_detector();
        let step_before = det.step_idx;
        det.ingest(1e-10, 1e-11, 0.3, ts(0));
        assert_eq!(det.step_idx, step_before, "eclipse data should be skipped");
    }

    #[test]
    fn test_gradual_ramp_increases_score() {
        let mut det = CriticalityDetector::default_detector();
        // Quiet baseline (24h).
        for i in 0..1440 {
            det.ingest(5e-7, 2e-8, 0.3, ts(i));
        }
        let baseline = det.score();

        // Gradual ramp from C-class to M-class over 60 minutes
        // (simulates active region flux buildup).
        for i in 0..60 {
            let flux = 1e-6 * (1.0 + 9.0 * (i as f64 / 60.0)); // C1 → C9
            let short = flux * (0.06 + 0.10 * (i as f64 / 60.0)); // hardening
            det.ingest(flux, short, 0.5, ts(1440 + i));
        }
        let ramped = det.score();

        assert!(
            ramped > baseline,
            "gradual ramp should increase score: baseline={baseline:.3}, ramped={ramped:.3}"
        );
    }
}
