//! CriticalLearningModel — near-critical Clifford lattice with explicit J control.
//!
//! This crate is intentionally independent from the simulator-facing model stack.
//! It imports lawful Clifford primitives from `harmonic-core` and builds a new
//! learning lineage around explicit criticality control.

use harmonic_core::clifford_cl3::{
    bivector_multivector, bivector_norm, bivector_part, bivector_to_multivector, byte_to_clifford,
    clifford_commutator, clifford_normalize, init_clifford_state, scalar_vector_ratio,
    BivectorDefectCache, Multivector,
};
use std::f32::consts::PI;
use std::path::Path;

pub const J_CRITICAL: f32 = 2.0 / PI;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum CouplingTopology {
    Ring,
    Complete,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CriticalLearningConfig {
    pub vocab_size: usize,
    pub n_sites: usize,
    pub dt: f32,
    pub gamma_init: f32,
    pub j_init: f32,
    pub inhibition_threshold_init: f32,
    pub discharge_gain_init: f32,
    pub standing_wave_amplitude_init: f32,
    pub standing_wave_cycles_init: f32,
    pub temporal_harmonic_amplitude_init: f32,
    pub temporal_harmonic_frequency_init: f32,
    pub forcing_gain_init: f32,
    pub readout_temperature_init: f32,
    pub topology: CouplingTopology,
    pub init_scalar_sign: f32,
    pub cache_size: usize,
    pub cache_threshold: f32,
}

impl Default for CriticalLearningConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            n_sites: 8,
            dt: 0.1,
            gamma_init: 0.05,
            j_init: J_CRITICAL,
            inhibition_threshold_init: 0.5,
            discharge_gain_init: 0.75,
            standing_wave_amplitude_init: 0.0,
            standing_wave_cycles_init: 1.0,
            temporal_harmonic_amplitude_init: 0.0,
            temporal_harmonic_frequency_init: 4.0,
            forcing_gain_init: 1.0,
            readout_temperature_init: 1.0,
            topology: CouplingTopology::Ring,
            init_scalar_sign: 1.0,
            cache_size: 256,
            cache_threshold: 0.7,
        }
    }
}

pub struct CriticalLearningModel {
    pub omega: [f32; 3],
    pub gamma: f32,
    pub j: f32,
    pub inhibition_threshold: f32,
    pub discharge_gain: f32,
    pub standing_wave_amplitude: f32,
    pub standing_wave_cycles: f32,
    pub temporal_harmonic_amplitude: f32,
    pub temporal_harmonic_frequency: f32,
    pub forcing_gain: f32,
    pub readout_temperature: f32,
    pub config: CriticalLearningConfig,
    pub defect_cache: BivectorDefectCache,
}

impl CriticalLearningModel {
    pub fn new(config: CriticalLearningConfig) -> Self {
        Self {
            omega: [0.0, 0.0, 1.0],
            gamma: config.gamma_init,
            j: config.j_init,
            inhibition_threshold: config.inhibition_threshold_init,
            discharge_gain: config.discharge_gain_init,
            standing_wave_amplitude: config.standing_wave_amplitude_init,
            standing_wave_cycles: config.standing_wave_cycles_init,
            temporal_harmonic_amplitude: config.temporal_harmonic_amplitude_init,
            temporal_harmonic_frequency: config.temporal_harmonic_frequency_init,
            forcing_gain: config.forcing_gain_init,
            readout_temperature: config.readout_temperature_init,
            defect_cache: BivectorDefectCache::new(config.cache_size, config.cache_threshold),
            config,
        }
    }

    pub fn n_params(&self) -> usize {
        13
    }

    pub fn init_states(&self) -> Vec<Multivector> {
        (0..self.config.n_sites)
            .map(|_| init_clifford_state(self.config.init_scalar_sign))
            .collect()
    }

    fn aggregate_state(&self, states: &[Multivector]) -> Multivector {
        let mut acc = [0.0f32; 8];
        for state in states {
            for i in 0..8 {
                acc[i] += state[i];
            }
        }
        let scale = 1.0 / states.len().max(1) as f32;
        for val in &mut acc {
            *val *= scale;
        }
        clifford_normalize(&acc)
    }

    fn sync_order(&self, states: &[Multivector]) -> f32 {
        if states.is_empty() {
            return 0.0;
        }
        let mut mean = [0.0f32; 3];
        for state in states {
            let b = bivector_part(state);
            let n = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).max(1e-8).sqrt();
            mean[0] += b[0] / n;
            mean[1] += b[1] / n;
            mean[2] += b[2] / n;
        }
        let inv = 1.0 / states.len() as f32;
        let norm =
            ((mean[0] * inv).powi(2) + (mean[1] * inv).powi(2) + (mean[2] * inv).powi(2)).sqrt();
        norm.clamp(0.0, 1.0)
    }

    fn site_dispersion(&self, states: &[Multivector]) -> f32 {
        if states.len() < 2 {
            return 0.0;
        }
        let agg = self.aggregate_state(states);
        let agg_b = bivector_part(&agg);
        let agg_n = (agg_b[0] * agg_b[0] + agg_b[1] * agg_b[1] + agg_b[2] * agg_b[2])
            .max(1e-8)
            .sqrt();
        let mut total = 0.0;
        for state in states {
            let b = bivector_part(state);
            let n = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).max(1e-8).sqrt();
            let cos = (b[0] * agg_b[0] + b[1] * agg_b[1] + b[2] * agg_b[2]) / (n * agg_n);
            total += 1.0 - cos.clamp(-1.0, 1.0);
        }
        total / states.len() as f32
    }

    fn local_alignment(&self, left: &Multivector, psi: &Multivector, right: &Multivector) -> f32 {
        fn cos_biv(a: &Multivector, b: &Multivector) -> f32 {
            let a_b = bivector_part(a);
            let b_b = bivector_part(b);
            let a_n = (a_b[0] * a_b[0] + a_b[1] * a_b[1] + a_b[2] * a_b[2])
                .max(1e-8)
                .sqrt();
            let b_n = (b_b[0] * b_b[0] + b_b[1] * b_b[1] + b_b[2] * b_b[2])
                .max(1e-8)
                .sqrt();
            let cos = (a_b[0] * b_b[0] + a_b[1] * b_b[1] + a_b[2] * b_b[2]) / (a_n * b_n);
            cos.clamp(-1.0, 1.0)
        }

        let pair_cos = 0.5 * (cos_biv(left, psi) + cos_biv(right, psi));
        0.5 * (pair_cos + 1.0)
    }

    fn standing_wave_factor(&self, site_idx: usize, n_sites: usize) -> f32 {
        if n_sites == 0 {
            return 1.0;
        }
        let phase = 2.0 * PI * self.standing_wave_cycles * site_idx as f32 / n_sites as f32;
        1.0 + self.standing_wave_amplitude * phase.sin()
    }

    fn temporal_harmonic_factor(&self, step_idx: usize) -> f32 {
        let phase = 2.0 * PI * self.temporal_harmonic_frequency * step_idx as f32 * self.config.dt;
        1.0 + self.temporal_harmonic_amplitude * phase.sin()
    }

    fn coupling_terms(&self, states: &[Multivector], i: usize) -> ([f32; 8], [f32; 8], f32) {
        let n_sites = states.len();
        let psi = &states[i];
        match self.config.topology {
            CouplingTopology::Ring => {
                let left = &states[(i + n_sites - 1) % n_sites];
                let right = &states[(i + 1) % n_sites];
                (
                    clifford_commutator(left, psi),
                    clifford_commutator(right, psi),
                    self.local_alignment(left, psi, right),
                )
            }
            CouplingTopology::Complete => {
                let mut total = [0.0f32; 8];
                let mut align_sum = 0.0f32;
                let mut count = 0usize;
                for (j, other) in states.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let comm = clifford_commutator(other, psi);
                    for k in 0..8 {
                        total[k] += comm[k];
                    }
                    let a = self.local_alignment(other, psi, other);
                    align_sum += a;
                    count += 1;
                }
                if count > 0 {
                    let inv = 1.0 / count as f32;
                    for val in &mut total {
                        *val *= inv;
                    }
                    (total, total, align_sum * inv)
                } else {
                    ([0.0; 8], [0.0; 8], 0.0)
                }
            }
        }
    }

    pub fn step(
        &self,
        states: &[Multivector],
        byte_val: u8,
        step_idx: usize,
    ) -> (Vec<Multivector>, CriticalStepStats) {
        let omega_mv = bivector_to_multivector(&self.omega);
        let kick = byte_to_clifford(byte_val);
        let kick_scale = self.forcing_gain / (self.config.n_sites.max(1) as f32).sqrt();
        let n_sites = states.len();
        let temporal_harmonic = self.temporal_harmonic_factor(step_idx);
        let mut next_states = Vec::with_capacity(n_sites);
        let mut total_delta_norm = 0.0f32;
        let mut total_discharge = 0.0f32;
        let mut inhibited_count = 0usize;

        for i in 0..n_sites {
            let psi = &states[i];
            let standing_wave = self.standing_wave_factor(i, n_sites);
            let modulation = standing_wave * temporal_harmonic;
            let comm_local = clifford_commutator(&omega_mv, psi);
            let (comm_left, comm_right, local_alignment) = self.coupling_terms(states, i);
            let psi_biv = bivector_multivector(psi);
            let accumulated_drive =
                self.j * local_alignment + modulation.abs() * kick_scale * bivector_norm(&kick);
            let threshold = self.inhibition_threshold.max(1e-4);
            let discharge = ((accumulated_drive - threshold) / threshold)
                .max(0.0)
                .tanh();
            let inhibited = accumulated_drive < threshold;

            let mut next = [0.0f32; 8];
            let mut delta_sq = 0.0f32;
            for k in 0..8 {
                let coupling = 0.5 * (comm_left[k] + comm_right[k]);
                let gated_kick =
                    (0.25 + 0.75 * local_alignment) * modulation * kick_scale * kick[k];
                let discharge_term = self.discharge_gain * discharge * coupling;
                let delta = modulation * comm_local[k] + self.j * coupling + discharge_term
                    - self.gamma.max(1e-6) * psi_biv[k]
                    + gated_kick;
                next[k] = psi[k] + self.config.dt * delta;
                delta_sq += delta * delta;
            }
            total_delta_norm += delta_sq.sqrt();
            total_discharge += discharge;
            inhibited_count += usize::from(inhibited);
            next_states.push(clifford_normalize(&next));
        }

        let sync = self.sync_order(&next_states);
        let dispersion = self.site_dispersion(&next_states);
        let adaptation_rate = (total_delta_norm / n_sites.max(1) as f32).tanh();
        let critical_gap = (self.j - J_CRITICAL).abs();
        let coherence_adaptability_balance = (1.0 - (sync - adaptation_rate).abs()).clamp(0.0, 1.0);
        let discharge_rate = total_discharge / n_sites.max(1) as f32;
        let inhibited_fraction = inhibited_count as f32 / n_sites.max(1) as f32;

        (
            next_states,
            CriticalStepStats {
                sync_order: sync,
                dispersion,
                adaptation_rate,
                critical_gap,
                coherence_adaptability_balance,
                discharge_rate,
                inhibited_fraction,
            },
        )
    }

    pub fn forward(&mut self, input: &[u8]) -> (Vec<Vec<f32>>, CriticalLearningDiagnostics) {
        let mut states = self.init_states();
        let mut logits_seq = Vec::with_capacity(input.len());
        let mut sync_order = Vec::with_capacity(input.len());
        let mut dispersion = Vec::with_capacity(input.len());
        let mut adaptation_rate = Vec::with_capacity(input.len());
        let mut balance = Vec::with_capacity(input.len());
        let mut discharge_rate = Vec::with_capacity(input.len());
        let mut inhibited_fraction = Vec::with_capacity(input.len());
        let mut bivector_norms = Vec::with_capacity(input.len());
        let mut scalar_vector_ratios = Vec::with_capacity(input.len());

        for (step_idx, &byte) in input.iter().enumerate() {
            let (next_states, step_stats) = self.step(&states, byte, step_idx);
            let aggregate = self.aggregate_state(&next_states);
            let mut logits = self.defect_cache.logits(&aggregate, self.config.vocab_size);
            let temp = self.readout_temperature.max(1e-3);
            for val in &mut logits {
                *val *= temp;
            }
            self.defect_cache.update(&aggregate, byte);

            bivector_norms.push(bivector_norm(&aggregate));
            scalar_vector_ratios.push(scalar_vector_ratio(&aggregate));
            sync_order.push(step_stats.sync_order);
            dispersion.push(step_stats.dispersion);
            adaptation_rate.push(step_stats.adaptation_rate);
            balance.push(step_stats.coherence_adaptability_balance);
            discharge_rate.push(step_stats.discharge_rate);
            inhibited_fraction.push(step_stats.inhibited_fraction);
            logits_seq.push(logits);
            states = next_states;
        }

        (
            logits_seq,
            CriticalLearningDiagnostics {
                sync_order,
                dispersion,
                adaptation_rate,
                bivector_norm: bivector_norms,
                scalar_vector_ratio: scalar_vector_ratios,
                coherence_adaptability_balance: balance,
                boundary_crossing_rate: discharge_rate.clone(),
                suppression_fraction: inhibited_fraction.clone(),
                discharge_rate,
                inhibited_fraction,
                critical_gap: (self.j - J_CRITICAL).abs(),
                j: self.j,
                j_critical: J_CRITICAL,
                gamma: self.gamma,
                omega: self.omega,
                topology: self.config.topology,
                inhibition_threshold: self.inhibition_threshold,
                discharge_gain: self.discharge_gain,
                standing_wave_amplitude: self.standing_wave_amplitude,
                standing_wave_cycles: self.standing_wave_cycles,
                temporal_harmonic_amplitude: self.temporal_harmonic_amplitude,
                temporal_harmonic_frequency: self.temporal_harmonic_frequency,
                forcing_gain: self.forcing_gain,
                readout_temperature: self.readout_temperature,
                cache_fill: self.defect_cache.fill(),
            },
        )
    }

    pub fn sequence_loss(&mut self, input: &[u8], targets: &[u8]) -> (f32, Vec<Vec<f32>>) {
        let (logits, _) = self.forward(input);
        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        for (t, target) in targets.iter().enumerate() {
            if t < logits.len() {
                let target_logit = logits[t][*target as usize];
                let target_sim = ((target_logit + 1.0) * 0.5).clamp(1e-8, 1.0);
                total_loss += -target_sim.ln();
                count += 1;
            }
        }
        let avg_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };
        (avg_loss, logits)
    }

    pub fn save_checkpoint(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let config_json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(dir.join("config.json"), config_json)?;

        let params = CriticalLearningCheckpoint {
            omega: self.omega,
            gamma: self.gamma,
            j: self.j,
            inhibition_threshold: self.inhibition_threshold,
            discharge_gain: self.discharge_gain,
            standing_wave_amplitude: self.standing_wave_amplitude,
            standing_wave_cycles: self.standing_wave_cycles,
            temporal_harmonic_amplitude: self.temporal_harmonic_amplitude,
            temporal_harmonic_frequency: self.temporal_harmonic_frequency,
            forcing_gain: self.forcing_gain,
            readout_temperature: self.readout_temperature,
            cache_bivectors: self.defect_cache.bivectors.clone(),
            cache_bytes: self.defect_cache.bytes.clone(),
        };
        let params_json = serde_json::to_string_pretty(&params)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(dir.join("params.json"), params_json)?;
        Ok(())
    }

    pub fn load_checkpoint(dir: &Path) -> std::io::Result<Self> {
        let config_str = std::fs::read_to_string(dir.join("config.json"))?;
        let config: CriticalLearningConfig = serde_json::from_str(&config_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let params_str = std::fs::read_to_string(dir.join("params.json"))?;
        let params: CriticalLearningCheckpoint = serde_json::from_str(&params_str)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut model = Self::new(config);
        model.omega = params.omega;
        model.gamma = params.gamma;
        model.j = params.j;
        model.inhibition_threshold = params.inhibition_threshold;
        model.discharge_gain = params.discharge_gain;
        model.standing_wave_amplitude = params.standing_wave_amplitude;
        model.standing_wave_cycles = params.standing_wave_cycles;
        model.temporal_harmonic_amplitude = params.temporal_harmonic_amplitude;
        model.temporal_harmonic_frequency = params.temporal_harmonic_frequency;
        model.forcing_gain = params.forcing_gain;
        model.readout_temperature = params.readout_temperature;
        model.defect_cache.bivectors = params.cache_bivectors;
        model.defect_cache.bytes = params.cache_bytes;
        Ok(model)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CriticalLearningCheckpoint {
    omega: [f32; 3],
    gamma: f32,
    j: f32,
    inhibition_threshold: f32,
    discharge_gain: f32,
    standing_wave_amplitude: f32,
    standing_wave_cycles: f32,
    temporal_harmonic_amplitude: f32,
    temporal_harmonic_frequency: f32,
    forcing_gain: f32,
    readout_temperature: f32,
    cache_bivectors: Vec<[f32; 3]>,
    cache_bytes: Vec<u8>,
}

pub struct CriticalStepStats {
    pub sync_order: f32,
    pub dispersion: f32,
    pub adaptation_rate: f32,
    pub critical_gap: f32,
    pub coherence_adaptability_balance: f32,
    /// Biological wording: coordinated release above threshold.
    /// KT/Coulomb wording: boundary-crossing or relevant-activation proxy.
    pub discharge_rate: f32,
    /// Biological wording: inhibited assembly fraction.
    /// KT/Coulomb wording: suppression or irrelevant-mode proxy.
    pub inhibited_fraction: f32,
}

pub struct CriticalLearningDiagnostics {
    pub sync_order: Vec<f32>,
    pub dispersion: Vec<f32>,
    pub adaptation_rate: Vec<f32>,
    pub bivector_norm: Vec<f32>,
    pub scalar_vector_ratio: Vec<f32>,
    pub coherence_adaptability_balance: Vec<f32>,
    pub boundary_crossing_rate: Vec<f32>,
    pub suppression_fraction: Vec<f32>,
    pub discharge_rate: Vec<f32>,
    pub inhibited_fraction: Vec<f32>,
    pub critical_gap: f32,
    pub j: f32,
    pub j_critical: f32,
    pub gamma: f32,
    pub omega: [f32; 3],
    pub topology: CouplingTopology,
    pub inhibition_threshold: f32,
    pub discharge_gain: f32,
    pub standing_wave_amplitude: f32,
    pub standing_wave_cycles: f32,
    pub temporal_harmonic_amplitude: f32,
    pub temporal_harmonic_frequency: f32,
    pub forcing_gain: f32,
    pub readout_temperature: f32,
    pub cache_fill: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmonic_core::clifford_cl3::{bivector_part, clifford_commutator, clifford_normalize};

    #[test]
    fn test_critical_learning_forward_finite() {
        let mut model = CriticalLearningModel::new(CriticalLearningConfig::default());
        let input = b"criticality";
        let (logits, diag) = model.forward(input);
        assert_eq!(logits.len(), input.len());
        assert_eq!(diag.sync_order.len(), input.len());
        assert!((diag.j_critical - J_CRITICAL).abs() < 1e-6);
        assert_eq!(diag.boundary_crossing_rate.len(), input.len());
        assert_eq!(diag.suppression_fraction.len(), input.len());
        for step in logits {
            for val in step {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_critical_learning_checkpoint_roundtrip() {
        let model = CriticalLearningModel::new(CriticalLearningConfig::default());
        let dir = std::env::temp_dir().join("critical_learning_ckpt_test");
        let _ = std::fs::remove_dir_all(&dir);
        model.save_checkpoint(&dir).unwrap();
        let loaded = CriticalLearningModel::load_checkpoint(&dir).unwrap();
        assert_eq!(loaded.n_params(), 13);
        assert!((loaded.j - J_CRITICAL).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_commutator_antisymmetry_property() {
        let a = clifford_normalize(&[0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05]);
        let b = clifford_normalize(&[-0.2, 0.6, 0.1, -0.4, 0.3, 0.5, -0.2, 0.1]);
        let ab = clifford_commutator(&a, &b);
        let ba = clifford_commutator(&b, &a);
        for i in 0..8 {
            assert!(
                (ab[i] + ba[i]).abs() < 1e-5,
                "commutator antisymmetry failed at index {i}: {} + {}",
                ab[i],
                ba[i]
            );
        }
    }

    #[test]
    fn test_bivector_grade_discipline() {
        let biv_a = [0.0, 0.0, 0.0, 0.0, 0.7, -0.2, 0.4, 0.0];
        let biv_b = [0.0, 0.0, 0.0, 0.0, -0.1, 0.6, 0.3, 0.0];
        let comm = clifford_commutator(&biv_a, &biv_b);

        assert!(
            comm[0].abs() < 1e-5,
            "scalar leakage from bivector commutator"
        );
        assert!(comm[1].abs() < 1e-5, "e1 leakage from bivector commutator");
        assert!(comm[2].abs() < 1e-5, "e2 leakage from bivector commutator");
        assert!(comm[3].abs() < 1e-5, "e3 leakage from bivector commutator");
        assert!(
            comm[7].abs() < 1e-5,
            "trivector leakage from bivector commutator"
        );

        let biv = bivector_part(&comm);
        let biv_norm_sq = biv[0] * biv[0] + biv[1] * biv[1] + biv[2] * biv[2];
        assert!(
            biv_norm_sq > 0.0,
            "bivector commutator should remain nontrivial"
        );
    }

    #[test]
    fn test_damping_acts_on_bivector_sector_only() {
        let mut model = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 1,
            gamma_init: 0.2,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        model.omega = [0.0, 0.0, 0.0];
        let psi = clifford_normalize(&[0.6, 0.2, -0.1, 0.15, 0.5, -0.25, 0.35, 0.18]);
        let states = vec![psi];

        let (next_states, _) = model.step(&states, 0, 0);
        let next = next_states[0];

        let psi_biv = harmonic_core::clifford_cl3::bivector_multivector(&psi);
        let mut expected = [0.0f32; 8];
        for i in 0..8 {
            expected[i] = psi[i] + model.config.dt * (-model.gamma * psi_biv[i]);
        }
        let expected = clifford_normalize(&expected);

        for i in 0..8 {
            assert!(
                (next[i] - expected[i]).abs() < 1e-5,
                "damping placement mismatch at index {i}: {} vs {}",
                next[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_j_sweep_regime_separation_smoke() {
        let low = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 12,
            j_init: 0.05,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        let crit = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 12,
            j_init: J_CRITICAL,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        let high = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 12,
            j_init: 1.5,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });

        let states: Vec<Multivector> = (0..12)
            .map(|i| {
                clifford_normalize(&[
                    0.7 + 0.02 * i as f32,
                    0.05 * (i as f32),
                    -0.03 * (i as f32),
                    0.02 * ((i % 3) as f32),
                    0.25 + 0.08 * ((i % 4) as f32),
                    -0.2 + 0.05 * ((i % 5) as f32),
                    0.15 - 0.04 * ((i % 6) as f32),
                    0.03 * ((i % 2) as f32),
                ])
            })
            .collect();

        let (_, low_stats) = low.step(&states, 0, 0);
        let (_, crit_stats) = crit.step(&states, 0, 0);
        let (_, high_stats) = high.step(&states, 0, 0);

        assert!(crit_stats.critical_gap <= low_stats.critical_gap + 1e-8);
        assert!(crit_stats.critical_gap <= high_stats.critical_gap + 1e-8);
        assert!(
            (high_stats.dispersion - low_stats.dispersion).abs() > 1e-4
                || (crit_stats.dispersion - low_stats.dispersion).abs() > 1e-4
                || (high_stats.adaptation_rate - low_stats.adaptation_rate).abs() > 1e-4
                || (crit_stats.adaptation_rate - low_stats.adaptation_rate).abs() > 1e-4,
            "J sweep did not produce measurable regime separation: \
             low_disp={}, crit_disp={}, high_disp={}, \
             low_adapt={}, crit_adapt={}, high_adapt={}",
            low_stats.dispersion,
            crit_stats.dispersion,
            high_stats.dispersion,
            low_stats.adaptation_rate,
            crit_stats.adaptation_rate,
            high_stats.adaptation_rate
        );
    }

    #[test]
    fn test_inhibition_threshold_gates_discharge() {
        let states: Vec<Multivector> = (0..6)
            .map(|_| clifford_normalize(&[1.0, 0.0, 0.0, 0.0, 0.35, 0.1, -0.05, 0.0]))
            .collect();

        let low_threshold = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 6,
            j_init: J_CRITICAL,
            forcing_gain_init: 0.2,
            inhibition_threshold_init: 0.2,
            ..CriticalLearningConfig::default()
        });
        let high_threshold = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 6,
            j_init: J_CRITICAL,
            forcing_gain_init: 0.2,
            inhibition_threshold_init: 2.0,
            ..CriticalLearningConfig::default()
        });

        let (_, low_stats) = low_threshold.step(&states, 97, 0);
        let (_, high_stats) = high_threshold.step(&states, 97, 0);

        assert!(
            low_stats.discharge_rate > high_stats.discharge_rate,
            "lower inhibition threshold should permit more discharge: low={}, high={}",
            low_stats.discharge_rate,
            high_stats.discharge_rate
        );
        assert!(
            low_stats.inhibited_fraction < high_stats.inhibited_fraction,
            "higher inhibition threshold should inhibit more sites: low={}, high={}",
            low_stats.inhibited_fraction,
            high_stats.inhibited_fraction
        );
    }

    #[test]
    fn test_theorem_language_aliases_match_biological_aliases() {
        let mut model = CriticalLearningModel::new(CriticalLearningConfig::default());
        let (_, diag) = model.forward(b"kt");
        assert_eq!(diag.boundary_crossing_rate, diag.discharge_rate);
        assert_eq!(diag.suppression_fraction, diag.inhibited_fraction);
    }

    #[test]
    fn test_topology_and_standing_wave_change_dynamics() {
        let states: Vec<Multivector> = (0..10)
            .map(|i| {
                clifford_normalize(&[
                    0.8 + 0.01 * i as f32,
                    0.03 * i as f32,
                    -0.02 * i as f32,
                    0.01 * ((i % 3) as f32),
                    0.2 + 0.07 * ((i % 4) as f32),
                    -0.15 + 0.05 * ((i % 5) as f32),
                    0.1 - 0.03 * ((i % 6) as f32),
                    0.02 * ((i % 2) as f32),
                ])
            })
            .collect();

        let ring = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 10,
            topology: CouplingTopology::Ring,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        let complete = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 10,
            topology: CouplingTopology::Complete,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        let standing = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 10,
            standing_wave_amplitude_init: 0.6,
            standing_wave_cycles_init: 2.0,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });
        let standing_phi = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 10,
            standing_wave_amplitude_init: 0.6,
            standing_wave_cycles_init: 1.618_034,
            forcing_gain_init: 0.0,
            ..CriticalLearningConfig::default()
        });

        let (_, ring_stats) = ring.step(&states, 0, 0);
        let (_, complete_stats) = complete.step(&states, 0, 0);
        let (_, standing_stats) = standing.step(&states, 0, 0);
        let (_, phi_stats) = standing_phi.step(&states, 0, 0);

        assert!(
            (ring_stats.discharge_rate - complete_stats.discharge_rate).abs() > 1e-5
                || (ring_stats.dispersion - complete_stats.dispersion).abs() > 1e-5
                || (ring_stats.adaptation_rate - complete_stats.adaptation_rate).abs() > 1e-5,
            "topology should measurably affect dynamics"
        );
        assert!(
            (standing_stats.discharge_rate - ring_stats.discharge_rate).abs() > 1e-5
                || (standing_stats.dispersion - ring_stats.dispersion).abs() > 1e-5
                || (standing_stats.adaptation_rate - ring_stats.adaptation_rate).abs() > 1e-5,
            "standing-wave modulation should measurably affect dynamics"
        );
        assert!(
            (phi_stats.discharge_rate - standing_stats.discharge_rate).abs() > 1e-5
                || (phi_stats.dispersion - standing_stats.dispersion).abs() > 1e-5
                || (phi_stats.adaptation_rate - standing_stats.adaptation_rate).abs() > 1e-5,
            "standing-wave cycle should measurably affect dynamics"
        );
    }

    #[test]
    fn test_temporal_harmonic_changes_dynamics_over_time() {
        let states: Vec<Multivector> = (0..8)
            .map(|i| {
                clifford_normalize(&[
                    0.9 + 0.01 * i as f32,
                    0.02 * i as f32,
                    -0.01 * i as f32,
                    0.01 * ((i % 2) as f32),
                    0.18 + 0.04 * ((i % 4) as f32),
                    -0.12 + 0.03 * ((i % 3) as f32),
                    0.08 - 0.02 * ((i % 5) as f32),
                    0.01 * ((i % 2) as f32),
                ])
            })
            .collect();

        let static_drive = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 8,
            temporal_harmonic_amplitude_init: 0.0,
            forcing_gain_init: 0.1,
            ..CriticalLearningConfig::default()
        });
        let theta_drive = CriticalLearningModel::new(CriticalLearningConfig {
            n_sites: 8,
            temporal_harmonic_amplitude_init: 0.6,
            temporal_harmonic_frequency_init: 6.0,
            forcing_gain_init: 0.1,
            ..CriticalLearningConfig::default()
        });

        let (_, static_t0) = static_drive.step(&states, 97, 0);
        let (_, static_t1) = static_drive.step(&states, 97, 1);
        let (_, theta_t0) = theta_drive.step(&states, 97, 0);
        let (_, theta_t1) = theta_drive.step(&states, 97, 1);

        assert!(
            (static_t0.adaptation_rate - static_t1.adaptation_rate).abs() < 1e-5
                && (static_t0.discharge_rate - static_t1.discharge_rate).abs() < 1e-5,
            "zero temporal harmonic should not change step-to-step modulation"
        );
        assert!(
            (theta_t0.adaptation_rate - theta_t1.adaptation_rate).abs() > 1e-5
                || (theta_t0.discharge_rate - theta_t1.discharge_rate).abs() > 1e-5,
            "temporal harmonic should create time-dependent modulation"
        );
    }
}
