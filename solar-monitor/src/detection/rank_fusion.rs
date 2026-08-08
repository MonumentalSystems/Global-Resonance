use chrono::{DateTime, Utc};
use serde::Serialize;

use super::criticality::CriticalityDetector;
use super::cusum::CusumDetector;
use super::hardness::HardnessRatioDetector;
use super::multichannel::MultichannelDetector;
use super::proton::ProtonDetector;
use super::rate_of_change::RateOfChangeDetector;
use super::zscore::ZScoreDetector;
use super::FlareOnset;
use crate::feeds::xray::FlareClass;

pub const N_DETECTORS: usize = 7;

/// Rank fusion of 6 OOD/anomaly detectors.
///
/// Inspired by the unsupervised rank fusion from the Kuramoto Layered OOD
/// experiments (0.9762 AUROC, no OOD training data needed).
///
/// Each detector produces a score in [0, 1]. Rank fusion:
/// 1. Collects scores from all N detectors.
/// 2. Ranks each score relative to its own recent history (percentile rank).
/// 3. Computes a weighted mean of percentile ranks.
///
/// Weights are derived from detector selectivity (TP fire rate / FP fire rate)
/// measured on real GOES 7-day data with the X1.5 event (2026-03-30).
pub struct RankFusionDetector {
    /// Individual detectors.
    pub zscore: ZScoreDetector,
    pub cusum: CusumDetector,
    pub hardness: HardnessRatioDetector,
    pub rate: RateOfChangeDetector,
    pub multichannel: MultichannelDetector,
    pub proton: ProtonDetector,
    pub criticality: CriticalityDetector,

    /// Score history per detector for percentile ranking.
    score_histories: [ScoreHistory; N_DETECTORS],

    /// Detector weights (selectivity-derived).
    /// Order: zscore, cusum, hardness, rate_of_change, multichannel, proton, criticality
    weights: [f64; N_DETECTORS],

    /// Fused score.
    fused_score: f64,
    /// Per-detector percentile ranks (for diagnostics).
    percentile_ranks: [f64; N_DETECTORS],

    /// Current state.
    current_flux: f64,
    current_time: Option<DateTime<Utc>>,
    /// Threshold for fused score to trigger alert.
    alert_threshold: f64,
    /// Minimum detector agreement (score > 0.3) required for alert.
    /// Eliminates FPs where only one noisy detector fires.
    min_agreement: usize,
    /// Whether each detector received the channels required for this observation.
    available: [bool; N_DETECTORS],
}

/// Names of the 7 detectors (for diagnostics).
pub const DETECTOR_NAMES: [&str; N_DETECTORS] = [
    "zscore",
    "cusum",
    "hardness",
    "rate_of_change",
    "multichannel",
    "proton",
    "criticality",
];

/// Diagnostics from the fused detector.
#[derive(Debug, Clone, Serialize)]
pub struct FusionDiagnostics {
    /// Fused score (0..1).
    pub fused_score: f64,
    /// Is the fused detector in alert state?
    pub alert: bool,
    /// Per-detector raw scores.
    pub raw_scores: Vec<DetectorScore>,
    /// Number of detectors agreeing on anomaly (score > 0.3).
    pub detector_agreement: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectorScore {
    pub name: &'static str,
    pub raw_score: f64,
    pub percentile_rank: f64,
    pub is_anomalous: bool,
    pub available: bool,
}

/// Maintains a sliding window of scores for percentile ranking.
struct ScoreHistory {
    values: Vec<f64>,
    capacity: usize,
    write_pos: usize,
    count: usize,
}

impl ScoreHistory {
    fn new(capacity: usize) -> Self {
        Self {
            values: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            count: 0,
        }
    }

    fn push(&mut self, score: f64) {
        self.values[self.write_pos] = score;
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Percentile rank of `score` within the history (0..1).
    fn percentile_rank(&self, score: f64) -> f64 {
        if self.count < 5 {
            return score;
        }
        let below = self.values[..self.count]
            .iter()
            .filter(|&&v| v < score)
            .count();
        below as f64 / self.count as f64
    }
}

impl RankFusionDetector {
    pub fn new(alert_threshold: f64) -> Self {
        Self::with_agreement(alert_threshold, 2)
    }

    pub fn with_agreement(alert_threshold: f64, min_agreement: usize) -> Self {
        Self {
            zscore: ZScoreDetector::default_detector(),
            cusum: CusumDetector::default_detector(),
            hardness: HardnessRatioDetector::default_detector(),
            rate: RateOfChangeDetector::default_detector(),
            multichannel: MultichannelDetector::default_detector(),
            proton: ProtonDetector::default_detector(),
            criticality: CriticalityDetector::default_detector(),
            score_histories: std::array::from_fn(|_| ScoreHistory::new(500)),
            // Weights optimized for commutator-enhanced 7-detector ensemble.
            // Relative weights for the six observational channels, normalized
            // after excluding the uncalibrated criticality research diagnostic
            // from live alert decisions.
            weights: [
                0.285714, // zscore
                0.181818, // cusum
                0.181818, // hardness
                0.038961, // rate_of_change
                0.181818, // multichannel
                0.129870, // proton
                0.0,      // criticality: research diagnostic only
            ],
            fused_score: 0.0,
            percentile_ranks: [0.0; N_DETECTORS],
            current_flux: 0.0,
            current_time: None,
            alert_threshold,
            min_agreement,
            available: [false; N_DETECTORS],
        }
    }

    /// Default: alert threshold 0.7, minimum 2 detectors agreeing.
    pub fn default_detector() -> Self {
        Self::new(0.7)
    }

    /// Update Kp index before ingesting. Propagates to criticality detector.
    pub fn update_kp(&mut self, kp: f64) {
        self.criticality.update_kp(kp);
    }

    /// Multi-channel ingest. All fields required for full detector coverage.
    ///
    /// - `xray_long`: 0.1-0.8nm flux (W/m^2) — primary flare channel
    /// - `xray_short`: 0.05-0.4nm flux (W/m^2) — hard X-ray channel for hardness ratio
    /// - `electron_flux`: >=2 MeV electrons (pfu) — for multichannel correlation
    /// - `proton_flux`: >=1 MeV protons (pfu) — for SEP detection
    pub fn ingest(
        &mut self,
        xray_long: f64,
        xray_short: f64,
        electron_flux: f64,
        proton_flux: f64,
        timestamp: DateTime<Utc>,
    ) {
        self.current_flux = xray_long;
        self.current_time = Some(timestamp);

        // Skip eclipse artifacts
        if xray_long < 1e-9 {
            return;
        }
        self.available = [true; N_DETECTORS];

        // Feed all detectors
        self.zscore.ingest(xray_long, timestamp);
        self.cusum.ingest(xray_long, timestamp);
        self.hardness.ingest(xray_short, xray_long, timestamp);
        self.rate.ingest(xray_long, timestamp);
        self.multichannel
            .ingest(xray_long, electron_flux, timestamp);
        self.proton.ingest(proton_flux, xray_long, timestamp);
        self.criticality
            .ingest(xray_long, xray_short, proton_flux, timestamp);

        // Collect raw scores
        let raw_scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];

        // Update histories and compute weighted percentile ranks
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (i, &score) in raw_scores.iter().enumerate() {
            self.percentile_ranks[i] = self.score_histories[i].percentile_rank(score);
            self.score_histories[i].push(score);
            weighted_sum += self.percentile_ranks[i] * self.weights[i];
            weight_sum += self.weights[i];
        }

        self.fused_score = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    /// Full ingest with IMF B-field vector for commutator-enhanced criticality.
    ///
    /// When solar wind magnetometer data is available (DSCOVR/ACE at L1),
    /// this method feeds Bx/By/Bz to the criticality detector which computes
    /// the actual bivector commutator ||B ∧ Ḃ|| for flare prediction.
    ///
    /// All other detectors receive the same scalar inputs as `ingest()`.
    pub fn ingest_full(
        &mut self,
        xray_long: f64,
        xray_short: f64,
        electron_flux: f64,
        proton_flux: f64,
        bx: f64,
        by: f64,
        bz: f64,
        timestamp: DateTime<Utc>,
    ) {
        self.current_flux = xray_long;
        self.current_time = Some(timestamp);

        if xray_long < 1e-9 {
            return;
        }
        self.available = [true; N_DETECTORS];

        // Scalar detectors — same as ingest().
        self.zscore.ingest(xray_long, timestamp);
        self.cusum.ingest(xray_long, timestamp);
        self.hardness.ingest(xray_short, xray_long, timestamp);
        self.rate.ingest(xray_long, timestamp);
        self.proton.ingest(proton_flux, xray_long, timestamp);

        // Multichannel gets B-field for 3-channel decorrelation detection.
        self.multichannel
            .ingest_with_bfield(xray_long, electron_flux, bx, by, bz, timestamp);

        // Criticality detector gets the B-field vector for commutator computation.
        self.criticality
            .ingest_with_bfield(bx, by, bz, xray_long, proton_flux, timestamp);

        // Collect raw scores and compute fusion (same as ingest).
        let raw_scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (i, &score) in raw_scores.iter().enumerate() {
            self.percentile_ranks[i] = self.score_histories[i].percentile_rank(score);
            self.score_histories[i].push(score);
            weighted_sum += self.percentile_ranks[i] * self.weights[i];
            weight_sum += self.weights[i];
        }

        self.fused_score = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    /// Full ingest with SHARP magnetogram parameters for the criticality detector.
    ///
    /// When SDO/HMI SHARP data is available (12-min cadence, ~3h latency),
    /// the criticality detector uses the actual photospheric magnetic field
    /// topology: current helicity IS the commutator, shear fraction IS the
    /// non-planarity, field gradient IS ∇F.
    ///
    /// This is the highest-fidelity input path for flare prediction.
    /// Full ingest with all SHARP parameters + B-field.
    ///
    /// Passes the complete SharpRecord through to the criticality detector
    /// for trajectory analysis and direct commutator computation.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_with_sharp(
        &mut self,
        xray_long: f64,
        xray_short: f64,
        electron_flux: f64,
        proton_flux: f64,
        bx: f64,
        by: f64,
        bz: f64,
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
        timestamp: DateTime<Utc>,
    ) {
        self.current_flux = xray_long;
        self.current_time = Some(timestamp);

        if xray_long < 1e-9 {
            return;
        }
        self.available = [true; N_DETECTORS];

        // Scalar detectors.
        self.zscore.ingest(xray_long, timestamp);
        self.cusum.ingest(xray_long, timestamp);
        self.hardness.ingest(xray_short, xray_long, timestamp);
        self.rate.ingest(xray_long, timestamp);
        self.proton.ingest(proton_flux, xray_long, timestamp);

        // Multichannel with B-field.
        self.multichannel
            .ingest_with_bfield(xray_long, electron_flux, bx, by, bz, timestamp);

        // Criticality with full SHARP — highest-fidelity path.
        self.criticality.ingest_with_sharp_full(
            usflux, meangbz, meanjzh, totusjh, shrgt45, r_value, totpot, totusjz, savncpp, absnjzh,
            meanalp, area_acr, xray_long, timestamp,
        );

        // Collect and fuse.
        let raw_scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (i, &score) in raw_scores.iter().enumerate() {
            self.percentile_ranks[i] = self.score_histories[i].percentile_rank(score);
            self.score_histories[i].push(score);
            weighted_sum += self.percentile_ranks[i] * self.weights[i];
            weight_sum += self.weights[i];
        }

        self.fused_score = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    /// Simplified ingest for backward compatibility (no short X-ray or proton).
    /// Estimates short X-ray channel from flux level using empirical
    /// short/long ratio relationship observed in real GOES data:
    /// - Quiet sun (B-class): ratio ~0.04
    /// - C-class: ratio ~0.08-0.12
    /// - M-class: ratio ~0.15-0.22
    /// - X-class: ratio ~0.22-0.29
    /// This enables the hardness ratio detector to work on synthetic data.
    pub fn ingest_simple(&mut self, xray_flux: f64, electron_flux: f64, timestamp: DateTime<Utc>) {
        // Empirical short/long ratio as function of flux level
        let ratio = if xray_flux >= 1e-4 {
            // X-class: 0.22-0.29, scale with log
            0.22 + 0.07 * (xray_flux / 1e-4).log10().min(1.0).max(0.0)
        } else if xray_flux >= 1e-5 {
            // M-class: 0.12-0.22
            0.12 + 0.10 * (xray_flux / 1e-5).log10().min(1.0).max(0.0)
        } else if xray_flux >= 1e-6 {
            // C-class: 0.06-0.12
            0.06 + 0.06 * (xray_flux / 1e-6).log10().min(1.0).max(0.0)
        } else {
            // B-class and below: 0.04
            0.04
        };
        let xray_short = xray_flux * ratio;
        self.ingest(xray_flux, xray_short, electron_flux, 0.3, timestamp);
    }

    /// Live ingestion with explicit channel availability.
    ///
    /// Missing channels are excluded from fusion and agreement. They are never
    /// replaced with synthetic quiet values.
    pub fn ingest_available(
        &mut self,
        xray_long: f64,
        xray_short: Option<f64>,
        electron_flux: Option<f64>,
        proton_flux: Option<f64>,
        bfield: Option<(f64, f64, f64)>,
        timestamp: DateTime<Utc>,
    ) {
        self.current_flux = xray_long;
        self.current_time = Some(timestamp);
        if xray_long < 1e-9 {
            return;
        }

        self.zscore.ingest(xray_long, timestamp);
        self.cusum.ingest(xray_long, timestamp);
        self.rate.ingest(xray_long, timestamp);

        if let Some(short) = xray_short {
            self.hardness.ingest(short, xray_long, timestamp);
        }
        match (electron_flux, bfield) {
            (Some(electrons), Some((bx, by, bz))) => self
                .multichannel
                .ingest_with_bfield(xray_long, electrons, bx, by, bz, timestamp),
            (Some(electrons), None) => self.multichannel.ingest(xray_long, electrons, timestamp),
            (None, _) => {}
        }
        if let Some(protons) = proton_flux {
            self.proton.ingest(protons, xray_long, timestamp);
        }

        // Criticality is a research-only diagnostic. Update it only when every
        // scalar input is measured; it remains excluded from live fusion weight.
        if let (Some(short), Some(protons)) = (xray_short, proton_flux) {
            if let Some((bx, by, bz)) = bfield {
                self.criticality
                    .ingest_with_bfield(bx, by, bz, xray_long, protons, timestamp);
            } else {
                self.criticality
                    .ingest(xray_long, short, protons, timestamp);
            }
        }

        self.available = [
            true,
            true,
            xray_short.is_some(),
            true,
            electron_flux.is_some(),
            proton_flux.is_some(),
            xray_short.is_some() && proton_flux.is_some(),
        ];
        self.fuse_available();
    }

    fn fuse_available(&mut self) {
        let raw_scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (i, &score) in raw_scores.iter().enumerate() {
            if self.available[i] {
                self.percentile_ranks[i] = self.score_histories[i].percentile_rank(score);
                self.score_histories[i].push(score);
                weighted_sum += self.percentile_ranks[i] * self.weights[i];
                weight_sum += self.weights[i];
            } else {
                self.percentile_ranks[i] = 0.0;
            }
        }
        self.fused_score = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };
    }

    /// Fused anomaly score (0..1).
    pub fn score(&self) -> f64 {
        self.fused_score
    }

    /// Raw criticality detector score (0..1), retained for research evaluation.
    /// It is excluded from live fusion weighting and detector agreement.
    pub fn criticality_score(&self) -> f64 {
        self.criticality.score()
    }

    /// Reset criticality EMA state. Call on large time gaps (year boundaries,
    /// multi-day data outages) to prevent stale AR state from contaminating
    /// subsequent periods.
    pub fn reset_criticality_ema(&mut self) {
        self.criticality.reset_ema();
    }

    /// Two-level multiplicative score (v7): optimized for short lead times (≤6h).
    pub fn criticality_score_v7(&self) -> f64 {
        self.criticality.compute_score_v7_twolevel()
    }

    /// Is the fused detector in alert state?
    /// Requires BOTH fused score above threshold AND minimum detector agreement.
    /// This dual gate eliminates FPs where a single noisy detector drives the
    /// fused score above threshold without corroboration.
    pub fn is_anomalous(&self) -> bool {
        self.fused_score > self.alert_threshold && self.detector_agreement() >= self.min_agreement
    }

    /// Number of individual detectors with score above soft threshold (0.3).
    pub fn detector_agreement(&self) -> usize {
        let scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];
        scores[..6]
            .iter()
            .enumerate()
            .filter(|(i, score)| self.available[*i] && **score > 0.3)
            .count()
    }

    /// Full diagnostics for the current state.
    pub fn diagnostics(&self) -> FusionDiagnostics {
        let raw_scores = [
            self.zscore.score(),
            self.cusum.score(),
            self.hardness.score(),
            self.rate.score(),
            self.multichannel.score(),
            self.proton.score(),
            self.criticality.score(),
        ];
        let anomalous = [
            self.zscore.is_anomalous(),
            self.cusum.is_anomalous(),
            self.hardness.is_anomalous(),
            self.rate.is_anomalous(),
            self.multichannel.is_anomalous(),
            self.proton.is_anomalous(),
            self.criticality.is_anomalous(),
        ];

        FusionDiagnostics {
            fused_score: self.fused_score,
            alert: self.is_anomalous(),
            raw_scores: (0..N_DETECTORS)
                .map(|i| DetectorScore {
                    name: DETECTOR_NAMES[i],
                    raw_score: raw_scores[i],
                    percentile_rank: self.percentile_ranks[i],
                    is_anomalous: anomalous[i],
                    available: self.available[i],
                })
                .collect(),
            detector_agreement: self.detector_agreement(),
        }
    }

    /// If fused detector is anomalous, return onset event.
    pub fn onset_event(&self) -> Option<FlareOnset> {
        if self.is_anomalous() {
            Some(FlareOnset {
                timestamp: self.current_time?,
                class: FlareClass::from_flux(self.current_flux),
                peak_flux: self.current_flux,
                anomaly_score: self.fused_score,
            })
        } else {
            None
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
    fn test_quiet_sun_no_alert() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..200 {
            det.ingest(5e-7, 2e-8, 100.0, 0.3, ts(i));
        }
        assert!(!det.is_anomalous());
        assert!(det.score() < 0.5);
        // With 7 detectors, at most 1 may show marginal activity during
        // quiet sun (criticality detector warm-up period).
        assert!(det.detector_agreement() <= 1);
    }

    #[test]
    fn test_flare_triggers_fusion() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..200 {
            det.ingest(5e-7, 2e-8, 100.0, 0.3, ts(i));
        }
        // X-class flare with hardened spectrum and proton increase
        det.ingest(3e-4, 8e-5, 50000.0, 15.0, ts(201));
        assert!(det.detector_agreement() >= 1);
        assert!(det.score() > 0.3);
    }

    #[test]
    fn sustained_observed_flare_reaches_live_alert_gate() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..200 {
            det.ingest(5e-7, 2e-8, 100.0, 0.3, ts(i));
        }
        for i in 201..211 {
            let progress = (i - 200) as f64 / 10.0;
            let long = 5e-7 + progress * 3e-4;
            det.ingest(long, long * 0.25, 50_000.0, 15.0, ts(i));
        }
        assert!(
            det.is_anomalous(),
            "observed flare sequence should cross live gate: score={:.3}, agreement={}",
            det.score(),
            det.detector_agreement(),
        );
    }

    #[test]
    fn missing_live_channels_are_excluded_not_synthesized() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..200 {
            det.ingest_available(5e-7, None, None, None, None, ts(i));
        }
        let diag = det.diagnostics();
        assert!(diag.raw_scores[0].available);
        assert!(diag.raw_scores[1].available);
        assert!(!diag.raw_scores[2].available); // hardness requires short XRS
        assert!(diag.raw_scores[3].available);
        assert!(!diag.raw_scores[4].available); // multichannel requires electrons
        assert!(!diag.raw_scores[5].available); // proton detector requires protons
        assert!(!diag.raw_scores[6].available); // criticality requires measured inputs
        assert_eq!(diag.raw_scores[2].percentile_rank, 0.0);
        assert_eq!(diag.raw_scores[4].percentile_rank, 0.0);
        assert_eq!(diag.raw_scores[5].percentile_rank, 0.0);
    }

    #[test]
    fn test_simple_ingest() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..50 {
            det.ingest_simple(5e-7, 100.0, ts(i));
        }
        assert!(det.score() >= 0.0 && det.score() <= 1.0);
    }

    #[test]
    fn test_diagnostics_structure() {
        let mut det = RankFusionDetector::default_detector();
        for i in 0..50 {
            det.ingest(5e-7, 2e-8, 100.0, 0.3, ts(i));
        }
        let diag = det.diagnostics();
        assert_eq!(diag.raw_scores.len(), N_DETECTORS);
        assert!(diag.fused_score >= 0.0 && diag.fused_score <= 1.0);
    }

    #[test]
    fn test_percentile_rank_calibrates() {
        let mut hist = ScoreHistory::new(100);
        for i in 0..100 {
            hist.push(i as f64 / 100.0);
        }
        let rank = hist.percentile_rank(0.5);
        assert!(rank > 0.4 && rank < 0.6);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let det = RankFusionDetector::default_detector();
        let sum: f64 = det.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_score_bounded() {
        let det = RankFusionDetector::default_detector();
        assert!(det.score() >= 0.0 && det.score() <= 1.0);
    }
}
