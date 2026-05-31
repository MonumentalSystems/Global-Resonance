//! Replay engine: runs historical data through the rank fusion detector
//! and evaluates detection performance against known flare events.

use chrono::{DateTime, Utc};
use serde::Serialize;

use super::loaders::{FlareEvent, HistoricalRecord};
use crate::coupling::StressorIndex;
use crate::detection::rank_fusion::RankFusionDetector;
use crate::feeds::xray::FlareClass;
use crate::feeds::FeedState;

/// Configuration for the backtest.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Fused detector alert threshold (0..1).
    pub alert_threshold: f64,
    /// Tolerance window: detection within this many hours of flare onset
    /// counts as a true positive.
    pub tolerance_hours: f64,
    /// Minimum flare class to evaluate (e.g., "M" skips C-class).
    pub min_class: char,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            alert_threshold: 0.7,
            tolerance_hours: 2.0,
            min_class: 'M',
        }
    }
}

/// Result of a single detection event during backtest.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub timestamp: DateTime<Utc>,
    pub fused_score: f64,
    pub detector_agreement: usize,
    pub flux: f64,
    pub class: FlareClass,
    /// Whether this detection matches a known flare (true positive).
    pub is_true_positive: bool,
    /// Matched flare class (if TP).
    pub matched_flare: Option<String>,
}

/// Summary statistics from a backtest run.
#[derive(Debug, Clone, Serialize)]
pub struct BacktestResults {
    /// Total historical records replayed.
    pub total_records: usize,
    /// Time span.
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    /// Known M/X-class flares in the period.
    pub total_flares: usize,
    /// Flares detected (within tolerance window).
    pub true_positives: usize,
    /// Flares missed.
    pub false_negatives: usize,
    /// Detections with no matching flare.
    pub false_positives: usize,
    /// Detection rate (recall).
    pub recall: f64,
    /// Precision.
    pub precision: f64,
    /// F1 score.
    pub f1: f64,
    /// Mean detection lead time (hours before flare peak, negative = early).
    pub mean_lead_time_hours: f64,
    /// All detections.
    pub detections: Vec<Detection>,
    /// Missed flares.
    pub missed_flares: Vec<MissedFlare>,
    /// Per-class breakdown.
    pub class_breakdown: Vec<ClassStats>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MissedFlare {
    pub begin: DateTime<Utc>,
    pub peak: DateTime<Utc>,
    pub class: String,
    pub max_fused_score: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassStats {
    pub class: String,
    pub total: usize,
    pub detected: usize,
    pub recall: f64,
}

/// Run a backtest: replay historical records through the detector.
pub fn run_backtest(
    records: &[HistoricalRecord],
    flares: &[FlareEvent],
    config: &BacktestConfig,
) -> BacktestResults {
    let mut detector = RankFusionDetector::new(config.alert_threshold);
    let mut stressor = StressorIndex::new();
    let mut feeds = FeedState::new();

    let mut detections: Vec<Detection> = Vec::new();
    let tolerance_secs = (config.tolerance_hours * 3600.0) as i64;

    // Replay each record
    for rec in records {
        // Update feed state for coupling pathways
        if let (Some(speed), Some(bz), Some(by), Some(density)) =
            (rec.solar_wind_speed, rec.bz, rec.by, rec.density)
        {
            feeds.append_solar_wind(vec![crate::feeds::solar_wind::SolarWindSample {
                time_tag: rec.timestamp,
                speed,
                bx: 0.0, // not in OMNI simplified
                by,
                bz,
                density,
            }]);
        }

        if let Some(kp) = rec.kp {
            feeds.append_kp_dst(vec![crate::feeds::kp_dst::KpDstSample {
                time_tag: rec.timestamp,
                kp,
                estimated_dst: rec.dst.unwrap_or(0.0),
            }]);
        }

        // Use a synthetic electron flux (correlated with X-ray for backtest)
        // Real electron data not in OMNI — approximate from Dst/AE
        let electron_flux = estimate_electron_flux(rec);

        // Feed the rank fusion detector
        detector.ingest_simple(rec.xray_flux, electron_flux, rec.timestamp);

        // Check for detection
        if detector.is_anomalous() {
            // Deduplicate: don't report if we already detected within last hour
            let dominated = detections.last().map_or(false, |last| {
                (rec.timestamp - last.timestamp).num_seconds() < 3600
            });

            if !dominated {
                let is_tp = flares.iter().any(|f| {
                    let dt = (rec.timestamp - f.begin).num_seconds().abs();
                    dt < tolerance_secs || (rec.timestamp >= f.begin && rec.timestamp <= f.end)
                });

                let matched = if is_tp {
                    flares
                        .iter()
                        .find(|f| {
                            let dt = (rec.timestamp - f.begin).num_seconds().abs();
                            dt < tolerance_secs
                                || (rec.timestamp >= f.begin && rec.timestamp <= f.end)
                        })
                        .map(|f| f.class.clone())
                } else {
                    None
                };

                detections.push(Detection {
                    timestamp: rec.timestamp,
                    fused_score: detector.score(),
                    detector_agreement: detector.detector_agreement(),
                    flux: rec.xray_flux,
                    class: FlareClass::from_flux(rec.xray_flux),
                    is_true_positive: is_tp,
                    matched_flare: matched,
                });
            }
        }

        // Update coupling pathways periodically (every hour in historical data)
        let onset = detector.onset_event();
        stressor.update(&feeds, onset.as_ref());
    }

    // Compute metrics
    let true_positives = detections.iter().filter(|d| d.is_true_positive).count();
    let false_positives = detections.iter().filter(|d| !d.is_true_positive).count();

    // Find missed flares
    let mut missed_flares = Vec::new();
    let mut detected_flare_indices: Vec<bool> = vec![false; flares.len()];

    for (fi, flare) in flares.iter().enumerate() {
        if flare.class.starts_with(config.min_class) || flare.class.starts_with('X') {
            let detected = detections.iter().any(|d| {
                d.is_true_positive
                    && ((d.timestamp - flare.begin).num_seconds().abs() < tolerance_secs
                        || (d.timestamp >= flare.begin && d.timestamp <= flare.end))
            });
            if detected {
                detected_flare_indices[fi] = true;
            } else {
                // Find max fused score during this flare's window
                let _count = records
                    .iter()
                    .filter(|r| r.timestamp >= flare.begin && r.timestamp <= flare.end)
                    .count();
                missed_flares.push(MissedFlare {
                    begin: flare.begin,
                    peak: flare.peak,
                    class: flare.class.clone(),
                    max_fused_score: 0.0, // would need to replay to get this
                });
            }
        }
    }

    let total_eval_flares = flares
        .iter()
        .filter(|f| f.class.starts_with(config.min_class) || f.class.starts_with('X'))
        .count();
    let false_negatives = total_eval_flares - true_positives.min(total_eval_flares);

    let recall = if total_eval_flares > 0 {
        true_positives as f64 / total_eval_flares as f64
    } else {
        0.0
    };
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    // Lead time: how early before flare peak did we detect?
    let lead_times: Vec<f64> = detections
        .iter()
        .filter(|d| d.is_true_positive)
        .filter_map(|d| {
            flares.iter().find_map(|f| {
                let dt = (d.timestamp - f.begin).num_seconds().abs();
                if dt < tolerance_secs || (d.timestamp >= f.begin && d.timestamp <= f.end) {
                    Some((f.peak - d.timestamp).num_seconds() as f64 / 3600.0)
                } else {
                    None
                }
            })
        })
        .collect();
    let mean_lead_time = if !lead_times.is_empty() {
        lead_times.iter().sum::<f64>() / lead_times.len() as f64
    } else {
        0.0
    };

    // Class breakdown
    let mut class_breakdown = Vec::new();
    for prefix in &["M", "X"] {
        let total = flares
            .iter()
            .filter(|f| f.class.starts_with(prefix))
            .count();
        let detected = detections
            .iter()
            .filter(|d| {
                d.is_true_positive
                    && d.matched_flare
                        .as_ref()
                        .map_or(false, |c| c.starts_with(prefix))
            })
            .count();
        if total > 0 {
            class_breakdown.push(ClassStats {
                class: prefix.to_string(),
                total,
                detected,
                recall: detected as f64 / total as f64,
            });
        }
    }

    BacktestResults {
        total_records: records.len(),
        start_time: records.first().map(|r| r.timestamp).unwrap_or_default(),
        end_time: records.last().map(|r| r.timestamp).unwrap_or_default(),
        total_flares: total_eval_flares,
        true_positives,
        false_negatives,
        false_positives,
        recall,
        precision,
        f1,
        mean_lead_time_hours: mean_lead_time,
        detections,
        missed_flares,
        class_breakdown,
    }
}

/// Estimate electron flux from available OMNI data.
/// Real >2 MeV electron flux isn't in OMNI, so we use a rough proxy
/// based on solar wind speed and Dst (empirical correlation).
fn estimate_electron_flux(rec: &HistoricalRecord) -> f64 {
    let base = 100.0; // quiet-time background ~100 pfu

    // High-speed streams enhance electron flux (2-3 day lag, but
    // we approximate as instantaneous for backtest)
    let speed_factor = if let Some(v) = rec.solar_wind_speed {
        if v > 600.0 {
            (v / 400.0).powi(2) // strong HSS
        } else if v > 500.0 {
            v / 400.0
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Storm-time enhancement
    let storm_factor = if let Some(dst) = rec.dst {
        if dst < -50.0 {
            ((-dst) / 50.0).min(10.0) // storms enhance electron flux
        } else {
            1.0
        }
    } else {
        1.0
    };

    base * speed_factor * storm_factor
}
