//! Detailed analysis: per-detector comparison, energy bias, detection breakdown.

use chrono::Datelike;
use std::path::PathBuf;

use solar_monitor::backtest::loaders::{self, FlareEvent};
use solar_monitor::backtest::replay;
use solar_monitor::detection::cusum::CusumDetector;
use solar_monitor::detection::energy::EnergyDetector;
use solar_monitor::detection::multichannel::MultichannelDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use solar_monitor::detection::rate_of_change::RateOfChangeDetector;
use solar_monitor::detection::zscore::ZScoreDetector;
use solar_monitor::feeds::xray::FlareClass;

fn main() {
    let data_dir = PathBuf::from(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind",
    );

    println!("=== Solar Flare Detection — Detailed Analysis ===\n");

    // Load data
    let flares = loaders::load_flares(&data_dir.join("solar_flares.csv")).unwrap();
    let omni = loaders::load_omni(&data_dir.join("omni_hourly.csv")).unwrap();
    let kp = loaders::load_kp(&data_dir.join("kp_3hourly.csv")).unwrap();
    let records = loaders::merge_datasets(&omni, &kp, &flares);

    println!(
        "Loaded {} records, {} M/X flares\n",
        records.len(),
        flares.len()
    );

    let tolerance_secs: i64 = 2 * 3600; // 2h tolerance

    // =========================================
    // 1. Per-detector individual performance
    // =========================================
    println!("========================================");
    println!("  PART 1: Individual Detector Comparison");
    println!("========================================\n");

    // Run each detector individually
    let detectors: Vec<(
        &str,
        Box<dyn FnMut(f64, f64, chrono::DateTime<chrono::Utc>) -> (f64, bool)>,
    )> = vec![
        (
            "Z-Score",
            Box::new({
                let mut d = ZScoreDetector::default_detector();
                move |xray, _electron, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "CUSUM",
            Box::new({
                let mut d = CusumDetector::default_detector();
                move |xray, _electron, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Energy",
            Box::new({
                let mut d = EnergyDetector::default_detector();
                move |xray, electron, ts| {
                    d.ingest(xray, electron, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Rate-of-Change",
            Box::new({
                let mut d = RateOfChangeDetector::default_detector();
                move |xray, _electron, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Multichannel",
            Box::new({
                let mut d = MultichannelDetector::default_detector();
                move |xray, electron, ts| {
                    d.ingest(xray, electron, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Rank Fusion",
            Box::new({
                let mut d = RankFusionDetector::new(0.5);
                move |xray, electron, ts| {
                    d.ingest_simple(xray, electron, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
    ];

    let mut all_results: Vec<(&str, DetectorStats)> = Vec::new();

    for (name, mut detect_fn) in detectors {
        let stats = run_single_detector(&records, &flares, tolerance_secs, &mut detect_fn);
        all_results.push((name, stats));
    }

    println!(
        "{:<16} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7}",
        "Detector", "TP", "FP", "FN", "Recall%", "Prec%", "F1"
    );
    println!("{}", "-".repeat(75));
    for (name, stats) in &all_results {
        println!(
            "{:<16} {:>7} {:>7} {:>7} {:>7.1}% {:>7.1}% {:>7.3}",
            name,
            stats.tp,
            stats.fp,
            stats.fn_count,
            stats.recall * 100.0,
            stats.precision * 100.0,
            stats.f1
        );
    }

    // =========================================
    // 2. Total detected vs not
    // =========================================
    println!("\n========================================");
    println!("  PART 2: Detection Coverage");
    println!("========================================\n");

    // Use the rank fusion at threshold 0.5 (best F1)
    let mut fusion = RankFusionDetector::new(0.5);
    let mut detected_flare_ids: Vec<bool> = vec![false; flares.len()];
    let mut detection_scores: Vec<(usize, f64)> = Vec::new(); // (flare_idx, max_score)

    // Track max score per flare
    let mut flare_max_scores: Vec<f64> = vec![0.0; flares.len()];

    for rec in &records {
        let electron = estimate_electron_flux(rec);
        fusion.ingest_simple(rec.xray_flux, electron, rec.timestamp);

        let score = fusion.score();

        // Check against all flares
        for (fi, flare) in flares.iter().enumerate() {
            if rec.timestamp >= flare.begin && rec.timestamp <= flare.end {
                if score > flare_max_scores[fi] {
                    flare_max_scores[fi] = score;
                }
            }
            // Also check tolerance window
            let dt = (rec.timestamp - flare.begin).num_seconds().abs();
            if dt < tolerance_secs || (rec.timestamp >= flare.begin && rec.timestamp <= flare.end) {
                if fusion.is_anomalous() {
                    detected_flare_ids[fi] = true;
                }
            }
        }
    }

    let total_detected = detected_flare_ids.iter().filter(|&&d| d).count();
    let total_missed = flares.len() - total_detected;

    println!("Total M/X flares: {}", flares.len());
    println!(
        "  Detected:  {} ({:.1}%)",
        total_detected,
        total_detected as f64 / flares.len() as f64 * 100.0
    );
    println!(
        "  Missed:    {} ({:.1}%)",
        total_missed,
        total_missed as f64 / flares.len() as f64 * 100.0
    );

    // Near-misses: flares where max score was > 0.3 but didn't cross threshold
    let near_misses: Vec<_> = flares
        .iter()
        .enumerate()
        .filter(|(i, _)| !detected_flare_ids[*i] && flare_max_scores[*i] > 0.3)
        .collect();
    println!(
        "  Near-misses (score > 0.3 but < threshold): {}",
        near_misses.len()
    );

    // =========================================
    // 3. Energy bias analysis
    // =========================================
    println!("\n========================================");
    println!("  PART 3: Energy Bias Analysis");
    println!("========================================\n");

    // Bin flares by class magnitude
    let bins: Vec<(&str, f64, f64)> = vec![
        ("M1.0-M1.9", 0.10, 0.20),
        ("M2.0-M4.9", 0.20, 0.50),
        ("M5.0-M9.9", 0.50, 1.00),
        ("X1.0-X2.9", 1.00, 3.00),
        ("X3.0-X9.9", 3.00, 10.0),
        ("X10.0+", 10.0, 1000.0),
    ];

    println!(
        "{:<14} {:>7} {:>9} {:>8} {:>10} {:>12}",
        "Class Range", "Total", "Detected", "Recall%", "Avg Score", "Avg Duration"
    );
    println!("{}", "-".repeat(70));

    for (label, lo, hi) in &bins {
        let in_bin: Vec<(usize, &FlareEvent)> = flares
            .iter()
            .enumerate()
            .filter(|(_, f)| f.class_numeric >= *lo && f.class_numeric < *hi)
            .collect();

        if in_bin.is_empty() {
            continue;
        }

        let detected = in_bin
            .iter()
            .filter(|(i, _)| detected_flare_ids[*i])
            .count();
        let avg_score: f64 = in_bin
            .iter()
            .map(|(i, _)| flare_max_scores[*i])
            .sum::<f64>()
            / in_bin.len() as f64;
        let avg_duration: f64 = in_bin
            .iter()
            .map(|(_, f)| (f.end - f.begin).num_minutes() as f64)
            .sum::<f64>()
            / in_bin.len() as f64;
        let recall = detected as f64 / in_bin.len() as f64;

        println!(
            "{:<14} {:>7} {:>9} {:>7.1}% {:>10.3} {:>10.0} min",
            label,
            in_bin.len(),
            detected,
            recall * 100.0,
            avg_score,
            avg_duration
        );
    }

    // Duration bias
    println!("\n--- Detection by Flare Duration ---\n");
    let duration_bins: Vec<(&str, i64, i64)> = vec![
        ("<15 min", 0, 15),
        ("15-30 min", 15, 30),
        ("30-60 min", 30, 60),
        ("1-2 hours", 60, 120),
        ("2-4 hours", 120, 240),
        (">4 hours", 240, 99999),
    ];

    println!(
        "{:<14} {:>7} {:>9} {:>8} {:>10}",
        "Duration", "Total", "Detected", "Recall%", "Avg Score"
    );
    println!("{}", "-".repeat(55));

    for (label, lo_min, hi_min) in &duration_bins {
        let in_bin: Vec<(usize, &FlareEvent)> = flares
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let dur = (f.end - f.begin).num_minutes();
                dur >= *lo_min && dur < *hi_min
            })
            .collect();

        if in_bin.is_empty() {
            continue;
        }

        let detected = in_bin
            .iter()
            .filter(|(i, _)| detected_flare_ids[*i])
            .count();
        let avg_score: f64 = in_bin
            .iter()
            .map(|(i, _)| flare_max_scores[*i])
            .sum::<f64>()
            / in_bin.len() as f64;
        let recall = detected as f64 / in_bin.len() as f64;

        println!(
            "{:<14} {:>7} {:>9} {:>7.1}% {:>10.3}",
            label,
            in_bin.len(),
            detected,
            recall * 100.0,
            avg_score
        );
    }

    // Solar cycle bias
    println!("\n--- Detection by Solar Cycle Phase ---\n");
    println!(
        "{:<10} {:>7} {:>9} {:>8} {:>7} {:>10}",
        "Year", "Flares", "Detected", "Recall%", "FP", "Avg Score"
    );
    println!("{}", "-".repeat(55));

    let min_year = flares.first().map(|f| f.begin.year()).unwrap_or(2010);
    let max_year = flares.last().map(|f| f.begin.year()).unwrap_or(2026);

    for year in min_year..=max_year {
        let in_year: Vec<(usize, &FlareEvent)> = flares
            .iter()
            .enumerate()
            .filter(|(_, f)| f.begin.year() == year)
            .collect();

        if in_year.is_empty() {
            continue;
        }

        let detected = in_year
            .iter()
            .filter(|(i, _)| detected_flare_ids[*i])
            .count();
        let avg_score: f64 = in_year
            .iter()
            .map(|(i, _)| flare_max_scores[*i])
            .sum::<f64>()
            / in_year.len() as f64;
        let recall = detected as f64 / in_year.len() as f64;

        println!(
            "{:<10} {:>7} {:>9} {:>7.1}% {:>7} {:>10.3}",
            year,
            in_year.len(),
            detected,
            recall * 100.0,
            "-",
            avg_score
        );
    }

    // Score distribution of detected vs missed
    println!("\n--- Score Distribution ---\n");
    let detected_scores: Vec<f64> = flares
        .iter()
        .enumerate()
        .filter(|(i, _)| detected_flare_ids[*i])
        .map(|(i, _)| flare_max_scores[i])
        .collect();
    let missed_scores: Vec<f64> = flares
        .iter()
        .enumerate()
        .filter(|(i, _)| !detected_flare_ids[*i])
        .map(|(i, _)| flare_max_scores[i])
        .collect();

    if !detected_scores.is_empty() {
        let det_mean: f64 = detected_scores.iter().sum::<f64>() / detected_scores.len() as f64;
        let det_min = detected_scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let det_max = detected_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "Detected flares:  mean_score={:.3}  min={:.3}  max={:.3}",
            det_mean, det_min, det_max
        );
    }
    if !missed_scores.is_empty() {
        let mis_mean: f64 = missed_scores.iter().sum::<f64>() / missed_scores.len() as f64;
        let mis_min = missed_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let mis_max = missed_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "Missed flares:    mean_score={:.3}  min={:.3}  max={:.3}",
            mis_mean, mis_min, mis_max
        );
    }

    // Score histogram
    println!("\nScore histogram (all flares, max score during event):");
    let hist_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01];
    for w in hist_bins.windows(2) {
        let count = flare_max_scores
            .iter()
            .filter(|&&s| s >= w[0] && s < w[1])
            .count();
        let det = flares
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                flare_max_scores[*i] >= w[0]
                    && flare_max_scores[*i] < w[1]
                    && detected_flare_ids[*i]
            })
            .count();
        let bar = "#".repeat((count as f64 / flares.len() as f64 * 50.0) as usize);
        println!(
            "  [{:.1}-{:.1}) {:>5} ({:>4} det) {}",
            w[0], w[1], count, det, bar
        );
    }
}

struct DetectorStats {
    tp: usize,
    fp: usize,
    fn_count: usize,
    recall: f64,
    precision: f64,
    f1: f64,
}

fn run_single_detector(
    records: &[loaders::HistoricalRecord],
    flares: &[FlareEvent],
    tolerance_secs: i64,
    detect_fn: &mut dyn FnMut(f64, f64, chrono::DateTime<chrono::Utc>) -> (f64, bool),
) -> DetectorStats {
    let mut detections: Vec<(chrono::DateTime<chrono::Utc>, bool)> = Vec::new();

    for rec in records {
        let electron = estimate_electron_flux(rec);
        let (_score, is_anomalous) = detect_fn(rec.xray_flux, electron, rec.timestamp);

        if is_anomalous {
            // Deduplicate within 1 hour
            let dominated = detections.last().map_or(false, |(last_ts, _)| {
                (rec.timestamp - *last_ts).num_seconds() < 3600
            });
            if !dominated {
                let is_tp = flares.iter().any(|f| {
                    let dt = (rec.timestamp - f.begin).num_seconds().abs();
                    dt < tolerance_secs || (rec.timestamp >= f.begin && rec.timestamp <= f.end)
                });
                detections.push((rec.timestamp, is_tp));
            }
        }
    }

    let tp = detections.iter().filter(|(_, is_tp)| *is_tp).count();
    let fp = detections.iter().filter(|(_, is_tp)| !*is_tp).count();

    // Count unique flares detected
    let mut detected_flares = vec![false; flares.len()];
    for (ts, is_tp) in &detections {
        if *is_tp {
            for (fi, flare) in flares.iter().enumerate() {
                let dt = (*ts - flare.begin).num_seconds().abs();
                if dt < tolerance_secs || (*ts >= flare.begin && *ts <= flare.end) {
                    detected_flares[fi] = true;
                }
            }
        }
    }
    let unique_tp = detected_flares.iter().filter(|&&d| d).count();
    let fn_count = flares.len() - unique_tp;

    let recall = if flares.is_empty() {
        0.0
    } else {
        unique_tp as f64 / flares.len() as f64
    };
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    DetectorStats {
        tp,
        fp,
        fn_count: fn_count,
        recall,
        precision,
        f1,
    }
}

fn estimate_electron_flux(rec: &loaders::HistoricalRecord) -> f64 {
    let base = 100.0;
    let speed_factor = if let Some(v) = rec.solar_wind_speed {
        if v > 600.0 {
            (v / 400.0).powi(2)
        } else if v > 500.0 {
            v / 400.0
        } else {
            1.0
        }
    } else {
        1.0
    };
    let storm_factor = if let Some(dst) = rec.dst {
        if dst < -50.0 {
            ((-dst) / 50.0).min(10.0)
        } else {
            1.0
        }
    } else {
        1.0
    };
    base * speed_factor * storm_factor
}
