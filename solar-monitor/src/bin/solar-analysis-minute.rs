//! Minute-cadence analysis: synthesize 1-min X-ray flux from flare catalog
//! and re-run all detectors to see rate-of-change and energy performance.

use chrono::{Datelike, NaiveDateTime};
use std::path::PathBuf;

use solar_monitor::backtest::loaders::{self, FlareEvent};
use solar_monitor::backtest::synthesize;
use solar_monitor::detection::cusum::CusumDetector;
use solar_monitor::detection::energy::EnergyDetector;
use solar_monitor::detection::multichannel::MultichannelDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use solar_monitor::detection::rate_of_change::RateOfChangeDetector;
use solar_monitor::detection::zscore::ZScoreDetector;

fn main() {
    let data_dir = PathBuf::from(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind",
    );

    // Parse year range (default: 2024 — most flares, 1036 M/X events)
    let mut year_start = 2024;
    let mut year_end = 2024;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--years" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    let parts: Vec<&str> = s.split('-').collect();
                    year_start = parts[0].parse().unwrap_or(2024);
                    year_end = parts
                        .get(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(year_start);
                }
            }
            "--help" | "-h" => {
                println!("solar-analysis-minute [--years START-END]");
                println!("  Default: --years 2024 (most active year, 1036 flares)");
                println!("  Example: --years 2022-2025");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Minute-Cadence Flare Detection Analysis ===");
    println!("Years: {}-{}\n", year_start, year_end);

    // Load datasets
    let all_flares = loaders::load_flares(&data_dir.join("solar_flares.csv")).unwrap();
    let omni = loaders::load_omni(&data_dir.join("omni_hourly.csv")).unwrap();
    let kp = loaders::load_kp(&data_dir.join("kp_3hourly.csv")).unwrap_or_default();

    // Filter flares to year range
    let flares: Vec<FlareEvent> = all_flares
        .into_iter()
        .filter(|f| f.begin.year() >= year_start && f.begin.year() <= year_end)
        .collect();
    println!("Flares in range: {} M/X-class", flares.len());

    // Synthesize minute-cadence X-ray flux
    let start = NaiveDateTime::parse_from_str(
        &format!("{}-01-01 00:00:00", year_start),
        "%Y-%m-%d %H:%M:%S",
    )
    .unwrap()
    .and_utc();
    let end =
        NaiveDateTime::parse_from_str(&format!("{}-12-31 23:59:00", year_end), "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc();

    print!("Synthesizing minute-cadence X-ray flux... ");
    let xray_minute = synthesize::synthesize_xray_minute(&flares, start, end);
    println!("{} minute samples", xray_minute.len());

    print!("Merging with OMNI/Kp... ");
    let records = synthesize::merge_minute_cadence(&xray_minute, &omni, &kp, &flares);
    println!("{} records\n", records.len());

    let tolerance_secs: i64 = 2 * 3600;

    // =========================================
    // Per-detector comparison at minute cadence
    // =========================================
    println!("========================================");
    println!("  Individual Detector Comparison (1-min)");
    println!("========================================\n");

    struct DetResult {
        name: &'static str,
        tp: usize,
        fp: usize,
        fn_count: usize,
        recall: f64,
        precision: f64,
        f1: f64,
    }

    let detector_configs: Vec<(
        &str,
        Box<dyn FnMut(f64, f64, chrono::DateTime<chrono::Utc>) -> (f64, bool)>,
    )> = vec![
        (
            "Z-Score",
            Box::new({
                let mut d = ZScoreDetector::default_detector();
                move |xray, _e, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "CUSUM",
            Box::new({
                let mut d = CusumDetector::new(120, 8.0); // tighter sensitivity
                move |xray, _e, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Energy",
            Box::new({
                let mut d = EnergyDetector::default_detector();
                move |xray, e, ts| {
                    d.ingest(xray, e, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Rate-of-Change",
            Box::new({
                let mut d = RateOfChangeDetector::default_detector();
                move |xray, _e, ts| {
                    d.ingest(xray, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Multichannel",
            Box::new({
                let mut d = MultichannelDetector::default_detector();
                move |xray, e, ts| {
                    d.ingest(xray, e, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
        (
            "Rank Fusion",
            Box::new({
                let mut d = RankFusionDetector::new(0.5);
                move |xray, e, ts| {
                    d.ingest_simple(xray, e, ts);
                    (d.score(), d.is_anomalous())
                }
            }),
        ),
    ];

    let mut results: Vec<DetResult> = Vec::new();

    for (name, mut detect_fn) in detector_configs {
        let (tp, fp, unique_tp) = run_detector(&records, &flares, tolerance_secs, &mut detect_fn);
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
        results.push(DetResult {
            name,
            tp,
            fp,
            fn_count,
            recall,
            precision,
            f1,
        });
    }

    println!(
        "{:<16} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7}",
        "Detector", "TP", "FP", "FN", "Recall%", "Prec%", "F1"
    );
    println!("{}", "-".repeat(75));
    for r in &results {
        println!(
            "{:<16} {:>7} {:>7} {:>7} {:>7.1}% {:>7.1}% {:>7.3}",
            r.name,
            r.tp,
            r.fp,
            r.fn_count,
            r.recall * 100.0,
            r.precision * 100.0,
            r.f1
        );
    }

    // =========================================
    // Energy bias at minute cadence
    // =========================================
    println!("\n========================================");
    println!("  Energy Bias (Rank Fusion, 1-min)");
    println!("========================================\n");

    // Run fusion and track per-flare max scores
    let mut fusion = RankFusionDetector::new(0.5);
    let mut flare_max_scores = vec![0.0f64; flares.len()];
    let mut detected = vec![false; flares.len()];

    for rec in &records {
        let electron = estimate_electron_flux(rec);
        fusion.ingest_simple(rec.xray_flux, electron, rec.timestamp);
        let score = fusion.score();
        let is_anom = fusion.is_anomalous();

        for (fi, flare) in flares.iter().enumerate() {
            let in_window = rec.timestamp >= flare.begin && rec.timestamp <= flare.end;
            let in_tolerance = (rec.timestamp - flare.begin).num_seconds().abs() < tolerance_secs;
            if in_window || in_tolerance {
                if score > flare_max_scores[fi] {
                    flare_max_scores[fi] = score;
                }
                if is_anom {
                    detected[fi] = true;
                }
            }
        }
    }

    let bins: Vec<(&str, f64, f64)> = vec![
        ("M1.0-M1.9", 0.10, 0.20),
        ("M2.0-M4.9", 0.20, 0.50),
        ("M5.0-M9.9", 0.50, 1.00),
        ("X1.0-X2.9", 1.00, 3.00),
        ("X3.0-X9.9", 3.00, 10.0),
        ("X10.0+", 10.0, 1000.0),
    ];

    println!(
        "{:<14} {:>7} {:>9} {:>8} {:>10}",
        "Class Range", "Total", "Detected", "Recall%", "Avg Score"
    );
    println!("{}", "-".repeat(55));

    for (label, lo, hi) in &bins {
        let in_bin: Vec<usize> = flares
            .iter()
            .enumerate()
            .filter(|(_, f)| f.class_numeric >= *lo && f.class_numeric < *hi)
            .map(|(i, _)| i)
            .collect();
        if in_bin.is_empty() {
            continue;
        }

        let det_count = in_bin.iter().filter(|&&i| detected[i]).count();
        let avg_score: f64 =
            in_bin.iter().map(|&i| flare_max_scores[i]).sum::<f64>() / in_bin.len() as f64;
        let recall = det_count as f64 / in_bin.len() as f64;

        println!(
            "{:<14} {:>7} {:>9} {:>7.1}% {:>10.3}",
            label,
            in_bin.len(),
            det_count,
            recall * 100.0,
            avg_score
        );
    }

    // Duration bias
    println!("\n--- Detection by Duration (1-min cadence) ---\n");
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
        let in_bin: Vec<usize> = flares
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                let dur = (f.end - f.begin).num_minutes();
                dur >= *lo_min && dur < *hi_min
            })
            .map(|(i, _)| i)
            .collect();
        if in_bin.is_empty() {
            continue;
        }

        let det_count = in_bin.iter().filter(|&&i| detected[i]).count();
        let avg_score: f64 =
            in_bin.iter().map(|&i| flare_max_scores[i]).sum::<f64>() / in_bin.len() as f64;
        let recall = det_count as f64 / in_bin.len() as f64;

        println!(
            "{:<14} {:>7} {:>9} {:>7.1}% {:>10.3}",
            label,
            in_bin.len(),
            det_count,
            recall * 100.0,
            avg_score
        );
    }

    // Total
    let total_det = detected.iter().filter(|&&d| d).count();
    println!(
        "\nTotal: {}/{} detected ({:.1}%)",
        total_det,
        flares.len(),
        total_det as f64 / flares.len() as f64 * 100.0
    );
}

fn run_detector(
    records: &[loaders::HistoricalRecord],
    flares: &[FlareEvent],
    tolerance_secs: i64,
    detect_fn: &mut dyn FnMut(f64, f64, chrono::DateTime<chrono::Utc>) -> (f64, bool),
) -> (usize, usize, usize) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut last_detection: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut detected_flares = vec![false; flares.len()];

    for rec in records {
        let electron = estimate_electron_flux(rec);
        let (_score, is_anomalous) = detect_fn(rec.xray_flux, electron, rec.timestamp);

        if is_anomalous {
            let dominated =
                last_detection.map_or(false, |last| (rec.timestamp - last).num_seconds() < 3600);
            if !dominated {
                let mut is_tp = false;
                for (fi, flare) in flares.iter().enumerate() {
                    let dt = (rec.timestamp - flare.begin).num_seconds().abs();
                    if dt < tolerance_secs
                        || (rec.timestamp >= flare.begin && rec.timestamp <= flare.end)
                    {
                        is_tp = true;
                        detected_flares[fi] = true;
                    }
                }
                if is_tp {
                    tp += 1;
                } else {
                    fp += 1;
                }
                last_detection = Some(rec.timestamp);
            }
        }
    }

    let unique_tp = detected_flares.iter().filter(|&&d| d).count();
    (tp, fp, unique_tp)
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
