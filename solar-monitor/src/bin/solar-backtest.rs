//! Backtest the rank fusion detector against historical solar data.
//!
//! Usage:
//!   solar-backtest [--data-dir <path>] [--threshold <0..1>] [--tolerance <hours>]
//!
//! Default data dir: /home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind/

use std::path::PathBuf;

use solar_monitor::backtest::loaders;
use solar_monitor::backtest::replay::{self, BacktestConfig};

fn main() {
    let mut data_dir = PathBuf::from(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind",
    );
    let mut threshold = 0.7;
    let mut tolerance = 2.0;
    let mut year_start: Option<i32> = None;
    let mut year_end: Option<i32> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                data_dir = PathBuf::from(args.get(i).expect("Missing data-dir value"));
            }
            "--threshold" => {
                i += 1;
                threshold = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.7);
            }
            "--tolerance" => {
                i += 1;
                tolerance = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            }
            "--year-start" => {
                i += 1;
                year_start = args.get(i).and_then(|s| s.parse().ok());
            }
            "--year-end" => {
                i += 1;
                year_end = args.get(i).and_then(|s| s.parse().ok());
            }
            "--help" | "-h" => {
                println!(
                    "solar-backtest — Replay historical solar data through rank fusion detector"
                );
                println!();
                println!("Usage: solar-backtest [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --data-dir <PATH>     Data directory (default: Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind/)");
                println!("  --threshold <0..1>    Fused alert threshold (default: 0.7)");
                println!("  --tolerance <HOURS>   Detection tolerance window (default: 2.0h)");
                println!("  --year-start <YEAR>   Filter: start year (inclusive)");
                println!("  --year-end <YEAR>     Filter: end year (inclusive)");
                println!("  -h, --help            Show this help");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    println!("=== Solar Flare Detection Backtest ===");
    println!("Data directory: {}", data_dir.display());
    println!("Alert threshold: {:.2}", threshold);
    println!("Tolerance window: {:.1}h", tolerance);
    println!();

    // Load datasets
    print!("Loading solar flares... ");
    let flares_path = data_dir.join("solar_flares.csv");
    let flares = match loaders::load_flares(&flares_path) {
        Ok(f) => {
            println!("{} M/X-class flares", f.len());
            f
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };

    print!("Loading OMNI hourly... ");
    let omni_path = data_dir.join("omni_hourly.csv");
    let omni = match loaders::load_omni(&omni_path) {
        Ok(o) => {
            println!("{} records", o.len());
            o
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };

    print!("Loading Kp index... ");
    let kp_path = data_dir.join("kp_3hourly.csv");
    let kp = match loaders::load_kp(&kp_path) {
        Ok(k) => {
            println!("{} records", k.len());
            k
        }
        Err(e) => {
            eprintln!("WARNING: {} (continuing without Kp)", e);
            std::collections::BTreeMap::new()
        }
    };

    println!();
    print!("Merging datasets... ");
    let mut records = loaders::merge_datasets(&omni, &kp, &flares);
    println!("{} merged records", records.len());

    // Filter by year if requested
    if let Some(ys) = year_start {
        records.retain(|r| r.timestamp.year() >= ys);
    }
    if let Some(ye) = year_end {
        records.retain(|r| r.timestamp.year() <= ye);
    }

    if records.is_empty() {
        eprintln!("No records in selected range");
        return;
    }

    use chrono::Datelike;
    println!(
        "Replay range: {} to {} ({} records)",
        records.first().unwrap().timestamp.format("%Y-%m-%d"),
        records.last().unwrap().timestamp.format("%Y-%m-%d"),
        records.len(),
    );

    // Count flares in range
    let range_flares: Vec<_> = flares
        .iter()
        .filter(|f| {
            f.begin >= records.first().unwrap().timestamp
                && f.begin <= records.last().unwrap().timestamp
        })
        .cloned()
        .collect();
    println!("Flares in range: {} M/X-class", range_flares.len());
    println!();

    // Run backtest
    println!("Running backtest...");
    let config = BacktestConfig {
        alert_threshold: threshold,
        tolerance_hours: tolerance,
        min_class: 'M',
    };

    let results = replay::run_backtest(&records, &range_flares, &config);

    // Print results
    println!();
    println!("=== Results ===");
    println!(
        "Period: {} to {}",
        results.start_time.format("%Y-%m-%d"),
        results.end_time.format("%Y-%m-%d"),
    );
    println!("Records replayed: {}", results.total_records);
    println!("Known M/X flares: {}", results.total_flares);
    println!();
    println!("True positives:   {}", results.true_positives);
    println!("False positives:  {}", results.false_positives);
    println!("False negatives:  {}", results.false_negatives);
    println!();
    println!("Recall:    {:.1}%", results.recall * 100.0);
    println!("Precision: {:.1}%", results.precision * 100.0);
    println!("F1 score:  {:.3}", results.f1);
    println!(
        "Mean lead time: {:.1}h (positive = detected before peak)",
        results.mean_lead_time_hours,
    );

    // Class breakdown
    if !results.class_breakdown.is_empty() {
        println!();
        println!("=== Per-Class Breakdown ===");
        for cs in &results.class_breakdown {
            println!(
                "  {}-class: {}/{} detected ({:.1}% recall)",
                cs.class,
                cs.detected,
                cs.total,
                cs.recall * 100.0,
            );
        }
    }

    // Top detections
    if !results.detections.is_empty() {
        println!();
        println!("=== Sample Detections (first 20) ===");
        for d in results.detections.iter().take(20) {
            let tp_marker = if d.is_true_positive { "TP" } else { "FP" };
            let matched = d.matched_flare.as_deref().unwrap_or("-");
            println!(
                "  {} {} fused={:.3} agree={}/5 flux={:.2e} matched={}",
                d.timestamp.format("%Y-%m-%d %H:%M"),
                tp_marker,
                d.fused_score,
                d.detector_agreement,
                d.flux,
                matched,
            );
        }
    }

    // Missed flares
    if !results.missed_flares.is_empty() {
        println!();
        println!(
            "=== Missed Flares (first 20 of {}) ===",
            results.missed_flares.len()
        );
        for m in results.missed_flares.iter().take(20) {
            println!(
                "  {} {}-class peak={}",
                m.begin.format("%Y-%m-%d %H:%M"),
                m.class,
                m.peak.format("%H:%M"),
            );
        }
    }

    // Threshold sweep
    println!();
    println!("=== Threshold Sweep ===");
    for t in &[0.5, 0.6, 0.7, 0.8, 0.9] {
        let sweep_config = BacktestConfig {
            alert_threshold: *t,
            tolerance_hours: tolerance,
            min_class: 'M',
        };
        let sweep = replay::run_backtest(&records, &range_flares, &sweep_config);
        println!(
            "  threshold={:.1}: recall={:.1}% precision={:.1}% F1={:.3} TP={} FP={} FN={}",
            t,
            sweep.recall * 100.0,
            sweep.precision * 100.0,
            sweep.f1,
            sweep.true_positives,
            sweep.false_positives,
            sweep.false_negatives,
        );
    }
}
