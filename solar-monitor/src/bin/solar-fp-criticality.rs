//! False positive analysis for the criticality detector.
//!
//! Counts how often the criticality score exceeds various thresholds
//! and classifies each trigger as:
//!   - True Positive: M/X flare within 24h after trigger
//!   - False Positive: no M/X flare within 24h after trigger
//!
//! Also computes: precision, recall, F1, and FAR (false alarm ratio)
//! at multiple threshold levels.

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Utc};
use solar_monitor::detection::criticality::CriticalityDetector;
use std::collections::BTreeMap;

struct Flare {
    begin: DateTime<Utc>,
    class: String,
}

fn main() {
    let dbdt_data = load_dbdt(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/atmospheric-analysis/data/frd_daily_dbdt_2010_2024.csv",
    );
    let f107_daily = load_f107(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/kp_daily.csv",
    );
    let flares = load_flares("solar-monitor/data/catalogs/solar_flares.csv");

    let flare_dates: Vec<NaiveDate> = flares
        .iter()
        .filter(|f| f.begin.year() >= 2010 && f.begin.year() <= 2024)
        .map(|f| f.begin.date_naive())
        .collect();
    let x_flare_dates: Vec<NaiveDate> = flares
        .iter()
        .filter(|f| f.begin.year() >= 2010 && f.begin.year() <= 2024 && f.class.starts_with('X'))
        .map(|f| f.begin.date_naive())
        .collect();

    let dbdt_map: BTreeMap<NaiveDate, (f64, f64, f64)> = dbdt_data
        .into_iter()
        .map(|(d, mean, max, p95, std)| (d, (p95, max, std)))
        .collect();
    let f107_map: BTreeMap<NaiveDate, f64> = f107_daily.into_iter().collect();

    // Run detector over full timeline.
    let mut det = CriticalityDetector::default_detector();
    let mut score_timeline: BTreeMap<NaiveDate, f64> = BTreeMap::new();

    let start = NaiveDate::from_ymd_opt(2010, 1, 31).unwrap();
    let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let mut day = start;

    while day <= end {
        let f107 = f107_map
            .range(..=day)
            .next_back()
            .map(|(_, &v)| v)
            .unwrap_or(100.0);
        let (dbdt_p95, dbdt_max, dbdt_std) = dbdt_map.get(&day).copied().unwrap_or((0.5, 1.0, 0.2));

        let (xray_long, xray_short, proton_flux) = proxy_observables(f107, dbdt_p95, dbdt_max);

        let ts = day.and_hms_opt(12, 0, 0).unwrap().and_utc();
        for h in [0, 3, 6, 9, 12, 15, 18, 21] {
            let t = ts - chrono::Duration::hours(12) + chrono::Duration::hours(h);
            det.ingest(xray_long, xray_short, proton_flux, t);
        }

        score_timeline.insert(day, det.score());
        day += chrono::Duration::days(1);
    }

    let total_days = score_timeline.len();

    // Threshold sweep.
    println!("=== False Positive Analysis: Criticality Detector (dB/dt input, 2010-2024) ===\n");
    println!("Total days evaluated: {}", total_days);
    println!(
        "Total M/X flare days: {} ({} unique days)",
        flare_dates.len(),
        unique_count(&flare_dates)
    );
    println!(
        "Total X-class flare days: {} ({} unique days)\n",
        x_flare_dates.len(),
        unique_count(&x_flare_dates)
    );

    println!("--- M/X Flare Detection (flare within 1-3 days after trigger) ---\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1", "FAR"
    );
    println!("{}", "-".repeat(78));

    for &threshold in &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] {
        let (tp, fp, fn_, tn) = compute_confusion(
            &score_timeline,
            &flare_dates,
            threshold,
            3, // flare within 3 days
        );

        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let far = fp as f64 / (tp + fp).max(1) as f64;

        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            threshold, tp, fp, fn_, tn, precision, recall, f1, far
        );
    }

    println!("\n--- X-Class Only Detection (X-flare within 3 days after trigger) ---\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1", "FAR"
    );
    println!("{}", "-".repeat(78));

    for &threshold in &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] {
        let (tp, fp, fn_, tn) = compute_confusion(&score_timeline, &x_flare_dates, threshold, 3);

        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let far = fp as f64 / (tp + fp).max(1) as f64;

        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            threshold, tp, fp, fn_, tn, precision, recall, f1, far
        );
    }

    // Also compute with 1-day lookahead (stricter).
    println!("\n--- M/X Flare Detection (flare within 1 day, stricter) ---\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1", "FAR"
    );
    println!("{}", "-".repeat(78));

    for &threshold in &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] {
        let (tp, fp, fn_, tn) = compute_confusion(&score_timeline, &flare_dates, threshold, 1);

        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let far = fp as f64 / (tp + fp).max(1) as f64;

        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            threshold, tp, fp, fn_, tn, precision, recall, f1, far
        );
    }

    // Baseline: what's the rate just from solar cycle phase?
    let unique_flare_days = unique_count(&flare_dates);
    let base_rate = unique_flare_days as f64 / total_days as f64;
    println!("\n--- Baseline ---");
    println!(
        "Base rate (any M/X flare day): {:.1}% ({}/{} days)",
        100.0 * base_rate,
        unique_flare_days,
        total_days
    );
    println!(
        "Base rate (X-class flare day): {:.2}% ({}/{} days)",
        100.0 * unique_count(&x_flare_dates) as f64 / total_days as f64,
        unique_count(&x_flare_dates),
        total_days
    );
}

/// Compute confusion matrix.
/// A day is a "positive prediction" if score > threshold.
/// A day is a "true positive" if an M/X flare occurs within `lookahead` days.
fn compute_confusion(
    scores: &BTreeMap<NaiveDate, f64>,
    flare_dates: &[NaiveDate],
    threshold: f64,
    lookahead_days: i64,
) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    let mut tn = 0;

    for (&day, &score) in scores {
        let predicted_positive = score > threshold;
        let actual_positive = flare_dates.iter().any(|&fd| {
            let diff = (fd - day).num_days();
            diff >= 0 && diff <= lookahead_days
        });

        match (predicted_positive, actual_positive) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }

    (tp, fp, fn_, tn)
}

fn unique_count(dates: &[NaiveDate]) -> usize {
    let mut s: Vec<NaiveDate> = dates.to_vec();
    s.sort();
    s.dedup();
    s.len()
}

fn proxy_observables(f107: f64, dbdt_p95: f64, dbdt_max: f64) -> (f64, f64, f64) {
    let xray_long = if f107 > 65.0 {
        let t = ((f107 - 70.0) / 230.0).clamp(0.0, 1.0);
        10.0_f64.powf(-7.0 + 2.0 * t)
    } else {
        1e-7
    };

    let hardness = if dbdt_p95 > 5.0 {
        0.15 + 0.10 * ((dbdt_p95 - 5.0) / 15.0).min(1.0)
    } else if dbdt_p95 > 1.0 {
        0.04 + 0.11 * ((dbdt_p95 - 1.0) / 4.0)
    } else {
        0.04
    };
    let xray_short = xray_long * hardness;

    let proton_flux = if dbdt_max > 10.0 {
        1.0 + 2.0 * (dbdt_max / 10.0).log10().min(2.0)
    } else if dbdt_max > 3.0 {
        0.3 + 0.7 * ((dbdt_max - 3.0) / 7.0)
    } else {
        0.3
    };

    (xray_long, xray_short, proton_flux)
}

fn load_dbdt(path: &str) -> Vec<(NaiveDate, f64, f64, f64, f64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 9 {
                return None;
            }
            let d = NaiveDate::parse_from_str(f[0], "%Y-%m-%d").ok()?;
            Some((
                d,
                f[1].parse().ok()?,
                f[2].parse().ok()?,
                f[3].parse().ok()?,
                f[4].parse().ok()?,
            ))
        })
        .collect()
}

fn load_f107(path: &str) -> Vec<(NaiveDate, f64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 10 {
                return None;
            }
            let y: i32 = f[0].parse().ok()?;
            if y < 2009 {
                return None;
            }
            let d = NaiveDate::from_ymd_opt(y, f[1].parse().ok()?, f[2].parse().ok()?)?;
            let v: f64 = f[9].parse().ok()?;
            if v > 0.0 {
                Some((d, v))
            } else {
                None
            }
        })
        .collect()
}

fn load_flares(path: &str) -> Vec<Flare> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 4 {
                return None;
            }
            let class = f[3].to_string();
            if !class.starts_with('M') && !class.starts_with('X') {
                return None;
            }
            let begin = NaiveDateTime::parse_from_str(f[0].trim(), "%Y-%m-%d %H:%M:%S")
                .ok()?
                .and_utc();
            Some(Flare { begin, class })
        })
        .collect()
}
