//! Backtest criticality detector v2 using the bivector commutator from
//! Fredericksburg magnetometer B-field vector components.
//!
//! Feeds B_x, B_y, B_z directly to `ingest_with_bfield()` which computes
//! the actual wedge product ||B ∧ Ḃ|| and uses it + loading fraction for scoring.

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Utc};
use solar_monitor::detection::criticality::CriticalityDetector;
use std::collections::BTreeMap;

struct DailyObs {
    date: NaiveDate,
    x_mean: f64,
    y_mean: f64,
    z_mean: f64,
    dbdt_max: f64,
}

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

    let flares: Vec<Flare> = flares
        .into_iter()
        .filter(|f| f.begin.year() >= 2010 && f.begin.year() <= 2024)
        .collect();
    let flare_dates: Vec<NaiveDate> = flares.iter().map(|f| f.begin.date_naive()).collect();
    let x_flare_dates: Vec<NaiveDate> = flares
        .iter()
        .filter(|f| f.class.starts_with('X'))
        .map(|f| f.begin.date_naive())
        .collect();

    println!("Loaded {} obs, {} flares", dbdt_data.len(), flares.len());

    let obs_map: BTreeMap<NaiveDate, &DailyObs> = dbdt_data.iter().map(|d| (d.date, d)).collect();
    let f107_map: BTreeMap<NaiveDate, f64> = f107_daily.into_iter().collect();

    // Run detector with B-field vector input.
    let mut det = CriticalityDetector::default_detector();
    let mut score_timeline: BTreeMap<NaiveDate, f64> = BTreeMap::new();

    let start = NaiveDate::from_ymd_opt(2010, 1, 31).unwrap();
    let end = NaiveDate::from_ymd_opt(2024, 12, 31).unwrap();
    let mut day = start;

    while day <= end {
        if let Some(obs) = obs_map.get(&day) {
            let f107 = f107_map
                .range(..=day)
                .next_back()
                .map(|(_, &v)| v)
                .unwrap_or(100.0);
            let xray_proxy = if f107 > 65.0 {
                let t = ((f107 - 70.0) / 230.0).clamp(0.0, 1.0);
                10.0_f64.powf(-7.0 + 2.0 * t)
            } else {
                1e-7
            };
            let proton_proxy = if obs.dbdt_max > 10.0 {
                1.0 + 2.0 * (obs.dbdt_max / 10.0).log10().min(2.0)
            } else if obs.dbdt_max > 3.0 {
                0.3 + 0.7 * ((obs.dbdt_max - 3.0) / 7.0)
            } else {
                0.3
            };

            // Feed 8 samples per day with actual B-field components.
            for h in [0, 3, 6, 9, 12, 15, 18, 21] {
                let ts = day.and_hms_opt(h as u32, 0, 0).unwrap().and_utc();
                det.ingest_with_bfield(
                    obs.x_mean,
                    obs.y_mean,
                    obs.z_mean,
                    xray_proxy,
                    proton_proxy,
                    ts,
                );
            }
        }

        score_timeline.insert(day, det.score());
        day += chrono::Duration::days(1);
    }

    println!("Processed {} days\n", score_timeline.len());

    // === Confusion matrix at multiple thresholds ===
    println!("=== Commutator-Enhanced Criticality Detector: FP Analysis (2010-2024) ===\n");

    let unique_flare_days = unique_count(&flare_dates);
    let total_days = score_timeline.len();
    println!(
        "Total days: {},  M/X flare days: {} ({:.1}%),  X-class days: {} ({:.2}%)\n",
        total_days,
        unique_flare_days,
        100.0 * unique_flare_days as f64 / total_days as f64,
        unique_count(&x_flare_dates),
        100.0 * unique_count(&x_flare_dates) as f64 / total_days as f64,
    );

    println!("--- M/X Flare Detection (flare within 3 days) ---\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1", "FAR"
    );
    println!("{}", "-".repeat(78));

    for &threshold in &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] {
        let (tp, fp, fn_, tn) = confusion(&score_timeline, &flare_dates, threshold, 3);
        let prec = tp as f64 / (tp + fp).max(1) as f64;
        let rec = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        let far = fp as f64 / (tp + fp).max(1) as f64;
        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            threshold, tp, fp, fn_, tn, prec, rec, f1, far
        );
    }

    println!("\n--- X-Class Only (flare within 3 days) ---\n");
    println!(
        "{:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Thresh", "TP", "FP", "FN", "TN", "Prec", "Recall", "F1", "FAR"
    );
    println!("{}", "-".repeat(78));

    for &threshold in &[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] {
        let (tp, fp, fn_, tn) = confusion(&score_timeline, &x_flare_dates, threshold, 3);
        let prec = tp as f64 / (tp + fp).max(1) as f64;
        let rec = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if prec + rec > 0.0 {
            2.0 * prec * rec / (prec + rec)
        } else {
            0.0
        };
        let far = fp as f64 / (tp + fp).max(1) as f64;
        println!(
            "{:>6.2} {:>6} {:>6} {:>6} {:>6} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            threshold, tp, fp, fn_, tn, prec, rec, f1, far
        );
    }

    // === Score distribution ===
    let pre_scores: Vec<f64> = flares
        .iter()
        .filter_map(|f| {
            let d = f.begin.date_naive();
            let scores: Vec<f64> = (1..=3)
                .filter_map(|i| {
                    score_timeline
                        .get(&(d - chrono::Duration::days(i)))
                        .copied()
                })
                .collect();
            if scores.is_empty() {
                None
            } else {
                Some(mean(&scores))
            }
        })
        .collect();

    let non_flare: Vec<f64> = score_timeline
        .iter()
        .filter(|(day, _)| {
            !flare_dates
                .iter()
                .any(|fd| (*fd - **day).num_days().abs() < 7)
        })
        .map(|(_, &s)| s)
        .collect();

    let sep = mean(&pre_scores) - mean(&non_flare);
    let pooled = ((std_dev(&pre_scores).powi(2) + std_dev(&non_flare).powi(2)) / 2.0).sqrt();

    println!("\n--- Score Distribution ---");
    println!(
        "Pre-flare (1-3d): mean={:.4}, std={:.4}, n={}",
        mean(&pre_scores),
        std_dev(&pre_scores),
        pre_scores.len()
    );
    println!(
        "Non-flare:        mean={:.4}, std={:.4}, n={}",
        mean(&non_flare),
        std_dev(&non_flare),
        non_flare.len()
    );
    println!("Separation:       {:.4}", sep);
    if pooled > 1e-8 {
        println!("Cohen's d:        {:.3}", sep / pooled);
    }

    // === Top X-class events ===
    println!("\n--- Top 15 X-class by Pre-Flare Score ---");
    println!(
        "{:<22} {:>6} {:>8} {:>8} {:>+8}",
        "Time", "Class", "PreMean", "BLMean", "Sep"
    );
    println!("{}", "-".repeat(56));

    let mut x_results: Vec<(DateTime<Utc>, String, f64, f64)> = flares
        .iter()
        .filter(|f| f.class.starts_with('X'))
        .filter_map(|f| {
            let d = f.begin.date_naive();
            let pre: Vec<f64> = (1..=3)
                .filter_map(|i| {
                    score_timeline
                        .get(&(d - chrono::Duration::days(i)))
                        .copied()
                })
                .collect();
            let bl: Vec<f64> = (8..=14)
                .filter_map(|i| {
                    score_timeline
                        .get(&(d - chrono::Duration::days(i)))
                        .copied()
                })
                .collect();
            if pre.is_empty() || bl.is_empty() {
                return None;
            }
            Some((f.begin, f.class.clone(), mean(&pre), mean(&bl)))
        })
        .collect();
    x_results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (begin, class, pre, bl) in x_results.iter().take(15) {
        println!(
            "{:<22} {:>6} {:>8.4} {:>8.4} {:>+8.4}",
            begin.format("%Y-%m-%d %H:%M"),
            class,
            pre,
            bl,
            pre - bl
        );
    }
}

fn confusion(
    scores: &BTreeMap<NaiveDate, f64>,
    flare_dates: &[NaiveDate],
    threshold: f64,
    lookahead: i64,
) -> (usize, usize, usize, usize) {
    let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
    for (&day, &score) in scores {
        let pred = score > threshold;
        let actual = flare_dates.iter().any(|&fd| {
            let d = (fd - day).num_days();
            d >= 0 && d <= lookahead
        });
        match (pred, actual) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    (tp, fp, fn_, tn)
}

fn unique_count(dates: &[NaiveDate]) -> usize {
    let mut s = dates.to_vec();
    s.sort();
    s.dedup();
    s.len()
}
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    if v.len() < 2 {
        return 0.0;
    }
    (v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn load_dbdt(path: &str) -> Vec<DailyObs> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() < 9 {
                return None;
            }
            Some(DailyObs {
                date: NaiveDate::parse_from_str(f[0], "%Y-%m-%d").ok()?,
                x_mean: f[5].parse().ok()?,
                y_mean: f[6].parse().ok()?,
                z_mean: f[7].parse().ok()?,
                dbdt_max: f[2].parse().ok()?,
            })
        })
        .collect()
}
fn load_f107(path: &str) -> Vec<(NaiveDate, f64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
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
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() < 4 {
                return None;
            }
            let c = f[3].to_string();
            if !c.starts_with('M') && !c.starts_with('X') {
                return None;
            }
            let b = NaiveDateTime::parse_from_str(f[0].trim(), "%Y-%m-%d %H:%M:%S")
                .ok()?
                .and_utc();
            Some(Flare { begin: b, class: c })
        })
        .collect()
}
