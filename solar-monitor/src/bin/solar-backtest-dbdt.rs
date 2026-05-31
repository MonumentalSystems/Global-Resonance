//! Backtest criticality detector using Fredericksburg dB/dt + daily F10.7
//! against the full solar flare catalog (2010-2024).
//!
//! dB/dt is the time derivative of the geomagnetic field — it directly
//! measures magnetic impulses from solar activity. The mapping:
//!   - F10.7 → proxy X-ray flux (background activity level)
//!   - dB/dt_p95 → proxy hardness (magnetic stress → spectral hardening)
//!   - dB/dt_max → proxy proton flux (sharp impulse → particle acceleration)
//!   - dB/dt_std → modulates J around J_c (variability → approach to criticality)

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Utc};
use solar_monitor::detection::criticality::CriticalityDetector;
use std::collections::BTreeMap;

struct DailyObs {
    date: NaiveDate,
    dbdt_mean: f64,
    dbdt_max: f64,
    dbdt_p95: f64,
    dbdt_std: f64,
    x_mean: f64,
    y_mean: f64,
    z_mean: f64,
}

struct Flare {
    begin: DateTime<Utc>,
    class: String,
    class_numeric: f64,
}

fn main() {
    let dbdt_data = load_dbdt(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/atmospheric-analysis/data/frd_daily_dbdt_2010_2024.csv",
    );
    let f107_daily = load_f107(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/kp_daily.csv",
    );
    let flares = load_flares("solar-monitor/data/catalogs/solar_flares.csv");

    // Only flares within dB/dt coverage (2010-2024).
    let flares: Vec<Flare> = flares
        .into_iter()
        .filter(|f| f.begin.year() >= 2010 && f.begin.year() <= 2024)
        .collect();

    println!("Loaded {} dB/dt daily records", dbdt_data.len());
    println!("Loaded {} daily F10.7 values", f107_daily.len());
    println!("Loaded {} M/X flares (2010-2024)\n", flares.len());

    // Build indexed lookups.
    let dbdt_map: BTreeMap<NaiveDate, &DailyObs> = dbdt_data.iter().map(|d| (d.date, d)).collect();
    let f107_map: BTreeMap<NaiveDate, f64> = f107_daily.into_iter().collect();

    // Run detector over the full timeline at daily cadence.
    // Feed 8 synthetic "minutes" per day to give the lattice time to evolve.
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

        let obs = dbdt_map.get(&day);
        let dbdt_p95 = obs.map(|o| o.dbdt_p95).unwrap_or(0.5);
        let dbdt_max = obs.map(|o| o.dbdt_max).unwrap_or(1.0);
        let dbdt_std = obs.map(|o| o.dbdt_std).unwrap_or(0.2);

        // Map to proxy observables using dB/dt.
        let (xray_long, xray_short, proton_flux) =
            proxy_observables_dbdt(f107, dbdt_p95, dbdt_max, dbdt_std);

        // Feed 8 samples per day (3-hour equivalent) to let lattice dynamics evolve.
        let ts = day.and_hms_opt(12, 0, 0).unwrap().and_utc();
        for hour_offset in [0, 3, 6, 9, 12, 15, 18, 21] {
            let t = ts - chrono::Duration::hours(12) + chrono::Duration::hours(hour_offset);
            det.ingest(xray_long, xray_short, proton_flux, t);
        }

        score_timeline.insert(day, det.score());
        day += chrono::Duration::days(1);
    }

    println!("Processed {} days\n", score_timeline.len());

    // Evaluate each flare.
    let mut results: Vec<FlareResult> = Vec::new();

    for flare in &flares {
        let flare_date = flare.begin.date_naive();

        // Pre-flare: 1-3 days before.
        let pre_scores: Vec<f64> = (1..=3)
            .filter_map(|d| {
                let day = flare_date - chrono::Duration::days(d);
                score_timeline.get(&day).copied()
            })
            .collect();

        // Baseline: 8-14 days before.
        let bl_scores: Vec<f64> = (8..=14)
            .filter_map(|d| {
                let day = flare_date - chrono::Duration::days(d);
                score_timeline.get(&day).copied()
            })
            .collect();

        if pre_scores.is_empty() || bl_scores.is_empty() {
            continue;
        }

        let pre_mean = mean(&pre_scores);
        let bl_mean = mean(&bl_scores);
        let elevated = pre_mean > bl_mean + 0.05;
        let strongly_elevated = pre_mean > bl_mean + 0.15;

        // Also get dB/dt for the pre-flare days.
        let pre_dbdt_max: f64 = (1..=3)
            .filter_map(|d| {
                let day = flare_date - chrono::Duration::days(d);
                dbdt_map.get(&day).map(|o| o.dbdt_max)
            })
            .fold(0.0_f64, f64::max);

        results.push(FlareResult {
            begin: flare.begin,
            class: flare.class.clone(),
            class_numeric: flare.class_numeric,
            pre_mean,
            bl_mean,
            elevated,
            strongly_elevated,
            pre_dbdt_max,
        });
    }

    // Summary.
    let total = results.len();
    let elevated_count = results.iter().filter(|r| r.elevated).count();
    let strong_count = results.iter().filter(|r| r.strongly_elevated).count();

    let x_flares: Vec<&FlareResult> = results
        .iter()
        .filter(|r| r.class.starts_with('X'))
        .collect();
    let m_flares: Vec<&FlareResult> = results
        .iter()
        .filter(|r| r.class.starts_with('M'))
        .collect();
    let x_elevated = x_flares.iter().filter(|r| r.elevated).count();
    let x_strong = x_flares.iter().filter(|r| r.strongly_elevated).count();
    let m_elevated = m_flares.iter().filter(|r| r.elevated).count();
    let m_strong = m_flares.iter().filter(|r| r.strongly_elevated).count();

    println!("=== Criticality + dB/dt Backtest: 2010-2024 M/X Flare Catalog ===\n");
    println!("Total flares evaluated:   {}", total);
    println!(
        "Elevated (pre > bl+0.05): {} ({:.1}%)",
        elevated_count,
        100.0 * elevated_count as f64 / total as f64
    );
    println!(
        "Strongly elevated (+0.15): {} ({:.1}%)\n",
        strong_count,
        100.0 * strong_count as f64 / total as f64
    );

    println!("--- By Class ---");
    println!(
        "X-class: {}/{} elevated ({:.1}%), {}/{} strong ({:.1}%)",
        x_elevated,
        x_flares.len(),
        100.0 * x_elevated as f64 / x_flares.len().max(1) as f64,
        x_strong,
        x_flares.len(),
        100.0 * x_strong as f64 / x_flares.len().max(1) as f64,
    );
    println!(
        "M-class: {}/{} elevated ({:.1}%), {}/{} strong ({:.1}%)\n",
        m_elevated,
        m_flares.len(),
        100.0 * m_elevated as f64 / m_flares.len().max(1) as f64,
        m_strong,
        m_flares.len(),
        100.0 * m_strong as f64 / m_flares.len().max(1) as f64,
    );

    // Top 20 X-class by pre-flare score.
    let mut x_sorted: Vec<&&FlareResult> = x_flares.iter().collect();
    x_sorted.sort_by(|a, b| b.pre_mean.partial_cmp(&a.pre_mean).unwrap());

    println!("--- Top 20 X-class Events by Pre-Flare Criticality Score ---");
    println!(
        "{:<22} {:>6} {:>8} {:>8} {:>8} {:>10} {:>5}",
        "Time", "Class", "PreMean", "BLMean", "Sep", "dBdt_max", "Elev?"
    );
    println!("{}", "-".repeat(75));
    for r in x_sorted.iter().take(20) {
        println!(
            "{:<22} {:>6} {:>8.4} {:>8.4} {:>+8.4} {:>10.2} {:>5}",
            r.begin.format("%Y-%m-%d %H:%M"),
            r.class,
            r.pre_mean,
            r.bl_mean,
            r.pre_mean - r.bl_mean,
            r.pre_dbdt_max,
            if r.elevated { "YES" } else { "no" },
        );
    }

    // Score distribution.
    let pre_all: Vec<f64> = results.iter().map(|r| r.pre_mean).collect();

    // Non-flare control: sample days >7 days from any flare.
    let flare_dates: Vec<NaiveDate> = flares.iter().map(|f| f.begin.date_naive()).collect();
    let non_flare: Vec<f64> = score_timeline
        .iter()
        .filter(|(day, _)| {
            !flare_dates
                .iter()
                .any(|fd| (*fd - **day).num_days().abs() < 7)
        })
        .map(|(_, &s)| s)
        .collect();

    println!("\n--- Score Distribution ---");
    println!(
        "Pre-flare (1-3d before): mean={:.4}, std={:.4}, median={:.4}, n={}",
        mean(&pre_all),
        std_dev(&pre_all),
        median(&pre_all),
        pre_all.len(),
    );
    println!(
        "Non-flare days:          mean={:.4}, std={:.4}, median={:.4}, n={}",
        mean(&non_flare),
        std_dev(&non_flare),
        median(&non_flare),
        non_flare.len(),
    );
    let sep = mean(&pre_all) - mean(&non_flare);
    let pooled = ((std_dev(&pre_all).powi(2) + std_dev(&non_flare).powi(2)) / 2.0).sqrt();
    println!("Separation:              {:.4}", sep);
    if pooled > 1e-8 {
        println!("Cohen's d:               {:.3}", sep / pooled);
    }

    // dB/dt correlation: do high-dB/dt days have higher criticality scores?
    println!("\n--- dB/dt vs Criticality Score Correlation ---");
    let mut dbdt_score_pairs: Vec<(f64, f64)> = Vec::new();
    for (day, &score) in &score_timeline {
        if let Some(obs) = dbdt_map.get(day) {
            dbdt_score_pairs.push((obs.dbdt_p95, score));
        }
    }
    if dbdt_score_pairs.len() > 10 {
        let xs: Vec<f64> = dbdt_score_pairs.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f64> = dbdt_score_pairs.iter().map(|(_, y)| *y).collect();
        let r = pearson_r(&xs, &ys);
        println!(
            "Pearson r(dB/dt_p95, criticality_score) = {:.4} (n={})",
            r,
            dbdt_score_pairs.len()
        );
    }
}

/// Map F10.7 and dB/dt to proxy solar observables.
fn proxy_observables_dbdt(
    f107: f64,
    dbdt_p95: f64,
    dbdt_max: f64,
    dbdt_std: f64,
) -> (f64, f64, f64) {
    // F10.7 → X-ray flux background.
    let xray_long = if f107 > 65.0 {
        let t = ((f107 - 70.0) / 230.0).clamp(0.0, 1.0);
        10.0_f64.powf(-7.0 + 2.0 * t)
    } else {
        1e-7
    };

    // dB/dt_p95 → spectral hardness.
    // Quiet: dB/dt_p95 ≈ 0.5-1.0 nT/s → hardness 0.04
    // Active: dB/dt_p95 ≈ 2-5 nT/s → hardness 0.10-0.20
    // Storm: dB/dt_p95 > 10 nT/s → hardness 0.25+
    let hardness = if dbdt_p95 > 5.0 {
        0.15 + 0.10 * ((dbdt_p95 - 5.0) / 15.0).min(1.0)
    } else if dbdt_p95 > 1.0 {
        0.04 + 0.11 * ((dbdt_p95 - 1.0) / 4.0)
    } else {
        0.04
    };
    let xray_short = xray_long * hardness;

    // dB/dt_max → proton flux proxy.
    // Sharp magnetic impulses indicate reconnection/SEP arrival.
    // Quiet: max ≈ 1-3 nT/s → proton 0.3 pfu
    // Active: max ≈ 5-20 nT/s → proton 1-10 pfu
    // Storm: max > 50 nT/s → proton 10+ pfu
    let proton_flux = if dbdt_max > 10.0 {
        1.0 + 2.0 * (dbdt_max / 10.0).log10().min(2.0)
    } else if dbdt_max > 3.0 {
        0.3 + 0.7 * ((dbdt_max - 3.0) / 7.0)
    } else {
        0.3
    };

    (xray_long, xray_short, proton_flux)
}

fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let cov: f64 = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| (x - mx) * (y - my))
        .sum::<f64>();
    let sx: f64 = xs.iter().map(|&x| (x - mx).powi(2)).sum::<f64>();
    let sy: f64 = ys.iter().map(|&y| (y - my).powi(2)).sum::<f64>();
    if sx * sy < 1e-16 {
        return 0.0;
    }
    cov / (sx * sy).sqrt()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}
fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    if v.len() < 2 {
        return 0.0;
    }
    (v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn load_dbdt(path: &str) -> Vec<DailyObs> {
    let content = std::fs::read_to_string(path).unwrap();
    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 9 {
                return None;
            }
            let date = NaiveDate::parse_from_str(f[0], "%Y-%m-%d").ok()?;
            Some(DailyObs {
                date,
                dbdt_mean: f[1].parse().ok()?,
                dbdt_max: f[2].parse().ok()?,
                dbdt_p95: f[3].parse().ok()?,
                dbdt_std: f[4].parse().ok()?,
                x_mean: f[5].parse().ok()?,
                y_mean: f[6].parse().ok()?,
                z_mean: f[7].parse().ok()?,
            })
        })
        .collect()
}

fn load_f107(path: &str) -> Vec<(NaiveDate, f64)> {
    let content = std::fs::read_to_string(path).unwrap();
    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 10 {
                return None;
            }
            let y: i32 = f[0].parse().ok()?;
            let m: u32 = f[1].parse().ok()?;
            let d: u32 = f[2].parse().ok()?;
            if y < 2009 {
                return None;
            }
            let date = NaiveDate::from_ymd_opt(y, m, d)?;
            let f107: f64 = f[9].parse().ok()?;
            if f107 > 0.0 {
                Some((date, f107))
            } else {
                None
            }
        })
        .collect()
}

fn load_flares(path: &str) -> Vec<Flare> {
    let content = std::fs::read_to_string(path).unwrap();
    content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 7 {
                return None;
            }
            let class = f[3].to_string();
            if !class.starts_with('M') && !class.starts_with('X') {
                return None;
            }
            let begin = NaiveDateTime::parse_from_str(f[0].trim(), "%Y-%m-%d %H:%M:%S")
                .ok()?
                .and_utc();
            let cn: f64 = f[6].parse().ok()?;
            Some(Flare {
                begin,
                class,
                class_numeric: cn,
            })
        })
        .collect()
}

struct FlareResult {
    begin: DateTime<Utc>,
    class: String,
    class_numeric: f64,
    pre_mean: f64,
    bl_mean: f64,
    elevated: bool,
    strongly_elevated: bool,
    pre_dbdt_max: f64,
}
