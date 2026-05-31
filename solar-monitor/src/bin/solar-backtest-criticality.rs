//! Full-catalog backtest of the criticality detector against 2,625 M/X-class
//! solar flares (2010-2026) using 3-hourly Kp and daily F10.7 data.
//!
//! Since we don't have minute-cadence GOES XRS for the full period, we use
//! proxy mappings from geomagnetic/solar indices:
//!   - F10.7 → proxy X-ray flux (well-known correlation)
//!   - Kp → proxy spectral hardness (disturbed → harder spectrum)
//!   - dKp/dt → proxy proton flux (rapid Kp rise → SEP arrival)
//!
//! For each flare, we measure the criticality score in a 24h window before
//! onset and compare to the 7-day baseline before that window.

use chrono::{DateTime, NaiveDateTime, Utc};
use solar_monitor::detection::criticality::CriticalityDetector;
use std::collections::BTreeMap;

/// A cataloged solar flare.
struct Flare {
    begin: DateTime<Utc>,
    class: String,
    class_numeric: f64,
}

/// A 3-hourly Kp observation.
struct KpObs {
    datetime: DateTime<Utc>,
    kp: f64,
    dkp_dt: f64,
}

fn main() {
    // Load data.
    let flares = load_flares("solar-monitor/data/catalogs/solar_flares.csv");
    let kp_data = load_kp_3hourly("solar-monitor/data/catalogs/kp_3hourly.csv");
    let f107_daily = load_kp_daily_f107(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/kp_daily.csv",
    );

    println!("Loaded {} M/X-class flares", flares.len());
    println!("Loaded {} Kp 3-hourly observations", kp_data.len());
    println!("Loaded {} daily F10.7 values", f107_daily.len());

    // Build time-indexed lookup for F10.7 (daily → nearest).
    let f107_map: BTreeMap<DateTime<Utc>, f64> = f107_daily.into_iter().collect();

    // Run detector over full Kp timeline, recording scores at each step.
    let mut det = CriticalityDetector::default_detector();
    let mut score_timeline: Vec<(DateTime<Utc>, f64)> = Vec::with_capacity(kp_data.len());

    for obs in &kp_data {
        // Map Kp/F10.7 to proxy solar observables.
        let f107 = find_nearest_f107(&f107_map, obs.datetime);
        let (xray_long, xray_short, proton_flux) = proxy_observables(f107, obs.kp, obs.dkp_dt);

        det.ingest(xray_long, xray_short, proton_flux, obs.datetime);
        score_timeline.push((obs.datetime, det.score()));
    }

    println!("Processed {} timesteps\n", score_timeline.len());

    // For each flare, measure pre-flare score and baseline score.
    let mut results: Vec<FlareResult> = Vec::new();

    for flare in &flares {
        // Pre-flare window: -24h to -1h before onset.
        let pre_start = flare.begin - chrono::Duration::hours(24);
        let pre_end = flare.begin - chrono::Duration::hours(1);

        // Baseline window: -8d to -1d before onset.
        let bl_start = flare.begin - chrono::Duration::days(8);
        let bl_end = flare.begin - chrono::Duration::days(1);

        let pre_scores: Vec<f64> = score_timeline
            .iter()
            .filter(|(t, _)| *t >= pre_start && *t <= pre_end)
            .map(|(_, s)| *s)
            .collect();

        let bl_scores: Vec<f64> = score_timeline
            .iter()
            .filter(|(t, _)| *t >= bl_start && *t <= bl_end)
            .map(|(_, s)| *s)
            .collect();

        if pre_scores.is_empty() || bl_scores.is_empty() {
            continue;
        }

        let pre_mean = mean(&pre_scores);
        let pre_max = pre_scores.iter().cloned().fold(0.0_f64, f64::max);
        let bl_mean = mean(&bl_scores);
        let elevated = pre_mean > bl_mean + 0.05;
        let strongly_elevated = pre_mean > bl_mean + 0.15;

        results.push(FlareResult {
            begin: flare.begin,
            class: flare.class.clone(),
            class_numeric: flare.class_numeric,
            pre_mean,
            pre_max,
            bl_mean,
            elevated,
            strongly_elevated,
        });
    }

    // Summary statistics.
    let total = results.len();
    let elevated_count = results.iter().filter(|r| r.elevated).count();
    let strong_count = results.iter().filter(|r| r.strongly_elevated).count();

    // Split by class.
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

    println!("=== Criticality Detector Backtest: 2010-2026 M/X Flare Catalog ===\n");
    println!("Total flares evaluated:  {}", total);
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

    // Top 20 X-class events with highest pre-flare scores.
    let mut x_sorted: Vec<&&FlareResult> = x_flares.iter().collect();
    x_sorted.sort_by(|a, b| b.pre_mean.partial_cmp(&a.pre_mean).unwrap());

    println!("--- Top 20 X-class Events by Pre-Flare Score ---");
    println!(
        "{:<22} {:>6} {:>8} {:>8} {:>8} {:>5}",
        "Time", "Class", "PreMean", "PreMax", "BLMean", "Elev?"
    );
    println!("{}", "-".repeat(62));
    for r in x_sorted.iter().take(20) {
        println!(
            "{:<22} {:>6} {:>8.4} {:>8.4} {:>8.4} {:>5}",
            r.begin.format("%Y-%m-%d %H:%M"),
            r.class,
            r.pre_mean,
            r.pre_max,
            r.bl_mean,
            if r.elevated { "YES" } else { "no" },
        );
    }

    // Score distribution for pre-flare vs random baseline periods.
    // Sample 1000 random 24h windows from the timeline as non-flare control.
    let mut non_flare_scores: Vec<f64> = Vec::new();
    let step = score_timeline.len() / 1000;
    for i in (0..score_timeline.len()).step_by(step.max(1)) {
        let t = score_timeline[i].0;
        // Skip if within 48h of any flare.
        let near_flare = flares.iter().any(|f| {
            let dt = (f.begin - t).num_hours().abs();
            dt < 48
        });
        if !near_flare {
            // Average score over 24h window starting at t.
            let window_end = t + chrono::Duration::hours(24);
            let window_scores: Vec<f64> = score_timeline
                .iter()
                .filter(|(ts, _)| *ts >= t && *ts <= window_end)
                .map(|(_, s)| *s)
                .collect();
            if !window_scores.is_empty() {
                non_flare_scores.push(mean(&window_scores));
            }
        }
    }

    let pre_flare_all: Vec<f64> = results.iter().map(|r| r.pre_mean).collect();

    println!("\n--- Score Distribution ---");
    println!(
        "Pre-flare windows:  mean={:.4}, std={:.4}, median={:.4}",
        mean(&pre_flare_all),
        std_dev(&pre_flare_all),
        median(&pre_flare_all),
    );
    println!(
        "Non-flare windows:  mean={:.4}, std={:.4}, median={:.4}",
        mean(&non_flare_scores),
        std_dev(&non_flare_scores),
        median(&non_flare_scores),
    );
    let separation = mean(&pre_flare_all) - mean(&non_flare_scores);
    println!("Separation:         {:.4}", separation);

    // Cohen's d effect size.
    let pooled_std =
        ((std_dev(&pre_flare_all).powi(2) + std_dev(&non_flare_scores).powi(2)) / 2.0).sqrt();
    if pooled_std > 1e-8 {
        println!("Cohen's d:          {:.3}", separation / pooled_std);
    }
}

/// Map F10.7 and Kp to proxy X-ray flux, hardness, and proton flux.
///
/// F10.7 correlates strongly with overall solar X-ray background:
///   - F10.7 = 70 (solar min) → ~B1 (1e-7 W/m²)
///   - F10.7 = 150 (moderate) → ~C1 (1e-6 W/m²)
///   - F10.7 = 250 (solar max) → ~C5 (5e-6 W/m²)
///
/// Kp modulates hardness (high Kp = geomagnetic storm = harder solar input):
///   - Kp 0-2: quiet → ratio 0.04
///   - Kp 3-5: moderate → ratio 0.08-0.15
///   - Kp 6+: storm → ratio 0.20+
///
/// dKp/dt > 0 indicates rapid geomagnetic onset → proton flux proxy.
fn proxy_observables(f107: f64, kp: f64, dkp_dt: f64) -> (f64, f64, f64) {
    // F10.7 → X-ray flux (log-linear relationship).
    let xray_long = if f107 > 65.0 {
        // Map [70, 300] → [1e-7, 1e-5] log-linearly.
        let t = ((f107 - 70.0) / 230.0).clamp(0.0, 1.0);
        10.0_f64.powf(-7.0 + 2.0 * t)
    } else {
        1e-7 // floor at B1
    };

    // Kp → hardness ratio.
    let hardness = if kp >= 6.0 {
        0.20 + 0.05 * ((kp - 6.0) / 3.0).min(1.0)
    } else if kp >= 3.0 {
        0.06 + 0.14 * ((kp - 3.0) / 3.0)
    } else {
        0.04 + 0.02 * (kp / 3.0)
    };
    let xray_short = xray_long * hardness;

    // dKp/dt → proton flux proxy.
    // Rapid positive dKp/dt indicates sudden storm commencement → SEP.
    let proton_flux = if dkp_dt > 1.0 {
        0.3 + 2.0 * (dkp_dt - 1.0).min(5.0)
    } else {
        0.3 // quiet background
    };

    (xray_long, xray_short, proton_flux)
}

fn find_nearest_f107(map: &BTreeMap<DateTime<Utc>, f64>, ts: DateTime<Utc>) -> f64 {
    map.range(..=ts)
        .next_back()
        .map(|(_, &v)| v)
        .unwrap_or(100.0)
}

fn load_flares(path: &str) -> Vec<Flare> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut flares = Vec::new();
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }
        let class = fields[3].to_string();
        // Only M and X class.
        if !class.starts_with('M') && !class.starts_with('X') {
            continue;
        }
        let begin = match parse_ts_opt(fields[0]) {
            Some(t) => t,
            None => continue,
        };
        let class_numeric: f64 = fields[6].parse().unwrap_or(0.0);
        flares.push(Flare {
            begin,
            class,
            class_numeric,
        });
    }
    flares
}

fn load_kp_3hourly(path: &str) -> Vec<KpObs> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut obs = Vec::new();
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9 {
            continue;
        }
        let datetime = match parse_ts_opt(fields[6]) {
            Some(t) => t,
            None => continue,
        };
        let kp: f64 = fields[4].parse().unwrap_or(0.0);
        let dkp_dt: f64 = fields[8].parse().unwrap_or(0.0);
        obs.push(KpObs {
            datetime,
            kp,
            dkp_dt,
        });
    }
    obs
}

fn load_kp_daily_f107(path: &str) -> Vec<(DateTime<Utc>, f64)> {
    let content = std::fs::read_to_string(path).unwrap();
    let mut data = Vec::new();
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 10 {
            continue;
        }
        let year: i32 = fields[0].parse().unwrap_or(0);
        let month: u32 = fields[1].parse().unwrap_or(0);
        let day: u32 = fields[2].parse().unwrap_or(0);
        if year < 2009 {
            continue; // Only need data from just before the flare catalog starts
        }
        let dt = match chrono::NaiveDate::from_ymd_opt(year, month, day) {
            Some(d) => d.and_hms_opt(12, 0, 0).unwrap().and_utc(),
            None => continue,
        };
        let f107: f64 = fields[9].parse().unwrap_or(0.0);
        if f107 > 0.0 {
            data.push((dt, f107));
        }
    }
    data
}

fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M:%S%.fZ"))
        .ok()
        .map(|ndt| ndt.and_utc())
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
    let var = v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

struct FlareResult {
    begin: DateTime<Utc>,
    class: String,
    class_numeric: f64,
    pre_mean: f64,
    pre_max: f64,
    bl_mean: f64,
    elevated: bool,
    strongly_elevated: bool,
}
