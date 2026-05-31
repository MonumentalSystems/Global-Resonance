//! Backtest the escalation state machine against historical data.
//! Uses synthetic minute-cadence X-ray from the flare catalog.
//! Tests: does escalation reach ACTIVE/FLARE before known M/X events?

use chrono::{Datelike, NaiveDateTime};
use std::path::PathBuf;

use solar_monitor::backtest::loaders::{self, FlareEvent};
use solar_monitor::backtest::synthesize;
use solar_monitor::detection::escalation::{EscalationLevel, EscalationMonitor};
use solar_monitor::detection::rank_fusion::RankFusionDetector;

fn main() {
    let data_dir = PathBuf::from(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind",
    );

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
            _ => {}
        }
        i += 1;
    }

    println!("=== Escalation Backtest {}-{} ===\n", year_start, year_end);

    let all_flares = loaders::load_flares(&data_dir.join("solar_flares.csv")).unwrap();
    let omni = loaders::load_omni(&data_dir.join("omni_hourly.csv")).unwrap();
    let kp = loaders::load_kp(&data_dir.join("kp_3hourly.csv")).unwrap_or_default();

    let flares: Vec<FlareEvent> = all_flares
        .into_iter()
        .filter(|f| f.begin.year() >= year_start && f.begin.year() <= year_end)
        .collect();
    println!("Flares: {} M/X-class", flares.len());

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

    print!("Synthesizing minute-cadence X-ray... ");
    let xray_minute = synthesize::synthesize_xray_minute(&flares, start, end);
    println!("{} samples", xray_minute.len());

    print!("Merging with OMNI/Kp... ");
    let records = synthesize::merge_minute_cadence(&xray_minute, &omni, &kp, &flares);
    println!("{} records\n", records.len());

    // Run escalation
    let mut fusion = RankFusionDetector::new(0.7);
    let mut esc = EscalationMonitor::new();

    // For each flare, track: what level were we at when it started?
    // And how long before onset did we reach ELEVATED/ACTIVE?
    struct FlareResult {
        class: String,
        begin: chrono::DateTime<chrono::Utc>,
        level_at_onset: EscalationLevel,
        minutes_elevated_before: Option<i64>,
        minutes_active_before: Option<i64>,
        minutes_flare_before: Option<i64>,
    }

    let mut results: Vec<FlareResult> = Vec::new();

    // Track when we entered each level
    let mut entered_elevated: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut entered_active: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut entered_flare: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut prev_level = EscalationLevel::Quiet;

    // Track which flare we're approaching
    let mut next_flare_idx = 0;

    for rec in &records {
        let electron = estimate_electron_flux(rec);
        fusion.ingest_simple(rec.xray_flux, electron, rec.timestamp);

        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();

        if let Some(transition) = esc.update(h, f, a, rec.timestamp) {
            // Track level entry times
            match transition.to {
                EscalationLevel::Elevated if prev_level < EscalationLevel::Elevated => {
                    entered_elevated = Some(rec.timestamp);
                }
                EscalationLevel::Active if prev_level < EscalationLevel::Active => {
                    entered_active = Some(rec.timestamp);
                }
                EscalationLevel::Flare if prev_level < EscalationLevel::Flare => {
                    entered_flare = Some(rec.timestamp);
                }
                EscalationLevel::Quiet => {
                    entered_elevated = None;
                    entered_active = None;
                    entered_flare = None;
                }
                _ => {}
            }
            prev_level = transition.to;
        }

        // Check if we've reached a flare onset
        while next_flare_idx < flares.len() && rec.timestamp >= flares[next_flare_idx].begin {
            let flare = &flares[next_flare_idx];
            let level_at_onset = esc.level;

            let mins_elev = entered_elevated
                .map(|t| (flare.begin - t).num_minutes())
                .filter(|&m| m > 0);
            let mins_active = entered_active
                .map(|t| (flare.begin - t).num_minutes())
                .filter(|&m| m > 0);
            let mins_flare = entered_flare
                .map(|t| (flare.begin - t).num_minutes())
                .filter(|&m| m > 0);

            results.push(FlareResult {
                class: flare.class.clone(),
                begin: flare.begin,
                level_at_onset: level_at_onset,
                minutes_elevated_before: mins_elev,
                minutes_active_before: mins_active,
                minutes_flare_before: mins_flare,
            });

            next_flare_idx += 1;
        }
    }

    // Summarize
    println!("=== Escalation Level at Flare Onset ===\n");

    let total = results.len();
    let at_quiet = results
        .iter()
        .filter(|r| r.level_at_onset == EscalationLevel::Quiet)
        .count();
    let at_elevated = results
        .iter()
        .filter(|r| r.level_at_onset == EscalationLevel::Elevated)
        .count();
    let at_active = results
        .iter()
        .filter(|r| r.level_at_onset == EscalationLevel::Active)
        .count();
    let at_flare = results
        .iter()
        .filter(|r| r.level_at_onset == EscalationLevel::Flare)
        .count();

    println!("Total M/X flares: {}", total);
    println!(
        "  At QUIET:    {:>4} ({:.1}%) — no warning",
        at_quiet,
        at_quiet as f64 / total as f64 * 100.0
    );
    println!(
        "  At ELEVATED: {:>4} ({:.1}%) — precursor detected",
        at_elevated,
        at_elevated as f64 / total as f64 * 100.0
    );
    println!(
        "  At ACTIVE:   {:>4} ({:.1}%) — active region monitored",
        at_active,
        at_active as f64 / total as f64 * 100.0
    );
    println!(
        "  At FLARE:    {:>4} ({:.1}%) — already in flare state",
        at_flare,
        at_flare as f64 / total as f64 * 100.0
    );
    println!(
        "  Warned (≥ELEVATED): {:>4} ({:.1}%)",
        at_elevated + at_active + at_flare,
        (at_elevated + at_active + at_flare) as f64 / total as f64 * 100.0
    );

    // By class
    println!("\n=== By Class ===\n");
    for prefix in &["M", "X"] {
        let class_results: Vec<&FlareResult> = results
            .iter()
            .filter(|r| r.class.starts_with(prefix))
            .collect();
        if class_results.is_empty() {
            continue;
        }
        let n = class_results.len();
        let warned = class_results
            .iter()
            .filter(|r| r.level_at_onset >= EscalationLevel::Elevated)
            .count();
        let active_or_flare = class_results
            .iter()
            .filter(|r| r.level_at_onset >= EscalationLevel::Active)
            .count();
        println!(
            "  {}-class: {}/{} warned ({:.1}%), {}/{} at ACTIVE+ ({:.1}%)",
            prefix,
            warned,
            n,
            warned as f64 / n as f64 * 100.0,
            active_or_flare,
            n,
            active_or_flare as f64 / n as f64 * 100.0
        );
    }

    // Lead time distribution
    println!("\n=== Lead Time (minutes before onset at ≥ELEVATED) ===\n");
    let lead_times: Vec<i64> = results
        .iter()
        .filter_map(|r| r.minutes_elevated_before)
        .collect();
    if !lead_times.is_empty() {
        let mean = lead_times.iter().sum::<i64>() as f64 / lead_times.len() as f64;
        let median = {
            let mut sorted = lead_times.clone();
            sorted.sort();
            sorted[sorted.len() / 2]
        };
        let min = *lead_times.iter().min().unwrap();
        let max = *lead_times.iter().max().unwrap();
        println!("  Flares with lead time: {}/{}", lead_times.len(), total);
        println!("  Mean:   {:.0} min ({:.1} hours)", mean, mean / 60.0);
        println!(
            "  Median: {} min ({:.1} hours)",
            median,
            median as f64 / 60.0
        );
        println!("  Min:    {} min", min);
        println!("  Max:    {} min ({:.1} hours)", max, max as f64 / 60.0);

        // Distribution
        println!("\n  Lead time distribution:");
        for (label, lo, hi) in &[
            ("0-10 min", 0, 10),
            ("10-30 min", 10, 30),
            ("30-60 min", 30, 60),
            ("1-3 hours", 60, 180),
            ("3-6 hours", 180, 360),
            ("6-12 hours", 360, 720),
            ("12-24 hours", 720, 1440),
            (">24 hours", 1440, 999999),
        ] {
            let count = lead_times.iter().filter(|&&m| m >= *lo && m < *hi).count();
            if count > 0 {
                let bar = "#".repeat((count as f64 / lead_times.len() as f64 * 40.0) as usize);
                println!(
                    "    {:<14} {:>4} ({:>5.1}%) {}",
                    label,
                    count,
                    count as f64 / lead_times.len() as f64 * 100.0,
                    bar
                );
            }
        }
    }
}

fn estimate_electron_flux(rec: &loaders::HistoricalRecord) -> f64 {
    let base = 100.0;
    let sf = if let Some(v) = rec.solar_wind_speed {
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
    let df = if let Some(d) = rec.dst {
        if d < -50.0 {
            ((-d) / 50.0).min(10.0)
        } else {
            1.0
        }
    } else {
        1.0
    };
    base * sf * df
}
