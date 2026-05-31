//! Backtest escalation against real dual-channel GOES-16 XRS data.
//!
//! Usage: solar-backtest-goes --goes <csv> --flares <csv>

use chrono::Datelike;
use std::path::PathBuf;

use solar_monitor::backtest::goes_loader;
use solar_monitor::backtest::loaders;
use solar_monitor::detection::escalation::{EscalationLevel, EscalationMonitor};
use solar_monitor::detection::rank_fusion::RankFusionDetector;

fn main() {
    let mut goes_path = PathBuf::from("solar-monitor/data/goes_data/goes16_xrs_2024.csv");
    let mut flares_path = PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--goes" => {
                i += 1;
                goes_path = PathBuf::from(&args[i]);
            }
            "--flares" => {
                i += 1;
                flares_path = PathBuf::from(&args[i]);
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Escalation Backtest — Real GOES-16 Dual-Channel Data ===\n");

    // Load data
    print!("Loading GOES XRS data... ");
    let goes = goes_loader::load_goes_csv(&goes_path).unwrap();
    println!(
        "{} records ({} to {})",
        goes.len(),
        goes.first()
            .map(|r| r.timestamp.format("%Y-%m-%d").to_string())
            .unwrap_or_default(),
        goes.last()
            .map(|r| r.timestamp.format("%Y-%m-%d").to_string())
            .unwrap_or_default(),
    );

    print!("Loading flare catalog... ");
    let all_flares = loaders::load_flares(&flares_path).unwrap();
    // Filter to GOES data range
    let start = goes.first().unwrap().timestamp;
    let end = goes.last().unwrap().timestamp;
    let flares: Vec<_> = all_flares
        .into_iter()
        .filter(|f| f.begin >= start && f.begin <= end)
        .collect();
    println!("{} M/X-class flares in range", flares.len());
    println!();

    // Run escalation with real dual-channel data
    let mut fusion = RankFusionDetector::new(0.7);
    let mut esc = EscalationMonitor::new();

    // Track results per flare
    struct FlareResult {
        class: String,
        level_at_onset: EscalationLevel,
        lead_minutes: Option<i64>,
    }
    let mut results: Vec<FlareResult> = Vec::new();

    let mut entered_elevated: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut prev_level = EscalationLevel::Quiet;
    let mut next_flare_idx = 0;
    let mut transition_count = 0usize;

    for rec in &goes {
        // Feed real dual-channel data directly (no synthetic estimation!)
        fusion.ingest(
            rec.xrsb, // long channel (0.1-0.8nm)
            rec.xrsa, // short channel (0.05-0.4nm)
            100.0,    // electron proxy (no electron data in XRS files)
            0.3,      // proton proxy
            rec.timestamp,
        );

        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();

        if let Some(transition) = esc.update_with_flux(h, f, a, rec.xrsb, rec.timestamp) {
            transition_count += 1;
            match transition.to {
                EscalationLevel::Elevated if prev_level < EscalationLevel::Elevated => {
                    entered_elevated = Some(rec.timestamp);
                }
                EscalationLevel::Active if prev_level < EscalationLevel::Active => {
                    if entered_elevated.is_none() {
                        entered_elevated = Some(rec.timestamp);
                    }
                }
                EscalationLevel::Flare if prev_level < EscalationLevel::Flare => {
                    if entered_elevated.is_none() {
                        entered_elevated = Some(rec.timestamp);
                    }
                }
                EscalationLevel::Quiet => {
                    entered_elevated = None;
                }
                _ => {}
            }
            prev_level = transition.to;
        }

        // Check flare onsets
        while next_flare_idx < flares.len() && rec.timestamp >= flares[next_flare_idx].begin {
            let flare = &flares[next_flare_idx];
            let lead = entered_elevated
                .map(|t| (flare.begin - t).num_minutes())
                .filter(|&m| m > 0);

            results.push(FlareResult {
                class: flare.class.clone(),
                level_at_onset: esc.level,
                lead_minutes: lead,
            });
            next_flare_idx += 1;
        }
    }

    // Report
    let total = results.len();
    println!("Total transitions: {}", transition_count);
    println!("Total M/X flares: {}\n", total);

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
    let warned = at_elevated + at_active + at_flare;

    println!("=== Escalation Level at Flare Onset ===\n");
    println!(
        "  QUIET:    {:>4} ({:.1}%)",
        at_quiet,
        at_quiet as f64 / total as f64 * 100.0
    );
    println!(
        "  ELEVATED: {:>4} ({:.1}%)",
        at_elevated,
        at_elevated as f64 / total as f64 * 100.0
    );
    println!(
        "  ACTIVE:   {:>4} ({:.1}%)",
        at_active,
        at_active as f64 / total as f64 * 100.0
    );
    println!(
        "  FLARE:    {:>4} ({:.1}%)",
        at_flare,
        at_flare as f64 / total as f64 * 100.0
    );
    println!(
        "  Warned:   {:>4} ({:.1}%)",
        warned,
        warned as f64 / total as f64 * 100.0
    );

    // By class
    println!("\n=== By Class ===\n");
    for prefix in &["M", "X"] {
        let cr: Vec<&FlareResult> = results
            .iter()
            .filter(|r| r.class.starts_with(prefix))
            .collect();
        if cr.is_empty() {
            continue;
        }
        let n = cr.len();
        let w = cr
            .iter()
            .filter(|r| r.level_at_onset >= EscalationLevel::Elevated)
            .count();
        let af = cr
            .iter()
            .filter(|r| r.level_at_onset >= EscalationLevel::Active)
            .count();
        println!(
            "  {}-class: {}/{} warned ({:.1}%), {}/{} at ACTIVE+ ({:.1}%)",
            prefix,
            w,
            n,
            w as f64 / n as f64 * 100.0,
            af,
            n,
            af as f64 / n as f64 * 100.0
        );
    }

    // Lead times
    let leads: Vec<i64> = results.iter().filter_map(|r| r.lead_minutes).collect();
    if !leads.is_empty() {
        println!("\n=== Lead Time (≥ELEVATED before onset) ===\n");
        let mean = leads.iter().sum::<i64>() as f64 / leads.len() as f64;
        let mut sorted = leads.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        println!("  Flares with lead: {}/{}", leads.len(), total);
        println!("  Mean:   {:.0} min ({:.1} hours)", mean, mean / 60.0);
        println!(
            "  Median: {} min ({:.1} hours)",
            median,
            median as f64 / 60.0
        );
        println!("  Min:    {} min", sorted[0]);
        println!(
            "  Max:    {} min ({:.1} hours)",
            sorted[sorted.len() - 1],
            sorted[sorted.len() - 1] as f64 / 60.0
        );

        println!("\n  Distribution:");
        for (label, lo, hi) in &[
            ("0-10 min", 0i64, 10),
            ("10-30 min", 10, 30),
            ("30-60 min", 30, 60),
            ("1-3 hours", 60, 180),
            ("3-6 hours", 180, 360),
            ("6-12 hours", 360, 720),
            ("12-24 hours", 720, 1440),
            (">24 hours", 1440, 999999),
        ] {
            let count = leads.iter().filter(|&&m| m >= *lo && m < *hi).count();
            if count > 0 {
                let bar = "#".repeat((count as f64 / leads.len() as f64 * 40.0) as usize);
                println!(
                    "    {:<14} {:>4} ({:>5.1}%) {}",
                    label,
                    count,
                    count as f64 / leads.len() as f64 * 100.0,
                    bar
                );
            }
        }
    }
}
