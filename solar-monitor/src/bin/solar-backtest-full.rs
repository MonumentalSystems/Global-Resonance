//! Full-stack backtest: GOES dual-channel XRS + OMNI solar wind + all detectors.
//!
//! Merges 1-min GOES data with hourly OMNI data (Bz, speed, density, Dst)
//! and runs all 9 detectors + escalation against the historical record.

use chrono::{DateTime, Datelike, NaiveDateTime, Utc};
use std::collections::BTreeMap;
use std::path::PathBuf;

use solar_monitor::backtest::{goes_loader, loaders};
use solar_monitor::detection::bz_southward::BzSouthwardDetector;
use solar_monitor::detection::escalation::{EscalationLevel, EscalationMonitor};
use solar_monitor::detection::flare_clustering::FlareClusteringDetector;
use solar_monitor::detection::pressure_jump::PressureJumpDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;

fn main() {
    let mut goes_path = PathBuf::from("solar-monitor/data/goes_data/goes16_xrs_2024.csv");
    let omni_path = PathBuf::from(
        "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind/omni_hourly.csv",
    );
    let flares_path = PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--goes" => {
                i += 1;
                goes_path = PathBuf::from(&args[i]);
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Full-Stack Solar Backtest ===\n");

    // Load GOES dual-channel
    print!("Loading GOES XRS... ");
    let goes = goes_loader::load_goes_csv(&goes_path).unwrap();
    println!(
        "{} records ({} to {})",
        goes.len(),
        goes.first().unwrap().timestamp.format("%Y-%m-%d"),
        goes.last().unwrap().timestamp.format("%Y-%m-%d")
    );

    // Load OMNI (hourly solar wind + Dst)
    print!("Loading OMNI... ");
    let omni = loaders::load_omni(&omni_path).unwrap();
    println!("{} records", omni.len());

    // Load flare catalog
    print!("Loading flares... ");
    let start = goes.first().unwrap().timestamp;
    let end = goes.last().unwrap().timestamp;
    let all_flares = loaders::load_flares(&flares_path).unwrap();
    let flares: Vec<_> = all_flares
        .into_iter()
        .filter(|f| f.begin >= start && f.begin <= end)
        .collect();
    println!("{} M/X-class in range\n", flares.len());

    // Build OMNI lookup (hold-previous interpolation to 1-min)
    let omni_vec: Vec<(&DateTime<Utc>, &loaders::OmniRecord)> = omni.iter().collect();

    // Initialize all detectors
    let mut fusion = RankFusionDetector::new(0.7);
    let mut bz_det = BzSouthwardDetector::default_detector();
    let mut pressure_det = PressureJumpDetector::default_detector();
    let mut cluster_det = FlareClusteringDetector::default_detector();
    let mut esc = EscalationMonitor::new();

    // Track results per flare
    struct FlareResult {
        class: String,
        esc_level: EscalationLevel,
        fused_score: f64,
        bz_score: f64,
        pressure_score: f64,
        cluster_score: f64,
        bz_at_onset: f64,
        speed_at_onset: f64,
        dst_at_onset: f64,
    }
    let mut results: Vec<FlareResult> = Vec::new();

    // Track time spent at each escalation level
    let mut minutes_at_level = [0u64; 4]; // Quiet, Elevated, Active, Flare
    let mut next_flare_idx = 0;
    let mut prev_level = EscalationLevel::Quiet;

    // Track new detector firing stats
    let mut bz_fires = 0u64;
    let mut pressure_fires = 0u64;
    let mut cluster_fires = 0u64;

    print!("Running full pipeline... ");
    for rec in &goes {
        // Find nearest OMNI record
        let omni_rec = find_nearest_omni(&omni_vec, rec.timestamp);
        let bz = omni_rec
            .map(|o| o.bz_gsm)
            .filter(|v| !v.is_nan())
            .unwrap_or(0.0);
        let speed = omni_rec
            .map(|o| o.v_sw)
            .filter(|v| !v.is_nan())
            .unwrap_or(400.0);
        let density = omni_rec
            .map(|o| o.n_proton)
            .filter(|v| !v.is_nan())
            .unwrap_or(5.0);
        let dst = omni_rec
            .map(|o| o.dst)
            .filter(|v| !v.is_nan())
            .unwrap_or(0.0);

        // Feed rank fusion (XRS dual-channel)
        fusion.ingest(rec.xrsb, rec.xrsa, 100.0, 0.3, rec.timestamp);

        // Feed new detectors
        bz_det.ingest(bz, rec.xrsb, rec.timestamp);
        pressure_det.ingest_raw(speed, density, rec.xrsb, rec.timestamp);
        cluster_det.ingest(rec.xrsb, rec.timestamp);

        // Feed escalation
        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();
        esc.update_with_flux(h, f, a, rec.xrsb, rec.timestamp);

        // Track time at level
        minutes_at_level[esc.level as usize] += 1;

        // Track detector fires
        if bz_det.is_anomalous() {
            bz_fires += 1;
        }
        if pressure_det.is_anomalous() {
            pressure_fires += 1;
        }
        if cluster_det.is_anomalous() {
            cluster_fires += 1;
        }

        // Check flare onsets
        while next_flare_idx < flares.len() && rec.timestamp >= flares[next_flare_idx].begin {
            let flare = &flares[next_flare_idx];
            results.push(FlareResult {
                class: flare.class.clone(),
                esc_level: esc.level,
                fused_score: fusion.score(),
                bz_score: bz_det.score(),
                pressure_score: pressure_det.score(),
                cluster_score: cluster_det.score(),
                bz_at_onset: bz,
                speed_at_onset: speed,
                dst_at_onset: dst,
            });
            next_flare_idx += 1;
        }
    }
    println!("done.\n");

    let total = results.len();
    let total_minutes = goes.len() as u64;

    // Escalation results
    println!("=== Escalation Level at Flare Onset ===\n");
    let at_quiet = results
        .iter()
        .filter(|r| r.esc_level == EscalationLevel::Quiet)
        .count();
    let at_elevated = results
        .iter()
        .filter(|r| r.esc_level == EscalationLevel::Elevated)
        .count();
    let at_active = results
        .iter()
        .filter(|r| r.esc_level == EscalationLevel::Active)
        .count();
    let at_flare = results
        .iter()
        .filter(|r| r.esc_level == EscalationLevel::Flare)
        .count();
    let warned = at_elevated + at_active + at_flare;

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
            .filter(|r| r.esc_level >= EscalationLevel::Elevated)
            .count();
        println!(
            "  {}-class: {}/{} warned ({:.1}%)",
            prefix,
            w,
            n,
            w as f64 / n as f64 * 100.0
        );
    }

    // Time at each level
    println!("\n=== Time Distribution ===\n");
    let labels = ["QUIET", "ELEVATED", "ACTIVE", "FLARE"];
    for (i, label) in labels.iter().enumerate() {
        let pct = minutes_at_level[i] as f64 / total_minutes as f64 * 100.0;
        let hours = minutes_at_level[i] as f64 / 60.0;
        println!("  {:<10} {:>6.0}h ({:.1}%)", label, hours, pct);
    }

    // New detector statistics
    println!("\n=== New Detector Stats ===\n");
    println!(
        "  Bz Southward:      fired {:.0} min ({:.1}% of time)",
        bz_fires,
        bz_fires as f64 / total_minutes as f64 * 100.0
    );
    println!(
        "  Pressure Jump:     fired {:.0} min ({:.1}% of time)",
        pressure_fires,
        pressure_fires as f64 / total_minutes as f64 * 100.0
    );
    println!(
        "  Flare Clustering:  fired {:.0} min ({:.1}% of time)",
        cluster_fires,
        cluster_fires as f64 / total_minutes as f64 * 100.0
    );

    // New detector scores at flare onset
    println!("\n=== New Detector Scores at Flare Onset ===\n");
    let bz_mean: f64 = results.iter().map(|r| r.bz_score).sum::<f64>() / total as f64;
    let pr_mean: f64 = results.iter().map(|r| r.pressure_score).sum::<f64>() / total as f64;
    let cl_mean: f64 = results.iter().map(|r| r.cluster_score).sum::<f64>() / total as f64;
    let bz_max = results.iter().map(|r| r.bz_score).fold(0.0f64, f64::max);
    let pr_max = results
        .iter()
        .map(|r| r.pressure_score)
        .fold(0.0f64, f64::max);
    let cl_max = results
        .iter()
        .map(|r| r.cluster_score)
        .fold(0.0f64, f64::max);
    let bz_above = results.iter().filter(|r| r.bz_score > 0.3).count();
    let pr_above = results.iter().filter(|r| r.pressure_score > 0.3).count();
    let cl_above = results.iter().filter(|r| r.cluster_score > 0.3).count();

    println!(
        "  {:<20} {:>8} {:>8} {:>8}",
        "Detector", "Mean", "Max", ">0.3"
    );
    println!("  {}", "-".repeat(50));
    println!(
        "  {:<20} {:>8.3} {:>8.3} {:>6}/{}",
        "Bz Southward", bz_mean, bz_max, bz_above, total
    );
    println!(
        "  {:<20} {:>8.3} {:>8.3} {:>6}/{}",
        "Pressure Jump", pr_mean, pr_max, pr_above, total
    );
    println!(
        "  {:<20} {:>8.3} {:>8.3} {:>6}/{}",
        "Flare Clustering", cl_mean, cl_max, cl_above, total
    );

    // Solar wind conditions at flare onset
    println!("\n=== Solar Wind Conditions at Flare Onset ===\n");
    let bz_values: Vec<f64> = results.iter().map(|r| r.bz_at_onset).collect();
    let speed_values: Vec<f64> = results.iter().map(|r| r.speed_at_onset).collect();
    let dst_values: Vec<f64> = results.iter().map(|r| r.dst_at_onset).collect();
    println!(
        "  Bz:    mean={:.1} nT, min={:.1}, max={:.1}",
        bz_values.iter().sum::<f64>() / total as f64,
        bz_values.iter().cloned().fold(f64::INFINITY, f64::min),
        bz_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "  Speed: mean={:.0} km/s, min={:.0}, max={:.0}",
        speed_values.iter().sum::<f64>() / total as f64,
        speed_values.iter().cloned().fold(f64::INFINITY, f64::min),
        speed_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "  Dst:   mean={:.0} nT, min={:.0}, max={:.0}",
        dst_values.iter().sum::<f64>() / total as f64,
        dst_values.iter().cloned().fold(f64::INFINITY, f64::min),
        dst_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Flares during disturbed geomagnetic conditions
    let bz_south_flares = results.iter().filter(|r| r.bz_at_onset < -5.0).count();
    let high_speed_flares = results.iter().filter(|r| r.speed_at_onset > 500.0).count();
    let storm_flares = results.iter().filter(|r| r.dst_at_onset < -50.0).count();
    println!(
        "\n  Flares during Bz<-5:    {}/{} ({:.1}%)",
        bz_south_flares,
        total,
        bz_south_flares as f64 / total as f64 * 100.0
    );
    println!(
        "  Flares during V>500:    {}/{} ({:.1}%)",
        high_speed_flares,
        total,
        high_speed_flares as f64 / total as f64 * 100.0
    );
    println!(
        "  Flares during Dst<-50:  {}/{} ({:.1}%)",
        storm_flares,
        total,
        storm_flares as f64 / total as f64 * 100.0
    );
}

fn find_nearest_omni<'a>(
    entries: &[(&DateTime<Utc>, &'a loaders::OmniRecord)],
    ts: DateTime<Utc>,
) -> Option<&'a loaders::OmniRecord> {
    match entries.binary_search_by_key(&ts, |(t, _)| **t) {
        Ok(idx) => Some(entries[idx].1),
        Err(idx) if idx > 0 => {
            if (ts - *entries[idx - 1].0).num_seconds() <= 7200 {
                Some(entries[idx - 1].1)
            } else {
                None
            }
        }
        _ => None,
    }
}
