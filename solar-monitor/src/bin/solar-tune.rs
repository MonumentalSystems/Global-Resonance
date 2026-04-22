//! Tune detector parameters against real 7-day GOES data.
//!
//! Loads X-ray, electron, and solar wind 7-day JSON files,
//! merges at 1-min cadence, and sweeps detector parameters.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::collections::BTreeMap;

use solar_monitor::detection::cusum::CusumDetector;
use solar_monitor::detection::energy::EnergyDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use solar_monitor::detection::rate_of_change::RateOfChangeDetector;

fn main() {
    println!("=== Detector Tuning Against Real 7-Day GOES Data ===\n");

    // Load all channels
    let xray = load_xray("/tmp/xrays-7-day.json");
    let electrons = load_electrons("/tmp/electrons-7day.json");
    let (mag, plasma) = load_solar_wind("/tmp/mag-7day.json", "/tmp/plasma-7day.json");

    println!("X-ray samples: {}", xray.len());
    println!("Electron samples: {}", electrons.len());
    println!("Solar wind mag: {}, plasma: {}", mag.len(), plasma.len());

    // Merge into unified timeline at 1-min cadence
    let samples = merge_real_data(&xray, &electrons, &mag, &plasma);
    println!("Merged samples: {}\n", samples.len());

    // Identify known events in this window for scoring
    // The X1.5 on March 30 03:08-03:35 is the main target
    let events: Vec<KnownEvent> = vec![
        KnownEvent {
            name: "X1.5 flare",
            begin: parse_ts("2026-03-30T02:50:00Z"),
            peak: parse_ts("2026-03-30T03:19:00Z"),
            end: parse_ts("2026-03-30T03:50:00Z"),
            class: 'X',
        },
        KnownEvent {
            name: "M1.1 flare",
            begin: parse_ts("2026-03-28T03:30:00Z"),
            peak: parse_ts("2026-03-28T04:15:00Z"),
            end: parse_ts("2026-03-28T06:00:00Z"),
            class: 'M',
        },
    ];

    println!("Known events: {}", events.len());
    for e in &events {
        println!("  {} at {}", e.name, e.peak.format("%m-%d %H:%M"));
    }
    println!();

    // =========================================
    // Part 1: Parameter sweeps
    // =========================================
    println!("========================================");
    println!("  Part 2: Rate-of-Change Sweep");
    println!("========================================\n");

    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>6}",
        "Config", "X-Det", "M-Det", "FP", "Score@X"
    );
    println!("{}", "-".repeat(68));

    for smooth in &[3usize, 5, 8, 12] {
        for thresh in &[0.05f64, 0.1, 0.2, 0.3, 0.5] {
            let label = format!("smooth={} thresh={:.2}", smooth, thresh);
            let (x_det, m_det, fp, x_score) = sweep_roc(&samples, &events, *smooth, *thresh);
            println!(
                "{:<30} {:>8} {:>8} {:>8} {:>6.3}",
                label, x_det, m_det, fp, x_score
            );
        }
    }

    println!("\n========================================");
    println!("  Part 3: Energy Detector Sweep");
    println!("========================================\n");

    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>6}",
        "Config", "X-Det", "M-Det", "FP", "Score@X"
    );
    println!("{}", "-".repeat(68));

    for window in &[30usize, 60, 120, 240] {
        let label = format!("window={}", window);
        let (x_det, m_det, fp, x_score) = sweep_energy(&samples, &events, *window);
        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>6.3}",
            label, x_det, m_det, fp, x_score
        );
    }

    println!("\n========================================");
    println!("  Part 4: CUSUM Sensitivity Sweep");
    println!("========================================\n");

    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>6}",
        "Config", "X-Det", "M-Det", "FP", "Score@X"
    );
    println!("{}", "-".repeat(68));

    for baseline in &[60usize, 120, 240, 480] {
        for sens in &[4.0f64, 8.0, 12.0, 16.0, 24.0, 32.0] {
            let label = format!("base={} sens={:.0}", baseline, sens);
            let (x_det, m_det, fp, x_score) = sweep_cusum(&samples, &events, *baseline, *sens);
            println!(
                "{:<30} {:>8} {:>8} {:>8} {:>6.3}",
                label, x_det, m_det, fp, x_score
            );
        }
    }

    // =========================================
    // Part 5: Fusion with tuned parameters
    // =========================================
    println!("\n========================================");
    println!("  Part 5: Rank Fusion Sweep");
    println!("========================================\n");

    println!(
        "{:<30} {:>8} {:>8} {:>8} {:>6}",
        "Threshold", "X-Det", "M-Det", "FP", "Score@X"
    );
    println!("{}", "-".repeat(68));

    for thresh in &[0.3f64, 0.4, 0.5, 0.6, 0.7, 0.8] {
        let (x_det, m_det, fp, x_score) = sweep_fusion(&samples, &events, *thresh);
        println!(
            "threshold={:.1}{:>19} {:>8} {:>8} {:>8} {:>6.3}",
            thresh, "", x_det, m_det, fp, x_score
        );
    }
}

// ----- Data structures -----

struct RealSample {
    timestamp: DateTime<Utc>,
    xray_flux: f64,
    electron_flux: f64,
}

struct KnownEvent {
    name: &'static str,
    begin: DateTime<Utc>,
    peak: DateTime<Utc>,
    end: DateTime<Utc>,
    class: char,
}

// ----- Loaders -----

fn load_xray(path: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct Rec {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }

    let data: Vec<Rec> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for r in data {
        if r.energy.as_deref() != Some("0.1-0.8nm") {
            continue;
        }
        if let (Some(ts), Some(flux)) = (parse_ts_opt(&r.time_tag), r.flux) {
            if flux > 0.0 {
                // Filter eclipse zeros
                map.insert(ts, flux);
            }
        }
    }
    map
}

fn load_electrons(path: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct Rec {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }

    let data: Vec<Rec> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for r in data {
        if r.energy.as_deref() != Some(">=2 MeV") {
            continue;
        }
        if let (Some(ts), Some(flux)) = (parse_ts_opt(&r.time_tag), r.flux) {
            map.insert(ts, flux);
        }
    }
    map
}

fn load_solar_wind(
    mag_path: &str,
    plasma_path: &str,
) -> (
    BTreeMap<DateTime<Utc>, [f64; 3]>,
    BTreeMap<DateTime<Utc>, [f64; 2]>,
) {
    let mag_raw: Vec<Vec<serde_json::Value>> =
        serde_json::from_str(&std::fs::read_to_string(mag_path).unwrap()).unwrap();
    let plasma_raw: Vec<Vec<serde_json::Value>> =
        serde_json::from_str(&std::fs::read_to_string(plasma_path).unwrap()).unwrap();

    let mut mag = BTreeMap::new();
    for row in mag_raw.iter().skip(1) {
        if row.len() < 4 {
            continue;
        }
        if let Some(ts) = row[0].as_str().and_then(|s| parse_ts_opt(s)) {
            let bx = parse_f64(&row[1]).unwrap_or(0.0);
            let by = parse_f64(&row[2]).unwrap_or(0.0);
            let bz = parse_f64(&row[3]).unwrap_or(0.0);
            mag.insert(ts, [bx, by, bz]);
        }
    }

    let mut plasma = BTreeMap::new();
    for row in plasma_raw.iter().skip(1) {
        if row.len() < 3 {
            continue;
        }
        if let Some(ts) = row[0].as_str().and_then(|s| parse_ts_opt(s)) {
            let density = parse_f64(&row[1]).unwrap_or(0.0);
            let speed = parse_f64(&row[2]).unwrap_or(0.0);
            plasma.insert(ts, [density, speed]);
        }
    }

    (mag, plasma)
}

fn merge_real_data(
    xray: &BTreeMap<DateTime<Utc>, f64>,
    electrons: &BTreeMap<DateTime<Utc>, f64>,
    _mag: &BTreeMap<DateTime<Utc>, [f64; 3]>,
    _plasma: &BTreeMap<DateTime<Utc>, [f64; 2]>,
) -> Vec<RealSample> {
    // Use X-ray timestamps as primary (1-min cadence)
    // Interpolate electrons (5-min cadence) with hold-previous
    let elec_vec: Vec<(DateTime<Utc>, f64)> = electrons.iter().map(|(&t, &v)| (t, v)).collect();

    xray.iter()
        .map(|(&ts, &flux)| {
            let electron = find_nearest_val(&elec_vec, ts).unwrap_or(100.0);
            RealSample {
                timestamp: ts,
                xray_flux: flux,
                electron_flux: electron,
            }
        })
        .collect()
}

fn find_nearest_val(entries: &[(DateTime<Utc>, f64)], ts: DateTime<Utc>) -> Option<f64> {
    match entries.binary_search_by_key(&ts, |(t, _)| *t) {
        Ok(idx) => Some(entries[idx].1),
        Err(idx) if idx > 0 => {
            if (ts - entries[idx - 1].0).num_seconds() < 600 {
                Some(entries[idx - 1].1)
            } else {
                None
            }
        }
        _ => None,
    }
}

// ----- Sweep functions -----

fn sweep_roc(
    samples: &[RealSample],
    events: &[KnownEvent],
    smooth: usize,
    thresh: f64,
) -> (bool, bool, usize, f64) {
    let mut d = RateOfChangeDetector::new(smooth, thresh);
    let mut x_score = 0.0f64;
    let mut x_det = false;
    let mut m_det = false;
    let mut fp = 0usize;
    let mut last_det: Option<DateTime<Utc>> = None;

    for s in samples {
        if s.xray_flux > 1e-9 {
            // skip eclipse
            d.ingest(s.xray_flux, s.timestamp);
        }

        // Track max score during X event
        for e in events {
            if s.timestamp >= e.begin && s.timestamp <= e.end && e.class == 'X' {
                if d.score() > x_score {
                    x_score = d.score();
                }
            }
        }

        if d.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (s.timestamp - l).num_seconds() < 1800);
            if !dominated {
                let mut matched = false;
                for e in events {
                    if s.timestamp >= e.begin && s.timestamp <= e.end {
                        matched = true;
                        if e.class == 'X' {
                            x_det = true;
                        }
                        if e.class == 'M' {
                            m_det = true;
                        }
                    }
                }
                if !matched {
                    fp += 1;
                }
                last_det = Some(s.timestamp);
            }
        }
    }
    (x_det, m_det, fp, x_score)
}

fn sweep_energy(
    samples: &[RealSample],
    events: &[KnownEvent],
    window: usize,
) -> (bool, bool, usize, f64) {
    let mut d = EnergyDetector::new(window);
    let mut x_score = 0.0f64;
    let mut x_det = false;
    let mut m_det = false;
    let mut fp = 0usize;
    let mut last_det: Option<DateTime<Utc>> = None;

    for s in samples {
        d.ingest(s.xray_flux, s.electron_flux, s.timestamp);

        for e in events {
            if s.timestamp >= e.begin && s.timestamp <= e.end && e.class == 'X' {
                if d.score() > x_score {
                    x_score = d.score();
                }
            }
        }

        if d.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (s.timestamp - l).num_seconds() < 1800);
            if !dominated {
                let mut matched = false;
                for e in events {
                    if s.timestamp >= e.begin && s.timestamp <= e.end {
                        matched = true;
                        if e.class == 'X' {
                            x_det = true;
                        }
                        if e.class == 'M' {
                            m_det = true;
                        }
                    }
                }
                if !matched {
                    fp += 1;
                }
                last_det = Some(s.timestamp);
            }
        }
    }
    (x_det, m_det, fp, x_score)
}

fn sweep_cusum(
    samples: &[RealSample],
    events: &[KnownEvent],
    baseline: usize,
    sens: f64,
) -> (bool, bool, usize, f64) {
    let mut d = CusumDetector::new(baseline, sens);
    let mut x_score = 0.0f64;
    let mut x_det = false;
    let mut m_det = false;
    let mut fp = 0usize;
    let mut last_det: Option<DateTime<Utc>> = None;

    for s in samples {
        if s.xray_flux > 1e-9 {
            d.ingest(s.xray_flux, s.timestamp);
        }

        for e in events {
            if s.timestamp >= e.begin && s.timestamp <= e.end && e.class == 'X' {
                if d.score() > x_score {
                    x_score = d.score();
                }
            }
        }

        if d.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (s.timestamp - l).num_seconds() < 1800);
            if !dominated {
                let mut matched = false;
                for e in events {
                    if s.timestamp >= e.begin && s.timestamp <= e.end {
                        matched = true;
                        if e.class == 'X' {
                            x_det = true;
                        }
                        if e.class == 'M' {
                            m_det = true;
                        }
                    }
                }
                if !matched {
                    fp += 1;
                }
                last_det = Some(s.timestamp);
            }
        }
    }
    (x_det, m_det, fp, x_score)
}

fn sweep_fusion(
    samples: &[RealSample],
    events: &[KnownEvent],
    threshold: f64,
) -> (bool, bool, usize, f64) {
    let mut d = RankFusionDetector::new(threshold);
    let mut x_score = 0.0f64;
    let mut x_det = false;
    let mut m_det = false;
    let mut fp = 0usize;
    let mut last_det: Option<DateTime<Utc>> = None;

    for s in samples {
        if s.xray_flux > 1e-9 {
            d.ingest_simple(s.xray_flux, s.electron_flux, s.timestamp);
        }

        for e in events {
            if s.timestamp >= e.begin && s.timestamp <= e.end && e.class == 'X' {
                if d.score() > x_score {
                    x_score = d.score();
                }
            }
        }

        if d.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (s.timestamp - l).num_seconds() < 1800);
            if !dominated {
                let mut matched = false;
                for e in events {
                    if s.timestamp >= e.begin && s.timestamp <= e.end {
                        matched = true;
                        if e.class == 'X' {
                            x_det = true;
                        }
                        if e.class == 'M' {
                            m_det = true;
                        }
                    }
                }
                if !matched {
                    fp += 1;
                }
                last_det = Some(s.timestamp);
            }
        }
    }
    (x_det, m_det, fp, x_score)
}

// ----- Utils -----

fn parse_ts(s: &str) -> DateTime<Utc> {
    parse_ts_opt(s).expect(&format!("Bad timestamp: {}", s))
}

fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

fn parse_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}
