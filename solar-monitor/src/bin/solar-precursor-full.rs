//! Precursor analysis using ingest_full() with merged X-ray + mag data.
//! This exercises the full commutator-enhanced pipeline:
//! - Criticality detector: ||B ∧ Ḃ|| commutator + loading fraction
//! - Multichannel detector: 3-channel X-ray/electron/B-field decorrelation
//! - Escalation: predictive escalation from criticality score

use chrono::{DateTime, NaiveDateTime, Timelike, Utc};
use serde::Deserialize;
use solar_monitor::detection::rank_fusion::{RankFusionDetector, DETECTOR_NAMES, N_DETECTORS};
use std::collections::BTreeMap;

fn main() {
    let xray_long = load_channel("/tmp/xrays-7-day.json", "0.1-0.8nm");
    let xray_short = load_channel("/tmp/xrays-7-day.json", "0.05-0.4nm");
    let electrons = load_channel_generic("/tmp/electrons-7day.json", ">=2 MeV");
    let protons = load_channel_generic("/tmp/protons-7day.json", ">=1 MeV");
    let mag = load_mag("/tmp/mag-7day.json");
    let electrons_smooth = ema_smooth(&electrons, 0.15);
    let protons_smooth = ema_smooth(&protons, 0.15);

    println!(
        "Data: {} xray, {} mag samples\n",
        xray_long.len(),
        mag.len()
    );

    let x_begin = parse_ts("2026-03-30T01:45:00Z");

    // Run with ingest_full (B-field enabled).
    let mut fusion_full = RankFusionDetector::new(0.5);
    // Also run without B-field for comparison.
    let mut fusion_scalar = RankFusionDetector::new(0.5);

    println!("=== Pre-X1.5 Window: ingest_full (with B-field) vs ingest (scalar only) ===\n");
    println!(
        "{:<14} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}  {:>7} {:>7}",
        "Time", "Xray", "FULL", "SCALAR", "Crit_F", "Crit_S", "Multi_F", "Multi_S", "Lead"
    );
    println!("{}", "-".repeat(100));

    let x_window_start = parse_ts("2026-03-29T19:45:00Z");
    let x_window_end = parse_ts("2026-03-30T03:30:00Z");

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);

        // Scalar path (no B-field).
        fusion_scalar.ingest(long_flux, short_flux, electron, proton, ts);

        // Full path (with B-field).
        if let Some((bx, by, bz)) = find_nearest_mag(&mag, ts) {
            fusion_full.ingest_full(long_flux, short_flux, electron, proton, bx, by, bz, ts);
        } else {
            fusion_full.ingest(long_flux, short_flux, electron, proton, ts);
        }

        if ts >= x_window_start && ts <= x_window_end && ts.minute() % 10 == 0 {
            let diag_f = fusion_full.diagnostics();
            let diag_s = fusion_scalar.diagnostics();
            let sf: Vec<f64> = diag_f.raw_scores.iter().map(|d| d.raw_score).collect();
            let ss: Vec<f64> = diag_s.raw_scores.iter().map(|d| d.raw_score).collect();
            let dt = (x_begin - ts).num_minutes();
            let marker = if dt > 0 {
                format!("-{}m", dt)
            } else {
                format!("+{}m", -dt)
            };

            println!(
                "{} {:>8.2e} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}  {:>7.3} {:>5}",
                ts.format("%m-%d %H:%M"),
                long_flux,
                diag_f.fused_score,
                diag_s.fused_score,
                sf[6], // criticality full
                ss[6], // criticality scalar
                sf[4], // multichannel full
                ss[4], // multichannel scalar
                marker
            );
        }
    }

    // First detection comparison.
    println!("\n=== First Detection (score > 0.3) ===\n");
    let mut fusion_f2 = RankFusionDetector::new(0.5);
    let mut fusion_s2 = RankFusionDetector::new(0.5);
    let mut first_full: [Option<i64>; N_DETECTORS] = [None; N_DETECTORS];
    let mut first_scalar: [Option<i64>; N_DETECTORS] = [None; N_DETECTORS];

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);

        fusion_s2.ingest(long_flux, short_flux, electron, proton, ts);
        if let Some((bx, by, bz)) = find_nearest_mag(&mag, ts) {
            fusion_f2.ingest_full(long_flux, short_flux, electron, proton, bx, by, bz, ts);
        } else {
            fusion_f2.ingest(long_flux, short_flux, electron, proton, ts);
        }

        let dt_x = (x_begin - ts).num_minutes();
        if dt_x > 0 && dt_x < 2880 {
            let diag_f = fusion_f2.diagnostics();
            let diag_s = fusion_s2.diagnostics();
            for (di, ds) in diag_f.raw_scores.iter().enumerate() {
                if ds.raw_score > 0.3 {
                    if first_full[di].is_none() || dt_x > first_full[di].unwrap() {
                        first_full[di] = Some(dt_x);
                    }
                }
            }
            for (di, ds) in diag_s.raw_scores.iter().enumerate() {
                if ds.raw_score > 0.3 {
                    if first_scalar[di].is_none() || dt_x > first_scalar[di].unwrap() {
                        first_scalar[di] = Some(dt_x);
                    }
                }
            }
        }
    }

    println!(
        "{:<16} {:>16} {:>16}",
        "Detector", "Full (B-field)", "Scalar only"
    );
    println!("{}", "-".repeat(52));
    for di in 0..N_DETECTORS {
        let f_str = first_full[di]
            .map(|m| format!("-{:.1}h", m as f64 / 60.0))
            .unwrap_or("never".into());
        let s_str = first_scalar[di]
            .map(|m| format!("-{:.1}h", m as f64 / 60.0))
            .unwrap_or("never".into());
        println!("{:<16} {:>16} {:>16}", DETECTOR_NAMES[di], f_str, s_str);
    }
}

struct MagEntry {
    bx: f64,
    by: f64,
    bz: f64,
}

fn load_mag(path: &str) -> BTreeMap<DateTime<Utc>, MagEntry> {
    let data: Vec<Vec<String>> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for row in data.iter().skip(1) {
        // skip header
        if row.len() < 4 {
            continue;
        }
        let ts = match parse_ts_opt(&row[0]) {
            Some(t) => t,
            None => continue,
        };
        let bx: f64 = row[1].parse().unwrap_or(0.0);
        let by: f64 = row[2].parse().unwrap_or(0.0);
        let bz: f64 = row[3].parse().unwrap_or(0.0);
        if bx.abs() < 500.0 && by.abs() < 500.0 && bz.abs() < 500.0 {
            map.insert(ts, MagEntry { bx, by, bz });
        }
    }
    map
}

fn find_nearest_mag(
    map: &BTreeMap<DateTime<Utc>, MagEntry>,
    ts: DateTime<Utc>,
) -> Option<(f64, f64, f64)> {
    map.range(..=ts).next_back().and_then(|(&t, m)| {
        if (ts - t).num_seconds() < 120 {
            Some((m.bx, m.by, m.bz))
        } else {
            None
        }
    })
}

// --- Shared data loading (same as solar-precursor) ---
fn load_channel(path: &str, energy: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct R {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<R> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    data.iter()
        .filter(|r| r.energy.as_deref() == Some(energy))
        .filter_map(|r| Some((parse_ts_opt(&r.time_tag)?, r.flux.filter(|&f| f > 0.0)?)))
        .collect()
}
fn load_channel_generic(path: &str, energy: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct R {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<R> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    data.iter()
        .filter(|r| r.energy.as_deref() == Some(energy))
        .filter_map(|r| Some((parse_ts_opt(&r.time_tag)?, r.flux?)))
        .collect()
}
fn ema_smooth(data: &BTreeMap<DateTime<Utc>, f64>, alpha: f64) -> Vec<(DateTime<Utc>, f64)> {
    let mut result = Vec::new();
    let mut ema = None;
    for (&ts, &val) in data {
        let lv = if val > 0.0 { val.log10() } else { -2.0 };
        let s = match ema {
            None => {
                ema = Some(lv);
                lv
            }
            Some(p) => {
                let s = alpha * lv + (1.0 - alpha) * p;
                ema = Some(s);
                s
            }
        };
        result.push((ts, 10.0_f64.powf(s)));
    }
    result
}
fn find_nearest_val(map: &BTreeMap<DateTime<Utc>, f64>, ts: DateTime<Utc>) -> Option<f64> {
    map.range(..=ts).next_back().and_then(|(&t, &v)| {
        if (ts - t).num_seconds() < 120 {
            Some(v)
        } else {
            None
        }
    })
}
fn find_nearest_smooth(entries: &[(DateTime<Utc>, f64)], ts: DateTime<Utc>) -> Option<f64> {
    match entries.binary_search_by_key(&ts, |(t, _)| *t) {
        Ok(i) => Some(entries[i].1),
        Err(i) if i > 0 => {
            if (ts - entries[i - 1].0).num_seconds() < 600 {
                Some(entries[i - 1].1)
            } else {
                None
            }
        }
        _ => None,
    }
}
fn parse_ts(s: &str) -> DateTime<Utc> {
    parse_ts_opt(s).unwrap()
}
fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
        .map(|ndt| ndt.and_utc())
}
