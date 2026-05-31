use chrono::Timelike;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use solar_monitor::detection::rank_fusion::{RankFusionDetector, DETECTOR_NAMES, N_DETECTORS};
use std::collections::BTreeMap;

fn main() {
    let xray_long = load_channel("/tmp/xrays-7-day.json", "0.1-0.8nm");
    let xray_short = load_channel("/tmp/xrays-7-day.json", "0.05-0.4nm");
    let electrons = load_channel_generic("/tmp/electrons-7day.json", ">=2 MeV");
    let protons = load_channel_generic("/tmp/protons-7day.json", ">=1 MeV");
    let electrons_smooth = ema_smooth(&electrons, 0.15);
    let protons_smooth = ema_smooth(&protons, 0.15);

    let m_begin = parse_ts("2026-03-28T02:23:00Z"); // actual GOES M-class start
    let x_begin = parse_ts("2026-03-30T01:45:00Z"); // actual GOES X-class start

    let mut fusion = RankFusionDetector::new(0.5);

    // Track per-detector scores in precursor windows
    println!("=== Pre-M1.1 Window (3h before onset at 02:23) ===\n");
    println!(
        "{:<14} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5}",
        "Time",
        "Xray",
        "Fused",
        "ZScore",
        "CUSUM",
        "Hardns",
        "RoC",
        "Multi",
        "Proton",
        "Crit",
        "Agree"
    );
    println!("{}", "-".repeat(112));

    let m_window_start = parse_ts("2026-03-27T23:23:00Z");
    let m_window_end = parse_ts("2026-03-28T03:00:00Z");

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);
        fusion.ingest(long_flux, short_flux, electron, proton, ts);

        if ts >= m_window_start && ts <= m_window_end && ts.minute() % 5 == 0 {
            let diag = fusion.diagnostics();
            let s: Vec<f64> = diag.raw_scores.iter().map(|d| d.raw_score).collect();
            let dt = (m_begin - ts).num_minutes();
            let marker = if dt > 0 {
                format!("-{}m", dt)
            } else {
                format!("+{}m", -dt)
            };
            println!(
                "{} {:>8.2e} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>3}/7  {}",
                ts.format("%m-%d %H:%M"),
                long_flux,
                diag.fused_score,
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                s[5],
                s[6],
                diag.detector_agreement,
                marker
            );
        }
    }

    // Reset and do X1.5
    let mut fusion2 = RankFusionDetector::new(0.5);
    println!("\n=== Pre-X1.5 Window (6h before onset at 01:45) ===\n");
    println!(
        "{:<14} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>5}",
        "Time",
        "Xray",
        "Fused",
        "ZScore",
        "CUSUM",
        "Hardns",
        "RoC",
        "Multi",
        "Proton",
        "Crit",
        "Agree"
    );
    println!("{}", "-".repeat(112));

    let x_window_start = parse_ts("2026-03-29T19:45:00Z");
    let x_window_end = parse_ts("2026-03-30T03:30:00Z");

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);
        fusion2.ingest(long_flux, short_flux, electron, proton, ts);

        if ts >= x_window_start && ts <= x_window_end && ts.minute() % 10 == 0 {
            let diag = fusion2.diagnostics();
            let s: Vec<f64> = diag.raw_scores.iter().map(|d| d.raw_score).collect();
            let dt = (x_begin - ts).num_minutes();
            let marker = if dt > 0 {
                format!("-{}m", dt)
            } else {
                format!("+{}m", -dt)
            };
            println!(
                "{} {:>8.2e} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>3}/7  {}",
                ts.format("%m-%d %H:%M"),
                long_flux,
                diag.fused_score,
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                s[5],
                s[6],
                diag.detector_agreement,
                marker
            );
        }
    }

    // Summary: which detectors had score > 0.3 earliest before each event?
    println!("\n=== First Detection (score > 0.3) Before Each Event ===\n");

    let mut fusion3 = RankFusionDetector::new(0.5);
    let mut first_above: [[Option<i64>; N_DETECTORS]; 2] = [[None; N_DETECTORS]; 2]; // [M, X][detector]

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);
        fusion3.ingest(long_flux, short_flux, electron, proton, ts);

        let diag = fusion3.diagnostics();
        for (di, ds) in diag.raw_scores.iter().enumerate() {
            if ds.raw_score > 0.3 {
                // Check M-class precursor
                let dt_m = (m_begin - ts).num_minutes();
                if dt_m > 0 && dt_m < 1440 {
                    // within 24h before
                    if first_above[0][di].is_none() || dt_m > first_above[0][di].unwrap() {
                        first_above[0][di] = Some(dt_m);
                    }
                }
                // Check X-class precursor
                let dt_x = (x_begin - ts).num_minutes();
                if dt_x > 0 && dt_x < 2880 {
                    if first_above[1][di].is_none() || dt_x > first_above[1][di].unwrap() {
                        first_above[1][di] = Some(dt_x);
                    }
                }
            }
        }
    }

    println!(
        "{:<16} {:>20} {:>20}",
        "Detector", "Before M1.1", "Before X1.5"
    );
    println!("{}", "-".repeat(60));
    for di in 0..N_DETECTORS {
        let m_str = first_above[0][di]
            .map(|m| format!("-{:.1}h", m as f64 / 60.0))
            .unwrap_or("never".into());
        let x_str = first_above[1][di]
            .map(|m| format!("-{:.1}h", m as f64 / 60.0))
            .unwrap_or("never".into());
        println!("{:<16} {:>20} {:>20}", DETECTOR_NAMES[di], m_str, x_str);
    }
}

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
        .ok()
        .map(|ndt| ndt.and_utc())
}
