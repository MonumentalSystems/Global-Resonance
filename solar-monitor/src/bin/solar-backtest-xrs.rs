//! Backtest the criticality detector against real GOES XRS 1-minute data
//! downloaded by solar-fetch-xrs.
//!
//! For each event file, runs the full rank fusion (7 detectors) and reports:
//! - Per-detector scores in the pre-flare window
//! - First detection time for each detector
//! - Criticality detector advantage (earlier signal vs others)

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use solar_monitor::detection::rank_fusion::{RankFusionDetector, DETECTOR_NAMES, N_DETECTORS};
use std::collections::BTreeMap;

/// Events to backtest (matching solar-fetch-xrs).
const EVENTS: &[(&str, &str, &str)] = &[
    (
        "/tmp/goes-xrs-x1.4_2011sep22.json",
        "X1.4",
        "2011-09-22 10:29:00",
    ),
    (
        "/tmp/goes-xrs-x1.5_2011mar09.json",
        "X1.5",
        "2011-03-09 23:13:00",
    ),
    (
        "/tmp/goes-xrs-x1.1_2013nov10.json",
        "X1.1",
        "2013-11-10 05:08:00",
    ),
    (
        "/tmp/goes-xrs-x1.6_2013may13.json",
        "X1.6",
        "2013-05-13 01:53:00",
    ),
    (
        "/tmp/goes-xrs-x1.9_2011sep24.json",
        "X1.9",
        "2011-09-24 09:21:00",
    ),
    (
        "/tmp/goes-xrs-x1.0_2014mar29.json",
        "X1.0",
        "2014-03-29 17:35:00",
    ),
    (
        "/tmp/goes-xrs-x2.2_2011feb15.json",
        "X2.2",
        "2011-02-15 01:44:00",
    ),
    (
        "/tmp/goes-xrs-x5.4_2012mar07.json",
        "X5.4",
        "2012-03-07 00:02:00",
    ),
    (
        "/tmp/goes-xrs-x9.3_2017sep06.json",
        "X9.3",
        "2017-09-06 11:53:00",
    ),
    (
        "/tmp/goes-xrs-x8.2_2017sep10.json",
        "X8.2",
        "2017-09-10 15:35:00",
    ),
];

fn main() {
    println!("=== Criticality Detector Backtest: Real GOES XRS 1-min Data ===\n");
    println!(
        "{:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Event",
        "PreMean",
        "PreMax",
        "BLMean",
        "Sep",
        "1st_Crit",
        "1st_ZSco",
        "1st_Hard",
        "1st_CSUM",
        "1st_Fuse"
    );
    println!("{}", "-".repeat(100));

    let mut total_crit_lead = 0i64;
    let mut total_best_other_lead = 0i64;
    let mut count = 0;

    for &(path, class, onset_str) in EVENTS {
        let onset = parse_ts(onset_str);

        // Load both channels.
        let (xray_long, xray_short) = load_xrs_channels(path);
        if xray_long.is_empty() {
            eprintln!("  SKIP {} (no data)", class);
            continue;
        }

        let mut fusion = RankFusionDetector::new(0.5);

        // Track per-detector scores and first detection times.
        let mut first_above: [Option<i64>; N_DETECTORS] = [None; N_DETECTORS];
        let mut pre_scores: Vec<f64> = Vec::new(); // criticality scores in pre-window
        let mut bl_scores: Vec<f64> = Vec::new(); // baseline scores

        // Pre-flare window: -6h to -10min.
        let pre_start = onset - chrono::Duration::hours(6);
        let pre_end = onset - chrono::Duration::minutes(10);
        // Baseline: -8d to -1d.
        let bl_start = onset - chrono::Duration::days(8);
        let bl_end = onset - chrono::Duration::days(1);

        for (&ts, &long_flux) in &xray_long {
            if long_flux < 1e-9 {
                continue;
            }
            let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
            // No proton data in XRS files — use background.
            fusion.ingest(long_flux, short_flux, 100.0, 0.3, ts);

            let diag = fusion.diagnostics();

            // Track criticality score in windows.
            if ts >= pre_start && ts <= pre_end {
                pre_scores.push(diag.raw_scores.last().map(|d| d.raw_score).unwrap_or(0.0));
            }
            if ts >= bl_start && ts <= bl_end {
                bl_scores.push(diag.raw_scores.last().map(|d| d.raw_score).unwrap_or(0.0));
            }

            // First detection (score > 0.3 before onset).
            let dt = (onset - ts).num_minutes();
            if dt > 0 && dt < 8 * 24 * 60 {
                for (di, ds) in diag.raw_scores.iter().enumerate() {
                    if ds.raw_score > 0.3 {
                        if first_above[di].is_none() || dt > first_above[di].unwrap() {
                            first_above[di] = Some(dt);
                        }
                    }
                }
            }
        }

        let pre_mean = mean(&pre_scores);
        let pre_max = pre_scores.iter().cloned().fold(0.0_f64, f64::max);
        let bl_mean = mean(&bl_scores);
        let sep = pre_mean - bl_mean;

        // Format first detection times.
        let fmt_lead = |idx: usize| -> String {
            first_above[idx]
                .map(|m| format!("-{:.1}h", m as f64 / 60.0))
                .unwrap_or("never".into())
        };

        // criticality is index 6, zscore=0, cusum=1, hardness=2, fused=overall
        let crit_lead = first_above[6].unwrap_or(0);
        let best_other = first_above[..6]
            .iter()
            .filter_map(|&x| x)
            .max()
            .unwrap_or(0);

        println!(
            "{:<8} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8} {:>8} {:>8} {:>8} {:>8}",
            class,
            pre_mean,
            pre_max,
            bl_mean,
            sep,
            fmt_lead(6),
            fmt_lead(0),
            fmt_lead(2),
            fmt_lead(1),
            format!("-{:.1}h", fusion.score()), // fused score at end
        );

        total_crit_lead += crit_lead;
        total_best_other_lead += best_other;
        count += 1;
    }

    println!("\n--- Summary ---");
    if count > 0 {
        println!(
            "Average criticality first detection: -{:.1}h",
            total_crit_lead as f64 / count as f64 / 60.0
        );
        println!(
            "Average best-other first detection:  -{:.1}h",
            total_best_other_lead as f64 / count as f64 / 60.0
        );
        println!(
            "Criticality advantage:               {:.1}h earlier on average",
            (total_crit_lead as f64 - total_best_other_lead as f64) / count as f64 / 60.0
        );
    }
}

fn load_xrs_channels(path: &str) -> (BTreeMap<DateTime<Utc>, f64>, BTreeMap<DateTime<Utc>, f64>) {
    #[derive(Deserialize)]
    struct R {
        time_tag: String,
        energy: String,
        flux: f64,
    }
    let data: Vec<R> = match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => return (BTreeMap::new(), BTreeMap::new()),
    };

    let mut long = BTreeMap::new();
    let mut short = BTreeMap::new();

    for r in &data {
        if let Some(ts) = parse_ts_opt(&r.time_tag) {
            if r.flux > 0.0 {
                if r.energy == "0.1-0.8nm" {
                    long.insert(ts, r.flux);
                } else if r.energy == "0.05-0.4nm" {
                    short.insert(ts, r.flux);
                }
            }
        }
    }

    (long, short)
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

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    parse_ts_opt(s).unwrap()
}

fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%dT%H:%M:%SZ"))
        .ok()
        .map(|ndt| ndt.and_utc())
}
