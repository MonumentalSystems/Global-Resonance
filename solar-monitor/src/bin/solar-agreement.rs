//! Analyze detector agreement patterns: which detectors fire on TPs vs FPs?

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::collections::BTreeMap;

use solar_monitor::detection::rank_fusion::{RankFusionDetector, DETECTOR_NAMES};
use solar_monitor::feeds::xray::FlareClass;

fn main() {
    println!("=== Detector Agreement Analysis ===\n");

    // Load real 7-day data
    let xray = load_xray("/tmp/xrays-7-day.json");
    let electrons = load_electrons("/tmp/electrons-7day.json");
    println!(
        "X-ray: {} samples, Electrons: {} samples\n",
        xray.len(),
        electrons.len()
    );

    let elec_vec: Vec<(DateTime<Utc>, f64)> = electrons.iter().map(|(&t, &v)| (t, v)).collect();

    // Known events
    let x_begin = parse_ts("2026-03-30T02:50:00Z");
    let x_end = parse_ts("2026-03-30T05:50:00Z");
    let m_begin = parse_ts("2026-03-28T03:30:00Z");
    let m_end = parse_ts("2026-03-28T05:00:00Z");

    let mut fusion = RankFusionDetector::new(0.5);
    let mut last_det: Option<DateTime<Utc>> = None;

    // Collect all detections with their per-detector state
    struct Detection {
        ts: DateTime<Utc>,
        flux: f64,
        fused_score: f64,
        per_detector_anomalous: [bool; 5],
        per_detector_scores: [f64; 5],
        per_detector_ranks: [f64; 5],
        agreement: usize,
        is_tp: bool,
        event_name: String,
    }

    let mut detections: Vec<Detection> = Vec::new();

    for (&ts, &flux) in &xray {
        if flux < 1e-9 {
            continue;
        }
        let electron = find_nearest(&elec_vec, ts).unwrap_or(100.0);
        fusion.ingest_simple(flux, electron, ts);

        if fusion.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (ts - l).num_seconds() < 1800);
            if !dominated {
                let diag = fusion.diagnostics();

                let in_x = ts >= x_begin && ts <= x_end;
                let in_m = ts >= m_begin && ts <= m_end;
                let is_tp = in_x || in_m;
                let event_name = if in_x {
                    "X1.5".to_string()
                } else if in_m {
                    "M1.1".to_string()
                } else {
                    format!("{}", FlareClass::from_flux(flux).label())
                };

                let mut per_anomalous = [false; 5];
                let mut per_scores = [0.0f64; 5];
                let mut per_ranks = [0.0f64; 5];
                for (i, d) in diag.raw_scores.iter().enumerate() {
                    per_anomalous[i] = d.is_anomalous;
                    per_scores[i] = d.raw_score;
                    per_ranks[i] = d.percentile_rank;
                }

                detections.push(Detection {
                    ts,
                    flux,
                    fused_score: diag.fused_score,
                    per_detector_anomalous: per_anomalous,
                    per_detector_scores: per_scores,
                    per_detector_ranks: per_ranks,
                    agreement: diag.detector_agreement,
                    is_tp,
                    event_name,
                });

                last_det = Some(ts);
            }
        }
    }

    let tps: Vec<&Detection> = detections.iter().filter(|d| d.is_tp).collect();
    let fps: Vec<&Detection> = detections.iter().filter(|d| !d.is_tp).collect();

    println!(
        "Total detections: {} (TP: {}, FP: {})\n",
        detections.len(),
        tps.len(),
        fps.len()
    );

    // =========================================
    // Agreement distribution
    // =========================================
    println!("=== Agreement Distribution ===\n");
    println!("{:<12} {:>6} {:>6}", "Agreement", "TP", "FP");
    println!("{}", "-".repeat(30));
    for agree in 0..=5 {
        let tp_count = tps.iter().filter(|d| d.agreement == agree).count();
        let fp_count = fps.iter().filter(|d| d.agreement == agree).count();
        if tp_count > 0 || fp_count > 0 {
            println!("{}/5 agree    {:>6} {:>6}", agree, tp_count, fp_count);
        }
    }

    // =========================================
    // Per-detector firing patterns
    // =========================================
    println!("\n=== Per-Detector Firing Rate ===\n");
    println!(
        "{:<16} {:>8} {:>8} {:>10}",
        "Detector", "TP fire%", "FP fire%", "Selectivity"
    );
    println!("{}", "-".repeat(50));
    for i in 0..5 {
        let tp_fires = tps.iter().filter(|d| d.per_detector_anomalous[i]).count();
        let fp_fires = fps.iter().filter(|d| d.per_detector_anomalous[i]).count();
        let tp_rate = if tps.is_empty() {
            0.0
        } else {
            tp_fires as f64 / tps.len() as f64
        };
        let fp_rate = if fps.is_empty() {
            0.0
        } else {
            fp_fires as f64 / fps.len() as f64
        };
        let selectivity = if fp_rate > 0.0 {
            tp_rate / fp_rate
        } else if tp_rate > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        println!(
            "{:<16} {:>7.1}% {:>7.1}% {:>10.1}x",
            DETECTOR_NAMES[i],
            tp_rate * 100.0,
            fp_rate * 100.0,
            selectivity
        );
    }

    // =========================================
    // Mean scores: TP vs FP
    // =========================================
    println!("\n=== Mean Raw Scores: TP vs FP ===\n");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "Detector", "TP mean", "FP mean", "Ratio"
    );
    println!("{}", "-".repeat(50));
    for i in 0..5 {
        let tp_mean: f64 = if tps.is_empty() {
            0.0
        } else {
            tps.iter().map(|d| d.per_detector_scores[i]).sum::<f64>() / tps.len() as f64
        };
        let fp_mean: f64 = if fps.is_empty() {
            0.0
        } else {
            fps.iter().map(|d| d.per_detector_scores[i]).sum::<f64>() / fps.len() as f64
        };
        let ratio = if fp_mean > 1e-6 {
            tp_mean / fp_mean
        } else {
            f64::INFINITY
        };
        println!(
            "{:<16} {:>10.4} {:>10.4} {:>10.1}x",
            DETECTOR_NAMES[i], tp_mean, fp_mean, ratio
        );
    }
    // Fused
    let tp_fused: f64 = if tps.is_empty() {
        0.0
    } else {
        tps.iter().map(|d| d.fused_score).sum::<f64>() / tps.len() as f64
    };
    let fp_fused: f64 = if fps.is_empty() {
        0.0
    } else {
        fps.iter().map(|d| d.fused_score).sum::<f64>() / fps.len() as f64
    };
    println!(
        "{:<16} {:>10.4} {:>10.4} {:>10.1}x",
        "FUSED",
        tp_fused,
        fp_fused,
        if fp_fused > 1e-6 {
            tp_fused / fp_fused
        } else {
            f64::INFINITY
        }
    );

    // =========================================
    // Mean percentile ranks: TP vs FP
    // =========================================
    println!("\n=== Mean Percentile Ranks: TP vs FP ===\n");
    println!(
        "{:<16} {:>10} {:>10} {:>10}",
        "Detector", "TP rank", "FP rank", "Gap"
    );
    println!("{}", "-".repeat(50));
    for i in 0..5 {
        let tp_mean: f64 = if tps.is_empty() {
            0.0
        } else {
            tps.iter().map(|d| d.per_detector_ranks[i]).sum::<f64>() / tps.len() as f64
        };
        let fp_mean: f64 = if fps.is_empty() {
            0.0
        } else {
            fps.iter().map(|d| d.per_detector_ranks[i]).sum::<f64>() / fps.len() as f64
        };
        println!(
            "{:<16} {:>10.4} {:>10.4} {:>10.4}",
            DETECTOR_NAMES[i],
            tp_mean,
            fp_mean,
            tp_mean - fp_mean
        );
    }

    // =========================================
    // Detector co-occurrence matrix
    // =========================================
    println!("\n=== Detector Pair Co-occurrence (TP only) ===\n");
    print!("{:<16}", "");
    for j in 0..5 {
        print!("{:>10}", DETECTOR_NAMES[j]);
    }
    println!();
    for i in 0..5 {
        print!("{:<16}", DETECTOR_NAMES[i]);
        for j in 0..5 {
            let both = tps
                .iter()
                .filter(|d| d.per_detector_anomalous[i] && d.per_detector_anomalous[j])
                .count();
            print!("{:>10}", both);
        }
        println!();
    }

    println!("\n=== Detector Pair Co-occurrence (FP only) ===\n");
    print!("{:<16}", "");
    for j in 0..5 {
        print!("{:>10}", DETECTOR_NAMES[j]);
    }
    println!();
    for i in 0..5 {
        print!("{:<16}", DETECTOR_NAMES[i]);
        for j in 0..5 {
            let both = fps
                .iter()
                .filter(|d| d.per_detector_anomalous[i] && d.per_detector_anomalous[j])
                .count();
            print!("{:>10}", both);
        }
        println!();
    }

    // =========================================
    // All detections detail
    // =========================================
    println!("\n=== All Detections ===\n");
    println!(
        "{:<18} {:>4} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Time", "TP", "Fused", "Agree", "ZScor", "CUSUM", "Enrgy", "RoC", "Multi", "Event"
    );
    println!("{}", "-".repeat(100));
    for d in &detections {
        let tp_str = if d.is_tp { "TP" } else { "FP" };
        println!(
            "{} {:>4} {:>6.3} {:>3}/5 {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3}  {}",
            d.ts.format("%m-%d %H:%M"),
            tp_str,
            d.fused_score,
            d.agreement,
            d.per_detector_scores[0],
            d.per_detector_scores[1],
            d.per_detector_scores[2],
            d.per_detector_scores[3],
            d.per_detector_scores[4],
            d.event_name
        );
    }
}

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

fn find_nearest(entries: &[(DateTime<Utc>, f64)], ts: DateTime<Utc>) -> Option<f64> {
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

fn parse_ts(s: &str) -> DateTime<Utc> {
    parse_ts_opt(s).unwrap()
}
fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .ok()
        .map(|ndt| ndt.and_utc())
}
