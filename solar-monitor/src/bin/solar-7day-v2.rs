//! V2: Run 6 detectors against real 7-day data with all channels.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::collections::BTreeMap;

use solar_monitor::detection::rank_fusion::{RankFusionDetector, DETECTOR_NAMES, N_DETECTORS};
use solar_monitor::feeds::xray::FlareClass;

fn main() {
    println!("=== 7-Day V2: 6 Detectors, All Channels ===\n");

    let xray_long = load_channel("/tmp/xrays-7-day.json", "0.1-0.8nm");
    let xray_short = load_channel("/tmp/xrays-7-day.json", "0.05-0.4nm");
    let electrons = load_channel_generic("/tmp/electrons-7day.json", ">=2 MeV");
    let protons = load_channel_generic("/tmp/protons-7day.json", ">=1 MeV");

    println!(
        "X-ray long: {}, short: {}",
        xray_long.len(),
        xray_short.len()
    );
    println!("Electrons: {}, Protons: {}", electrons.len(), protons.len());

    // EMA-smooth the 5-min feeds
    let electrons_smooth = ema_smooth(&electrons, 0.15);
    let protons_smooth = ema_smooth(&protons, 0.15);

    // Known events
    let x_begin = parse_ts("2026-03-30T02:50:00Z");
    let x_end = parse_ts("2026-03-30T05:50:00Z");
    let m_begin = parse_ts("2026-03-28T03:30:00Z");
    let m_end = parse_ts("2026-03-28T05:00:00Z");

    let mut fusion = RankFusionDetector::new(0.5);
    let mut last_det: Option<DateTime<Utc>> = None;
    let mut tp_count = 0usize;
    let mut fp_count = 0usize;

    // Track per-detector scores during events
    let mut x_max_scores = [0.0f64; N_DETECTORS];
    let mut x_max_fused = 0.0f64;
    let mut m_max_scores = [0.0f64; N_DETECTORS];
    let mut m_max_fused = 0.0f64;

    // All detections
    struct Det {
        ts: DateTime<Utc>,
        fused: f64,
        agree: usize,
        scores: [f64; N_DETECTORS],
        is_tp: bool,
        label: String,
    }
    let mut detections: Vec<Det> = Vec::new();

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }

        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);

        fusion.ingest(long_flux, short_flux, electron, proton, ts);

        let diag = fusion.diagnostics();
        let scores: [f64; N_DETECTORS] = std::array::from_fn(|i| diag.raw_scores[i].raw_score);

        // Track max during events
        if ts >= x_begin && ts <= x_end {
            for i in 0..N_DETECTORS {
                x_max_scores[i] = x_max_scores[i].max(scores[i]);
            }
            x_max_fused = x_max_fused.max(diag.fused_score);
        }
        if ts >= m_begin && ts <= m_end {
            for i in 0..N_DETECTORS {
                m_max_scores[i] = m_max_scores[i].max(scores[i]);
            }
            m_max_fused = m_max_fused.max(diag.fused_score);
        }

        if fusion.is_anomalous() {
            let dominated = last_det.map_or(false, |l| (ts - l).num_seconds() < 1800);
            if !dominated {
                let in_x = ts >= x_begin && ts <= x_end;
                let in_m = ts >= m_begin && ts <= m_end;
                let is_tp = in_x || in_m;
                if is_tp {
                    tp_count += 1;
                } else {
                    fp_count += 1;
                }
                let label = if in_x {
                    "X1.5"
                } else if in_m {
                    "M1.1"
                } else {
                    FlareClass::from_flux(long_flux).label()
                };
                detections.push(Det {
                    ts,
                    fused: diag.fused_score,
                    agree: diag.detector_agreement,
                    scores,
                    is_tp,
                    label: label.to_string(),
                });
                last_det = Some(ts);
            }
        }
    }

    // Results
    println!("\n=== Max Scores During Events ===\n");
    println!("{:<16} {:>10} {:>10}", "Detector", "X1.5 max", "M1.1 max");
    println!("{}", "-".repeat(40));
    for i in 0..N_DETECTORS {
        println!(
            "{:<16} {:>10.3} {:>10.3}",
            DETECTOR_NAMES[i], x_max_scores[i], m_max_scores[i]
        );
    }
    println!(
        "{:<16} {:>10.3} {:>10.3}",
        "FUSED", x_max_fused, m_max_fused
    );

    println!("\n=== Detection Summary ===\n");
    println!(
        "TP: {}, FP: {}, Ratio: 1:{:.1}",
        tp_count,
        fp_count,
        if tp_count > 0 {
            fp_count as f64 / tp_count as f64
        } else {
            f64::INFINITY
        }
    );

    // Agreement distribution
    println!("\n{:<12} {:>6} {:>6}", "Agreement", "TP", "FP");
    println!("{}", "-".repeat(28));
    for agree in 0..=6 {
        let tp = detections
            .iter()
            .filter(|d| d.is_tp && d.agree == agree)
            .count();
        let fp = detections
            .iter()
            .filter(|d| !d.is_tp && d.agree == agree)
            .count();
        if tp > 0 || fp > 0 {
            println!("{}/6 agree    {:>6} {:>6}", agree, tp, fp);
        }
    }

    // Print all detections with timing relative to events
    println!("\n=== All Detections ===\n");
    for d in &detections {
        let tp = if d.is_tp { "TP" } else { "FP" };
        // Hours relative to nearest event
        let x_dt = (d.ts - x_begin).num_seconds() as f64 / 3600.0;
        let m_dt = (d.ts - m_begin).num_seconds() as f64 / 3600.0;
        let nearest = if x_dt.abs() < m_dt.abs() {
            format!("X{:+.1}h", x_dt)
        } else {
            format!("M{:+.1}h", m_dt)
        };
        println!(
            "{} {} fused={:.3} agree={}/6 near={}",
            d.ts.format("%m-%d %H:%M"),
            tp,
            d.fused,
            d.agree,
            nearest
        );
    }

    // Sweep: threshold × min_agreement
    println!("\n=== Threshold × Agreement Sweep ===\n");
    println!(
        "{:<12} {:>5} {:>5} {:>5} {:>8} {:>8}",
        "Config", "TP", "FP", "FN", "Prec%", "FP:TP"
    );
    println!("{}", "-".repeat(55));

    for thresh in &[0.4f64, 0.5, 0.6, 0.7] {
        for min_agree in &[1usize, 2, 3] {
            let mut f = RankFusionDetector::with_agreement(*thresh, *min_agree);
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut last: Option<DateTime<Utc>> = None;

            for (&ts, &long_flux) in &xray_long {
                if long_flux < 1e-9 {
                    continue;
                }
                let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
                let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
                let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);
                f.ingest(long_flux, short_flux, electron, proton, ts);

                if f.is_anomalous() {
                    let dom = last.map_or(false, |l| (ts - l).num_seconds() < 1800);
                    if !dom {
                        let is_tp =
                            (ts >= x_begin && ts <= x_end) || (ts >= m_begin && ts <= m_end);
                        if is_tp {
                            tp += 1;
                        } else {
                            fp += 1;
                        }
                        last = Some(ts);
                    }
                }
            }
            let fn_count = 2 - tp.min(2);
            let prec = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64 * 100.0
            } else {
                0.0
            };
            let ratio = if tp > 0 {
                format!("1:{:.1}", fp as f64 / tp as f64)
            } else {
                "inf".into()
            };
            println!(
                "t={:.1} a>={:<5} {:>5} {:>5} {:>5} {:>7.1}% {:>8}",
                thresh, min_agree, tp, fp, fn_count, prec, ratio
            );
        }
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

/// EMA smooth a 5-min feed into a continuous series.
fn ema_smooth(data: &BTreeMap<DateTime<Utc>, f64>, alpha: f64) -> Vec<(DateTime<Utc>, f64)> {
    let mut result = Vec::with_capacity(data.len());
    let mut ema = None;
    for (&ts, &val) in data {
        let log_val = if val > 0.0 { val.log10() } else { -2.0 };
        let smoothed = match ema {
            None => {
                ema = Some(log_val);
                log_val
            }
            Some(prev) => {
                let s = alpha * log_val + (1.0 - alpha) * prev;
                ema = Some(s);
                s
            }
        };
        result.push((ts, 10.0_f64.powf(smoothed)));
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
