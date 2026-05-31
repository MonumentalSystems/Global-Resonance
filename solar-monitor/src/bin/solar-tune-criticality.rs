//! Parameter sweep for the criticality detector.
//!
//! Sweeps gamma, inhibition_threshold, discharge_gain, temporal_harmonic_amplitude,
//! and standing_wave_amplitude against real GOES data. Measures:
//!
//! - Quiet-sun baseline score (want LOW)
//! - Pre-flare window score, -360m to -60m before X1.5 (want HIGH)
//! - Signal-to-noise ratio = pre_flare_mean / (baseline_mean + 0.01)
//! - Separation = pre_flare_mean - baseline_mean

use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use solar_monitor::detection::criticality::CriticalityDetector;
use std::collections::BTreeMap;

/// Parameter combination to test.
struct ParamSet {
    gamma: f32,
    inhibition_threshold: f32,
    discharge_gain: f32,
    temporal_harmonic_amplitude: f32,
    standing_wave_amplitude: f32,
}

/// Result of one parameter evaluation.
struct EvalResult {
    params: ParamSet,
    baseline_mean: f64,
    baseline_std: f64,
    preflare_mean: f64,
    preflare_std: f64,
    snr: f64,
    separation: f64,
}

fn main() {
    // Load data (same as solar-precursor).
    let xray_long = load_channel("/tmp/xrays-7-day.json", "0.1-0.8nm");
    let xray_short = load_channel("/tmp/xrays-7-day.json", "0.05-0.4nm");
    let protons = load_channel_generic("/tmp/protons-7day.json", ">=1 MeV");
    let protons_smooth = ema_smooth(&protons, 0.15);

    let x_begin = parse_ts("2026-03-30T01:45:00Z");

    // Quiet window: 24h of quiet sun well before either event.
    let quiet_start = parse_ts("2026-03-25T00:00:00Z");
    let quiet_end = parse_ts("2026-03-26T00:00:00Z");

    // Pre-flare window: -6h to -1h before X1.5 (the interesting gap).
    let preflare_start = parse_ts("2026-03-29T19:45:00Z");
    let preflare_end = parse_ts("2026-03-30T00:45:00Z");

    // Parameter grid.
    let gammas = [0.02, 0.05, 0.10, 0.15, 0.25];
    let inhibitions = [0.3, 0.5, 0.7, 1.0, 1.5];
    let discharges = [0.5, 0.75, 1.0, 1.5];
    let temporal_amps = [0.0, 0.2, 0.4, 0.6, 0.8];
    let standing_amps = [0.0, 0.15, 0.3, 0.5];

    let total = gammas.len()
        * inhibitions.len()
        * discharges.len()
        * temporal_amps.len()
        * standing_amps.len();
    println!("Sweeping {} parameter combinations...\n", total);

    let mut results: Vec<EvalResult> = Vec::with_capacity(total);
    let mut count = 0;

    for &gamma in &gammas {
        for &inhib in &inhibitions {
            for &discharge in &discharges {
                for &temp_amp in &temporal_amps {
                    for &stand_amp in &standing_amps {
                        let params = ParamSet {
                            gamma,
                            inhibition_threshold: inhib,
                            discharge_gain: discharge,
                            temporal_harmonic_amplitude: temp_amp,
                            standing_wave_amplitude: stand_amp,
                        };

                        let result = evaluate(
                            &params,
                            &xray_long,
                            &xray_short,
                            &protons_smooth,
                            quiet_start,
                            quiet_end,
                            preflare_start,
                            preflare_end,
                        );

                        results.push(result);
                        count += 1;
                        if count % 100 == 0 {
                            eprint!("\r  {}/{}", count, total);
                        }
                    }
                }
            }
        }
    }
    eprintln!("\r  {}/{} done.", count, total);

    // Sort by SNR descending.
    results.sort_by(|a, b| {
        b.snr
            .partial_cmp(&a.snr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print top 20.
    println!("\n=== Top 20 by Signal-to-Noise Ratio ===\n");
    println!(
        "{:>5} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Rank",
        "gamma",
        "inhib",
        "disch",
        "t_amp",
        "s_amp",
        "BL_mean",
        "BL_std",
        "PF_mean",
        "PF_std",
        "SNR",
        "Sep"
    );
    println!("{}", "-".repeat(108));

    for (i, r) in results.iter().take(20).enumerate() {
        println!(
            "{:>5} {:>6.3} {:>6.2} {:>6.2} {:>6.2} {:>6.2}  {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.2} {:>7.4}",
            i + 1,
            r.params.gamma,
            r.params.inhibition_threshold,
            r.params.discharge_gain,
            r.params.temporal_harmonic_amplitude,
            r.params.standing_wave_amplitude,
            r.baseline_mean,
            r.baseline_std,
            r.preflare_mean,
            r.preflare_std,
            r.snr,
            r.separation,
        );
    }

    // Also print top 20 by separation (absolute difference).
    results.sort_by(|a, b| {
        b.separation
            .partial_cmp(&a.separation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("\n=== Top 20 by Absolute Separation ===\n");
    println!(
        "{:>5} {:>6} {:>6} {:>6} {:>6} {:>6}  {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Rank",
        "gamma",
        "inhib",
        "disch",
        "t_amp",
        "s_amp",
        "BL_mean",
        "BL_std",
        "PF_mean",
        "PF_std",
        "SNR",
        "Sep"
    );
    println!("{}", "-".repeat(108));

    for (i, r) in results.iter().take(20).enumerate() {
        println!(
            "{:>5} {:>6.3} {:>6.2} {:>6.2} {:>6.2} {:>6.2}  {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.2} {:>7.4}",
            i + 1,
            r.params.gamma,
            r.params.inhibition_threshold,
            r.params.discharge_gain,
            r.params.temporal_harmonic_amplitude,
            r.params.standing_wave_amplitude,
            r.baseline_mean,
            r.baseline_std,
            r.preflare_mean,
            r.preflare_std,
            r.snr,
            r.separation,
        );
    }

    // Print the best overall (highest SNR with separation > 0.15).
    let best = results
        .iter()
        .filter(|r| r.separation > 0.15 && r.baseline_mean < 0.35)
        .max_by(|a, b| {
            a.snr
                .partial_cmp(&b.snr)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(b) = best {
        println!("\n=== BEST (SNR with sep>0.15, baseline<0.35) ===");
        println!("  gamma:                     {:.3}", b.params.gamma);
        println!(
            "  inhibition_threshold:      {:.2}",
            b.params.inhibition_threshold
        );
        println!(
            "  discharge_gain:            {:.2}",
            b.params.discharge_gain
        );
        println!(
            "  temporal_harmonic_amp:     {:.2}",
            b.params.temporal_harmonic_amplitude
        );
        println!(
            "  standing_wave_amp:         {:.2}",
            b.params.standing_wave_amplitude
        );
        println!("  baseline_mean:             {:.4}", b.baseline_mean);
        println!("  preflare_mean:             {:.4}", b.preflare_mean);
        println!("  SNR:                       {:.2}", b.snr);
        println!("  separation:                {:.4}", b.separation);
    }
}

fn evaluate(
    params: &ParamSet,
    xray_long: &BTreeMap<DateTime<Utc>, f64>,
    xray_short: &BTreeMap<DateTime<Utc>, f64>,
    protons_smooth: &[(DateTime<Utc>, f64)],
    quiet_start: DateTime<Utc>,
    quiet_end: DateTime<Utc>,
    preflare_start: DateTime<Utc>,
    preflare_end: DateTime<Utc>,
) -> EvalResult {
    let mut det = CriticalityDetector::with_params(
        0.6,
        params.gamma,
        params.inhibition_threshold,
        params.discharge_gain,
        params.temporal_harmonic_amplitude,
        params.standing_wave_amplitude,
    );

    let mut quiet_scores: Vec<f64> = Vec::new();
    let mut preflare_scores: Vec<f64> = Vec::new();

    for (&ts, &long_flux) in xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(xray_short, ts).unwrap_or(long_flux * 0.04);
        let proton = find_nearest_smooth(protons_smooth, ts).unwrap_or(0.3);
        det.ingest(long_flux, short_flux, proton, ts);

        let score = det.score();

        if ts >= quiet_start && ts <= quiet_end {
            quiet_scores.push(score);
        }
        if ts >= preflare_start && ts <= preflare_end {
            preflare_scores.push(score);
        }
    }

    let baseline_mean = mean(&quiet_scores);
    let baseline_std = std_dev(&quiet_scores, baseline_mean);
    let preflare_mean = mean(&preflare_scores);
    let preflare_std = std_dev(&preflare_scores, preflare_mean);
    let snr = preflare_mean / (baseline_mean + 0.01);
    let separation = preflare_mean - baseline_mean;

    EvalResult {
        params: ParamSet {
            gamma: params.gamma,
            inhibition_threshold: params.inhibition_threshold,
            discharge_gain: params.discharge_gain,
            temporal_harmonic_amplitude: params.temporal_harmonic_amplitude,
            standing_wave_amplitude: params.standing_wave_amplitude,
        },
        baseline_mean,
        baseline_std,
        preflare_mean,
        preflare_std,
        snr,
        separation,
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

// --- Data loading (same as solar-precursor) ---

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
