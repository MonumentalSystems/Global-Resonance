//! Combined GOES X-ray + SHARP training for the criticality detector.
//!
//! Replays both data streams chronologically through the detector:
//! - GOES X-ray at 1-minute cadence → ingest() for timing/hardness
//! - SHARP at 12-minute cadence → ingest_with_sharp_full() for magnetic topology
//!
//! This is how the detector operates in production. X-ray gives the
//! energy state (when is J near J_c), SHARP gives the magnetic capability
//! (can this active region flare).
//!
//! Usage:
//!   solar-train-combined --goes-dir solar-monitor/data/goes_data \
//!                        --sharp-dir solar-monitor/data/sharp_data \
//!                        --flares solar-monitor/data/catalogs/solar_flares.csv

use chrono::{Datelike, Timelike, NaiveDateTime, Utc, TimeZone, Duration};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use solar_monitor::detection::criticality::CriticalityDetector;
use solar_monitor::detection::planetary_kan::{PlanetaryKAN, N_BODIES, date_to_jd};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

const PREDICTION_WINDOW_MIN: i64 = 120;
const LR_KAN: f32 = 0.05;
const LR_WEIGHTS: f64 = 0.005;
const EVAL_STRIDE: usize = 10; // evaluate every 10 minutes
const CHECKPOINT_PATH: &str = "solar-monitor/data/combined_checkpoint.json";

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    weights: [f64; 5],
    kan: PlanetaryKAN,
    best_tss: f64,
    epoch: usize,
}

/// SHARP snapshot for a single active region at one timestamp.
struct SharpSnapshot {
    usflux: f64, meangbz: f64, meanjzh: f64, totusjh: f64, shrgt45: f64,
    r_value: f64, totpot: f64, totusjz: f64, savncpp: f64, absnjzh: f64,
    meanalp: f64, area_acr: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut goes_dir = PathBuf::from("solar-monitor/data/goes_data");
    let mut sharp_dir = PathBuf::from("solar-monitor/data/sharp_data");
    let mut flare_path = PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv");
    let mut epochs = 3usize;
    let mut year_start = 2017u32;
    let mut year_end = 2025u32;
    let mut patience_cfg = 20usize;
    let mut resume = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--goes-dir" => { i += 1; goes_dir = PathBuf::from(&args[i]); }
            "--sharp-dir" => { i += 1; sharp_dir = PathBuf::from(&args[i]); }
            "--flares" => { i += 1; flare_path = PathBuf::from(&args[i]); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap_or(3); }
            "--patience" => { i += 1; patience_cfg = args[i].parse().unwrap_or(20); }
            "--resume" => { resume = true; }
            "--years" => { i += 1; let parts: Vec<&str> = args[i].split('-').collect();
                if parts.len() == 2 {
                    year_start = parts[0].parse().unwrap_or(2017);
                    year_end = parts[1].parse().unwrap_or(2025);
                }
            }
            _ => {}
        }
        i += 1;
    }

    println!("={:=<60}", "");
    println!("  COMBINED GOES + SHARP CRITICALITY TRAINING");
    println!("={:=<60}", "");
    println!("  Years: {}-{}", year_start, year_end);

    // Load flare catalog
    let flares = load_flare_datetimes(&flare_path);
    println!("  Flares: {}", flares.len());

    let (mut weights, mut kan, mut best_tss, start_epoch) = if resume {
        match std::fs::read_to_string(CHECKPOINT_PATH) {
            Ok(json) => match serde_json::from_str::<Checkpoint>(&json) {
                Ok(ckpt) => {
                    println!("  Resumed from checkpoint: epoch {}, TSS {:.4}", ckpt.epoch, ckpt.best_tss);
                    (ckpt.weights, ckpt.kan, ckpt.best_tss, ckpt.epoch)
                }
                Err(e) => { eprintln!("  Bad checkpoint: {e}, starting fresh"); ([0.25, 0.20, 0.25, 0.20, 0.10], PlanetaryKAN::new(8), 0.0, 0) }
            }
            Err(_) => { println!("  No checkpoint found, starting fresh"); ([0.25, 0.20, 0.25, 0.20, 0.10], PlanetaryKAN::new(8), 0.0, 0) }
        }
    } else {
        ([0.25f64, 0.20, 0.25, 0.20, 0.10], PlanetaryKAN::new(8), 0.0f64, 0)
    };
    let mut best_weights = weights;
    let mut best_kan = kan.clone();
    let mut patience_counter = 0usize;

    println!("\n{:>6} | {:>8} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>8} {:>6}",
        "Epoch", "Loss", "TSS", "F1", "w_bal", "w_dis", "w_com", "w_lod", "w_syn", "KAN_E", "evals", "secs");
    println!("{}", "-".repeat(105));

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();
        let mut detector = CriticalityDetector::new(0.5);
        detector.set_score_weights(weights);

        let mut total_loss = 0.0f64;
        let mut n_eval = 0usize;
        let mut preds = Vec::new();
        let mut labels_vec = Vec::new();
        let mut d_weights = [0.0f64; 5];
        let mut kan_grad_acc: Option<KanGrads> = None;

        // Process each year
        for year in year_start..=year_end {
            // Load SHARP data for this year into a time-indexed map
            let sharp_map = load_sharp_year(&sharp_dir, year);
            let sharp_records: usize = sharp_map.values().map(|v| v.len()).sum();
            if epoch == 0 {
                println!("  {} — SHARP: {} bins, {} records", year, sharp_map.len(), sharp_records);
            }

            // Load GOES X-ray for this year
            let goes_path = goes_dir.join(format!("goes16_xrs_{}.csv", year));
            let mut rdr = match csv::Reader::from_path(&goes_path) {
                Ok(r) => r,
                Err(_) => { if epoch == 0 { eprintln!("  {} — No GOES data", year); } continue; }
            };

            let mut step = 0usize;
            let mut last_sharp_key = String::new();

            for result in rdr.records() {
                let record = match result { Ok(r) => r, Err(_) => continue };

                let time_str = match record.get(0) { Some(s) => s, None => continue };
                let xrsa: f64 = record.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let xrsb: f64 = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                if xrsb < 1e-9 { step += 1; continue; }

                let ts = match NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S") {
                    Ok(dt) => Utc.from_utc_datetime(&dt),
                    Err(_) => { step += 1; continue; }
                };

                // Check if we have SHARP data at this timestamp (12-min cadence)
                // Round down to nearest 12 minutes
                let minute = ts.format("%M").to_string().parse::<u32>().unwrap_or(0);
                let sharp_minute = (minute / 12) * 12;
                let sharp_key = format!("{}-{:02}", ts.format("%Y-%m-%dT%H"), sharp_minute);

                if sharp_key != last_sharp_key {
                    if let Some(snaps) = sharp_map.get(&sharp_key) {
                        // Feed strongest AR's SHARP data + X-ray.
                        // Do NOT call plain ingest() between SHARP bins —
                        // it uses v1 scoring which washes out v3 signal via score_ema.
                        if let Some(s) = snaps.first() {
                            detector.ingest_with_sharp_full(
                                s.usflux, s.meangbz, s.meanjzh, s.totusjh, s.shrgt45,
                                s.r_value, s.totpot, s.totusjz, s.savncpp, s.absnjzh,
                                s.meanalp, s.area_acr,
                                xrsb, ts,
                            );
                            eval_step(&detector, &kan, &flares, ts, epoch, n_eval,
                                &weights, &mut total_loss, &mut preds, &mut labels_vec,
                                &mut d_weights, &mut kan_grad_acc, &mut n_eval);
                        }
                        last_sharp_key = sharp_key.clone();
                        step += 1;
                    }
                }
                // X-ray-only minutes: skip (v1 scoring dilutes v3 via score_ema)
            }
        }

        // Apply gradients
        if n_eval > 0 {
            let scale = 1.0 / n_eval as f64;
            for i in 0..5 {
                weights[i] -= LR_WEIGHTS * scale * d_weights[i];
                weights[i] = weights[i].max(0.01);
            }
            let w_sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() { *w /= w_sum; }

            if let Some(ref grads) = kan_grad_acc {
                let s = 1.0 / n_eval as f32;
                let mut scaled = grads.clone();
                scaled.d_bias *= s;
                for i in 0..N_BODIES {
                    scaled.d_weights[i] *= s;
                    for k in 0..kan.n_knots { scaled.d_spline_coeffs[i][k] *= s; }
                }
                kan.sgd_step(&scaled, LR_KAN);
            }
        }

        let avg_loss = total_loss / n_eval.max(1) as f64;
        let (tss, f1) = compute_tss_f1(&preds, &labels_vec);
        let kan_energy: f32 = kan.splines.iter()
            .map(|s| s.coeffs.iter().map(|c| c*c).sum::<f32>()).sum::<f32>().sqrt();
        let elapsed = t0.elapsed().as_secs_f64();

        // Score distribution diagnostic (first epoch only)
        if epoch == 0 && !preds.is_empty() {
            let mut sorted = preds.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let pos: Vec<&f32> = preds.iter().zip(labels_vec.iter()).filter(|(_, l)| **l > 0.5).map(|(p, _)| p).collect();
            let neg: Vec<&f32> = preds.iter().zip(labels_vec.iter()).filter(|(_, l)| **l <= 0.5).map(|(p, _)| p).collect();
            let pos_mean = if pos.is_empty() { 0.0 } else { pos.iter().map(|&&v| v as f64).sum::<f64>() / pos.len() as f64 };
            let neg_mean = if neg.is_empty() { 0.0 } else { neg.iter().map(|&&v| v as f64).sum::<f64>() / neg.len() as f64 };
            println!("  Score dist: min={:.4} p25={:.4} p50={:.4} p75={:.4} max={:.4}",
                sorted[0], sorted[n/4], sorted[n/2], sorted[3*n/4], sorted[n-1]);
            println!("  Pos mean={:.4} (n={}), Neg mean={:.4} (n={})",
                pos_mean, pos.len(), neg_mean, neg.len());
        }

        let epoch_num = start_epoch + epoch + 1;
        print!("{:6} | {:8.4} {:8.4} {:8.4} | {:6.3} {:6.3} {:6.3} {:6.3} {:6.3} | {:6.3} {:>8} {:6.1}",
            epoch_num, avg_loss, tss, f1,
            weights[0], weights[1], weights[2], weights[3], weights[4],
            kan_energy, n_eval, elapsed);

        if tss > best_tss {
            best_tss = tss;
            best_weights = weights;
            best_kan = kan.clone();
            patience_counter = 0;
            // Save checkpoint
            let ckpt = Checkpoint { weights, kan: kan.clone(), best_tss: tss, epoch: epoch_num };
            if let Ok(json) = serde_json::to_string_pretty(&ckpt) {
                let _ = std::fs::write(CHECKPOINT_PATH, json);
            }
            print!(" * saved");
        } else {
            patience_counter += 1;
            if patience_counter >= patience_cfg {
                println!("\nEarly stop at epoch {} (no improvement for {} epochs)", epoch_num, patience_cfg);
                break;
            }
        }
        println!();
    }

    println!("\nBest TSS: {:.4}", best_tss);
    println!("Best weights: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        best_weights[0], best_weights[1], best_weights[2], best_weights[3], best_weights[4]);
    best_kan.print_weights();
    println!("Checkpoint: {}", CHECKPOINT_PATH);
}

use solar_monitor::detection::planetary_kan::PlanetaryKANGrads as KanGrads;

fn eval_step(
    detector: &CriticalityDetector, kan: &PlanetaryKAN,
    flares: &[chrono::DateTime<Utc>], ts: chrono::DateTime<Utc>,
    epoch: usize, step: usize,
    weights: &[f64; 5],
    total_loss: &mut f64, preds: &mut Vec<f32>, labels_vec: &mut Vec<f32>,
    d_weights: &mut [f64; 5], kan_grad_acc: &mut Option<KanGrads>,
    n_eval: &mut usize,
) {
    let window_end = ts + Duration::minutes(PREDICTION_WINDOW_MIN);
    let label = flares.iter().any(|&ft| ft > ts && ft <= window_end);
    let label_f = if label { 1.0f64 } else { 0.0 };

    let base_score = detector.raw_physics_score();
    let jd = date_to_jd(ts.year(), ts.month(), ts.day());
    let angles = PlanetaryKAN::angles_from_jd(jd);
    let kan_mod = kan.forward(&angles) as f64;
    let final_score = (base_score * kan_mod).clamp(0.0, 1.0);

    let p = final_score.clamp(1e-7, 1.0 - 1e-7);
    let loss = -(label_f * p.ln() + (1.0 - label_f) * (1.0 - p).ln());
    *total_loss += loss;

    // KAN gradient
    let d_final = -label_f / p + (1.0 - label_f) / (1.0 - p);
    let d_kan = d_final * base_score;
    let kan_grads = kan.backward(&angles, d_kan as f32);
    match kan_grad_acc {
        None => *kan_grad_acc = Some(kan_grads),
        Some(acc) => {
            acc.d_bias += kan_grads.d_bias;
            for i in 0..N_BODIES {
                acc.d_weights[i] += kan_grads.d_weights[i];
                for k in 0..kan.n_knots { acc.d_spline_coeffs[i][k] += kan_grads.d_spline_coeffs[i][k]; }
            }
        }
    }

    // SPSA weight gradient
    {
        let eps = 0.02;
        let hash = ((*n_eval as u64).wrapping_mul(6364136223846793005).wrapping_add(epoch as u64)) as u32;
        let perturbation: Vec<f64> = (0..5).map(|i| if (hash >> i) & 1 == 0 { eps } else { -eps }).collect();
        let w_sum: f64 = weights.iter().sum();
        let p_sum: f64 = perturbation.iter().sum();
        let ps = (base_score * (1.0 + p_sum / w_sum.max(0.01)) * kan_mod).clamp(0.0, 1.0);
        let pp = ps.clamp(1e-7, 1.0 - 1e-7);
        let lp = -(label_f * pp.ln() + (1.0 - label_f) * (1.0 - pp).ln());
        for i in 0..5 { d_weights[i] += (lp - loss) / perturbation[i]; }
    }

    preds.push(final_score as f32);
    labels_vec.push(if label { 1.0 } else { 0.0 });
    *n_eval += 1;
}

/// Load SHARP data for one year into a BTreeMap keyed by "YYYY-MM-DDTHH-MM" (12-min bins).
fn load_sharp_year(sharp_dir: &Path, year: u32) -> BTreeMap<String, Vec<SharpSnapshot>> {
    let mut map: BTreeMap<String, Vec<SharpSnapshot>> = BTreeMap::new();

    for month in 1..=12u32 {
        let path = sharp_dir.join(format!("sharp_{}_{:02}.csv.gz", year, month));
        if !path.exists() { continue; }

        let file = match std::fs::File::open(&path) { Ok(f) => f, Err(_) => continue };
        let decoder = GzDecoder::new(file);
        let mut rdr = csv::Reader::from_reader(decoder);

        for result in rdr.records() {
            let record = match result { Ok(r) => r, Err(_) => continue };
            let time_str = match record.get(0) { Some(s) => s, None => continue };

            // Parse timestamp: "2017.01.01_00:00:00_TAI" or "2023-01-01T00:00:00"
            let ts_clean = time_str
                .replace("_TAI", "")   // strip suffix BEFORE replacing T
                .replacen('_', " ", 1) // date_time separator
                .replace('T', " ");    // ISO separator
            let dt = match NaiveDateTime::parse_from_str(&ts_clean, "%Y.%m.%d %H:%M:%S")
                .or_else(|_| NaiveDateTime::parse_from_str(&ts_clean, "%Y-%m-%d %H:%M:%S")) {
                Ok(dt) => dt,
                Err(_) => continue,
            };

            // Round to 12-min bin
            let minute = dt.minute();
            let sharp_minute = (minute / 12) * 12;
            let key = format!("{}-{:02}", dt.format("%Y-%m-%dT%H"), sharp_minute);

            let snap = SharpSnapshot {
                usflux: record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                meangbz: record.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                meanjzh: record.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                totusjh: record.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                shrgt45: record.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                area_acr: record.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                r_value: record.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                totpot: record.get(9).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                totusjz: record.get(10).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                savncpp: record.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                absnjzh: record.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                meanalp: record.get(13).and_then(|s| s.parse().ok()).unwrap_or(0.0),
            };

            map.entry(key).or_default().push(snap);
        }
    }

    // Sort each bin by totpot (strongest AR first)
    for snaps in map.values_mut() {
        snaps.sort_by(|a, b| b.totpot.partial_cmp(&a.totpot).unwrap_or(std::cmp::Ordering::Equal));
    }

    map
}

fn load_flare_datetimes(path: &Path) -> Vec<chrono::DateTime<Utc>> {
    let mut times = Vec::new();
    if let Ok(mut rdr) = csv::Reader::from_path(path) {
        for result in rdr.records() {
            if let Ok(record) = result {
                if let Some(ts) = record.get(0) {
                    if let Ok(dt) = NaiveDateTime::parse_from_str(ts, "%Y-%m-%d %H:%M:%S") {
                        times.push(Utc.from_utc_datetime(&dt));
                    }
                }
            }
        }
    }
    times.sort();
    times
}

fn compute_tss_f1(preds: &[f32], labels: &[f32]) -> (f64, f64) {
    let mut best_tss = 0.0f64;
    let mut best_f1 = 0.0f64;
    for thresh_i in 0..20 {
        let thresh = thresh_i as f32 * 0.05;
        let (mut tp, mut fp, mut tn, mut r#fn) = (0u32, 0u32, 0u32, 0u32);
        for (p, l) in preds.iter().zip(labels) {
            match (*p >= thresh, *l > 0.5) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => r#fn += 1,
            }
        }
        let tpr = if tp + r#fn > 0 { tp as f64 / (tp + r#fn) as f64 } else { 0.0 };
        let fpr = if fp + tn > 0 { fp as f64 / (fp + tn) as f64 } else { 0.0 };
        let tss = tpr - fpr;
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let f1 = if prec + tpr > 0.0 { 2.0 * prec * tpr / (prec + tpr) } else { 0.0 };
        if tss > best_tss { best_tss = tss; best_f1 = f1; }
    }
    (best_tss, best_f1)
}
