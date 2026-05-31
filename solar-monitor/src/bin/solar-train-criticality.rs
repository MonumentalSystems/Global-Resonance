//! Train the criticality detector's learnable weights on historical data.
//!
//! Replays SHARP time series through the criticality detector in chronological
//! order (streaming, continuous state). At each timestep, compares the detector's
//! score against whether a flare occurred within the prediction window.
//!
//! Learns:
//!   - 5 sub-score weights (balance, discharge, commutator, loading, sync)
//!   - 82 PlanetaryKAN params (9 bodies × 8 knots + 9 weights + 1 bias)
//!   = 87 total learnable parameters on top of the physics backbone
//!
//! Usage:
//!   solar-train-criticality --sharp-dir solar-monitor/data/sharp_data \
//!                           --flares solar-monitor/data/catalogs/solar_flares.csv \
//!                           --epochs 50

use chrono::{Datelike, NaiveDateTime, Utc, TimeZone};
use serde::{Deserialize, Serialize};
use solar_monitor::backtest::sharp_dataset::{SharpDataset, DatasetConfig};
use solar_monitor::detection::criticality::CriticalityDetector;
use solar_monitor::detection::planetary_kan::{PlanetaryKAN, PlanetaryKANGrads, date_to_jd, N_BODIES};
use std::path::PathBuf;

/// Prediction window: flare within this many minutes counts as positive.
const PREDICTION_WINDOW_MIN: i64 = 120;

/// Learning rate for score weights.
const LR_WEIGHTS: f64 = 0.01;
/// Learning rate for KAN params (high because 87 params, big dataset).
const LR_KAN: f32 = 0.05;
const CHECKPOINT_PATH: &str = "solar-monitor/data/criticality_checkpoint.json";

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    weights: [f64; 5],
    kan: PlanetaryKAN,
    best_tss: f64,
    epoch: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut sharp_dir: Option<PathBuf> = None;
    let mut flare_path: Option<PathBuf> = None;
    let mut epochs = 50usize;
    let mut patience_cfg = 20usize;
    let mut resume = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sharp-dir" => { i += 1; sharp_dir = Some(PathBuf::from(&args[i])); }
            "--flares" => { i += 1; flare_path = Some(PathBuf::from(&args[i])); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap_or(50); }
            "--patience" => { i += 1; patience_cfg = args[i].parse().unwrap_or(20); }
            "--resume" => { resume = true; }
            "--help" | "-h" => {
                println!("solar-train-criticality — Train criticality detector weights + PlanetaryKAN");
                println!("  --sharp-dir DIR   SHARP data directory");
                println!("  --flares FILE     Flare catalog CSV");
                println!("  --epochs N        Training epochs (default: 50)");
                println!("  --patience N      Early stop after N epochs without improvement (default: 20)");
                println!("  --resume          Resume from checkpoint");
                return;
            }
            _ => { eprintln!("Unknown: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    let sharp_dir = sharp_dir.unwrap_or_else(|| PathBuf::from("solar-monitor/data/sharp_data"));
    let flare_path = flare_path.unwrap_or_else(|| PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv"));

    println!("={:=<60}", "");
    println!("  CRITICALITY DETECTOR TRAINING");
    println!("  87 learnable params on physics backbone (sep=0.61)");
    println!("={:=<60}", "");

    // Load flare catalog with timestamps
    println!("\nLoading flare catalog...");
    let flare_times = load_flare_times(&flare_path);
    println!("  {} flares loaded", flare_times.len());

    // Load SHARP dataset
    println!("Loading SHARP data...");
    let config = DatasetConfig { seq_len: 10, ..Default::default() };
    let dataset = match SharpDataset::load_gzipped_dir(&sharp_dir, &flare_path, config) {
        Ok(ds) => ds,
        Err(e) => { eprintln!("Failed: {e}"); std::process::exit(1); }
    };
    dataset.print_stats();

    // Initialize detector with KAN
    let mut detector = CriticalityDetector::new(0.5);
    detector.enable_planetary_kan(8);
    let mut kan = PlanetaryKAN::new(8);

    // Initialize score weights — from checkpoint or tuned defaults
    let (mut weights, start_epoch) = if resume {
        match std::fs::read_to_string(CHECKPOINT_PATH) {
            Ok(json) => match serde_json::from_str::<Checkpoint>(&json) {
                Ok(ckpt) => {
                    println!("  Resumed from checkpoint: epoch {}, TSS {:.4}", ckpt.epoch, ckpt.best_tss);
                    kan = ckpt.kan;
                    ([ckpt.weights[0], ckpt.weights[1], ckpt.weights[2], ckpt.weights[3], ckpt.weights[4]], ckpt.epoch)
                }
                Err(e) => { eprintln!("  Bad checkpoint: {e}, starting fresh"); ([0.25, 0.20, 0.25, 0.20, 0.10], 0) }
            }
            Err(_) => { println!("  No checkpoint found, starting fresh"); ([0.25, 0.20, 0.25, 0.20, 0.10], 0) }
        }
    } else {
        ([0.25f64, 0.20, 0.25, 0.20, 0.10], 0)
    };

    println!("\nLearnable params: 5 weights + {} KAN = {} total",
        kan.param_count(), 5 + kan.param_count());
    println!("Epochs: {}, patience: {}", epochs, patience_cfg);
    println!();

    // Training: replay samples chronologically, accumulate gradients
    let all_samples = &dataset.train;
    println!("{:>6} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>6}",
        "Epoch", "Loss", "TSS", "F1", "w_bal", "w_dis", "w_com", "w_lod", "w_syn", "KAN_E");
    println!("{}", "-".repeat(100));

    let mut best_tss = f64::NEG_INFINITY;
    let mut best_weights = weights;
    let mut best_kan = kan.clone();
    let mut patience_counter = 0usize;
    let patience = patience_cfg;

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();

        // Accumulate gradients over all samples
        let mut d_weights = [0.0f64; 5];
        let mut kan_grad_acc: Option<PlanetaryKANGrads> = None;
        let mut total_loss = 0.0f64;
        let mut n_samples = 0usize;
        let mut preds = Vec::new();
        let mut labels = Vec::new();

        // Reset detector state for each epoch (fresh pass through data)
        detector = CriticalityDetector::new(0.5);
        detector.set_score_weights(weights);

        for sample in all_samples.iter() {
            if sample.jd < 1.0 { continue; } // skip samples without timestamps

            // Compute the detector's raw sub-scores by stepping through the lattice
            // We need the 5 sub-scores individually for weight gradient
            // For now, use the full score as a proxy and compute weight gradients
            // via finite-difference style perturbation

            // Synthesize X-ray from flare proximity
            let xray = synthesize_xray(sample.jd, &flare_times);

            // Build timestamp from JD
            let ts = jd_to_datetime(sample.jd);

            // Feed the last SHARP observation through the detector
            let sharp = &sample.features;
            let seq_len = 10;
            let last = seq_len - 1;
            if sharp.len() >= seq_len * 9 {
                let base = last * 9;
                detector.ingest_with_sharp_full(
                    sharp[base + 2] as f64 * 1e22,  // USFLUX (denormalize approx)
                    0.0, 0.0,                        // meangbz, meanjzh (not in 9-field set)
                    sharp[base + 0] as f64 * 1e4,   // TOTUSJH
                    0.0,                              // shrgt45
                    sharp[base + 4] as f64 * 1e4,   // R_VALUE
                    sharp[base + 5] as f64 * 1e24,  // TOTPOT
                    sharp[base + 1] as f64 * 1e4,   // TOTUSJZ
                    sharp[base + 6] as f64 * 1e4,   // SAVNCPP
                    sharp[base + 8] as f64 * 1e4,   // ABSNJZH
                    sharp[base + 3] as f64,          // MEANALP
                    sharp[base + 7] as f64 * 1e12,  // AREA_ACR
                    xray,
                    ts,
                );
            }

            // Get score and label
            let score = detector.raw_physics_score();
            let label = sample.label as f64;

            // Planetary KAN forward
            let angles = PlanetaryKAN::angles_from_jd(sample.jd);
            let kan_mod = kan.forward(&angles) as f64;
            let final_score = (score * kan_mod).clamp(0.0, 1.0);

            // BCE loss
            let p = final_score.clamp(1e-7, 1.0 - 1e-7);
            let loss = -(label * p.ln() + (1.0 - label) * (1.0 - p).ln());
            total_loss += loss;

            // Gradient of BCE w.r.t. final_score
            let d_final = -label / p + (1.0 - label) / (1.0 - p);

            // d_final/d_kan_mod = score
            let d_kan = d_final * score;
            let kan_grads = kan.backward(&angles, d_kan as f32);
            match &mut kan_grad_acc {
                None => kan_grad_acc = Some(kan_grads),
                Some(acc) => {
                    acc.d_bias += kan_grads.d_bias;
                    for i in 0..N_BODIES {
                        acc.d_weights[i] += kan_grads.d_weights[i];
                        for k in 0..kan.n_knots {
                            acc.d_spline_coeffs[i][k] += kan_grads.d_spline_coeffs[i][k];
                        }
                    }
                }
            }

            // Weight gradient via SPSA (simultaneous perturbation):
            // Perturb all weights by random ±ε, measure loss change.
            // Efficient: one extra forward per sample (not 5).
            {
                let eps = 0.02;
                let hash = ((n_samples as u64).wrapping_mul(6364136223846793005).wrapping_add(epoch as u64)) as u32;
                let perturbation: Vec<f64> = (0..5).map(|i| {
                    if (hash >> i) & 1 == 0 { eps } else { -eps }
                }).collect();

                // Perturbed score = base_score computed with perturbed weights
                // We approximate: perturbed_score ≈ score + Σ_i perturbation_i * sub_score_i
                // Since we don't have sub_scores, use: Δscore ≈ score * Σ_i perturbation_i / Σ_i w_i
                let w_sum: f64 = weights.iter().sum();
                let p_sum: f64 = perturbation.iter().sum();
                let perturbed_score = (score * (1.0 + p_sum / w_sum.max(0.01)) * kan_mod).clamp(0.0, 1.0);
                let p_p = perturbed_score.clamp(1e-7, 1.0 - 1e-7);
                let loss_plus = -(label * p_p.ln() + (1.0 - label) * (1.0 - p_p).ln());
                let d_loss = loss_plus - loss;

                for i in 0..5 {
                    d_weights[i] += d_loss / perturbation[i];
                }
            }

            preds.push(final_score as f32);
            labels.push(sample.label);
            n_samples += 1;
        }

        // Apply weight gradients (SPSA)
        {
            let scale = 1.0 / n_samples.max(1) as f64;
            for i in 0..5 {
                weights[i] -= LR_WEIGHTS * scale * d_weights[i];
                weights[i] = weights[i].max(0.01); // keep positive
            }
            // Normalize weights to sum to 1
            let w_sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() { *w /= w_sum; }
            detector.set_score_weights(weights);
        }

        // Apply KAN gradients
        if let Some(ref grads) = kan_grad_acc {
            let scale = 1.0 / n_samples.max(1) as f32;
            let mut scaled = grads.clone();
            scaled.d_bias *= scale;
            for i in 0..N_BODIES {
                scaled.d_weights[i] *= scale;
                for k in 0..kan.n_knots {
                    scaled.d_spline_coeffs[i][k] *= scale;
                }
            }
            kan.sgd_step(&scaled, LR_KAN);
        }

        // Evaluate TSS
        let avg_loss = total_loss / n_samples.max(1) as f64;
        let (tss, f1) = compute_tss_f1(&preds, &labels);

        // KAN energy (how much the splines have learned)
        let kan_energy: f32 = kan.splines.iter()
            .map(|s| s.coeffs.iter().map(|c| c * c).sum::<f32>())
            .sum::<f32>().sqrt();

        let epoch_num = start_epoch + epoch + 1;
        print!("{:6} | {:8.4} {:8.4} {:8.4} | {:8.4} {:8.4} {:8.4} {:8.4} {:8.4} | {:6.3}",
            epoch_num, avg_loss, tss, f1,
            weights[0], weights[1], weights[2], weights[3], weights[4],
            kan_energy);

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
            if patience_counter >= patience {
                println!("\nEarly stop at epoch {} (no improvement for {} epochs)", epoch_num, patience);
                break;
            }
        }
        println!();
    }

    println!("\nBest TSS: {:.4}", best_tss);
    println!("Best weights: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        best_weights[0], best_weights[1], best_weights[2], best_weights[3], best_weights[4]);
    println!("Checkpoint: {}", CHECKPOINT_PATH);

    // Print learned KAN weights from best checkpoint
    best_kan.print_weights();
}

/// Load flare times as Julian dates for quick lookup.
fn load_flare_times(path: &std::path::Path) -> Vec<f64> {
    let mut times = Vec::new();
    if let Ok(mut rdr) = csv::Reader::from_path(path) {
        for result in rdr.records() {
            if let Ok(record) = result {
                if let Some(time_str) = record.get(0) {
                    if let Ok(dt) = NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S") {
                        let utc = Utc.from_utc_datetime(&dt);
                        let jd = date_to_jd(utc.year() as i32, utc.month(), utc.day());
                        times.push(jd);
                    }
                }
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times
}

/// Synthesize X-ray flux from proximity to flares.
/// Background C1 (1e-6), elevated near flares.
fn synthesize_xray(jd: f64, flare_times: &[f64]) -> f64 {
    let background = 1e-6; // C1 background
    // Find nearest flare
    let idx = flare_times.partition_point(|&t| t < jd);
    let mut min_dist = f64::MAX;
    if idx > 0 { min_dist = min_dist.min((jd - flare_times[idx - 1]).abs()); }
    if idx < flare_times.len() { min_dist = min_dist.min((flare_times[idx] - jd).abs()); }

    // Elevate X-ray within 0.1 day (~2.4 hours) of a flare
    if min_dist < 0.1 {
        let proximity = 1.0 - min_dist / 0.1; // 1.0 at flare, 0.0 at edge
        background + proximity * 1e-4 // up to M1 at peak
    } else {
        background
    }
}

/// Convert Julian date to DateTime<Utc> (approximate).
fn jd_to_datetime(jd: f64) -> chrono::DateTime<Utc> {
    use chrono::Duration;
    // J2000 = 2451545.0 = 2000-01-01 12:00:00 UTC
    let days_since_j2000 = jd - 2451545.0;
    let j2000 = Utc.with_ymd_and_hms(2000, 1, 1, 12, 0, 0).unwrap();
    j2000 + Duration::seconds((days_since_j2000 * 86400.0) as i64)
}

/// Compute TSS and F1 from predictions and labels.
fn compute_tss_f1(preds: &[f32], labels: &[f32]) -> (f64, f64) {
    // Sweep thresholds
    let mut best_tss = 0.0f64;
    let mut best_f1 = 0.0f64;
    for thresh_i in 0..20 {
        let thresh = thresh_i as f32 * 0.05;
        let mut tp = 0u32; let mut fp = 0u32;
        let mut tn = 0u32; let mut r#fn = 0u32;
        for (p, l) in preds.iter().zip(labels) {
            let pred_pos = *p >= thresh;
            let true_pos = *l > 0.5;
            match (pred_pos, true_pos) {
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
