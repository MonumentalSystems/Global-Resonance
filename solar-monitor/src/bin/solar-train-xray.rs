//! Train criticality detector on actual GOES X-ray time series.
//!
//! Replays 1-minute GOES-16 XRS data chronologically through the
//! criticality detector's native ingest() method. This is the
//! detector's designed input — not SHARP proxies.
//!
//! Labels come from the flare catalog: if a flare starts within
//! the prediction window, the current timestep is positive.
//!
//! Learns: 5 sub-score weights + 82 PlanetaryKAN params = 87 total
//!
//! Usage:
//!   solar-train-xray --goes-dir solar-monitor/data/goes_data \
//!                    --flares solar-monitor/data/catalogs/solar_flares.csv

use chrono::{Datelike, NaiveDateTime, Utc, TimeZone, Duration};
use solar_monitor::detection::criticality::CriticalityDetector;
use solar_monitor::detection::planetary_kan::{PlanetaryKAN, N_BODIES, date_to_jd};
use std::path::{Path, PathBuf};

const PREDICTION_WINDOW_MIN: i64 = 120;
const LR_KAN: f32 = 0.05;
const LR_WEIGHTS: f64 = 0.005;
/// Subsample: evaluate every N minutes (1 = every minute, 10 = every 10 min)
const EVAL_STRIDE: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut goes_dir = PathBuf::from("solar-monitor/data/goes_data");
    let mut flare_path = PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv");
    let mut epochs = 3usize; // full passes through all years

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--goes-dir" => { i += 1; goes_dir = PathBuf::from(&args[i]); }
            "--flares" => { i += 1; flare_path = PathBuf::from(&args[i]); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap_or(3); }
            _ => {}
        }
        i += 1;
    }

    println!("={:=<60}", "");
    println!("  CRITICALITY TRAINING ON GOES X-RAY (native input)");
    println!("={:=<60}", "");

    // Load flare catalog
    let flares = load_flare_datetimes(&flare_path);
    println!("Flare catalog: {} events", flares.len());

    // Find GOES CSV files
    let mut goes_files: Vec<PathBuf> = std::fs::read_dir(&goes_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "csv"))
        .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with("goes16"))
        .collect();
    goes_files.sort();
    println!("GOES files: {}", goes_files.len());
    for f in &goes_files {
        println!("  {}", f.file_name().unwrap().to_str().unwrap());
    }

    let mut weights = [0.25f64, 0.20, 0.25, 0.20, 0.10];
    let mut kan = PlanetaryKAN::new(8);
    let mut best_tss = 0.0f64;

    println!("\n{:>6} | {:>8} {:>8} {:>8} | {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>8}",
        "Epoch", "Loss", "TSS", "F1", "w_bal", "w_dis", "w_com", "w_lod", "w_syn", "KAN_E", "samples");
    println!("{}", "-".repeat(100));

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();
        let mut detector = CriticalityDetector::new(0.5);
        detector.set_score_weights(weights);

        let mut total_loss = 0.0f64;
        let mut n_eval = 0usize;
        let mut preds = Vec::new();
        let mut labels = Vec::new();
        let mut d_weights = [0.0f64; 5];
        let mut kan_grad_acc: Option<super_kan_grads> = None;

        // Replay each year chronologically
        for goes_file in &goes_files {
            let mut rdr = match csv::Reader::from_path(goes_file) {
                Ok(r) => r,
                Err(_) => continue,
            };

            let mut step = 0usize;
            for result in rdr.records() {
                let record = match result {
                    Ok(r) => r,
                    Err(_) => continue,
                };

                // Parse: time_tag, xrsa_flux, xrsb_flux
                let time_str = match record.get(0) { Some(s) => s, None => continue };
                let xrsa: f64 = record.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let xrsb: f64 = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);

                if xrsb < 1e-9 { step += 1; continue; } // eclipse

                let ts = match NaiveDateTime::parse_from_str(time_str, "%Y-%m-%d %H:%M:%S") {
                    Ok(dt) => Utc.from_utc_datetime(&dt),
                    Err(_) => { step += 1; continue; }
                };

                // Feed to detector (native ingest — X-ray + hardness)
                let proton = 0.1; // default quiet proton flux
                detector.ingest(xrsb, xrsa, proton, ts);

                // Evaluate every EVAL_STRIDE steps
                step += 1;
                if step % EVAL_STRIDE != 0 { continue; }

                // Label: flare within prediction window?
                let window_end = ts + Duration::minutes(PREDICTION_WINDOW_MIN);
                let label = flares.iter().any(|&ft| ft > ts && ft <= window_end);
                let label_f = if label { 1.0f64 } else { 0.0 };

                // Scores
                let base_score = detector.raw_physics_score();
                let jd = date_to_jd(ts.year(), ts.month(), ts.day());
                let angles = PlanetaryKAN::angles_from_jd(jd);
                let kan_mod = kan.forward(&angles) as f64;
                let final_score = (base_score * kan_mod).clamp(0.0, 1.0);

                // BCE loss
                let p = final_score.clamp(1e-7, 1.0 - 1e-7);
                let loss = -(label_f * p.ln() + (1.0 - label_f) * (1.0 - p).ln());
                total_loss += loss;

                // KAN gradient
                let d_final = -label_f / p + (1.0 - label_f) / (1.0 - p);
                let d_kan = d_final * base_score;
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

                // Weight SPSA gradient
                {
                    let eps = 0.02;
                    let hash = ((n_eval as u64).wrapping_mul(6364136223846793005).wrapping_add(epoch as u64)) as u32;
                    let perturbation: Vec<f64> = (0..5).map(|i| {
                        if (hash >> i) & 1 == 0 { eps } else { -eps }
                    }).collect();
                    let w_sum: f64 = weights.iter().sum();
                    let p_sum: f64 = perturbation.iter().sum();
                    let ps = (base_score * (1.0 + p_sum / w_sum.max(0.01)) * kan_mod).clamp(0.0, 1.0);
                    let pp = ps.clamp(1e-7, 1.0 - 1e-7);
                    let lp = -(label_f * pp.ln() + (1.0 - label_f) * (1.0 - pp).ln());
                    for i in 0..5 { d_weights[i] += (lp - loss) / perturbation[i]; }
                }

                preds.push(final_score as f32);
                labels.push(if label { 1.0f32 } else { 0.0 });
                n_eval += 1;
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
        let (tss, f1) = compute_tss_f1(&preds, &labels);
        let kan_energy: f32 = kan.splines.iter()
            .map(|s| s.coeffs.iter().map(|c| c*c).sum::<f32>()).sum::<f32>().sqrt();

        if tss > best_tss { best_tss = tss; }

        println!("{:6} | {:8.4} {:8.4} {:8.4} | {:6.3} {:6.3} {:6.3} {:6.3} {:6.3} | {:6.3} {:>8}",
            epoch + 1, avg_loss, tss, f1,
            weights[0], weights[1], weights[2], weights[3], weights[4],
            kan_energy, n_eval);

        println!("       ({:.1}s)", t0.elapsed().as_secs_f64());
    }

    println!("\nBest TSS: {:.4}", best_tss);
    kan.print_weights();
}

// Type alias for KAN grads to avoid full path
use solar_monitor::detection::planetary_kan::PlanetaryKANGrads as super_kan_grads;

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
