//! Training binary for SolarFlareV2 (Clifford lattice dynamics).
//!
//! Usage:
//!   solar-train-v2 --sharp-dir data/sharp_data --flares data/catalogs/solar_flares.csv
//!   solar-train-v2                              # synthetic data demo
//!
//! Key diagnostic: watch J converge toward J_c = 2/π ≈ 0.6366.

use solar_monitor::models::solar_flare_v2::{SolarFlareV2, SolarFlareV2Config, V2Grads, J_CRITICAL, N_FIELDS, N_ORBITAL, N_INPUT};
use rayon::prelude::*;
use solar_monitor::backtest::kp_lookup::KpLookup;
use solar_monitor::backtest::orbital;
use std::sync::Arc;
use solar_monitor::backtest::sharp_dataset::{
    brier_score, brier_skill_score, shuffle, ClassificationMetrics, DatasetConfig, SharpDataset,
};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut sharp_dir: Option<PathBuf> = None;
    let mut sharp_path: Option<PathBuf> = None;
    let mut flare_path: Option<PathBuf> = None;
    let mut checkpoint_path: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("checkpoints/solar-flare-v2");
    let mut epochs = 30;
    let mut batch_size = 64usize;
    let mut seed = 42u64;
    let mut freeze_j = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sharp-dir" => { i += 1; sharp_dir = Some(PathBuf::from(&args[i])); }
            "--sharp" => { i += 1; sharp_path = Some(PathBuf::from(&args[i])); }
            "--flares" => { i += 1; flare_path = Some(PathBuf::from(&args[i])); }
            "--checkpoint" => { i += 1; checkpoint_path = Some(PathBuf::from(&args[i])); }
            "--output" | "-o" => { i += 1; output_dir = PathBuf::from(&args[i]); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap_or(30); }
            "--batch-size" => { i += 1; batch_size = args[i].parse().unwrap_or(64); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap_or(42); }
            "--freeze-j" => { freeze_j = true; }
            "--help" | "-h" => { print_usage(); return; }
            _ => { eprintln!("Unknown: {}", args[i]); print_usage(); std::process::exit(1); }
        }
        i += 1;
    }

    // Load dataset
    let dataset = if let Some(ref sd) = sharp_dir {
        let fp = flare_path.as_deref().unwrap_or_else(|| {
            eprintln!("--flares required with --sharp-dir");
            std::process::exit(1);
        });
        println!("Loading SHARP data from: {}", sd.display());
        println!("Loading flare catalog from: {}", fp.display());
        let config = DatasetConfig { seq_len: 10, seed, ..Default::default() };
        match SharpDataset::load_gzipped_dir(sd, fp, config) {
            Ok(ds) => ds,
            Err(e) => { eprintln!("Failed: {e}"); std::process::exit(1); }
        }
    } else if let (Some(sp), Some(fp)) = (&sharp_path, &flare_path) {
        println!("Loading SHARP CSV: {}", sp.display());
        let config = DatasetConfig { seq_len: 10, seed, ..Default::default() };
        match SharpDataset::load(sp, fp, config) {
            Ok(ds) => ds,
            Err(e) => { eprintln!("Failed: {e}"); std::process::exit(1); }
        }
    } else {
        println!("No data — using synthetic (300 samples)");
        println!("For real training: solar-train-v2 --sharp-dir data/sharp_data --flares data/catalogs/solar_flares.csv");
        SharpDataset::synthetic(10, 300, seed)
    };

    dataset.print_stats();

    // Load Kp index for geomagnetic input
    let kp_path = PathBuf::from("solar-monitor/data/catalogs/kp_3hourly.csv");
    let kp_lookup = if kp_path.exists() {
        match KpLookup::load(&kp_path) {
            Ok(kp) => Some(Arc::new(kp)),
            Err(e) => { eprintln!("Kp load failed: {e}"); None }
        }
    } else {
        println!("  No Kp data found, using defaults");
        None
    };
    println!();

    // Initialize model
    let mut model = if let Some(cp) = &checkpoint_path {
        println!("Resuming from: {}", cp.display());
        SolarFlareV2::load_checkpoint(cp).unwrap_or_else(|e| {
            eprintln!("Failed: {e}"); std::process::exit(1);
        })
    } else {
        let n_pos = dataset.train.iter().filter(|s| s.label > 0.5).count();
        let n_neg = dataset.train.len() - n_pos;
        let pos_weight = if n_pos > 0 { (n_neg as f32 / n_pos as f32).min(50.0) } else { 1.0 };

        let config = SolarFlareV2Config {
            pos_weight,
            ..Default::default()
        };
        SolarFlareV2::new(config)
    };

    if freeze_j {
        model.config.freeze_j_sun = true;
        let (_, w_sum) = model.j_diagnostic();
        println!("FREEZE J_sun at Jw={:.4} (target: 1/π={:.4})", w_sum, 1.0/std::f32::consts::PI);
    }
    println!("Parameters: {}", model.param_count());
    let (j_model, w_sum) = model.j_diagnostic();
    println!("J_model: {:.4} (fixed at J_c), J_sun weights sum: {:.4}", j_model, w_sum);
    println!("Batch size: {batch_size}");
    println!();

    // Pre-training diagnostic
    {
        let (preds, labels) = evaluate(&model, &dataset.test[..dataset.test.len().min(200)], kp_lookup.as_deref());
        let pos_mean = mean_where(&preds, &labels, true);
        let neg_mean = mean_where(&preds, &labels, false);
        println!("PRE-TRAIN: pos_mean={pos_mean:.4} neg_mean={neg_mean:.4} sep={:.4}", pos_mean - neg_mean);
        println!();
    }

    // Training loop
    let mut best_tss = f64::NEG_INFINITY;
    let mut best_epoch = 0;
    let mut train_samples = dataset.train.clone();

    // With analytical BPTT, full dataset is tractable (~0.9s/epoch for 2K samples).
    // Use 8192 samples per epoch for faster iteration with good coverage.
    let samples_per_epoch = 8192usize.min(train_samples.len());

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();
        shuffle(&mut train_samples, seed + epoch as u64);
        let epoch_samples = &train_samples[..samples_per_epoch];

        let mut epoch_loss = 0.0f64;
        let mut n_samples = 0;

        for batch in epoch_samples.chunks(batch_size) {
            // Parallel over samples — analytical BPTT backward doesn't use rayon internally
            let kp_ref = kp_lookup.as_deref();
            let per_sample: Vec<(f64, V2Grads)> = batch
                .par_iter()
                .filter_map(|sample| {
                    let expanded = expand_with_orbital(&sample.features, 10, sample.jd, kp_ref);
                    let (_prob, acts) = model.forward(&expanded);
                    let (loss, grads) = model.backward(&expanded, &acts, sample.label);
                    if loss.is_finite() { Some((loss as f64, grads)) } else { None }
                })
                .collect();

            let bc = per_sample.len();
            if bc > 0 {
                let mut total_loss = 0.0f64;
                let mut grad_acc: Option<V2Grads> = None;
                for (loss, grads) in per_sample {
                    total_loss += loss;
                    match &mut grad_acc {
                        None => grad_acc = Some(grads),
                        Some(a) => add_grads(a, &grads),
                    }
                }
                if let Some(ref mut a) = grad_acc {
                    scale_grads(a, 1.0 / bc as f32);
                    model.sgd_step(a);
                }
                epoch_loss += total_loss;
                n_samples += bc;
            }
        }

        let elapsed = t0.elapsed();
        let avg_loss = epoch_loss / n_samples.max(1) as f64;

        // Evaluate
        let (test_preds, test_labels) = evaluate(&model, &dataset.test, kp_lookup.as_deref());
        let clean_preds: Vec<f32> = test_preds.iter()
            .map(|&p| if p.is_finite() { p } else { 0.5 })
            .collect();
        let (metrics, best_thresh) = ClassificationMetrics::optimal_tss(&clean_preds, &test_labels);
        let bs = brier_score(&clean_preds, &test_labels);

        // J diagnostic
        let (_, w_sum) = model.j_diagnostic();

        println!(
            "Epoch {:3}/{} | loss={:.4} | TSS={:.4}@{:.2} | F1={:.4} | BS={:.4} | Jw={:.3} | {:.1}s",
            epoch + 1, epochs, avg_loss, metrics.tss, best_thresh, metrics.f1, bs,
            w_sum, elapsed.as_secs_f64()
        );

        if metrics.tss > best_tss {
            best_tss = metrics.tss;
            best_epoch = epoch + 1;
            let _ = std::fs::create_dir_all(&output_dir);
            let cp_path = output_dir.join("best_model.json");
            if let Err(e) = model.save_checkpoint(&cp_path) {
                eprintln!("  [!] Save failed: {e}");
            }
        }
    }

    println!();
    println!("=== Training Complete ===");
    println!("Best TSS: {:.4} at epoch {}", best_tss, best_epoch);

    let (_, w_sum_final) = model.j_diagnostic();
    println!("J_model: fixed at J_c = {:.4}", J_CRITICAL);
    println!("J_sun weights sum: {:.4}", w_sum_final);
    println!("J_sun weights: {:?}", model.j_sun_weights);
    println!("J_sun bias: {:.4}", model.j_sun_bias);

    // Final eval
    let (test_preds, test_labels) = evaluate(&model, &dataset.test, kp_lookup.as_deref());
    let final_metrics = ClassificationMetrics::compute(&test_preds, &test_labels, 0.5);
    final_metrics.print("Final Test");
    final_metrics.print_sota_comparison();

    // Save final
    let _ = std::fs::create_dir_all(&output_dir);
    let final_path = output_dir.join("final_model.json");
    match model.save_checkpoint(&final_path) {
        Ok(()) => println!("\nSaved: {}", final_path.display()),
        Err(e) => eprintln!("Save failed: {e}"),
    }
}

fn evaluate(model: &SolarFlareV2, samples: &[solar_monitor::backtest::sharp_dataset::SharpSample], kp: Option<&KpLookup>) -> (Vec<f32>, Vec<f32>) {
    let results: Vec<(f32, f32)> = samples
        .par_iter()
        .map(|s| {
            let expanded = expand_with_orbital(&s.features, 10, s.jd, kp);
            let (p, _) = model.forward(&expanded);
            (p, s.label)
        })
        .collect();
    let mut preds = Vec::with_capacity(results.len());
    let mut labels = Vec::with_capacity(results.len());
    for (p, l) in results { preds.push(p); labels.push(l); }
    (preds, labels)
}

/// Expand a (seq_len × N_FIELDS) feature vector to (seq_len × N_INPUT) by
/// appending orbital angles at each timestep.
///
/// Uses the sample's Julian date for exact orbital angles. The planetary
/// positions barely change over a 2-hour sequence (10 steps × 12 min),
/// so we use the same angles for all timesteps in the sequence.
/// Expand features with orbital angles and geomagnetic data.
/// Input: (seq_len × N_FIELDS) SHARP features
/// Output: (seq_len × N_INPUT) with orbital + Kp appended
fn expand_with_orbital(features: &[f32], seq_len: usize, jd: f64, kp: Option<&KpLookup>) -> Vec<f32> {
    let orb = orbital::orbital_inputs(jd);
    let (kp_norm, dkp_norm) = match kp {
        Some(lookup) => lookup.lookup(jd),
        None => (0.33f32, 0.0f32), // default: moderate, steady
    };

    let mut out = Vec::with_capacity(seq_len * N_INPUT);
    for ti in 0..seq_len {
        // SHARP fields
        for fi in 0..N_FIELDS {
            out.push(features[ti * N_FIELDS + fi]);
        }
        // Orbital angles
        for &o in &orb {
            out.push(o);
        }
        // Geomagnetic: Kp (normalized 0-1) + dKp/dt (normalized)
        out.push(kp_norm);
        out.push(dkp_norm);
    }
    out
}

fn mean_where(preds: &[f32], labels: &[f32], positive: bool) -> f32 {
    let (sum, count) = preds.iter().zip(labels.iter())
        .filter(|(_, &l)| if positive { l > 0.5 } else { l < 0.5 })
        .fold((0.0f32, 0usize), |(s, c), (&p, _)| (s + p, c + 1));
    if count > 0 { sum / count as f32 } else { 0.0 }
}

fn add_grads(dst: &mut V2Grads, src: &V2Grads) {
    fn add(a: &mut [f32], b: &[f32]) { for (x, y) in a.iter_mut().zip(b) { *x += *y; } }
    add(&mut dst.d_encoder_a, &src.d_encoder_a);
    add(&mut dst.d_encoder_b, &src.d_encoder_b);
    add(&mut dst.d_helmholtz_gamma, &src.d_helmholtz_gamma);
    for i in 0..3 { dst.d_omega[i] += src.d_omega[i]; }
    dst.d_gamma += src.d_gamma;
    add(&mut dst.d_j_sun_weights, &src.d_j_sun_weights);
    add(&mut dst.d_j_sun_orbital, &src.d_j_sun_orbital);
    add(&mut dst.d_j_sun_geomag, &src.d_j_sun_geomag);
    dst.d_j_sun_bias += src.d_j_sun_bias;
    add(&mut dst.d_coupling_matrix, &src.d_coupling_matrix);
    dst.d_thomson_beta += src.d_thomson_beta;
    add(&mut dst.d_head_w1, &src.d_head_w1);
    add(&mut dst.d_head_b1, &src.d_head_b1);
    add(&mut dst.d_head_w2, &src.d_head_w2);
    dst.d_head_b2 += src.d_head_b2;
}

fn scale_grads(g: &mut V2Grads, s: f32) {
    fn sc(v: &mut [f32], s: f32) { for x in v.iter_mut() { *x *= s; } }
    sc(&mut g.d_encoder_a, s);
    sc(&mut g.d_encoder_b, s);
    sc(&mut g.d_helmholtz_gamma, s);
    for x in g.d_omega.iter_mut() { *x *= s; }
    g.d_gamma *= s;
    sc(&mut g.d_j_sun_weights, s);
    sc(&mut g.d_j_sun_orbital, s);
    sc(&mut g.d_j_sun_geomag, s);
    g.d_j_sun_bias *= s;
    sc(&mut g.d_coupling_matrix, s);
    g.d_thomson_beta *= s;
    sc(&mut g.d_head_w1, s);
    sc(&mut g.d_head_b1, s);
    sc(&mut g.d_head_w2, s);
    g.d_head_b2 *= s;
}

fn print_usage() {
    println!("solar-train-v2 — Train SolarFlareV2 (Clifford lattice dynamics)");
    println!();
    println!("USAGE:");
    println!("  solar-train-v2 --sharp-dir data/sharp_data --flares data/catalogs/solar_flares.csv");
    println!("  solar-train-v2                             # synthetic demo");
    println!();
    println!("OPTIONS:");
    println!("  --sharp-dir <dir>    Directory of .csv.gz SHARP files");
    println!("  --sharp <csv>        Single SHARP CSV (uncompressed)");
    println!("  --flares <csv>       GOES flare catalog CSV");
    println!("  --checkpoint <json>  Resume from saved model");
    println!("  --output, -o <dir>   Output directory (default: checkpoints/solar-flare-v2)");
    println!("  --epochs <n>         Training epochs (default: 30)");
    println!("  --batch-size <n>     Batch size (default: 64)");
    println!("  --seed <n>           Random seed (default: 42)");
}
