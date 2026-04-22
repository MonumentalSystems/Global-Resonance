//! Training binary for SolarFlareModel on SHARP time series.
//!
//! Usage:
//!   solar-train                        # Train on synthetic data (demo)
//!   solar-train --sharp data/sharp.csv --flares data/goes_flares.csv
//!   solar-train --checkpoint out/model  # Resume from checkpoint
//!
//! Trains the manifold-native solar flare predictor using:
//! - ManifoldEncoder → HelmholtzTemporal → LoheSync → CliffordReadout → ClassificationHead
//! - Geodesic SGD on S^{d-1} for encoder weights
//! - Standard SGD for scalar params (classifier, coupling, gamma)
//! - Binary cross-entropy loss with sigmoid output
//!
//! Evaluates TSS, BACC, F1, Brier Score on held-out test set each epoch.

use crate::models::solar_flare::{SolarFlareConfig, SolarFlareGrads, SolarFlareModel};
use rayon::prelude::*;
use solar_monitor::backtest::sharp_dataset::{
    brier_score, brier_skill_score, shuffle, ClassificationMetrics, DatasetConfig, SharpDataset,
};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse CLI args
    let mut sharp_path: Option<PathBuf> = None;
    let mut flare_path: Option<PathBuf> = None;
    let mut checkpoint_path: Option<PathBuf> = None;
    let mut output_dir = PathBuf::from("checkpoints/solar-flare");
    let mut epochs = 50;
    let mut seq_len = 10;
    let mut lr_manifold = 0.01f32;
    let mut lr_scalar = 0.01f32;
    let mut batch_size = 256usize;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sharp" => {
                i += 1;
                sharp_path = Some(PathBuf::from(&args[i]));
            }
            "--flares" => {
                i += 1;
                flare_path = Some(PathBuf::from(&args[i]));
            }
            "--checkpoint" => {
                i += 1;
                checkpoint_path = Some(PathBuf::from(&args[i]));
            }
            "--output" | "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().unwrap_or(50);
            }
            "--seq-len" => {
                i += 1;
                seq_len = args[i].parse().unwrap_or(10);
            }
            "--lr-manifold" => {
                i += 1;
                lr_manifold = args[i].parse().unwrap_or(0.01);
            }
            "--lr-scalar" => {
                i += 1;
                lr_scalar = args[i].parse().unwrap_or(0.001);
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().unwrap_or(32);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(42);
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load or create dataset
    let dataset = if let (Some(sp), Some(fp)) = (&sharp_path, &flare_path) {
        println!("Loading SHARP data from: {}", sp.display());
        println!("Loading flare catalog from: {}", fp.display());
        let config = DatasetConfig {
            seq_len,
            seed,
            ..Default::default()
        };
        match SharpDataset::load(sp, fp, config) {
            Ok(ds) => ds,
            Err(e) => {
                eprintln!("Failed to load dataset: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("No data files specified — using synthetic dataset (300 samples)");
        println!("For real training: solar-train --sharp <path> --flares <path>");
        SharpDataset::synthetic(seq_len, 300, seed)
    };

    dataset.print_stats();
    println!();

    // Initialize or load model
    let mut model = if let Some(cp) = &checkpoint_path {
        println!("Resuming from checkpoint: {}", cp.display());
        match SolarFlareModel::load_checkpoint(cp) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load checkpoint: {e}");
                std::process::exit(1);
            }
        }
    } else {
        // Compute class imbalance weight from training data
        let n_pos = dataset.train.iter().filter(|s| s.label > 0.5).count();
        let n_neg = dataset.train.len() - n_pos;
        let pos_weight = if n_pos > 0 {
            n_neg as f32 / n_pos as f32
        } else {
            1.0
        };
        // Cap pos_weight to avoid extreme gradients on rare events
        let pos_weight = pos_weight.min(50.0);
        let pos_rate = n_pos as f32 / dataset.train.len() as f32;

        let config = SolarFlareConfig {
            seq_len,
            lr_manifold,
            lr_scalar,
            pos_weight,
            ..Default::default()
        };
        println!(
            "Model config: d_osc={}, seq_len={}, hidden={}, sync_steps={}",
            config.d_osc, config.seq_len, config.hidden_dim, config.sync_steps
        );
        println!(
            "LR: manifold={}, scalar={}, pos_weight={:.1} (pos_rate={:.3})",
            config.lr_manifold, config.lr_scalar, pos_weight, pos_rate
        );
        let mut m = SolarFlareModel::new(config);
        // Set output bias to true class prior logit
        if let Some(ref mut bias) = m.head_w2.bias {
            bias[0] = (pos_rate / (1.0 - pos_rate)).ln();
        }
        m
    };

    let param_count = model.param_count();
    println!("Parameters: {param_count}");
    println!();

    // Training loop with mini-batch gradient accumulation
    let mut best_tss = f64::NEG_INFINITY;
    let mut best_epoch = 0;
    let mut train_samples = dataset.train.clone();

    println!("Batch size: {batch_size}");
    println!();

    // --- Diagnostic: trace forward pass on 3 samples before training ---
    {
        for (i, sample) in dataset.test.iter().take(3).enumerate() {
            let (prob, acts) = model.forward(&sample.features);
            let input = &sample.features;
            let i_min = input.iter().cloned().fold(f32::INFINITY, f32::min);
            let i_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let r_min = acts.readout.iter().cloned().fold(f32::INFINITY, f32::min);
            let r_max = acts
                .readout
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let r_nan = acts.readout.iter().filter(|x| !x.is_finite()).count();
            let h_min = acts.hidden.iter().cloned().fold(f32::INFINITY, f32::min);
            let h_max = acts
                .hidden
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let (loss, grads) = model.backward(&acts, sample.label);
            let g_head: f32 = grads
                .d_head_w2_weight
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            let g_enc: f32 = grads.d_encoder_u.iter().map(|g| g * g).sum::<f32>().sqrt();
            let g_helm: f32 = grads
                .d_helmholtz_gamma
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            // Readout variance across features
            let r_var: f32 = {
                let mean = acts.readout.iter().sum::<f32>() / acts.readout.len() as f32;
                acts.readout
                    .iter()
                    .map(|x| (x - mean) * (x - mean))
                    .sum::<f32>()
                    / acts.readout.len() as f32
            };
            // Check encoded vs synced: how much did Lohe sync change?
            let sync_diff: f32 = acts
                .encoded
                .iter()
                .zip(acts.synced.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / acts.encoded.len() as f32;
            let g_coup: f32 = grads
                .d_coupling_matrix
                .iter()
                .map(|g| g * g)
                .sum::<f32>()
                .sqrt();
            println!("DIAG[{i}] label={:.0} input=[{i_min:.3}..{i_max:.3}] readout=[{r_min:.4}..{r_max:.4}] r_var={r_var:.6} sync_diff={sync_diff:.6} logit={:.4} prob={prob:.4} loss={loss:.4} |g_head|={g_head:.6} |g_enc|={g_enc:.6} |g_coup|={g_coup:.6}", sample.label, acts.logit);
        }
        // Check separation on 500 test samples
        let (preds, labels) = evaluate(&model, &dataset.test[..500.min(dataset.test.len())]);
        let pos_mean: f32 = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &l)| l > 0.5)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / preds
                .iter()
                .zip(labels.iter())
                .filter(|(_, &l)| l > 0.5)
                .count()
                .max(1) as f32;
        let neg_mean: f32 = preds
            .iter()
            .zip(labels.iter())
            .filter(|(_, &l)| l < 0.5)
            .map(|(&p, _)| p)
            .sum::<f32>()
            / preds
                .iter()
                .zip(labels.iter())
                .filter(|(_, &l)| l < 0.5)
                .count()
                .max(1) as f32;
        println!(
            "DIAG: pos_mean={pos_mean:.6} neg_mean={neg_mean:.6} sep={:.6}",
            pos_mean - neg_mean
        );
        println!();
    }

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();

        // Shuffle training data each epoch
        shuffle(&mut train_samples, seed + epoch as u64);

        let mut epoch_loss = 0.0f64;
        let mut n_samples = 0;

        // Mini-batch training: parallel forward+backward, reduce gradients, then step
        for batch in train_samples.chunks(batch_size) {
            // Parallel map: each sample computes forward+backward independently.
            // Model is read-only during forward+backward (weights only mutate in sgd_step).
            let per_sample: Vec<(f64, SolarFlareGrads)> = batch
                .par_iter()
                .filter_map(|sample| {
                    let (_prob, acts) = model.forward(&sample.features);
                    let (loss, grads) = model.backward(&acts, sample.label);
                    if loss.is_nan() || !loss.is_finite() {
                        None
                    } else {
                        Some((loss as f64, grads))
                    }
                })
                .collect();

            let bc = per_sample.len();
            if bc > 0 {
                let mut total_loss = 0.0f64;
                let mut grad_acc: Option<SolarFlareGrads> = None;
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

        // Evaluate on test set
        let (test_preds, test_labels) = evaluate(&model, &dataset.test);

        // Filter NaN predictions (diagnostic: count them)
        let nan_count = test_preds.iter().filter(|p| p.is_nan()).count();
        if nan_count > 0 && epoch == 0 {
            eprintln!(
                "  [diag] {nan_count}/{} test predictions are NaN",
                test_preds.len()
            );
        }
        // Replace NaN with 0.5 for metric computation
        let clean_preds: Vec<f32> = test_preds
            .iter()
            .map(|&p| if p.is_finite() { p } else { 0.5 })
            .collect();

        let (metrics, best_thresh) = ClassificationMetrics::optimal_tss(&clean_preds, &test_labels);
        let bs = brier_score(&clean_preds, &test_labels);
        let bss = brier_skill_score(&clean_preds, &test_labels);

        // Prediction spread diagnostic
        let pred_min = clean_preds.iter().cloned().fold(f32::INFINITY, f32::min);
        let pred_max = clean_preds
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Sync order (diagnostic — mean coupling strength relative to J_c)
        let mean_coupling: f32 = model.coupling_matrix.iter().map(|v| v.abs()).sum::<f32>()
            / model.coupling_matrix.len() as f32;

        println!(
            "Epoch {:3}/{} | loss={:.4} | TSS={:.4} @{:.2} | BACC={:.4} | F1={:.4} | BS={:.4} | BSS={:.4} | |K|={:.4} | pred=[{:.3}..{:.3}] | {:.1}s",
            epoch + 1, epochs, avg_loss, metrics.tss, best_thresh, metrics.bacc, metrics.f1, bs, bss, mean_coupling, pred_min, pred_max, elapsed.as_secs_f64()
        );

        // Save best model
        if metrics.tss > best_tss {
            best_tss = metrics.tss;
            best_epoch = epoch + 1;
            let _ = std::fs::create_dir_all(&output_dir);
            let cp_path = output_dir.join("best_model.json");
            if let Err(e) = model.save_checkpoint(&cp_path) {
                eprintln!("  [!] Failed to save checkpoint: {e}");
            }
        }
    }

    println!();
    println!("=== Training Complete ===");
    println!("Best TSS: {:.4} at epoch {}", best_tss, best_epoch);

    // Final evaluation
    let (test_preds, test_labels) = evaluate(&model, &dataset.test);
    let final_metrics = ClassificationMetrics::compute(&test_preds, &test_labels, 0.5);
    final_metrics.print("Final Test Metrics");
    final_metrics.print_sota_comparison();

    // Save final model
    let _ = std::fs::create_dir_all(&output_dir);
    let final_path = output_dir.join("final_model.json");
    match model.save_checkpoint(&final_path) {
        Ok(()) => println!("\nFinal model saved to: {}", final_path.display()),
        Err(e) => eprintln!("Failed to save final model: {e}"),
    }

    // Also save norm stats for inference
    let norm_path = output_dir.join("norm_stats.json");
    if let Ok(json) = serde_json::to_string_pretty(&dataset.norm) {
        let _ = std::fs::write(&norm_path, json);
        println!("Norm stats saved to: {}", norm_path.display());
    }
}

/// Run model on test samples, collect predictions and labels.
fn evaluate(
    model: &SolarFlareModel,
    samples: &[solar_monitor::backtest::sharp_dataset::SharpSample],
) -> (Vec<f32>, Vec<f32>) {
    let results: Vec<(f32, f32)> = samples
        .par_iter()
        .map(|sample| {
            let (prob, _) = model.forward(&sample.features);
            (prob, sample.label)
        })
        .collect();
    let mut preds = Vec::with_capacity(results.len());
    let mut labels = Vec::with_capacity(results.len());
    for (p, l) in results {
        preds.push(p);
        labels.push(l);
    }
    (preds, labels)
}

/// Element-wise add `src` gradients into `dst`.
fn add_grads(
    dst: &mut crate::models::solar_flare::SolarFlareGrads,
    src: &crate::models::solar_flare::SolarFlareGrads,
) {
    fn add_vecs(a: &mut [f32], b: &[f32]) {
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x += *y;
        }
    }
    add_vecs(&mut dst.d_encoder_u, &src.d_encoder_u);
    add_vecs(&mut dst.d_encoder_v, &src.d_encoder_v);
    add_vecs(&mut dst.d_helmholtz_gamma, &src.d_helmholtz_gamma);
    add_vecs(&mut dst.d_coupling_matrix, &src.d_coupling_matrix);
    add_vecs(&mut dst.d_head_w1_weight, &src.d_head_w1_weight);
    add_vecs(&mut dst.d_head_w1_bias, &src.d_head_w1_bias);
    add_vecs(&mut dst.d_head_w2_weight, &src.d_head_w2_weight);
    add_vecs(&mut dst.d_head_w2_bias, &src.d_head_w2_bias);
}

/// Scale all gradient components by `s` (for averaging over batch).
fn scale_grads(g: &mut crate::models::solar_flare::SolarFlareGrads, s: f32) {
    fn scale_vec(v: &mut [f32], s: f32) {
        for x in v.iter_mut() {
            *x *= s;
        }
    }
    scale_vec(&mut g.d_encoder_u, s);
    scale_vec(&mut g.d_encoder_v, s);
    scale_vec(&mut g.d_helmholtz_gamma, s);
    scale_vec(&mut g.d_coupling_matrix, s);
    scale_vec(&mut g.d_head_w1_weight, s);
    scale_vec(&mut g.d_head_w1_bias, s);
    scale_vec(&mut g.d_head_w2_weight, s);
    scale_vec(&mut g.d_head_w2_bias, s);
}

fn print_usage() {
    println!("solar-train — Train SolarFlareModel on SHARP magnetogram time series");
    println!();
    println!("USAGE:");
    println!("  solar-train                                    # Synthetic data demo");
    println!("  solar-train --sharp <csv> --flares <csv>       # Real data");
    println!("  solar-train --checkpoint <dir>/model.json      # Resume");
    println!();
    println!("OPTIONS:");
    println!("  --sharp <path>        SHARP parameters CSV (JSOC export)");
    println!("  --flares <path>       GOES flare catalog CSV");
    println!("  --checkpoint <path>   Resume from saved model JSON");
    println!("  --output, -o <dir>    Output directory (default: checkpoints/solar-flare)");
    println!("  --epochs <n>          Number of training epochs (default: 50)");
    println!("  --seq-len <n>         Input sequence length (default: 10)");
    println!("  --lr-manifold <f>     Manifold param learning rate (default: 0.01)");
    println!("  --lr-scalar <f>       Scalar param learning rate (default: 0.001)");
    println!("  --batch-size <n>      Mini-batch size (default: 32)");
    println!("  --seed <n>            Random seed (default: 42)");
}
