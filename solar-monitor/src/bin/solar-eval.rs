//! Evaluation binary for SolarFlareModel.
//!
//! Usage:
//!   solar-eval --checkpoint checkpoints/solar-flare/best_model.json
//!   solar-eval                    # Evaluate on synthetic data (demo)
//!
//! Evaluates on held-out test set and prints:
//! - TSS, BACC, F1, Precision, Recall
//! - Brier Score and Brier Skill Score
//! - Comparison table vs SolarFlareNet and Doria Rosales
//! - Threshold sweep for optimal TSS

use solar_monitor::models::solar_flare::SolarFlareModel;
use solar_monitor::backtest::sharp_dataset::{
    brier_score, brier_skill_score, ClassificationMetrics, DatasetConfig, SharpDataset,
};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut checkpoint_path: Option<PathBuf> = None;
    let mut sharp_path: Option<PathBuf> = None;
    let mut flare_path: Option<PathBuf> = None;
    let mut seq_len = 10;
    let mut seed = 42u64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" | "-c" => {
                i += 1;
                checkpoint_path = Some(PathBuf::from(&args[i]));
            }
            "--sharp" => {
                i += 1;
                sharp_path = Some(PathBuf::from(&args[i]));
            }
            "--flares" => {
                i += 1;
                flare_path = Some(PathBuf::from(&args[i]));
            }
            "--seq-len" => {
                i += 1;
                seq_len = args[i].parse().unwrap_or(10);
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(42);
            }
            "--help" | "-h" => {
                println!("solar-eval — Evaluate SolarFlareModel on held-out test set");
                println!();
                println!("USAGE:");
                println!("  solar-eval                              # Synthetic demo");
                println!("  solar-eval -c model.json                # From checkpoint");
                println!("  solar-eval -c model.json --sharp <csv> --flares <csv>");
                println!();
                println!("OPTIONS:");
                println!("  --checkpoint, -c <path>   Model checkpoint JSON");
                println!("  --sharp <path>            SHARP parameters CSV");
                println!("  --flares <path>           GOES flare catalog CSV");
                println!("  --seq-len <n>             Sequence length (default: 10)");
                println!("  --seed <n>                Random seed (default: 42)");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Load model
    let model = if let Some(cp) = &checkpoint_path {
        println!("Loading model from: {}", cp.display());
        match SolarFlareModel::load_checkpoint(cp) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load model: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("No checkpoint specified — using freshly initialized model");
        solar_monitor::models::solar_flare::SolarFlareModel::new(Default::default())
    };

    println!("Parameters: {}", model.param_count());
    println!();

    // Load dataset
    let dataset = if let (Some(sp), Some(fp)) = (&sharp_path, &flare_path) {
        let config = DatasetConfig {
            seq_len: model.config.seq_len,
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
        println!("No data files — using synthetic dataset");
        SharpDataset::synthetic(model.config.seq_len, 300, seed)
    };

    dataset.print_stats();
    println!();

    // Collect predictions on test set
    let mut preds = Vec::with_capacity(dataset.test.len());
    let mut labels = Vec::with_capacity(dataset.test.len());
    for sample in &dataset.test {
        let (prob, _) = model.forward(&sample.features);
        preds.push(prob);
        labels.push(sample.label);
    }

    // Metrics at default threshold (0.5)
    let metrics = ClassificationMetrics::compute(&preds, &labels, 0.5);
    metrics.print("Test Set (threshold=0.5)");
    println!();

    // Brier scores
    let bs = brier_score(&preds, &labels);
    let bss = brier_skill_score(&preds, &labels);
    println!("Brier Score:       {:.6}", bs);
    println!("Brier Skill Score: {:.6}", bss);
    println!();

    // Threshold sweep for optimal TSS
    println!("=== Threshold Sweep ===");
    println!("  Thresh |  TSS   |  BACC  | Prec   | Recall | F1");
    println!("  -------|--------|--------|--------|--------|------");
    let mut best_tss = f64::NEG_INFINITY;
    let mut best_thresh = 0.5;
    for t_idx in 1..20 {
        let thresh = t_idx as f32 * 0.05;
        let m = ClassificationMetrics::compute(&preds, &labels, thresh);
        if m.tss > best_tss {
            best_tss = m.tss;
            best_thresh = thresh as f64;
        }
        println!(
            "  {:.2}   | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            thresh, m.tss, m.bacc, m.precision, m.recall, m.f1
        );
    }
    println!();
    println!("Best TSS: {:.4} at threshold {:.2}", best_tss, best_thresh);
    println!();

    // SOTA comparison
    let best_metrics = ClassificationMetrics::compute(&preds, &labels, best_thresh as f32);
    best_metrics.print_sota_comparison();

    // Prediction distribution
    let pos_preds: Vec<f32> = preds
        .iter()
        .zip(labels.iter())
        .filter(|(_, &l)| l > 0.5)
        .map(|(&p, _)| p)
        .collect();
    let neg_preds: Vec<f32> = preds
        .iter()
        .zip(labels.iter())
        .filter(|(_, &l)| l <= 0.5)
        .map(|(&p, _)| p)
        .collect();

    if !pos_preds.is_empty() && !neg_preds.is_empty() {
        let pos_mean = pos_preds.iter().sum::<f32>() / pos_preds.len() as f32;
        let neg_mean = neg_preds.iter().sum::<f32>() / neg_preds.len() as f32;
        let pos_min = pos_preds.iter().cloned().fold(f32::INFINITY, f32::min);
        let pos_max = pos_preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let neg_min = neg_preds.iter().cloned().fold(f32::INFINITY, f32::min);
        let neg_max = neg_preds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!();
        println!("=== Prediction Distribution ===");
        println!(
            "Positive (n={}): mean={:.4}, min={:.4}, max={:.4}",
            pos_preds.len(),
            pos_mean,
            pos_min,
            pos_max
        );
        println!(
            "Negative (n={}): mean={:.4}, min={:.4}, max={:.4}",
            neg_preds.len(),
            neg_mean,
            neg_min,
            neg_max
        );
        println!("Separation: {:.4}", pos_mean - neg_mean);
    }
}
