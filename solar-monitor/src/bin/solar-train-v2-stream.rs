//! Streaming V2 training: full AR time series through SolarFlareV2.step().
//!
//! Unlike window-based training, the Helmholtz accumulators and lattice states
//! evolve continuously across each AR's lifetime (~days to weeks). This gives
//! the model temporal context far beyond the 10-step (2-hour) window.
//!
//! Gradient computation: SPSA (simultaneous perturbation) on model params.
//! Each AR's full stream produces one loss; SPSA estimates the gradient with
//! 2 perturbed forward passes per AR.
//!
//! Usage:
//!   solar-train-v2-stream --sharp-dir solar-monitor/data/sharp_data \
//!                         --flares solar-monitor/data/catalogs/solar_flares.csv \
//!                         --epochs 100 --patience 20

use solar_monitor::models::solar_flare_v2::{
    SolarFlareV2, SolarFlareV2Config, V2StreamState,
    J_CRITICAL, N_FIELDS, N_ORBITAL, N_INPUT,
};
use solar_monitor::backtest::kp_lookup::KpLookup;
use solar_monitor::backtest::orbital;
use solar_monitor::backtest::sharp_dataset::{ArDataset, ArTimeSeries, DatasetConfig};
use serde_json;
use std::path::PathBuf;

const LR: f32 = 0.002;
const SPSA_EPS: f32 = 0.01;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut sharp_dir = PathBuf::from("solar-monitor/data/sharp_data");
    let mut flare_path = PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv");
    let mut kp_path = PathBuf::from("solar-monitor/data/catalogs/kp_3hourly.csv");
    let mut epochs = 100usize;
    let mut patience_cfg = 20usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sharp-dir" => { i += 1; sharp_dir = PathBuf::from(&args[i]); }
            "--flares" => { i += 1; flare_path = PathBuf::from(&args[i]); }
            "--kp" => { i += 1; kp_path = PathBuf::from(&args[i]); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap_or(100); }
            "--patience" => { i += 1; patience_cfg = args[i].parse().unwrap_or(20); }
            _ => {}
        }
        i += 1;
    }

    println!("={:=<60}", "");
    println!("  STREAMING V2 TRAINING (full AR time series)");
    println!("={:=<60}", "");

    // Load data
    let config = DatasetConfig { seq_len: 10, ..Default::default() };
    let dataset = match ArDataset::load_gzipped_dir(&sharp_dir, &flare_path, &config) {
        Ok(ds) => ds,
        Err(e) => { eprintln!("Failed: {e}"); std::process::exit(1); }
    };

    let kp_lookup = if kp_path.exists() {
        KpLookup::load(&kp_path).ok()
    } else {
        None
    };

    // Initialize model
    let model_config = SolarFlareV2Config {
        seq_len: 10, // used by window-based forward; streaming ignores this
        pos_weight: 5.0,
        ..Default::default()
    };
    let model = SolarFlareV2::new(model_config);
    let n_params = model.param_count();
    println!("  Model params: {}", n_params);
    println!("  Train ARs: {}, Test ARs: {}", dataset.train_ars.len(), dataset.test_ars.len());
    println!("  Epochs: {}, patience: {}", epochs, patience_cfg);

    // Flatten model params for SPSA
    let mut params = model_to_params(&model);
    let mut best_tss = f64::NEG_INFINITY;
    let mut best_params = params.clone();
    let mut patience_counter = 0usize;

    println!("\n{:>6} | {:>8} {:>8} {:>8} | {:>8} {:>8} | {:>6}",
        "Epoch", "Loss", "TSS", "F1", "pos_mean", "neg_mean", "secs");
    println!("{}", "-".repeat(75));

    for epoch in 0..epochs {
        let t0 = std::time::Instant::now();

        // SPSA gradient estimation: perturb all params, measure loss change
        let perturbation = spsa_perturbation(params.len(), epoch as u64);
        let mut params_plus = params.clone();
        let mut params_minus = params.clone();
        for i in 0..params.len() {
            params_plus[i] += SPSA_EPS * perturbation[i];
            params_minus[i] -= SPSA_EPS * perturbation[i];
        }

        let model_plus = params_to_model(&params_plus, &model);
        let model_minus = params_to_model(&params_minus, &model);

        // Subsample training ARs for SPSA gradient (speed)
        let ar_subset: Vec<&ArTimeSeries> = dataset.train_ars.iter()
            .enumerate()
            .filter(|(i, _)| {
                let h = (*i as u64).wrapping_mul(2654435761).wrapping_add(epoch as u64);
                (h % 5) == 0 // ~20% of ARs per epoch
            })
            .map(|(_, ar)| ar)
            .collect();
        let train_slice: Vec<ArTimeSeries> = ar_subset.iter().map(|a| (*a).clone()).collect();

        let loss_plus = eval_streaming_loss(&model_plus, &train_slice, kp_lookup.as_ref());
        let loss_minus = eval_streaming_loss(&model_minus, &train_slice, kp_lookup.as_ref());

        // SPSA gradient: g_i = (L+ - L-) / (2 * eps * delta_i)
        let d_loss = loss_plus - loss_minus;
        if d_loss.is_finite() {
            for i in 0..params.len() {
                let grad = d_loss / (2.0 * SPSA_EPS as f64 * perturbation[i] as f64);
                params[i] -= LR * grad.clamp(-1.0, 1.0) as f32;
            }
        }

        // Evaluate on test set
        let model_current = params_to_model(&params, &model);
        let (test_loss, preds, labels) = eval_streaming_detailed(
            &model_current, &dataset.test_ars, kp_lookup.as_ref(),
        );
        let (tss, f1) = compute_tss_f1(&preds, &labels);

        let pos_mean = mean_where(&preds, &labels, true);
        let neg_mean = mean_where(&preds, &labels, false);
        let elapsed = t0.elapsed().as_secs_f64();

        print!("{:6} | {:8.4} {:8.4} {:8.4} | {:8.4} {:8.4} | {:6.1}",
            epoch + 1, test_loss, tss, f1, pos_mean, neg_mean, elapsed);

        if tss > best_tss {
            best_tss = tss;
            best_params = params.clone();
            patience_counter = 0;
            print!(" *");
        } else {
            patience_counter += 1;
            if patience_counter >= patience_cfg {
                println!("\nEarly stop at epoch {} (no improvement for {} epochs)", epoch + 1, patience_cfg);
                break;
            }
        }
        println!();
    }

    println!("\nBest TSS: {:.4}", best_tss);
}

// ── Model <-> param vector conversion ──

fn model_to_params(model: &SolarFlareV2) -> Vec<f32> {
    let mut p = Vec::new();
    for a in &model.encoder_a { p.extend_from_slice(a); }
    for b in &model.encoder_b { p.extend_from_slice(b); }
    p.extend_from_slice(&model.helmholtz_gamma);
    p.extend_from_slice(&model.omega);
    p.push(model.gamma);
    p.extend_from_slice(&model.j_sun_weights);
    p.extend_from_slice(&model.j_sun_orbital);
    p.extend_from_slice(&model.j_sun_geomag);
    p.push(model.j_sun_bias);
    p.extend_from_slice(&model.coupling_matrix);
    p.push(model.thomson_beta);
    p.extend_from_slice(&model.head_w1);
    p.extend_from_slice(&model.head_b1);
    p.extend_from_slice(&model.head_w2);
    p.push(model.head_b2);
    p
}

fn clone_model(m: &SolarFlareV2) -> SolarFlareV2 {
    let json = serde_json::to_string(m).unwrap();
    serde_json::from_str(&json).unwrap()
}

fn params_to_model(params: &[f32], template: &SolarFlareV2) -> SolarFlareV2 {
    let mut model = clone_model(template);
    let mut idx = 0usize;

    for a in model.encoder_a.iter_mut() {
        for d in 0..8 { a[d] = params[idx]; idx += 1; }
    }
    for b in model.encoder_b.iter_mut() {
        for d in 0..8 { b[d] = params[idx]; idx += 1; }
    }
    for v in model.helmholtz_gamma.iter_mut() { *v = params[idx]; idx += 1; }
    for d in 0..3 { model.omega[d] = params[idx]; idx += 1; }
    model.gamma = params[idx]; idx += 1;
    for v in model.j_sun_weights.iter_mut() { *v = params[idx]; idx += 1; }
    for v in model.j_sun_orbital.iter_mut() { *v = params[idx]; idx += 1; }
    for v in model.j_sun_geomag.iter_mut() { *v = params[idx]; idx += 1; }
    model.j_sun_bias = params[idx]; idx += 1;
    for v in model.coupling_matrix.iter_mut() { *v = params[idx]; idx += 1; }
    model.thomson_beta = params[idx]; idx += 1;
    for v in model.head_w1.iter_mut() { *v = params[idx]; idx += 1; }
    for v in model.head_b1.iter_mut() { *v = params[idx]; idx += 1; }
    for v in model.head_w2.iter_mut() { *v = params[idx]; idx += 1; }
    model.head_b2 = params[idx]; // idx += 1;
    model
}

// ── Streaming evaluation ──

fn stream_ar(
    model: &SolarFlareV2,
    ar: &ArTimeSeries,
    kp: Option<&KpLookup>,
) -> Vec<(f32, f32)> {
    let mut state = V2StreamState::new();
    let mut results = Vec::with_capacity(ar.observations.len());

    for obs in &ar.observations {
        let expanded = expand_obs(&obs.sharp_norm, obs.jd, kp);
        let prob = model.step(&expanded, &mut state);
        // Clamp Helmholtz accumulators to prevent NaN from perturbed gammas
        for v in state.helmholtz_state.iter_mut() {
            *v = v.clamp(-10.0, 10.0);
        }
        if prob.is_finite() {
            results.push((prob, obs.label));
        } else {
            results.push((0.5, obs.label)); // safe fallback
        }
    }
    results
}

fn expand_obs(sharp: &[f32; 9], jd: f64, kp: Option<&KpLookup>) -> Vec<f32> {
    let mut out = Vec::with_capacity(N_INPUT);
    out.extend_from_slice(sharp);
    let orb = orbital::orbital_inputs(jd);
    out.extend_from_slice(&orb);
    let (kp_norm, dkp_norm) = match kp {
        Some(lookup) => lookup.lookup(jd),
        None => (0.33f32, 0.0f32),
    };
    out.push(kp_norm);
    out.push(dkp_norm);
    out
}

fn eval_streaming_loss(
    model: &SolarFlareV2,
    ars: &[ArTimeSeries],
    kp: Option<&KpLookup>,
) -> f64 {
    let mut total_loss = 0.0f64;
    let mut n = 0usize;
    for ar in ars {
        for (prob, label) in stream_ar(model, ar, kp) {
            let p = (prob as f64).clamp(1e-7, 1.0 - 1e-7);
            let l = label as f64;
            let w = if l > 0.5 { 5.0 } else { 1.0 };
            total_loss += -(w * l * p.ln() + (1.0 - l) * (1.0 - p).ln());
            n += 1;
        }
    }
    total_loss / n.max(1) as f64
}

fn eval_streaming_detailed(
    model: &SolarFlareV2,
    ars: &[ArTimeSeries],
    kp: Option<&KpLookup>,
) -> (f64, Vec<f32>, Vec<f32>) {
    let mut total_loss = 0.0f64;
    let mut preds = Vec::new();
    let mut labels = Vec::new();
    for ar in ars {
        for (prob, label) in stream_ar(model, ar, kp) {
            let p = (prob as f64).clamp(1e-7, 1.0 - 1e-7);
            let l = label as f64;
            let w = if l > 0.5 { 5.0 } else { 1.0 };
            total_loss += -(w * l * p.ln() + (1.0 - l) * (1.0 - p).ln());
            preds.push(prob);
            labels.push(label);
        }
    }
    let n = preds.len().max(1) as f64;
    (total_loss / n, preds, labels)
}

// ── SPSA perturbation (Rademacher: +1 or -1) ──

fn spsa_perturbation(n: usize, seed: u64) -> Vec<f32> {
    (0..n).map(|i| {
        let hash = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed.wrapping_mul(1442695040888963407));
        if hash & 1 == 0 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

fn mean_where(preds: &[f32], labels: &[f32], positive: bool) -> f32 {
    let (sum, count) = preds.iter().zip(labels.iter())
        .filter(|(_, &l)| if positive { l > 0.5 } else { l < 0.5 })
        .fold((0.0f32, 0usize), |(s, c), (&p, _)| (s + p, c + 1));
    if count > 0 { sum / count as f32 } else { 0.0 }
}
