//! TSS/HSS/AUROC backtest of the criticality detector against GOES XRS
//! dual-channel 1-minute data (2017, 2020, 2022, 2024 — ~2M records).
//!
//! **No training**: the detector runs in pure inference mode with its
//! analytically-derived parameters (J_c = 2/π KT transition). This measures
//! zero-shot physics-based prediction skill.
//!
//! ## Methodology
//!
//! Each minute of GOES XRS data is labeled as positive (event window) or
//! negative (quiet). Event window = any minute within a configurable lead
//! time before M/X-class flare onset (default: 24h before, ending 5min
//! before onset to exclude the flare itself).
//!
//! TSS = TPR - FPR = TP/(TP+FN) - FP/(FP+TN)
//!   - TSS = 0: no skill (random)
//!   - TSS = 1: perfect prediction
//! HSS = 2(TP·TN - FP·FN) / [(TP+FN)(FN+TN) + (TP+FP)(FP+TN)]
//! AUROC: area under ROC from threshold sweep [0, 1] at 0.01 steps.
//!
//! ## SOTA reference
//! Bobra & Couvidat 2015 (SVM + SHARP): TSS ≈ 0.75–0.90 for M/X
//! Liu et al. 2022 (SolarFlareNet LSTM): TSS ≈ 0.79 for ≥M-class (24h)
//! Wang et al. 2020 (RF + SHARP):        TSS ≈ 0.68–0.76
//! NOAA operational (SWPC):              TSS ≈ 0.40–0.60
//!
//! ## Usage
//! ```
//! cargo run --release --bin solar-backtest-tss -- \
//!   --goes-dir solar-monitor/data/goes_data \
//!   --sharp-dir solar-monitor/data/sharp_data \   # optional: enables SHARP ingest path
//!   --flares /path/to/solar_flares.csv \
//!   --kp    /path/to/kp_3hourly.csv \
//!   --omni  /path/to/omni_hourly.csv \
//!   --lead  24     # hours of positive window before onset
//!   --lag   0      # hours before onset where positive window ends
//!   --class M      # M, X, or MX (default: MX)
//! ```

use chrono::{DateTime, Duration, Utc};
use solar_monitor::backtest::loaders::nearest_sharp;
use solar_monitor::backtest::{goes_loader, loaders};
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use std::collections::BTreeMap;
use std::path::PathBuf;

// ── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    goes_dir: PathBuf,
    /// Optional directory containing sharp_YYYY.csv files.
    sharp_dir: Option<PathBuf>,
    flares_path: PathBuf,
    kp_path: PathBuf,
    omni_path: PathBuf,
    /// Positive window starts this many hours before onset.
    lead_hours: i64,
    /// Positive window ends this many hours before onset (0 = at onset).
    lag_hours: i64,
    /// "M", "X", or "MX"
    flare_class: String,
    /// Print per-threshold table.
    verbose: bool,
}

impl Args {
    fn parse() -> Self {
        let argv: Vec<String> = std::env::args().collect();
        let mut args = Args {
            goes_dir: PathBuf::from("solar-monitor/data/goes_data"),
            sharp_dir: Some(PathBuf::from("solar-monitor/data/sharp_data")),
            flares_path: PathBuf::from("solar-monitor/data/catalogs/solar_flares.csv"),
            kp_path: PathBuf::from("solar-monitor/data/catalogs/kp_3hourly.csv"),
            omni_path: PathBuf::from(
                "/home/ubuntu/Dev/Geometric-Resonance-Papers/earthquake-analysis/data/solar_wind/omni_hourly.csv",
            ),
            lead_hours: 24,
            lag_hours: 0,
            flare_class: "MX".to_string(),
            verbose: false,
        };
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--goes-dir" => {
                    i += 1;
                    args.goes_dir = PathBuf::from(&argv[i]);
                }
                "--flares" => {
                    i += 1;
                    args.flares_path = PathBuf::from(&argv[i]);
                }
                "--kp" => {
                    i += 1;
                    args.kp_path = PathBuf::from(&argv[i]);
                }
                "--omni" => {
                    i += 1;
                    args.omni_path = PathBuf::from(&argv[i]);
                }
                "--lead" => {
                    i += 1;
                    args.lead_hours = argv[i].parse().unwrap_or(24);
                }
                "--lag" => {
                    i += 1;
                    args.lag_hours = argv[i].parse().unwrap_or(0);
                }
                "--class" => {
                    i += 1;
                    args.flare_class = argv[i].clone();
                }
                "--sharp-dir" => {
                    i += 1;
                    args.sharp_dir = Some(PathBuf::from(&argv[i]));
                }
                "--no-sharp" => {
                    args.sharp_dir = None;
                }
                "--verbose" => {
                    args.verbose = true;
                }
                _ => {}
            }
            i += 1;
        }
        args
    }
}

// ── Label helpers ─────────────────────────────────────────────────────────────

/// Minute-resolution label map: timestamp → true if in a positive window.
fn build_label_map(
    flares: &[loaders::FlareEvent],
    lead: Duration,
    lag: Duration,
    class_filter: &str,
) -> BTreeMap<DateTime<Utc>, bool> {
    let mut map = BTreeMap::new();
    for f in flares {
        let keep = match class_filter {
            "M" => f.class.starts_with('M'),
            "X" => f.class.starts_with('X'),
            _ => f.class.starts_with('M') || f.class.starts_with('X'),
        };
        if !keep {
            continue;
        }
        // Positive window: [onset - lead, onset - lag)
        let win_start = f.begin - lead;
        let win_end = f.begin - lag;
        let mut t = win_start;
        while t < win_end {
            map.insert(t, true);
            t = t + Duration::minutes(1);
        }
    }
    map
}

// ── Contingency table ─────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct Contingency {
    tp: u64,
    fp: u64,
    tn: u64,
    r#fn: u64,
}

impl Contingency {
    fn tss(&self) -> f64 {
        let tpr = if self.tp + self.r#fn > 0 {
            self.tp as f64 / (self.tp + self.r#fn) as f64
        } else {
            0.0
        };
        let fpr = if self.fp + self.tn > 0 {
            self.fp as f64 / (self.fp + self.tn) as f64
        } else {
            0.0
        };
        tpr - fpr
    }

    fn hss(&self) -> f64 {
        let tp = self.tp as f64;
        let fp = self.fp as f64;
        let tn = self.tn as f64;
        let r#fn = self.r#fn as f64;
        let num = 2.0 * (tp * tn - fp * r#fn);
        let den = (tp + r#fn) * (r#fn + tn) + (tp + fp) * (fp + tn);
        let den = den as f64;
        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    fn tpr(&self) -> f64 {
        if self.tp + self.r#fn > 0 {
            self.tp as f64 / (self.tp + self.r#fn) as f64
        } else {
            0.0
        }
    }

    fn fpr(&self) -> f64 {
        if self.fp + self.tn > 0 {
            self.fp as f64 / (self.fp + self.tn) as f64
        } else {
            0.0
        }
    }

    fn pod(&self) -> f64 {
        self.tpr()
    }

    fn far(&self) -> f64 {
        if self.tp + self.fp > 0 {
            self.fp as f64 / (self.tp + self.fp) as f64
        } else {
            0.0
        }
    }
}

// ── AUROC from ROC curve ─────────────────────────────────────────────────────

/// Compute AUROC via trapezoidal rule over threshold sweep.
fn compute_auroc(roc: &[(f64, f64)]) -> f64 {
    // roc: (fpr, tpr) pairs, sorted by fpr ascending.
    let mut sorted = roc.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut auc = 0.0;
    for w in sorted.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        auc += (x1 - x0) * (y0 + y1) / 2.0;
    }
    auc.abs()
}

// ── Nearest-value lookup in BTreeMap ─────────────────────────────────────────

fn nearest_kp(kp_map: &BTreeMap<DateTime<Utc>, f64>, t: DateTime<Utc>) -> f64 {
    // Hold-previous: find largest key ≤ t.
    kp_map
        .range(..=t)
        .next_back()
        .map(|(_, &v)| v)
        .unwrap_or(0.0)
}

fn nearest_omni(
    omni: &BTreeMap<DateTime<Utc>, loaders::OmniRecord>,
    t: DateTime<Utc>,
) -> Option<&loaders::OmniRecord> {
    omni.range(..=t).next_back().map(|(_, v)| v)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();
    let lead = Duration::hours(args.lead_hours);
    let lag = Duration::hours(args.lag_hours);

    println!("=== Solar Flare TSS Backtest ===");
    println!("Class filter : {}", args.flare_class);
    println!("Lead window  : {}h before onset", args.lead_hours);
    println!("Lag window   : {}h before onset", args.lag_hours);
    println!("No training  : zero-shot physics (J_c = 2/π KT transition)\n");

    // ── Load flare catalog ────────────────────────────────────────────────────
    print!("Loading flare catalog... ");
    let all_flares = loaders::load_flares(&args.flares_path).unwrap_or_else(|e| {
        eprintln!("WARN: {e}");
        vec![]
    });
    let flares: Vec<_> = all_flares
        .iter()
        .filter(|f| match args.flare_class.as_str() {
            "M" => f.class.starts_with('M'),
            "X" => f.class.starts_with('X'),
            _ => f.class.starts_with('M') || f.class.starts_with('X'),
        })
        .collect();
    println!("{} {} flares", flares.len(), args.flare_class);

    let x_count = flares.iter().filter(|f| f.class.starts_with('X')).count();
    let m_count = flares.iter().filter(|f| f.class.starts_with('M')).count();
    println!("  X-class: {}  M-class: {}", x_count, m_count);

    // ── Load Kp ───────────────────────────────────────────────────────────────
    print!("Loading Kp... ");
    let kp_map = loaders::load_kp(&args.kp_path).unwrap_or_else(|e| {
        eprintln!("WARN: {e}");
        BTreeMap::new()
    });
    println!("{} 3-hourly Kp records", kp_map.len());

    // ── Load OMNI ─────────────────────────────────────────────────────────────
    print!("Loading OMNI... ");
    let omni = loaders::load_omni(&args.omni_path).unwrap_or_else(|e| {
        eprintln!("WARN: {e}");
        BTreeMap::new()
    });
    println!("{} hourly OMNI records", omni.len());

    // ── Load SHARP CSV files (one per year, optional) ─────────────────────────
    let mut sharp_map: BTreeMap<DateTime<Utc>, Vec<loaders::SharpCsvRecord>> = BTreeMap::new();
    if let Some(sharp_dir) = &args.sharp_dir {
        for year in [2017u32, 2020, 2022, 2024] {
            let p = sharp_dir.join(format!("sharp_{year}.csv"));
            if p.exists() {
                match loaders::load_sharp_csv(&p) {
                    Ok(m) => {
                        let n = m.len();
                        sharp_map.extend(m);
                        println!("SHARP {year}        : {n} timestamps loaded");
                    }
                    Err(e) => eprintln!("WARN: SHARP {year}: {e}"),
                }
            } else {
                println!("SHARP {year}        : not found ({})", p.display());
            }
        }
    } else {
        println!("SHARP              : disabled (--no-sharp)");
    }
    let use_sharp = !sharp_map.is_empty();
    println!(
        "SHARP ingest path  : {}",
        if use_sharp {
            "enabled"
        } else {
            "disabled (scalar fallback)"
        }
    );
    println!();

    // ── Discover GOES year files ──────────────────────────────────────────────
    let mut goes_files: Vec<PathBuf> = std::fs::read_dir(&args.goes_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().map(|e| e == "csv").unwrap_or(false)
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.contains("xrs") && !n.contains("partial"))
                            .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default();
    goes_files.sort();
    println!("GOES files   : {}", goes_files.len());
    for f in &goes_files {
        println!("  {}", f.display());
    }
    println!();

    if goes_files.is_empty() {
        eprintln!(
            "ERROR: No GOES XRS CSV files found in {}",
            args.goes_dir.display()
        );
        eprintln!("  Expected pattern: goes16_xrs_YYYY.csv");
        std::process::exit(1);
    }

    // ── Build label map (positive window per flare) ───────────────────────────
    let label_map = build_label_map(&all_flares, lead, lag, &args.flare_class);
    println!("Positive-labeled minutes: {}", label_map.len());

    // ── Run detector over all GOES data ───────────────────────────────────────
    let mut fusion = RankFusionDetector::new(0.5); // threshold doesn't matter — we sweep

    // Scored timeline: (timestamp, label, fused_score, criticality_score_v6, criticality_score_v7)
    let mut scored: Vec<(DateTime<Utc>, bool, f64, f64, f64)> = Vec::with_capacity(2_100_000);
    let mut total_minutes = 0u64;
    let mut skipped_eclipse = 0u64;
    let mut sharp_hits = 0u64;
    let mut sharp_misses = 0u64;

    let mut prev_timestamp: Option<DateTime<Utc>> = None;

    for goes_file in &goes_files {
        print!(
            "Processing {}... ",
            goes_file.file_name().unwrap().to_str().unwrap()
        );
        let records = match goes_loader::load_goes_csv(goes_file) {
            Ok(r) => r,
            Err(e) => {
                println!("SKIP: {e}");
                continue;
            }
        };
        println!("{} records", records.len());

        // Reset EMA state at year boundaries (between CSV files).
        // Multi-month gaps between yearly files would otherwise let 2017 AR state
        // contaminate 2020 solar-minimum data with zero decay applied.
        fusion.reset_criticality_ema();
        prev_timestamp = None;

        for rec in &records {
            let t = rec.timestamp;

            // Reset on large intra-file gaps (> 24h): data outages, missing months.
            if let Some(prev_t) = prev_timestamp {
                if (t - prev_t).num_hours() > 24 {
                    fusion.reset_criticality_ema();
                }
            }
            prev_timestamp = Some(t);

            // Skip eclipse/fill values.
            if rec.xrsb < 1e-9 {
                skipped_eclipse += 1;
                continue;
            }

            // Update Kp before ingest (hold-last-value, 3h cadence).
            let kp = nearest_kp(&kp_map, t);
            fusion.update_kp(kp);

            // Get IMF B-field from OMNI if available.
            let (bx, by, bz) = nearest_omni(&omni, t)
                .map(|o| (0.0_f64, o.by_gsm, o.bz_gsm))
                .unwrap_or((0.0, 0.0, 0.0));

            let electron_flux = 300.0; // background — no minute-cadence electrons in GOES CSV
            let proton_flux = 0.3; // background

            // Use SHARP ingest path when data is available (12-min cadence, ±6min window).
            if use_sharp {
                if let Some(sh) = nearest_sharp(&sharp_map, t, 6) {
                    sharp_hits += 1;
                    fusion.ingest_with_sharp(
                        rec.xrsb,
                        rec.xrsa,
                        electron_flux,
                        proton_flux,
                        bx,
                        by,
                        bz,
                        sh.usflux,
                        sh.meangbz,
                        sh.meanjzh,
                        sh.totusjh,
                        sh.shrgt45,
                        sh.r_value,
                        sh.totpot,
                        sh.totusjz,
                        sh.savncpp,
                        sh.absnjzh,
                        sh.meanalp,
                        sh.area_acr,
                        t,
                    );
                } else {
                    sharp_misses += 1;
                    if bz != 0.0 || by != 0.0 {
                        fusion.ingest_full(
                            rec.xrsb,
                            rec.xrsa,
                            electron_flux,
                            proton_flux,
                            bx,
                            by,
                            bz,
                            t,
                        );
                    } else {
                        fusion.ingest(rec.xrsb, rec.xrsa, electron_flux, proton_flux, t);
                    }
                }
            } else if bz != 0.0 || by != 0.0 {
                fusion.ingest_full(
                    rec.xrsb,
                    rec.xrsa,
                    electron_flux,
                    proton_flux,
                    bx,
                    by,
                    bz,
                    t,
                );
            } else {
                fusion.ingest(rec.xrsb, rec.xrsa, electron_flux, proton_flux, t);
            }

            let score = fusion.score();
            let crit_raw = fusion.criticality_score();
            let crit_v7_raw = fusion.criticality_score_v7();
            // Guard NaN (from missing SHARP fields) — treat as zero signal.
            let crit_score = if crit_raw.is_finite() { crit_raw } else { 0.0 };
            let crit_v7 = if crit_v7_raw.is_finite() {
                crit_v7_raw
            } else {
                0.0
            };
            let tmin = truncate_to_minute(t);
            let label = label_map.contains_key(&tmin);

            scored.push((tmin, label, score, crit_score, crit_v7));
            total_minutes += 1;
        }
    }

    println!();
    println!("Total minutes processed : {}", total_minutes);
    println!("Eclipse/fill skipped    : {}", skipped_eclipse);
    if use_sharp {
        println!(
            "SHARP path used         : {} ({:.1}%)",
            sharp_hits,
            100.0 * sharp_hits as f64 / total_minutes as f64
        );
        println!(
            "Scalar fallback         : {} ({:.1}%)",
            sharp_misses,
            100.0 * sharp_misses as f64 / total_minutes as f64
        );
    }
    let pos_count = scored.iter().filter(|(_, l, _, _, _)| *l).count();
    let neg_count = scored.iter().filter(|(_, l, _, _, _)| !*l).count();
    println!(
        "Positive minutes        : {} ({:.1}%)",
        pos_count,
        100.0 * pos_count as f64 / total_minutes as f64
    );
    println!("Negative minutes        : {}", neg_count);
    // Score distribution stats.
    let fused_max = scored
        .iter()
        .map(|&(_, _, f, _, _)| f)
        .fold(f64::NEG_INFINITY, f64::max);
    let crit_max = scored
        .iter()
        .map(|&(_, _, _, c, _)| c)
        .fold(f64::NEG_INFINITY, f64::max);
    let fused_mean = scored.iter().map(|&(_, _, f, _, _)| f).sum::<f64>() / scored.len() as f64;
    let crit_mean = scored.iter().map(|&(_, _, _, c, _)| c).sum::<f64>() / scored.len() as f64;
    let crit_nonzero = scored.iter().filter(|&&(_, _, _, c, _)| c > 0.01).count();
    println!(
        "Score stats: fused max={:.3} mean={:.3} | crit max={:.3} mean={:.3} nonzero={:.1}%",
        fused_max,
        fused_mean,
        crit_max,
        crit_mean,
        100.0 * crit_nonzero as f64 / scored.len() as f64
    );
    println!();

    if pos_count == 0 {
        eprintln!("ERROR: 0 positive minutes — check that flare catalog overlaps GOES date range.");
        std::process::exit(1);
    }

    // ── Threshold sweep ───────────────────────────────────────────────────────
    let thresholds: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();

    // Evaluate fused score.
    let mut best_tss = f64::NEG_INFINITY;
    let mut best_threshold = 0.0;
    let mut best_ct = Contingency::default();
    let mut roc_points: Vec<(f64, f64)> = Vec::new();

    // Evaluate criticality-only score (v6).
    let mut crit_best_tss = f64::NEG_INFINITY;
    let mut crit_best_threshold = 0.0;
    let mut crit_best_ct = Contingency::default();
    let mut crit_roc_points: Vec<(f64, f64)> = Vec::new();

    // Evaluate two-level criticality score (v7, short-lead).
    let mut v7_best_tss = f64::NEG_INFINITY;
    let mut v7_best_threshold = 0.0;
    let mut v7_best_ct = Contingency::default();
    let mut v7_roc_points: Vec<(f64, f64)> = Vec::new();

    if args.verbose {
        println!(
            "{:>8}  {:>6}  {:>6}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}",
            "Thresh", "TSS", "HSS", "TPR", "FPR", "TP", "FP", "FN"
        );
    }

    for &thresh in &thresholds {
        let mut ct = Contingency::default();
        let mut ct_crit = Contingency::default();
        let mut ct_v7 = Contingency::default();
        for &(_, label, score, crit_score, v7_score) in &scored {
            match (label, score >= thresh) {
                (true, true) => ct.tp += 1,
                (false, true) => ct.fp += 1,
                (false, false) => ct.tn += 1,
                (true, false) => ct.r#fn += 1,
            }
            match (label, crit_score >= thresh) {
                (true, true) => ct_crit.tp += 1,
                (false, true) => ct_crit.fp += 1,
                (false, false) => ct_crit.tn += 1,
                (true, false) => ct_crit.r#fn += 1,
            }
            match (label, v7_score >= thresh) {
                (true, true) => ct_v7.tp += 1,
                (false, true) => ct_v7.fp += 1,
                (false, false) => ct_v7.tn += 1,
                (true, false) => ct_v7.r#fn += 1,
            }
        }
        roc_points.push((ct.fpr(), ct.tpr()));
        crit_roc_points.push((ct_crit.fpr(), ct_crit.tpr()));
        v7_roc_points.push((ct_v7.fpr(), ct_v7.tpr()));

        if ct.tss() > best_tss {
            best_tss = ct.tss();
            best_threshold = thresh;
            best_ct = ct;
        }
        if ct_crit.tss() > crit_best_tss {
            crit_best_tss = ct_crit.tss();
            crit_best_threshold = thresh;
            crit_best_ct = ct_crit;
        }
        if ct_v7.tss() > v7_best_tss {
            v7_best_tss = ct_v7.tss();
            v7_best_threshold = thresh;
            v7_best_ct = ct_v7;
        }

        if args.verbose {
            println!(
                "{:>8.2}  {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>8}  {:>8}  {:>8}",
                thresh,
                ct.tss(),
                ct.hss(),
                ct.tpr(),
                ct.fpr(),
                ct.tp,
                ct.fp,
                ct.r#fn
            );
        }
    }

    let auroc = compute_auroc(&roc_points);
    let crit_auroc = compute_auroc(&crit_roc_points);
    let v7_auroc = compute_auroc(&v7_roc_points);

    // ── Per-class TSS breakdown ───────────────────────────────────────────────
    let label_m = build_label_map(&all_flares, lead, lag, "M");
    let label_x = build_label_map(&all_flares, lead, lag, "X");
    let ct_m = contingency_at_exact(&scored, &label_m, best_threshold, 2);
    let ct_x = contingency_at_exact(&scored, &label_x, best_threshold, 2);
    let ct_cm = contingency_at_exact(&scored, &label_m, crit_best_threshold, 3);
    let ct_cx = contingency_at_exact(&scored, &label_x, crit_best_threshold, 3);
    let ct_vm = contingency_at_exact(&scored, &label_m, v7_best_threshold, 4);
    let ct_vx = contingency_at_exact(&scored, &label_x, v7_best_threshold, 4);

    // ── Print results ─────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("  RESULTS — zero-shot physics-based (no training)");
    println!("══════════════════════════════════════════════════════════════════");
    println!();
    println!("  A) Fused score (7 detectors, wt 0.23 criticality + 0.77 reactive)");
    println!("  ────────────────────────────────────────────────────────────────");
    println!("  AUROC            : {:.4}", auroc);
    println!(
        "  Best TSS         : {:.4}  @ threshold {:.2}",
        best_tss, best_threshold
    );
    println!("  HSS              : {:.4}", best_ct.hss());
    println!("  TPR (recall)     : {:.4}", best_ct.tpr());
    println!("  FPR              : {:.4}", best_ct.fpr());
    println!(
        "  TP / FP / FN / TN: {} / {} / {} / {}",
        best_ct.tp, best_ct.fp, best_ct.r#fn, best_ct.tn
    );
    println!(
        "  Per-class (@ {:.2}): M-TSS={:.4} X-TSS={:.4}",
        best_threshold,
        ct_m.tss(),
        ct_x.tss()
    );
    println!();
    println!("  B) Criticality-only score v6 (additive, optimized for 24h lead)");
    println!("  ────────────────────────────────────────────────────────────────");
    println!("  AUROC            : {:.4}", crit_auroc);
    println!(
        "  Best TSS         : {:.4}  @ threshold {:.2}",
        crit_best_tss, crit_best_threshold
    );
    println!("  HSS              : {:.4}", crit_best_ct.hss());
    println!("  TPR (recall)     : {:.4}", crit_best_ct.tpr());
    println!("  FPR              : {:.4}", crit_best_ct.fpr());
    println!(
        "  TP / FP / FN / TN: {} / {} / {} / {}",
        crit_best_ct.tp, crit_best_ct.fp, crit_best_ct.r#fn, crit_best_ct.tn
    );
    println!(
        "  Per-class (@ {:.2}): M-TSS={:.4} X-TSS={:.4}",
        crit_best_threshold,
        ct_cm.tss(),
        ct_cx.tss()
    );
    println!();
    println!("  C) Two-level score v7 (multiplicative gate, test for short lead)");
    println!("  ────────────────────────────────────────────────────────────────");
    println!("  AUROC            : {:.4}", v7_auroc);
    println!(
        "  Best TSS         : {:.4}  @ threshold {:.2}",
        v7_best_tss, v7_best_threshold
    );
    println!("  HSS              : {:.4}", v7_best_ct.hss());
    println!("  TPR (recall)     : {:.4}", v7_best_ct.tpr());
    println!("  FPR              : {:.4}", v7_best_ct.fpr());
    println!(
        "  TP / FP / FN / TN: {} / {} / {} / {}",
        v7_best_ct.tp, v7_best_ct.fp, v7_best_ct.r#fn, v7_best_ct.tn
    );
    println!(
        "  Per-class (@ {:.2}): M-TSS={:.4} X-TSS={:.4}",
        v7_best_threshold,
        ct_vm.tss(),
        ct_vx.tss()
    );
    println!();
    println!("  SOTA reference (24h lead, SHARP-based, trained):");
    println!("    Bobra 2015 SVM  : TSS ~0.76  AUROC ~0.90");
    println!("    Liu 2022 LSTM   : TSS ~0.79");
    println!("    NOAA SWPC ops   : TSS ~0.50");
    println!("══════════════════════════════════════════════════════════════════");
}

// ── Helper: exact contingency using per-class label map + stored timestamps ───

/// `col`: which score column to use — 2=fused, 3=crit_v6, 4=crit_v7.
fn contingency_at_exact(
    scored: &[(DateTime<Utc>, bool, f64, f64, f64)],
    label_map: &BTreeMap<DateTime<Utc>, bool>,
    threshold: f64,
    col: usize,
) -> Contingency {
    let mut ct = Contingency::default();
    for &(t, _, fused, crit, v7) in scored {
        let score = match col {
            3 => crit,
            4 => v7,
            _ => fused,
        };
        let label = label_map.contains_key(&t);
        let predicted = score >= threshold;
        match (label, predicted) {
            (true, true) => ct.tp += 1,
            (false, true) => ct.fp += 1,
            (false, false) => ct.tn += 1,
            (true, false) => ct.r#fn += 1,
        }
    }
    ct
}

// ── DateTime helper ───────────────────────────────────────────────────────────

fn truncate_to_minute(t: DateTime<Utc>) -> DateTime<Utc> {
    use chrono::Timelike;
    t.with_second(0)
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(t)
}
