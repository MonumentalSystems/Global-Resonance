//! SHARP time series dataset for training SolarFlareModel.
//!
//! Loads SHARP magnetogram parameter time series and GOES flare labels,
//! creating (features, label) pairs for binary classification:
//!   - features: (T, 9) min-max normalized SHARP params
//!   - label: 1.0 if flare ≥ C5.0 within prediction window, 0.0 otherwise
//!
//! Data sources:
//! - SHARP: JSOC hmi.sharp_cea_720s at 12-min cadence (CSV export)
//! - Flares: GOES X-ray event list (NCEI catalog)
//!
//! Following SolarFlareNet (Abduallah et al. 2023) for fair comparison:
//! - 9 SHARP parameters: TOTUSJH, TOTUSJZ, USFLUX, MEANALP, R_VALUE, TOTPOT,
//!                       SAVNCPP, AREA_ACR, ABSNJZH
//! - AR-level train/test split by HARP number (no data leakage)
//! - GWN augmentation for minority class (training only)

use chrono::{DateTime, NaiveDateTime, Utc};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// The 9 SolarFlareNet SHARP parameter names (column order).
pub const SHARP_FIELD_NAMES: [&str; 9] = [
    "TOTUSJH", "TOTUSJZ", "USFLUX", "MEANALP", "R_VALUE", "TOTPOT", "SAVNCPP", "AREA_ACR",
    "ABSNJZH",
];

/// A single SHARP time series sample with label.
#[derive(Debug, Clone)]
pub struct SharpSample {
    /// SHARP features: (seq_len * 9) flat row-major, min-max normalized to [0, 1].
    pub features: Vec<f32>,
    /// Binary label: 1.0 = flare ≥ C5.0 within prediction window, 0.0 = quiet.
    pub label: f32,
    /// HARP number (for AR-level split).
    pub harpnum: u32,
    /// Timestamp of the LAST row in the sequence (for orbital angle computation).
    /// Julian date. 0.0 if unavailable.
    pub jd: f64,
}

/// Raw SHARP record from CSV (one row per 12-min observation).
#[derive(Debug, Clone, Deserialize)]
pub struct RawSharpRow {
    #[serde(alias = "T_REC", alias = "time", alias = "time_tag")]
    pub time: String,
    #[serde(alias = "HARPNUM", alias = "harpnum")]
    pub harpnum: u32,
    #[serde(alias = "TOTUSJH", default)]
    pub totusjh: f64,
    #[serde(alias = "TOTUSJZ", default)]
    pub totusjz: f64,
    #[serde(alias = "USFLUX", default)]
    pub usflux: f64,
    #[serde(alias = "MEANALP", default)]
    pub meanalp: f64,
    #[serde(alias = "R_VALUE", default)]
    pub r_value: f64,
    #[serde(alias = "TOTPOT", default)]
    pub totpot: f64,
    #[serde(alias = "SAVNCPP", default)]
    pub savncpp: f64,
    #[serde(alias = "AREA_ACR", default)]
    pub area_acr: f64,
    #[serde(alias = "ABSNJZH", default)]
    pub absnjzh: f64,
}

impl RawSharpRow {
    /// Extract the 9 SolarFlareNet fields as an array.
    pub fn to_array(&self) -> [f64; 9] {
        [
            self.totusjh,
            self.totusjz,
            self.usflux,
            self.meanalp,
            self.r_value,
            self.totpot,
            self.savncpp,
            self.area_acr,
            self.absnjzh,
        ]
    }
}

/// Flare event from GOES catalog.
#[derive(Debug, Clone, Deserialize)]
pub struct FlareEvent {
    /// Start time (ISO 8601 or similar).
    #[serde(alias = "start_time", alias = "begin_time", alias = "beginTime")]
    pub start_time: String,
    /// Flare class string: e.g., "C5.3", "M1.2", "X1.5".
    #[serde(alias = "fl_goescls", alias = "class", alias = "classType")]
    pub class: String,
    /// NOAA active region number.
    #[serde(
        alias = "noaa_ar",
        alias = "ar_num",
        alias = "activeRegionNum",
        default,
        deserialize_with = "deserialize_f64_or_empty"
    )]
    pub noaa_ar: f64,
}

impl FlareEvent {
    /// Parse flare class to peak flux in W/m². Returns None for invalid class.
    pub fn peak_flux(&self) -> Option<f64> {
        let class = self.class.trim();
        if class.is_empty() {
            return None;
        }
        let (prefix, mantissa_str) = class.split_at(1);
        let mantissa: f64 = mantissa_str.parse().ok()?;
        let base = match prefix {
            "A" => 1e-8,
            "B" => 1e-7,
            "C" => 1e-6,
            "M" => 1e-5,
            "X" => 1e-4,
            _ => return None,
        };
        Some(base * mantissa)
    }

    /// Whether this flare is ≥ C5.0 class (5e-6 W/m²).
    pub fn is_c5_or_above(&self) -> bool {
        self.peak_flux().map_or(false, |f| f >= 5e-6)
    }
}

/// Min-max normalization statistics per field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormStats {
    pub min: [f64; 9],
    pub max: [f64; 9],
}

/// Which SHARP fields should be log-transformed before normalization.
/// SHARP fields are heavy-tailed (span orders of magnitude), so log-transform
/// is standard practice (SolarFlareNet, Bobra & Couvidat do this).
/// All fields are positive except MEANALP (index 3) which can be negative.
const LOG_TRANSFORM: [bool; 9] = [
    true,  // TOTUSJH  — total unsigned current helicity
    true,  // TOTUSJZ  — total unsigned vertical current
    true,  // USFLUX   — total unsigned flux
    false, // MEANALP  — mean twist (can be negative)
    true,  // R_VALUE  — total PIL flux
    true,  // TOTPOT   — total free energy
    true,  // SAVNCPP  — sum of net current
    true,  // AREA_ACR — AR area
    true,  // ABSNJZH  — absolute net current helicity
];

impl NormStats {
    /// Apply log-transform to a raw value for fields that need it.
    /// Uses log10(|x| + 1) to handle zeros and preserve ordering.
    fn transform(field_idx: usize, val: f64) -> f64 {
        if LOG_TRANSFORM[field_idx] {
            (val.abs() + 1.0).log10()
        } else {
            val
        }
    }

    /// Compute min/max from training data (in transformed space).
    pub fn from_data(data: &[[f64; 9]]) -> Self {
        let mut min = [f64::INFINITY; 9];
        let mut max = [f64::NEG_INFINITY; 9];
        for row in data {
            for i in 0..9 {
                if row[i].is_finite() {
                    let v = Self::transform(i, row[i]);
                    min[i] = min[i].min(v);
                    max[i] = max[i].max(v);
                }
            }
        }
        // Prevent div-by-zero
        for i in 0..9 {
            if (max[i] - min[i]).abs() < 1e-10 {
                max[i] = min[i] + 1.0;
            }
        }
        NormStats { min, max }
    }

    /// Normalize a single value to [0, 1] (applies log-transform first).
    pub fn normalize(&self, field_idx: usize, val: f64) -> f32 {
        let v = Self::transform(field_idx, val);
        let n = ((v - self.min[field_idx]) / (self.max[field_idx] - self.min[field_idx]))
            .clamp(0.0, 1.0);
        n as f32
    }

    /// Normalize a full 9-field row.
    pub fn normalize_row(&self, row: &[f64; 9]) -> [f32; 9] {
        let mut out = [0.0f32; 9];
        for i in 0..9 {
            out[i] = self.normalize(i, row[i]);
        }
        out
    }
}

/// Dataset configuration.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Number of timesteps per sample.
    pub seq_len: usize,
    /// Prediction window in minutes (e.g., 120 for 2h, 1440 for 24h).
    pub prediction_window_min: u64,
    /// Training set fraction (rest is test).
    pub train_fraction: f64,
    /// Whether to apply GWN augmentation to minority class.
    pub augment: bool,
    /// GWN noise standard deviation (fraction of range).
    pub noise_std: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        DatasetConfig {
            seq_len: 10,
            prediction_window_min: 120, // 2h, matching Doria Rosales
            train_fraction: 0.7,
            augment: true,
            noise_std: 0.05,
            seed: 42,
        }
    }
}

/// Loaded and split dataset.
pub struct SharpDataset {
    pub train: Vec<SharpSample>,
    pub test: Vec<SharpSample>,
    pub norm: NormStats,
    pub config: DatasetConfig,
    pub n_positive_train: usize,
    pub n_negative_train: usize,
    pub n_positive_test: usize,
    pub n_negative_test: usize,
}

impl SharpDataset {
    /// Load dataset from SHARP CSV and GOES flare catalog CSV.
    ///
    /// SHARP CSV expected columns: T_REC, HARPNUM, TOTUSJH, TOTUSJZ, USFLUX,
    /// MEANALP, R_VALUE, TOTPOT, SAVNCPP, AREA_ACR, ABSNJZH
    ///
    /// Flare CSV expected columns: start_time, fl_goescls, noaa_ar
    pub fn load(
        sharp_path: &Path,
        flare_path: &Path,
        config: DatasetConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Parse SHARP CSV
        let mut reader = csv::Reader::from_path(sharp_path)?;
        let mut rows: Vec<RawSharpRow> = Vec::new();
        for result in reader.deserialize() {
            let row: RawSharpRow = result?;
            rows.push(row);
        }

        // Parse flare catalog
        let mut flare_reader = csv::Reader::from_path(flare_path)?;
        let mut flares: Vec<FlareEvent> = Vec::new();
        for result in flare_reader.deserialize() {
            let event: FlareEvent = result?;
            if event.is_c5_or_above() {
                flares.push(event);
            }
        }

        // Group SHARP rows by HARP number, sorted by time.
        // Filter out records with zero USFLUX or TOTPOT (invalid/missing measurements)
        // which create false 0→value→0 discontinuities in the time series.
        let mut by_harp: HashMap<u32, Vec<(DateTime<Utc>, [f64; 9])>> = HashMap::new();
        for row in &rows {
            let arr = row.to_array();
            // arr[2] = USFLUX, arr[5] = TOTPOT — skip invalid records
            if arr[2] <= 0.0 || arr[5] <= 0.0 {
                continue;
            }
            if let Some(dt) = parse_flexible_time(&row.time) {
                by_harp.entry(row.harpnum).or_default().push((dt, arr));
            }
        }
        // Sort each HARP's records by time
        for records in by_harp.values_mut() {
            records.sort_by_key(|r| r.0);
        }

        // Collect all data rows for normalization (training set only after split)
        let all_rows: Vec<[f64; 9]> = rows.iter().map(|r| r.to_array()).collect();

        // AR-level split: assign HARPs to train or test
        let mut harps: Vec<u32> = by_harp.keys().copied().collect();
        harps.sort(); // Deterministic
                      // Simple hash-based split using seed
        let mut train_harps = HashSet::new();
        let mut test_harps = HashSet::new();
        for &harp in &harps {
            let hash = (harp as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(config.seed);
            if (hash % 1000) < (config.train_fraction * 1000.0) as u64 {
                train_harps.insert(harp);
            } else {
                test_harps.insert(harp);
            }
        }

        // Compute normalization stats from training HARPs only
        let train_rows: Vec<[f64; 9]> = rows
            .iter()
            .filter(|r| train_harps.contains(&r.harpnum))
            .map(|r| r.to_array())
            .collect();
        let norm = if train_rows.is_empty() {
            NormStats::from_data(&all_rows)
        } else {
            NormStats::from_data(&train_rows)
        };

        // Parse flare times into DateTime for proper temporal matching.
        let mut flare_datetimes: Vec<DateTime<Utc>> = flares
            .iter()
            .filter_map(|f| parse_flexible_time(&f.start_time))
            .collect();
        flare_datetimes.sort();

        let window_duration = chrono::Duration::minutes(config.prediction_window_min as i64);

        // Create samples: sliding window of seq_len records per HARP.
        // Stride = seq_len (non-overlapping) to avoid near-duplicate samples.
        // With 12-min cadence and seq_len=10, each sample covers 2h.
        //
        // Parallelized: each HARP is processed independently via rayon.
        let seq_len = config.seq_len;
        let max_gap = chrono::Duration::minutes(15);

        let harp_entries: Vec<(&u32, &Vec<(DateTime<Utc>, [f64; 9])>)> = by_harp.iter().collect();

        let all_samples: Vec<(SharpSample, bool)> = harp_entries
            .par_iter()
            .flat_map(|(&harp, records)| {
                if records.len() < seq_len {
                    return Vec::new();
                }
                let is_train = train_harps.contains(&harp);
                let stride = seq_len;
                let mut samples = Vec::new();
                let mut start = 0;

                while start + seq_len <= records.len() {
                    let window = &records[start..start + seq_len];

                    // Check temporal continuity: reject windows with gaps > 15 min
                    let is_continuous = (1..seq_len).all(|i| {
                        let gap = window[i].0 - window[i - 1].0;
                        gap <= max_gap && gap >= chrono::Duration::zero()
                    });

                    if !is_continuous {
                        start += 1;
                        continue;
                    }

                    let last_dt = window[seq_len - 1].0;
                    let window_end = last_dt + window_duration;

                    // Binary search for flares in [last_dt, window_end]
                    let idx = flare_datetimes.partition_point(|ft| *ft < last_dt);
                    let has_flare =
                        idx < flare_datetimes.len() && flare_datetimes[idx] <= window_end;
                    let label = if has_flare { 1.0f32 } else { 0.0f32 };

                    start += stride;

                    // Normalize features
                    let mut features = Vec::with_capacity(seq_len * 9);
                    for (_, row) in window {
                        let normed = norm.normalize_row(row);
                        features.extend_from_slice(&normed);
                    }

                    let jd = datetime_to_jd(&last_dt);
                    samples.push((
                        SharpSample {
                            features,
                            label,
                            harpnum: harp,
                            jd,
                        },
                        is_train,
                    ));
                }
                samples
            })
            .collect();

        let mut train_samples: Vec<SharpSample> = Vec::new();
        let mut test_samples: Vec<SharpSample> = Vec::new();
        for (sample, is_train) in all_samples {
            if is_train {
                train_samples.push(sample);
            } else {
                test_samples.push(sample);
            }
        }

        // Count class balance
        let n_positive_train = train_samples.iter().filter(|s| s.label > 0.5).count();
        let n_negative_train = train_samples.len() - n_positive_train;
        let n_positive_test = test_samples.iter().filter(|s| s.label > 0.5).count();
        let n_negative_test = test_samples.len() - n_positive_test;

        // GWN augmentation on minority class (positive) in training set
        if config.augment && n_positive_train > 0 && n_positive_train < n_negative_train {
            let augment_ratio = (n_negative_train / n_positive_train).min(5);
            let positive_samples: Vec<SharpSample> = train_samples
                .iter()
                .filter(|s| s.label > 0.5)
                .cloned()
                .collect();

            for rep in 0..augment_ratio {
                for sample in &positive_samples {
                    let mut augmented = sample.clone();
                    // Add GWN with seed-dependent noise
                    for (i, v) in augmented.features.iter_mut().enumerate() {
                        let noise_seed = (i as u64)
                            .wrapping_mul(2654435761)
                            .wrapping_add(rep as u64 * 12345)
                            .wrapping_add(config.seed);
                        let noise =
                            ((noise_seed % 10000) as f32 / 10000.0 - 0.5) * 2.0 * config.noise_std;
                        *v = (*v + noise).clamp(0.0, 1.0);
                    }
                    train_samples.push(augmented);
                }
            }
        }

        Ok(SharpDataset {
            train: train_samples,
            test: test_samples,
            norm,
            config,
            n_positive_train,
            n_negative_train,
            n_positive_test,
            n_negative_test,
        })
    }

    /// Load dataset from a directory of gzipped SHARP CSVs + a flare catalog.
    ///
    /// Reads all `*.csv.gz` files in `sharp_dir`, concatenates them,
    /// then proceeds with the same pipeline as `load()`.
    pub fn load_gzipped_dir(
        sharp_dir: &Path,
        flare_path: &Path,
        config: DatasetConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Collect all .csv.gz files
        let mut gz_files: Vec<std::path::PathBuf> = std::fs::read_dir(sharp_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map_or(false, |ext| ext == "gz")
                    && p.to_string_lossy().ends_with(".csv.gz")
            })
            .collect();
        gz_files.sort();

        if gz_files.is_empty() {
            return Err(format!("No .csv.gz files found in {}", sharp_dir.display()).into());
        }

        // Read all gzipped CSVs into rows
        let mut rows: Vec<RawSharpRow> = Vec::new();
        for gz_path in &gz_files {
            let file = std::fs::File::open(gz_path)?;
            let decoder = GzDecoder::new(file);
            let mut reader = csv::Reader::from_reader(decoder);
            let mut count = 0usize;
            for result in reader.deserialize() {
                match result {
                    Ok(row) => {
                        rows.push(row);
                        count += 1;
                    }
                    Err(_) => continue, // skip malformed rows
                }
            }
            eprintln!(
                "  Loaded {} records from {}",
                count,
                gz_path.file_name().unwrap_or_default().to_string_lossy()
            );
        }
        eprintln!("  Total SHARP records: {}", rows.len());

        // From here, reuse the same logic as load() but with pre-parsed rows
        Self::from_rows(rows, flare_path, config)
    }

    /// Internal: build dataset from pre-parsed rows + flare catalog.
    fn from_rows(
        rows: Vec<RawSharpRow>,
        flare_path: &Path,
        config: DatasetConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Parse flare catalog
        let mut flare_reader = csv::Reader::from_path(flare_path)?;
        let mut flares: Vec<FlareEvent> = Vec::new();
        for result in flare_reader.deserialize() {
            let event: FlareEvent = result?;
            if event.is_c5_or_above() {
                flares.push(event);
            }
        }

        // Group SHARP rows by HARP number, sorted by time.
        let mut by_harp: HashMap<u32, Vec<(DateTime<Utc>, [f64; 9])>> = HashMap::new();
        for row in &rows {
            let arr = row.to_array();
            if arr[2] <= 0.0 || arr[5] <= 0.0 {
                continue;
            }
            if let Some(dt) = parse_flexible_time(&row.time) {
                by_harp.entry(row.harpnum).or_default().push((dt, arr));
            }
        }
        for records in by_harp.values_mut() {
            records.sort_by_key(|r| r.0);
        }

        let all_rows: Vec<[f64; 9]> = rows.iter().map(|r| r.to_array()).collect();

        // AR-level split
        let mut harps: Vec<u32> = by_harp.keys().copied().collect();
        harps.sort();
        let mut train_harps = HashSet::new();
        let mut test_harps = HashSet::new();
        for &harp in &harps {
            let hash = (harp as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(config.seed);
            if (hash % 1000) < (config.train_fraction * 1000.0) as u64 {
                train_harps.insert(harp);
            } else {
                test_harps.insert(harp);
            }
        }

        let train_rows: Vec<[f64; 9]> = rows
            .iter()
            .filter(|r| train_harps.contains(&r.harpnum))
            .map(|r| r.to_array())
            .collect();
        let norm = if train_rows.is_empty() {
            NormStats::from_data(&all_rows)
        } else {
            NormStats::from_data(&train_rows)
        };

        let mut flare_datetimes: Vec<DateTime<Utc>> = flares
            .iter()
            .filter_map(|f| parse_flexible_time(&f.start_time))
            .collect();
        flare_datetimes.sort();

        let window_duration = chrono::Duration::minutes(config.prediction_window_min as i64);
        let seq_len = config.seq_len;
        let max_gap = chrono::Duration::minutes(15);

        let harp_entries: Vec<(&u32, &Vec<(DateTime<Utc>, [f64; 9])>)> = by_harp.iter().collect();

        let all_samples: Vec<(SharpSample, bool)> = harp_entries
            .par_iter()
            .flat_map(|(&harp, records)| {
                if records.len() < seq_len {
                    return Vec::new();
                }
                let is_train = train_harps.contains(&harp);
                let stride = seq_len;
                let mut samples = Vec::new();
                let mut start = 0;

                while start + seq_len <= records.len() {
                    let window = &records[start..start + seq_len];
                    let is_continuous = (1..seq_len).all(|i| {
                        let gap = window[i].0 - window[i - 1].0;
                        gap <= max_gap && gap >= chrono::Duration::zero()
                    });
                    if !is_continuous {
                        start += 1;
                        continue;
                    }
                    let last_dt = window[seq_len - 1].0;
                    let window_end = last_dt + window_duration;
                    let idx = flare_datetimes.partition_point(|ft| *ft < last_dt);
                    let has_flare =
                        idx < flare_datetimes.len() && flare_datetimes[idx] <= window_end;
                    let label = if has_flare { 1.0f32 } else { 0.0f32 };
                    start += stride;
                    let mut features = Vec::with_capacity(seq_len * 9);
                    for (_, row) in window {
                        let normed = norm.normalize_row(row);
                        features.extend_from_slice(&normed);
                    }
                    let jd = datetime_to_jd(&last_dt);
                    samples.push((
                        SharpSample {
                            features,
                            label,
                            harpnum: harp,
                            jd,
                        },
                        is_train,
                    ));
                }
                samples
            })
            .collect();

        let mut train_samples: Vec<SharpSample> = Vec::new();
        let mut test_samples: Vec<SharpSample> = Vec::new();
        for (sample, is_train) in all_samples {
            if is_train {
                train_samples.push(sample);
            } else {
                test_samples.push(sample);
            }
        }

        let n_positive_train = train_samples.iter().filter(|s| s.label > 0.5).count();
        let n_negative_train = train_samples.len() - n_positive_train;
        let n_positive_test = test_samples.iter().filter(|s| s.label > 0.5).count();
        let n_negative_test = test_samples.len() - n_positive_test;

        // GWN augmentation
        if config.augment && n_positive_train > 0 && n_positive_train < n_negative_train {
            let augment_ratio = (n_negative_train / n_positive_train).min(5);
            let positive_samples: Vec<SharpSample> = train_samples
                .iter()
                .filter(|s| s.label > 0.5)
                .cloned()
                .collect();
            for rep in 0..augment_ratio {
                for sample in &positive_samples {
                    let mut augmented = sample.clone();
                    for (i, v) in augmented.features.iter_mut().enumerate() {
                        let noise_seed = (i as u64)
                            .wrapping_mul(2654435761)
                            .wrapping_add(rep as u64 * 12345)
                            .wrapping_add(config.seed);
                        let noise =
                            ((noise_seed % 10000) as f32 / 10000.0 - 0.5) * 2.0 * config.noise_std;
                        *v = (*v + noise).clamp(0.0, 1.0);
                    }
                    train_samples.push(augmented);
                }
            }
        }

        Ok(SharpDataset {
            train: train_samples,
            test: test_samples,
            norm,
            config,
            n_positive_train,
            n_negative_train,
            n_positive_test,
            n_negative_test,
        })
    }

    /// Create a synthetic dataset for testing (no file I/O needed).
    ///
    /// Generates random SHARP-like time series with known labels.
    /// Positive samples have elevated TOTPOT and R_VALUE (mimicking flare-productive ARs).
    pub fn synthetic(seq_len: usize, n_samples: usize, seed: u64) -> Self {
        let mut train = Vec::new();
        let mut test = Vec::new();

        for i in 0..n_samples {
            let is_positive = i % 3 == 0; // 33% positive rate
            let is_train = (i * 7 + seed as usize) % 10 < 7; // 70% train

            let mut features = vec![0.0f32; seq_len * 9];
            for ti in 0..seq_len {
                for fi in 0..9 {
                    let base = if is_positive {
                        // Flare-productive AR: elevated values, rising trend
                        0.5 + 0.3 * (ti as f32 / seq_len as f32)
                    } else {
                        // Quiet AR: low, flat values
                        0.1 + 0.1 * ((fi * 13 + ti * 7 + i * 3) as f32 * 0.1).sin().abs()
                    };
                    // Add some noise
                    let noise_hash = ((ti * 9 + fi + i * 100) as u64)
                        .wrapping_mul(2654435761)
                        .wrapping_add(seed);
                    let noise = ((noise_hash % 10000) as f32 / 10000.0 - 0.5) * 0.1;
                    features[ti * 9 + fi] = (base + noise).clamp(0.0, 1.0);
                }
            }

            let sample = SharpSample {
                features,
                label: if is_positive { 1.0 } else { 0.0 },
                harpnum: i as u32,
                jd: 2460115.0 + i as f64, // synthetic: ~mid-2023, one day apart
            };

            if is_train {
                train.push(sample);
            } else {
                test.push(sample);
            }
        }

        let n_positive_train = train.iter().filter(|s| s.label > 0.5).count();
        let n_negative_train = train.len() - n_positive_train;
        let n_positive_test = test.iter().filter(|s| s.label > 0.5).count();
        let n_negative_test = test.len() - n_positive_test;

        SharpDataset {
            train,
            test,
            norm: NormStats {
                min: [0.0; 9],
                max: [1.0; 9],
            },
            config: DatasetConfig {
                seq_len,
                ..Default::default()
            },
            n_positive_train,
            n_negative_train,
            n_positive_test,
            n_negative_test,
        }
    }

    /// Print dataset statistics.
    pub fn print_stats(&self) {
        println!("=== SHARP Dataset ===");
        println!(
            "Train: {} samples ({} positive, {} negative, ratio {:.1}%)",
            self.train.len(),
            self.n_positive_train,
            self.n_negative_train,
            100.0 * self.n_positive_train as f64 / self.train.len().max(1) as f64
        );
        println!(
            "Test:  {} samples ({} positive, {} negative, ratio {:.1}%)",
            self.test.len(),
            self.n_positive_test,
            self.n_negative_test,
            100.0 * self.n_positive_test as f64 / self.test.len().max(1) as f64
        );
        println!("Seq len: {}", self.config.seq_len);
        println!(
            "Prediction window: {} min",
            self.config.prediction_window_min
        );
    }
}

/// Deserialize f64 that may be empty string (CSV fields like "11059.0" or "").
fn deserialize_f64_or_empty<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    if s.is_empty() {
        Ok(0.0)
    } else {
        s.parse::<f64>().map_err(serde::de::Error::custom)
    }
}

/// Parse a timestamp string in multiple formats.
/// Supports: "2024-05-01T00:00:00", "2024-01-01 00:00:00", ISO 8601 with timezone.
fn parse_flexible_time(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    // Try ISO 8601 with T separator
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(ndt.and_utc());
    }
    // Try space separator
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(ndt.and_utc());
    }
    // Try with fractional seconds
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(ndt.and_utc());
    }
    // Try without seconds (e.g., DONKI API: "2024-01-01T08:33")
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M") {
        return Some(ndt.and_utc());
    }
    // Try with trailing Z (e.g., "2024-01-01T08:33Z")
    let s_no_z = s.trim_end_matches('Z');
    if s_no_z != s {
        return parse_flexible_time(s_no_z);
    }
    None
}

/// Shuffle a dataset in place using Fisher-Yates with deterministic seed.
pub fn shuffle(samples: &mut [SharpSample], seed: u64) {
    let n = samples.len();
    for i in (1..n).rev() {
        let j = ((i as u64).wrapping_mul(2654435761).wrapping_add(seed) % (i as u64 + 1)) as usize;
        samples.swap(i, j);
    }
}

// ============================================================================
// Evaluation metrics
// ============================================================================

/// Binary classification metrics.
#[derive(Debug, Clone, Serialize)]
pub struct ClassificationMetrics {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub tss: f64,
    pub bacc: f64,
}

impl ClassificationMetrics {
    /// Compute metrics from predictions and labels.
    pub fn compute(predictions: &[f32], labels: &[f32], threshold: f32) -> Self {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut fn_ = 0usize;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            let predicted_positive = *pred >= threshold;
            let actual_positive = *label > 0.5;
            match (predicted_positive, actual_positive) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let accuracy = (tp + tn) as f64 / (tp + fp + tn + fn_).max(1) as f64;
        let precision = tp as f64 / (tp + fp).max(1) as f64;
        let recall = tp as f64 / (tp + fn_).max(1) as f64;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        // TSS = TP/(TP+FN) - FP/(FP+TN) = Recall - FPR
        let sensitivity = tp as f64 / (tp + fn_).max(1) as f64;
        let specificity = tn as f64 / (tn + fp).max(1) as f64;
        let tss = sensitivity + specificity - 1.0;
        let bacc = (sensitivity + specificity) / 2.0;

        ClassificationMetrics {
            tp,
            fp,
            tn,
            fn_,
            accuracy,
            precision,
            recall,
            f1,
            tss,
            bacc,
        }
    }

    /// Print metrics in a formatted table.
    pub fn print(&self, label: &str) {
        println!("--- {} ---", label);
        println!(
            "  TP={} FP={} TN={} FN={}",
            self.tp, self.fp, self.tn, self.fn_
        );
        println!("  Accuracy:  {:.4}", self.accuracy);
        println!("  Precision: {:.4}", self.precision);
        println!("  Recall:    {:.4}", self.recall);
        println!("  F1:        {:.4}", self.f1);
        println!("  TSS:       {:.4}", self.tss);
        println!("  BACC:      {:.4}", self.bacc);
    }

    /// Find the threshold that maximizes TSS, scanning 100 thresholds in [0, 1].
    /// Returns (best_metrics, best_threshold).
    pub fn optimal_tss(predictions: &[f32], labels: &[f32]) -> (Self, f32) {
        let mut best_tss = f64::NEG_INFINITY;
        let mut best_thresh = 0.5f32;
        let mut best_metrics = Self::compute(predictions, labels, 0.5);
        for i in 1..100 {
            let thresh = i as f32 / 100.0;
            let m = Self::compute(predictions, labels, thresh);
            if m.tss > best_tss {
                best_tss = m.tss;
                best_thresh = thresh;
                best_metrics = m;
            }
        }
        (best_metrics, best_thresh)
    }

    /// Print comparison with SOTA methods.
    pub fn print_sota_comparison(&self) {
        println!("\n=== SOTA Comparison ===");
        println!("| Method                | TSS   | Recall | Precision | F1    |");
        println!("|----------------------|-------|--------|-----------|-------|");
        println!(
            "| SolarFlareModel (ours)| {:.3} | {:.3}  | {:.3}     | {:.3} |",
            self.tss, self.recall, self.precision, self.f1
        );
        println!("| SolarFlareNet (24h)  | 0.835 | 0.891  | 0.949     | 0.919 |");
        println!("| Doria Rosales (2h)   | 0.862 | 0.976  | 0.879     | 0.925 |");
    }
}

/// Brier Score: mean squared error of probabilistic predictions.
pub fn brier_score(predictions: &[f32], labels: &[f32]) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = predictions
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| ((p as f64) - (y as f64)).powi(2))
        .sum();
    sum / n as f64
}

/// Brier Skill Score: 1 - BS/BS_ref where BS_ref uses climatological frequency.
pub fn brier_skill_score(predictions: &[f32], labels: &[f32]) -> f64 {
    let bs = brier_score(predictions, labels);
    let mean_y: f64 = labels.iter().map(|&y| y as f64).sum::<f64>() / labels.len().max(1) as f64;
    let bs_ref: f64 = labels
        .iter()
        .map(|&y| (mean_y - y as f64).powi(2))
        .sum::<f64>()
        / labels.len().max(1) as f64;
    if bs_ref > 1e-10 {
        1.0 - bs / bs_ref
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let ds = SharpDataset::synthetic(10, 100, 42);
        assert!(!ds.train.is_empty());
        assert!(!ds.test.is_empty());
        assert_eq!(ds.train[0].features.len(), 10 * 9);
        assert!(ds.n_positive_train > 0);
        assert!(ds.n_negative_train > 0);
        ds.print_stats();
    }

    #[test]
    fn test_classification_metrics() {
        let preds = vec![0.9, 0.8, 0.3, 0.1, 0.7, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let m = ClassificationMetrics::compute(&preds, &labels, 0.5);
        assert_eq!(m.tp, 3);
        assert_eq!(m.tn, 3);
        assert_eq!(m.fp, 0);
        assert_eq!(m.fn_, 0);
        assert!((m.tss - 1.0).abs() < 1e-6, "Perfect TSS expected");
    }

    #[test]
    fn test_brier_score() {
        let preds = vec![1.0, 0.0, 1.0, 0.0];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let bs = brier_score(&preds, &labels);
        assert!(bs < 1e-6, "Perfect predictions should have BS ≈ 0: {}", bs);
    }

    #[test]
    fn test_flare_event_parsing() {
        let event = FlareEvent {
            start_time: "2024-01-01T00:00:00Z".to_string(),
            class: "M1.5".to_string(),
            noaa_ar: 13500,
        };
        assert!(event.is_c5_or_above());
        assert!((event.peak_flux().unwrap() - 1.5e-5).abs() < 1e-7);

        let quiet = FlareEvent {
            start_time: "2024-01-01T00:00:00Z".to_string(),
            class: "B5.0".to_string(),
            noaa_ar: 13500,
        };
        assert!(!quiet.is_c5_or_above());
    }

    #[test]
    fn test_norm_stats() {
        let data = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        ];
        let norm = NormStats::from_data(&data);
        assert!((norm.normalize(0, 1.0)).abs() < 1e-6); // min → 0
        assert!((norm.normalize(0, 10.0) - 1.0).abs() < 1e-6); // max → 1
        assert!((norm.normalize(0, 5.5) - 0.5).abs() < 1e-6); // mid → 0.5
    }
}

/// Convert a chrono DateTime<Utc> to Julian date.
// ── AR-level time series for streaming training ──────────────────────

/// A single 12-minute observation in an AR time series.
#[derive(Debug, Clone)]
pub struct ArObservation {
    /// Julian date (for orbital angle computation).
    pub jd: f64,
    /// Min-max normalized SHARP fields [0, 1].
    pub sharp_norm: [f32; 9],
    /// 1.0 if flare >= C5.0 within prediction window of this step.
    pub label: f32,
}

/// Full chronological time series for one active region.
#[derive(Debug, Clone)]
pub struct ArTimeSeries {
    pub harpnum: u32,
    pub observations: Vec<ArObservation>,
}

/// AR-level dataset for streaming training.
pub struct ArDataset {
    pub train_ars: Vec<ArTimeSeries>,
    pub test_ars: Vec<ArTimeSeries>,
    pub norm: NormStats,
}

impl ArDataset {
    /// Load SHARP data as per-AR chronological series.
    ///
    /// Reuses the same data pipeline as SharpDataset (log-transform, min-max normalization,
    /// AR-level train/test split by HARP hash) but returns full time series per AR
    /// instead of windowed samples.
    pub fn load_gzipped_dir(
        sharp_dir: &std::path::Path,
        flare_path: &std::path::Path,
        config: &DatasetConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load all CSV rows (reuse existing pipeline)
        let mut gz_files: Vec<std::path::PathBuf> = std::fs::read_dir(sharp_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "gz")
                && p.to_string_lossy().ends_with(".csv.gz"))
            .collect();
        gz_files.sort();

        let mut rows: Vec<RawSharpRow> = Vec::new();
        for gz_path in &gz_files {
            let file = std::fs::File::open(gz_path)?;
            let decoder = GzDecoder::new(file);
            let mut reader = csv::Reader::from_reader(decoder);
            for result in reader.deserialize() {
                if let Ok(row) = result { rows.push(row); }
            }
        }
        eprintln!("  Total SHARP records: {}", rows.len());

        // Parse flares
        let mut flare_reader = csv::Reader::from_path(flare_path)?;
        let mut flare_datetimes: Vec<DateTime<Utc>> = Vec::new();
        for result in flare_reader.deserialize() {
            let event: FlareEvent = result?;
            if event.is_c5_or_above() {
                if let Some(dt) = parse_flexible_time(&event.start_time) {
                    flare_datetimes.push(dt);
                }
            }
        }
        flare_datetimes.sort();

        // Group by HARP, sorted chronologically
        let mut by_harp: std::collections::HashMap<u32, Vec<(DateTime<Utc>, [f64; 9])>> = std::collections::HashMap::new();
        for row in &rows {
            let arr = row.to_array();
            if arr[2] <= 0.0 || arr[5] <= 0.0 { continue; } // skip invalid USFLUX/TOTPOT
            if let Some(dt) = parse_flexible_time(&row.time) {
                by_harp.entry(row.harpnum).or_default().push((dt, arr));
            }
        }
        for records in by_harp.values_mut() {
            records.sort_by_key(|r| r.0);
        }

        // Compute normalization from training ARs
        let mut harps: Vec<u32> = by_harp.keys().copied().collect();
        harps.sort();
        let mut train_harps = std::collections::HashSet::new();
        let mut test_harps = std::collections::HashSet::new();
        for &harp in &harps {
            let hash = (harp as u64).wrapping_mul(2654435761).wrapping_add(config.seed);
            if (hash % 1000) < (config.train_fraction * 1000.0) as u64 {
                train_harps.insert(harp);
            } else {
                test_harps.insert(harp);
            }
        }

        let train_rows: Vec<[f64; 9]> = rows.iter()
            .filter(|r| train_harps.contains(&r.harpnum))
            .map(|r| r.to_array())
            .collect();
        let all_rows: Vec<[f64; 9]> = rows.iter().map(|r| r.to_array()).collect();
        let norm = if train_rows.is_empty() {
            NormStats::from_data(&all_rows)
        } else {
            NormStats::from_data(&train_rows)
        };

        let window_duration = chrono::Duration::minutes(config.prediction_window_min as i64);

        // Build per-AR time series
        let mut train_ars = Vec::new();
        let mut test_ars = Vec::new();

        for (&harp, records) in &by_harp {
            if records.len() < 3 { continue; } // skip very short ARs

            let observations: Vec<ArObservation> = records.iter().map(|(dt, raw)| {
                let sharp_norm = norm.normalize_row(raw);
                let jd = datetime_to_jd(dt);
                let window_end = *dt + window_duration;
                let idx = flare_datetimes.partition_point(|ft| *ft < *dt);
                let has_flare = idx < flare_datetimes.len() && flare_datetimes[idx] <= window_end;
                ArObservation {
                    jd,
                    sharp_norm,
                    label: if has_flare { 1.0 } else { 0.0 },
                }
            }).collect();

            let ar = ArTimeSeries { harpnum: harp, observations };
            if train_harps.contains(&harp) {
                train_ars.push(ar);
            } else {
                test_ars.push(ar);
            }
        }

        // Sort by harpnum for reproducibility
        train_ars.sort_by_key(|a| a.harpnum);
        test_ars.sort_by_key(|a| a.harpnum);

        let train_obs: usize = train_ars.iter().map(|a| a.observations.len()).sum();
        let test_obs: usize = test_ars.iter().map(|a| a.observations.len()).sum();
        let train_pos: usize = train_ars.iter()
            .flat_map(|a| &a.observations).filter(|o| o.label > 0.5).count();
        let test_pos: usize = test_ars.iter()
            .flat_map(|a| &a.observations).filter(|o| o.label > 0.5).count();

        eprintln!("  Train: {} ARs, {} obs ({} positive, {:.1}%)",
            train_ars.len(), train_obs, train_pos,
            100.0 * train_pos as f64 / train_obs.max(1) as f64);
        eprintln!("  Test:  {} ARs, {} obs ({} positive, {:.1}%)",
            test_ars.len(), test_obs, test_pos,
            100.0 * test_pos as f64 / test_obs.max(1) as f64);

        Ok(ArDataset { train_ars, test_ars, norm })
    }
}

fn datetime_to_jd(dt: &DateTime<Utc>) -> f64 {
    use chrono::{Datelike, Timelike};
    let y = dt.year();
    let m = dt.month();
    let d = dt.day();
    let (y, m) = if m <= 2 { (y - 1, m + 12) } else { (y, m) };
    let yf = y as f64;
    let mf = m as f64;
    let a = (yf / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd = (365.25 * (yf + 4716.0)).floor()
        + (30.6001 * (mf + 1.0)).floor()
        + d as f64
        + b
        - 1524.5;
    // Add fractional day from time
    let h = dt.hour() as f64 + dt.minute() as f64 / 60.0 + dt.second() as f64 / 3600.0;
    jd + h / 24.0
}
