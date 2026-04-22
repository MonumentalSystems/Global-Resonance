//! CSV loaders for historical solar/geomagnetic datasets.
//!
//! Loads data from the Geometric-Resonance-Papers repo:
//! - OMNI hourly (2000-2026): solar wind, Bz, Dst, AE
//! - Solar flares (2010-2026): GOES events with class/time
//! - Kp 3-hourly (1980-present): geomagnetic index
//! - Dst hourly (2021-present): ring current index

use chrono::{DateTime, NaiveDateTime, Utc};
use std::collections::BTreeMap;
use std::path::Path;

/// A unified historical record at a given timestamp.
/// Fields are Option because not all datasets have all channels.
#[derive(Debug, Clone)]
pub struct HistoricalRecord {
    pub timestamp: DateTime<Utc>,
    /// X-ray flux estimate (W/m^2). From flare catalog class_numeric,
    /// or quiet-sun background (~5e-7) when no flare active.
    pub xray_flux: f64,
    /// Solar wind speed (km/s). From OMNI.
    pub solar_wind_speed: Option<f64>,
    /// IMF Bz GSM (nT). From OMNI.
    pub bz: Option<f64>,
    /// IMF By GSM (nT). From OMNI.
    pub by: Option<f64>,
    /// Proton density (n/cm^3). From OMNI.
    pub density: Option<f64>,
    /// Dst index (nT). From OMNI or Dst file.
    pub dst: Option<f64>,
    /// Kp index. From Kp file.
    pub kp: Option<f64>,
    /// Whether a known flare is active at this time.
    pub flare_active: bool,
    /// Flare class if active (e.g., "M2.0", "X1.5").
    pub flare_class: Option<String>,
}

/// A known flare event (ground truth for evaluation).
#[derive(Debug, Clone)]
pub struct FlareEvent {
    pub begin: DateTime<Utc>,
    pub peak: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub class: String,
    /// Numeric class (M1.0 = 0.1, X1.0 = 1.0, etc.)
    pub class_numeric: f64,
}

/// Load solar flare catalog.
/// CSV format: beginTime,peakTime,endTime,classType,sourceLocation,activeRegionNum,class_numeric,day_number,hour
pub fn load_flares(path: &Path) -> Result<Vec<FlareEvent>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut flares = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        let begin = parse_datetime(fields[0])
            .ok_or_else(|| format!("Bad begin time line {}: {}", i + 1, fields[0]))?;
        let peak = parse_datetime(fields[1]).unwrap_or(begin);
        let end = parse_datetime(fields[2]).unwrap_or(begin);
        let class = fields[3].to_string();
        let class_numeric: f64 = fields[6].parse().unwrap_or(0.0);

        // Only keep M and X class flares for evaluation
        if class.starts_with('M') || class.starts_with('X') {
            flares.push(FlareEvent {
                begin,
                peak,
                end,
                class,
                class_numeric,
            });
        }
    }

    flares.sort_by_key(|f| f.begin);
    Ok(flares)
}

/// Load OMNI hourly data.
/// CSV: year,doy,hour,datetime,bz_gse,bz_gsm,by_gse,b_mag,v_sw,n_proton,dst,ae,day_number
pub fn load_omni(path: &Path) -> Result<BTreeMap<DateTime<Utc>, OmniRecord>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut records = BTreeMap::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 11 {
            continue;
        }

        let timestamp = match parse_datetime(fields[3]) {
            Some(t) => t,
            None => continue,
        };

        let bz_gsm: f64 = fields[5].parse().unwrap_or(f64::NAN);
        let by_gse: f64 = fields[6].parse().unwrap_or(f64::NAN);
        let v_sw: f64 = fields[8].parse().unwrap_or(f64::NAN);
        let n_proton: f64 = fields[9].parse().unwrap_or(f64::NAN);
        let dst: f64 = fields[10].parse().unwrap_or(f64::NAN);

        // Skip fill values (OMNI uses 9999.99 etc.)
        if v_sw > 9000.0 || bz_gsm.abs() > 900.0 {
            continue;
        }

        records.insert(
            timestamp,
            OmniRecord {
                timestamp,
                bz_gsm,
                by_gsm: by_gse, // approximate (GSE ≈ GSM for By to first order)
                v_sw,
                n_proton,
                dst,
            },
        );
    }

    Ok(records)
}

#[derive(Debug, Clone)]
pub struct OmniRecord {
    pub timestamp: DateTime<Utc>,
    pub bz_gsm: f64,
    pub by_gsm: f64,
    pub v_sw: f64,
    pub n_proton: f64,
    pub dst: f64,
}

/// Load Kp 3-hourly data.
/// CSV: year,month,day,hour,kp,ap,datetime,day_number,dkp_dt
pub fn load_kp(path: &Path) -> Result<BTreeMap<DateTime<Utc>, f64>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut records = BTreeMap::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        let timestamp = match parse_datetime(fields[6]) {
            Some(t) => t,
            None => continue,
        };
        let kp: f64 = fields[4].parse().unwrap_or(0.0);
        records.insert(timestamp, kp);
    }

    Ok(records)
}

/// Merge all datasets into a time-aligned sequence of HistoricalRecords.
///
/// Uses OMNI hourly timestamps as the primary timeline, interpolates
/// Kp (3-hourly) to hourly by holding constant, and overlays flare
/// active periods.
pub fn merge_datasets(
    omni: &BTreeMap<DateTime<Utc>, OmniRecord>,
    kp: &BTreeMap<DateTime<Utc>, f64>,
    flares: &[FlareEvent],
) -> Vec<HistoricalRecord> {
    let mut records = Vec::with_capacity(omni.len());

    // Pre-sort Kp for lookup
    let kp_entries: Vec<(DateTime<Utc>, f64)> = kp.iter().map(|(&t, &v)| (t, v)).collect();

    for (&timestamp, omni_rec) in omni {
        // Find nearest Kp value (hold previous)
        let kp_val = find_nearest_kp(&kp_entries, timestamp);

        // Check if a flare is active at this timestamp
        let active_flare = flares
            .iter()
            .find(|f| timestamp >= f.begin && timestamp <= f.end);
        let flare_active = active_flare.is_some();
        let flare_class = active_flare.map(|f| f.class.clone());

        // X-ray flux: during flare use class_numeric * 1e-4 (X-class scale),
        // otherwise quiet sun background
        let xray_flux = if let Some(flare) = active_flare {
            // class_numeric: M1.0 = 0.1, X1.0 = 1.0, etc.
            // Convert to W/m^2: class_numeric * 1e-4
            flare.class_numeric * 1e-4
        } else {
            // Quiet sun: B-class background
            5e-7
        };

        records.push(HistoricalRecord {
            timestamp,
            xray_flux,
            solar_wind_speed: if omni_rec.v_sw.is_nan() {
                None
            } else {
                Some(omni_rec.v_sw)
            },
            bz: if omni_rec.bz_gsm.is_nan() {
                None
            } else {
                Some(omni_rec.bz_gsm)
            },
            by: if omni_rec.by_gsm.is_nan() {
                None
            } else {
                Some(omni_rec.by_gsm)
            },
            density: if omni_rec.n_proton.is_nan() {
                None
            } else {
                Some(omni_rec.n_proton)
            },
            dst: if omni_rec.dst.is_nan() {
                None
            } else {
                Some(omni_rec.dst)
            },
            kp: kp_val,
            flare_active,
            flare_class,
        });
    }

    records.sort_by_key(|r| r.timestamp);
    records
}

fn find_nearest_kp(kp_entries: &[(DateTime<Utc>, f64)], timestamp: DateTime<Utc>) -> Option<f64> {
    // Binary search for nearest Kp entry at or before timestamp
    match kp_entries.binary_search_by_key(&timestamp, |(t, _)| *t) {
        Ok(idx) => Some(kp_entries[idx].1),
        Err(idx) => {
            if idx > 0 {
                // Use previous entry (hold-previous interpolation)
                let prev = &kp_entries[idx - 1];
                // Only use if within 6 hours
                if (timestamp - prev.0).num_hours() <= 6 {
                    Some(prev.1)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}

fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_datetime() {
        let t = parse_datetime("2010-04-03 09:04:00").unwrap();
        assert_eq!(t.year(), 2010);
    }

    use chrono::Datelike;

    #[test]
    fn test_load_flares_if_available() {
        let path = Path::new("solar-monitor/data/catalogs/solar_flares.csv");
        if path.exists() {
            let flares = load_flares(path).unwrap();
            assert!(!flares.is_empty());
            // All should be M or X class
            for f in &flares {
                assert!(f.class.starts_with('M') || f.class.starts_with('X'));
            }
            println!("Loaded {} M/X-class flares", flares.len());
        }
    }
}

// ── SHARP CSV loader ──────────────────────────────────────────────────────────

/// One SHARP record (12-min cadence, per active region).
#[derive(Debug, Clone)]
pub struct SharpCsvRecord {
    pub time_tag: DateTime<Utc>,
    pub harpnum: u32,
    pub usflux: f64,
    pub meangbz: f64,
    pub meanjzh: f64,
    pub totusjh: f64,
    pub shrgt45: f64,
    pub area_acr: f64,
    pub r_value: f64,
    pub totpot: f64,
    pub totusjz: f64,
    pub savncpp: f64,
    pub absnjzh: f64,
    pub meanalp: f64,
}

/// Load SHARP bulk CSV (output of fetch_sharp_bulk.py).
///
/// Returns a map from minute-truncated UTC timestamp → Vec of active region
/// records at that time. Multiple HARPs can be active simultaneously; the
/// caller picks the highest-risk one.
///
/// CSV columns: time_tag,harpnum,usflux,meangbz,meanjzh,totusjh,shrgt45,
///              area_acr,r_value,totpot,totusjz,savncpp,absnjzh,meanalp
pub fn load_sharp_csv(
    path: &Path,
) -> Result<std::collections::BTreeMap<DateTime<Utc>, Vec<SharpCsvRecord>>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut map: std::collections::BTreeMap<DateTime<Utc>, Vec<SharpCsvRecord>> =
        std::collections::BTreeMap::new();

    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let f: Vec<&str> = line.splitn(15, ',').collect();
        if f.len() < 14 {
            continue;
        }

        let t = match parse_datetime(f[0]) {
            Some(t) => truncate_to_minute(t),
            None => continue,
        };
        let harpnum: u32 = f[1].trim().parse().unwrap_or(0);
        if harpnum == 0 {
            continue;
        }

        let pf = |s: &str| -> f64 { s.trim().parse::<f64>().unwrap_or(f64::NAN) };

        let rec = SharpCsvRecord {
            time_tag: t,
            harpnum,
            usflux: pf(f[2]),
            meangbz: pf(f[3]),
            meanjzh: pf(f[4]),
            totusjh: pf(f[5]),
            shrgt45: pf(f[6]),
            area_acr: pf(f[7]),
            r_value: pf(f[8]),
            totpot: pf(f[9]),
            totusjz: pf(f[10]),
            savncpp: pf(f[11]),
            absnjzh: pf(f[12]),
            meanalp: pf(f[13]),
        };

        map.entry(t).or_default().push(rec);
    }

    Ok(map)
}

/// Find the highest-risk SHARP record within ±N minutes of `t`.
///
/// Returns None if no SHARP data exists within the window.
/// Risk metric: totpot + r_value ranking (same as sharp_flare_risk in feeds).
pub fn nearest_sharp(
    sharp_map: &std::collections::BTreeMap<DateTime<Utc>, Vec<SharpCsvRecord>>,
    t: DateTime<Utc>,
    window_minutes: i64,
) -> Option<SharpCsvRecord> {
    use chrono::Duration;
    let lo = t - Duration::minutes(window_minutes);
    let hi = t + Duration::minutes(window_minutes);

    let mut best: Option<SharpCsvRecord> = None;
    let mut best_risk = f64::NEG_INFINITY;

    for (_, recs) in sharp_map.range(lo..=hi) {
        for rec in recs {
            let risk = sharp_risk(rec);
            if risk > best_risk {
                best_risk = risk;
                best = Some(rec.clone());
            }
        }
    }
    best
}

fn sharp_risk(r: &SharpCsvRecord) -> f64 {
    let totpot_norm = if r.totpot > 0.0 && r.totpot.is_finite() {
        (r.totpot.log10() - 2.0).max(0.0) / 3.0
    } else {
        0.0
    };
    let r_value_norm = if r.r_value > 0.0 && r.r_value.is_finite() {
        (r.r_value.log10() - 1.0).max(0.0) / 3.0
    } else {
        0.0
    };
    let shear_norm = (r.shrgt45 / 50.0).min(1.0);
    0.40 * totpot_norm + 0.35 * r_value_norm + 0.25 * shear_norm
}

fn truncate_to_minute(t: DateTime<Utc>) -> DateTime<Utc> {
    use chrono::Timelike;
    t.with_second(0)
        .and_then(|t| t.with_nanosecond(0))
        .unwrap_or(t)
}
