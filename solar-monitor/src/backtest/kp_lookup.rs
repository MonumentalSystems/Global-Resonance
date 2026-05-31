//! Kp index lookup by Julian date.
//!
//! Loads kp_3hourly.csv and provides fast lookup of Kp and dKp/dt
//! for any timestamp. Uses binary search on sorted JD array.

use std::path::Path;

/// Kp lookup table — sorted by Julian date for binary search.
pub struct KpLookup {
    /// Julian dates (sorted ascending).
    jds: Vec<f64>,
    /// Kp values (0-9 scale), normalized to [0, 1] by dividing by 9.
    kp_norm: Vec<f32>,
    /// dKp/dt values, normalized to [-1, 1] by dividing by 9.
    dkp_norm: Vec<f32>,
}

impl KpLookup {
    /// Load from kp_3hourly.csv.
    /// Columns: year,month,day,hour,kp,ap,datetime,day_number,dkp_dt
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut jds = Vec::new();
        let mut kp_norm = Vec::new();
        let mut dkp_norm = Vec::new();

        let mut rdr = csv::Reader::from_path(path)?;
        for result in rdr.records() {
            let record = result?;
            // Parse year, month, day, hour
            let year: i32 = record.get(0).unwrap_or("2000").parse().unwrap_or(2000);
            let month: u32 = record.get(1).unwrap_or("1").parse().unwrap_or(1);
            let day: u32 = record.get(2).unwrap_or("1").parse().unwrap_or(1);
            let hour: u32 = record.get(3).unwrap_or("0").parse().unwrap_or(0);
            let kp: f64 = record.get(4).unwrap_or("0").parse().unwrap_or(0.0);
            let dkp: f64 = record.get(8).unwrap_or("0").parse().unwrap_or(0.0);

            let jd = date_hour_to_jd(year, month, day, hour);
            jds.push(jd);
            kp_norm.push((kp / 9.0) as f32);
            dkp_norm.push((dkp / 9.0).clamp(-1.0, 1.0) as f32);
        }

        println!("  Loaded {} Kp records ({:.0}-{:.0} JD)",
            jds.len(),
            jds.first().unwrap_or(&0.0),
            jds.last().unwrap_or(&0.0));

        Ok(KpLookup { jds, kp_norm, dkp_norm })
    }

    /// Look up Kp and dKp/dt at a given Julian date.
    /// Returns (kp_normalized, dkp_normalized) using nearest-earlier entry.
    pub fn lookup(&self, jd: f64) -> (f32, f32) {
        if self.jds.is_empty() {
            return (0.33, 0.0); // default: moderate, steady
        }
        // Binary search for nearest-earlier entry
        let idx = match self.jds.binary_search_by(|v| v.partial_cmp(&jd).unwrap()) {
            Ok(i) => i,
            Err(i) => if i > 0 { i - 1 } else { 0 },
        };
        (self.kp_norm[idx], self.dkp_norm[idx])
    }
}

fn date_hour_to_jd(year: i32, month: u32, day: u32, hour: u32) -> f64 {
    let (y, m) = if month <= 2 { (year - 1, month + 12) } else { (year, month) };
    let yf = y as f64;
    let mf = m as f64;
    let a = (yf / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    let jd = (365.25 * (yf + 4716.0)).floor()
        + (30.6001 * (mf + 1.0)).floor()
        + day as f64
        + b
        - 1524.5;
    jd + hour as f64 / 24.0
}
