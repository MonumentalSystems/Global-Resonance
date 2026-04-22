//! Loader for real GOES-16 XRS dual-channel CSV data.
//!
//! CSV format (from fetch_goes_year.py):
//! time_tag,xrsa_flux,xrsb_flux
//! 2024-01-01 00:00:00,3.45e-08,5.67e-07

use chrono::{DateTime, NaiveDateTime, Utc};
use std::path::Path;

/// A single GOES XRS dual-channel record.
#[derive(Debug, Clone)]
pub struct GoesXrsRecord {
    pub timestamp: DateTime<Utc>,
    /// Short channel flux (0.05-0.4nm) in W/m^2.
    pub xrsa: f64,
    /// Long channel flux (0.1-0.8nm) in W/m^2.
    pub xrsb: f64,
}

/// Load GOES XRS dual-channel CSV.
pub fn load_goes_csv(path: &Path) -> Result<Vec<GoesXrsRecord>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut records = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 {
            continue; // skip header
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 3 {
            continue;
        }

        let timestamp = match NaiveDateTime::parse_from_str(fields[0].trim(), "%Y-%m-%d %H:%M:%S") {
            Ok(ndt) => ndt.and_utc(),
            Err(_) => continue,
        };

        let xrsa: f64 = match fields[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let xrsb: f64 = match fields[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        if xrsa > 0.0 && xrsb > 0.0 {
            records.push(GoesXrsRecord {
                timestamp,
                xrsa,
                xrsb,
            });
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_goes_record() {
        let csv = "time_tag,xrsa_flux,xrsb_flux\n2024-01-01 00:00:00,3.45e-08,5.67e-07\n";
        let tmp = std::env::temp_dir().join("test_goes.csv");
        std::fs::write(&tmp, csv).unwrap();
        let records = load_goes_csv(&tmp).unwrap();
        assert_eq!(records.len(), 1);
        assert!((records[0].xrsa - 3.45e-8).abs() < 1e-12);
        assert!((records[0].xrsb - 5.67e-7).abs() < 1e-12);
        std::fs::remove_file(&tmp).ok();
    }
}
