//! Synthesize minute-cadence X-ray flux from the flare catalog.
//!
//! Uses the impulsive flare model from the Geometric-Resonance-Papers
//! (xray_events.py): linear rise from begin→peak, exponential decay
//! from peak→end, with quiet-sun background between events.
//!
//! This gives us 1-minute resolution across the full 2010-2026 flare
//! catalog, enabling the rate-of-change and energy detectors to work.

use chrono::{DateTime, Duration, Utc};
use std::collections::BTreeMap;

use super::loaders::{FlareEvent, HistoricalRecord, OmniRecord};

/// Quiet-sun X-ray background (mid-B class).
const QUIET_BG: f64 = 5e-7;

/// Generate minute-cadence synthetic X-ray flux from flare catalog.
///
/// For each flare: linear rise (begin→peak), exponential decay (peak→end).
/// Between flares: quiet background. Overlapping flares are summed.
///
/// Returns a BTreeMap of timestamp → flux for the full time range.
pub fn synthesize_xray_minute(
    flares: &[FlareEvent],
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> BTreeMap<DateTime<Utc>, f64> {
    let mut flux_map: BTreeMap<DateTime<Utc>, f64> = BTreeMap::new();

    // Initialize with quiet background at 1-min cadence
    let mut t = start;
    while t <= end {
        flux_map.insert(t, QUIET_BG);
        t = t + Duration::minutes(1);
    }

    // Overlay each flare's profile
    for flare in flares {
        if flare.begin > end || flare.end < start {
            continue;
        }

        let peak_flux = flare.class_numeric * 1e-4; // class_numeric: M1=0.1, X1=1.0
        if peak_flux <= QUIET_BG {
            continue;
        }

        let rise_secs = (flare.peak - flare.begin).num_seconds() as f64;
        let decay_secs = (flare.end - flare.peak).num_seconds() as f64;

        // Exponential decay time constant
        let tau = if decay_secs > 0.0 && peak_flux > QUIET_BG * 3.0 {
            decay_secs / (peak_flux / (QUIET_BG * 3.0).max(1e-8)).ln()
        } else {
            300.0 // fallback: 5-minute e-folding
        };

        // Generate profile at 1-min cadence
        let flare_start = flare.begin.max(start);
        let flare_end = (flare.end + Duration::minutes(30)).min(end); // extend tail
        let mut ft = flare_start;

        while ft <= flare_end {
            let flux = if ft < flare.begin {
                0.0
            } else if ft <= flare.peak {
                // Linear rise
                if rise_secs > 0.0 {
                    let frac = (ft - flare.begin).num_seconds() as f64 / rise_secs;
                    QUIET_BG + (peak_flux - QUIET_BG) * frac
                } else {
                    peak_flux
                }
            } else {
                // Exponential decay
                let dt = (ft - flare.peak).num_seconds() as f64;
                QUIET_BG + (peak_flux - QUIET_BG) * (-dt / tau).exp()
            };

            // Add to existing (handles overlapping flares)
            if let Some(existing) = flux_map.get_mut(&ft) {
                // Sum flare contribution above background
                *existing = (*existing - QUIET_BG).max(0.0) + flux;
            } else {
                flux_map.insert(ft, flux);
            }

            ft = ft + Duration::minutes(1);
        }
    }

    flux_map
}

/// Merge minute-cadence synthetic X-ray with hourly OMNI into
/// minute-cadence HistoricalRecords.
///
/// OMNI values are held constant (step interpolation) within each hour.
/// Kp is held constant within each 3-hour window.
pub fn merge_minute_cadence(
    xray_minute: &BTreeMap<DateTime<Utc>, f64>,
    omni: &BTreeMap<DateTime<Utc>, OmniRecord>,
    kp: &BTreeMap<DateTime<Utc>, f64>,
    flares: &[FlareEvent],
) -> Vec<HistoricalRecord> {
    let omni_entries: Vec<(&DateTime<Utc>, &OmniRecord)> = omni.iter().collect();
    let kp_entries: Vec<(DateTime<Utc>, f64)> = kp.iter().map(|(&t, &v)| (t, v)).collect();

    let mut records = Vec::with_capacity(xray_minute.len());

    for (&timestamp, &xray_flux) in xray_minute {
        // Find nearest OMNI entry (hourly, hold-previous)
        let omni_rec = find_nearest_omni(&omni_entries, timestamp);

        // Find nearest Kp
        let kp_val = find_nearest(&kp_entries, timestamp, 6 * 3600);

        // Check flare active
        let active_flare = flares
            .iter()
            .find(|f| timestamp >= f.begin && timestamp <= f.end);

        records.push(HistoricalRecord {
            timestamp,
            xray_flux,
            solar_wind_speed: omni_rec.map(|o| o.v_sw).filter(|v| !v.is_nan()),
            bz: omni_rec.map(|o| o.bz_gsm).filter(|v| !v.is_nan()),
            by: omni_rec.map(|o| o.by_gsm).filter(|v| !v.is_nan()),
            density: omni_rec.map(|o| o.n_proton).filter(|v| !v.is_nan()),
            dst: omni_rec.map(|o| o.dst).filter(|v| !v.is_nan()),
            kp: kp_val,
            flare_active: active_flare.is_some(),
            flare_class: active_flare.map(|f| f.class.clone()),
        });
    }

    records
}

fn find_nearest_omni<'a>(
    entries: &[(&DateTime<Utc>, &'a OmniRecord)],
    timestamp: DateTime<Utc>,
) -> Option<&'a OmniRecord> {
    // Binary search for nearest entry at or before timestamp
    match entries.binary_search_by_key(&timestamp, |(t, _)| **t) {
        Ok(idx) => Some(entries[idx].1),
        Err(idx) => {
            if idx > 0 {
                let prev = entries[idx - 1];
                // Within 2 hours
                if (timestamp - *prev.0).num_seconds() <= 7200 {
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

fn find_nearest(
    entries: &[(DateTime<Utc>, f64)],
    timestamp: DateTime<Utc>,
    max_gap_secs: i64,
) -> Option<f64> {
    match entries.binary_search_by_key(&timestamp, |(t, _)| *t) {
        Ok(idx) => Some(entries[idx].1),
        Err(idx) => {
            if idx > 0 {
                let prev = &entries[idx - 1];
                if (timestamp - prev.0).num_seconds() <= max_gap_secs {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ndt(s: &str) -> DateTime<Utc> {
        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .unwrap()
            .and_utc()
    }

    #[test]
    fn test_synthesize_single_flare() {
        let flare = FlareEvent {
            begin: ndt("2024-01-01 12:00:00"),
            peak: ndt("2024-01-01 12:10:00"),
            end: ndt("2024-01-01 12:30:00"),
            class: "M5.0".into(),
            class_numeric: 0.5,
        };

        let start = ndt("2024-01-01 11:50:00");
        let end = ndt("2024-01-01 13:00:00");

        let flux = synthesize_xray_minute(&[flare.clone()], start, end);

        // Before flare: quiet
        let pre_flux = flux[&start];
        assert!((pre_flux - QUIET_BG).abs() < 1e-9);

        // At peak: should be near M5.0 = 5e-5
        let peak_flux = flux[&flare.peak];
        assert!(
            peak_flux > 1e-5,
            "Peak flux should be > 1e-5, got {}",
            peak_flux
        );

        // After end: decaying toward background
        let post = ndt("2024-01-01 12:50:00");
        let post_flux = flux[&post];
        assert!(post_flux < peak_flux, "Post-flare should be less than peak");
        assert!(
            post_flux > QUIET_BG,
            "Post-flare should still be above background"
        );
    }

    #[test]
    fn test_minute_count() {
        let start = ndt("2024-01-01 00:00:00");
        let end = ndt("2024-01-01 01:00:00");
        let flux = synthesize_xray_minute(&[], start, end);
        assert_eq!(flux.len(), 61); // 0..60 inclusive
    }
}
