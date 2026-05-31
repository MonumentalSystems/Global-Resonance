//! Download GOES-15 XRS 2-second CSV data from NCEI and aggregate to 1-minute
//! JSON matching the SWPC real-time format. Downloads a window of days around
//! specified events for backtesting with real X-ray data.
//!
//! Output: /tmp/goes-xrs-{event_label}.json in SWPC format.
//!
//! NCEI URL pattern (GOES-15, 2010-2020):
//!   https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/full/
//!     {YYYY}/{MM}/goes15/csv/g15_xrs_2s_{YYYYMMDD}_{YYYYMMDD}.csv
//!
//! NOAA scaling factors (must be removed for true flux):
//!   A_FLUX (short): divide by 0.85
//!   B_FLUX (long): divide by 0.7

use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Write;

/// Events to download data for (date, label, days_before, days_after).
const EVENTS: &[(&str, &str, i64, i64)] = &[
    // Top X-class events from backtest (GOES-15 era: 2010-2020)
    ("2011-09-22", "x1.4_2011sep22", 8, 2),
    ("2011-03-09", "x1.5_2011mar09", 8, 2),
    ("2013-11-10", "x1.1_2013nov10", 8, 2),
    ("2013-05-13", "x1.6_2013may13", 8, 2),
    ("2011-09-24", "x1.9_2011sep24", 8, 2),
    ("2014-03-29", "x1.0_2014mar29", 8, 2),
    ("2011-02-15", "x2.2_2011feb15", 8, 2),
    ("2012-03-07", "x5.4_2012mar07", 8, 2),
    ("2017-09-06", "x9.3_2017sep06", 8, 2),
    ("2017-09-10", "x8.2_2017sep10", 8, 2),
];

const NCEI_BASE: &str = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/full";

#[derive(Serialize)]
struct XrayRecord {
    time_tag: String,
    energy: String,
    flux: f64,
}

fn main() {
    for &(date_str, label, days_before, days_after) in EVENTS {
        let center = NaiveDate::parse_from_str(date_str, "%Y-%m-%d").unwrap();
        let start = center - Duration::days(days_before);
        let end = center + Duration::days(days_after);

        println!("=== Fetching {} ({} to {}) ===", label, start, end);

        let mut all_records: Vec<XrayRecord> = Vec::new();
        let mut day = start;

        while day <= end {
            let url = format!(
                "{}/{:04}/{:02}/goes15/csv/g15_xrs_2s_{:04}{:02}{:02}_{:04}{:02}{:02}.csv",
                NCEI_BASE,
                day.year(),
                day.month(),
                day.year(),
                day.month(),
                day.day(),
                day.year(),
                day.month(),
                day.day(),
            );

            eprint!("  Fetching {} ... ", day);

            match download_and_parse(&url) {
                Ok(records) => {
                    eprintln!("{} records", records.len());
                    // Aggregate 2-sec to 1-min averages.
                    let minute_avgs = aggregate_to_minutes(&records);
                    all_records.extend(minute_avgs);
                }
                Err(e) => {
                    eprintln!("SKIP ({})", e);
                }
            }

            day += Duration::days(1);
        }

        // Write output.
        let out_path = format!("/tmp/goes-xrs-{}.json", label);
        let json = serde_json::to_string(&all_records).unwrap();
        let mut f = std::fs::File::create(&out_path).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        println!("  Wrote {} records to {}\n", all_records.len(), out_path);
    }

    println!("Done! Files in /tmp/goes-xrs-*.json");
}

/// Raw 2-second record parsed from CSV.
struct RawXrs {
    time_tag: String,
    a_flux: f64, // short channel (0.05-0.4nm)
    b_flux: f64, // long channel (0.1-0.8nm)
}

fn download_and_parse(url: &str) -> Result<Vec<RawXrs>, String> {
    let resp = reqwest::blocking::get(url).map_err(|e| format!("HTTP error: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let body = resp.text().map_err(|e| format!("Read error: {}", e))?;

    // Find the data section (after "data:" line).
    let mut in_data = false;
    let mut records = Vec::new();

    for line in body.lines() {
        if line.starts_with("data:") {
            in_data = true;
            continue;
        }
        if !in_data {
            continue;
        }
        // Skip header line.
        if line.starts_with("time_tag,") {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }

        let a_flux: f64 = fields[3].trim().parse().unwrap_or(-1.0);
        let b_flux: f64 = fields[6].trim().parse().unwrap_or(-1.0);

        if a_flux < 0.0 || b_flux < 0.0 {
            continue;
        }

        // Remove NOAA scaling factors for true flux.
        let a_true = a_flux / 0.85;
        let b_true = b_flux / 0.7;

        records.push(RawXrs {
            time_tag: fields[0].to_string(),
            a_flux: a_true,
            b_flux: b_true,
        });
    }

    Ok(records)
}

fn aggregate_to_minutes(records: &[RawXrs]) -> Vec<XrayRecord> {
    // Group by minute (truncate seconds).
    let mut long_bins: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut short_bins: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for r in records {
        // Truncate to minute: "2015-05-01 00:00:02.120" → "2015-05-01 00:00:00"
        let minute_key = if r.time_tag.len() >= 16 {
            format!("{}:00", &r.time_tag[..16])
        } else {
            continue;
        };

        long_bins
            .entry(minute_key.clone())
            .or_default()
            .push(r.b_flux);
        short_bins.entry(minute_key).or_default().push(r.a_flux);
    }

    let mut output = Vec::new();

    for (minute, long_vals) in &long_bins {
        let long_mean = long_vals.iter().sum::<f64>() / long_vals.len() as f64;
        output.push(XrayRecord {
            time_tag: minute.clone(),
            energy: "0.1-0.8nm".to_string(),
            flux: long_mean,
        });

        if let Some(short_vals) = short_bins.get(minute) {
            let short_mean = short_vals.iter().sum::<f64>() / short_vals.len() as f64;
            output.push(XrayRecord {
                time_tag: minute.clone(),
                energy: "0.05-0.4nm".to_string(),
                flux: short_mean,
            });
        }
    }

    output
}
