use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// DSCOVR/ACE solar wind plasma + magnetic field sample.
/// Source: https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json
///         https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarWindSample {
    pub time_tag: DateTime<Utc>,
    /// Solar wind bulk speed (km/s).
    pub speed: f64,
    /// IMF Bx (nT, GSM).
    pub bx: f64,
    /// IMF By (nT, GSM).
    pub by: f64,
    /// IMF Bz (nT, GSM).
    pub bz: f64,
    /// Proton density (n/cm^3).
    pub density: f64,
}

/// Fetch solar wind data from SWPC (merged plasma + mag).
///
/// SWPC provides these as separate JSON arrays where [0] is the header row
/// and subsequent rows are arrays of string values.
pub async fn fetch(client: &reqwest::Client) -> Result<Vec<SolarWindSample>, super::FeedError> {
    let (mag_result, plasma_result) = tokio::join!(fetch_mag(client), fetch_plasma(client),);

    let mag_data = mag_result?;
    let plasma_data = plasma_result?;

    // Merge by time_tag: mag has bx/by/bz, plasma has speed/density
    let mut samples = Vec::new();
    for m in &mag_data {
        // Find matching plasma entry (closest within 1 minute)
        if let Some(p) = plasma_data
            .iter()
            .find(|p| (m.time_tag - p.time_tag).num_seconds().unsigned_abs() < 120)
        {
            samples.push(SolarWindSample {
                time_tag: m.time_tag,
                speed: p.speed,
                bx: m.bx,
                by: m.by,
                bz: m.bz,
                density: p.density,
            });
        }
    }

    Ok(samples)
}

struct MagEntry {
    time_tag: DateTime<Utc>,
    bx: f64,
    by: f64,
    bz: f64,
}

struct PlasmaEntry {
    time_tag: DateTime<Utc>,
    speed: f64,
    density: f64,
}

async fn fetch_mag(client: &reqwest::Client) -> Result<Vec<MagEntry>, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "solar_wind_mag".into(),
        message: format!("HTTP error: {e}"),
    })?;

    // Format: array of arrays, first element is header
    // ["time_tag","bx_gsm","by_gsm","bz_gsm","lon_gsm","lat_gsm","bt"]
    let rows: Vec<Vec<serde_json::Value>> = resp.json().await.map_err(|e| super::FeedError {
        feed: "solar_wind_mag".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut entries = Vec::new();
    for row in rows.iter().skip(1) {
        if row.len() < 4 {
            continue;
        }
        let time_str = row[0].as_str().unwrap_or("");
        let time_tag = match parse_time(time_str) {
            Some(t) => t,
            None => continue,
        };
        let bx = parse_f64(&row[1]).unwrap_or(0.0);
        let by = parse_f64(&row[2]).unwrap_or(0.0);
        let bz = parse_f64(&row[3]).unwrap_or(0.0);
        entries.push(MagEntry {
            time_tag,
            bx,
            by,
            bz,
        });
    }

    Ok(entries)
}

async fn fetch_plasma(client: &reqwest::Client) -> Result<Vec<PlasmaEntry>, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "solar_wind_plasma".into(),
        message: format!("HTTP error: {e}"),
    })?;

    // Format: ["time_tag","density","speed","temperature"]
    let rows: Vec<Vec<serde_json::Value>> = resp.json().await.map_err(|e| super::FeedError {
        feed: "solar_wind_plasma".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut entries = Vec::new();
    for row in rows.iter().skip(1) {
        if row.len() < 3 {
            continue;
        }
        let time_str = row[0].as_str().unwrap_or("");
        let time_tag = match parse_time(time_str) {
            Some(t) => t,
            None => continue,
        };
        let density = parse_f64(&row[1]).unwrap_or(0.0);
        let speed = parse_f64(&row[2]).unwrap_or(0.0);
        entries.push(PlasmaEntry {
            time_tag,
            density,
            speed,
        });
    }

    Ok(entries)
}

fn parse_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_f64_from_string() {
        let v = serde_json::Value::String("3.14".into());
        assert!((parse_f64(&v).unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_parse_f64_from_number() {
        let v = serde_json::json!(42.0);
        assert!((parse_f64(&v).unwrap() - 42.0).abs() < 1e-10);
    }
}
