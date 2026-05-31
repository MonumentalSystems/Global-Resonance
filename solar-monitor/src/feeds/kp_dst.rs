use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Kp planetary geomagnetic index sample.
/// Source: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KpDstSample {
    pub time_tag: DateTime<Utc>,
    /// Kp index (0-9, can be fractional e.g. 3.33 for Kp=3+).
    pub kp: f64,
    /// Estimated Dst from Kp (Burton et al. empirical).
    /// Real Dst requires Kyoto WDC which has different latency.
    pub estimated_dst: f64,
}

/// Raw row as returned by the NOAA API (object format).
#[derive(Debug, Deserialize)]
struct KpRow {
    time_tag: String,
    #[serde(rename = "Kp")]
    kp: f64,
}

/// Fetch NOAA planetary K-index (3-hour cadence, ~8 points/day).
pub async fn fetch(client: &reqwest::Client) -> Result<Vec<KpDstSample>, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "kp_dst".into(),
        message: format!("HTTP error: {e}"),
    })?;

    // Format: array of objects {"time_tag","Kp","a_running","station_count"}
    let rows: Vec<KpRow> = resp.json().await.map_err(|e| super::FeedError {
        feed: "kp_dst".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut samples = Vec::new();
    for row in &rows {
        let time_tag = match parse_time(&row.time_tag) {
            Some(t) => t,
            None => continue,
        };
        let estimated_dst = estimate_dst_from_kp(row.kp);
        samples.push(KpDstSample {
            time_tag,
            kp: row.kp,
            estimated_dst,
        });
    }

    Ok(samples)
}

/// Rough empirical Dst from Kp (not a substitute for real Dst data).
/// Based on Kp-Dst statistical correlations.
fn estimate_dst_from_kp(kp: f64) -> f64 {
    if kp <= 2.0 {
        // Quiet: Dst ~ 0 to -20
        -5.0 * kp
    } else if kp <= 5.0 {
        // Moderate: Dst ~ -20 to -100
        -10.0 - 18.0 * (kp - 2.0)
    } else {
        // Storm: Dst drops steeply
        -64.0 - 40.0 * (kp - 5.0)
    }
}

fn parse_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dst_estimate_quiet() {
        let dst = estimate_dst_from_kp(1.0);
        assert!(dst > -20.0 && dst < 0.0);
    }

    #[test]
    fn test_dst_estimate_storm() {
        let dst = estimate_dst_from_kp(7.0);
        assert!(dst < -100.0);
    }
}
