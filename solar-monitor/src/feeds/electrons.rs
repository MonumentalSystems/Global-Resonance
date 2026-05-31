use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// GOES >2 MeV integral electron flux sample.
/// Source: https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronSample {
    pub time_tag: DateTime<Utc>,
    pub satellite: u16,
    /// Integral electron flux (>2 MeV) in pfu (particles/cm^2/s/sr).
    pub flux: f64,
}

#[derive(Debug, Deserialize)]
struct SwpcElectronEntry {
    time_tag: String,
    satellite: Option<u16>,
    flux: Option<f64>,
}

/// Fetch GOES integral electron flux 1-day data.
pub async fn fetch(client: &reqwest::Client) -> Result<Vec<ElectronSample>, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "electrons".into(),
        message: format!("HTTP error: {e}"),
    })?;

    let entries: Vec<SwpcElectronEntry> = resp.json().await.map_err(|e| super::FeedError {
        feed: "electrons".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut samples = Vec::with_capacity(entries.len());
    for entry in entries {
        let time_tag = parse_time(&entry.time_tag).ok_or_else(|| super::FeedError {
            feed: "electrons".into(),
            message: format!("Bad time_tag: {}", entry.time_tag),
        })?;
        if let Some(flux) = entry.flux {
            samples.push(ElectronSample {
                time_tag,
                satellite: entry.satellite.unwrap_or(0),
                flux,
            });
        }
    }

    Ok(samples)
}

fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time() {
        assert!(parse_time("2026-04-02 13:45:00.000").is_some());
    }
}
