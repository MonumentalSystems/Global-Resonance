use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// GOES >=1 MeV integral proton flux sample.
/// Source: https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json
/// 5-minute cadence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonSample {
    pub time_tag: DateTime<Utc>,
    pub satellite: u16,
    /// Integral proton flux (>=1 MeV) in pfu (particles/cm^2/s/sr).
    pub flux: f64,
}

#[derive(Debug, Deserialize)]
struct SwpcProtonEntry {
    time_tag: String,
    satellite: Option<u16>,
    flux: Option<f64>,
    energy: Option<String>,
}

/// Fetch GOES integral proton flux (>=1 MeV channel only).
pub async fn fetch(client: &reqwest::Client) -> Result<Vec<ProtonSample>, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "protons".into(),
        message: format!("HTTP error: {e}"),
    })?;

    let entries: Vec<SwpcProtonEntry> = resp.json().await.map_err(|e| super::FeedError {
        feed: "protons".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut samples = Vec::new();
    for entry in entries {
        // Only >=1 MeV channel (most responsive to flares)
        if entry.energy.as_deref() != Some(">=1 MeV") {
            continue;
        }
        let time_tag = match parse_time(&entry.time_tag) {
            Some(t) => t,
            None => continue,
        };
        if let Some(flux) = entry.flux {
            samples.push(ProtonSample {
                time_tag,
                satellite: entry.satellite.unwrap_or(0),
                flux,
            });
        }
    }

    Ok(samples)
}

fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .ok()
        .map(|ndt| ndt.and_utc())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time() {
        assert!(parse_time("2026-04-02T13:45:00Z").is_some());
        assert!(parse_time("2026-04-02 13:45:00.000").is_some());
    }
}
