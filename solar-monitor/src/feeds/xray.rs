use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// GOES X-ray flux 1-min sample.
/// Source: https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XraySample {
    pub time_tag: DateTime<Utc>,
    pub satellite: u16,
    /// Current flux in W/m^2 (1-8 Angstrom, long channel).
    pub flux: f64,
    /// Current X-ray class string (e.g. "B3.2", "M1.5", "X2.1").
    #[serde(default)]
    pub current_class: Option<String>,
}

/// Flare class thresholds in W/m^2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FlareClass {
    A,
    B,
    C,
    M,
    X,
}

impl FlareClass {
    pub fn from_flux(flux_wm2: f64) -> Self {
        if flux_wm2 >= 1e-4 {
            FlareClass::X
        } else if flux_wm2 >= 1e-5 {
            FlareClass::M
        } else if flux_wm2 >= 1e-6 {
            FlareClass::C
        } else if flux_wm2 >= 1e-7 {
            FlareClass::B
        } else {
            FlareClass::A
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            FlareClass::A => "A",
            FlareClass::B => "B",
            FlareClass::C => "C",
            FlareClass::M => "M",
            FlareClass::X => "X",
        }
    }
}

/// Short X-ray (0.05-0.4nm) sample — same structure, separate ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrayShortSample {
    pub time_tag: DateTime<Utc>,
    pub satellite: u16,
    /// Flux in W/m^2 (0.05-0.4 Angstrom, short/hard channel).
    pub flux: f64,
}

/// Raw JSON shape from SWPC (fields vary — we pick what we need).
#[derive(Debug, Deserialize)]
struct SwpcXrayEntry {
    time_tag: String,
    satellite: Option<u16>,
    flux: Option<f64>,
    current_class: Option<String>,
    energy: Option<String>,
}

/// Both X-ray channels fetched together (they're interleaved in the JSON).
pub struct XrayBothChannels {
    pub long: Vec<XraySample>,
    pub short: Vec<XrayShortSample>,
}

/// Fetch GOES X-ray 1-day data from SWPC (both channels).
pub async fn fetch(client: &reqwest::Client) -> Result<Vec<XraySample>, super::FeedError> {
    let both = fetch_both(client).await?;
    Ok(both.long)
}

/// Fetch both X-ray channels (long 0.1-0.8nm + short 0.05-0.4nm).
pub async fn fetch_both(client: &reqwest::Client) -> Result<XrayBothChannels, super::FeedError> {
    let url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json";
    let resp = client.get(url).send().await.map_err(|e| super::FeedError {
        feed: "xray".into(),
        message: format!("HTTP error: {e}"),
    })?;

    let entries: Vec<SwpcXrayEntry> = resp.json().await.map_err(|e| super::FeedError {
        feed: "xray".into(),
        message: format!("JSON parse error: {e}"),
    })?;

    let mut long = Vec::new();
    let mut short = Vec::new();

    for entry in entries {
        let time_tag = parse_swpc_time(&entry.time_tag).ok_or_else(|| super::FeedError {
            feed: "xray".into(),
            message: format!("Bad time_tag: {}", entry.time_tag),
        })?;
        let flux = match entry.flux {
            Some(f) if f > 0.0 => f,
            _ => continue,
        };
        let sat = entry.satellite.unwrap_or(0);

        match entry.energy.as_deref() {
            Some("0.1-0.8nm") | None => {
                long.push(XraySample {
                    time_tag,
                    satellite: sat,
                    flux,
                    current_class: entry.current_class,
                });
            }
            Some("0.05-0.4nm") => {
                short.push(XrayShortSample {
                    time_tag,
                    satellite: sat,
                    flux,
                });
            }
            _ => {}
        }
    }

    Ok(XrayBothChannels { long, short })
}

/// Parse SWPC time_tag format. Handles both legacy and current formats:
/// - "2026-04-02 12:00:00.000" (legacy)
/// - "2026-04-01T21:48:00Z" (current ISO 8601)
fn parse_swpc_time(s: &str) -> Option<DateTime<Utc>> {
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
    fn test_flare_class() {
        assert_eq!(FlareClass::from_flux(2e-4), FlareClass::X);
        assert_eq!(FlareClass::from_flux(5e-5), FlareClass::M);
        assert_eq!(FlareClass::from_flux(3e-6), FlareClass::C);
        assert_eq!(FlareClass::from_flux(5e-7), FlareClass::B);
        assert_eq!(FlareClass::from_flux(1e-8), FlareClass::A);
    }

    #[test]
    fn test_parse_swpc_time() {
        let t = parse_swpc_time("2026-04-02 12:00:00.000").unwrap();
        assert_eq!(t.year(), 2026);
        let t2 = parse_swpc_time("2026-04-02 12:00:00").unwrap();
        assert_eq!(chrono::Timelike::minute(&t2), 0);
    }

    use chrono::Datelike;
}
