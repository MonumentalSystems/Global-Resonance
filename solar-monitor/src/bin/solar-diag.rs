use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use solar_monitor::detection::energy::EnergyDetector;
use solar_monitor::detection::rate_of_change::RateOfChangeDetector;
use solar_monitor::feeds::xray::FlareClass;
use std::collections::BTreeMap;

fn main() {
    let xray = load_xray("/tmp/xrays-7-day.json");
    let electrons = load_electrons("/tmp/electrons-7day.json");
    let elec_vec: Vec<(DateTime<Utc>, f64)> = electrons.iter().map(|(&t, &v)| (t, v)).collect();

    let mut energy = EnergyDetector::new(30);
    let mut roc = RateOfChangeDetector::new(8, 0.10);

    // Focus on the X1.5 window: 02:30 - 04:00 on March 30
    let window_start = parse_ts("2026-03-30T02:30:00Z");
    let window_end = parse_ts("2026-03-30T04:00:00Z");

    println!("=== Energy + RoC Detail During X1.5 Flare ===\n");
    println!(
        "{:<18} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "Time",
        "Xray",
        "Electron",
        "ch_xray",
        "ch_elec",
        "ch_ratio",
        "E_score",
        "RoC_rate",
        "RoC_scr",
        "Class"
    );
    println!("{}", "-".repeat(115));

    for (&ts, &flux) in &xray {
        if flux < 1e-9 {
            continue;
        }
        let electron = find_nearest(&elec_vec, ts).unwrap_or(100.0);

        energy.ingest(flux, electron, ts);
        roc.ingest(flux, ts);

        if ts >= window_start && ts <= window_end {
            // Compute channels manually for display
            let ch_xray = flux.log10();
            let ch_electron = electron.log10();
            let ch_ratio = (electron / (flux * 1e9)).max(1e-6).log10();

            println!(
                "{} {:>10.2e} {:>10.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.4} {:>8.3} {:>6}",
                ts.format("%m-%d %H:%M"),
                flux,
                electron,
                ch_xray,
                ch_electron,
                ch_ratio,
                energy.score(),
                roc.rate(),
                roc.score(),
                FlareClass::from_flux(flux).label()
            );
        }
    }
}

fn load_xray(path: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct Rec {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<Rec> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for r in data {
        if r.energy.as_deref() != Some("0.1-0.8nm") {
            continue;
        }
        if let (Some(ts), Some(flux)) = (parse_ts_opt(&r.time_tag), r.flux) {
            if flux > 0.0 {
                map.insert(ts, flux);
            }
        }
    }
    map
}

fn load_electrons(path: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct Rec {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<Rec> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut map = BTreeMap::new();
    for r in data {
        if r.energy.as_deref() != Some(">=2 MeV") {
            continue;
        }
        if let (Some(ts), Some(flux)) = (parse_ts_opt(&r.time_tag), r.flux) {
            map.insert(ts, flux);
        }
    }
    map
}

fn find_nearest(entries: &[(DateTime<Utc>, f64)], ts: DateTime<Utc>) -> Option<f64> {
    match entries.binary_search_by_key(&ts, |(t, _)| *t) {
        Ok(idx) => Some(entries[idx].1),
        Err(idx) if idx > 0 => {
            if (ts - entries[idx - 1].0).num_seconds() < 600 {
                Some(entries[idx - 1].1)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    parse_ts_opt(s).unwrap()
}
fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .ok()
        .map(|ndt| ndt.and_utc())
}
