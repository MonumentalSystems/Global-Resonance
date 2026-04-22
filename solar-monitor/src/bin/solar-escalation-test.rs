use chrono::{DateTime, NaiveDateTime, Timelike, Utc};
use serde::Deserialize;
use solar_monitor::detection::escalation::EscalationMonitor;
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use std::collections::BTreeMap;

fn main() {
    let xray_long = load_channel("/tmp/xrays-7-day.json", "0.1-0.8nm");
    let xray_short = load_channel("/tmp/xrays-7-day.json", "0.05-0.4nm");
    let electrons = load_channel_generic("/tmp/electrons-7day.json", ">=2 MeV");
    let protons = load_channel_generic("/tmp/protons-7day.json", ">=1 MeV");
    let electrons_smooth = ema_smooth(&electrons, 0.15);
    let protons_smooth = ema_smooth(&protons, 0.15);

    let mut fusion = RankFusionDetector::new(0.7);
    let mut esc = EscalationMonitor::new();

    println!("=== Escalation Timeline (7-day real data) ===\n");
    println!(
        "{:<18} {:>10} {:>8} {:>8} {:>5} {:>8} {}",
        "Time", "Level", "Hardness", "Fused", "Agree", "Xray", "Transition"
    );
    println!("{}", "-".repeat(90));

    for (&ts, &long_flux) in &xray_long {
        if long_flux < 1e-9 {
            continue;
        }
        let short_flux = find_nearest_val(&xray_short, ts).unwrap_or(long_flux * 0.04);
        let electron = find_nearest_smooth(&electrons_smooth, ts).unwrap_or(100.0);
        let proton = find_nearest_smooth(&protons_smooth, ts).unwrap_or(0.3);
        fusion.ingest(long_flux, short_flux, electron, proton, ts);

        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();
        let old_level = esc.level;

        if let Some(transition) = esc.update(h, f, a, ts) {
            println!(
                "{} {:>10} {:>8.3} {:>8.3} {:>3}/6 {:>8.2e} << {} -> {} : {}",
                ts.format("%m-%d %H:%M"),
                transition.to.label(),
                h,
                f,
                a,
                long_flux,
                transition.from.label(),
                transition.to.label(),
                transition.reason
            );
        }
    }

    let status = esc.status(chrono::Utc::now());
    println!(
        "\nFinal state: {} (since {:?})",
        status.level_label, status.since
    );
    println!(
        "Hardness spikes in window: {}",
        status.hardness_spikes_in_window
    );
    println!("Peak hardness: {:.3}", status.peak_hardness);
    println!("Peak fused: {:.3}", status.peak_fused);
    println!("Flare triggers: {}", status.flare_triggers);
}

fn load_channel(path: &str, energy: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct R {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<R> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    data.iter()
        .filter(|r| r.energy.as_deref() == Some(energy))
        .filter_map(|r| Some((parse_ts_opt(&r.time_tag)?, r.flux.filter(|&f| f > 0.0)?)))
        .collect()
}
fn load_channel_generic(path: &str, energy: &str) -> BTreeMap<DateTime<Utc>, f64> {
    #[derive(Deserialize)]
    struct R {
        time_tag: String,
        flux: Option<f64>,
        energy: Option<String>,
    }
    let data: Vec<R> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    data.iter()
        .filter(|r| r.energy.as_deref() == Some(energy))
        .filter_map(|r| Some((parse_ts_opt(&r.time_tag)?, r.flux?)))
        .collect()
}
fn ema_smooth(data: &BTreeMap<DateTime<Utc>, f64>, alpha: f64) -> Vec<(DateTime<Utc>, f64)> {
    let mut r = Vec::new();
    let mut e = None;
    for (&t, &v) in data {
        let l = if v > 0.0 { v.log10() } else { -2.0 };
        let s = match e {
            None => {
                e = Some(l);
                l
            }
            Some(p) => {
                let s = alpha * l + (1.0 - alpha) * p;
                e = Some(s);
                s
            }
        };
        r.push((t, 10.0_f64.powf(s)));
    }
    r
}
fn find_nearest_val(m: &BTreeMap<DateTime<Utc>, f64>, ts: DateTime<Utc>) -> Option<f64> {
    m.range(..=ts).next_back().and_then(|(&t, &v)| {
        if (ts - t).num_seconds() < 120 {
            Some(v)
        } else {
            None
        }
    })
}
fn find_nearest_smooth(e: &[(DateTime<Utc>, f64)], ts: DateTime<Utc>) -> Option<f64> {
    match e.binary_search_by_key(&ts, |(t, _)| *t) {
        Ok(i) => Some(e[i].1),
        Err(i) if i > 0 => {
            if (ts - e[i - 1].0).num_seconds() < 600 {
                Some(e[i - 1].1)
            } else {
                None
            }
        }
        _ => None,
    }
}
fn parse_ts_opt(s: &str) -> Option<DateTime<Utc>> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.fZ"))
        .ok()
        .map(|n| n.and_utc())
}
