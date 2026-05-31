//! Run all detectors against the live 7-day GOES X-ray data.
//! This has real 1-minute cadence and includes any recent flares.

use chrono::{DateTime, Utc};
use solar_monitor::detection::cusum::CusumDetector;
use solar_monitor::detection::energy::EnergyDetector;
use solar_monitor::detection::multichannel::MultichannelDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;
use solar_monitor::detection::rate_of_change::RateOfChangeDetector;
use solar_monitor::detection::zscore::ZScoreDetector;
use solar_monitor::feeds::xray::FlareClass;

#[derive(serde::Deserialize)]
struct SwpcXrayRecord {
    time_tag: String,
    flux: Option<f64>,
    energy: Option<String>,
}

fn main() {
    // Load from file or fetch
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/xrays-7-day.json".to_string());

    println!("=== 7-Day GOES X-ray Analysis (1-min cadence) ===");
    println!("Source: {}\n", path);

    let content = std::fs::read_to_string(&path)
        .expect("Failed to read file. Run: curl -s https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json -o /tmp/xrays-7-day.json");

    let raw: Vec<SwpcXrayRecord> = serde_json::from_str(&content).unwrap();

    // Filter to 0.1-0.8nm channel (long wavelength, standard for flare classification)
    let mut samples: Vec<(DateTime<Utc>, f64)> = raw
        .iter()
        .filter(|r| r.energy.as_deref() == Some("0.1-0.8nm"))
        .filter_map(|r| {
            let ts = chrono::NaiveDateTime::parse_from_str(&r.time_tag, "%Y-%m-%dT%H:%M:%SZ")
                .or_else(|_| {
                    chrono::NaiveDateTime::parse_from_str(&r.time_tag, "%Y-%m-%dT%H:%M:%S%.fZ")
                })
                .ok()?
                .and_utc();
            Some((ts, r.flux.unwrap_or(0.0)))
        })
        .collect();

    samples.sort_by_key(|(ts, _)| *ts);
    println!("Samples: {} (1-min cadence)", samples.len());
    println!(
        "Range: {} to {}",
        samples.first().unwrap().0.format("%Y-%m-%d %H:%M"),
        samples.last().unwrap().0.format("%Y-%m-%d %H:%M")
    );

    // Find peak
    let peak = samples
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!(
        "Peak flux: {:.2e} W/m² ({}-class) at {}",
        peak.1,
        FlareClass::from_flux(peak.1).label(),
        peak.0.format("%Y-%m-%d %H:%M")
    );

    // Count class thresholds
    let c_count = samples.iter().filter(|(_, f)| *f >= 1e-6).count();
    let m_count = samples.iter().filter(|(_, f)| *f >= 1e-5).count();
    let x_count = samples.iter().filter(|(_, f)| *f >= 1e-4).count();
    println!(
        "Minutes at C+: {}, M+: {}, X+: {}\n",
        c_count, m_count, x_count
    );

    // Run all detectors
    println!("========================================");
    println!("  Detector Responses (1-min real data)");
    println!("========================================\n");

    let mut zscore = ZScoreDetector::default_detector();
    let mut cusum = CusumDetector::new(120, 8.0);
    let mut energy = EnergyDetector::default_detector();
    let mut roc = RateOfChangeDetector::default_detector();
    let mut multi = MultichannelDetector::default_detector();
    let mut fusion = RankFusionDetector::new(0.5);

    // Track detections
    struct DetectorTracker {
        name: &'static str,
        detections: Vec<(DateTime<Utc>, f64, f64)>, // (time, score, flux)
        last_detection: Option<DateTime<Utc>>,
    }

    impl DetectorTracker {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                detections: Vec::new(),
                last_detection: None,
            }
        }
        fn maybe_record(&mut self, ts: DateTime<Utc>, score: f64, is_anom: bool, flux: f64) {
            if is_anom {
                let dominated = self.last_detection.map_or(false, |last| {
                    (ts - last).num_seconds() < 1800 // 30-min dedup for minute data
                });
                if !dominated {
                    self.detections.push((ts, score, flux));
                    self.last_detection = Some(ts);
                }
            }
        }
    }

    let mut trackers = vec![
        DetectorTracker::new("Z-Score"),
        DetectorTracker::new("CUSUM"),
        DetectorTracker::new("Energy"),
        DetectorTracker::new("Rate-of-Change"),
        DetectorTracker::new("Multichannel"),
        DetectorTracker::new("Rank Fusion"),
    ];

    // Use constant electron proxy (no real electron data in this feed)
    let electron_proxy = 100.0;

    for &(ts, flux) in &samples {
        zscore.ingest(flux, ts);
        cusum.ingest(flux, ts);
        energy.ingest(flux, electron_proxy, ts);
        roc.ingest(flux, ts);
        multi.ingest(flux, electron_proxy, ts);
        fusion.ingest_simple(flux, electron_proxy, ts);

        trackers[0].maybe_record(ts, zscore.score(), zscore.is_anomalous(), flux);
        trackers[1].maybe_record(ts, cusum.score(), cusum.is_anomalous(), flux);
        trackers[2].maybe_record(ts, energy.score(), energy.is_anomalous(), flux);
        trackers[3].maybe_record(ts, roc.score(), roc.is_anomalous(), flux);
        trackers[4].maybe_record(ts, multi.score(), multi.is_anomalous(), flux);
        trackers[5].maybe_record(ts, fusion.score(), fusion.is_anomalous(), flux);
    }

    println!("{:<16} {:>10} {:>10}", "Detector", "Detections", "X-class");
    println!("{}", "-".repeat(40));
    for t in &trackers {
        let x_det = t.detections.iter().filter(|(_, _, f)| *f >= 1e-4).count();
        println!("{:<16} {:>10} {:>10}", t.name, t.detections.len(), x_det);
    }

    // Print detection timeline for each detector
    for t in &trackers {
        if t.detections.is_empty() {
            continue;
        }
        println!("\n--- {} detections ---", t.name);
        for (ts, score, flux) in &t.detections {
            let class = FlareClass::from_flux(*flux);
            println!(
                "  {} score={:.3} flux={:.2e} class={}",
                ts.format("%m-%d %H:%M"),
                score,
                flux,
                class.label()
            );
        }
    }

    // Fusion detail around the X-class event
    println!("\n========================================");
    println!("  Fusion Detail Around X-class Peak");
    println!("========================================\n");

    // Find X-class window
    let x_start = samples.iter().position(|(_, f)| *f >= 1e-4);
    if let Some(start_idx) = x_start {
        // Show 30 min before to 30 min after
        let window_start = start_idx.saturating_sub(30);
        let window_end = (start_idx + 60).min(samples.len());

        // Re-run fusion for this window with diagnostics
        let mut fusion2 = RankFusionDetector::new(0.5);
        // Warm up
        for &(ts, flux) in &samples[..window_start] {
            fusion2.ingest_simple(flux, electron_proxy, ts);
        }

        println!(
            "{:<18} {:>10} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>5}",
            "Time", "Flux", "ZScor", "CUSUM", "Enrgy", "RoC", "Multi", "Fused", "Agr"
        );
        println!("{}", "-".repeat(95));

        for &(ts, flux) in &samples[window_start..window_end] {
            fusion2.ingest_simple(flux, electron_proxy, ts);
            let diag = fusion2.diagnostics();
            let scores: Vec<f64> = diag.raw_scores.iter().map(|d| d.raw_score).collect();
            let class = FlareClass::from_flux(flux);
            let marker = if flux >= 1e-4 {
                " X"
            } else if flux >= 1e-5 {
                " M"
            } else {
                ""
            };
            println!(
                "{} {:>10.2e}{:<2} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7.3} {:>3}/5",
                ts.format("%m-%d %H:%M"),
                flux,
                marker,
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                scores[4],
                diag.fused_score,
                diag.detector_agreement
            );
        }
    }
}
