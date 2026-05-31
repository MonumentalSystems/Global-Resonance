use solar_monitor::backtest::{goes_loader, loaders};
use solar_monitor::detection::escalation::{EscalationLevel, EscalationMonitor};
use solar_monitor::detection::flare_clustering::FlareClusteringDetector;
use solar_monitor::detection::rank_fusion::RankFusionDetector;

fn main() {
    let goes = goes_loader::load_goes_csv(std::path::Path::new(
        "solar-monitor/data/goes_data/goes16_xrs_2024.csv",
    ))
    .unwrap();
    let all_flares = loaders::load_flares(std::path::Path::new(
        "solar-monitor/data/catalogs/solar_flares.csv",
    ))
    .unwrap();
    let start = goes.first().unwrap().timestamp;
    let end = goes.last().unwrap().timestamp;
    let flares: Vec<_> = all_flares
        .into_iter()
        .filter(|f| f.begin >= start && f.begin <= end)
        .collect();

    let mut fusion = RankFusionDetector::new(0.7);
    let mut esc = EscalationMonitor::new();
    let mut cluster = FlareClusteringDetector::default_detector();
    let mut next = 0;

    println!("=== Missed Flares (QUIET at onset) — 2024 ===\n");
    for rec in &goes {
        fusion.ingest(rec.xrsb, rec.xrsa, 100.0, 0.3, rec.timestamp);
        cluster.ingest(rec.xrsb, rec.timestamp);
        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();
        esc.update_with_flux(h, f, a, rec.xrsb, rec.timestamp);

        while next < flares.len() && rec.timestamp >= flares[next].begin {
            if esc.level == EscalationLevel::Quiet {
                let fl = &flares[next];
                println!(
                    "{} {} (numeric={:.2})",
                    fl.begin.format("%Y-%m-%d %H:%M"),
                    fl.class,
                    fl.class_numeric,
                );
                println!(
                    "  hardness={:.3} fused={:.3} cluster={:.3} agree={} xrsb={:.2e}",
                    fusion.hardness.score(),
                    fusion.score(),
                    cluster.score(),
                    fusion.detector_agreement(),
                    rec.xrsb,
                );
                // Check what happened in the hour before
                let hour_before = fl.begin - chrono::Duration::hours(1);
                let pre_samples: Vec<_> = goes
                    .iter()
                    .filter(|r| r.timestamp >= hour_before && r.timestamp < fl.begin)
                    .collect();
                if !pre_samples.is_empty() {
                    let max_flux = pre_samples.iter().map(|r| r.xrsb).fold(0.0f64, f64::max);
                    let min_flux = pre_samples
                        .iter()
                        .map(|r| r.xrsb)
                        .fold(f64::INFINITY, f64::min);
                    let ratio = if min_flux > 0.0 {
                        max_flux / min_flux
                    } else {
                        0.0
                    };
                    println!(
                        "  1h before: flux {:.2e} to {:.2e} (range {:.1}x), {} samples",
                        min_flux,
                        max_flux,
                        ratio,
                        pre_samples.len(),
                    );
                }
                println!();
            }
            next += 1;
        }
    }
}
