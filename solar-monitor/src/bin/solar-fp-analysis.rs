use solar_monitor::backtest::{goes_loader, loaders};
use solar_monitor::detection::escalation::{EscalationLevel, EscalationMonitor};
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

    // Count minutes at each level, binned by actual X-ray flux
    // Flux bins: A (<1e-7), B (1e-7 to 1e-6), C (1e-6 to 1e-5), M+ (>=1e-5)
    let mut level_flux = [[0u64; 4]; 4]; // [level][flux_bin]

    for rec in &goes {
        if rec.xrsb < 1e-9 {
            continue;
        }
        fusion.ingest(rec.xrsb, rec.xrsa, 100.0, 0.3, rec.timestamp);
        let h = fusion.hardness.score();
        let f = fusion.score();
        let a = fusion.detector_agreement();
        esc.update_with_flux(h, f, a, rec.xrsb, rec.timestamp);

        let level_idx = esc.level as usize;
        let flux_bin = if rec.xrsb >= 1e-5 {
            3
        } else if rec.xrsb >= 1e-6 {
            2
        } else if rec.xrsb >= 1e-7 {
            1
        } else {
            0
        };
        level_flux[level_idx][flux_bin] += 1;
    }

    let levels = ["QUIET", "ELEVATED", "ACTIVE", "FLARE"];
    let bins = ["A (<1e-7)", "B (1e-7-1e-6)", "C (1e-6-1e-5)", "M+ (>=1e-5)"];

    println!("=== Escalation Level vs Actual X-ray Flux (2024) ===\n");
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Level", bins[0], bins[1], bins[2], bins[3], "Total"
    );
    println!("{}", "-".repeat(72));
    for i in 0..4 {
        let total: u64 = level_flux[i].iter().sum();
        println!(
            "{:<10} {:>10}m {:>10}m {:>10}m {:>10}m {:>8}m",
            levels[i],
            level_flux[i][0],
            level_flux[i][1],
            level_flux[i][2],
            level_flux[i][3],
            total
        );
    }

    // True FP: minutes at ACTIVE/FLARE when flux is A or B class (truly quiet sun)
    let active_quiet = level_flux[2][0] + level_flux[2][1]; // ACTIVE at A/B flux
    let flare_quiet = level_flux[3][0] + level_flux[3][1]; // FLARE at A/B flux
    let total_active_flare: u64 =
        level_flux[2].iter().sum::<u64>() + level_flux[3].iter().sum::<u64>();
    let total_ab = active_quiet + flare_quiet;

    println!();
    println!("=== True False Positive Analysis ===\n");
    println!("Minutes at ACTIVE/FLARE with A/B-class flux (truly quiet):");
    println!(
        "  ACTIVE at A/B: {} min ({:.2}h)",
        active_quiet,
        active_quiet as f64 / 60.0
    );
    println!(
        "  FLARE at A/B:  {} min ({:.2}h)",
        flare_quiet,
        flare_quiet as f64 / 60.0
    );
    println!(
        "  Total FP:      {} min ({:.1}h)",
        total_ab,
        total_ab as f64 / 60.0
    );
    println!(
        "  Out of:        {} min ({:.0}h) at ACTIVE+FLARE",
        total_active_flare,
        total_active_flare as f64 / 60.0
    );
    println!(
        "  True FP rate:  {:.2}%",
        total_ab as f64 / total_active_flare as f64 * 100.0
    );

    // Also: minutes at QUIET/ELEVATED with C/M flux (missed activity)
    let quiet_active = level_flux[0][2] + level_flux[0][3]; // QUIET at C/M flux
    let elev_active = level_flux[1][2] + level_flux[1][3];
    println!();
    println!("Minutes at QUIET/ELEVATED with C+/M+ flux (missed activity):");
    println!("  QUIET at C+:     {} min", quiet_active);
    println!("  ELEVATED at C+:  {} min", elev_active);
}
