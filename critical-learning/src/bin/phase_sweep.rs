use critical_learning::{
    CouplingTopology, CriticalLearningConfig, CriticalLearningModel, J_CRITICAL,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
struct CliArgs {
    output_json: Option<PathBuf>,
    output_md: Option<PathBuf>,
    sequence: String,
    sequence_repeat: usize,
    n_sites: usize,
    dt_values: Vec<f32>,
    j_values: Vec<f32>,
    threshold_values: Vec<f32>,
    discharge_values: Vec<f32>,
    standing_wave_amplitudes: Vec<f32>,
    standing_wave_cycles: Vec<f32>,
    temporal_harmonic_amplitudes: Vec<f32>,
    temporal_harmonic_frequencies: Vec<f32>,
    topologies: Vec<CouplingTopology>,
}

#[derive(Debug, Serialize)]
struct SweepPoint {
    dt: f32,
    j: f32,
    inhibition_threshold: f32,
    discharge_gain: f32,
    standing_wave_amplitude: f32,
    standing_wave_cycles: f32,
    temporal_harmonic_amplitude: f32,
    temporal_harmonic_frequency: f32,
    topology: &'static str,
    critical_gap: f32,
    mean_sync_order: f32,
    mean_dispersion: f32,
    mean_adaptation_rate: f32,
    mean_balance: f32,
    mean_boundary_crossing_rate: f32,
    mean_suppression_fraction: f32,
    final_bivector_norm: f32,
    final_scalar_vector_ratio: f32,
    cache_fill: usize,
}

#[derive(Debug, Serialize)]
struct SweepReport {
    sequence: String,
    n_sites: usize,
    j_critical: f32,
    points: Vec<SweepPoint>,
}

#[derive(Debug, Clone)]
struct GroupSummary {
    count: usize,
    mean_boundary_crossing_rate: f32,
    mean_suppression_fraction: f32,
    mean_sync_order: f32,
    mean_adaptation_rate: f32,
    mean_balance: f32,
    best_gap: f32,
}

fn parse_csv_f32(raw: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().map_err(|e| e.into()))
        .collect()
}

fn parse_topologies(raw: &str) -> Result<Vec<CouplingTopology>, Box<dyn Error>> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| match s {
            "ring" => Ok(CouplingTopology::Ring),
            "complete" => Ok(CouplingTopology::Complete),
            other => Err(format!("unknown topology: {other}").into()),
        })
        .collect()
}

fn parse_wave_presets(raw: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut values: Vec<f32> = Vec::new();
    for preset in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match preset {
            "phi" => values.push(1.618_034),
            "theta" => values.extend([4.0, 6.0]),
            "alpha" => values.push(2.5),
            "critical" => values.push(J_CRITICAL),
            other => {
                return Err(format!("unknown wave preset: {other}").into());
            }
        }
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-6_f32);
    Ok(values)
}

fn parse_temporal_presets(raw: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut values: Vec<f32> = Vec::new();
    for preset in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match preset {
            "critical" => values.push(J_CRITICAL),
            "phi" => values.push(1.618_034),
            "theta" => values.extend([4.0, 6.0]),
            "alpha" => values.extend([8.0, 10.0, 12.0]),
            "beta" => values.extend([16.0, 20.0, 24.0]),
            "gamma" => values.extend([30.0, 40.0, 50.0]),
            "brainwaves" => values.extend([2.0, 6.0, 10.0, 20.0, 40.0]),
            "e" => values.push(std::f32::consts::E),
            "pi" => values.push(std::f32::consts::PI),
            "zeta" => values.push(14.134_726),
            other => {
                return Err(format!("unknown temporal preset: {other}").into());
            }
        }
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-6_f32);
    Ok(values)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f32>() / xs.len() as f32
    }
}

fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut output_json = None;
    let mut output_md = None;
    let mut sequence = String::from("criticality criticality criticality");
    let mut sequence_repeat = 1usize;
    let mut n_sites = 12usize;
    let mut dt_values = vec![0.1];
    let mut j_values = vec![0.25, J_CRITICAL, 1.0];
    let mut threshold_values = vec![0.25, 0.5, 1.0];
    let mut discharge_values = vec![0.25, 0.75, 1.25];
    let mut standing_wave_amplitudes = vec![0.0, 0.5];
    let mut standing_wave_cycles = vec![1.0];
    let mut temporal_harmonic_amplitudes = vec![0.0];
    let mut temporal_harmonic_frequencies = vec![4.0];
    let mut topologies = vec![CouplingTopology::Ring, CouplingTopology::Complete];

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output-json" => output_json = args.next().map(PathBuf::from),
            "--output-md" => output_md = args.next().map(PathBuf::from),
            "--sequence" => {
                if let Some(val) = args.next() {
                    sequence = val;
                }
            }
            "--sequence-repeat" => {
                if let Some(val) = args.next() {
                    sequence_repeat = val.parse()?;
                }
            }
            "--n-sites" => {
                if let Some(val) = args.next() {
                    n_sites = val.parse()?;
                }
            }
            "--dt-values" => {
                if let Some(val) = args.next() {
                    dt_values = parse_csv_f32(&val)?;
                }
            }
            "--j-values" => {
                if let Some(val) = args.next() {
                    j_values = parse_csv_f32(&val)?;
                }
            }
            "--threshold-values" => {
                if let Some(val) = args.next() {
                    threshold_values = parse_csv_f32(&val)?;
                }
            }
            "--discharge-values" => {
                if let Some(val) = args.next() {
                    discharge_values = parse_csv_f32(&val)?;
                }
            }
            "--standing-wave-amplitudes" => {
                if let Some(val) = args.next() {
                    standing_wave_amplitudes = parse_csv_f32(&val)?;
                }
            }
            "--standing-wave-cycles" => {
                if let Some(val) = args.next() {
                    standing_wave_cycles = parse_csv_f32(&val)?;
                }
            }
            "--temporal-harmonic-amplitudes" => {
                if let Some(val) = args.next() {
                    temporal_harmonic_amplitudes = parse_csv_f32(&val)?;
                }
            }
            "--temporal-harmonic-frequencies" => {
                if let Some(val) = args.next() {
                    temporal_harmonic_frequencies = parse_csv_f32(&val)?;
                }
            }
            "--wave-presets" => {
                if let Some(val) = args.next() {
                    standing_wave_cycles = parse_wave_presets(&val)?;
                }
            }
            "--temporal-presets" => {
                if let Some(val) = args.next() {
                    temporal_harmonic_frequencies = parse_temporal_presets(&val)?;
                }
            }
            "--topologies" => {
                if let Some(val) = args.next() {
                    topologies = parse_topologies(&val)?;
                }
            }
            "--help" | "-h" => {
                println!(
                    "phase_sweep [--output-json PATH] [--output-md PATH] [--sequence TEXT] \
                     [--sequence-repeat N] [--n-sites N] [--dt-values CSV] \
                     [--j-values CSV] [--threshold-values CSV] \
                     [--discharge-values CSV] [--standing-wave-amplitudes CSV] \
                     [--standing-wave-cycles CSV|--wave-presets theta,phi,alpha,critical] \
                     [--temporal-harmonic-amplitudes CSV] \
                     [--temporal-harmonic-frequencies CSV|--temporal-presets critical,phi,theta,alpha,beta,gamma,brainwaves,e,pi,zeta] \
                     [--topologies ring,complete]"
                );
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
    }

    Ok(CliArgs {
        output_json,
        output_md,
        sequence,
        sequence_repeat,
        n_sites,
        dt_values,
        j_values,
        threshold_values,
        discharge_values,
        standing_wave_amplitudes,
        standing_wave_cycles,
        temporal_harmonic_amplitudes,
        temporal_harmonic_frequencies,
        topologies,
    })
}

fn build_report(args: &CliArgs) -> SweepReport {
    let mut points = Vec::new();
    let sequence = std::iter::repeat_n(args.sequence.as_str(), args.sequence_repeat.max(1))
        .collect::<Vec<_>>()
        .join(" ");
    let input = sequence.as_bytes();

    for &dt in &args.dt_values {
        for &j in &args.j_values {
            for &threshold in &args.threshold_values {
                for &discharge_gain in &args.discharge_values {
                    for &standing_wave_amplitude in &args.standing_wave_amplitudes {
                        for &standing_wave_cycles in &args.standing_wave_cycles {
                            for &temporal_harmonic_amplitude in &args.temporal_harmonic_amplitudes {
                                for &temporal_harmonic_frequency in
                                    &args.temporal_harmonic_frequencies
                                {
                                    for &topology in &args.topologies {
                                        let config = CriticalLearningConfig {
                                            n_sites: args.n_sites,
                                            dt,
                                            j_init: j,
                                            inhibition_threshold_init: threshold,
                                            discharge_gain_init: discharge_gain,
                                            standing_wave_amplitude_init: standing_wave_amplitude,
                                            standing_wave_cycles_init: standing_wave_cycles,
                                            temporal_harmonic_amplitude_init:
                                                temporal_harmonic_amplitude,
                                            temporal_harmonic_frequency_init:
                                                temporal_harmonic_frequency,
                                            topology,
                                            ..CriticalLearningConfig::default()
                                        };
                                        let mut model = CriticalLearningModel::new(config);
                                        let (_, diag) = model.forward(input);
                                        points.push(SweepPoint {
                                            dt,
                                            j,
                                            inhibition_threshold: threshold,
                                            discharge_gain,
                                            standing_wave_amplitude,
                                            standing_wave_cycles,
                                            temporal_harmonic_amplitude,
                                            temporal_harmonic_frequency,
                                            topology: match topology {
                                                CouplingTopology::Ring => "ring",
                                                CouplingTopology::Complete => "complete",
                                            },
                                            critical_gap: diag.critical_gap,
                                            mean_sync_order: mean(&diag.sync_order),
                                            mean_dispersion: mean(&diag.dispersion),
                                            mean_adaptation_rate: mean(&diag.adaptation_rate),
                                            mean_balance: mean(
                                                &diag.coherence_adaptability_balance,
                                            ),
                                            mean_boundary_crossing_rate: mean(
                                                &diag.boundary_crossing_rate,
                                            ),
                                            mean_suppression_fraction: mean(
                                                &diag.suppression_fraction,
                                            ),
                                            final_bivector_norm: *diag
                                                .bivector_norm
                                                .last()
                                                .unwrap_or(&0.0),
                                            final_scalar_vector_ratio: *diag
                                                .scalar_vector_ratio
                                                .last()
                                                .unwrap_or(&0.0),
                                            cache_fill: diag.cache_fill,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    points.sort_by(|a, b| {
        a.critical_gap
            .partial_cmp(&b.critical_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.mean_boundary_crossing_rate
                    .partial_cmp(&a.mean_boundary_crossing_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    SweepReport {
        sequence,
        n_sites: args.n_sites,
        j_critical: J_CRITICAL,
        points,
    }
}

fn summarize_by_key<F>(points: &[SweepPoint], key_fn: F) -> BTreeMap<String, GroupSummary>
where
    F: Fn(&SweepPoint) -> String,
{
    let mut buckets: BTreeMap<String, Vec<&SweepPoint>> = BTreeMap::new();
    for point in points {
        buckets.entry(key_fn(point)).or_default().push(point);
    }

    buckets
        .into_iter()
        .map(|(key, bucket)| {
            let count = bucket.len();
            let inv = 1.0 / count.max(1) as f32;
            let summary = GroupSummary {
                count,
                mean_boundary_crossing_rate: bucket
                    .iter()
                    .map(|p| p.mean_boundary_crossing_rate)
                    .sum::<f32>()
                    * inv,
                mean_suppression_fraction: bucket
                    .iter()
                    .map(|p| p.mean_suppression_fraction)
                    .sum::<f32>()
                    * inv,
                mean_sync_order: bucket.iter().map(|p| p.mean_sync_order).sum::<f32>() * inv,
                mean_adaptation_rate: bucket.iter().map(|p| p.mean_adaptation_rate).sum::<f32>()
                    * inv,
                mean_balance: bucket.iter().map(|p| p.mean_balance).sum::<f32>() * inv,
                best_gap: bucket
                    .iter()
                    .map(|p| p.critical_gap)
                    .fold(f32::INFINITY, f32::min),
            };
            (key, summary)
        })
        .collect()
}

fn write_markdown(path: &PathBuf, report: &SweepReport) -> Result<(), Box<dyn Error>> {
    let mut lines = vec![
        String::from("# Critical Learning Phase Sweep"),
        String::new(),
        format!("- sequence: `{}`", report.sequence),
        format!("- n_sites: `{}`", report.n_sites),
        format!("- J_c: `{:.6}`", report.j_critical),
        String::new(),
        String::from("| topology | dt | wave_amp | wave_cycles | temp_amp | temp_freq | J | threshold | discharge | gap | boundary_crossing | suppression | sync | adapt | balance |"),
        String::from("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"),
    ];

    for point in &report.points {
        lines.push(format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            point.topology,
            point.dt,
            point.standing_wave_amplitude,
            point.standing_wave_cycles,
            point.temporal_harmonic_amplitude,
            point.temporal_harmonic_frequency,
            point.j,
            point.inhibition_threshold,
            point.discharge_gain,
            point.critical_gap,
            point.mean_boundary_crossing_rate,
            point.mean_suppression_fraction,
            point.mean_sync_order,
            point.mean_adaptation_rate,
            point.mean_balance
        ));
    }

    if let Some(best) = report.points.first() {
        lines.push(String::new());
        lines.push(String::from("## Nearest Critical Point"));
        lines.push(String::new());
        lines.push(format!(
            "`topology={}`, `dt={:.4}`, `wave_amp={:.4}`, `wave_cycles={:.4}`, `temp_amp={:.4}`, `temp_freq={:.4}`, `J={:.4}`, `threshold={:.4}`, `discharge={:.4}`, `gap={:.4}`, \
             `boundary_crossing={:.4}`, `suppression={:.4}`",
            best.topology,
            best.dt,
            best.standing_wave_amplitude,
            best.standing_wave_cycles,
            best.temporal_harmonic_amplitude,
            best.temporal_harmonic_frequency,
            best.j,
            best.inhibition_threshold,
            best.discharge_gain,
            best.critical_gap,
            best.mean_boundary_crossing_rate,
            best.mean_suppression_fraction
        ));
    }

    if let Some(best_balance) = report.points.iter().max_by(|a, b| {
        a.mean_balance
            .partial_cmp(&b.mean_balance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.critical_gap
                    .partial_cmp(&a.critical_gap)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.mean_suppression_fraction
                    .partial_cmp(&b.mean_suppression_fraction)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }) {
        lines.push(String::new());
        lines.push(String::from("## Best Balance"));
        lines.push(String::new());
        lines.push(format!(
            "`topology={}`, `dt={:.4}`, `wave_amp={:.4}`, `wave_cycles={:.4}`, `temp_amp={:.4}`, `temp_freq={:.4}`, `J={:.4}`, `threshold={:.4}`, `discharge={:.4}`, `gap={:.4}`, \
             `boundary_crossing={:.4}`, `suppression={:.4}`, `sync={:.4}`, `adapt={:.4}`, `balance={:.4}`",
            best_balance.topology,
            best_balance.dt,
            best_balance.standing_wave_amplitude,
            best_balance.standing_wave_cycles,
            best_balance.temporal_harmonic_amplitude,
            best_balance.temporal_harmonic_frequency,
            best_balance.j,
            best_balance.inhibition_threshold,
            best_balance.discharge_gain,
            best_balance.critical_gap,
            best_balance.mean_boundary_crossing_rate,
            best_balance.mean_suppression_fraction,
            best_balance.mean_sync_order,
            best_balance.mean_adaptation_rate,
            best_balance.mean_balance
        ));
    }

    if let Some(best_release) = report.points.iter().min_by(|a, b| {
        a.critical_gap
            .partial_cmp(&b.critical_gap)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.mean_boundary_crossing_rate
                    .partial_cmp(&a.mean_boundary_crossing_rate)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.mean_suppression_fraction
                    .partial_cmp(&b.mean_suppression_fraction)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }) {
        lines.push(String::new());
        lines.push(String::from("## Best Critical Release"));
        lines.push(String::new());
        lines.push(format!(
            "`topology={}`, `dt={:.4}`, `wave_amp={:.4}`, `wave_cycles={:.4}`, `temp_amp={:.4}`, `temp_freq={:.4}`, `J={:.4}`, `threshold={:.4}`, `discharge={:.4}`, `gap={:.4}`, \
             `boundary_crossing={:.4}`, `suppression={:.4}`, `sync={:.4}`, `adapt={:.4}`, `balance={:.4}`",
            best_release.topology,
            best_release.dt,
            best_release.standing_wave_amplitude,
            best_release.standing_wave_cycles,
            best_release.temporal_harmonic_amplitude,
            best_release.temporal_harmonic_frequency,
            best_release.j,
            best_release.inhibition_threshold,
            best_release.discharge_gain,
            best_release.critical_gap,
            best_release.mean_boundary_crossing_rate,
            best_release.mean_suppression_fraction,
            best_release.mean_sync_order,
            best_release.mean_adaptation_rate,
            best_release.mean_balance
        ));
    }

    let by_dt = summarize_by_key(&report.points, |p| format!("{:.4}", p.dt));
    lines.push(String::new());
    lines.push(String::from("## By Dt"));
    lines.push(String::new());
    lines.push(String::from("| dt | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---:|---:|---:|---:|---:|---:|---:|---:|"));
    for (dt, summary) in by_dt {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            dt,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    let by_topology = summarize_by_key(&report.points, |p| p.topology.to_string());
    lines.push(String::new());
    lines.push(String::from("## By Topology"));
    lines.push(String::new());
    lines.push(String::from("| topology | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---|---:|---:|---:|---:|---:|---:|---:|"));
    for (topology, summary) in by_topology {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            topology,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    let by_cycle = summarize_by_key(&report.points, |p| format!("{:.4}", p.standing_wave_cycles));
    lines.push(String::new());
    lines.push(String::from("## By Wave Cycle"));
    lines.push(String::new());
    lines.push(String::from("| wave_cycles | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---:|---:|---:|---:|---:|---:|---:|---:|"));
    for (wave_cycles, summary) in by_cycle {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            wave_cycles,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    let by_temporal = summarize_by_key(&report.points, |p| {
        format!(
            "{:.4} / {:.4}",
            p.temporal_harmonic_amplitude, p.temporal_harmonic_frequency
        )
    });
    lines.push(String::new());
    lines.push(String::from("## By Temporal Harmonic"));
    lines.push(String::new());
    lines.push(String::from("| temp_amp / temp_freq | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---|---:|---:|---:|---:|---:|---:|---:|"));
    for (label, summary) in by_temporal {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    let by_wave = summarize_by_key(&report.points, |p| {
        format!("{:.4}", p.standing_wave_amplitude)
    });
    lines.push(String::new());
    lines.push(String::from("## By Standing Wave"));
    lines.push(String::new());
    lines.push(String::from("| wave_amp | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---:|---:|---:|---:|---:|---:|---:|---:|"));
    for (wave_amp, summary) in by_wave {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            wave_amp,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    let by_topology_and_cycle = summarize_by_key(&report.points, |p| {
        format!("{} / {:.4}", p.topology, p.standing_wave_cycles)
    });
    lines.push(String::new());
    lines.push(String::from("## By Topology And Wave Cycle"));
    lines.push(String::new());
    lines.push(String::from("| topology / wave_cycles | count | best_gap | mean_crossing | mean_suppression | mean_sync | mean_adapt | mean_balance |"));
    lines.push(String::from("|---|---:|---:|---:|---:|---:|---:|---:|"));
    for (label, summary) in by_topology_and_cycle {
        lines.push(format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            label,
            summary.count,
            summary.best_gap,
            summary.mean_boundary_crossing_rate,
            summary.mean_suppression_fraction,
            summary.mean_sync_order,
            summary.mean_adaptation_rate,
            summary.mean_balance
        ));
    }

    fs::write(path, lines.join("\n"))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let report = build_report(&args);

    if let Some(path) = &args.output_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }

    if let Some(path) = &args.output_md {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        write_markdown(path, &report)?;
    }

    println!(
        "phase sweep complete: {} points around J_c={:.6}",
        report.points.len(),
        report.j_critical
    );
    if let Some(best) = report.points.first() {
        println!(
            "nearest critical point: J={:.4}, threshold={:.4}, discharge={:.4}, gap={:.4}, crossing={:.4}, suppression={:.4}",
            best.j,
            best.inhibition_threshold,
            best.discharge_gain,
            best.critical_gap,
            best.mean_boundary_crossing_rate,
            best.mean_suppression_fraction
        );
    }

    Ok(())
}
