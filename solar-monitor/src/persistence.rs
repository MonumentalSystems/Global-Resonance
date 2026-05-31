//! Append-only JSONL persistence for live solar monitor data.
//!
//! Writes one JSON line per poll cycle to daily-rotated files:
//!   `{data_dir}/live/2026-04-04.jsonl`
//!
//! Each line is a compact snapshot: timestamp, feed values, detector scores,
//! and escalation state. ~300-400 bytes/line, ~1 MB/day at 30s polling,
//! ~100 MB/year uncompressed.

use chrono::{DateTime, Datelike, Utc};
use serde::Serialize;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::detection::escalation::EscalationStatus;
use crate::detection::rank_fusion::FusionDiagnostics;
use crate::feeds::FeedState;

/// One JSONL record per poll cycle.
#[derive(Serialize)]
pub struct LiveRecord {
    /// UTC timestamp of this snapshot.
    pub ts: DateTime<Utc>,
    // -- Feed values (latest from each ring buffer) --
    /// GOES 0.1-0.8nm X-ray flux (W/m²).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xray: Option<f64>,
    /// GOES 0.05-0.4nm X-ray flux (W/m²).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xray_short: Option<f64>,
    /// >2 MeV electron flux (pfu).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub electron: Option<f64>,
    /// >=1 MeV proton flux (pfu).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proton: Option<f64>,
    /// Solar wind speed (km/s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sw_speed: Option<f64>,
    /// IMF Bx (nT).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sw_bx: Option<f64>,
    /// IMF By (nT).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sw_by: Option<f64>,
    /// IMF Bz (nT).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sw_bz: Option<f64>,
    /// Solar wind density (cm⁻³).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sw_density: Option<f64>,
    /// Kp index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kp: Option<f64>,
    /// Estimated Dst (nT).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dst: Option<f64>,
    // -- Detector state --
    /// Fused anomaly score (0..1).
    pub fused: f64,
    /// Number of detectors agreeing on anomaly.
    pub agreement: usize,
    /// Escalation level label.
    pub escalation: String,
    /// Per-detector scores: [(name, raw_score, is_anomalous)].
    pub detectors: Vec<(String, f64, bool)>,
    // -- Feed buffer sizes (for diagnosing data gaps) --
    pub buf_xray: usize,
    pub buf_sw: usize,
    pub buf_electron: usize,
    pub buf_kp: usize,
}

impl LiveRecord {
    /// Build a record from current feed and detector state.
    pub fn from_state(
        feeds: &FeedState,
        diagnostics: &FusionDiagnostics,
        escalation: &EscalationStatus,
    ) -> Self {
        let sw = feeds.solar_wind.back();
        Self {
            ts: Utc::now(),
            xray: feeds.xray.back().map(|s| s.flux),
            xray_short: feeds.xray_short.back().map(|s| s.flux),
            electron: feeds.electrons.back().map(|s| s.flux),
            proton: feeds.protons.back().map(|s| s.flux),
            sw_speed: sw.map(|s| s.speed),
            sw_bx: sw.map(|s| s.bx),
            sw_by: sw.map(|s| s.by),
            sw_bz: sw.map(|s| s.bz),
            sw_density: sw.map(|s| s.density),
            kp: feeds.kp_dst.back().map(|s| s.kp),
            dst: feeds.kp_dst.back().map(|s| s.estimated_dst),
            fused: diagnostics.fused_score,
            agreement: diagnostics.detector_agreement,
            escalation: escalation.level_label.to_string(),
            detectors: diagnostics
                .raw_scores
                .iter()
                .map(|d| (d.name.to_string(), d.raw_score, d.is_anomalous))
                .collect(),
            buf_xray: feeds.xray.len(),
            buf_sw: feeds.solar_wind.len(),
            buf_electron: feeds.electrons.len(),
            buf_kp: feeds.kp_dst.len(),
        }
    }
}

/// Daily-rotating JSONL writer.
pub struct LiveLogger {
    data_dir: PathBuf,
    current_date: Option<(i32, u32, u32)>,
    writer: Option<BufWriter<File>>,
    lines_written: u64,
}

impl LiveLogger {
    /// Create a new logger writing to `{data_dir}/live/`.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            current_date: None,
            writer: None,
            lines_written: 0,
        }
    }

    /// Append a record. Rotates file on date change.
    pub fn append(&mut self, record: &LiveRecord) {
        let date = (record.ts.year(), record.ts.month(), record.ts.day());

        // Rotate on date change
        if self.current_date != Some(date) {
            self.rotate(date);
        }

        if let Some(ref mut w) = self.writer {
            match serde_json::to_string(record) {
                Ok(line) => {
                    if writeln!(w, "{}", line).is_ok() {
                        self.lines_written += 1;
                        // Flush every line — polling is 30s so throughput is trivial
                        let _ = w.flush();
                    }
                }
                Err(e) => {
                    tracing::warn!("LiveLogger serialize error: {}", e);
                }
            }
        }
    }

    /// Flush buffered data to disk.
    pub fn flush(&mut self) {
        if let Some(ref mut w) = self.writer {
            let _ = w.flush();
        }
    }

    fn rotate(&mut self, date: (i32, u32, u32)) {
        // Flush and drop old writer
        self.flush();
        self.writer = None;

        let live_dir = self.data_dir.join("live");
        if let Err(e) = fs::create_dir_all(&live_dir) {
            tracing::error!("LiveLogger: cannot create {}: {}", live_dir.display(), e);
            return;
        }

        let filename = format!("{:04}-{:02}-{:02}.jsonl", date.0, date.1, date.2);
        let path = live_dir.join(&filename);

        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => {
                tracing::info!("LiveLogger: writing to {}", path.display());
                self.writer = Some(BufWriter::new(file));
                self.current_date = Some(date);
            }
            Err(e) => {
                tracing::error!("LiveLogger: cannot open {}: {}", path.display(), e);
            }
        }
    }

    /// Total lines written this session.
    pub fn total_lines(&self) -> u64 {
        self.lines_written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_live_record_serializes() {
        let feeds = FeedState::new();
        let diag = FusionDiagnostics {
            fused_score: 0.5,
            alert: false,
            raw_scores: vec![],
            detector_agreement: 0,
        };
        let esc = EscalationStatus {
            level: crate::detection::escalation::EscalationLevel::Quiet,
            level_label: "QUIET",
            level_numeric: 0,
            since: None,
            hardness_spikes_in_window: 0,
            peak_hardness: 0.0,
            peak_fused: 0.0,
            flare_triggers: 0,
            minutes_at_level: 0,
        };
        let record = LiveRecord::from_state(&feeds, &diag, &esc);
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"fused\":0.5"));
        assert!(json.contains("\"escalation\":\"QUIET\""));
        // None fields should be skipped
        assert!(!json.contains("\"xray\":"));
    }

    #[test]
    fn test_daily_rotation() {
        let dir = TempDir::new().unwrap();
        let mut logger = LiveLogger::new(dir.path());

        let feeds = FeedState::new();
        let diag = FusionDiagnostics {
            fused_score: 0.0,
            alert: false,
            raw_scores: vec![],
            detector_agreement: 0,
        };
        let esc = EscalationStatus {
            level: crate::detection::escalation::EscalationLevel::Quiet,
            level_label: "QUIET",
            level_numeric: 0,
            since: None,
            hardness_spikes_in_window: 0,
            peak_hardness: 0.0,
            peak_fused: 0.0,
            flare_triggers: 0,
            minutes_at_level: 0,
        };
        let record = LiveRecord::from_state(&feeds, &diag, &esc);
        logger.append(&record);
        logger.flush();

        let today = Utc::now();
        let filename = format!(
            "{:04}-{:02}-{:02}.jsonl",
            today.year(),
            today.month(),
            today.day()
        );
        let path = dir.path().join("live").join(filename);
        assert!(path.exists());
        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content.lines().count(), 1);
    }

    #[test]
    fn test_append_multiple() {
        let dir = TempDir::new().unwrap();
        let mut logger = LiveLogger::new(dir.path());

        let feeds = FeedState::new();
        let diag = FusionDiagnostics {
            fused_score: 0.0,
            alert: false,
            raw_scores: vec![],
            detector_agreement: 0,
        };
        let esc = EscalationStatus {
            level: crate::detection::escalation::EscalationLevel::Quiet,
            level_label: "QUIET",
            level_numeric: 0,
            since: None,
            hardness_spikes_in_window: 0,
            peak_hardness: 0.0,
            peak_fused: 0.0,
            flare_triggers: 0,
            minutes_at_level: 0,
        };

        for _ in 0..5 {
            let record = LiveRecord::from_state(&feeds, &diag, &esc);
            logger.append(&record);
        }
        logger.flush();
        assert_eq!(logger.total_lines(), 5);
    }
}
