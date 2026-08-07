use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::coupling::{StressorIndex, StressorScore};
use crate::detection::escalation::{EscalationMonitor, EscalationStatus};
use crate::detection::rank_fusion::{FusionDiagnostics, RankFusionDetector};
use crate::feeds::{FeedQuality, FeedState};
use crate::persistence::{LiveLogger, LiveRecord};

pub const DETECTOR_WARMUP_SAMPLES: usize = 200;

fn pending_xray_observations(
    feeds: &FeedState,
    last_processed: Option<chrono::DateTime<chrono::Utc>>,
) -> Vec<crate::feeds::xray::XraySample> {
    let mut pending: Vec<_> = feeds
        .xray
        .iter()
        .filter(|sample| {
            sample.flux >= 1e-9
                && last_processed
                    .map(|last| sample.time_tag > last)
                    .unwrap_or(true)
        })
        .cloned()
        .collect();
    if last_processed.is_none() && pending.len() > DETECTOR_WARMUP_SAMPLES {
        pending = pending.split_off(pending.len() - DETECTOR_WARMUP_SAMPLES);
    }
    pending
}

/// Configuration for the solar monitor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SolarConfig {
    /// Feed polling interval in seconds.
    pub poll_interval_secs: u64,
    /// Fused detector alert threshold (0..1).
    pub fusion_alert_threshold: f64,
    /// Pathway weights [forbush, heep, ssc, mansurov, lunar].
    pub pathway_weights: [f64; 5],
    /// Data directory for persistent JSONL logs. Logs go to `{data_dir}/live/`.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
}

fn default_data_dir() -> String {
    "data".to_string()
}

impl Default for SolarConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: 60,
            fusion_alert_threshold: 0.7,
            pathway_weights: [1.0; 5],
            data_dir: default_data_dir(),
        }
    }
}

/// Shared state for the solar monitor.
#[derive(Clone)]
pub struct SolarMonitorState {
    pub feeds: Arc<RwLock<FeedState>>,
    pub detector: Arc<RwLock<RankFusionDetector>>,
    pub escalation: Arc<RwLock<EscalationMonitor>>,
    pub stressor: Arc<RwLock<StressorIndex>>,
    pub config: Arc<RwLock<SolarConfig>>,
    pub alert_tx: broadcast::Sender<SolarAlert>,
    pub metrics_tx: broadcast::Sender<SolarMetrics>,
    pub logger: Arc<Mutex<LiveLogger>>,
    last_detector_observation: Arc<Mutex<Option<chrono::DateTime<chrono::Utc>>>>,
    detector_alert_active: Arc<Mutex<bool>>,
    detector_samples: Arc<Mutex<usize>>,
}

/// Alert event broadcast via SSE.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SolarAlert {
    pub alert_type: AlertType,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub severity: f64,
    /// Number of observational detectors agreeing on anomaly (0-6).
    pub detector_agreement: usize,
    pub source: &'static str,
    pub observation_time: Option<chrono::DateTime<chrono::Utc>>,
    pub data_status: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertType {
    /// Escalation level changed (Quiet → Elevated → Active → Flare).
    Escalation,
    /// Multi-detector flare onset confirmed.
    FlareOnset,
    ForbushPrediction,
    HeepEvent,
    SscPrecursor,
    StressorThreshold,
}

/// Periodic metrics broadcast via SSE.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SolarMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub xray_flux: Option<f64>,
    pub electron_flux: Option<f64>,
    pub solar_wind_speed: Option<f64>,
    pub bz: Option<f64>,
    pub kp: Option<f64>,
    /// Fused anomaly score from rank fusion (0..1).
    pub fused_flare_score: Option<f64>,
    /// Per-detector diagnostics.
    pub fusion_diagnostics: FusionDiagnostics,
    /// Current escalation level and status.
    pub escalation: EscalationStatus,
    pub stressor_total: f64,
    pub pathway_scores: Vec<f64>,
    pub stressor: StressorScore,
    pub data_quality: FeedQuality,
}

impl SolarMonitorState {
    pub fn new(config: SolarConfig) -> Self {
        let (alert_tx, _) = broadcast::channel(256);
        let (metrics_tx, _) = broadcast::channel(256);
        let logger = LiveLogger::new(&config.data_dir);

        Self {
            feeds: Arc::new(RwLock::new(FeedState::new())),
            detector: Arc::new(RwLock::new(RankFusionDetector::new(
                config.fusion_alert_threshold,
            ))),
            escalation: Arc::new(RwLock::new(EscalationMonitor::new())),
            stressor: Arc::new(RwLock::new(StressorIndex::new())),
            config: Arc::new(RwLock::new(config)),
            alert_tx,
            metrics_tx,
            logger: Arc::new(Mutex::new(logger)),
            last_detector_observation: Arc::new(Mutex::new(None)),
            detector_alert_active: Arc::new(Mutex::new(false)),
            detector_samples: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn quality(&self) -> FeedQuality {
        let feeds = self.feeds.read().await;
        let samples = *self.detector_samples.lock().await;
        feeds
            .quality(chrono::Utc::now())
            .with_detector_samples(samples, DETECTOR_WARMUP_SAMPLES)
    }

    /// Spawn the feed polling loop. Returns a JoinHandle.
    pub fn spawn_poll_loop(&self) -> tokio::task::JoinHandle<()> {
        let state = self.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client");

            loop {
                let interval = {
                    let cfg = state.config.read().await;
                    Duration::from_secs(cfg.poll_interval_secs)
                };

                // Fetch all feeds concurrently
                let (xray_res, electron_res, proton_res, sw_res, kp_res, sharp_res) = tokio::join!(
                    crate::feeds::xray::fetch_both(&client),
                    crate::feeds::electrons::fetch(&client),
                    crate::feeds::protons::fetch(&client),
                    crate::feeds::solar_wind::fetch(&client),
                    crate::feeds::kp_dst::fetch(&client),
                    crate::feeds::sharp::fetch_latest(&client),
                );

                let any_observations = xray_res
                    .as_ref()
                    .map(|both| !both.long.is_empty())
                    .unwrap_or(false)
                    || electron_res
                        .as_ref()
                        .map(|v| !v.is_empty())
                        .unwrap_or(false)
                    || proton_res.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
                    || sw_res.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
                    || kp_res.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
                    || sharp_res.as_ref().map(|v| !v.is_empty()).unwrap_or(false);

                // Update feed state
                {
                    let mut feeds = state.feeds.write().await;
                    feeds.errors.clear();
                    let poll_time = chrono::Utc::now();
                    feeds.last_poll = Some(poll_time);

                    match xray_res {
                        Ok(both) => {
                            feeds.append_xray(both.long);
                            feeds.append_xray_short(both.short);
                        }
                        Err(e) => {
                            tracing::warn!("X-ray feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }
                    match electron_res {
                        Ok(samples) => feeds.append_electrons(samples),
                        Err(e) => {
                            tracing::warn!("Electron feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }
                    match proton_res {
                        Ok(samples) => feeds.append_protons(samples),
                        Err(e) => {
                            tracing::warn!("Proton feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }
                    match sw_res {
                        Ok(samples) => feeds.append_solar_wind(samples),
                        Err(e) => {
                            tracing::warn!("Solar wind feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }
                    match kp_res {
                        Ok(samples) => feeds.append_kp_dst(samples),
                        Err(e) => {
                            tracing::warn!("Kp/Dst feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }
                    match sharp_res {
                        Ok(records) => {
                            if !records.is_empty() {
                                tracing::debug!("SHARP: {} active regions", records.len());
                            }
                            feeds.sharp = records;
                        }
                        Err(e) => {
                            tracing::warn!("SHARP feed error: {}", e);
                            feeds.errors.push(e);
                        }
                    }

                    if any_observations {
                        feeds.last_update = Some(poll_time);
                    }
                }

                // Warm with distinct real XRS observations, then process every new
                // observation once. Optional channels are used only when measured,
                // fresh, and contemporaneous with the live XRS sample.
                let processed_observation = {
                    let feeds = state.feeds.read().await;
                    let feed_quality = feeds.quality(chrono::Utc::now());
                    let mut detector = state.detector.write().await;
                    let mut last_processed = state.last_detector_observation.lock().await;
                    let mut sample_count = state.detector_samples.lock().await;
                    let was_ready = *sample_count >= DETECTOR_WARMUP_SAMPLES;

                    let pending = pending_xray_observations(&feeds, *last_processed);

                    let latest_time = feeds.xray.back().map(|sample| sample.time_tag);
                    let mut processed_any = false;
                    for observation in pending {
                        let is_live_tail = Some(observation.time_tag) == latest_time;
                        let short = feeds
                            .xray_short
                            .iter()
                            .find(|sample| sample.time_tag == observation.time_tag)
                            .map(|sample| sample.flux);

                        let electron = if is_live_tail && feed_quality.electrons.fresh {
                            feeds.electrons.back().and_then(|sample| {
                                ((observation.time_tag - sample.time_tag).num_seconds().abs()
                                    <= 10 * 60)
                                    .then_some(sample.flux)
                            })
                        } else {
                            None
                        };
                        let proton = if is_live_tail && feed_quality.protons.fresh {
                            feeds.protons.back().and_then(|sample| {
                                ((observation.time_tag - sample.time_tag).num_seconds().abs()
                                    <= 10 * 60)
                                    .then_some(sample.flux)
                            })
                        } else {
                            None
                        };
                        let bfield = if is_live_tail && feed_quality.solar_wind.fresh {
                            feeds.solar_wind.back().and_then(|sample| {
                                ((observation.time_tag - sample.time_tag).num_seconds().abs()
                                    <= 5 * 60)
                                    .then_some((sample.bx, sample.by, sample.bz))
                            })
                        } else {
                            None
                        };

                        if is_live_tail && feed_quality.kp_dst.fresh {
                            if let Some(kp_sample) = feeds.kp_dst.back() {
                                detector.update_kp(kp_sample.kp);
                            }
                        }
                        detector.ingest_available(
                            observation.flux,
                            short,
                            electron,
                            proton,
                            bfield,
                            observation.time_tag,
                        );
                        *last_processed = Some(observation.time_tag);
                        *sample_count += 1;
                        processed_any = true;
                    }

                    // The initialization replay establishes baselines but cannot
                    // itself emit a live transition.
                    was_ready && processed_any
                };

                // Alert only on the false -> true edge. A sustained detector state
                // remains visible in status but does not spam identical SSE alerts.
                let flare_onset = if processed_observation {
                    let detector = state.detector.read().await;
                    let is_alert = detector.is_anomalous();
                    let mut was_alert = state.detector_alert_active.lock().await;
                    let onset = if is_alert && !*was_alert {
                        detector.onset_event()
                    } else {
                        None
                    };
                    *was_alert = is_alert;
                    onset
                } else {
                    None
                };

                // Update escalation monitor
                if processed_observation {
                    let feeds = state.feeds.read().await;
                    let detector = state.detector.read().await;
                    let mut esc = state.escalation.write().await;
                    let now = chrono::Utc::now();
                    let hardness_score = detector.hardness.score();
                    let fused_score = detector.score();
                    let agreement = detector.detector_agreement();
                    let xray_flux = feeds.xray.back().map(|s| s.flux).unwrap_or(0.0);
                    let criticality_score = detector.criticality.score();
                    let samples = *state.detector_samples.lock().await;
                    let quality = feeds
                        .quality(now)
                        .with_detector_samples(samples, DETECTOR_WARMUP_SAMPLES);

                    if let Some(transition) = esc.update_full(
                        hardness_score,
                        fused_score,
                        agreement,
                        xray_flux,
                        criticality_score,
                        now,
                    ) {
                        let severity = transition.to.numeric() as f64 / 3.0;
                        let _ = state.alert_tx.send(SolarAlert {
                            alert_type: AlertType::Escalation,
                            message: format!(
                                "Escalation: {} → {} — {}",
                                transition.from.label(),
                                transition.to.label(),
                                transition.reason,
                            ),
                            timestamp: transition.timestamp,
                            severity,
                            detector_agreement: agreement,
                            source: "NOAA SWPC GOES XRS + rank-fusion detector",
                            observation_time: feeds.xray.back().map(|sample| sample.time_tag),
                            data_status: quality.status,
                        });
                    }
                }

                // Emit flare alert if fused detector triggers
                if let Some(ref onset) = flare_onset {
                    let (agreement, quality) = {
                        let detector = state.detector.read().await;
                        let feeds = state.feeds.read().await;
                        let samples = *state.detector_samples.lock().await;
                        (
                            detector.detector_agreement(),
                            feeds
                                .quality(chrono::Utc::now())
                                .with_detector_samples(samples, DETECTOR_WARMUP_SAMPLES),
                        )
                    };
                    let _ = state.alert_tx.send(SolarAlert {
                        alert_type: AlertType::FlareOnset,
                        message: format!(
                            "{}-class flare detected (fused score {:.2}, {}/6 observational detectors agree), flux {:.2e} W/m²",
                            onset.class.label(),
                            onset.anomaly_score,
                            agreement,
                            onset.peak_flux,
                        ),
                        timestamp: onset.timestamp,
                        severity: onset.anomaly_score,
                        detector_agreement: agreement,
                        source: "NOAA SWPC GOES XRS + rank-fusion detector",
                        observation_time: Some(onset.timestamp),
                        data_status: quality.status,
                    });
                }

                // Update coupling pathways
                {
                    let feeds = state.feeds.read().await;
                    let mut stressor = state.stressor.write().await;
                    stressor.update(&feeds, flare_onset.as_ref());
                }

                // Broadcast metrics + persist to JSONL
                {
                    let feeds = state.feeds.read().await;
                    let stressor = state.stressor.read().await;
                    let detector = state.detector.read().await;
                    let esc = state.escalation.read().await;
                    let now = chrono::Utc::now();
                    let score = stressor.compute();
                    let diag = detector.diagnostics();
                    let esc_status = esc.status(now);
                    let samples = *state.detector_samples.lock().await;
                    let quality = feeds
                        .quality(now)
                        .with_detector_samples(samples, DETECTOR_WARMUP_SAMPLES);

                    let metrics = SolarMetrics {
                        timestamp: now,
                        xray_flux: quality
                            .xray
                            .fresh
                            .then(|| feeds.xray.back().map(|s| s.flux))
                            .flatten(),
                        electron_flux: quality
                            .electrons
                            .fresh
                            .then(|| feeds.electrons.back().map(|s| s.flux))
                            .flatten(),
                        solar_wind_speed: quality
                            .solar_wind
                            .fresh
                            .then(|| feeds.solar_wind.back().map(|s| s.speed))
                            .flatten(),
                        bz: quality
                            .solar_wind
                            .fresh
                            .then(|| feeds.solar_wind.back().map(|s| s.bz))
                            .flatten(),
                        kp: quality
                            .kp_dst
                            .fresh
                            .then(|| feeds.kp_dst.back().map(|s| s.kp))
                            .flatten(),
                        fused_flare_score: quality.alerting_ready.then(|| detector.score()),
                        fusion_diagnostics: diag.clone(),
                        escalation: esc_status.clone(),
                        stressor_total: score.total,
                        pathway_scores: score.pathways.iter().map(|p| p.score).collect(),
                        stressor: score,
                        data_quality: quality,
                    };

                    let _ = state.metrics_tx.send(metrics);

                    // Persist snapshot to JSONL
                    let record = LiveRecord::from_state(&feeds, &diag, &esc_status);
                    let mut logger = state.logger.lock().await;
                    logger.append(&record);
                }

                tokio::time::sleep(interval).await;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    #[tokio::test]
    async fn readiness_requires_200_distinct_observed_xray_samples() {
        let state = SolarMonitorState::new(SolarConfig::default());
        let now = Utc::now();
        let mut observations = Vec::new();
        for i in 0..DETECTOR_WARMUP_SAMPLES {
            let sample = crate::feeds::xray::XraySample {
                time_tag: now - Duration::seconds((DETECTOR_WARMUP_SAMPLES - i) as i64),
                satellite: 18,
                flux: 5e-7,
                current_class: None,
            };
            observations.push(sample.clone());
            observations.push(sample);
        }
        {
            let mut feeds = state.feeds.write().await;
            feeds.append_xray(observations);
            assert_eq!(feeds.xray.len(), DETECTOR_WARMUP_SAMPLES);
            assert_eq!(
                pending_xray_observations(&feeds, None).len(),
                DETECTOR_WARMUP_SAMPLES
            );
        }

        *state.detector_samples.lock().await = DETECTOR_WARMUP_SAMPLES - 1;
        let warming = state.quality().await;
        assert_eq!(warming.status, "warming_up");
        assert!(!warming.detector_ready);
        assert!(!warming.alerting_ready);

        *state.detector_samples.lock().await = DETECTOR_WARMUP_SAMPLES;
        let ready = state.quality().await;
        assert!(ready.detector_ready);
        assert!(ready.alerting_ready);
    }
}
