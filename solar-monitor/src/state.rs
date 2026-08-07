use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, Mutex, RwLock};

use crate::coupling::{StressorIndex, StressorScore};
use crate::detection::escalation::{EscalationMonitor, EscalationStatus};
use crate::detection::rank_fusion::{FusionDiagnostics, RankFusionDetector};
use crate::feeds::{FeedQuality, FeedState};
use crate::persistence::{LiveLogger, LiveRecord};

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
    /// Experimental SHARP helicity interaction, exposed for validation only.
    /// This value is not used by rank fusion, escalation, or flare probability.
    pub sharp_helicity_diagnostic: Option<crate::feeds::sharp::HelicityInteractionDiagnostic>,
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
        }
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
                    || electron_res.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
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

                // Process each fresh XRS observation exactly once. Replaying the
                // cached tail changes detector history without new evidence.
                let processed_observation = {
                    let feeds = state.feeds.read().await;
                    let mut detector = state.detector.write().await;
                    let quality = feeds.quality(chrono::Utc::now());
                    let latest_time = feeds.xray.back().map(|sample| sample.time_tag);
                    let mut last_processed = state.last_detector_observation.lock().await;
                    let should_process = quality.alerting_ready
                        && latest_time.is_some()
                        && latest_time != *last_processed;

                    if !should_process {
                        false
                    } else {

                    let xray_long = feeds.xray.back().map(|s| s.flux).unwrap_or(0.0);
                    let xray_short = feeds
                        .xray_short
                        .back()
                        .map(|s| s.flux)
                        .unwrap_or(xray_long * 0.04);
                    let electron_flux = feeds.electrons.back().map(|s| s.flux).unwrap_or(100.0);
                    let proton_flux = feeds.protons.back().map(|s| s.flux).unwrap_or(0.3);

                    // Feed Kp into criticality detector before ingest.
                    if let Some(kp_sample) = feeds.kp_dst.back() {
                        detector.update_kp(kp_sample.kp);
                    }

                    if let Some(latest) = feeds.xray.back() {
                        // Use the highest-fidelity ingest path available:
                        // 1. SHARP + B-field: photospheric magnetogram + IMF commutator
                        // 2. B-field only: IMF commutator from DSCOVR/ACE
                        // 3. Scalar only: X-ray + proton flux
                        let sw = feeds.solar_wind.back();
                        // Use the most flare-prone active region (highest risk).
                        let best_sharp = feeds.sharp.iter().max_by(|a, b| {
                            crate::feeds::sharp::sharp_flare_risk(a)
                                .partial_cmp(&crate::feeds::sharp::sharp_flare_risk(b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });

                        match (best_sharp, sw) {
                            (Some(sharp), Some(sw)) => {
                                // Tier 1: SHARP + B-field (best)
                                detector.ingest_with_sharp(
                                    xray_long,
                                    xray_short,
                                    electron_flux,
                                    proton_flux,
                                    sw.bx,
                                    sw.by,
                                    sw.bz,
                                    sharp.usflux,
                                    sharp.meangbz,
                                    sharp.meanjzh,
                                    sharp.totusjh,
                                    sharp.shrgt45,
                                    sharp.r_value,
                                    sharp.totpot,
                                    sharp.totusjz,
                                    sharp.savncpp,
                                    sharp.absnjzh,
                                    sharp.meanalp,
                                    sharp.area_acr,
                                    latest.time_tag,
                                );
                            }
                            (_, Some(sw)) => {
                                // Tier 2: B-field only
                                detector.ingest_full(
                                    xray_long,
                                    xray_short,
                                    electron_flux,
                                    proton_flux,
                                    sw.bx,
                                    sw.by,
                                    sw.bz,
                                    latest.time_tag,
                                );
                            }
                            _ => {
                                // Tier 3: scalar only
                                detector.ingest(
                                    xray_long,
                                    xray_short,
                                    electron_flux,
                                    proton_flux,
                                    latest.time_tag,
                                );
                            }
                        }
                    }

                        *last_processed = latest_time;
                        true
                    }
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
                    let quality = feeds.quality(now);

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
                        (detector.detector_agreement(), feeds.quality(chrono::Utc::now()))
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
                    let quality = feeds.quality(now);

                    let metrics = SolarMetrics {
                        timestamp: now,
                        xray_flux: feeds.xray.back().map(|s| s.flux),
                        electron_flux: feeds.electrons.back().map(|s| s.flux),
                        solar_wind_speed: feeds.solar_wind.back().map(|s| s.speed),
                        bz: feeds.solar_wind.back().map(|s| s.bz),
                        kp: feeds.kp_dst.back().map(|s| s.kp),
                        sharp_helicity_diagnostic: feeds
                            .sharp
                            .iter()
                            .max_by(|a, b| {
                                crate::feeds::sharp::sharp_flare_risk(a)
                                    .partial_cmp(&crate::feeds::sharp::sharp_flare_risk(b))
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(crate::feeds::sharp::helicity_interaction_diagnostic),
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
