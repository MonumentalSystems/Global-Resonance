use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::state::{SolarConfig, SolarMonitorState};

/// Build the solar monitor router.
pub fn solar_routes(state: SolarMonitorState) -> Router {
    Router::new()
        .route("/api/solar/status", get(get_status))
        .route("/api/solar/feeds", get(get_feeds))
        .route("/api/solar/feeds/xray", get(get_xray))
        .route("/api/solar/feeds/electrons", get(get_electrons))
        .route("/api/solar/feeds/solar-wind", get(get_solar_wind))
        .route("/api/solar/feeds/kp-dst", get(get_kp_dst))
        .route("/api/solar/pathways", get(get_pathways))
        .route("/api/solar/detectors", get(get_detectors))
        .route("/api/solar/escalation", get(get_escalation))
        .route("/api/solar/alerts", get(sse_alerts))
        .route("/api/solar/metrics", get(sse_metrics))
        .route("/api/solar/health", get(get_health))
        .route("/api/solar/state", get(get_solar_state))
        .route("/api/solar/config", post(post_config))
        .with_state(state)
}

/// GET /api/solar/status — Current stressor index + all pathway scores.
async fn get_status(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    let quality = feeds.quality(chrono::Utc::now());
    let stressor = state.stressor.read().await;
    let score = stressor.compute();
    let detector = state.detector.read().await;

    let diag = detector.diagnostics();

    let esc = state.escalation.read().await;
    let now = chrono::Utc::now();

    Json(serde_json::json!({
        "escalation": esc.status(now),
        "stressor": score,
        "flare_detected": quality.alerting_ready && detector.is_anomalous(),
        "fused_flare_score": quality.alerting_ready.then(|| detector.score()),
        "detector_agreement": quality.alerting_ready.then(|| detector.detector_agreement()),
        "fusion_diagnostics": diag,
        "data_quality": quality,
        "semantics": {
            "alert_scope": "observed solar X-ray anomaly detection",
            "coupling_scope": "experimental research indicators",
            "forecast": false,
            "note": "Not an operational earthquake, weather, or CME forecast.",
        },
    }))
}

/// GET /api/solar/escalation — Current escalation level and precursor status.
async fn get_escalation(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let esc = state.escalation.read().await;
    let feeds = state.feeds.read().await;
    let now = chrono::Utc::now();
    Json(serde_json::json!({
        "escalation": esc.status(now),
        "data_quality": feeds.quality(now),
    }))
}

/// GET /api/solar/feeds — Latest values from all feeds.
async fn get_feeds(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "last_update": feeds.last_update,
        "last_poll": feeds.last_poll,
        "xray_count": feeds.xray.len(),
        "electron_count": feeds.electrons.len(),
        "solar_wind_count": feeds.solar_wind.len(),
        "kp_dst_count": feeds.kp_dst.len(),
        "xray_latest": feeds.xray.back(),
        "electron_latest": feeds.electrons.back(),
        "solar_wind_latest": feeds.solar_wind.back(),
        "kp_dst_latest": feeds.kp_dst.back(),
        "errors": feeds.errors,
        "data_quality": feeds.quality(chrono::Utc::now()),
    }))
}

/// GET /api/solar/feeds/xray — Full X-ray ring buffer.
async fn get_xray(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "count": feeds.xray.len(),
        "samples": feeds.xray,
    }))
}

/// GET /api/solar/feeds/electrons — Full electron flux ring buffer.
async fn get_electrons(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "count": feeds.electrons.len(),
        "samples": feeds.electrons,
    }))
}

/// GET /api/solar/feeds/solar-wind — Full solar wind ring buffer.
async fn get_solar_wind(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "count": feeds.solar_wind.len(),
        "samples": feeds.solar_wind,
    }))
}

/// GET /api/solar/feeds/kp-dst — Full Kp/Dst ring buffer.
async fn get_kp_dst(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "count": feeds.kp_dst.len(),
        "samples": feeds.kp_dst,
    }))
}

/// GET /api/solar/pathways — All 5 pathway statuses.
async fn get_pathways(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let stressor = state.stressor.read().await;
    let score = stressor.compute();
    let feeds = state.feeds.read().await;
    Json(serde_json::json!({
        "pathways": score.pathways,
        "total": score.total,
        "timestamp": score.timestamp,
        "data_quality": feeds.quality(chrono::Utc::now()),
    }))
}

/// GET /api/solar/detectors — Rank fusion diagnostics (all 5 detectors + fused score).
async fn get_detectors(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let detector = state.detector.read().await;
    let feeds = state.feeds.read().await;
    let quality = feeds.quality(chrono::Utc::now());
    Json(serde_json::json!({
        "fusion_diagnostics": detector.diagnostics(),
        "alerting_ready": quality.alerting_ready,
        "data_quality": quality,
    }))
}

/// GET /api/solar/alerts — SSE stream of flare and coupling alerts.
async fn sse_alerts(
    State(state): State<SolarMonitorState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.alert_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        result.ok().map(|alert| {
            Ok(Event::default()
                .event("alert")
                .json_data(&alert)
                .unwrap_or_else(|_| Event::default().data("error")))
        })
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// GET /api/solar/metrics — SSE stream of periodic metrics.
async fn sse_metrics(
    State(state): State<SolarMonitorState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.metrics_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|result| {
        result.ok().map(|metrics| {
            Ok(Event::default()
                .event("metrics")
                .json_data(&metrics)
                .unwrap_or_else(|_| Event::default().data("error")))
        })
    });

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// GET /api/solar/health — Feed freshness check.
async fn get_health(State(state): State<SolarMonitorState>) -> impl IntoResponse {
    let feeds = state.feeds.read().await;
    let now = chrono::Utc::now();
    let quality = feeds.quality(now);
    let status_code = if quality.alerting_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status_code, Json(serde_json::json!({
        "status": quality.status,
        "alerting_ready": quality.alerting_ready,
        "data_quality": quality,
        "errors": feeds.errors,
    })))
}

/// GET /api/solar/state — Complete solar state snapshot (all layers).
/// Fetches live data from DONKI + SWPC on each request (~2-5s).
async fn get_solar_state(State(_state): State<SolarMonitorState>) -> impl IntoResponse {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap();

    match crate::solar_state::donki::fetch_solar_state(&client).await {
        Ok(solar_state) => Json(serde_json::to_value(&solar_state).unwrap()).into_response(),
        Err(e) => Json(serde_json::json!({
            "error": e,
        }))
        .into_response(),
    }
}

/// POST /api/solar/config — Update configuration.
async fn post_config(
    State(state): State<SolarMonitorState>,
    Json(new_config): Json<SolarConfig>,
) -> impl IntoResponse {
    if std::env::var("SOLAR_MONITOR_ALLOW_CONFIG_WRITE").as_deref() != Ok("1") {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "runtime configuration mutation is disabled",
                "hint": "set SOLAR_MONITOR_ALLOW_CONFIG_WRITE=1 only on a trusted internal deployment",
            })),
        );
    }
    let mut config = state.config.write().await;
    *config = new_config.clone();

    // Update stressor weights
    let mut stressor = state.stressor.write().await;
    stressor.weights = new_config.pathway_weights;

    (StatusCode::OK, Json(serde_json::json!({
        "status": "updated",
        "config": new_config,
    })))
}
