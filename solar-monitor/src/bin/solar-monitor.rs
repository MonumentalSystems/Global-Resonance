use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tower_http::cors::{Any, CorsLayer};

use solar_monitor::server::solar_routes;
use solar_monitor::state::{SolarConfig, SolarMonitorState};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Parse args
    let mut port: u16 = 8089;
    let mut host = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let mut poll_interval: u64 = 60;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8089);
            }
            "--host" => {
                i += 1;
                host = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
            }
            "--poll-interval" => {
                i += 1;
                poll_interval = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(60);
            }
            "--help" | "-h" => {
                println!("solar-monitor — Real-time solar flare monitoring and atmospheric coupling alerting");
                println!();
                println!("Usage: solar-monitor [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --port <PORT>            HTTP port (default: 8089)");
                println!("  --host <ADDRESS>         Bind address (default: 127.0.0.1)");
                println!(
                    "  --poll-interval <SECS>   Feed polling interval in seconds (default: 60)"
                );
                println!("  -h, --help               Show this help");
                println!();
                println!("Endpoints:");
                println!("  GET  /api/solar/status    Current stressor index + pathway scores");
                println!("  GET  /api/solar/feeds     Latest feed values");
                println!("  GET  /api/solar/pathways  All 5 coupling pathway statuses");
                println!("  GET  /api/solar/alerts    SSE stream of flare/coupling alerts");
                println!("  GET  /api/solar/metrics   SSE stream of periodic metrics");
                println!("  GET  /api/solar/health    Feed freshness check");
                println!("  POST /api/solar/config    Update configuration");
                return;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    let config = SolarConfig {
        poll_interval_secs: poll_interval,
        ..Default::default()
    };

    let state = SolarMonitorState::new(config);

    // Start feed polling loop
    let _poll_handle = state.spawn_poll_loop();
    tracing::info!("Feed polling started (interval: {}s)", poll_interval);

    // Build router
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = solar_routes(state).layer(cors);

    let addr = SocketAddr::new(host, port);
    tracing::info!("Solar monitor listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind");
    axum::serve(listener, app).await.expect("Server error");
}
