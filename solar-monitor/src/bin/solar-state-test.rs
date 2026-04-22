#[tokio::main]
async fn main() {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();

    println!("Fetching solar state from live APIs...\n");
    match solar_monitor::solar_state::donki::fetch_solar_state(&client).await {
        Ok(state) => {
            println!("{}", serde_json::to_string_pretty(&state).unwrap());
        }
        Err(e) => println!("Error: {}", e),
    }
}
