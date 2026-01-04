//! Ballchasing.com replay downloader.
//!
//! Downloads Rocket League replays from ballchasing.com organized by rank.

use anyhow::Result;
use config::CONFIG;
use database::{initialize_pool, run_migrations};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with console and file output
    // Ensure directory exists
    std::fs::create_dir_all("/workspace/target/logs").expect("Failed to create .cursor directory");
    let log_file = std::fs::File::create("/workspace/target/logs/debug.log")
        .expect("Failed to create debug log file");

    let env_filter = EnvFilter::new("info");

    // Console layer
    let console_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_writer(std::io::stdout);

    // File layer
    let file_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_writer(log_file);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    info!("Ballchasing replay downloader starting");

    // Connect to database
    initialize_pool(&CONFIG.database_url).await?;

    // Run migrations
    run_migrations().await?;

    // Run the downloader
    ballchasing_downloader::run().await?;

    Ok(())
}
