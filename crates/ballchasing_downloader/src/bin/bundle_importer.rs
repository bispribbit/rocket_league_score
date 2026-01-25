//! Bundle importer for local replay files.
//!
//! Imports replays from a local directory with a JSONL metadata file.

use std::path::Path;

use anyhow::Result;
use config::CONFIG;
use database::{initialize_pool, run_migrations};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, fmt};

/// Path to the bundle directory containing metadata.jsonl and replay files.
const BUNDLE_PATH: &str = "/workspace/replays/3v3";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with console and file output
    std::fs::create_dir_all("/workspace/target/logs").expect("Failed to create logs directory");
    let log_file = std::fs::File::create("/workspace/target/logs/bundle_importer.log")
        .expect("Failed to create log file");

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

    info!("Bundle importer starting");
    info!("Importing from: {BUNDLE_PATH}");

    // Connect to database
    initialize_pool(&CONFIG.database_url).await?;

    // Run migrations
    run_migrations().await?;

    // Run the bundle importer
    ballchasing_downloader::import_bundle(Path::new(BUNDLE_PATH)).await?;

    info!("Bundle import completed successfully");

    Ok(())
}
