//! Ballchasing.com replay downloader.
//!
//! Downloads Rocket League replays from ballchasing.com organized by rank.

use anyhow::Result;
use config::{CONFIG, get_base_path};
use database::{
    initialize_pool, run_migrations, synchronize_replay_download_status_with_filesystem,
};
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

    let replay_base = get_base_path();
    info!(
        "Synchronizing replay download status with files under {}",
        replay_base.display()
    );
    let sync_summary = synchronize_replay_download_status_with_filesystem(&replay_base).await?;
    info!(
        failed_reset = sync_summary.failed_reset_to_not_downloaded,
        downloaded_checked = sync_summary.downloaded_rows_checked,
        existing_on_disk = sync_summary.downloaded_existing_on_disk,
        missing_reset = sync_summary.missing_files_reset_to_not_downloaded,
        not_downloaded_checked = sync_summary.not_downloaded_rows_checked,
        found_marked_downloaded = sync_summary.existing_files_marked_downloaded,
        still_missing = sync_summary.not_downloaded_still_missing,
        "Replay download status sync finished"
    );

    // Run the downloader
    ballchasing_downloader::run().await?;

    Ok(())
}
