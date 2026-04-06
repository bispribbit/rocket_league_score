//! Binary: Verify downloaded replays exist on disk.
//!
//! Aligns `download_status` in the database with files under `REPLAY_BASE_PATH` (same logic as
//! `database::synchronize_replay_download_status_with_filesystem`, which runs automatically when
//! starting `ballchasing_downloader`).
//!
//! Environment variables:
//! - `DATABASE_URL` — PostgreSQL connection string (required)
//! - `REPLAY_BASE_PATH` — Base path for replay files (optional; platform default if unset)

use std::path::PathBuf;

use anyhow::Result;
use database::{initialize_pool, synchronize_replay_download_status_with_filesystem};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    #[cfg(target_os = "linux")]
    let base_path_unwrap = PathBuf::from("/workspace/ballchasing");

    #[cfg(target_os = "windows")]
    let base_path_unwrap = PathBuf::from(r"C:\github\rocket-league-score\ballchasing");

    let base_path = match std::env::var("REPLAY_BASE_PATH") {
        Ok(path) => PathBuf::from(path),
        Err(_) => base_path_unwrap,
    };

    info!("=== Verify Downloaded Replays ===");
    info!("Base path: {}\n", base_path.display());

    let summary = synchronize_replay_download_status_with_filesystem(&base_path).await?;

    info!(
        "\n=== Failed downloads reset to not_downloaded (for retry) ===\n{}",
        summary.failed_reset_to_not_downloaded
    );

    info!("\n=== Summary (Downloaded Replays) ===");
    info!(
        "Total downloaded replays: {}",
        summary.downloaded_rows_checked
    );
    info!(
        "Existing on disk:         {}",
        summary.downloaded_existing_on_disk
    );
    info!(
        "Missing (marked as not_downloaded): {}",
        summary.missing_files_reset_to_not_downloaded
    );

    info!("\n=== Summary (Not Downloaded Replays) ===");
    info!(
        "Total not_downloaded replays: {}",
        summary.not_downloaded_rows_checked
    );
    info!(
        "Found on disk (marked as downloaded): {}",
        summary.existing_files_marked_downloaded
    );
    info!("Still missing: {}", summary.not_downloaded_still_missing);

    Ok(())
}
