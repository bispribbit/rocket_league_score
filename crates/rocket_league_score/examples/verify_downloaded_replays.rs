//! Example: Verify downloaded replays exist on disk.
//!
//! This script:
//! - Selects all replays where `download_status` = 'downloaded'
//! - Checks if each file path exists on disk
//! - Marks replays as '`not_downloaded`' if the file doesn't exist
//!
//! Usage:
//!   cargo run --example `verify_downloaded_replays`
//!
//! Environment Variables:
//!   `DATABASE_URL` - `PostgreSQL` connection string (required)
//!   `REPLAY_BASE_PATH` - Base path for replay files (required)

use std::path::PathBuf;

use anyhow::Result;
use database::{get_pool, initialize_pool};
use replay_structs::DownloadStatus;
use tracing::info;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

/// Represents a downloaded replay record from the database.
struct DownloadedReplay {
    id: Uuid,
    file_path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    // Initialize database
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    // Get the base path for replay files
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

    // Select all replays where download_status = 'downloaded'
    let pool = get_pool();
    let rows = sqlx::query!(
        "
        SELECT id, file_path
        FROM replays
        WHERE download_status = 'downloaded'
        "
    )
    .fetch_all(pool)
    .await?;

    let downloaded_replays: Vec<DownloadedReplay> = rows
        .into_iter()
        .map(|row| DownloadedReplay {
            id: row.id,
            file_path: row.file_path,
        })
        .collect();

    info!(
        "Found {} replays with download_status = 'downloaded'\n",
        downloaded_replays.len()
    );

    // Check each file path and mark as 'not_downloaded' if it doesn't exist
    let mut missing_count = 0;
    let mut existing_count = 0;

    for replay in &downloaded_replays {
        let path = base_path.join(&replay.file_path);

        if !path.exists() {
            info!("Missing: {} (ID: {})", replay.file_path, replay.id);

            // Mark as 'not_downloaded'
            sqlx::query!(
                "
                UPDATE replays
                SET download_status = $1, updated_at = NOW()
                WHERE id = $2
                ",
                DownloadStatus::NotDownloaded as DownloadStatus,
                replay.id
            )
            .execute(pool)
            .await?;

            missing_count += 1;
        } else {
            existing_count += 1;
        }
    }

    info!("\n=== Summary ===");
    info!("Total downloaded replays: {}", downloaded_replays.len());
    info!("Existing on disk:         {existing_count}");
    info!("Missing (marked as not_downloaded): {missing_count}");

    Ok(())
}
