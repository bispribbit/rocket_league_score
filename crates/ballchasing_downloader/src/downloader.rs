//! Core downloader logic that runs fetch and download in parallel.

use core::fmt::Write as _;
use core::time::Duration;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use config::OBJECT_STORE;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_structs::{BallchasingRank, DownloadStatus, ReplaySummary};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::api::client::BallchasingClient;

/// Target number of replays per rank.
const TARGET_REPLAYS_PER_RANK: usize = 1100;

/// How often to log progress (in seconds).
const PROGRESS_LOG_INTERVAL_SECONDS: u64 = 30;

/// Maximum concurrent downloads.
const MAX_CONCURRENT_DOWNLOADS: usize = 2;

/// Runs the complete download process.
///
/// Fetches metadata and downloads replays in parallel until complete.
///
/// # Errors
///
/// Returns an error if the process fails.
pub async fn run() -> Result<()> {
    let client = Arc::new(BallchasingClient::new()?);

    // Reset any stuck in-progress downloads
    let reset_count = database::reset_in_progress_ballchasing_downloads().await?;
    if reset_count > 0 {
        info!("Reset {reset_count} in-progress downloads");
    }

    // Run fetch and download tasks in parallel
    let fetch_client = Arc::clone(&client);
    let download_client = Arc::clone(&client);

    let fetch_task = tokio::spawn(async move { fetch_all_metadata(&fetch_client).await });

    let download_task = tokio::spawn(async move { download_all_replays(download_client).await });

    // Wait for both tasks
    let (fetch_result, download_result) = tokio::join!(fetch_task, download_task);

    fetch_result??;
    download_result??;

    info!("Download process complete");
    print_stats().await?;

    Ok(())
}

/// Fetches metadata for all ranks until we have enough replays.
async fn fetch_all_metadata(client: &BallchasingClient) -> Result<()> {
    info!("Starting metadata fetch (target: {TARGET_REPLAYS_PER_RANK} per rank)");

    loop {
        let mut all_complete = true;

        for rank in BallchasingRank::all_ranked() {
            let current = database::count_ballchasing_replays_by_rank(rank).await?;

            if current >= TARGET_REPLAYS_PER_RANK as i64 {
                continue;
            }

            all_complete = false;
            let needed = TARGET_REPLAYS_PER_RANK - current as usize;

            info!("Rank {rank}: have {current}, fetching up to {needed} more");

            // Get existing IDs
            let existing = database::list_ballchasing_replays_by_rank(rank, None).await?;
            let existing_ids: HashSet<Uuid> = existing.iter().map(|r| r.id).collect();

            // Fetch from API
            let replays: Vec<ReplaySummary> = client
                .fetch_replays_for_rank(rank.as_api_string(), needed, &existing_ids)
                .await?;

            // Store new replays
            let mut ids = Vec::new();
            let mut ranks = Vec::new();
            let mut metadata_values = Vec::new();

            for replay in replays {
                let Ok(id) = replay.id.parse::<Uuid>() else {
                    warn!("Invalid replay ID: {}", replay.id);
                    continue;
                };

                if existing_ids.contains(&id) {
                    continue;
                }

                let metadata = serde_json::to_value(&replay)?;
                ids.push(id);
                ranks.push(rank);
                metadata_values.push(metadata);
            }

            if !ids.is_empty() {
                let created =
                    database::insert_ballchasing_replays(&ids, &ranks, &metadata_values).await?;
                info!("Stored {created} new replays for rank {rank}");
            }
        }

        if all_complete {
            info!("All ranks have {TARGET_REPLAYS_PER_RANK} replays");
            break;
        }

        // Small delay before checking again
        sleep(Duration::from_secs(1)).await;
    }

    Ok(())
}

/// Downloads all pending replays.
async fn download_all_replays(client: Arc<BallchasingClient>) -> Result<()> {
    info!("Starting replay downloads");

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_DOWNLOADS));
    let mut last_progress_log = std::time::Instant::now();

    loop {
        // Get batch of pending downloads
        let pending = database::list_pending_ballchasing_downloads(None, 100).await?;

        if pending.is_empty() {
            // Check if fetch is still running by waiting a bit and checking again
            sleep(Duration::from_secs(2)).await;

            let still_pending = database::list_pending_ballchasing_downloads(None, 1).await?;

            if still_pending.is_empty() {
                // Double-check all ranks are complete
                let mut all_downloaded = true;
                for rank in BallchasingRank::all_ranked() {
                    let not_downloaded = database::count_ballchasing_replays_by_rank_and_status(
                        rank,
                        DownloadStatus::NotDownloaded,
                    )
                    .await?;
                    if not_downloaded > 0 {
                        all_downloaded = false;
                        break;
                    }
                }

                if all_downloaded {
                    info!("All downloads complete");
                    break;
                }
            }
            continue;
        }

        // Log progress periodically
        if last_progress_log.elapsed() > Duration::from_secs(PROGRESS_LOG_INTERVAL_SECONDS) {
            log_download_progress().await?;
            last_progress_log = std::time::Instant::now();
        }

        // Download in parallel with semaphore limiting concurrency
        let mut handles = Vec::new();

        for replay in pending {
            let permit = Arc::clone(&semaphore).acquire_owned().await?;
            let client = Arc::clone(&client);

            let handle = tokio::spawn(async move {
                let result = download_single(&client, replay.id, replay.rank).await;
                drop(permit);
                result
            });

            handles.push(handle);
        }

        // Wait for batch to complete
        for handle in handles {
            if let Err(error) = handle.await? {
                debug!("Download error: {error}");
            }
        }
    }

    Ok(())
}

/// Downloads a single replay file.
async fn download_single(
    client: &BallchasingClient,
    replay_id: Uuid,
    rank: BallchasingRank,
) -> Result<()> {
    // Mark as in progress
    database::mark_ballchasing_replay_in_progress(replay_id).await?;

    // Attempt download
    match download_and_save(client, replay_id, rank).await {
        Ok(relative_path) => {
            database::mark_ballchasing_replay_downloaded(replay_id, &relative_path).await?;
            debug!("Downloaded {replay_id}");
            Ok(())
        }
        Err(error) => {
            let error_msg = format!("{error:#}");
            database::mark_ballchasing_replay_failed(replay_id, &error_msg).await?;
            error!("Failed to download {replay_id}: {error}");
            Err(error)
        }
    }
}

/// Downloads replay data and saves to file.
///
/// Returns a relative path from the base data directory.
async fn download_and_save(
    client: &BallchasingClient,
    replay_id: Uuid,
    rank: BallchasingRank,
) -> Result<String> {
    let data: Bytes = client.download_replay(replay_id).await?;

    if data.is_empty() {
        anyhow::bail!("Downloaded replay is empty");
    }

    // Build object store path using forward slashes (object_store handles normalization)
    let object_path = ObjectStorePath::from(format!(
        "replays/3v3/{}/{}.replay",
        rank.as_folder_name(),
        replay_id
    ));

    // Write to object store
    OBJECT_STORE
        .put(&object_path, data.into())
        .await
        .context("Failed to write replay to object store")?;

    // Return relative path (object_store paths use forward slashes)
    Ok(object_path.to_string())
}

/// Logs current download progress.
async fn log_download_progress() -> Result<()> {
    let mut total_downloaded = 0i64;
    let mut total_pending = 0i64;
    let mut status = String::new();

    for rank in BallchasingRank::all_ranked() {
        let downloaded = database::count_ballchasing_replays_by_rank_and_status(
            rank,
            DownloadStatus::Downloaded,
        )
        .await?;
        let pending = database::count_ballchasing_replays_by_rank_and_status(
            rank,
            DownloadStatus::NotDownloaded,
        )
        .await?;

        total_downloaded += downloaded;
        total_pending += pending;

        if pending > 0 {
            let _ = write!(status, " {}:{}/{}", rank, downloaded, downloaded + pending);
        }
    }

    let total = total_downloaded + total_pending;
    let pct = if total > 0 {
        (total_downloaded as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    info!("Progress: {total_downloaded}/{total} ({pct:.1}%){status}");

    Ok(())
}

/// Prints final statistics.
async fn print_stats() -> Result<()> {
    info!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Rank", "Downloaded", "Failed", "Total"
    );
    info!("{}", "-".repeat(58));

    let mut grand_downloaded = 0i64;
    let mut grand_failed = 0i64;

    for rank in BallchasingRank::all_ranked() {
        let downloaded = database::count_ballchasing_replays_by_rank_and_status(
            rank,
            DownloadStatus::Downloaded,
        )
        .await?;
        let failed =
            database::count_ballchasing_replays_by_rank_and_status(rank, DownloadStatus::Failed)
                .await?;
        let total = downloaded + failed;

        info!(
            "{:<20} {:>12} {:>12} {:>12}",
            rank.as_api_string(),
            downloaded,
            failed,
            total
        );

        grand_downloaded += downloaded;
        grand_failed += failed;
    }

    info!("{}", "-".repeat(58));
    info!(
        "{:<20} {:>12} {:>12} {:>12}",
        "TOTAL",
        grand_downloaded,
        grand_failed,
        grand_downloaded + grand_failed
    );

    Ok(())
}
