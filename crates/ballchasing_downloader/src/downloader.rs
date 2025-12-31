//! Core downloader logic that runs fetch and download in parallel.

use core::fmt::Write as _;
use core::str::FromStr;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use chrono::Utc;
use config::OBJECT_STORE;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_structs::{DownloadStatus, GameMode, Rank, Replay, ReplayPlayer, ReplaySummary};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::api::client::{BallchasingClient, RateLimitedError};
use crate::players::extract_players_from_metadata;

/// Target number of replays per rank.
const TARGET_REPLAYS_PER_RANK: usize = 1100;

/// How often to log progress (in seconds).
const PROGRESS_LOG_INTERVAL_SECONDS: u64 = 30;

/// Maximum concurrent downloads.
const MAX_CONCURRENT_DOWNLOADS: usize = 1;

/// How long to pause when rate limited (429 error).
/// With 1000 downloads/hour, we wait 10 minutes to let the rate limit reset.
const RATE_LIMIT_PAUSE_SECONDS: u64 = 600;

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
    let reset_count = database::reset_in_progress_downloads().await?;
    if reset_count > 0 {
        info!("Reset {reset_count} in-progress downloads");
    }

    let download_client = client.clone();
    let _replay = download_all_replays(download_client)
        .await
        .inspect_err(|err| {
            error!("Download error: {err}");
        });

    /*
       let fetch_client = client.clone();
       let download_client = client.clone();

       let fetch_task = tokio::spawn(async move { fetch_all_metadata(fetch_client).await });

       let download_task = tokio::spawn(async move { download_all_replays(download_client).await });

       // Wait for both tasks
       let (fetch_result, download_result) = tokio::join!(fetch_task, download_task);

       fetch_result??;
       download_result??;
    */

    info!("Download process complete");
    print_stats().await?;

    Ok(())
}

/// Fetches metadata for all ranks until we have enough replays.
async fn fetch_all_metadata(client: Arc<BallchasingClient>) -> Result<()> {
    info!("Starting metadata fetch (target: {TARGET_REPLAYS_PER_RANK} per rank)");

    loop {
        let mut all_complete = true;

        for rank in Rank::all_ranked() {
            let current = database::count_replays_by_rank(rank).await?;

            if current >= TARGET_REPLAYS_PER_RANK as i64 {
                continue;
            }

            all_complete = false;
            let needed = TARGET_REPLAYS_PER_RANK - current as usize;

            info!("Rank {rank}: have {current}, fetching up to {needed} more");

            // Get existing IDs
            let existing = database::list_replays_by_rank(rank, None).await?;
            let existing_ids: HashSet<Uuid> = existing.iter().map(|replay| replay.id).collect();

            // Fetch from API
            let replays: Vec<ReplaySummary> = client
                .fetch_replays_for_rank(rank.as_api_string(), needed, &existing_ids)
                .await?;

            // Store new replays
            let mut ids = Vec::new();
            let mut game_modes = Vec::new();
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

                // Extract game_mode from playlist_id, default to RankedStandard if not available
                let game_mode =
                    replay
                        .playlist_id
                        .as_ref()
                        .map_or(GameMode::RankedStandard, |playlist_id| {
                            GameMode::from_str(playlist_id).unwrap_or_else(|_| {
                                warn!(
                                    "Invalid playlist_id: {}, defaulting to ranked_standard",
                                    playlist_id
                                );
                                GameMode::RankedStandard
                            })
                        });

                let metadata = serde_json::to_value(&replay)?;
                ids.push(id);
                game_modes.push(game_mode);
                ranks.push(rank);
                metadata_values.push(metadata);
            }

            if ids.is_empty() {
                continue;
            }

            let created =
                database::insert_replays(&ids, &game_modes, &ranks, &metadata_values).await?;
            info!("Stored {created} new replays for rank {rank}");

            // Extract and insert players for each new replay
            let mut all_players = Vec::new();

            for (replay_id, metadata) in ids.iter().zip(metadata_values.iter()) {
                match extract_players_from_metadata(metadata) {
                    Ok(players) => {
                        for player in players {
                            all_players.push(ReplayPlayer {
                                id: 0, // Will be auto-generated by database
                                replay_id: *replay_id,
                                player_name: player.player_name,
                                team: player.team,
                                rank_division: player.rank_division,
                                created_at: Utc::now(), // Will be auto-generated by database
                            });
                        }
                    }
                    Err(e) => {
                        warn!(
                            replay_id = %replay_id,
                            error = %e,
                            "Failed to extract players from metadata"
                        );
                    }
                }
            }

            if all_players.is_empty() {
                continue;
            }

            database::insert_replay_players(&all_players).await?;
            info!(
                "Inserted {} player records for {} replays",
                all_players.len(),
                ids.len()
            );
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
    let rate_limited = Arc::new(AtomicBool::new(false));
    let mut last_progress_log = std::time::Instant::now();

    loop {
        // Check if we're rate limited and need to wait
        if rate_limited.load(Ordering::SeqCst) {
            warn!(
                "Rate limited, pausing downloads for {} seconds",
                RATE_LIMIT_PAUSE_SECONDS
            );
            sleep(Duration::from_secs(RATE_LIMIT_PAUSE_SECONDS)).await;
            rate_limited.store(false, Ordering::SeqCst);
            info!("Resuming downloads after rate limit pause");
        }

        // Get batch of pending downloads
        let pending = database::list_pending_downloads(None, 100).await?;

        if pending.is_empty() {
            // Check if fetch is still running by waiting a bit and checking again
            sleep(Duration::from_secs(2)).await;

            let still_pending = database::list_pending_downloads(None, 1).await?;

            if still_pending.is_empty() {
                // Double-check all ranks are complete
                let mut all_downloaded = true;
                for rank in Rank::all_ranked() {
                    let not_downloaded = database::count_replays_by_rank_and_status(
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
            let rate_limited = Arc::clone(&rate_limited);

            let handle = tokio::spawn(async move {
                // Check if we're rate limited BEFORE making the API call
                // This prevents spamming the API when a previous task already hit 429
                if rate_limited.load(Ordering::SeqCst) {
                    drop(permit);
                    // Return Ok so we don't mark as failed - it will be retried
                    return Ok(());
                }

                let result = download_single(&client, &replay).await;

                // Check if this was a rate limit error - set flag BEFORE dropping permit
                // to prevent race condition where next task starts before flag is set
                if let Err(e) = &result
                    && e.downcast_ref::<RateLimitedError>().is_some()
                {
                    rate_limited.store(true, Ordering::SeqCst);
                }

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

        // If rate limited, reset in-progress downloads so they can be retried
        if rate_limited.load(Ordering::SeqCst) {
            let reset_count = database::reset_in_progress_downloads().await?;
            if reset_count > 0 {
                info!("Reset {reset_count} in-progress downloads due to rate limiting");
            }
        }
    }

    Ok(())
}

/// Downloads a single replay file.
async fn download_single(client: &BallchasingClient, replay: &Replay) -> Result<()> {
    // Mark as in progress
    database::mark_replay_download_in_progress(replay.id).await?;

    // Attempt download
    match download_and_save(client, replay).await {
        Ok(relative_path) => {
            database::mark_replay_downloaded(replay.id, &relative_path).await?;
            debug!("Downloaded {}", replay.id);
            Ok(())
        }
        Err(error) => {
            let error_msg = format!("{error:#}");
            database::mark_replay_failed(replay.id, &error_msg).await?;
            error!("Failed to download {}: {error}", replay.id);
            Err(error)
        }
    }
}

/// Downloads replay data and saves to file.
///
/// Returns a relative path from the base data directory.
async fn download_and_save(client: &BallchasingClient, replay: &Replay) -> Result<String> {
    let data: Bytes = client.download_replay(replay).await?;

    if data.is_empty() {
        anyhow::bail!("Downloaded replay is empty");
    }

    let object_path = ObjectStorePath::from(replay.file_path.as_str());

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

    for rank in Rank::all_ranked() {
        let downloaded =
            database::count_replays_by_rank_and_status(rank, DownloadStatus::Downloaded).await?;
        let pending =
            database::count_replays_by_rank_and_status(rank, DownloadStatus::NotDownloaded).await?;

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

    for rank in Rank::all_ranked() {
        let downloaded =
            database::count_replays_by_rank_and_status(rank, DownloadStatus::Downloaded).await?;
        let failed =
            database::count_replays_by_rank_and_status(rank, DownloadStatus::Failed).await?;
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
