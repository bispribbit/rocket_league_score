//! Core downloader logic that runs fetch and download in parallel.

use core::fmt::Write as _;
use core::str::FromStr;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
use crate::bundle::extract_players_with_average_rank;
use crate::players::extract_players_from_metadata;

/// How many replays to collect per rank of one playlist.
#[derive(Debug, Clone, Copy)]
struct PlaylistTarget {
    game_mode: GameMode,
    replays_per_rank: usize,
}

/// Replays per rank, per ranked playlist. Standard was sized for the old LSTM. Duels and
/// doubles are sized from the tree model's learning curve (`learning_curve` binary): on
/// standard, accuracy flattens past ~8,000 training replays (each doubling after that buys
/// 2–3 MMR), and 350 per rank over 22 ranks is ~7,700 replays per playlist.
const PLAYLIST_TARGETS: [PlaylistTarget; 3] = [
    PlaylistTarget {
        game_mode: GameMode::RankedStandard,
        replays_per_rank: 1200,
    },
    PlaylistTarget {
        game_mode: GameMode::RankedDoubles,
        replays_per_rank: TARGET_DOUBLES_REPLAYS_PER_RANK,
    },
    PlaylistTarget {
        game_mode: GameMode::RankedDuels,
        replays_per_rank: TARGET_DUELS_REPLAYS_PER_RANK,
    },
];

/// Doubles replays per rank.
const TARGET_DOUBLES_REPLAYS_PER_RANK: usize = 350;

/// Duels replays per rank.
const TARGET_DUELS_REPLAYS_PER_RANK: usize = 350;

/// Pause after a failed replay list request before trying the next bucket.
const FETCH_RETRY_SECONDS: u64 = 60;

/// One (playlist, rank) bucket the fetcher fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FetchBucket {
    game_mode: GameMode,
    rank: Rank,
}

/// How often to log progress (in seconds).
const PROGRESS_LOG_INTERVAL_SECONDS: u64 = 30;

/// Maximum concurrent downloads.
const MAX_CONCURRENT_DOWNLOADS: usize = 1;

/// Default pause when the server returns HTTP 429 without a `Retry-After` header.
const DEFAULT_PAUSE_ON_RATE_LIMIT_SECONDS: u64 = 600;

/// Maximum pause we apply when honoring `Retry-After` (avoid multi-hour sleeps from bad headers).
const MAX_PAUSE_ON_RATE_LIMIT_SECONDS: u64 = 600;

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

    let fetch_client = client.clone();
    let download_client = client.clone();

    let fetch_finished = Arc::new(AtomicBool::new(false));
    let fetch_finished_by_fetcher = Arc::clone(&fetch_finished);
    let fetch_task = tokio::spawn(async move {
        let result = fetch_all_metadata(fetch_client).await;
        if let Err(error) = &result {
            error!("Metadata fetch stopped: {error:#}");
        }
        fetch_finished_by_fetcher.store(true, Ordering::SeqCst);
        result
    });

    let download_task =
        tokio::spawn(async move { download_all_replays(download_client, fetch_finished).await });

    // Wait for both tasks
    let (fetch_result, download_result) = tokio::join!(fetch_task, download_task);

    fetch_result??;
    download_result??;

    info!("Download process complete");
    print_stats().await?;

    Ok(())
}

/// Fetches metadata for every playlist and rank until each has its target.
///
/// A bucket whose search comes back empty is not retried: some playlist/rank pairs (for
/// example Supersonic Legend duels) never reach the target.
async fn fetch_all_metadata(client: Arc<BallchasingClient>) -> Result<()> {
    info!("Starting metadata fetch (targets: {PLAYLIST_TARGETS:?})");
    let mut exhausted: HashSet<FetchBucket> = HashSet::new();

    loop {
        let mut all_complete = true;

        for target in PLAYLIST_TARGETS {
            for rank in Rank::all_ranked() {
                let bucket = FetchBucket {
                    game_mode: target.game_mode,
                    rank,
                };
                if exhausted.contains(&bucket) {
                    continue;
                }
                let current = database::count_replays_by_rank(target.game_mode, rank).await?;
                let Some(needed) = usize::try_from(current)
                    .ok()
                    .and_then(|current| target.replays_per_rank.checked_sub(current))
                    .filter(|needed| *needed > 0)
                else {
                    continue;
                };
                all_complete = false;
                // A failed list request (timeout, 5xx, 429) must not end the whole fetch:
                // the downloads would carry on and the missing buckets would go unnoticed.
                let created = match fetch_bucket(&client, bucket, needed).await {
                    Ok(created) => created,
                    Err(error) => {
                        warn!(
                            "{} {rank}: fetch failed, retrying in {FETCH_RETRY_SECONDS} s: {error:#}",
                            target.game_mode.as_api_string()
                        );
                        sleep(Duration::from_secs(FETCH_RETRY_SECONDS)).await;
                        continue;
                    }
                };
                if created == 0 {
                    info!(
                        "{} {rank}: no new replays available, stopping at {current}",
                        target.game_mode.as_api_string()
                    );
                    exhausted.insert(bucket);
                }
            }
        }

        if all_complete || exhausted.len() >= PLAYLIST_TARGETS.len() * Rank::all_ranked().count() {
            info!("Metadata fetch finished");
            break;
        }

        // Small delay before checking again
        sleep(Duration::from_secs(1)).await;
    }

    Ok(())
}

/// Fetches and stores up to `needed` new replays of one bucket. Returns how many were stored.
async fn fetch_bucket(
    client: &BallchasingClient,
    bucket: FetchBucket,
    needed: usize,
) -> Result<usize> {
    let FetchBucket { game_mode, rank } = bucket;
    info!(
        "{} {rank}: fetching up to {needed} more",
        game_mode.as_api_string()
    );

    // Skip any replay ID already stored under any rank (not only this rank).
    // Otherwise the API can return replays we already have from another rank
    // filter; inserts would no-op and this rank would never reach its target.
    let all_replay_ids: HashSet<Uuid> = database::list_all_replay_ids().await?;

    let replays: Vec<ReplaySummary> = client
        .fetch_replays_for_rank(game_mode, rank, needed, &all_replay_ids)
        .await?;

    let mut ids = Vec::new();
    let mut game_modes = Vec::new();
    let mut ranks = Vec::new();
    let mut metadata_values = Vec::new();

    for replay in replays {
        let Ok(id) = replay.id.parse::<Uuid>() else {
            warn!("Invalid replay ID: {}", replay.id);
            continue;
        };

        if all_replay_ids.contains(&id) {
            continue;
        }

        // The playlist the API reports, else the one we searched.
        let replay_game_mode = replay
            .playlist_id
            .as_ref()
            .map_or(game_mode, |playlist_id| {
                GameMode::from_str(playlist_id).unwrap_or_else(|_| {
                    warn!(
                        "Invalid playlist_id: {playlist_id}, defaulting to {}",
                        game_mode.as_api_string()
                    );
                    game_mode
                })
            });

        let storage_rank = match extract_players_with_average_rank(id, &replay) {
            Ok(calculation) => calculation.folder_rank,
            Err(_) => rank,
        };

        let metadata = serde_json::to_value(&replay)?;
        ids.push(id);
        game_modes.push(replay_game_mode);
        ranks.push(storage_rank);
        metadata_values.push(metadata);
    }

    if ids.is_empty() {
        return Ok(0);
    }

    let created = database::insert_replays(&ids, &game_modes, &ranks, &metadata_values).await?;
    info!(
        "Stored {created} new {} replays for rank {rank}",
        game_mode.as_api_string()
    );

    // Extract and insert players for each new replay
    let mut all_players = Vec::new();

    for (replay_id, metadata) in ids.iter().zip(metadata_values.iter()) {
        match extract_players_from_metadata(metadata) {
            Ok(players) => {
                for player in players {
                    all_players.push(ReplayPlayer {
                        id: 0,
                        replay_id: *replay_id,
                        player_name: player.player_name,
                        team: player.team,
                        rank_division: player.rank_division,
                        rank_known: player.rank_known,
                        created_at: Utc::now(),
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

    if !all_players.is_empty() {
        database::insert_replay_players(&all_players).await?;
        info!(
            "Inserted {} player records for {} replays",
            all_players.len(),
            ids.len()
        );
    }

    Ok(created)
}

/// Downloads pending replays until the metadata fetch has finished and nothing is left.
async fn download_all_replays(
    client: Arc<BallchasingClient>,
    fetch_finished: Arc<AtomicBool>,
) -> Result<()> {
    info!("Starting replay downloads");

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_DOWNLOADS));
    let rate_limited = Arc::new(AtomicBool::new(false));
    let pause_on_rate_limit_seconds = Arc::new(AtomicU64::new(DEFAULT_PAUSE_ON_RATE_LIMIT_SECONDS));
    let mut last_progress_log = std::time::Instant::now();

    loop {
        // Check if we're rate limited and need to wait
        if rate_limited.load(Ordering::SeqCst) {
            let pause_seconds = pause_on_rate_limit_seconds.load(Ordering::SeqCst);
            warn!(
                "Rate limited, pausing downloads for {} seconds",
                pause_seconds
            );
            sleep(Duration::from_secs(pause_seconds)).await;
            rate_limited.store(false, Ordering::SeqCst);
            pause_on_rate_limit_seconds
                .store(DEFAULT_PAUSE_ON_RATE_LIMIT_SECONDS, Ordering::SeqCst);
            info!("Resuming downloads after rate limit pause");
        }

        // Get batch of pending downloads
        let pending = database::list_pending_downloads(None, 100).await?;

        if pending.is_empty() {
            // Check if fetch is still running by waiting a bit and checking again
            sleep(Duration::from_secs(2)).await;

            let still_pending = database::list_pending_downloads(None, 1).await?;

            if still_pending.is_empty() && fetch_finished.load(Ordering::SeqCst) {
                info!("All downloads complete");
                break;
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
            let pause_on_rate_limit_seconds = Arc::clone(&pause_on_rate_limit_seconds);

            let handle = tokio::spawn(async move {
                // Check if we're rate limited BEFORE making the API call
                // This prevents spamming the API when a previous task already hit 429
                if rate_limited.load(Ordering::SeqCst) {
                    drop(permit);
                    // Return Ok so we don't mark as failed - it will be retried
                    return Ok(());
                }

                let downloads_remaining =
                    database::count_replays_pending_download_completion().await?;
                info!("Downloads remaining: {downloads_remaining}");

                let result = download_single(&client, &replay).await;

                // Check if this was a rate limit error - set flag BEFORE dropping permit
                // to prevent race condition where next task starts before flag is set
                if let Err(error) = &result
                    && let Some(rate_limited_error) = error.downcast_ref::<RateLimitedError>()
                {
                    let pause_seconds = rate_limited_error
                        .retry_after
                        .map_or(DEFAULT_PAUSE_ON_RATE_LIMIT_SECONDS, |duration| {
                            duration.as_secs().clamp(1, MAX_PAUSE_ON_RATE_LIMIT_SECONDS)
                        });
                    pause_on_rate_limit_seconds.store(pause_seconds, Ordering::SeqCst);
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
            if error.downcast_ref::<RateLimitedError>().is_some() {
                // Leave status as `in_progress`; `reset_in_progress_downloads` clears it.
                return Err(error);
            }
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

    for target in PLAYLIST_TARGETS {
        for rank in Rank::all_ranked() {
            let downloaded = database::count_replays_by_rank_and_status(
                target.game_mode,
                rank,
                DownloadStatus::Downloaded,
            )
            .await?;
            let pending = database::count_replays_by_rank_and_status(
                target.game_mode,
                rank,
                DownloadStatus::NotDownloaded,
            )
            .await?;

            total_downloaded += downloaded;
            total_pending += pending;

            if pending > 0 {
                let _ = write!(
                    status,
                    " {}/{rank}:{downloaded}/{}",
                    target.game_mode.as_api_string(),
                    downloaded + pending
                );
            }
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
        "Playlist and rank", "Downloaded", "Failed", "Total"
    );
    info!("{}", "-".repeat(58));

    let mut grand_downloaded = 0i64;
    let mut grand_failed = 0i64;

    for target in PLAYLIST_TARGETS {
        for rank in Rank::all_ranked() {
            let downloaded = database::count_replays_by_rank_and_status(
                target.game_mode,
                rank,
                DownloadStatus::Downloaded,
            )
            .await?;
            let failed = database::count_replays_by_rank_and_status(
                target.game_mode,
                rank,
                DownloadStatus::Failed,
            )
            .await?;
            let total = downloaded + failed;

            info!(
                "{:<20} {:>12} {:>12} {:>12}",
                format!(
                    "{} {}",
                    target.game_mode.as_api_string(),
                    rank.as_api_string()
                ),
                downloaded,
                failed,
                total
            );

            grand_downloaded += downloaded;
            grand_failed += failed;
        }
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
