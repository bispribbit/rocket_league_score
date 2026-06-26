//! Targeted fetch of mid-ladder, big-delta (party) lobbies.
//!
//! Ranked solo-queue matches are near rank-homogeneous, so a model trained on
//! them learns to read *lobby* skill rather than the focal player's own skill.
//! Parties let a high-skill player sit in a lower-MMR lobby with a correct label
//! — the exact within-lobby contrast a smurf detector must learn.
//!
//! This module scans the ballchasing **list** endpoint over a wide rank band
//! (per-player ranks are in the summary, no downloads needed), keeps lobbies
//! whose top rank-known player is at least `min_top_gap_mmr` above the lobby
//! median (coarse-rank canonical MMR — identical to `database::list_mixed_rank_replays`),
//! inserts their metadata + players, then downloads the replay files.

use core::time::Duration;
use std::collections::HashSet;

use anyhow::{Context, Result};
use chrono::Utc;
use config::OBJECT_STORE;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_structs::{GameMode, Rank, RankDivision, ReplayPlayer};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::api::client::{BallchasingClient, RateLimitedError, next_replay_list_url_with_max_rank};
use crate::bundle::extract_players_with_average_rank;

/// Page size for the list endpoint (API max is 200).
const PAGE_SIZE: usize = 200;

/// Replays inserted per DB batch during discovery.
const INSERT_BATCH_SIZE: usize = 200;

/// Default pause when rate limited without a usable `Retry-After`.
const DEFAULT_RATE_LIMIT_PAUSE_SECONDS: u64 = 600;

/// Configuration for a mixed-rank fetch run.
pub struct MixedRankFetchConfig {
    /// Lower bound of the API rank band (tier string, e.g. `"gold-1"`).
    pub min_rank: String,
    /// Upper bound of the API rank band (tier string, e.g. `"champion-3"`).
    pub max_rank: String,
    /// Minimum MMR the top rank-known player must sit above the lobby median.
    pub min_top_gap_mmr: f64,
    /// Stop discovery once this many *new* qualifying replays have been inserted.
    pub target_new_replays: usize,
    /// Hard cap on how many list summaries to scan (safety bound on API usage).
    pub max_summaries_to_scan: usize,
    /// Whether to download the replay files after inserting metadata.
    pub do_download: bool,
    /// When true, skip the discovery/scan phase entirely and only download replays
    /// that are already in the DB with `not_downloaded` status. Use this to resume
    /// an interrupted download run.
    pub download_only: bool,
}

/// Coarse-rank canonical MMR (tier midpoint), matching the SQL mapping used by
/// `database::list_mixed_rank_replays`. Returns `None` for unranked.
fn coarse_mmr(rank_division: RankDivision) -> Option<f64> {
    let mmr = match Rank::from(rank_division) {
        Rank::Unranked => return None,
        Rank::Bronze1 => 130.0,
        Rank::Bronze2 => 194.0,
        Rank::Bronze3 => 257.0,
        Rank::Silver1 => 321.0,
        Rank::Silver2 => 386.0,
        Rank::Silver3 => 451.0,
        Rank::Gold1 => 516.0,
        Rank::Gold2 => 580.0,
        Rank::Gold3 => 644.0,
        Rank::Platinum1 => 709.0,
        Rank::Platinum2 => 773.0,
        Rank::Platinum3 => 837.0,
        Rank::Diamond1 => 902.0,
        Rank::Diamond2 => 966.0,
        Rank::Diamond3 => 1030.0,
        Rank::Champion1 => 1127.0,
        Rank::Champion2 => 1258.0,
        Rank::Champion3 => 1388.0,
        Rank::GrandChampion1 => 1520.0,
        Rank::GrandChampion2 => 1651.0,
        Rank::GrandChampion3 => 1782.0,
        Rank::SupersonicLegend => 2200.0,
    };
    Some(mmr)
}

fn median_sorted(values: &[f64]) -> f64 {
    let mid = values.len() / 2;
    let upper = values.get(mid).copied().unwrap_or(0.0);
    if values.len().is_multiple_of(2) {
        let lower = values.get(mid.wrapping_sub(1)).copied().unwrap_or(upper);
        f64::midpoint(lower, upper)
    } else {
        upper
    }
}

/// Within-lobby top gap (`max - median`) over rank-known players, using coarse
/// canonical MMR. Returns `None` if fewer than three players have a known rank.
fn top_gap_mmr(players: &[ReplayPlayer]) -> Option<f64> {
    let mut mmrs: Vec<f64> = players
        .iter()
        .filter(|player| player.rank_known)
        .filter_map(|player| coarse_mmr(player.rank_division))
        .collect();
    if mmrs.len() < 3 {
        return None;
    }
    mmrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = median_sorted(&mmrs);
    let max = mmrs.last().copied().unwrap_or(median);
    Some(max - median)
}

/// Runs discovery (+ optional download) for mixed-rank replays.
///
/// When `config.download_only` is true, discovery is skipped and the function goes
/// straight to downloading all `not_downloaded` replays that already satisfy the
/// `min_top_gap_mmr` criterion. Use this to resume an interrupted download run
/// without re-scanning the API.
///
/// # Errors
///
/// Returns an error if the API, database, or object store operations fail.
pub async fn run(config: &MixedRankFetchConfig) -> Result<()> {
    let client = BallchasingClient::new()?;

    if config.download_only {
        info!("Skipping discovery phase (--download-only)");
    } else {
        let inserted = discover_and_insert(&client, config).await?;
        info!(
            inserted_new = inserted,
            target = config.target_new_replays,
            "Discovery phase complete"
        );
    }

    if config.do_download {
        download_pending(&client, config.min_top_gap_mmr).await?;
    } else {
        info!("Skipping download phase (do_download = false)");
    }

    Ok(())
}

/// A buffer of pending DB inserts (parallel arrays for `insert_replays`).
#[derive(Default)]
struct InsertBuffer {
    ids: Vec<Uuid>,
    game_modes: Vec<GameMode>,
    ranks: Vec<Rank>,
    metadata: Vec<serde_json::Value>,
    players: Vec<ReplayPlayer>,
}

impl InsertBuffer {
    const fn len(&self) -> usize {
        self.ids.len()
    }

    async fn flush(&mut self) -> Result<usize> {
        if self.ids.is_empty() {
            return Ok(0);
        }
        let created =
            database::insert_replays(&self.ids, &self.game_modes, &self.ranks, &self.metadata)
                .await
                .context("Failed to insert mixed-rank replays")?;
        database::insert_replay_players(&self.players)
            .await
            .context("Failed to insert mixed-rank replay players")?;
        let inserted = self.ids.len();
        self.ids.clear();
        self.game_modes.clear();
        self.ranks.clear();
        self.metadata.clear();
        self.players.clear();
        Ok(created.max(inserted))
    }
}

/// Scans the band and inserts new qualifying replays until the target is reached.
async fn discover_and_insert(
    client: &BallchasingClient,
    config: &MixedRankFetchConfig,
) -> Result<usize> {
    let existing_ids: HashSet<Uuid> = database::list_all_replay_ids().await?;
    let mut seen_this_run: HashSet<Uuid> = HashSet::new();
    let mut buffer = InsertBuffer::default();

    let mut scanned = 0usize;
    let mut collected = 0usize;
    let mut qualifying_already_present = 0usize;

    info!(
        band_min = %config.min_rank,
        band_max = %config.max_rank,
        min_gap = config.min_top_gap_mmr,
        target = config.target_new_replays,
        "Starting mixed-rank discovery"
    );

    let first = client
        .list_replays_in_band(
            GameMode::RankedStandard,
            &config.min_rank,
            &config.max_rank,
            PAGE_SIZE,
        )
        .await
        .context("Initial list request failed")?;
    let mut next_url = first.next.clone();
    let mut page = first;

    loop {
        for summary in &page.list {
            scanned += 1;

            let Ok(id) = summary.id.parse::<Uuid>() else {
                continue;
            };
            if !seen_this_run.insert(id) {
                continue;
            }

            let Ok(calculation) = extract_players_with_average_rank(id, summary) else {
                continue;
            };
            let Some(gap) = top_gap_mmr(&calculation.players) else {
                continue;
            };
            if gap < config.min_top_gap_mmr {
                continue;
            }

            // Qualifying — but skip if we already have it.
            if existing_ids.contains(&id) {
                qualifying_already_present += 1;
                continue;
            }

            buffer.ids.push(id);
            buffer.game_modes.push(GameMode::RankedStandard);
            buffer.ranks.push(calculation.folder_rank);
            buffer.metadata.push(serde_json::to_value(summary)?);
            for player in calculation.players {
                buffer.players.push(ReplayPlayer {
                    id: 0,
                    replay_id: id,
                    player_name: player.player_name,
                    team: player.team,
                    rank_division: player.rank_division,
                    rank_known: player.rank_known,
                    created_at: Utc::now(),
                });
            }
            collected += 1;

            if buffer.len() >= INSERT_BATCH_SIZE {
                buffer.flush().await?;
                info!(
                    collected,
                    scanned, qualifying_already_present, "Discovery progress"
                );
            }

            if collected >= config.target_new_replays {
                break;
            }
        }

        if collected >= config.target_new_replays {
            info!(collected, "Reached target");
            break;
        }
        if scanned >= config.max_summaries_to_scan {
            warn!(
                scanned,
                collected, "Hit scan cap before reaching target — widen band or raise cap"
            );
            break;
        }

        let Some(raw_next) = next_url.as_deref() else {
            warn!(
                scanned,
                collected, "Reached end of results before target — widen band / date range"
            );
            break;
        };
        let page_url = next_replay_list_url_with_max_rank(raw_next, &config.max_rank)?;
        page = match client.fetch_replay_list_page(&page_url).await {
            Ok(page) => page,
            Err(error) => {
                warn!(%error, "Page fetch failed; stopping discovery early");
                break;
            }
        };
        next_url.clone_from(&page.next);
        if page.list.is_empty() {
            break;
        }
    }

    buffer.flush().await?;
    info!(
        collected,
        scanned, qualifying_already_present, "Discovery finished"
    );
    Ok(collected)
}

/// Downloads all not-yet-downloaded mixed-rank replays at or above the gap.
async fn download_pending(client: &BallchasingClient, min_top_gap_mmr: f64) -> Result<()> {
    let reset = database::reset_in_progress_downloads().await?;
    if reset > 0 {
        info!(reset, "Reset stuck in-progress downloads");
    }

    let mut downloaded = 0usize;
    let mut failed = 0usize;

    loop {
        let pending = database::list_pending_mixed_rank_replays(min_top_gap_mmr, 100).await?;
        if pending.is_empty() {
            break;
        }
        info!(
            batch = pending.len(),
            downloaded, failed, "Downloading batch"
        );

        for replay in &pending {
            database::mark_replay_download_in_progress(replay.id).await?;
            match client.download_replay(replay).await {
                Ok(bytes) => {
                    if bytes.is_empty() {
                        database::mark_replay_failed(replay.id, "Downloaded replay is empty")
                            .await?;
                        failed += 1;
                        continue;
                    }
                    let object_path = ObjectStorePath::from(replay.file_path.as_str());
                    OBJECT_STORE
                        .put(&object_path, bytes.into())
                        .await
                        .context("Failed to write replay to object store")?;
                    database::mark_replay_downloaded(replay.id, replay.file_path.as_str()).await?;
                    downloaded += 1;
                    if downloaded.is_multiple_of(25) {
                        info!(downloaded, failed, "Download progress");
                    }
                }
                Err(error) => {
                    if let Some(rate_limited) = error.downcast_ref::<RateLimitedError>() {
                        let pause = rate_limited.retry_after.map_or(
                            DEFAULT_RATE_LIMIT_PAUSE_SECONDS,
                            |duration| {
                                duration
                                    .as_secs()
                                    .clamp(1, DEFAULT_RATE_LIMIT_PAUSE_SECONDS)
                            },
                        );
                        warn!(pause_seconds = pause, "Rate limited; pausing");
                        database::reset_in_progress_downloads().await?;
                        tokio::time::sleep(Duration::from_secs(pause)).await;
                        break;
                    }
                    database::mark_replay_failed(replay.id, &format!("{error:#}")).await?;
                    error!(replay_id = %replay.id, %error, "Download failed");
                    failed += 1;
                }
            }
        }
    }

    info!(downloaded, failed, "Mixed-rank download phase complete");
    Ok(())
}
