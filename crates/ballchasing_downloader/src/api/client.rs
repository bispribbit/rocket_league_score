//! Rate-limited HTTP client for ballchasing.com API.

use core::num::NonZeroU32;
use core::str::FromStr;
use core::time::Duration;

use anyhow::{Context, Result};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use config::CONFIG;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use replay_structs::{GameMode, Rank, Replay, ReplaySummary};
use reqwest::Client;
use thiserror::Error;
use tracing::{info, warn};
use uuid::Uuid;

use super::models::ReplayListResponse;

/// Error returned when we hit API rate limits (429).
#[derive(Debug, Clone, Error)]
#[error("Rate limited by ballchasing API (429)")]
pub struct RateLimitedError;

/// Rate limit: 2 requests per second
const RATE_LIMIT_PER_SECOND: u32 = 1;

/// Rate limit: 1000 requests per hour
const RATE_LIMIT_PER_HOUR: u32 = 1000;

/// Base URL for the ballchasing API
const API_BASE_URL: &str = "https://ballchasing.com/api";

type RateLimiterType = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

/// Rate-limited client for ballchasing.com API.
pub struct BallchasingClient {
    client: Client,
    per_second_limiter: RateLimiterType,
    per_hour_limiter: RateLimiterType,
}

impl BallchasingClient {
    /// Creates a new client with rate limiting.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new() -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        // Per-second rate limiter (burst of 2 per second)
        let per_second_quota = Quota::per_second(
            NonZeroU32::new(RATE_LIMIT_PER_SECOND).expect("rate limit should be non-zero"),
        );
        let per_second_limiter = RateLimiter::direct(per_second_quota);

        // Per-hour rate limiter (500 per hour)
        let per_hour_quota = Quota::per_hour(
            NonZeroU32::new(RATE_LIMIT_PER_HOUR).expect("rate limit should be non-zero"),
        );
        let per_hour_limiter = RateLimiter::direct(per_hour_quota);

        Ok(Self {
            client,
            per_second_limiter,
            per_hour_limiter,
        })
    }

    /// Waits for rate limiters before making a request.
    async fn wait_for_rate_limit(&self) {
        // Wait for both rate limiters
        self.per_second_limiter.until_ready().await;
        self.per_hour_limiter.until_ready().await;
    }

    /// Lists replays with the given filters.
    ///
    /// # Arguments
    ///
    /// * `playlist` - Playlist filter (e.g., "ranked-standard")
    /// * `min_rank` - Minimum rank filter (e.g., "bronze-1")
    /// * `max_rank` - Maximum rank filter (same as min for exact rank)
    /// * `count` - Number of results (max 200)
    /// * `before` - Date filter for pagination (only include replays created before this date)
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn list_replays(
        &self,
        playlist: GameMode,
        min_rank: Rank,
        max_rank: Rank,
        count: usize,
        before: Option<DateTime<Utc>>,
    ) -> Result<ReplayListResponse> {
        self.wait_for_rate_limit().await;

        let playlist_str = playlist.as_api_string();
        let min_rank_str = min_rank.as_api_string();
        let max_rank_str = max_rank.as_api_string();

        info!(
            playlist = playlist_str,
            min_rank = min_rank_str,
            max_rank = max_rank_str,
            count = count,
            before = ?before,
            "Listing replays",
        );

        let count = count.min(200); // API max is 200

        let mut url = format!(
            "{API_BASE_URL}/replays?playlist={playlist_str}&min-rank={min_rank_str}&max-rank={max_rank_str}&count={count}&sort-by=upload-date&sort-dir=desc"
        );

        if let Some(before_date) = before {
            // Format DateTime as RFC3339 with Z suffix (API requires this format)
            let rfc3339_date = before_date.format("%Y-%m-%dT%H:%M:%SZ").to_string();
            use core::fmt::Write;
            let _ = write!(url, "&created-before={rfc3339_date}");
        }

        info!("Fetching replays: {url}");

        let response = self
            .client
            .get(&url)
            .header("Authorization", &CONFIG.ballchasing_api_key)
            .send()
            .await
            .context("Failed to send request to ballchasing API")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed with status {status}: {body}");
        }

        let data: ReplayListResponse = response
            .json()
            .await
            .context("Failed to parse replay list response")?;

        info!("Received {} replays", data.list.len());

        Ok(data)
    }

    /// Downloads a replay file by ID.
    ///
    /// # Arguments
    ///
    /// * `replay_id` - The replay ID to download
    ///
    /// # Returns
    ///
    /// The raw bytes of the replay file.
    ///
    /// # Errors
    ///
    /// Returns `RateLimitedError` if rate limited (429), allowing caller to pause.
    /// Returns other errors if the download fails.
    pub async fn download_replay(&self, replay: &Replay) -> Result<Bytes> {
        self.wait_for_rate_limit().await;

        let replay_id = replay.id;

        info!(
            replay_id = %replay_id,
            "Downloading replay"
        );

        let url = format!("{API_BASE_URL}/replays/{replay_id}/file");

        let response = self
            .client
            .get(&url)
            .header("Authorization", &CONFIG.ballchasing_api_key)
            .send()
            .await
            .context("Failed to send download request")?;

        let status = response.status();

        // Return RateLimitedError on 429 so caller can pause all downloads
        if status == 429 {
            warn!(
                replay_id = %replay_id,
                "Rate limited (429), need to pause downloads"
            );
            return Err(RateLimitedError.into());
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Download failed with status {status}: {body}");
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read replay file bytes")?;

        info!(
            replay_id = %replay_id,
            bytes = bytes.len(),
            "Downloaded replay"
        );

        Ok(bytes)
    }

    /// Fetches replays for a specific rank, handling pagination.
    ///
    /// This method will fetch up to `target_count` replays, handling
    /// pagination automatically.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank to fetch (API format, e.g., "bronze-1")
    /// * `target_count` - Target number of replays to fetch
    /// * `existing_ids` - Set of already-known replay IDs to skip
    ///
    /// # Returns
    ///
    /// A vector of replay summaries.
    ///
    /// # Errors
    ///
    /// Returns an error if any API request fails.
    pub async fn fetch_replays_for_rank(
        &self,
        rank: &str,
        target_count: usize,
        existing_ids: &std::collections::HashSet<Uuid>,
    ) -> Result<Vec<ReplaySummary>> {
        let mut all_replays = Vec::new();
        let mut seen_ids = existing_ids.clone();
        let mut before_date: Option<DateTime<Utc>> = None;
        let mut consecutive_duplicates = 0;

        info!(
            "Fetching replays for rank {rank}, target count: {target_count}",
            rank = rank,
            target_count = target_count,
        );

        while all_replays.len() < target_count {
            // Always request at least 50 replays to have a good chance of finding new ones,
            // even if we only need 1 more. We'll stop adding once we have enough.
            let batch_size = 200;

            let ballchasing_rank =
                Rank::from_str(rank).with_context(|| format!("Invalid rank: {rank}"))?;

            let response = self
                .list_replays(
                    GameMode::RankedStandard,
                    ballchasing_rank,
                    ballchasing_rank,
                    batch_size,
                    before_date,
                )
                .await?;

            if response.list.is_empty() {
                info!("No more replays available for rank {rank}");
                break;
            }

            let mut new_count = 0;
            for replay in &response.list {
                // Stop if we have enough replays
                if all_replays.len() >= target_count {
                    break;
                }

                if let Ok(id) = replay.id.parse::<Uuid>() {
                    // Skip if already in database or already seen in this fetch
                    if seen_ids.contains(&id) {
                        continue;
                    }
                    seen_ids.insert(id);
                    new_count += 1;
                    all_replays.push(replay.clone());
                }
            }

            if new_count == 0 {
                consecutive_duplicates += 1;
                if consecutive_duplicates >= 10 {
                    warn!(
                        "Too many consecutive duplicate batches for rank {rank}, going back one day"
                    );
                    consecutive_duplicates = 0;
                    // Subtract one day from before_date to look for older replays
                    before_date = Some(
                        before_date
                            .unwrap_or_else(Utc::now)
                            .checked_sub_signed(chrono::Duration::weeks(1))
                            .unwrap_or_else(Utc::now),
                    );
                    continue;
                }
            } else {
                consecutive_duplicates = 0;
            }

            // Get the created date of the last replay for pagination
            // Parse the created field (when replay was uploaded) for created-before filter
            // With created-before and descending sort, we use the exact date to get older replays
            if let Some(last) = response.list.last() {
                // Parse the created field (when the replay was uploaded)
                let parsed_datetime = chrono::DateTime::parse_from_rfc3339(&last.created)
                    .map_or_else(
                        |_| {
                            if let Ok(dt) = chrono::DateTime::parse_from_str(
                                &last.created,
                                "%Y-%m-%dT%H:%M:%S%z",
                            ) {
                                Some(dt.with_timezone(&Utc))
                            } else {
                                warn!(
                                    replay_id = %last.id,
                                    created = last.created,
                                    "Failed to parse replay created date"
                                );
                                None
                            }
                        },
                        |dt| Some(dt.with_timezone(&Utc)),
                    );

                if let Some(dt) = parsed_datetime {
                    // Use the exact created date - created-before will exclude this date and get older replays
                    before_date = Some(dt);
                } else {
                    // Fallback: use yesterday to ensure we don't get stuck
                    warn!(
                        replay_id = %last.id,
                        "Failed to parse created date, using current date minus 1 week"
                    );
                    before_date = Some(Utc::now() - chrono::Duration::weeks(1));
                }
            }

            // Check if there are more pages
            if response.next.is_none() {
                info!("No more pages for rank {rank}");
                break;
            }
        }

        Ok(all_replays)
    }
}
