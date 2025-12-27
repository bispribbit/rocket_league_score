//! Rate-limited HTTP client for ballchasing.com API.

use core::num::NonZeroU32;
use core::str::FromStr;
use core::time::Duration;

use anyhow::{Context, Result};
use backon::{ExponentialBuilder, Retryable};
use bytes::Bytes;
use config::CONFIG;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use replay_structs::{GameMode, Rank, Replay, ReplaySummary};
use reqwest::Client;
use tracing::{info, warn};
use uuid::Uuid;

use super::models::ReplayListResponse;

/// Rate limit: 2 requests per second
const RATE_LIMIT_PER_SECOND: u32 = 1;

/// Rate limit: 500 requests per hour
const RATE_LIMIT_PER_HOUR: u32 = 500;

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
    /// * `after` - Cursor for pagination (replay ID to start after)
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
        after: Option<&str>,
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
            after = after,
            "Listing replays",
        );

        let count = count.min(200); // API max is 200

        let mut url = format!(
            "{API_BASE_URL}/replays?playlist={playlist_str}&min-rank={min_rank_str}&max-rank={max_rank_str}&count={count}&sort-by=replay-date&sort-dir=desc"
        );

        if let Some(after_id) = after {
            use core::fmt::Write;
            let _ = write!(url, "&after={after_id}");
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

    /// Downloads a replay file by ID with retry logic for rate limiting.
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
    /// Returns an error if the download fails after retries.
    pub async fn download_replay(&self, replay: &Replay) -> Result<Bytes> {
        let client = &self.client;
        let replay_id_for_closure = replay.id;

        (|| async {
            self.wait_for_rate_limit().await;

            info!(
                replay_id = %replay_id_for_closure,
                "Downloading replay"
            );

            let url = format!("{API_BASE_URL}/replays/{replay_id_for_closure}/file");

            let response = client
                .get(&url)
                .header("Authorization", &CONFIG.ballchasing_api_key)
                .send()
                .await
                .context("Failed to send download request")?;

            let status = response.status();

            // Only retry on 429 Too Many Requests
            if status == 429 {
                let body = response.text().await.unwrap_or_default();
                warn!(
                    replay_id = %replay_id_for_closure,
                    "Rate limited (429), will retry"
                );
                anyhow::bail!("Rate limited (429): {body}");
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
                replay_id = %replay_id_for_closure,
                bytes = bytes.len(),
                "Downloaded replay"
            );

            Ok(bytes)
        })
        .retry(
            &ExponentialBuilder::default()
                .with_max_times(3)
                .with_min_delay(Duration::from_secs(1))
                .with_max_delay(Duration::from_secs(8)),
        )
        .await
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
        let mut after: Option<String> = None;
        let mut consecutive_duplicates = 0;

        info!(
            "Fetching replays for rank {rank}, target count: {target_count}",
            rank = rank,
            target_count = target_count,
        );

        while all_replays.len() < target_count {
            let batch_size = (target_count - all_replays.len()).min(200);

            let ballchasing_rank =
                Rank::from_str(rank).with_context(|| format!("Invalid rank: {rank}"))?;

            let response = self
                .list_replays(
                    GameMode::RankedStandard,
                    ballchasing_rank,
                    ballchasing_rank,
                    batch_size,
                    after.as_deref(),
                )
                .await?;

            if response.list.is_empty() {
                info!("No more replays available for rank {rank}");
                break;
            }

            let mut new_count = 0;
            for replay in &response.list {
                if let Ok(id) = replay.id.parse::<Uuid>()
                    && !existing_ids.contains(&id)
                {
                    new_count += 1;
                }
            }

            if new_count == 0 {
                consecutive_duplicates += 1;
                if consecutive_duplicates >= 3 {
                    warn!("Too many consecutive duplicate batches for rank {rank}, stopping fetch");
                    break;
                }
            } else {
                consecutive_duplicates = 0;
            }

            // Get the last ID for pagination
            if let Some(last) = response.list.last() {
                after = Some(last.id.clone());
            }

            all_replays.extend(response.list);

            // Check if there are more pages
            if response.next.is_none() {
                info!("No more pages for rank {rank}");
                break;
            }
        }

        Ok(all_replays)
    }
}
