//! Rate-limited HTTP client for ballchasing.com API.

use core::num::NonZeroU32;
use core::time::Duration;
use std::sync::Arc;

use anyhow::{Context, Result};
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use reqwest::Client;
use tracing::{debug, warn};
use uuid::Uuid;

use super::models::ReplayListResponse;
use crate::Config;

/// Rate limit: 2 requests per second
const RATE_LIMIT_PER_SECOND: u32 = 2;

/// Rate limit: 500 requests per hour
const RATE_LIMIT_PER_HOUR: u32 = 500;

/// Base URL for the ballchasing API
const API_BASE_URL: &str = "https://ballchasing.com/api";

type RateLimiterType = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

/// Rate-limited client for ballchasing.com API.
pub struct BallchasingClient {
    client: Client,
    api_key: String,
    per_second_limiter: Arc<RateLimiterType>,
    per_hour_limiter: Arc<RateLimiterType>,
}

impl BallchasingClient {
    /// Creates a new client with rate limiting.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP client cannot be created.
    pub fn new(config: &Config) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        // Per-second rate limiter (burst of 2 per second)
        let per_second_quota = Quota::per_second(
            NonZeroU32::new(RATE_LIMIT_PER_SECOND).expect("rate limit should be non-zero"),
        );
        let per_second_limiter = Arc::new(RateLimiter::direct(per_second_quota));

        // Per-hour rate limiter (500 per hour)
        let per_hour_quota = Quota::per_hour(
            NonZeroU32::new(RATE_LIMIT_PER_HOUR).expect("rate limit should be non-zero"),
        );
        let per_hour_limiter = Arc::new(RateLimiter::direct(per_hour_quota));

        Ok(Self {
            client,
            api_key: config.ballchasing_api_key.clone(),
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
        playlist: &str,
        min_rank: &str,
        max_rank: &str,
        count: usize,
        after: Option<&str>,
    ) -> Result<ReplayListResponse> {
        self.wait_for_rate_limit().await;

        let count = count.min(200); // API max is 200

        let mut url = format!(
            "{API_BASE_URL}/replays?playlist={playlist}&min-rank={min_rank}&max-rank={max_rank}&count={count}&sort-by=replay-date&sort-dir=desc"
        );

        if let Some(after_id) = after {
            use core::fmt::Write;
            let _ = write!(url, "&after={after_id}");
        }

        debug!("Fetching replays: {url}");

        let response = self
            .client
            .get(&url)
            .header("Authorization", &self.api_key)
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

        debug!("Received {} replays", data.list.len());

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
    /// Returns an error if the download fails.
    pub async fn download_replay(&self, replay_id: Uuid) -> Result<Vec<u8>> {
        self.wait_for_rate_limit().await;

        let url = format!("{API_BASE_URL}/replays/{replay_id}/file");

        debug!("Downloading replay: {url}");

        let response = self
            .client
            .get(&url)
            .header("Authorization", &self.api_key)
            .send()
            .await
            .context("Failed to send download request")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Download failed with status {status}: {body}");
        }

        let bytes = response
            .bytes()
            .await
            .context("Failed to read replay file bytes")?;

        debug!("Downloaded {} bytes for replay {replay_id}", bytes.len());

        Ok(bytes.to_vec())
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
    ) -> Result<Vec<super::models::ReplaySummary>> {
        let mut all_replays = Vec::new();
        let mut after: Option<String> = None;
        let mut consecutive_duplicates = 0;

        while all_replays.len() < target_count {
            let batch_size = (target_count - all_replays.len()).min(200);

            let response = self
                .list_replays(
                    "ranked-standard",
                    rank,
                    rank,
                    batch_size,
                    after.as_deref(),
                )
                .await?;

            if response.list.is_empty() {
                debug!("No more replays available for rank {rank}");
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
                    warn!(
                        "Too many consecutive duplicate batches for rank {rank}, stopping fetch"
                    );
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
                debug!("No more pages for rank {rank}");
                break;
            }
        }

        Ok(all_replays)
    }
}

