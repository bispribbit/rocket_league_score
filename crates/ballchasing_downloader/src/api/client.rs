//! Rate-limited HTTP client for ballchasing.com API.

use core::num::NonZeroU32;
use core::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;
use config::CONFIG;
use governor::clock::DefaultClock;
use governor::state::{InMemoryState, NotKeyed};
use governor::{Quota, RateLimiter};
use replay_structs::{GameMode, Rank, RankDivision, Replay, ReplaySummary};
use reqwest::Client;
use reqwest::header::RETRY_AFTER;
use thiserror::Error;
use tokio::sync::Mutex;
use tracing::{info, warn};
use url::Url;
use uuid::Uuid;

use super::models::ReplayListResponse;

/// Error returned when we hit API rate limits (429).
#[derive(Debug, Clone, Error)]
#[error("Rate limited by ballchasing API (429)")]
pub struct RateLimitedError {
    /// `Retry-After` response header when present (how long the server asks us to wait).
    pub retry_after: Option<Duration>,
}

fn replay_summary_matches_coarse_rank(replay: &ReplaySummary, rank: Rank) -> bool {
    let Some(min_rank) = &replay.min_rank else {
        return true;
    };
    let division = RankDivision::from(min_rank.clone());
    Rank::from(division) == rank
}

/// Replay list (JSON) requests per second. Separate from file downloads on ballchasing.
const LIST_REPLAYS_REQUESTS_PER_SECOND: u32 = 1;

/// Replay list (JSON) requests per hour.
const LIST_REPLAYS_REQUESTS_PER_HOUR: u32 = 1000;

/// Replay file download requests per hour (typical API tier; adjust if your key differs).
const DOWNLOAD_REPLAY_FILE_REQUESTS_PER_HOUR: u32 = 200;

/// Minimum time between the **start** of consecutive replay file download HTTP requests.
///
/// Ballchasing applies a strict limit on file downloads. In practice, ~1.25 seconds between
/// starts still returned HTTP 429 on the second request; two full seconds is safer. This is
/// enforced in addition to the hourly download quota.
const MINIMUM_DURATION_BETWEEN_DOWNLOAD_REPLAY_FILE_REQUESTS: Duration = Duration::from_secs(2);

/// Base URL for the ballchasing API
const API_BASE_URL: &str = "https://ballchasing.com/api";

type RateLimiterType = RateLimiter<NotKeyed, InMemoryState, DefaultClock>;

/// Upper bound when parsing the `Retry-After` header (seconds).
const MAX_RETRY_AFTER_HEADER_SECONDS: u64 = 3600;

fn parse_retry_after_http_header(response: &reqwest::Response) -> Option<Duration> {
    let header_value = response.headers().get(RETRY_AFTER)?;
    let header_text = header_value.to_str().ok()?;
    let seconds: u64 = header_text.parse().ok()?;
    Some(Duration::from_secs(
        seconds.clamp(1, MAX_RETRY_AFTER_HEADER_SECONDS),
    ))
}

/// Rebuilds the pagination URL so `max-rank` appears exactly once.
///
/// The API `next` link may include an empty `max-rank=` query pair; appending
/// `&max-rank=...` would duplicate the parameter and confuse servers.
fn next_replay_list_url_with_max_rank(next_url: &str, max_rank: &str) -> Result<String> {
    let mut url = Url::parse(next_url).context("Failed to parse ballchasing pagination URL")?;
    let pairs: Vec<(String, String)> = url
        .query_pairs()
        .into_owned()
        .filter(|(name, _)| name != "max-rank")
        .chain(std::iter::once((
            "max-rank".to_string(),
            max_rank.to_string(),
        )))
        .collect();
    url.set_query(None);
    {
        let mut serializer = url.query_pairs_mut();
        for (name, value) in pairs {
            serializer.append_pair(&name, &value);
        }
    }
    Ok(url.to_string())
}

/// Rate-limited client for ballchasing.com API.
pub struct BallchasingClient {
    client: Client,
    list_replays_per_second_limiter: RateLimiterType,
    list_replays_per_hour_limiter: RateLimiterType,
    download_replay_file_per_hour_limiter: RateLimiterType,
    /// Instant when the last replay file download HTTP request was allowed to start (after waits).
    last_download_replay_file_request_start: Mutex<Option<Instant>>,
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

        let list_replays_per_second_quota = Quota::per_second(
            NonZeroU32::new(LIST_REPLAYS_REQUESTS_PER_SECOND)
                .expect("list replays per-second rate limit should be non-zero"),
        );
        let list_replays_per_second_limiter = RateLimiter::direct(list_replays_per_second_quota);

        let list_replays_per_hour_quota = Quota::per_hour(
            NonZeroU32::new(LIST_REPLAYS_REQUESTS_PER_HOUR)
                .expect("list replays per-hour rate limit should be non-zero"),
        );
        let list_replays_per_hour_limiter = RateLimiter::direct(list_replays_per_hour_quota);

        let download_replay_file_per_hour_quota = Quota::per_hour(
            NonZeroU32::new(DOWNLOAD_REPLAY_FILE_REQUESTS_PER_HOUR)
                .expect("download replay file per-hour rate limit should be non-zero"),
        );
        let download_replay_file_per_hour_limiter =
            RateLimiter::direct(download_replay_file_per_hour_quota);

        Ok(Self {
            client,
            list_replays_per_second_limiter,
            list_replays_per_hour_limiter,
            download_replay_file_per_hour_limiter,
            last_download_replay_file_request_start: Mutex::new(None),
        })
    }

    /// Waits for replay list (JSON) endpoint rate limits.
    async fn wait_for_list_replays_rate_limit(&self) {
        self.list_replays_per_second_limiter.until_ready().await;
        self.list_replays_per_hour_limiter.until_ready().await;
    }

    /// Waits for replay file download endpoint rate limits (separate from list endpoints).
    async fn wait_for_download_replay_file_rate_limit(&self) {
        self.download_replay_file_per_hour_limiter
            .until_ready()
            .await;

        let mut last_start_guard = self.last_download_replay_file_request_start.lock().await;
        let now = Instant::now();
        if let Some(previous_start) = *last_start_guard {
            let elapsed = now.saturating_duration_since(previous_start);
            if elapsed < MINIMUM_DURATION_BETWEEN_DOWNLOAD_REPLAY_FILE_REQUESTS {
                let sleep_duration =
                    MINIMUM_DURATION_BETWEEN_DOWNLOAD_REPLAY_FILE_REQUESTS.saturating_sub(elapsed);
                drop(last_start_guard);
                tokio::time::sleep(sleep_duration).await;
                last_start_guard = self.last_download_replay_file_request_start.lock().await;
            }
        }
        *last_start_guard = Some(Instant::now());
    }

    /// Lists replays with the given filters.
    ///
    /// # Arguments
    ///
    /// * `playlist` - Playlist filter (e.g., "ranked-standard")
    /// * `min_rank` - Minimum rank filter (e.g., "bronze-1")
    /// * `count` - Number of results (max 200)
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn list_replays(
        &self,
        playlist: GameMode,
        min_rank: Rank,
        count: usize,
    ) -> Result<ReplayListResponse> {
        let playlist_str = playlist.as_api_string();
        let min_rank_str = min_rank.as_api_string();
        let max_rank = min_rank.saturating_add(4);
        let max_rank_str = max_rank.as_api_string();
        let count = count.min(200); // API max is 200

        let url = format!(
            "{API_BASE_URL}/replays?playlist={playlist_str}&min-rank={min_rank_str}&max-rank={max_rank_str}&count={count}&sort-by=created&sort-dir=desc"
        );

        self.fetch_replay_list(&url).await
    }

    /// Fetches a replay list from a URL (used for initial request and pagination).
    ///
    /// # Arguments
    ///
    /// * `url` - The full URL to fetch
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    async fn fetch_replay_list(&self, url: &str) -> Result<ReplayListResponse> {
        self.wait_for_list_replays_rate_limit().await;

        info!("Fetching replays: {url}");

        let response = self
            .client
            .get(url)
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
    ///
    /// `downloads_remaining` is how many replays are still not successfully downloaded
    /// (including this one) when the HTTP request starts.
    pub async fn download_replay(&self, replay: &Replay) -> Result<Bytes> {
        self.wait_for_download_replay_file_rate_limit().await;

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
            let retry_after = parse_retry_after_http_header(&response);
            warn!(
                replay_id = %replay_id,
                retry_after = ?retry_after,
                "Rate limited (429), need to pause downloads"
            );
            return Err(RateLimitedError { retry_after }.into());
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
    /// pagination automatically using the API's native `next` pagination.
    ///
    /// # Arguments
    ///
    /// * `rank` - The rank to fetch (API format, e.g., "bronze-1")
    /// * `target_count` - Target number of replays to fetch
    /// * `existing_ids` - Replay IDs already present in the database (any rank).
    ///   The API can return the same replay for different rank filters; this set
    ///   must be global so pagination advances past duplicates.
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
        rank: Rank,
        target_count: usize,
        existing_ids: &std::collections::HashSet<Uuid>,
    ) -> Result<Vec<ReplaySummary>> {
        let mut all_replays = Vec::new();
        let mut seen_ids = existing_ids.clone();
        let mut next_url: Option<String> = None;

        info!(
            "Fetching replays for rank {rank}, target count: {target_count}",
            rank = rank,
            target_count = target_count,
        );

        while all_replays.len() < target_count {
            // Fetch either the next page or the initial request
            let response = match &next_url {
                Some(url) => self.fetch_replay_list(url).await?,
                None => {
                    self.list_replays(GameMode::RankedStandard, rank, 200)
                        .await?
                }
            };

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

                if !replay_summary_matches_coarse_rank(replay, rank) {
                    continue;
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

            info!(
                "Rank {rank}: found {new_count} new replays in batch, total: {}",
                all_replays.len()
            );

            // Use API's native pagination
            if let Some(url) = response.next {
                let max_rank = rank.saturating_add(4);
                let max_rank_str = max_rank.as_api_string();
                next_url = Some(
                    next_replay_list_url_with_max_rank(&url, max_rank_str)
                        .with_context(|| format!("Failed to normalize pagination URL: {url}"))?,
                );
            } else {
                info!("No more pages available for rank {rank}");
                break;
            }
        }

        info!(
            "Finished fetching for rank {rank}: {} new replays found",
            all_replays.len()
        );

        Ok(all_replays)
    }
}

#[cfg(test)]
mod tests {
    use super::next_replay_list_url_with_max_rank;

    #[test]
    fn next_replay_list_url_replaces_empty_max_rank_instead_of_duplicating() {
        let api_next = "https://ballchasing.com/api/replays?after=cursor&max-rank=&min-rank=supersonic-legend&playlist=ranked-standard";
        let normalized = next_replay_list_url_with_max_rank(api_next, "supersonic-legend").unwrap();
        assert_eq!(
            normalized.matches("max-rank=").count(),
            1,
            "expected single max-rank: {normalized}"
        );
        assert!(
            normalized.contains("max-rank=supersonic-legend"),
            "{normalized}"
        );
    }
}
