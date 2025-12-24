//! API response types for ballchasing.com.

use replay_structs::ReplaySummary;
use serde::{Deserialize, Serialize};

/// Response from GET /api/replays endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayListResponse {
    /// List of replay summaries
    pub list: Vec<ReplaySummary>,

    /// Pagination count
    pub count: i32,

    /// URL for next page of results (if available)
    pub next: Option<String>,
}

/// Uploader information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Uploader {
    /// Steam ID
    pub steam_id: Option<String>,

    /// Display name
    pub name: Option<String>,

    /// Profile URL
    pub profile_url: Option<String>,

    /// Avatar URL
    pub avatar: Option<String>,
}
