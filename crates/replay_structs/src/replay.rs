use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::Rank;
use crate::rank::RankDivision;

/// Player data for a specific replay.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ReplayPlayer {
    pub id: i32,
    pub replay_id: Uuid,
    pub player_name: String,
    pub team: i16,
    pub rank_division: RankDivision,
    pub created_at: DateTime<Utc>,
}

/// Download status for tracking replay file downloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, sqlx::Type)]
#[sqlx(type_name = "download_status", rename_all = "snake_case")]
pub enum DownloadStatus {
    NotDownloaded,
    InProgress,
    Downloaded,
    Failed,
}

impl core::fmt::Display for DownloadStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotDownloaded => write!(f, "not_downloaded"),
            Self::InProgress => write!(f, "in_progress"),
            Self::Downloaded => write!(f, "downloaded"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Replay metadata and download tracking.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Replay {
    pub id: Uuid,
    pub rank: Rank,
    pub metadata: serde_json::Value,
    pub download_status: DownloadStatus,
    pub file_path: String,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
