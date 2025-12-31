use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::rank::RankDivision;
use crate::{GameMode, Rank};

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

/// Dataset split assignment for machine learning.
/// Used to ensure training and evaluation sets remain separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, sqlx::Type)]
#[sqlx(type_name = "dataset_split", rename_all = "snake_case")]
pub enum DatasetSplit {
    /// Replay is used for training the model.
    Training,
    /// Replay is held out for evaluation/testing.
    Evaluation,
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
    pub game_mode: GameMode,
    pub rank: Rank,
    pub metadata: serde_json::Value,
    pub download_status: DownloadStatus,
    pub file_path: String,
    pub error_message: Option<String>,
    /// Dataset split assignment (None = unassigned).
    pub dataset_split: Option<DatasetSplit>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
