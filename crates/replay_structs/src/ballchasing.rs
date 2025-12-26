//! Database model types.

use core::str::FromStr;

use sqlx::types::chrono::{DateTime, Utc};
use uuid::Uuid;

/// Game mode enum matching the `PostgreSQL` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "game_mode", rename_all = "snake_case")]
pub enum GameMode {
    Soccar3v3,
    Soccar2v2,
    Soccar1v1,
    Hoops,
    Rumble,
    Dropshot,
    Snowday,
}

impl FromStr for GameMode {
    type Err = anyhow::Error;

    /// Returns the game mode from a string representation.
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "3v3" | "soccar_3v3" => Ok(Self::Soccar3v3),
            "2v2" | "soccar_2v2" => Ok(Self::Soccar2v2),
            "1v1" | "soccar_1v1" => Ok(Self::Soccar1v1),
            "hoops" => Ok(Self::Hoops),
            "rumble" => Ok(Self::Rumble),
            "dropshot" => Ok(Self::Dropshot),
            "snowday" => Ok(Self::Snowday),
            _ => Err(anyhow::anyhow!("Invalid game mode: {s}")),
        }
    }
}

/// ML model metadata stored in the database.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Model {
    pub id: Uuid,
    pub name: String,
    pub version: i32,
    pub checkpoint_path: String,
    pub training_config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
    pub trained_at: DateTime<Utc>,
}

// ============================================================================
// Ballchasing Downloader Models
// ============================================================================

/// Rank enum matching ballchasing API filter options.
///
/// These correspond to the min-rank/max-rank filter values in the ballchasing API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, sqlx::Type)]
#[sqlx(type_name = "ballchasing_rank", rename_all = "snake_case")]
pub enum BallchasingRank {
    Unranked,
    Bronze1,
    Bronze2,
    Bronze3,
    Silver1,
    Silver2,
    Silver3,
    Gold1,
    Gold2,
    Gold3,
    Platinum1,
    Platinum2,
    Platinum3,
    Diamond1,
    Diamond2,
    Diamond3,
    Champion1,
    Champion2,
    Champion3,
    GrandChampion,
}

impl BallchasingRank {
    /// Returns the API string representation for this rank.
    #[must_use]
    pub const fn as_api_string(self) -> &'static str {
        match self {
            Self::Unranked => "unranked",
            Self::Bronze1 => "bronze-1",
            Self::Bronze2 => "bronze-2",
            Self::Bronze3 => "bronze-3",
            Self::Silver1 => "silver-1",
            Self::Silver2 => "silver-2",
            Self::Silver3 => "silver-3",
            Self::Gold1 => "gold-1",
            Self::Gold2 => "gold-2",
            Self::Gold3 => "gold-3",
            Self::Platinum1 => "platinum-1",
            Self::Platinum2 => "platinum-2",
            Self::Platinum3 => "platinum-3",
            Self::Diamond1 => "diamond-1",
            Self::Diamond2 => "diamond-2",
            Self::Diamond3 => "diamond-3",
            Self::Champion1 => "champion-1",
            Self::Champion2 => "champion-2",
            Self::Champion3 => "champion-3",
            Self::GrandChampion => "grand-champion",
        }
    }

    /// Returns the folder name for storing replays of this rank.
    #[must_use]
    pub const fn as_folder_name(self) -> &'static str {
        match self {
            Self::Unranked => "unranked",
            Self::Bronze1 => "bronze-1",
            Self::Bronze2 => "bronze-2",
            Self::Bronze3 => "bronze-3",
            Self::Silver1 => "silver-1",
            Self::Silver2 => "silver-2",
            Self::Silver3 => "silver-3",
            Self::Gold1 => "gold-1",
            Self::Gold2 => "gold-2",
            Self::Gold3 => "gold-3",
            Self::Platinum1 => "platinum-1",
            Self::Platinum2 => "platinum-2",
            Self::Platinum3 => "platinum-3",
            Self::Diamond1 => "diamond-1",
            Self::Diamond2 => "diamond-2",
            Self::Diamond3 => "diamond-3",
            Self::Champion1 => "champion-1",
            Self::Champion2 => "champion-2",
            Self::Champion3 => "champion-3",
            Self::GrandChampion => "grand-champion",
        }
    }

    /// Returns an iterator over all ranked tiers (excluding unranked).
    pub fn all_ranked() -> impl Iterator<Item = Self> {
        [
            Self::Bronze1,
            Self::Bronze2,
            Self::Bronze3,
            Self::Silver1,
            Self::Silver2,
            Self::Silver3,
            Self::Gold1,
            Self::Gold2,
            Self::Gold3,
            Self::Platinum1,
            Self::Platinum2,
            Self::Platinum3,
            Self::Diamond1,
            Self::Diamond2,
            Self::Diamond3,
            Self::Champion1,
            Self::Champion2,
            Self::Champion3,
            Self::GrandChampion,
        ]
        .into_iter()
    }

    /// Returns an iterator over all ranks including unranked.
    pub fn all() -> impl Iterator<Item = Self> {
        core::iter::once(Self::Unranked).chain(Self::all_ranked())
    }
}

impl core::fmt::Display for BallchasingRank {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_api_string())
    }
}

impl FromStr for BallchasingRank {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "unranked" => Ok(Self::Unranked),
            "bronze-1" => Ok(Self::Bronze1),
            "bronze-2" => Ok(Self::Bronze2),
            "bronze-3" => Ok(Self::Bronze3),
            "silver-1" => Ok(Self::Silver1),
            "silver-2" => Ok(Self::Silver2),
            "silver-3" => Ok(Self::Silver3),
            "gold-1" => Ok(Self::Gold1),
            "gold-2" => Ok(Self::Gold2),
            "gold-3" => Ok(Self::Gold3),
            "platinum-1" => Ok(Self::Platinum1),
            "platinum-2" => Ok(Self::Platinum2),
            "platinum-3" => Ok(Self::Platinum3),
            "diamond-1" => Ok(Self::Diamond1),
            "diamond-2" => Ok(Self::Diamond2),
            "diamond-3" => Ok(Self::Diamond3),
            "champion-1" => Ok(Self::Champion1),
            "champion-2" => Ok(Self::Champion2),
            "champion-3" => Ok(Self::Champion3),
            "grand-champion" => Ok(Self::GrandChampion),
            _ => Err(anyhow::anyhow!("Invalid ballchasing rank: {s}")),
        }
    }
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

/// Ballchasing replay metadata and download tracking.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct BallchasingReplay {
    pub id: Uuid,
    pub rank: BallchasingRank,
    pub metadata: serde_json::Value,
    pub download_status: DownloadStatus,
    pub file_path: Option<String>,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Statistics for a ballchasing rank showing download status counts.
#[derive(Debug, Clone)]
pub struct BallchasingRankStats {
    pub rank: BallchasingRank,
    pub not_downloaded: i64,
    pub in_progress: i64,
    pub downloaded: i64,
    pub failed: i64,
}
