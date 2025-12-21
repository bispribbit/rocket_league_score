//! Database model types.

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

impl GameMode {
    /// Returns the game mode from a string representation.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "3v3" | "soccar_3v3" => Some(Self::Soccar3v3),
            "2v2" | "soccar_2v2" => Some(Self::Soccar2v2),
            "1v1" | "soccar_1v1" => Some(Self::Soccar1v1),
            "hoops" => Some(Self::Hoops),
            "rumble" => Some(Self::Rumble),
            "dropshot" => Some(Self::Dropshot),
            "snowday" => Some(Self::Snowday),
            _ => None,
        }
    }
}

/// Replay metadata stored in the database.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Replay {
    pub id: Uuid,
    pub file_path: String,
    pub game_mode: GameMode,
    pub processed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Player data for a specific replay.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ReplayPlayer {
    pub id: Uuid,
    pub replay_id: Uuid,
    pub player_name: String,
    pub team: i16,
    pub skill_rating: i32,
    pub created_at: DateTime<Utc>,
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

/// Input for creating a new replay record.
#[derive(Debug, Clone)]
pub struct CreateReplay {
    pub file_path: String,
    pub game_mode: GameMode,
}

/// Input for creating a new player record.
#[derive(Debug, Clone)]
pub struct CreateReplayPlayer {
    pub replay_id: Uuid,
    pub player_name: String,
    pub team: i16,
    pub skill_rating: i32,
}

/// Input for creating a new model record.
#[derive(Debug, Clone)]
pub struct CreateModel {
    pub name: String,
    pub version: i32,
    pub checkpoint_path: String,
    pub training_config: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
}

