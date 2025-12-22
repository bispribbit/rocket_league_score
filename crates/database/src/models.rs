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
