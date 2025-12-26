use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::ballchasing::GameMode;

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
