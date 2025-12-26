//! Common structs for replay metadata shared across crates.

use serde::{Deserialize, Serialize};

mod game_mode;
mod model;
mod rank;
mod replay;

pub use game_mode::*;
pub use model::*;
pub use rank::*;
pub use replay::*;

/// Rank information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RankInfo {
    /// Rank ID (e.g., "grand-champion-3")
    pub id: String,

    /// Tier number
    pub tier: Option<i32>,

    /// Division number (1-4)
    pub division: Option<i32>,

    /// Human-readable name
    pub name: Option<String>,
}

/// Player ID information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlayerId {
    /// Platform (steam, ps4, xbox, epic)
    pub platform: Option<String>,

    /// Platform-specific ID
    pub id: Option<String>,
}

/// Player summary information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlayerSummary {
    /// Player name
    pub name: Option<String>,

    /// Player ID
    pub id: Option<PlayerId>,

    /// Player rank
    pub rank: Option<RankInfo>,

    /// Whether player is MVP
    pub mvp: Option<bool>,

    /// Car ID
    pub car_id: Option<i32>,

    /// Car name
    pub car_name: Option<String>,
}

/// Team data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TeamData {
    /// Team color
    pub color: Option<String>,

    /// Team players
    pub players: Option<Vec<PlayerSummary>>,

    /// Team goals
    pub goals: Option<i32>,
}

/// Summary information for a replay from the list endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplaySummary {
    /// Unique replay ID
    pub id: String,

    /// Link to the replay API endpoint
    pub link: String,

    /// When the replay was uploaded
    pub created: String,

    /// Replay uploader information
    pub uploader: Option<Uploader>,

    /// Replay status
    pub status: Option<String>,

    /// Rocket League internal ID
    pub rocket_league_id: Option<String>,

    /// Match GUID
    pub match_guid: Option<String>,

    /// Replay title
    pub title: Option<String>,

    /// Map code
    pub map_code: Option<String>,

    /// Match type (Online, Offline, etc.)
    pub match_type: Option<String>,

    /// Team size
    pub team_size: Option<i32>,

    /// Playlist ID (e.g., "ranked-standard")
    pub playlist_id: Option<String>,

    /// Duration in seconds
    pub duration: Option<i32>,

    /// Whether the match went to overtime
    pub overtime: Option<bool>,

    /// Season number
    pub season: Option<i32>,

    /// Season type (e.g., "free2play")
    pub season_type: Option<String>,

    /// Match date
    pub date: Option<String>,

    /// Whether the date has timezone info
    pub date_has_timezone: Option<bool>,

    /// Visibility (public, private, etc.)
    pub visibility: Option<String>,

    /// Minimum rank in the match
    pub min_rank: Option<RankInfo>,

    /// Maximum rank in the match
    pub max_rank: Option<RankInfo>,

    /// Blue team data
    pub blue: Option<TeamData>,

    /// Orange team data
    pub orange: Option<TeamData>,

    /// Playlist name
    pub playlist_name: Option<String>,

    /// Map name
    pub map_name: Option<String>,
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

/// Game metadata structure (for pipeline.rs compatibility).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GameMetadata {
    /// Game ID
    pub id: String,

    /// Game data
    pub data: GameData,
}

/// Game data structure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GameData {
    /// Blue team
    pub blue: TeamInfo,

    /// Orange team
    pub orange: TeamInfo,
}

/// Team info from metadata (for pipeline.rs compatibility).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TeamInfo {
    /// Team players
    pub players: Vec<PlayerInfo>,
}

/// Player info from metadata (for pipeline.rs compatibility).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlayerInfo {
    /// Player name
    pub name: String,

    /// Player rank
    pub rank: Option<PlayerRank>,
}

/// Player rank from metadata (for pipeline.rs compatibility).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlayerRank {
    /// Rank ID
    pub id: String,

    /// Tier number
    pub tier: Option<i32>,

    /// Division number
    pub division: Option<i32>,
}

/// Player with rating information (replaces tuple usage).
#[derive(Debug, Clone)]
pub struct PlayerWithRating {
    /// Player name
    pub player_name: String,

    /// Player MMR
    pub mmr: i32,
}
