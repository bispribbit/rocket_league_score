use serde::{Deserialize, Serialize};

use crate::PlayerSummary;

/// Team assignment for a player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, strum::Display)]
pub enum Team {
    #[default]
    Blue,
    Orange,
}

impl From<u8> for Team {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Blue,
            _ => Self::Orange,
        }
    }
}

impl From<Team> for u8 {
    fn from(team: Team) -> Self {
        match team {
            Team::Blue => 0,
            Team::Orange => 1,
        }
    }
}

impl From<i32> for Team {
    fn from(value: i32) -> Self {
        match value % 2 {
            0 => Self::Blue,
            _ => Self::Orange,
        }
    }
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
