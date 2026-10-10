#![expect(clippy::indexing_slicing)]

//! Per-player match statistics from parsed Rocket League replays.
//!
//! [`compute_player_match_stats`] summarises how each of the six players played a whole
//! match (movement, boost, positioning, ball control, mechanics, scoreboard…), and
//! [`compute_player_window_stats`] does the same for a slice of it. These summaries are the
//! inputs of the skill model (`skill_model`).

use std::collections::{BTreeMap, HashSet};

use replay_structs::{ParsedReplay, Team};

mod match_stats;
pub use match_stats::{
    MATCH_STAT_COUNT, MATCH_STAT_NAMES, PlayerMatchStats, compute_player_match_stats,
    compute_player_window_stats, goal_windows, time_windows,
};

/// Number of players per team in a 3v3 match.
pub const PLAYERS_PER_TEAM: usize = 3;
/// Number of players in a 3v3 match.
pub const TOTAL_PLAYERS: usize = PLAYERS_PER_TEAM * 2;

/// How a name was seen in a replay.
#[derive(Debug, Clone, Default)]
struct Sightings {
    name: String,
    blue_frames: usize,
    orange_frames: usize,
    /// Team from the header scoreboard, which outranks the per-frame attribute.
    header_team: Option<Team>,
}

impl Sightings {
    fn frames(&self) -> usize {
        self.blue_frames + self.orange_frames
    }

    fn blue_share(&self) -> f64 {
        self.blue_frames as f64 / self.frames().max(1) as f64
    }
}

/// Names the parser gives cars it could not link to a player (`Player_<actor id>`).
fn is_placeholder_name(name: &str) -> bool {
    name.strip_prefix("Player_")
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
}

/// Canonical ordered roster for a 3v3 match.
///
/// Slots 0–2 hold the blue team sorted alphabetically by player name; slots 3–5 hold the
/// orange team sorted the same way. Empty strings mark unused slots.
#[derive(Debug, Clone)]
pub struct PlayerRoster {
    /// Six names: `[blue_0, blue_1, blue_2, orange_0, orange_1, orange_2]`.
    pub names: [String; TOTAL_PLAYERS],
}

impl PlayerRoster {
    /// Builds the roster from a parsed replay.
    ///
    /// Neither source is complete on its own: the header scoreboard leaves out players who
    /// quit early, and the per-frame team attribute flips back and forth in some replays.
    /// So every real name from either source is a candidate; the header decides the team
    /// when it lists the player, otherwise the team the player is seen on most (a full lobby
    /// without header teams is split 3/3 by how often each name is seen on blue). Each team
    /// keeps the three players seen in the most frames.
    #[must_use]
    pub fn from_parsed(parsed: &ParsedReplay) -> Self {
        let mut by_name: BTreeMap<String, Sightings> = BTreeMap::new();
        for frame in &parsed.frames {
            for player in &frame.players {
                if is_placeholder_name(&player.name) {
                    continue;
                }
                let sightings =
                    by_name
                        .entry((*player.name).clone())
                        .or_insert_with(|| Sightings {
                            name: (*player.name).clone(),
                            ..Sightings::default()
                        });
                match player.team {
                    Team::Blue => sightings.blue_frames += 1,
                    Team::Orange => sightings.orange_frames += 1,
                }
            }
        }
        for player in &parsed.header_player_stats {
            if player.name.is_empty() {
                continue;
            }
            let sightings = by_name
                .entry(player.name.clone())
                .or_insert_with(|| Sightings {
                    name: player.name.clone(),
                    ..Sightings::default()
                });
            sightings.header_team = player.team;
        }

        let mut candidates: Vec<Sightings> = by_name.into_values().collect();
        let no_header_teams = candidates.iter().all(|s| s.header_team.is_none());
        let mut blue: Vec<Sightings> = Vec::new();
        let mut orange: Vec<Sightings> = Vec::new();
        if no_header_teams && candidates.len() == TOTAL_PLAYERS {
            candidates.sort_by(|a, b| b.blue_share().total_cmp(&a.blue_share()));
            for (index, sightings) in candidates.into_iter().enumerate() {
                if index < PLAYERS_PER_TEAM {
                    blue.push(sightings);
                } else {
                    orange.push(sightings);
                }
            }
        } else {
            for sightings in candidates {
                let team = sightings.header_team.unwrap_or(
                    if sightings.blue_frames >= sightings.orange_frames {
                        Team::Blue
                    } else {
                        Team::Orange
                    },
                );
                match team {
                    Team::Blue => blue.push(sightings),
                    Team::Orange => orange.push(sightings),
                }
            }
        }
        let keep_three = |mut team: Vec<Sightings>| -> Vec<String> {
            // Header-listed players first, then by time on the pitch.
            team.sort_by(|a, b| {
                b.header_team
                    .is_some()
                    .cmp(&a.header_team.is_some())
                    .then(b.frames().cmp(&a.frames()))
            });
            let mut names: Vec<String> = team
                .into_iter()
                .take(PLAYERS_PER_TEAM)
                .map(|s| s.name)
                .collect();
            names.sort();
            names
        };
        let blue = keep_three(blue);
        let orange = keep_three(orange);
        Self {
            names: core::array::from_fn(|slot| {
                if slot < PLAYERS_PER_TEAM {
                    blue.get(slot).cloned().unwrap_or_default()
                } else {
                    orange
                        .get(slot - PLAYERS_PER_TEAM)
                        .cloned()
                        .unwrap_or_default()
                }
            }),
        }
    }
}

/// Frame indices inside goal-replay windows: from a goal frame (exclusive) to the next
/// kickoff (exclusive), capped at 600 frames. Those frames show the celebration camera,
/// teleporting cars and a paused clock, so the stats skip them.
fn build_goal_replay_excluded_set(
    goal_frames: &[usize],
    kickoff_frames: &[usize],
) -> HashSet<usize> {
    let mut excluded = HashSet::new();
    for &goal in goal_frames {
        let next_kickoff = kickoff_frames
            .iter()
            .find(|&&kickoff| kickoff > goal)
            .copied()
            .unwrap_or(usize::MAX);
        for frame_index in (goal + 1)..next_kickoff.min(goal + 600) {
            excluded.insert(frame_index);
        }
    }
    excluded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_replay_runs_until_the_next_kickoff() {
        let excluded = build_goal_replay_excluded_set(&[100], &[50, 130]);
        assert!(!excluded.contains(&100));
        assert!(excluded.contains(&101));
        assert!(excluded.contains(&129));
        assert!(!excluded.contains(&130));
    }
}
