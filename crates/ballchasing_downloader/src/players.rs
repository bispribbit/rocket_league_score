/// Player information extracted from metadata.
#[derive(Debug, Clone)]
pub struct ExtractedPlayer {
    /// Player name
    pub player_name: String,

    /// Team (0 for blue, 1 for orange)
    pub team: i16,

    /// Rank division
    pub rank_division: replay_structs::RankDivision,

    /// Whether the rank came directly from the player's profile (true)
    /// or was inferred from the lobby min/max midpoint (false).
    pub rank_known: bool,
}

/// Extracts player information from ballchasing metadata.
///
/// # Arguments
///
/// * `metadata` - JSON value from `replays.metadata` field
///
/// # Returns
///
/// Vector of player information with name, team, and rank division.
///
/// If a player's rank is null, the function will use the middle division of `min_rank` and `max_rank`
/// from the metadata if available.
///
/// # Errors
///
/// Returns an error if the metadata cannot be deserialized or if rank conversion fails.
#[expect(clippy::option_if_let_else, reason = "readibility")]
pub fn extract_players_from_metadata(
    metadata: &serde_json::Value,
) -> anyhow::Result<Vec<ExtractedPlayer>> {
    use anyhow::Context;
    use replay_structs::{RankDivision, RankInfo, ReplaySummary};

    let summary: ReplaySummary = serde_json::from_value(metadata.clone())
        .context("Failed to deserialize replay metadata")?;

    let mut players = Vec::new();

    // Helper function to extract rank division from a player's rank or use midpoint
    /// Result of resolving a player's rank: the division and whether it was
    /// directly reported by the API (`rank_known = true`) or inferred from
    /// lobby min/max (`rank_known = false`).
    struct ResolvedRank {
        division: RankDivision,
        rank_known: bool,
    }

    let get_rank_division = |player_rank: &Option<RankInfo>,
                             min_rank: &Option<RankInfo>,
                             max_rank: &Option<RankInfo>|
     -> anyhow::Result<Option<ResolvedRank>> {
        if let Some(rank) = player_rank {
            Ok(Some(ResolvedRank {
                division: rank.clone().into(),
                rank_known: true,
            }))
        } else if let (Some(min_rank), Some(max_rank)) = (min_rank, max_rank) {
            let min_division: RankDivision = min_rank.clone().into();
            let max_division: RankDivision = max_rank.clone().into();

            let min_idx = min_division.as_index();
            let max_idx = max_division.as_index();
            let mid_idx = usize::midpoint(min_idx, max_idx);

            let all_divisions: Vec<RankDivision> = RankDivision::all().collect();
            Ok(all_divisions
                .get(mid_idx)
                .copied()
                .map(|division| ResolvedRank {
                    division,
                    rank_known: false,
                }))
        } else {
            Ok(None)
        }
    };

    // Helper function to process a team's players
    let process_team = |team_players: &[replay_structs::PlayerSummary],
                        team_number: i16,
                        min_rank: &Option<RankInfo>,
                        max_rank: &Option<RankInfo>|
     -> anyhow::Result<Vec<ExtractedPlayer>> {
        let mut team_players_vec = Vec::new();
        for player in team_players {
            let resolved = get_rank_division(&player.rank, min_rank, max_rank)?;
            if let (Some(name), Some(resolved)) = (player.name.as_ref(), resolved) {
                team_players_vec.push(ExtractedPlayer {
                    player_name: name.clone(),
                    team: team_number,
                    rank_division: resolved.division,
                    rank_known: resolved.rank_known,
                });
            }
        }
        Ok(team_players_vec)
    };

    // Extract blue team players (team 0)
    if let Some(blue_team) = &summary.blue
        && let Some(team_players) = &blue_team.players
    {
        players.extend(process_team(
            team_players,
            0,
            &summary.min_rank,
            &summary.max_rank,
        )?);
    }

    // Extract orange team players (team 1)
    if let Some(orange_team) = &summary.orange
        && let Some(team_players) = &orange_team.players
    {
        players.extend(process_team(
            team_players,
            1,
            &summary.min_rank,
            &summary.max_rank,
        )?);
    }

    Ok(players)
}
