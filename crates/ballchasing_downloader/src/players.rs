/// Player information extracted from metadata.
#[derive(Debug, Clone)]
pub struct ExtractedPlayer {
    /// Player name
    pub player_name: String,

    /// Team (0 for blue, 1 for orange)
    pub team: i16,

    /// Rank division
    pub rank_division: replay_structs::RankDivision,
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
pub fn extract_players_from_metadata(
    metadata: &serde_json::Value,
) -> anyhow::Result<Vec<ExtractedPlayer>> {
    use anyhow::Context;
    use replay_structs::{RankDivision, ReplaySummary};

    let summary: ReplaySummary = serde_json::from_value(metadata.clone())
        .context("Failed to deserialize replay metadata")?;

    let mut players = Vec::new();

    // Extract blue team players (team 0)
    if let Some(blue_team) = &summary.blue
        && let Some(team_players) = &blue_team.players
    {
        for player in team_players {
            let rank_division = if let Some(rank) = &player.rank {
                RankDivision::from_rank_id(&rank.id, rank.division)
            } else if let (Some(min_rank), Some(max_rank)) = (&summary.min_rank, &summary.max_rank)
            {
                // Use middle division if player rank is null
                let min_division = RankDivision::from_rank_id(&min_rank.id, min_rank.division)
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert min_rank to RankDivision"))?;
                let max_division = RankDivision::from_rank_id(&max_rank.id, max_rank.division)
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert max_rank to RankDivision"))?;

                // Find middle division between min and max
                let all_divisions: Vec<RankDivision> = RankDivision::all().collect();
                let min_idx = all_divisions
                    .iter()
                    .position(|&d| d == min_division)
                    .ok_or_else(|| anyhow::anyhow!("min_rank not found in all divisions"))?;
                let max_idx = all_divisions
                    .iter()
                    .position(|&d| d == max_division)
                    .ok_or_else(|| anyhow::anyhow!("max_rank not found in all divisions"))?;
                let mid_idx = usize::midpoint(min_idx, max_idx);
                all_divisions.get(mid_idx).copied()
            } else {
                None
            };

            if let (Some(name), Some(rank_div)) = (player.name.as_ref(), rank_division) {
                players.push(ExtractedPlayer {
                    player_name: name.clone(),
                    team: 0,
                    rank_division: rank_div,
                });
            }
        }
    }

    // Extract orange team players (team 1)
    if let Some(orange_team) = &summary.orange
        && let Some(team_players) = &orange_team.players
    {
        for player in team_players {
            let rank_division = if let Some(rank) = &player.rank {
                RankDivision::from_rank_id(&rank.id, rank.division)
            } else if let (Some(min_rank), Some(max_rank)) = (&summary.min_rank, &summary.max_rank)
            {
                // Use middle division if player rank is null
                let min_division = RankDivision::from_rank_id(&min_rank.id, min_rank.division)
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert min_rank to RankDivision"))?;
                let max_division = RankDivision::from_rank_id(&max_rank.id, max_rank.division)
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert max_rank to RankDivision"))?;

                // Find middle division between min and max
                let all_divisions: Vec<RankDivision> = RankDivision::all().collect();
                let min_idx = all_divisions
                    .iter()
                    .position(|&d| d == min_division)
                    .ok_or_else(|| anyhow::anyhow!("min_rank not found in all divisions"))?;
                let max_idx = all_divisions
                    .iter()
                    .position(|&d| d == max_division)
                    .ok_or_else(|| anyhow::anyhow!("max_rank not found in all divisions"))?;
                let mid_idx = usize::midpoint(min_idx, max_idx);
                all_divisions.get(mid_idx).copied()
            } else {
                None
            };

            if let (Some(name), Some(rank_div)) = (player.name.as_ref(), rank_division) {
                players.push(ExtractedPlayer {
                    player_name: name.clone(),
                    team: 1,
                    rank_division: rank_div,
                });
            }
        }
    }

    Ok(players)
}
