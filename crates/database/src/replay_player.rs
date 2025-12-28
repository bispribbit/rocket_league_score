//! Repository functions for replay player operations.

use replay_structs::{RankDivision, ReplayPlayer};
use uuid::Uuid;

use crate::get_pool;

/// Inserts multiple player records for a replay.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_replay_players(players: &[ReplayPlayer]) -> Result<(), sqlx::Error> {
    if players.is_empty() {
        return Ok(());
    }

    let replay_ids: Vec<Uuid> = players.iter().map(|p| p.replay_id).collect();
    let player_names: Vec<String> = players.iter().map(|p| p.player_name.clone()).collect();
    let teams: Vec<i16> = players.iter().map(|p| p.team).collect();
    let rank_divisions: Vec<RankDivision> = players.iter().map(|p| p.rank_division).collect();

    let pool = get_pool();

    sqlx::query!(
        r#"
        INSERT INTO replay_players (replay_id, player_name, team, rank_division)
        SELECT * FROM unnest($1::uuid[], $2::text[], $3::smallint[], $4::rank_division[])
        ON CONFLICT (replay_id, player_name) DO NOTHING
        "#,
        &replay_ids as &[Uuid],
        &player_names as &[String],
        &teams as &[i16],
        &rank_divisions as &[RankDivision]
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Lists all players for a given replay.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_replay_players_by_replay(
    replay_id: Uuid,
) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        ReplayPlayer,
        r#"
        SELECT id, replay_id, player_name, team, rank_division as "rank_division: RankDivision", created_at
        FROM replay_players
        WHERE replay_id = $1
        ORDER BY team, player_name
        "#,
        replay_id
    )
    .fetch_all(pool)
    .await
}

/// Gets the average skill rating for a replay.
///
/// Uses the middle MMR value of each player's rank division.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn get_average_rating_for_replay(replay_id: Uuid) -> Result<Option<f64>, sqlx::Error> {
    let players = list_replay_players_by_replay(replay_id).await?;

    if players.is_empty() {
        return Ok(None);
    }

    let sum: i64 = players
        .iter()
        .map(|p| i64::from(p.rank_division.mmr_middle()))
        .sum();
    let count = players.len() as f64;

    Ok(Some(sum as f64 / count))
}
