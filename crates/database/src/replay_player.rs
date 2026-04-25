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
    let rank_knowns: Vec<bool> = players.iter().map(|p| p.rank_known).collect();

    let pool = get_pool();

    sqlx::query!(
        r#"
        INSERT INTO replay_players (replay_id, player_name, team, rank_division, rank_known)
        SELECT * FROM unnest($1::uuid[], $2::text[], $3::smallint[], $4::rank_division[], $5::boolean[])
        ON CONFLICT (replay_id, player_name) DO NOTHING
        "#,
        &replay_ids as &[Uuid],
        &player_names as &[String],
        &teams as &[i16],
        &rank_divisions as &[RankDivision],
        &rank_knowns as &[bool]
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
        SELECT id, replay_id, player_name, team, rank_division as "rank_division: RankDivision", rank_known, created_at
        FROM replay_players
        WHERE replay_id = $1
        ORDER BY team, player_name
        "#,
        replay_id
    )
    .fetch_all(pool)
    .await
}

/// Updates the `smurf_score` for a specific player in a replay.
///
/// `smurf_score` = predicted_mmr − mmr_middle(rank_division).
/// A large positive value indicates above-rank performance.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn update_smurf_score(
    replay_id: Uuid,
    player_name: &str,
    smurf_score: f32,
) -> Result<(), sqlx::Error> {
    let pool = get_pool();
    sqlx::query!(
        r#"
        UPDATE replay_players
        SET smurf_score = $3
        WHERE replay_id = $1 AND player_name = $2
        "#,
        replay_id,
        player_name,
        smurf_score as f64,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Bulk-update smurf scores for many players in a single statement.
///
/// `updates` is a slice of `(replay_id, player_name, smurf_score)` triples.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn bulk_update_smurf_scores(updates: &[(Uuid, String, f32)]) -> Result<(), sqlx::Error> {
    if updates.is_empty() {
        return Ok(());
    }
    let pool = get_pool();
    let replay_ids: Vec<Uuid> = updates.iter().map(|(id, _, _)| *id).collect();
    let player_names: Vec<String> = updates.iter().map(|(_, name, _)| name.clone()).collect();
    let scores: Vec<f64> = updates
        .iter()
        .map(|(_, _, score)| f64::from(*score))
        .collect();

    sqlx::query!(
        r#"
        UPDATE replay_players AS rp
        SET smurf_score = data.score
        FROM unnest($1::uuid[], $2::text[], $3::float8[]) AS data(replay_id, player_name, score)
        WHERE rp.replay_id = data.replay_id AND rp.player_name = data.player_name
        "#,
        &replay_ids as &[Uuid],
        &player_names as &[String],
        &scores as &[f64],
    )
    .execute(pool)
    .await?;

    Ok(())
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
