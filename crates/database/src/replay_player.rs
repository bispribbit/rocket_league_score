//! Repository functions for replay player operations.

use replay_structs::ReplayPlayer;
use uuid::Uuid;

use crate::get_pool;

/// Inserts multiple player records for a replay.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_replay_players(
    replay_ids: &[Uuid],
    player_names: &[String],
    teams: &[i16],
    skill_ratings: &[i32],
) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
    if replay_ids.is_empty() {
        return Ok(Vec::new());
    }

    let pool = get_pool();
    let ids: Vec<Uuid> = (0..replay_ids.len()).map(|_| Uuid::new_v4()).collect();

    sqlx::query_as::<_, ReplayPlayer>(
        "INSERT INTO replay_players (id, replay_id, player_name, team, skill_rating) SELECT * FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::smallint[], $5::integer[]) RETURNING id, replay_id, player_name, team, skill_rating, created_at",
    )
    .bind(&ids)
    .bind(replay_ids)
    .bind(player_names)
    .bind(teams)
    .bind(skill_ratings)
    .fetch_all(pool)
    .await
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
        SELECT id, replay_id, player_name, team, skill_rating, created_at
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
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn get_average_rating_for_replay(replay_id: Uuid) -> Result<Option<f64>, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        SELECT AVG(skill_rating::float) as average FROM replay_players WHERE replay_id = $1
        "#,
        replay_id
    )
    .fetch_one(pool)
    .await?;

    Ok(result.average)
}
