//! Repository functions for replay operations.

use uuid::Uuid;

use crate::get_pool;
use crate::models::{CreateReplay, GameMode, Replay};

/// Creates a new replay record.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_replay(input: CreateReplay) -> Result<Replay, sqlx::Error> {
    let id = Uuid::new_v4();
    let pool = get_pool();

    sqlx::query_as!(
        Replay,
        r#"
        INSERT INTO replays (id, file_path, game_mode)
        VALUES ($1, $2, $3)
        RETURNING id, file_path, game_mode as "game_mode: GameMode", processed_at, created_at
        "#,
        id,
        input.file_path,
        input.game_mode as GameMode
    )
    .fetch_one(pool)
    .await
}

/// Finds a replay by its file path.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn find_replay_by_path(file_path: &str) -> Result<Option<Replay>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        Replay,
        r#"
        SELECT id, file_path, game_mode as "game_mode: GameMode", processed_at, created_at
        FROM replays
        WHERE file_path = $1
        "#,
        file_path
    )
    .fetch_optional(pool)
    .await
}

/// Finds a replay by its ID.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn find_replay_by_id(id: Uuid) -> Result<Option<Replay>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        Replay,
        r#"
        SELECT id, file_path, game_mode as "game_mode: GameMode", processed_at, created_at
        FROM replays
        WHERE id = $1
        "#,
        id
    )
    .fetch_optional(pool)
    .await
}

/// Lists all replays for a given game mode.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_replays_by_game_mode(game_mode: GameMode) -> Result<Vec<Replay>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        Replay,
        r#"
        SELECT id, file_path, game_mode as "game_mode: GameMode", processed_at, created_at
        FROM replays
        WHERE game_mode = $1
        ORDER BY created_at DESC
        "#,
        game_mode as GameMode
    )
    .fetch_all(pool)
    .await
}

/// Marks a replay as processed.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_replay_processed(id: Uuid) -> Result<(), sqlx::Error> {
    let pool = get_pool();
    sqlx::query!(
        r#"
        UPDATE replays
        SET processed_at = NOW()
        WHERE id = $1
        "#,
        id
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Counts replays by game mode.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn count_replays_by_game_mode(game_mode: GameMode) -> Result<i64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        SELECT COUNT(*) as "count!" FROM replays WHERE game_mode = $1
        "#,
        game_mode as GameMode
    )
    .fetch_one(pool)
    .await?;

    Ok(result.count)
}

