//! Repository functions for ballchasing replay operations.

use replay_structs::{BallchasingRank, BallchasingRankStats, BallchasingReplay, DownloadStatus};
use uuid::Uuid;

use crate::get_pool;

/// Inserts multiple ballchasing replay records, skipping duplicates.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_ballchasing_replays(
    ids: &[Uuid],
    ranks: &[BallchasingRank],
    metadata: &[serde_json::Value],
) -> Result<usize, sqlx::Error> {
    if ids.is_empty() {
        return Ok(0);
    }

    let pool = get_pool();

    let result = sqlx::query!(
        r#"
        INSERT INTO ballchasing_replays (id, rank, metadata)
        SELECT * FROM unnest($1::uuid[], $2::ballchasing_rank[], $3::jsonb[])
        ON CONFLICT (id) DO NOTHING
        "#,
        ids as &[Uuid],
        ranks as &[BallchasingRank],
        metadata as &[serde_json::Value]
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

/// Checks if a replay exists by ID.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn ballchasing_replay_exists(id: Uuid) -> Result<bool, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT EXISTS(SELECT 1 FROM ballchasing_replays WHERE id = $1) as "exists!""#,
        id
    )
    .fetch_one(pool)
    .await?;

    Ok(result.exists)
}

/// Finds a replay by its ID.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn find_ballchasing_replay_by_id(
    id: Uuid,
) -> Result<Option<BallchasingReplay>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        BallchasingReplay,
        r#"
        SELECT id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
        FROM ballchasing_replays
        WHERE id = $1
        "#,
        id
    )
    .fetch_optional(pool)
    .await
}

/// Lists replays by rank with optional status filter.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_ballchasing_replays_by_rank(
    rank: BallchasingRank,
    status: Option<DownloadStatus>,
) -> Result<Vec<BallchasingReplay>, sqlx::Error> {
    let pool = get_pool();
    match status {
        Some(status) => {
            sqlx::query_as!(
                BallchasingReplay,
                r#"
                SELECT id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
                FROM ballchasing_replays
                WHERE rank = $1 AND download_status = $2
                ORDER BY created_at
                "#,
                rank as BallchasingRank,
                status as DownloadStatus
            )
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as!(
                BallchasingReplay,
                r#"
                SELECT id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
                FROM ballchasing_replays
                WHERE rank = $1
                ORDER BY created_at
                "#,
                rank as BallchasingRank
            )
            .fetch_all(pool)
            .await
        }
    }
}

/// Lists replays pending download (status = `not_downloaded`).
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_pending_ballchasing_downloads(
    rank: Option<BallchasingRank>,
    limit: i64,
) -> Result<Vec<BallchasingReplay>, sqlx::Error> {
    let pool = get_pool();
    match rank {
        Some(rank) => {
            sqlx::query_as!(
                BallchasingReplay,
                r#"
                SELECT id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
                FROM ballchasing_replays
                WHERE rank = $1 AND download_status = 'not_downloaded'
                ORDER BY created_at
                LIMIT $2
                "#,
                rank as BallchasingRank,
                limit
            )
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as!(
                BallchasingReplay,
                r#"
                SELECT id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
                FROM ballchasing_replays
                WHERE download_status = 'not_downloaded'
                ORDER BY created_at
                LIMIT $1
                "#,
                limit
            )
            .fetch_all(pool)
            .await
        }
    }
}

/// Updates the download status of a replay.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn update_ballchasing_replay_status(
    id: Uuid,
    status: DownloadStatus,
    file_path: Option<&str>,
    error_message: Option<&str>,
) -> Result<(), sqlx::Error> {
    let pool = get_pool();
    sqlx::query!(
        r#"
        UPDATE ballchasing_replays
        SET 
            download_status = $2,
            file_path = COALESCE($3, file_path),
            error_message = $4,
            updated_at = NOW()
        WHERE id = $1
        "#,
        id,
        status as DownloadStatus,
        file_path,
        error_message
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Marks a replay as in progress.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_ballchasing_replay_in_progress(id: Uuid) -> Result<(), sqlx::Error> {
    update_ballchasing_replay_status(id, DownloadStatus::InProgress, None, None).await
}

/// Marks a replay as downloaded with its file path.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_ballchasing_replay_downloaded(
    id: Uuid,
    file_path: &str,
) -> Result<(), sqlx::Error> {
    update_ballchasing_replay_status(id, DownloadStatus::Downloaded, Some(file_path), None).await
}

/// Marks a replay as failed with an error message.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_ballchasing_replay_failed(
    id: Uuid,
    error_message: &str,
) -> Result<(), sqlx::Error> {
    update_ballchasing_replay_status(id, DownloadStatus::Failed, None, Some(error_message)).await
}

/// Resets failed downloads back to `not_downloaded` status.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn reset_failed_ballchasing_downloads() -> Result<u64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        UPDATE ballchasing_replays
        SET 
            download_status = 'not_downloaded',
            error_message = NULL,
            updated_at = NOW()
        WHERE download_status = 'failed'
        "#
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Resets in-progress downloads back to `not_downloaded` status.
/// Useful for recovering from interrupted downloads.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn reset_in_progress_ballchasing_downloads() -> Result<u64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        UPDATE ballchasing_replays
        SET 
            download_status = 'not_downloaded',
            updated_at = NOW()
        WHERE download_status = 'in_progress'
        "#
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}

/// Counts replays by rank.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn count_ballchasing_replays_by_rank(rank: BallchasingRank) -> Result<i64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT COUNT(*) as "count!" FROM ballchasing_replays WHERE rank = $1"#,
        rank as BallchasingRank
    )
    .fetch_one(pool)
    .await?;

    Ok(result.count)
}

/// Counts replays by rank and status.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn count_ballchasing_replays_by_rank_and_status(
    rank: BallchasingRank,
    status: DownloadStatus,
) -> Result<i64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT COUNT(*) as "count!" FROM ballchasing_replays WHERE rank = $1 AND download_status = $2"#,
        rank as BallchasingRank,
        status as DownloadStatus
    )
    .fetch_one(pool)
    .await?;

    Ok(result.count)
}

/// Gets statistics for all ranks.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn get_ballchasing_replay_stats() -> Result<Vec<BallchasingRankStats>, sqlx::Error> {
    let pool = get_pool();
    let rows = sqlx::query!(
        r#"
        SELECT 
            rank as "rank!: BallchasingRank",
            COUNT(*) FILTER (WHERE download_status = 'not_downloaded') as "not_downloaded!",
            COUNT(*) FILTER (WHERE download_status = 'in_progress') as "in_progress!",
            COUNT(*) FILTER (WHERE download_status = 'downloaded') as "downloaded!",
            COUNT(*) FILTER (WHERE download_status = 'failed') as "failed!"
        FROM ballchasing_replays
        GROUP BY rank
        ORDER BY rank
        "#
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| BallchasingRankStats {
            rank: row.rank,
            not_downloaded: row.not_downloaded,
            in_progress: row.in_progress,
            downloaded: row.downloaded,
            failed: row.failed,
        })
        .collect())
}
