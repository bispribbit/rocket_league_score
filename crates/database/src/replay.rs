//! Repository functions for replay operations.

use std::path::Path;

use replay_structs::{DatasetSplit, DownloadStatus, GameMode, Rank, Replay};
use uuid::Uuid;

use crate::get_pool;

/// Constructs the file path for a replay based on its ID and rank.
///
/// The path format is: `replays/{game_mode}/{rank}/{uuid}.replay`
fn replay_file_path(replay_id: Uuid, game_mode: GameMode, rank: Rank) -> String {
    format!(
        "replays/{}/{}/{replay_id}.replay",
        game_mode.as_api_string(),
        rank.as_folder_name()
    )
}

/// Inserts multiple replay records, skipping duplicates.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_replays(
    ids: &[Uuid],
    game_modes: &[GameMode],
    ranks: &[Rank],
    metadata: &[serde_json::Value],
) -> Result<usize, sqlx::Error> {
    if ids.is_empty() {
        return Ok(0);
    }

    // Construct file paths for all replays
    let file_paths: Vec<String> = ids
        .iter()
        .zip(game_modes.iter())
        .zip(ranks.iter())
        .map(|((id, game_mode), rank)| replay_file_path(*id, *game_mode, *rank))
        .collect();

    let pool = get_pool();

    let result = sqlx::query!(
        r#"
        INSERT INTO replays (id, game_mode, rank, metadata, file_path)
        SELECT * FROM unnest($1::uuid[], $2::game_mode[], $3::rank[], $4::jsonb[], $5::text[])
        ON CONFLICT (id) DO NOTHING
        "#,
        ids as &[Uuid],
        game_modes as &[GameMode],
        ranks as &[Rank],
        metadata as &[serde_json::Value],
        &file_paths as &[String]
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
pub async fn replay_exists(id: Uuid) -> Result<bool, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT EXISTS(SELECT 1 FROM replays WHERE id = $1) as "exists!""#,
        id
    )
    .fetch_one(pool)
    .await?;

    Ok(result.exists)
}

/// Finds a replay by its file path.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn find_replay_by_path(file_path: &Path) -> Result<Option<Replay>, sqlx::Error> {
    let pool = get_pool();
    let file_path_str = file_path.to_string_lossy();
    sqlx::query_as!(
        Replay,
        r#"
        SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata, 
               download_status as "download_status: DownloadStatus", file_path, error_message,
               dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
        FROM replays
        WHERE file_path = $1
        "#,
        &file_path_str
    )
    .fetch_optional(pool)
    .await
}

/// Lists replays by rank with optional status filter.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_replays_by_rank(
    rank: Rank,
    status: Option<DownloadStatus>,
) -> Result<Vec<Replay>, sqlx::Error> {
    let pool = get_pool();
    match status {
        Some(status) => {
            sqlx::query_as!(
                Replay,
                r#"
                SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata,
                       download_status as "download_status: DownloadStatus", file_path, error_message,
                       dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
                FROM replays
                WHERE rank = $1 AND download_status = $2
                ORDER BY created_at
                "#,
                rank as Rank,
                status as DownloadStatus
            )
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as!(
                Replay,
                r#"
                SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata,
                       download_status as "download_status: DownloadStatus", file_path, error_message,
                       dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
                FROM replays
                WHERE rank = $1
                ORDER BY created_at
                "#,
                rank as Rank
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
pub async fn list_pending_downloads(
    rank: Option<Rank>,
    limit: i64,
) -> Result<Vec<Replay>, sqlx::Error> {
    let pool = get_pool();
    match rank {
        Some(rank) => {
            sqlx::query_as!(
                Replay,
                r#"
                SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata,
                       download_status as "download_status: DownloadStatus", file_path, error_message,
                       dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
                FROM replays
                WHERE rank = $1 AND download_status = 'not_downloaded'
                ORDER BY created_at
                LIMIT $2
                "#,
                rank as Rank,
                limit
            )
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as!(
                Replay,
                r#"
                SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata,
                       download_status as "download_status: DownloadStatus", file_path, error_message,
                       dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
                FROM replays
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
pub async fn update_replay_status(
    id: Uuid,
    status: DownloadStatus,
    file_path: Option<&str>,
    error_message: Option<&str>,
) -> Result<(), sqlx::Error> {
    let pool = get_pool();
    sqlx::query!(
        r#"
        UPDATE replays
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
pub async fn mark_replay_download_in_progress(id: Uuid) -> Result<(), sqlx::Error> {
    update_replay_status(id, DownloadStatus::InProgress, None, None).await
}

/// Marks a replay as downloaded with its file path.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_replay_downloaded(id: Uuid, file_path: &str) -> Result<(), sqlx::Error> {
    update_replay_status(id, DownloadStatus::Downloaded, Some(file_path), None).await
}

/// Marks a replay as failed with an error message.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn mark_replay_failed(id: Uuid, error_message: &str) -> Result<(), sqlx::Error> {
    update_replay_status(id, DownloadStatus::Failed, None, Some(error_message)).await
}

/// Resets failed downloads back to `not_downloaded` status.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn reset_failed_downloads() -> Result<u64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        UPDATE replays
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
pub async fn reset_in_progress_downloads() -> Result<u64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"
        UPDATE replays
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
pub async fn count_replays_by_rank(rank: Rank) -> Result<i64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT COUNT(*) as "count!" FROM replays WHERE rank = $1"#,
        rank as Rank
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
pub async fn count_replays_by_rank_and_status(
    rank: Rank,
    status: DownloadStatus,
) -> Result<i64, sqlx::Error> {
    let pool = get_pool();
    let result = sqlx::query!(
        r#"SELECT COUNT(*) as "count!" FROM replays WHERE rank = $1 AND download_status = $2"#,
        rank as Rank,
        status as DownloadStatus
    )
    .fetch_one(pool)
    .await?;

    Ok(result.count)
}

// ============================================================================
// Dataset Split Functions
// ============================================================================

/// Assigns dataset splits to unassigned downloaded replays.
///
/// Uses random assignment to achieve the target training ratio.
/// Only assigns replays that have `dataset_split = NULL` and `download_status = 'downloaded'`.
///
/// # Arguments
///
/// * `train_ratio` - Target ratio for training set (e.g., 0.9 for 90% training).
///
/// # Returns
///
/// A tuple of (`training_count`, `evaluation_count`) for newly assigned replays.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn assign_dataset_splits(train_ratio: f64) -> Result<DatasetSplitCounts, sqlx::Error> {
    let pool = get_pool();

    // First, count how many unassigned downloaded replays we have
    let unassigned = sqlx::query!(
        r#"
        SELECT COUNT(*) as "count!"
        FROM replays
        WHERE dataset_split IS NULL AND download_status = 'downloaded'
        "#
    )
    .fetch_one(pool)
    .await?;

    let total_unassigned = unassigned.count;
    if total_unassigned == 0 {
        return Ok(DatasetSplitCounts {
            training: 0,
            evaluation: 0,
        });
    }

    // Calculate how many should go to training
    let training_count = (total_unassigned as f64 * train_ratio).round() as i64;

    // Assign training set (random selection)
    let training_result = sqlx::query!(
        r#"
        UPDATE replays
        SET dataset_split = 'training', updated_at = NOW()
        WHERE id IN (
            SELECT id FROM replays
            WHERE dataset_split IS NULL AND download_status = 'downloaded'
            ORDER BY RANDOM()
            LIMIT $1
        )
        "#,
        training_count
    )
    .execute(pool)
    .await?;

    // Assign remaining to evaluation
    let eval_result = sqlx::query!(
        r#"
        UPDATE replays
        SET dataset_split = 'evaluation', updated_at = NOW()
        WHERE dataset_split IS NULL AND download_status = 'downloaded'
        "#
    )
    .execute(pool)
    .await?;

    Ok(DatasetSplitCounts {
        training: training_result.rows_affected() as i64,
        evaluation: eval_result.rows_affected() as i64,
    })
}

/// Counts of replays in each dataset split.
#[derive(Debug, Clone, Default)]
pub struct DatasetSplitCounts {
    /// Number of replays in training set.
    pub training: i64,
    /// Number of replays in evaluation set.
    pub evaluation: i64,
}

/// Returns counts of replays by dataset split.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn count_replays_by_split() -> Result<DatasetSplitCounts, sqlx::Error> {
    let pool = get_pool();

    let training = sqlx::query!(
        r#"
        SELECT COUNT(*) as "count!"
        FROM replays
        WHERE dataset_split = 'training' AND download_status = 'downloaded'
        "#
    )
    .fetch_one(pool)
    .await?;

    let evaluation = sqlx::query!(
        r#"
        SELECT COUNT(*) as "count!"
        FROM replays
        WHERE dataset_split = 'evaluation' AND download_status = 'downloaded'
        "#
    )
    .fetch_one(pool)
    .await?;

    Ok(DatasetSplitCounts {
        training: training.count,
        evaluation: evaluation.count,
    })
}

/// Lists replays by dataset split.
///
/// Only returns downloaded replays.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn list_replays_by_split(split: DatasetSplit) -> Result<Vec<Replay>, sqlx::Error> {
    let pool = get_pool();
    sqlx::query_as!(
        Replay,
        r#"
        SELECT id, game_mode as "game_mode: GameMode", rank as "rank: Rank", metadata,
               download_status as "download_status: DownloadStatus", file_path, error_message,
               dataset_split as "dataset_split: DatasetSplit", created_at, updated_at
        FROM replays
        WHERE dataset_split = $1 AND download_status = 'downloaded'
        ORDER BY created_at
        "#,
        split as DatasetSplit
    )
    .fetch_all(pool)
    .await
}

/// Updates the dataset split for a specific replay.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn update_replay_split(replay_id: Uuid, split: DatasetSplit) -> Result<(), sqlx::Error> {
    let pool = get_pool();
    sqlx::query!(
        r#"
        UPDATE replays
        SET dataset_split = $2, updated_at = NOW()
        WHERE id = $1
        "#,
        replay_id,
        split as DatasetSplit
    )
    .execute(pool)
    .await?;

    Ok(())
}
