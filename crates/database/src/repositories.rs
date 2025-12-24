//! Repository functions for database operations.

use uuid::Uuid;

use crate::get_pool;
use crate::models::{
    BallchasingRank, BallchasingRankStats, BallchasingReplay, CreateBallchasingReplay, CreateModel,
    CreateReplay, CreateReplayPlayer, DownloadStatus, GameMode, Model, Replay, ReplayPlayer,
};

/// Repository for replay operations.
pub struct ReplayRepository;

impl ReplayRepository {
    /// Creates a new replay record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create(input: CreateReplay) -> Result<Replay, sqlx::Error> {
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
    pub async fn find_by_path(file_path: &str) -> Result<Option<Replay>, sqlx::Error> {
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
    pub async fn find_by_id(id: Uuid) -> Result<Option<Replay>, sqlx::Error> {
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
    pub async fn list_by_game_mode(game_mode: GameMode) -> Result<Vec<Replay>, sqlx::Error> {
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
    pub async fn mark_processed(id: Uuid) -> Result<(), sqlx::Error> {
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
    pub async fn count_by_game_mode(game_mode: GameMode) -> Result<i64, sqlx::Error> {
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
}

/// Repository for replay player operations.
pub struct ReplayPlayerRepository;

impl ReplayPlayerRepository {
    /// Creates a new replay player record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create(input: CreateReplayPlayer) -> Result<ReplayPlayer, sqlx::Error> {
        let id = Uuid::new_v4();
        let pool = get_pool();

        sqlx::query_as!(
            ReplayPlayer,
            r#"
            INSERT INTO replay_players (id, replay_id, player_name, team, skill_rating)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, replay_id, player_name, team, skill_rating, created_at
            "#,
            id,
            input.replay_id,
            input.player_name,
            input.team,
            input.skill_rating
        )
        .fetch_one(pool)
        .await
    }

    /// Creates multiple player records for a replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create_many(
        inputs: Vec<CreateReplayPlayer>,
    ) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
        let mut players = Vec::with_capacity(inputs.len());

        for input in inputs {
            let player = Self::create(input).await?;
            players.push(player);
        }

        Ok(players)
    }

    /// Lists all players for a given replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn list_by_replay(replay_id: Uuid) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
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
    pub async fn average_rating_for_replay(replay_id: Uuid) -> Result<Option<f64>, sqlx::Error> {
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
}

/// Repository for model operations.
pub struct ModelRepository;

impl ModelRepository {
    /// Creates a new model record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create(input: CreateModel) -> Result<Model, sqlx::Error> {
        let id = Uuid::new_v4();
        let pool = get_pool();

        sqlx::query_as!(
            Model,
            r#"
            INSERT INTO models (id, name, version, checkpoint_path, training_config, metrics)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, name, version, checkpoint_path, training_config, metrics, trained_at
            "#,
            id,
            input.name,
            input.version,
            input.checkpoint_path,
            input.training_config,
            input.metrics
        )
        .fetch_one(pool)
        .await
    }

    /// Finds a model by name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_by_name_version(
        name: &str,
        version: i32,
    ) -> Result<Option<Model>, sqlx::Error> {
        let pool = get_pool();
        sqlx::query_as!(
            Model,
            r#"
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1 AND version = $2
            "#,
            name,
            version
        )
        .fetch_optional(pool)
        .await
    }

    /// Gets the latest version of a model by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_latest(name: &str) -> Result<Option<Model>, sqlx::Error> {
        let pool = get_pool();
        sqlx::query_as!(
            Model,
            r#"
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1
            ORDER BY version DESC
            LIMIT 1
            "#,
            name
        )
        .fetch_optional(pool)
        .await
    }

    /// Gets the next version number for a model name.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn next_version(name: &str) -> Result<i32, sqlx::Error> {
        let pool = get_pool();
        let result = sqlx::query!(
            r#"
            SELECT MAX(version) as max_version FROM models WHERE name = $1
            "#,
            name
        )
        .fetch_one(pool)
        .await?;

        Ok(result.max_version.unwrap_or(0) + 1)
    }

    /// Lists all versions of a model.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn list_versions(name: &str) -> Result<Vec<Model>, sqlx::Error> {
        let pool = get_pool();
        sqlx::query_as!(
            Model,
            r#"
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1
            ORDER BY version DESC
            "#,
            name
        )
        .fetch_all(pool)
        .await
    }
}

// ============================================================================
// Ballchasing Replay Repository
// ============================================================================

/// Repository for ballchasing replay operations.
pub struct BallchasingReplayRepository;

impl BallchasingReplayRepository {
    /// Creates a new ballchasing replay record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create(input: CreateBallchasingReplay) -> Result<BallchasingReplay, sqlx::Error> {
        let pool = get_pool();
        sqlx::query_as!(
            BallchasingReplay,
            r#"
            INSERT INTO ballchasing_replays (id, rank, metadata)
            VALUES ($1, $2, $3)
            RETURNING id, rank as "rank: BallchasingRank", metadata, download_status as "download_status: DownloadStatus", file_path, error_message, created_at, updated_at
            "#,
            input.id,
            input.rank as BallchasingRank,
            input.metadata
        )
        .fetch_one(pool)
        .await
    }

    /// Creates multiple ballchasing replay records, skipping duplicates.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create_many(inputs: Vec<CreateBallchasingReplay>) -> Result<usize, sqlx::Error> {
        let pool = get_pool();
        let mut created = 0;

        for input in inputs {
            let result = sqlx::query!(
                r#"
                INSERT INTO ballchasing_replays (id, rank, metadata)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO NOTHING
                "#,
                input.id,
                input.rank as BallchasingRank,
                input.metadata
            )
            .execute(pool)
            .await?;

            if result.rows_affected() > 0 {
                created += 1;
            }
        }

        Ok(created)
    }

    /// Checks if a replay exists by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn exists(id: Uuid) -> Result<bool, sqlx::Error> {
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
    pub async fn find_by_id(id: Uuid) -> Result<Option<BallchasingReplay>, sqlx::Error> {
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
    pub async fn list_by_rank(
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
    pub async fn list_pending_downloads(
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
    pub async fn update_status(
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
    pub async fn mark_in_progress(id: Uuid) -> Result<(), sqlx::Error> {
        Self::update_status(id, DownloadStatus::InProgress, None, None).await
    }

    /// Marks a replay as downloaded with its file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn mark_downloaded(id: Uuid, file_path: &str) -> Result<(), sqlx::Error> {
        Self::update_status(id, DownloadStatus::Downloaded, Some(file_path), None).await
    }

    /// Marks a replay as failed with an error message.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn mark_failed(id: Uuid, error_message: &str) -> Result<(), sqlx::Error> {
        Self::update_status(id, DownloadStatus::Failed, None, Some(error_message)).await
    }

    /// Resets failed downloads back to `not_downloaded` status.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn reset_failed() -> Result<u64, sqlx::Error> {
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
    pub async fn reset_in_progress() -> Result<u64, sqlx::Error> {
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
    pub async fn count_by_rank(rank: BallchasingRank) -> Result<i64, sqlx::Error> {
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
    pub async fn count_by_rank_and_status(
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
    pub async fn get_stats() -> Result<Vec<BallchasingRankStats>, sqlx::Error> {
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
}
