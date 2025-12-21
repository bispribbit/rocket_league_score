//! Repository functions for database operations.

use sqlx::PgPool;
use uuid::Uuid;

use crate::models::{
    CreateModel, CreateReplay, CreateReplayPlayer, GameMode, Model, Replay, ReplayPlayer,
};

/// Repository for replay operations.
pub struct ReplayRepository;

impl ReplayRepository {
    /// Creates a new replay record.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create(pool: &PgPool, input: CreateReplay) -> Result<Replay, sqlx::Error> {
        let id = Uuid::new_v4();

        sqlx::query_as::<_, Replay>(
            "
            INSERT INTO replays (id, file_path, game_mode)
            VALUES ($1, $2, $3)
            RETURNING id, file_path, game_mode, processed_at, created_at
            ",
        )
        .bind(id)
        .bind(&input.file_path)
        .bind(input.game_mode)
        .fetch_one(pool)
        .await
    }

    /// Finds a replay by its file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_by_path(pool: &PgPool, file_path: &str) -> Result<Option<Replay>, sqlx::Error> {
        sqlx::query_as::<_, Replay>(
            "
            SELECT id, file_path, game_mode, processed_at, created_at
            FROM replays
            WHERE file_path = $1
            ",
        )
        .bind(file_path)
        .fetch_optional(pool)
        .await
    }

    /// Finds a replay by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_by_id(pool: &PgPool, id: Uuid) -> Result<Option<Replay>, sqlx::Error> {
        sqlx::query_as::<_, Replay>(
            "
            SELECT id, file_path, game_mode, processed_at, created_at
            FROM replays
            WHERE id = $1
            ",
        )
        .bind(id)
        .fetch_optional(pool)
        .await
    }

    /// Lists all replays for a given game mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn list_by_game_mode(
        pool: &PgPool,
        game_mode: GameMode,
    ) -> Result<Vec<Replay>, sqlx::Error> {
        sqlx::query_as::<_, Replay>(
            "
            SELECT id, file_path, game_mode, processed_at, created_at
            FROM replays
            WHERE game_mode = $1
            ORDER BY created_at DESC
            ",
        )
        .bind(game_mode)
        .fetch_all(pool)
        .await
    }

    /// Marks a replay as processed.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn mark_processed(pool: &PgPool, id: Uuid) -> Result<(), sqlx::Error> {
        sqlx::query(
            "
            UPDATE replays
            SET processed_at = NOW()
            WHERE id = $1
            ",
        )
        .bind(id)
        .execute(pool)
        .await?;

        Ok(())
    }

    /// Counts replays by game mode.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn count_by_game_mode(pool: &PgPool, game_mode: GameMode) -> Result<i64, sqlx::Error> {
        let result: (i64,) = sqlx::query_as(
            "
            SELECT COUNT(*) FROM replays WHERE game_mode = $1
            ",
        )
        .bind(game_mode)
        .fetch_one(pool)
        .await?;

        Ok(result.0)
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
    pub async fn create(pool: &PgPool, input: CreateReplayPlayer) -> Result<ReplayPlayer, sqlx::Error> {
        let id = Uuid::new_v4();

        sqlx::query_as::<_, ReplayPlayer>(
            "
            INSERT INTO replay_players (id, replay_id, player_name, team, skill_rating)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, replay_id, player_name, team, skill_rating, created_at
            ",
        )
        .bind(id)
        .bind(input.replay_id)
        .bind(&input.player_name)
        .bind(input.team)
        .bind(input.skill_rating)
        .fetch_one(pool)
        .await
    }

    /// Creates multiple player records for a replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn create_many(
        pool: &PgPool,
        inputs: Vec<CreateReplayPlayer>,
    ) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
        let mut players = Vec::with_capacity(inputs.len());

        for input in inputs {
            let player = Self::create(pool, input).await?;
            players.push(player);
        }

        Ok(players)
    }

    /// Lists all players for a given replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn list_by_replay(pool: &PgPool, replay_id: Uuid) -> Result<Vec<ReplayPlayer>, sqlx::Error> {
        sqlx::query_as::<_, ReplayPlayer>(
            "
            SELECT id, replay_id, player_name, team, skill_rating, created_at
            FROM replay_players
            WHERE replay_id = $1
            ORDER BY team, player_name
            ",
        )
        .bind(replay_id)
        .fetch_all(pool)
        .await
    }

    /// Gets the average skill rating for a replay.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn average_rating_for_replay(pool: &PgPool, replay_id: Uuid) -> Result<Option<f64>, sqlx::Error> {
        let result: (Option<f64>,) = sqlx::query_as(
            "
            SELECT AVG(skill_rating::float) FROM replay_players WHERE replay_id = $1
            ",
        )
        .bind(replay_id)
        .fetch_one(pool)
        .await?;

        Ok(result.0)
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
    pub async fn create(pool: &PgPool, input: CreateModel) -> Result<Model, sqlx::Error> {
        let id = Uuid::new_v4();

        sqlx::query_as::<_, Model>(
            "
            INSERT INTO models (id, name, version, checkpoint_path, training_config, metrics)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, name, version, checkpoint_path, training_config, metrics, trained_at
            ",
        )
        .bind(id)
        .bind(&input.name)
        .bind(input.version)
        .bind(&input.checkpoint_path)
        .bind(&input.training_config)
        .bind(&input.metrics)
        .fetch_one(pool)
        .await
    }

    /// Finds a model by name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_by_name_version(
        pool: &PgPool,
        name: &str,
        version: i32,
    ) -> Result<Option<Model>, sqlx::Error> {
        sqlx::query_as::<_, Model>(
            "
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1 AND version = $2
            ",
        )
        .bind(name)
        .bind(version)
        .fetch_optional(pool)
        .await
    }

    /// Gets the latest version of a model by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn find_latest(pool: &PgPool, name: &str) -> Result<Option<Model>, sqlx::Error> {
        sqlx::query_as::<_, Model>(
            "
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1
            ORDER BY version DESC
            LIMIT 1
            ",
        )
        .bind(name)
        .fetch_optional(pool)
        .await
    }

    /// Gets the next version number for a model name.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn next_version(pool: &PgPool, name: &str) -> Result<i32, sqlx::Error> {
        let result: (Option<i32>,) = sqlx::query_as(
            "
            SELECT MAX(version) FROM models WHERE name = $1
            ",
        )
        .bind(name)
        .fetch_one(pool)
        .await?;

        Ok(result.0.unwrap_or(0) + 1)
    }

    /// Lists all versions of a model.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn list_versions(pool: &PgPool, name: &str) -> Result<Vec<Model>, sqlx::Error> {
        sqlx::query_as::<_, Model>(
            "
            SELECT id, name, version, checkpoint_path, training_config, metrics, trained_at
            FROM models
            WHERE name = $1
            ORDER BY version DESC
            ",
        )
        .bind(name)
        .fetch_all(pool)
        .await
    }
}

