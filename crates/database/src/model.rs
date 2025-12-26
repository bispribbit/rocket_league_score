//! Repository functions for model operations.

use replay_structs::Model;
use uuid::Uuid;

use crate::get_pool;

/// Creates a new model record.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn insert_model(
    name: &str,
    version: i32,
    checkpoint_path: &str,
    training_config: Option<serde_json::Value>,
    metrics: Option<serde_json::Value>,
) -> Result<Model, sqlx::Error> {
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
        name,
        version,
        checkpoint_path,
        training_config,
        metrics
    )
    .fetch_one(pool)
    .await
}

/// Finds a model by name and version.
///
/// # Errors
///
/// Returns an error if the database operation fails.
pub async fn find_model_by_name_version(
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
pub async fn find_latest_model(name: &str) -> Result<Option<Model>, sqlx::Error> {
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
pub async fn get_next_model_version(name: &str) -> Result<i32, sqlx::Error> {
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
pub async fn list_model_versions(name: &str) -> Result<Vec<Model>, sqlx::Error> {
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
