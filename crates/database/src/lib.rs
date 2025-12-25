//! Database crate for Rocket League impact score calculator.
//!
//! Provides connection pooling, migrations, and repository functions
//! for replays, players, and ML models.

use std::sync::LazyLock;

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub mod ballchasing_replay;
pub mod model;
pub mod models;
pub mod path_utils;
pub mod replay;
pub mod replay_player;

pub use ballchasing_replay::*;
pub use model::*;
pub use models::*;
pub use path_utils::{read_from_object_store, resolve_file_path};
pub use replay::*;
pub use replay_player::*;

/// Internal storage for the pool using `OnceLock` for runtime initialization.
static POOL_INNER: std::sync::OnceLock<PgPool> = std::sync::OnceLock::new();

/// Global database connection pool.
static POOL: LazyLock<&'static PgPool> = LazyLock::new(|| {
    POOL_INNER.get().expect(
        "Database pool not initialized. Call initialize_pool() before using database functions.",
    )
});

/// Initializes the global database connection pool.
///
/// # Errors
///
/// Returns an error if the connection to the database fails.
pub async fn initialize_pool(database_url: &str) -> Result<(), sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    if POOL_INNER.set(pool).is_err() {
        return Err(sqlx::Error::Configuration(
            "Pool already initialized".into(),
        ));
    }

    Ok(())
}

/// Gets a reference to the global database pool.
///
/// # Panics
///
/// Panics if the pool has not been initialized.
#[must_use]
pub fn get_pool() -> &'static PgPool {
    *POOL
}

/// Creates a connection pool to the `PostgreSQL` database.
///
/// # Errors
///
/// Returns an error if the connection to the database fails.
pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
}

/// Runs all pending migrations.
///
/// # Errors
///
/// Returns an error if running migrations fails.
pub async fn run_migrations() -> Result<(), sqlx::migrate::MigrateError> {
    let pool = get_pool();
    sqlx::migrate!("./migrations").run(pool).await
}
