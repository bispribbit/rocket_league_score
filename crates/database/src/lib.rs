use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

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
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::migrate::MigrateError> {
    sqlx::migrate!("./migrations").run(pool).await
}
