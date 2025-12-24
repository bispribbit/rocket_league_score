//! Example: Run database migrations.
//!
//! Usage: cargo run --example migrate
//!
//! Requires `DATABASE_URL` environment variable to be set.

use anyhow::Result;
use database::{initialize_pool, run_migrations};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    initialize_pool(&database_url).await?;

    run_migrations().await?;
    info!("Migrations completed successfully");

    Ok(())
}
