//! Ballchasing.com replay downloader.
//!
//! Downloads Rocket League replays from ballchasing.com organized by rank.

use anyhow::Result;
use ballchasing_downloader::Config;
use database::{create_pool, run_migrations};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file
    dotenvy::dotenv().ok();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    info!("Ballchasing replay downloader starting");

    // Load config
    let config = Config::from_env()?;

    // Connect to database
    let pool = create_pool(&config.database_url).await?;

    // Run migrations
    run_migrations(&pool).await?;
    info!("Database ready");

    // Run the downloader
    ballchasing_downloader::run(&pool, &config).await?;

    Ok(())
}
