//! Example: Run end-to-end pipeline test (no database required).
//!
//! Usage: cargo run --example `test_pipeline`
//!
//! This example does not require a database connection.

use anyhow::Result;
use database::initialize_pool;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

const NUM_REPLAYS: usize = 10;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    initialize_pool(&database_url).await?;

    commands::pipeline::run(NUM_REPLAYS).await?;

    Ok(())
}
