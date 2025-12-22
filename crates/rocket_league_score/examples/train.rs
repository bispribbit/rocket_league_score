//! Example: Train the ML model on ingested replays.
//!
//! Usage: cargo run --example train
//!
//! Requires `DATABASE_URL` environment variable to be set.
//! Make sure to run the ingest example first to populate the database.

use anyhow::Result;
use database::create_pool;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

const MODEL_NAME: &str = "impact_model";
const EPOCHS: usize = 100;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 0.0001;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    let pool = create_pool(&database_url).await?;

    commands::train::run(&pool, MODEL_NAME, EPOCHS, BATCH_SIZE, LEARNING_RATE).await?;

    Ok(())
}
