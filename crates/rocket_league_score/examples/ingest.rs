//! Example: Ingest replays from the 3v3 folder into the database.
//!
//! Usage: cargo run --example ingest
//!
//! Requires `DATABASE_URL` environment variable to be set.

use std::path::PathBuf;

use anyhow::Result;
use database::initialize_pool;
use replay_structs::GameMode;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

const REPLAY_FOLDER: &str = "C:/GitHub/rocket_league_score/workspace/replays/3v3";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    initialize_pool(&database_url).await?;

    let folder = PathBuf::from(REPLAY_FOLDER);

    commands::ingest::run(&folder, GameMode::Soccar3v3, None).await?;

    Ok(())
}
