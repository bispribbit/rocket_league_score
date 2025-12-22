//! Example: Predict impact scores for a replay.
//!
//! Usage: cargo run --example predict
//!
//! Requires `DATABASE_URL` environment variable to be set.
//! Make sure to run the train example first to create a model.

use std::path::PathBuf;

use anyhow::Result;
use database::create_pool;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

const REPLAY_FOLDER: &str = "/workspace/replays/3v3";
const MODEL_NAME: &str = "impact_model";

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    let pool = create_pool(&database_url).await?;

    // Find the first replay in the folder
    let replay_folder = PathBuf::from(REPLAY_FOLDER);
    let replay_path = find_first_replay(&replay_folder)?;

    tracing::info!(replay = %replay_path.display(), "Using replay");

    commands::predict::run(&pool, &replay_path, MODEL_NAME, None).await?;

    Ok(())
}

/// Finds the first .replay file in the given directory.
fn find_first_replay(folder: &std::path::Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && let Some(ext) = path.extension()
            && ext == "replay"
        {
            return Ok(path);
        }
    }

    anyhow::bail!("No .replay files found in {}", folder.display())
}

