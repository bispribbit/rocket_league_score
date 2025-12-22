//! Example: Run end-to-end pipeline test (no database required).
//!
//! Usage: cargo run --example `test_pipeline`
//!
//! This example does not require a database connection.

use std::path::PathBuf;

use anyhow::Result;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

const REPLAY_DIR: &str = "/workspace/replays/3v3";
const METADATA_PATH: &str = "/workspace/replays/3v3/metadata.jsonl";
const NUM_REPLAYS: usize = 3;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let replay_dir = PathBuf::from(REPLAY_DIR);
    let metadata = PathBuf::from(METADATA_PATH);

    commands::test_pipeline::run(&replay_dir, &metadata, NUM_REPLAYS)?;

    Ok(())
}
