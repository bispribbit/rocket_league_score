//! Binary: Compare database player rows to replay-file player-count diagnostics for downloaded replays.
//!
//! Environment variables:
//! - `DATABASE_URL` — PostgreSQL connection string (required)
//! - `REPLAY_BASE_PATH` — Base path for the local object store (optional; platform default if unset)

use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    rocket_league_score::commands::validate_ballchasing_player_counts::run().await
}
