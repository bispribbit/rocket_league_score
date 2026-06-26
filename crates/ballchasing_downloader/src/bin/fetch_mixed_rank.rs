//! Fetches mid-ladder, big-delta (party) lobbies for smurf-detection training.
//!
//! Scans the ballchasing list endpoint for ranked-standard lobbies where the top
//! rank-known player sits well above the lobby median, inserts their metadata +
//! players, then downloads the replay files.
//!
//! Usage (defaults shown):
//!   cargo run --release --bin fetch_mixed_rank -- \
//!       --target 2500 --min-gap 150 --min-rank silver-1 --max-rank champion-3 \
//!       --max-scan 400000
//!   # discovery only, no file downloads:
//!   cargo run --release --bin fetch_mixed_rank -- --no-download
//!   # resume interrupted downloads without re-scanning the API:
//!   cargo run --release --bin fetch_mixed_rank -- --download-only --min-gap 150
//!
//! Requires `BALLCHASING_API_KEY` and `DATABASE_URL` (read by the shared config).

use anyhow::Result;
use ballchasing_downloader::{MixedRankFetchConfig, run_mixed_rank_fetch};
use config::CONFIG;
use database::{initialize_pool, run_migrations};
use tracing::info;
use tracing_subscriber::EnvFilter;

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let args: Vec<String> = std::env::args().collect();

    let config = MixedRankFetchConfig {
        min_rank: arg_value(&args, "--min-rank").unwrap_or_else(|| "gold-1".to_string()),
        max_rank: arg_value(&args, "--max-rank").unwrap_or_else(|| "champion-3".to_string()),
        min_top_gap_mmr: arg_value(&args, "--min-gap")
            .and_then(|v| v.parse().ok())
            .unwrap_or(150.0),
        target_new_replays: arg_value(&args, "--target")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2500),
        max_summaries_to_scan: arg_value(&args, "--max-scan")
            .and_then(|v| v.parse().ok())
            .unwrap_or(400_000),
        do_download: !args.iter().any(|a| a == "--no-download"),
        download_only: args.iter().any(|a| a == "--download-only"),
    };

    info!(
        min_rank = %config.min_rank,
        max_rank = %config.max_rank,
        min_gap = config.min_top_gap_mmr,
        target = config.target_new_replays,
        download = config.do_download,
        "fetch_mixed_rank starting"
    );

    initialize_pool(&CONFIG.database_url).await?;
    run_migrations().await?;

    run_mixed_rank_fetch(&config).await?;

    info!("fetch_mixed_rank complete");
    Ok(())
}
