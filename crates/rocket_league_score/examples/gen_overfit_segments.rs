//! Generate v6 segment cache for a small per-rank sample used by `overfit_test`.
//!
//! Queries the database for up to `--n-per-rank` downloaded replays per rank,
//! parses each one, and writes `.features` files to the same path the
//! `overfit_test` binary reads from.
//!
//! Usage:
//!   cargo run --example gen_overfit_segments --release -- [--n-per-rank 2]
//!
//! Environment Variables:
//!   DATABASE_URL  - PostgreSQL connection string (required)

use anyhow::{Context, Result};
use config::{OBJECT_STORE, get_base_path};
use database::initialize_pool;
use feature_extractor::{PlayerRating, extract_player_centric_game_sequence};
use ml_model_training::segment_cache::SegmentStoreBuilder;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::Rank;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let n_per_rank: usize = args
        .windows(2)
        .find(|w| w[0] == "--n-per-rank")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(3);

    let sequence_length: usize = 300; // matches TrainingConfig sequence_length

    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    let base_path = get_base_path();

    // All 3v3 ranked standard ranks, in ascending MMR order.
    let all_ranks = [
        Rank::Bronze1,
        Rank::Bronze2,
        Rank::Bronze3,
        Rank::Silver1,
        Rank::Silver2,
        Rank::Silver3,
        Rank::Gold1,
        Rank::Gold2,
        Rank::Gold3,
        Rank::Platinum1,
        Rank::Platinum2,
        Rank::Platinum3,
        Rank::Diamond1,
        Rank::Diamond2,
        Rank::Diamond3,
        Rank::Champion1,
        Rank::Champion2,
        Rank::Champion3,
        Rank::GrandChampion1,
        Rank::GrandChampion2,
        Rank::GrandChampion3,
        Rank::SupersonicLegend,
    ];

    let mut total_cached = 0usize;
    let mut total_skipped = 0usize;
    let mut total_errors = 0usize;

    for rank in all_ranks {
        let replays = database::list_replays_by_rank(rank)
            .await
            .with_context(|| format!("DB query failed for {rank:?}"))?;

        let sample: Vec<_> = replays.into_iter().take(n_per_rank).collect();

        if sample.is_empty() {
            warn!("No replays found for {rank:?}, skipping");
            continue;
        }

        info!("Processing {} replays for {rank:?}", sample.len());

        for replay in &sample {
            let db_players =
                database::list_replay_players_by_replay(replay.id).await?;

            if db_players.is_empty() {
                warn!(replay_id = %replay.id, "No players in DB, skipping");
                total_skipped += 1;
                continue;
            }

            // Build a thin SegmentStoreBuilder just to write the .features files.
            let mut builder = SegmentStoreBuilder::new(
                base_path.clone(),
                "overfit_gen".to_string(),
                sequence_length,
            );

            // Skip if already cached.
            {
                use ml_model_training::segment_cache::segment_directory;
                let seg_dir =
                    segment_directory(&base_path, &replay.file_path, replay.id);
                if seg_dir.exists()
                    && std::fs::read_dir(&seg_dir)
                        .map(|mut d| d.next().is_some())
                        .unwrap_or(false)
                {
                    info!(replay_id = %replay.id, "Already cached, skipping");
                    total_cached += 1;
                    builder.add_replay(&replay.file_path, replay.id, [0.0; 6]);
                    continue;
                }
            }

            // Read replay bytes from object store.
            let object_path = ObjectStorePath::from(replay.file_path.clone());
            let replay_data = match OBJECT_STORE
                .get(&object_path)
                .await
                .context("object store get")?
                .bytes()
                .await
            {
                Ok(b) => b,
                Err(e) => {
                    error!(replay_id = %replay.id, error = %e, "Failed to read bytes");
                    total_errors += 1;
                    continue;
                }
            };

            let parsed = match parse_replay_from_bytes(&replay_data) {
                Ok(p) => p,
                Err(e) => {
                    warn!(replay_id = %replay.id, error = %e, "Parse failed");
                    total_errors += 1;
                    continue;
                }
            };

            if parsed.frames.is_empty() {
                warn!(replay_id = %replay.id, "No frames");
                total_errors += 1;
                continue;
            }

            let player_ratings: Vec<PlayerRating> = db_players
                .iter()
                .map(|p| PlayerRating {
                    player_name: p.player_name.clone(),
                    team: p.team,
                    mmr: p.rank_division.mmr_middle(),
                })
                .collect();

            let game_sequence = extract_player_centric_game_sequence(
                &parsed,
                &player_ratings,
                sequence_length,
            );

            if let Err(e) = builder.ensure_player_centric_segments_cached(
                &replay.file_path,
                replay.id,
                Some(&game_sequence.player_frames),
                Some(game_sequence.player_frames.len()),
            ) {
                error!(replay_id = %replay.id, error = %e, "Cache write failed");
                total_errors += 1;
                continue;
            }

            total_cached += 1;
            info!(replay_id = %replay.id, rank = ?rank, "Cached");
        }
    }

    info!(
        total_cached,
        total_skipped,
        total_errors,
        "Done — segment cache generation complete"
    );
    Ok(())
}
