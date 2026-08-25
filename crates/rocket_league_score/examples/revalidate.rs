//! Re-score an existing checkpoint without re-training.
//!
//! Runs one validation pass with frozen weights and prints the full diagnostic block,
//! including the whole-match and within-lobby metrics added in row 20 of `experiment.md`.
//! Those metrics were introduced after the `lstm_v20` run finished, so no existing
//! checkpoint has ever been scored with them; this example is how they get read for the
//! first time (step 1 of `docs/smurf-detection-handoff.md`) without paying for a
//! ~6-day re-train.
//!
//! It deliberately calls [`ml_model_training::compute_validation_loss`] — the same
//! function the training loop calls — rather than reimplementing the metrics, so the
//! numbers here cannot drift from the numbers printed during a run.
//!
//! Segments are loaded **cache-only**: replays with no cached segment directory are
//! skipped and counted rather than parsed. Re-scoring is a read-only operation and
//! should not populate the cache as a side effect. The skip count is printed so a
//! thin cache cannot be mistaken for a full evaluation set.
//!
//! Usage:
//!   cargo run --release --example revalidate -- \
//!       --model models/lstm_v20/checkpoint_best --split evaluation
//!
//! Requires `DATABASE_URL`. Does not require the replay object store.

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;
use config::get_base_path;
use database::{initialize_pool, list_replay_players_by_replay, list_replays_by_split};
use feature_extractor::TOTAL_PLAYERS;
use ml_model::SequenceModel;
use ml_model_training::segment_cache::{SegmentStore, SegmentStoreBuilder, segment_directory};
use ml_model_training::{SequenceBatcher, compute_validation_loss, load_checkpoint};
use replay_structs::DatasetSplit;
use std::path::Path;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

// Eval-only path: NdArray on CPU avoids GPU autotune issues (e.g. WSL `dzn` lacking
// subgroup/"plane" instructions) and matches the other eval binaries in this crate.
type InferenceBackend = NdArray;

#[derive(Parser, Debug)]
#[command(name = "revalidate")]
#[command(about = "Re-score an existing checkpoint on a dataset split", long_about = None)]
struct Args {
    /// Path to the model checkpoint (without extension).
    #[arg(short = 'm', long, default_value = "models/lstm_v20/checkpoint_best")]
    model: String,

    /// Dataset split to score ("evaluation" or "training").
    #[arg(long, default_value = "evaluation")]
    split: String,

    /// Sequence length (must match the checkpoint).
    #[arg(long, default_value_t = 300)]
    seq_len: usize,

    /// Validation batch size. Only affects speed, not the metrics.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Cap the number of replays loaded (for a quick smoke run).
    #[arg(long)]
    limit: Option<usize>,
}

fn parse_split(raw: &str) -> Result<DatasetSplit> {
    match raw.to_ascii_lowercase().as_str() {
        "evaluation" | "eval" | "valid" | "validation" => Ok(DatasetSplit::Evaluation),
        "training" | "train" => Ok(DatasetSplit::Training),
        other => anyhow::bail!("unknown split {other:?} (expected \"evaluation\" or \"training\")"),
    }
}

/// True when this replay already has at least one cached `.features` segment.
///
/// Mirrors the cache probe in `full_pipeline`, which is private to that module.
fn segments_cached(base_path: &Path, file_path: &str, replay_id: uuid::Uuid) -> bool {
    let dir = segment_directory(base_path, file_path, replay_id);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry
            .path()
            .extension()
            .is_some_and(|ext| ext == "features")
    })
}

/// Builds the label row for a replay, `0.0` in slots whose rank is unknown.
///
/// Slot order must match the training path exactly (blue then orange, each sorted by
/// player name) or every within-lobby metric would be scored against shuffled labels.
fn target_mmr_from_players(players: &[replay_structs::ReplayPlayer]) -> [f32; TOTAL_PLAYERS] {
    let mut target_mmr = [0.0f32; TOTAL_PLAYERS];

    let mut blue: Vec<_> = players.iter().filter(|p| p.team == 0).collect();
    let mut orange: Vec<_> = players.iter().filter(|p| p.team == 1).collect();

    blue.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    for (i, player) in blue.iter().take(3).enumerate() {
        if player.rank_known
            && let Some(slot) = target_mmr.get_mut(i)
        {
            *slot = player.rank_division.mmr_middle() as f32;
        }
    }
    for (i, player) in orange.iter().take(3).enumerate() {
        if player.rank_known
            && let Some(slot) = target_mmr.get_mut(i + 3)
        {
            *slot = player.rank_division.mmr_middle() as f32;
        }
    }

    target_mmr
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let args = Args::parse();
    let split = parse_split(&args.split)?;

    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;
    initialize_pool(&database_url).await?;

    let mut replays = list_replays_by_split(split).await?;
    if let Some(limit) = args.limit {
        replays.truncate(limit);
    }
    anyhow::ensure!(
        !replays.is_empty(),
        "no downloaded replays found for split {split:?}"
    );
    info!(
        replays = replays.len(),
        split = %args.split,
        "Loading cached segments"
    );

    let base_path = get_base_path();
    let mut builder =
        SegmentStoreBuilder::new(base_path.clone(), "revalidate".to_string(), args.seq_len);

    let mut skipped_uncached = 0usize;
    let mut skipped_no_players = 0usize;

    for replay in &replays {
        let db_players = list_replay_players_by_replay(replay.id).await?;
        if db_players.is_empty() {
            skipped_no_players += 1;
            continue;
        }
        if !segments_cached(&base_path, &replay.file_path, replay.id) {
            skipped_uncached += 1;
            continue;
        }
        builder.add_replay(
            &replay.file_path,
            replay.id,
            target_mmr_from_players(&db_players),
        );
    }

    let dataset: Arc<SegmentStore> = Arc::new(builder.build());
    anyhow::ensure!(
        !dataset.is_empty(),
        "no cached segments found for split {split:?} — run the pipeline first to populate the cache"
    );

    info!(
        segments = dataset.len(),
        replays_used = replays.len() - skipped_uncached - skipped_no_players,
        skipped_uncached,
        skipped_no_players,
        "Segment store ready"
    );

    let device = NdArrayDevice::Cpu;
    info!(model = %args.model, "Loading checkpoint");
    let model: SequenceModel<InferenceBackend> =
        load_checkpoint(&args.model, &device).context("Failed to load model checkpoint")?;

    let batcher = SequenceBatcher::<InferenceBackend>::new(device, args.seq_len);

    info!("Scoring (frozen weights, no optimiser step)...");
    let result = compute_validation_loss(&model, &dataset, &batcher, args.batch_size);

    // compute_validation_loss already logs the per-rank table, collapse diagnostics and
    // the within-lobby block. Print a short summary so a piped run has a final line.
    println!(
        "\nrevalidate: model={} split={} segments={} loss={:.6} pred_std={:.1} MMR pearson_r={:.4}",
        args.model,
        args.split,
        dataset.len(),
        result.loss,
        result.pred_std_mmr,
        result.pearson_r,
    );
    match result.within_lobby.as_ref() {
        Some(m) => println!(
            "within-lobby: within={:.1} MMR between={:.1} MMR concordance={:.3} over {} pairs \
             | mixed n={} top1={:.1}% detect={:.1}% margin={:+.0} MMR",
            m.within_rmse_mmr,
            m.between_rmse_mmr,
            m.concordance,
            m.comparable_pairs,
            m.mixed_lobbies,
            100.0 * m.mixed_top1_rate,
            100.0 * m.mixed_detection_rate,
            m.mixed_mean_margin_mmr,
        ),
        None => println!("within-lobby: unavailable (no lobby had two rank-known players)"),
    }

    Ok(())
}
