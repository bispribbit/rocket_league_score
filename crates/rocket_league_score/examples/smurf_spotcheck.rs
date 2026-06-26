#![allow(clippy::indexing_slicing)]

//! Coherent smurf-detection spot-check.
//!
//! Runs a trained model over REAL mixed-rank lobbies (whole replays where one
//! rank-known player sits well above their lobbymates) and reports how often the
//! model flags that known outlier using the production rule from `is_this_a_smurf`
//! (`player_pred > lobby_median(preds) + 200`).
//!
//! Unlike `mixed_rank_eval`, this never stitches players from different games:
//! every lobby is a single real match, so the ball/score/opponent context embedded
//! in each player's features is physically consistent.
//!
//! Usage:
//!   cargo run --release --example smurf_spotcheck -- \
//!       --model models/lstm_v20/checkpoint_best --min-top-gap 300
//!
//! Requires `DATABASE_URL` and access to the replay object store.

use std::collections::HashMap;

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;
use config::OBJECT_STORE;
use database::{initialize_pool, list_mixed_rank_replays, list_replay_players_by_replay};
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{SequenceModel, predict_player_centric_per_segment};
use ml_model_training::load_checkpoint;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

// Eval-only path: NdArray on CPU avoids GPU autotune issues (e.g. WSL `dzn`
// lacking subgroup/"plane" instructions) and is plenty fast for ~100 lobbies.
type InferenceBackend = NdArray;

/// MMR a player's prediction must exceed the lobby median by to be flagged
/// (mirrors `is_this_a_smurf::SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN`).
const SMURF_THRESHOLD_MMR: f32 = 200.0;

#[derive(Parser, Debug)]
#[command(name = "smurf_spotcheck")]
#[command(about = "Spot-check smurf detection on real mixed-rank lobbies", long_about = None)]
struct Args {
    /// Path to the model checkpoint (without extension).
    #[arg(short = 'm', long, default_value = "models/lstm_v20/checkpoint_best")]
    model: String,

    /// Minimum top-gap (MMR above lobby median of the known outlier) to include a lobby.
    #[arg(long, default_value_t = 300.0)]
    min_top_gap: f64,

    /// Sequence length (must match the checkpoint).
    #[arg(long, default_value_t = 300)]
    seq_len: usize,

    /// Only evaluate this split ("training" or "evaluation"). Default: all.
    #[arg(long)]
    split: Option<String>,

    /// Cap the number of lobbies evaluated.
    #[arg(long)]
    limit: Option<usize>,
}

/// Running tally for one cohort (overall or a single split).
#[derive(Default)]
struct Tally {
    lobbies: usize,
    detected: usize,
    top1: usize,
    margin_sum: f64,
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f32::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let args = Args::parse();

    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;
    initialize_pool(&database_url).await?;

    let device = NdArrayDevice::Cpu;
    info!(model = %args.model, "Loading model checkpoint");
    let model: SequenceModel<InferenceBackend> =
        load_checkpoint(&args.model, &device).context("Failed to load model checkpoint")?;

    let mut candidates = list_mixed_rank_replays(args.min_top_gap).await?;
    if let Some(split) = args.split.as_deref() {
        candidates.retain(|c| {
            c.dataset_split
                .is_some_and(|s| format!("{s:?}").to_lowercase().contains(split))
        });
    }
    if let Some(limit) = args.limit {
        candidates.truncate(limit);
    }

    info!(
        candidates = candidates.len(),
        min_top_gap = args.min_top_gap,
        threshold_mmr = SMURF_THRESHOLD_MMR,
        "Evaluating real mixed-rank lobbies"
    );
    println!(
        "\n{:<10} {:>11} {:>9} {:>9} {:>11} {:>9} {:>6} {:>5}",
        "replay", "split", "high_lbl", "high_prd", "lobby_med", "margin", "flag", "top1"
    );

    let mut overall = Tally::default();
    let mut by_split: HashMap<String, Tally> = HashMap::new();
    let mut skipped = 0usize;

    for candidate in &candidates {
        let object_path = ObjectStorePath::from(candidate.file_path.clone());
        let bytes = match OBJECT_STORE.get(&object_path).await {
            Ok(result) => match result.bytes().await {
                Ok(b) => b,
                Err(error) => {
                    warn!(replay_id = %candidate.id, %error, "Failed to read replay bytes");
                    skipped += 1;
                    continue;
                }
            },
            Err(error) => {
                warn!(replay_id = %candidate.id, %error, "Failed to get replay from object store");
                skipped += 1;
                continue;
            }
        };

        let parsed = match parse_replay_from_bytes(&bytes) {
            Ok(p) if !p.frames.is_empty() => p,
            _ => {
                skipped += 1;
                continue;
            }
        };

        let per_segment =
            predict_player_centric_per_segment(&model, &parsed.frames, &device, args.seq_len);
        if per_segment.is_empty() {
            skipped += 1;
            continue;
        }

        // Per-slot prediction = median across segments (matches is_this_a_smurf aggregation).
        let mut slot_preds = [0.0f32; TOTAL_PLAYERS];
        for (slot, slot_pred) in slot_preds.iter_mut().enumerate() {
            let mut across_segments: Vec<f32> = per_segment
                .iter()
                .map(|seg| seg.player_predictions[slot])
                .collect();
            *slot_pred = median(&mut across_segments);
        }

        let player_names: Vec<String> = parsed
            .frames
            .first()
            .map(|frame| {
                frame
                    .players
                    .iter()
                    .map(|p| p.name.as_ref().clone())
                    .collect()
            })
            .unwrap_or_default();

        let db_players = list_replay_players_by_replay(candidate.id).await?;

        // Ground-truth outlier = rank-known player with the highest label MMR.
        let Some(high_player) = db_players
            .iter()
            .filter(|p| p.rank_known)
            .max_by_key(|p| p.rank_division.mmr_middle())
        else {
            skipped += 1;
            continue;
        };
        let high_label = high_player.rank_division.mmr_middle() as f32;

        let Some(high_slot) = player_names
            .iter()
            .position(|name| name == &high_player.player_name)
        else {
            // Name mismatch between parsed frames and DB roster — cannot locate the outlier.
            skipped += 1;
            continue;
        };

        let mut lobby_vals: Vec<f32> = slot_preds.to_vec();
        let lobby_median = median(&mut lobby_vals);
        let high_pred = slot_preds[high_slot];
        let margin = high_pred - lobby_median;
        let detected = high_pred > lobby_median + SMURF_THRESHOLD_MMR;

        let argmax_slot = slot_preds
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(high_slot, |(idx, _)| idx);
        let top1 = argmax_slot == high_slot;

        let split_label = candidate
            .dataset_split
            .map_or_else(|| "none".to_string(), |s| format!("{s:?}"));

        let short_id: String = candidate.id.to_string().chars().take(8).collect();
        println!(
            "{:<10} {:>11} {:>9.0} {:>9.0} {:>11.0} {:>9.0} {:>6} {:>5}",
            short_id,
            split_label,
            high_label,
            high_pred,
            lobby_median,
            margin,
            if detected { "YES" } else { "no" },
            if top1 { "YES" } else { "no" },
        );

        for tally in [&mut overall, by_split.entry(split_label).or_default()] {
            tally.lobbies += 1;
            tally.detected += usize::from(detected);
            tally.top1 += usize::from(top1);
            tally.margin_sum += margin as f64;
        }
    }

    print_summary("OVERALL", &overall);
    let mut splits: Vec<&String> = by_split.keys().collect();
    splits.sort();
    for split in splits {
        if let Some(tally) = by_split.get(split) {
            print_summary(split, tally);
        }
    }
    if skipped > 0 {
        println!("\n(skipped {skipped} candidate(s): unreadable, unparsable, or name mismatch)");
    }

    Ok(())
}

fn print_summary(label: &str, tally: &Tally) {
    if tally.lobbies == 0 {
        println!("\n[{label}] no lobbies evaluated");
        return;
    }
    let n = tally.lobbies as f64;
    println!(
        "\n[{label}] lobbies={}  detection_rate={:.1}%  top1_rate={:.1}%  mean_margin={:.0} MMR",
        tally.lobbies,
        100.0 * tally.detected as f64 / n,
        100.0 * tally.top1 as f64 / n,
        tally.margin_sum / n,
    );
}
