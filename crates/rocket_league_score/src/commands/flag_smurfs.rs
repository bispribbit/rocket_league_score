//! Post-training smurf-detection flagging command.
//!
//! Runs the final trained model over every downloaded replay in the database,
//! computes `predicted_mmr − label_mmr` per player slot, and writes the result
//! to the `replay_players.smurf_score` column.
//!
//! A large positive `smurf_score` (e.g. > 500) indicates that the model thinks the
//! player performed significantly above their rated MMR — a common smurf pattern.
//!
//! # Requirements
//!
//! * `DATABASE_URL` — PostgreSQL connection string
//! * The migration `20260419000001_add_smurf_score_to_replay_players.sql` must have run.
//!
//! # Usage
//!
//! ```ignore
//! cargo run --bin rocket_league_score --release -- flag-smurfs \
//!     --model models/lstm_v13/checkpoint_best
//! ```

use anyhow::{Context, Result};
use burn::backend::Wgpu;
use config::{OBJECT_STORE, get_base_path};
use database::{
    bulk_update_smurf_scores, initialize_pool, list_downloaded_replays,
    list_replay_players_by_replay,
};
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{SequenceModel, predict_player_centric_per_segment};
use ml_model_training::load_checkpoint;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use tracing::{info, warn};

type InferenceBackend = Wgpu;

/// Default sequence length (must match the checkpoint).
const DEFAULT_SEQUENCE_LENGTH: usize = 300;

/// Number of smurf-score updates to accumulate before flushing to the DB.
const DB_FLUSH_INTERVAL: usize = 500;

/// Configuration for the flag-smurfs command.
pub struct FlagSmurfsConfig {
    /// Path to the trained model checkpoint (without extension).
    pub model_path: String,
    /// Sequence length to use during inference. Defaults to [`DEFAULT_SEQUENCE_LENGTH`].
    pub sequence_length: usize,
}

impl Default for FlagSmurfsConfig {
    fn default() -> Self {
        Self {
            model_path: String::from("models/lstm_v13/checkpoint_best"),
            sequence_length: DEFAULT_SEQUENCE_LENGTH,
        }
    }
}

/// Runs the flag-smurfs command.
///
/// # Errors
///
/// Returns an error if DB access, model loading, or replay parsing fails.
pub async fn run(config: &FlagSmurfsConfig) -> Result<()> {
    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;

    initialize_pool(&database_url).await?;

    let device = super::init_device();

    info!(model_path = %config.model_path, "Loading model checkpoint");
    let model: SequenceModel<InferenceBackend> =
        load_checkpoint(&config.model_path, &device).context("Failed to load model checkpoint")?;

    let _base_path = get_base_path();
    let replays = list_downloaded_replays().await?;

    info!(
        total_replays = replays.len(),
        sequence_length = config.sequence_length,
        "Starting smurf score computation"
    );

    let mut pending_updates: Vec<(uuid::Uuid, String, f32)> = Vec::new();
    let mut replays_processed = 0usize;
    let mut replays_failed = 0usize;
    let mut players_scored = 0usize;

    for replay in &replays {
        let object_path = ObjectStorePath::from(replay.file_path.clone());
        let replay_data = match OBJECT_STORE
            .get(&object_path)
            .await
            .context("Failed to read object store")?
            .bytes()
            .await
        {
            Ok(bytes) => bytes,
            Err(error) => {
                warn!(replay_id = %replay.id, %error, "Failed to read replay bytes");
                replays_failed += 1;
                continue;
            }
        };

        let parsed = match parse_replay_from_bytes(&replay_data) {
            Ok(p) => p,
            Err(error) => {
                warn!(replay_id = %replay.id, %error, "Failed to parse replay");
                replays_failed += 1;
                continue;
            }
        };

        if parsed.frames.is_empty() {
            replays_failed += 1;
            continue;
        }

        // Load DB player labels (for label_mmr).
        let db_players = match list_replay_players_by_replay(replay.id).await {
            Ok(p) => p,
            Err(error) => {
                warn!(replay_id = %replay.id, %error, "Failed to load replay players");
                replays_failed += 1;
                continue;
            }
        };

        if db_players.is_empty() {
            continue;
        }

        // Build player-name → label-MMR map.
        let label_mmr_map: std::collections::HashMap<String, f32> = db_players
            .iter()
            .map(|p| (p.player_name.clone(), p.rank_division.mmr_middle() as f32))
            .collect();

        // Run inference (no ratings needed — inference uses default context).
        let per_segment = predict_player_centric_per_segment(
            &model,
            &parsed.frames,
            &device,
            config.sequence_length,
        );

        if per_segment.is_empty() {
            continue;
        }

        // Average predictions across segments.
        let mut sum_preds = [0.0f32; TOTAL_PLAYERS];
        for seg in &per_segment {
            for (i, pred) in seg.player_predictions.iter().enumerate() {
                if let Some(slot) = sum_preds.get_mut(i) {
                    *slot += pred;
                }
            }
        }
        let n_segs = per_segment.len() as f32;
        for slot in &mut sum_preds {
            *slot /= n_segs;
        }

        // Map slot predictions to player names using the frame ordering from the replay.
        // `predict_player_centric_per_segment` preserves the player ordering from the first frame.
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

        for (slot_idx, player_name) in player_names.iter().enumerate().take(TOTAL_PLAYERS) {
            let Some(&predicted_mmr) = sum_preds.get(slot_idx) else {
                continue;
            };
            let Some(&label_mmr) = label_mmr_map.get(player_name) else {
                continue;
            };
            let smurf_score = predicted_mmr - label_mmr;
            pending_updates.push((replay.id, player_name.clone(), smurf_score));
            players_scored += 1;
        }

        replays_processed += 1;

        if pending_updates.len() >= DB_FLUSH_INTERVAL {
            bulk_update_smurf_scores(&pending_updates)
                .await
                .context("Failed to write smurf scores to DB")?;
            info!(
                replays_processed,
                players_flushed = pending_updates.len(),
                "Flushed smurf scores to DB"
            );
            pending_updates.clear();
        }
    }

    // Final flush.
    if !pending_updates.is_empty() {
        bulk_update_smurf_scores(&pending_updates)
            .await
            .context("Failed to write final smurf scores to DB")?;
    }

    info!(
        replays_processed,
        replays_failed, players_scored, "Smurf score computation complete"
    );

    Ok(())
}
