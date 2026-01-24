//! Full training pipeline command with checkpoint support and database tracking.
//!
//! This command implements a production-ready training pipeline that:
//! - Tracks training/evaluation split in the database
//! - Supports resume from checkpoints
//! - Saves checkpoints every N epochs
//! - Validates during training
//!
//! # Example
//!
//! ```ignore
//! // Train from scratch with 90% training split
//! full_train::run("lstm_v1", 0.9, 100, 32, 0.001, false).await?;
//!
//! // Resume training from last checkpoint
//! full_train::run("lstm_v1", 0.9, 100, 32, 0.001, true).await?;
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Cuda};
use config::{OBJECT_STORE, get_base_path};
use feature_extractor::{PlayerRating, TOTAL_PLAYERS, extract_player_centric_game_sequence};
use ml_model::segment_cache::{SegmentStore, SegmentStoreBuilder};
use ml_model::{
    CheckpointConfig, ModelConfig, TrainingConfig, TrainingState, create_model, load_checkpoint,
    save_checkpoint, train,
};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, Replay};
use tracing::{debug, error, info, warn};

use super::init_device;

/// Training requires Autodiff wrapper for automatic differentiation.
type TrainBackend = Autodiff<Cuda>;

/// Configuration for the full training pipeline.
#[derive(Debug, Clone)]
pub struct FullTrainConfig {
    /// Model name for saving/loading.
    pub model_name: String,
    /// Ratio of data to use for training (e.g., 0.9 for 90%).
    pub train_ratio: f64,
    /// Number of epochs to train.
    pub epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Learning rate for optimizer.
    pub learning_rate: f64,
    /// Whether to resume from last checkpoint.
    pub resume: bool,
    /// Save checkpoint every N epochs (0 to disable).
    pub checkpoint_every_n_epochs: usize,
    /// Maximum number of replays to use (None = use all available).
    /// When set, replays are sampled across ranks for balanced distribution.
    pub max_replays: Option<usize>,
}

impl Default for FullTrainConfig {
    fn default() -> Self {
        Self {
            model_name: String::from("lstm_v6"),
            train_ratio: 0.9,
            epochs: 100,
            batch_size: 128,
            learning_rate: 0.001,
            resume: false,
            checkpoint_every_n_epochs: 5,
            max_replays: None,
        }
    }
}

/// Runs the full training pipeline.
///
/// This is the main entry point for production training. It:
/// 1. Assigns dataset splits to unassigned replays
/// 2. Loads training data from the training split
/// 3. Creates or loads model (if resuming)
/// 4. Trains with checkpoint saving
/// 5. Saves final model to database
///
/// # Arguments
///
/// * `max_replays` - Optional limit on number of replays. When set, replays are
///   sampled across ranks for balanced testing. When None, uses all available replays.
///
/// # Errors
///
/// Returns an error if training fails.
pub async fn run(
    model_name: &str,
    train_ratio: f64,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    resume: bool,
    max_replays: Option<usize>,
) -> anyhow::Result<()> {
    let config = FullTrainConfig {
        model_name: model_name.to_string(),
        train_ratio,
        epochs,
        batch_size,
        learning_rate,
        resume,
        checkpoint_every_n_epochs: 5,
        max_replays,
    };

    Box::pin(run_with_config(&config)).await?;

    Ok(())
}

/// Runs the full training pipeline with a config struct.
///
/// Uses lazy loading to avoid loading all replay data into memory at once.
///
/// # Errors
///
/// Returns an error if training fails.
pub async fn run_with_config(config: &FullTrainConfig) -> Result<()> {
    let pipeline_start = Instant::now();
    info!(
        model_name = %config.model_name,
        train_ratio = config.train_ratio,
        epochs = config.epochs,
        resume = config.resume,
        "Starting full training pipeline (lazy loading mode)"
    );

    let model_config = ModelConfig::new();
    let training_config = TrainingConfig::new(model_config.clone())
        .with_learning_rate(config.learning_rate)
        .with_epochs(config.epochs)
        .with_batch_size(config.batch_size);

    // Step 1: Assign dataset splits to unassigned replays (skip when using max_replays limit)
    let step1_start = Instant::now();
    let current_counts = if config.max_replays.is_some() {
        info!("Step 1: Using sampled replays (skipping dataset split assignment)...");
        // When testing with limited replays, we skip split assignment
        database::DatasetSplitCounts {
            training: 0,
            evaluation: 0,
        }
    } else {
        info!("Step 1: Checking and assigning dataset splits...");
        let split_counts = database::assign_dataset_splits(config.train_ratio).await?;

        if split_counts.training > 0 || split_counts.evaluation > 0 {
            info!(
                newly_assigned_training = split_counts.training,
                newly_assigned_evaluation = split_counts.evaluation,
                "Assigned new replays to dataset splits"
            );
        }

        // Get current split counts
        let counts = database::count_replays_by_split().await?;
        info!(
            training_replays = counts.training,
            evaluation_replays = counts.evaluation,
            "Current dataset split sizes"
        );

        if counts.training == 0 {
            anyhow::bail!(
                "No training replays available. Please ingest and download replays first."
            );
        }
        counts
    };
    let step1_duration = step1_start.elapsed();
    info!(duration_ms = step1_duration.as_millis(), "Step 1 complete");

    // Step 2: Load training data using segment cache
    let step2_start = Instant::now();
    info!("Step 2: Loading training data (using segment cache)...");

    // Load replays based on configuration
    let training_replays = if let Some(max_replays) = config.max_replays {
        info!(max_replays, "Sampling replays across ranks for testing");
        database::list_replays_sampled(max_replays).await?
    } else {
        database::list_replays_by_split(DatasetSplit::Training).await?
    };

    if training_replays.is_empty() {
        anyhow::bail!("No valid replays found. Please ingest and download replays first.");
    }

    info!(
        replay_count = training_replays.len(),
        "Loading training replays (cache-first)..."
    );

    // Get base path for segment cache
    let base_path = get_base_path();

    // Load training data using segment cache (zero-copy)
    let train_dataset: Arc<SegmentStore> = load_training_data_cached(
        &training_replays,
        training_config.sequence_length,
        base_path.clone(),
    )
    .await?;

    let step2_duration = step2_start.elapsed();

    if train_dataset.is_empty() {
        anyhow::bail!("No valid training segments found.");
    }

    info!(
        segments_loaded = train_dataset.len(),
        duration_ms = step2_duration.as_millis(),
        duration_sec = step2_duration.as_secs_f64(),
        "Training data loaded via segment cache"
    );

    // Load validation data if not using max_replays mode
    let valid_dataset = if config.max_replays.is_none() {
        let valid_replays = database::list_replays_by_split(DatasetSplit::Evaluation).await?;
        if !valid_replays.is_empty() {
            info!(
                replay_count = valid_replays.len(),
                "Loading validation replays..."
            );
            let valid_ds: Arc<SegmentStore> = load_validation_data_cached(
                &valid_replays,
                training_config.sequence_length,
                base_path,
            )
            .await?;
            info!(segments_loaded = valid_ds.len(), "Validation data loaded");
            Some(valid_ds)
        } else {
            None
        }
    } else {
        None
    };

    // Step 3: Create or load model
    let step3_start = Instant::now();
    info!("Step 3: Initializing model...");
    let device = init_device();

    let checkpoint_dir = format!("models/{}", config.model_name);
    let checkpoint_prefix = format!("{checkpoint_dir}/checkpoint");

    let (mut model, start_state) = if config.resume {
        // Try to load from latest checkpoint
        let latest_checkpoint = find_latest_checkpoint(&checkpoint_prefix)?;
        if let Some(checkpoint_path) = latest_checkpoint {
            info!(checkpoint = %checkpoint_path, "Resuming from checkpoint");
            let model: ml_model::SequenceModel<TrainBackend> =
                load_checkpoint(&checkpoint_path, &device)?;

            // Extract epoch from checkpoint path
            let start_epoch = extract_epoch_from_checkpoint(&checkpoint_path).unwrap_or(0);
            let state = TrainingState::new(start_epoch);

            (model, Some(state))
        } else {
            warn!("No checkpoint found, starting fresh");
            (create_model::<TrainBackend>(&device, &model_config), None)
        }
    } else {
        info!("Creating new model");
        (create_model::<TrainBackend>(&device, &model_config), None)
    };
    let step3_duration = step3_start.elapsed();

    info!(
        lstm_hidden_1 = model_config.lstm_hidden_1,
        lstm_hidden_2 = model_config.lstm_hidden_2,
        feedforward_hidden = model_config.feedforward_hidden,
        dropout = model_config.dropout,
        duration_ms = step3_duration.as_millis(),
        "Model architecture"
    );

    // Step 4: Train model using memory-mapped segments
    let step4_start = Instant::now();
    info!("Step 4: Starting training with mmap segments...");

    let checkpoint_config = CheckpointConfig {
        path_prefix: checkpoint_prefix.clone(),
        save_every_n_epochs: config.checkpoint_every_n_epochs,
        save_on_improvement: true,
    };

    let output = train(
        &mut model,
        train_dataset.clone(),
        valid_dataset.as_ref(),
        &training_config,
        Some(checkpoint_config),
        start_state,
    )?;

    let train_rmse = output.final_train_loss.sqrt();
    let valid_rmse = output.final_valid_loss.map(f32::sqrt);
    let step4_duration = step4_start.elapsed();

    info!(
        final_train_loss = output.final_train_loss,
        train_rmse,
        final_valid_loss = output.final_valid_loss,
        valid_rmse,
        epochs_completed = output.epochs_completed,
        checkpoints_saved = output.checkpoint_paths.len(),
        duration_ms = step4_duration.as_millis(),
        duration_sec = step4_duration.as_secs_f64(),
        avg_epoch_ms = if output.epochs_completed > 0 {
            step4_duration.as_millis() / output.epochs_completed as u128
        } else {
            0
        },
        "Training completed"
    );

    // Step 5: Save final model to database
    let step5_start = Instant::now();
    info!("Step 5: Saving model to database...");
    let next_version = database::get_next_model_version(&config.model_name).await?;
    let final_checkpoint_path = format!("{checkpoint_dir}/final_v{next_version}");

    save_checkpoint(&model, &final_checkpoint_path, &training_config)?;

    let training_config_json = serde_json::json!({
        "model_type": "lstm_sequence",
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "sequence_length": training_config.sequence_length,
        "lstm_hidden_1": model_config.lstm_hidden_1,
        "lstm_hidden_2": model_config.lstm_hidden_2,
        "feedforward_hidden": model_config.feedforward_hidden,
        "dropout": model_config.dropout,
        "train_ratio": config.train_ratio,
        "training_segments": train_dataset.len(),
        "validation_segments": valid_dataset.as_ref().map_or(0, |ds: &Arc<SegmentStore>| ds.len()),
        "evaluation_games": current_counts.evaluation,
    });

    let training_metrics_json = serde_json::json!({
        "final_train_loss": output.final_train_loss,
        "final_train_rmse": train_rmse,
        "final_valid_loss": output.final_valid_loss,
        "final_valid_rmse": valid_rmse,
        "epochs_completed": output.epochs_completed,
        "checkpoints": output.checkpoint_paths,
    });

    database::insert_model(
        &config.model_name,
        next_version,
        &final_checkpoint_path,
        Some(training_config_json),
        Some(training_metrics_json),
    )
    .await?;

    let step5_duration = step5_start.elapsed();
    info!(
        model_name = %config.model_name,
        version = next_version,
        checkpoint_path = %final_checkpoint_path,
        duration_ms = step5_duration.as_millis(),
        "Model saved to database"
    );

    // Step 6: Report evaluation results (already computed during training)
    let step6_start = Instant::now();
    if config.max_replays.is_some() {
        info!("Step 6: Skipping evaluation summary (using sampled replays mode)...");
    } else if let Some(valid_loss) = output.final_valid_loss {
        // Evaluation was already done during training using the evaluation split
        let valid_rmse = valid_loss.sqrt();
        let valid_segments = valid_dataset
            .as_ref()
            .map_or(0, |ds: &Arc<SegmentStore>| ds.len());
        info!(
            eval_segments = valid_segments,
            eval_loss = valid_loss,
            eval_rmse = valid_rmse,
            "Evaluation summary (from training validation)"
        );
    } else {
        info!("Step 6: No validation data was used during training");
    }
    let step6_duration = step6_start.elapsed();

    let total_duration = pipeline_start.elapsed();
    info!(
        total_duration_ms = total_duration.as_millis(),
        total_duration_sec = total_duration.as_secs_f64(),
        "=== Training pipeline timing summary ==="
    );
    info!(
        step1_splits_ms = step1_duration.as_millis(),
        step2_data_loading_ms = step2_duration.as_millis(),
        step3_model_init_ms = step3_duration.as_millis(),
        step4_training_ms = step4_duration.as_millis(),
        step5_save_model_ms = step5_duration.as_millis(),
        step6_evaluation_ms = step6_duration.as_millis(),
        "=== Detailed timing breakdown ==="
    );

    info!("=== Training pipeline completed successfully ===");
    Ok(())
}

/// Loads training data using the segment cache for zero-copy access.
///
/// This function:
/// 1. Checks if cached segments exist for each replay
/// 2. For replays without cached segments: parses replay, extracts features, writes segments
/// 3. Memory-maps all segment files for zero-copy access
///
/// # Arguments
///
/// * `replays` - List of replays to load
/// * `segment_length` - Number of frames per segment
/// * `base_path` - Base path for segment cache storage
///
/// # Returns
///
/// Loads training data with lazy-loaded segments.
async fn load_training_data_cached(
    replays: &[Replay],
    segment_length: usize,
    base_path: PathBuf,
) -> Result<Arc<SegmentStore>> {
    const BATCH_SIZE: usize = 100;

    let mut builder =
        SegmentStoreBuilder::new(base_path.clone(), "training".to_string(), segment_length);

    let mut skipped_no_players = 0;
    let mut skipped_parse_error = 0;
    let mut skipped_read_error = 0;
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    let total_replays = replays.len();
    let num_batches = total_replays.div_ceil(BATCH_SIZE);

    for (batch_idx, batch) in replays.chunks(BATCH_SIZE).enumerate() {
        debug!(
            "Processing batch {}/{} ({} replays)...",
            batch_idx + 1,
            num_batches,
            batch.len()
        );

        for replay in batch {
            // Get player ratings for this replay
            let db_players = database::list_replay_players_by_replay(replay.id).await?;

            if db_players.is_empty() {
                skipped_no_players += 1;
                continue;
            }

            // Build target MMR array
            let target_mmr = build_target_mmr_from_players(&db_players);

            // Check if segments already exist
            // For checking, we need to know the frame count - we'll estimate from typical replay size
            // If the cache doesn't exist, we'll parse and get the exact frame count
            let segments_exist = check_replay_segments_cached(
                &base_path,
                &replay.file_path,
                replay.id,
                segment_length,
            );

            if segments_exist {
                // Segments exist, just load them
                cache_hits += 1;
                builder.add_replay(&replay.file_path, replay.id, target_mmr);
                continue;
            }

            // Segments don't exist, need to parse replay and create them
            cache_misses += 1;

            // Read from object_store as bytes
            let object_path = ObjectStorePath::from(replay.file_path.clone());
            let replay_data = match OBJECT_STORE
                .get(&object_path)
                .await
                .context("Failed to read from object_store")
            {
                Ok(get_result) => match get_result.bytes().await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!(replay = %object_path.to_string(), error = %e, "Failed to read bytes");
                        skipped_read_error += 1;
                        continue;
                    }
                },
                Err(e) => {
                    error!(replay = %object_path.to_string(), error = %e, "Failed to get from object_store");
                    skipped_read_error += 1;
                    continue;
                }
            };

            // Parse the replay from bytes
            let parsed = match parse_replay_from_bytes(&replay_data) {
                Ok(p) => p,
                Err(e) => {
                    warn!(replay = %object_path.to_string(), error = %e, "Failed to parse replay");
                    skipped_parse_error += 1;
                    continue;
                }
            };

            if parsed.frames.is_empty() {
                warn!(replay = %object_path.to_string(), "Replay has no frames");
                skipped_parse_error += 1;
                continue;
            }

            // Convert to feature extractor format
            let player_ratings: Vec<PlayerRating> = db_players
                .iter()
                .map(|p| PlayerRating {
                    player_name: p.player_name.clone(),
                    team: p.team,
                    mmr: p.rank_division.mmr_middle(),
                })
                .collect();

            let game_sequence =
                extract_player_centric_game_sequence(&parsed.frames, &player_ratings);

            // Ensure segments are cached, then add replay
            if let Err(e) = builder.ensure_player_centric_segments_cached(
                &replay.file_path,
                replay.id,
                Some(&game_sequence.player_frames),
                Some(game_sequence.player_frames.len()),
            ) {
                warn!(
                    replay_id = %replay.id,
                    error = %e,
                    "Failed to cache segments"
                );
                continue;
            }

            builder.add_replay(&replay.file_path, replay.id, target_mmr);
        }

        let (replays_loaded, replays_cached, segments_loaded) = builder.stats();
        info!(
            "Batch {}/{} complete. Loaded: {}, Cached: {}, Segments: {}",
            batch_idx + 1,
            num_batches,
            replays_loaded,
            replays_cached,
            segments_loaded
        );
    }

    let (replays_loaded, replays_cached, segments_loaded) = builder.stats();
    info!(
        cache_hits,
        cache_misses,
        replays_loaded,
        replays_newly_cached = replays_cached,
        segments_loaded,
        skipped_no_players,
        skipped_parse_error,
        skipped_read_error,
        "Segment cache loading summary"
    );

    let store = builder.build();
    Ok(Arc::new(store))
}

/// Loads validation data with all segments preloaded into memory.
///
/// This function is similar to `load_training_data_cached` but preloads
/// all segments into memory for faster validation access.
/// Validation datasets are typically smaller and accessed frequently,
/// so keeping them in memory is a good trade-off.
///
/// # Arguments
///
/// * `replays` - List of replays to load
/// * `segment_length` - Number of frames per segment
/// * `base_path` - Base path for segment cache storage
///
/// # Returns
///
/// Loads validation data with all segments preloaded into memory.
async fn load_validation_data_cached(
    replays: &[Replay],
    segment_length: usize,
    base_path: PathBuf,
) -> Result<Arc<SegmentStore>> {
    const BATCH_SIZE: usize = 100;

    let mut builder =
        SegmentStoreBuilder::new(base_path.clone(), "validation".to_string(), segment_length);

    let mut skipped_no_players = 0;
    let mut skipped_parse_error = 0;
    let mut skipped_read_error = 0;
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    let total_replays = replays.len();
    let num_batches = total_replays.div_ceil(BATCH_SIZE);

    for (batch_idx, batch) in replays.chunks(BATCH_SIZE).enumerate() {
        debug!(
            "Processing validation batch {}/{} ({} replays)...",
            batch_idx + 1,
            num_batches,
            batch.len()
        );

        for replay in batch {
            // Get player ratings for this replay
            let db_players = database::list_replay_players_by_replay(replay.id).await?;

            if db_players.is_empty() {
                skipped_no_players += 1;
                continue;
            }

            // Build target MMR array
            let target_mmr = build_target_mmr_from_players(&db_players);

            // Check if segments already exist
            let segments_exist = check_replay_segments_cached(
                &base_path,
                &replay.file_path,
                replay.id,
                segment_length,
            );

            if segments_exist {
                // Segments exist, just load them
                cache_hits += 1;
                builder.add_replay(&replay.file_path, replay.id, target_mmr);
                continue;
            }

            // Segments don't exist, need to parse replay and create them
            cache_misses += 1;

            // Read from object_store as bytes
            let object_path = ObjectStorePath::from(replay.file_path.clone());
            let replay_data = match OBJECT_STORE
                .get(&object_path)
                .await
                .context("Failed to read from object_store")
            {
                Ok(get_result) => match get_result.bytes().await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        error!(replay = %object_path.to_string(), error = %e, "Failed to read bytes");
                        skipped_read_error += 1;
                        continue;
                    }
                },
                Err(e) => {
                    error!(replay = %object_path.to_string(), error = %e, "Failed to get from object_store");
                    skipped_read_error += 1;
                    continue;
                }
            };

            // Parse the replay from bytes
            let parsed = match parse_replay_from_bytes(&replay_data) {
                Ok(p) => p,
                Err(e) => {
                    warn!(replay = %object_path.to_string(), error = %e, "Failed to parse replay");
                    skipped_parse_error += 1;
                    continue;
                }
            };

            if parsed.frames.is_empty() {
                warn!(replay = %object_path.to_string(), "Replay has no frames");
                skipped_parse_error += 1;
                continue;
            }

            // Convert to feature extractor format
            let player_ratings: Vec<PlayerRating> = db_players
                .iter()
                .map(|p| PlayerRating {
                    player_name: p.player_name.clone(),
                    team: p.team,
                    mmr: p.rank_division.mmr_middle(),
                })
                .collect();

            let game_sequence =
                extract_player_centric_game_sequence(&parsed.frames, &player_ratings);

            // Ensure segments are cached, then add replay
            if let Err(e) = builder.ensure_player_centric_segments_cached(
                &replay.file_path,
                replay.id,
                Some(&game_sequence.player_frames),
                Some(game_sequence.player_frames.len()),
            ) {
                warn!(
                    replay_id = %replay.id,
                    error = %e,
                    "Failed to cache segments"
                );
                continue;
            }

            builder.add_replay(&replay.file_path, replay.id, target_mmr);
        }

        let (replays_loaded, replays_cached, segments_loaded) = builder.stats();
        info!(
            "Validation batch {}/{} complete. Loaded: {}, Cached: {}, Segments: {}",
            batch_idx + 1,
            num_batches,
            replays_loaded,
            replays_cached,
            segments_loaded
        );
    }

    let (replays_loaded, replays_cached, segments_loaded) = builder.stats();
    info!(
        cache_hits,
        cache_misses,
        replays_loaded,
        replays_newly_cached = replays_cached,
        segments_loaded,
        skipped_no_players,
        skipped_parse_error,
        skipped_read_error,
        "Validation segment cache loading summary"
    );

    // Preload all segments into memory for faster validation access
    let mut store = builder.build();
    info!(
        store_name = %store.name,
        "Preloading all segments into memory"
    );
    store.preload_all_segments()?;

    Ok(Arc::new(store))
}

/// Checks if cached segments exist for a replay.
///
/// This does a lightweight check to see if the segment directory exists
/// and has at least one segment file.
fn check_replay_segments_cached(
    base_path: &std::path::Path,
    file_path: &str,
    replay_id: uuid::Uuid,
    _segment_length: usize,
) -> bool {
    use ml_model::segment_cache::segment_directory;

    let segment_dir = segment_directory(base_path, file_path, replay_id);

    if !segment_dir.exists() {
        return false;
    }

    // Check if there's at least one .features file
    if let Ok(entries) = std::fs::read_dir(&segment_dir) {
        for entry in entries.flatten() {
            if entry
                .path()
                .extension()
                .is_some_and(|ext| ext == "features")
            {
                return true;
            }
        }
    }

    false
}

/// Builds a target MMR array from database player records.
fn build_target_mmr_from_players(players: &[replay_structs::ReplayPlayer]) -> [f32; TOTAL_PLAYERS] {
    let mut target_mmr = [1000.0f32; TOTAL_PLAYERS];

    // Separate by team and sort
    let mut blue_players: Vec<_> = players.iter().filter(|p| p.team == 0).collect();
    let mut orange_players: Vec<_> = players.iter().filter(|p| p.team == 1).collect();

    // Sort by name for consistent ordering
    blue_players.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange_players.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    // Fill blue team (first 3 slots)
    for (i, player) in blue_players.iter().take(3).enumerate() {
        if let Some(slot) = target_mmr.get_mut(i) {
            *slot = player.rank_division.mmr_middle() as f32;
        }
    }

    // Fill orange team (slots 3-5)
    for (i, player) in orange_players.iter().take(3).enumerate() {
        if let Some(slot) = target_mmr.get_mut(i + 3) {
            *slot = player.rank_division.mmr_middle() as f32;
        }
    }

    target_mmr
}

/// Finds the latest checkpoint file for a given prefix.
fn find_latest_checkpoint(prefix: &str) -> Result<Option<String>> {
    let parent = std::path::Path::new(prefix)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

    if !parent.exists() {
        return Ok(None);
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(parent)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let name = path.file_name()?.to_str()?;

            // Look for checkpoint files (ending in .mpk)
            if name.starts_with("checkpoint_")
                && std::path::Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("mpk"))
            {
                Some(path.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();

    if checkpoints.is_empty() {
        return Ok(None);
    }

    // Sort by modification time (newest first)
    checkpoints.sort_by(|a, b| {
        let a_time = std::fs::metadata(a)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let b_time = std::fs::metadata(b)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        b_time.cmp(&a_time)
    });

    // Return the path without the .mpk extension (as expected by load_checkpoint)
    if let Some(latest) = checkpoints.first() {
        let path = latest.trim_end_matches(".mpk");
        Ok(Some(path.to_string()))
    } else {
        Ok(None)
    }
}

/// Extracts the epoch number from a checkpoint filename.
fn extract_epoch_from_checkpoint(path: &str) -> Option<usize> {
    // Pattern: checkpoint_epoch{N} or checkpoint_epoch{N}.mpk
    let filename = std::path::Path::new(path).file_name()?.to_str()?;

    if let Some(rest) = filename.strip_prefix("checkpoint_epoch") {
        let epoch_str = rest.trim_end_matches(".mpk");
        epoch_str.parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_epoch_from_checkpoint() {
        assert_eq!(extract_epoch_from_checkpoint("checkpoint_epoch5"), Some(5));
        assert_eq!(
            extract_epoch_from_checkpoint("checkpoint_epoch10.mpk"),
            Some(10)
        );
        assert_eq!(
            extract_epoch_from_checkpoint("/path/to/checkpoint_epoch25.mpk"),
            Some(25)
        );
        assert_eq!(extract_epoch_from_checkpoint("checkpoint_best"), None);
        assert_eq!(extract_epoch_from_checkpoint("checkpoint_final"), None);
    }

    #[test]
    fn test_full_train_config_default() {
        let config = FullTrainConfig::default();
        assert!((config.train_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 128);
        assert!(!config.resume);
        assert!(config.max_replays.is_none());
    }
}
