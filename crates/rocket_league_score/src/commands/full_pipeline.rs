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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use config::OBJECT_STORE;
use feature_extractor::{FrameFeatures, PlayerRating, TOTAL_PLAYERS, extract_game_sequence};
use ml_model::{
    CheckpointConfig, GameLoader, GameMetadata, LazySegmentDataset, ModelConfig, SequenceSample,
    SequenceTrainingData, TrainingConfig, TrainingState, create_model, load_checkpoint,
    save_checkpoint, train_with_dataset,
};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use parking_lot::RwLock;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, Replay};
use tracing::{error, info, warn};

use super::init_device;

/// Training requires Autodiff wrapper for automatic differentiation.
type TrainBackend = Autodiff<Wgpu>;

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
            model_name: String::from("lstm_v2"),
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

    // Step 2: Collect metadata (lightweight - no frame data yet)
    let step2_start = Instant::now();
    info!("Step 2: Collecting game metadata...");

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

    info!(replay_count = training_replays.len(), "Loaded replays");

    let (train_metadata, train_player_map) =
        collect_game_metadata(&training_replays, config.train_ratio).await?;
    let step2_duration = step2_start.elapsed();

    if train_metadata.is_empty() {
        anyhow::bail!("No valid training games found.");
    }

    // Split metadata into train/valid (last 10% for validation)
    let valid_count = (train_metadata.len() as f64 * 0.1) as usize;
    let train_count = train_metadata.len() - valid_count;

    let (train_games, valid_games) = train_metadata.split_at(train_count);
    let train_games = train_games.to_vec();
    let valid_games = valid_games.to_vec();

    info!(
        train_games = train_games.len(),
        valid_games = valid_games.len(),
        duration_ms = step2_duration.as_millis(),
        "Split metadata into train/valid"
    );

    // Step 3: Create lazy datasets with game loader
    let step3_start = Instant::now();
    info!("Step 3: Creating lazy datasets...");
    let loader = Arc::new(ReplayLoader::new(train_player_map));

    let segment_length = 90; // 3 seconds at 30fps
    let train_dataset =
        LazySegmentDataset::new(train_games.clone(), loader.clone(), segment_length);
    let valid_dataset = if valid_games.is_empty() {
        None
    } else {
        Some(LazySegmentDataset::new(
            valid_games.clone(),
            loader,
            segment_length,
        ))
    };

    let step3_duration = step3_start.elapsed();
    info!(
        train_segments = train_dataset.len(),
        valid_segments = valid_dataset.as_ref().map_or(0, LazySegmentDataset::len),
        duration_ms = step3_duration.as_millis(),
        "Created lazy datasets"
    );

    // Step 4: Create or load model
    let step4_start = Instant::now();
    info!("Step 4: Initializing model...");
    let device = init_device();
    let model_config = ModelConfig::new();
    let training_config = TrainingConfig::new(model_config.clone())
        .with_learning_rate(config.learning_rate)
        .with_epochs(config.epochs)
        .with_batch_size(config.batch_size)
        .with_sequence_length(segment_length);

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
    let step4_duration = step4_start.elapsed();

    info!(
        lstm_hidden_1 = model_config.lstm_hidden_1,
        lstm_hidden_2 = model_config.lstm_hidden_2,
        feedforward_hidden = model_config.feedforward_hidden,
        dropout = model_config.dropout,
        duration_ms = step4_duration.as_millis(),
        "Model architecture"
    );

    // Step 5: Train with lazy datasets
    let step5_start = Instant::now();
    info!("Step 5: Starting training with lazy loading...");
    let checkpoint_config = CheckpointConfig {
        path_prefix: checkpoint_prefix.clone(),
        save_every_n_epochs: config.checkpoint_every_n_epochs,
        save_on_improvement: true,
    };

    let output = train_with_dataset(
        &mut model,
        train_dataset,
        valid_dataset,
        &training_config,
        Some(checkpoint_config),
        start_state,
        train_games.len(),
        valid_games.len(),
    )?;

    let train_rmse = output.final_train_loss.sqrt();
    let valid_rmse = output.final_valid_loss.map(f32::sqrt);
    let step5_duration = step5_start.elapsed();

    info!(
        final_train_loss = output.final_train_loss,
        train_rmse,
        final_valid_loss = output.final_valid_loss,
        valid_rmse,
        epochs_completed = output.epochs_completed,
        checkpoints_saved = output.checkpoint_paths.len(),
        duration_ms = step5_duration.as_millis(),
        duration_sec = step5_duration.as_secs_f64(),
        avg_epoch_ms = if output.epochs_completed > 0 {
            step5_duration.as_millis() / output.epochs_completed as u128
        } else {
            0
        },
        "Training completed"
    );

    // Step 6: Save final model to database
    let step6_start = Instant::now();
    info!("Step 6: Saving model to database...");
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
        "training_games": train_games.len(),
        "validation_games": valid_games.len(),
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

    let step6_duration = step6_start.elapsed();
    info!(
        model_name = %config.model_name,
        version = next_version,
        checkpoint_path = %final_checkpoint_path,
        duration_ms = step6_duration.as_millis(),
        "Model saved to database"
    );

    // Step 7: Run evaluation on held-out test set (skip when using max_replays)
    let step7_start = Instant::now();
    if config.max_replays.is_some() {
        info!("Step 7: Skipping evaluation (using sampled replays mode)...");
    } else {
        info!("Step 7: Running evaluation on test set...");
        let eval_replays = database::list_replays_by_split(DatasetSplit::Evaluation).await?;

        if !eval_replays.is_empty() {
            let eval_data = load_training_data_from_replays(&eval_replays).await?;
            if !eval_data.is_empty() {
                let eval_loss = evaluate_model(&model, &eval_data, &training_config, &device);
                let eval_rmse = eval_loss.sqrt();
                let step7_duration = step7_start.elapsed();
                info!(
                    eval_samples = eval_data.len(),
                    eval_loss,
                    eval_rmse,
                    duration_ms = step7_duration.as_millis(),
                    "Evaluation completed"
                );
            } else {
                warn!("No valid evaluation samples extracted");
            }
        } else {
            warn!("No evaluation replays available");
        }
    }
    let step7_duration = step7_start.elapsed();

    let total_duration = pipeline_start.elapsed();
    info!(
        total_duration_ms = total_duration.as_millis(),
        total_duration_sec = total_duration.as_secs_f64(),
        "=== Training pipeline timing summary ==="
    );
    info!(
        step1_splits_ms = step1_duration.as_millis(),
        step2_metadata_ms = step2_duration.as_millis(),
        step3_datasets_ms = step3_duration.as_millis(),
        step4_model_init_ms = step4_duration.as_millis(),
        step5_training_ms = step5_duration.as_millis(),
        step6_save_model_ms = step6_duration.as_millis(),
        step7_evaluation_ms = step7_duration.as_millis(),
        "=== Detailed timing breakdown ==="
    );

    info!("=== Training pipeline completed successfully ===");
    Ok(())
}

/// Loads training data from a list of replays in batches to manage memory.
///
/// Processes replays in batches to avoid loading all raw replay data at once.
/// Frame data is subsampled (every 3rd frame by default) to further reduce memory.
async fn load_training_data_from_replays(replays: &[Replay]) -> Result<SequenceTrainingData> {
    const BATCH_SIZE: usize = 100; // Process 100 replays at a time

    let mut data = SequenceTrainingData::new();

    let mut skipped_no_players = 0;
    let mut skipped_parse_error = 0;
    let mut skipped_read_error = 0;

    let total_replays = replays.len();
    let num_batches = total_replays.div_ceil(BATCH_SIZE);

    for (batch_idx, batch) in replays.chunks(BATCH_SIZE).enumerate() {
        info!(
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

            // Convert to feature extractor format
            let player_ratings: Vec<PlayerRating> = db_players
                .iter()
                .map(|p| PlayerRating {
                    player_name: p.player_name.clone(),
                    team: p.team,
                    mmr: p.rank_division.mmr_middle(),
                })
                .collect();

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

            // Extract game sequence with subsampling (every 3rd frame = ~10fps instead of 30fps)
            // This reduces memory by 3x while preserving temporal patterns
            let game_sequence = extract_game_sequence(
                &parsed.frames,
                &player_ratings,
                Some(&parsed.goal_frames),
                Some(&parsed.goals),
            );

            // Convert to SequenceSample for ml_model
            data.add_sample(SequenceSample {
                frames: game_sequence.frames,
                target_mmr: game_sequence.target_mmr,
            });

            // parsed and replay_data are dropped here, freeing memory before next iteration
        }

        // Log progress with memory estimate
        let estimated_memory_mb = data.len() * 9000 * 152 * 4 / (1024 * 1024); // ~9000 frames at 30fps, 152 features, 4 bytes
        info!(
            "Batch {}/{} complete. Total samples: {}, estimated memory: ~{}MB",
            batch_idx + 1,
            num_batches,
            data.len(),
            estimated_memory_mb
        );
    }

    if skipped_no_players > 0 || skipped_parse_error > 0 || skipped_read_error > 0 {
        info!(
            skipped_no_players,
            skipped_parse_error,
            skipped_read_error,
            processed = data.len(),
            "Data loading summary"
        );
    }

    Ok(data)
}

/// Stored info about a game for lazy loading.
#[derive(Debug, Clone)]
struct GameInfo {
    file_path: String,
    player_ratings: Vec<PlayerRating>,
}

/// Loader that loads replay data on-demand from object store.
struct ReplayLoader {
    /// Map from `game_id` to game info for loading.
    game_info: HashMap<String, GameInfo>,
    /// Cache of recently loaded games (LRU-style, limited size).
    cache: RwLock<HashMap<String, Vec<FrameFeatures>>>,
}

impl ReplayLoader {
    fn new(game_info: HashMap<String, GameInfo>) -> Self {
        Self {
            game_info,
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl GameLoader for ReplayLoader {
    fn load_game(&self, game_id: &str) -> Option<Vec<FrameFeatures>> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(frames) = cache.get(game_id) {
                return Some(frames.clone());
            }
        }

        // Load from object store
        let info = self.game_info.get(game_id)?;

        // Use block_in_place to move blocking work to a thread pool
        // This allows us to call block_on from within an async runtime
        let file_path = info.file_path.clone();
        let player_ratings = info.player_ratings.clone();

        let frames = tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::try_current().ok()?;
            handle.block_on(async {
                let object_path = ObjectStorePath::from(file_path);
                let get_result = OBJECT_STORE.get(&object_path).await.ok()?;
                let bytes = get_result.bytes().await.ok()?;
                let parsed = parse_replay_from_bytes(&bytes).ok()?;

                if parsed.frames.is_empty() {
                    return None;
                }

                let game_sequence = extract_game_sequence(
                    &parsed.frames,
                    &player_ratings,
                    Some(&parsed.goal_frames),
                    Some(&parsed.goals),
                );

                Some(game_sequence.frames)
            })
        })?;

        // Store in cache (simple cache, no eviction for now)
        // In production, you'd want LRU eviction to limit cache size
        {
            let mut cache = self.cache.write();
            // Limit cache to ~100 games to avoid memory issues
            if cache.len() >= 100 {
                // Simple eviction: clear half the cache
                let keys_to_remove: Vec<_> = cache.keys().take(50).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
            cache.insert(game_id.to_string(), frames.clone());
        }

        Some(frames)
    }
}

/// Collects lightweight metadata for all games without loading frame data.
///
/// Returns `(game_metadata, player_map)` where `player_map` is used by the loader.
async fn collect_game_metadata(
    replays: &[Replay],
    _train_ratio: f64,
) -> Result<(Vec<GameMetadata>, HashMap<String, GameInfo>)> {
    let mut metadata = Vec::with_capacity(replays.len());
    let mut player_map = HashMap::with_capacity(replays.len());

    let mut skipped_no_players = 0;

    for replay in replays {
        let game_id = replay.id.to_string();

        // Get player ratings for this replay
        let db_players = database::list_replay_players_by_replay(replay.id).await?;

        if db_players.is_empty() {
            skipped_no_players += 1;
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

        // Build target MMR array
        let target_mmr = build_target_mmr_from_ratings(&player_ratings);

        // Estimate frame count (5 min game at 30fps = 9000 frames)
        let estimated_frame_count = 9000;

        metadata.push(GameMetadata {
            game_id: game_id.clone(),
            target_mmr,
            estimated_frame_count,
        });

        player_map.insert(
            game_id,
            GameInfo {
                file_path: replay.file_path.clone(),
                player_ratings,
            },
        );
    }

    if skipped_no_players > 0 {
        info!(skipped_no_players, "Skipped games with no player data");
    }

    info!(
        total_games = metadata.len(),
        "Collected game metadata (no frame data loaded yet)"
    );

    Ok((metadata, player_map))
}

/// Builds target MMR array from player ratings.
fn build_target_mmr_from_ratings(player_ratings: &[PlayerRating]) -> [f32; TOTAL_PLAYERS] {
    let mut target_mmr = [1000.0f32; TOTAL_PLAYERS];

    // Team values: 0 = Blue, 1 = Orange (matches replay_structs::Team enum)
    const TEAM_BLUE: i16 = 0;
    const TEAM_ORANGE: i16 = 1;

    // Sort players by team, then by name for consistency
    let mut blue_players: Vec<_> = player_ratings
        .iter()
        .filter(|p| p.team == TEAM_BLUE)
        .collect();
    let mut orange_players: Vec<_> = player_ratings
        .iter()
        .filter(|p| p.team == TEAM_ORANGE)
        .collect();

    blue_players.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange_players.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    // Fill slots: first 3 for blue, next 3 for orange
    for (i, player) in blue_players.iter().take(3).enumerate() {
        let Some(target_mmr_player) = target_mmr.get_mut(i) else {
            error!("Blue team has 3 players");
            continue;
        };
        *target_mmr_player = player.mmr as f32;
    }
    for (i, player) in orange_players.iter().take(3).enumerate() {
        let Some(target_mmr_player) = target_mmr.get_mut(3 + i) else {
            error!("Orange team has 3 players");
            continue;
        };
        *target_mmr_player = player.mmr as f32;
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

/// Evaluates the model on a dataset and returns the loss.
fn evaluate_model<B: burn::prelude::Backend>(
    model: &ml_model::SequenceModel<B>,
    data: &SequenceTrainingData,
    config: &TrainingConfig,
    device: &B::Device,
) -> f32 {
    use burn::data::dataset::Dataset;
    use burn::nn::loss::MseLoss;
    use ml_model::{SegmentDataset, SequenceBatcher};

    let dataset = SegmentDataset::new(data.samples.clone(), config.sequence_length);
    let batcher = SequenceBatcher::<B>::new(device.clone(), config.sequence_length);
    let loss_fn = MseLoss::new();

    let mut total_loss = 0.0f64;
    let mut batch_count = 0;

    const BATCH_SIZE: usize = 32;
    let num_samples = dataset.len();

    for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(num_samples);

        let items: Vec<_> = (batch_start..batch_end)
            .filter_map(|i| dataset.get(i))
            .collect();

        if items.is_empty() {
            continue;
        }

        let batch = batcher.batch(items);
        let predictions = model.forward(batch.inputs);
        let loss = loss_fn.forward(predictions, batch.targets, burn::nn::loss::Reduction::Mean);

        let loss_value: f32 = loss
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![0.0])
            .first()
            .copied()
            .unwrap_or(0.0);

        total_loss += loss_value as f64;
        batch_count += 1;
    }

    if batch_count > 0 {
        (total_loss / batch_count as f64) as f32
    } else {
        0.0
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
        assert_eq!(config.model_name, "lstm_v2");
        assert!((config.train_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 128);
        assert!(!config.resume);
        assert!(config.max_replays.is_none());
    }
}
