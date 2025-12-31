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

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use config::OBJECT_STORE;
use feature_extractor::{PlayerRating, extract_game_sequence};
use ml_model::{
    CheckpointConfig, ModelConfig, SequenceSample, SequenceTrainingData, TrainingConfig,
    TrainingState, create_model, load_checkpoint, save_checkpoint, train_with_checkpoints,
};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, DownloadStatus, Replay};
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
}

impl Default for FullTrainConfig {
    fn default() -> Self {
        Self {
            model_name: String::from("lstm_v1"),
            train_ratio: 0.9,
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            resume: false,
            checkpoint_every_n_epochs: 5,
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
) -> Result<()> {
    let config = FullTrainConfig {
        model_name: model_name.to_string(),
        train_ratio,
        epochs,
        batch_size,
        learning_rate,
        resume,
        checkpoint_every_n_epochs: 5,
    };

    run_with_config(&config).await
}

/// Runs the full training pipeline with a config struct.
///
/// # Errors
///
/// Returns an error if training fails.
pub async fn run_with_config(config: &FullTrainConfig) -> Result<()> {
    info!(
        model_name = %config.model_name,
        train_ratio = config.train_ratio,
        epochs = config.epochs,
        resume = config.resume,
        "Starting full training pipeline"
    );

    // Step 1: Assign dataset splits to unassigned replays
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
    let current_counts = database::count_replays_by_split().await?;
    info!(
        training_replays = current_counts.training,
        evaluation_replays = current_counts.evaluation,
        "Current dataset split sizes"
    );

    if current_counts.training == 0 {
        anyhow::bail!("No training replays available. Please ingest and download replays first.");
    }

    // Step 2: Load training data
    info!("Step 2: Loading training data from database...");
    let training_replays = database::list_replays_by_split(DatasetSplit::Training).await?;
    let training_data = load_training_data_from_replays(&training_replays).await?;

    if training_data.is_empty() {
        anyhow::bail!("No valid training samples extracted from replays.");
    }

    info!(
        samples = training_data.len(),
        "Loaded training samples"
    );

    // Step 3: Create or load model
    info!("Step 3: Initializing model...");
    let device = init_device();
    let model_config = ModelConfig::new();
    let training_config = TrainingConfig::new(model_config.clone())
        .with_learning_rate(config.learning_rate)
        .with_epochs(config.epochs)
        .with_batch_size(config.batch_size);

    let checkpoint_dir = format!("models/{}", config.model_name);
    let checkpoint_prefix = format!("{}/checkpoint", checkpoint_dir);

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

    info!(
        lstm_hidden_1 = model_config.lstm_hidden_1,
        lstm_hidden_2 = model_config.lstm_hidden_2,
        feedforward_hidden = model_config.feedforward_hidden,
        dropout = model_config.dropout,
        "Model architecture"
    );

    // Step 4: Train with checkpoints
    info!("Step 4: Starting training...");
    let checkpoint_config = CheckpointConfig {
        path_prefix: checkpoint_prefix.clone(),
        save_every_n_epochs: config.checkpoint_every_n_epochs,
        save_on_improvement: true,
    };

    let output = train_with_checkpoints(
        &mut model,
        &training_data,
        &training_config,
        Some(checkpoint_config),
        start_state,
    )?;

    let train_rmse = output.final_train_loss.sqrt();
    let valid_rmse = output.final_valid_loss.map(f32::sqrt);

    info!(
        final_train_loss = output.final_train_loss,
        train_rmse,
        final_valid_loss = output.final_valid_loss,
        valid_rmse,
        epochs_completed = output.epochs_completed,
        checkpoints_saved = output.checkpoint_paths.len(),
        "Training completed"
    );

    // Step 5: Save final model to database
    info!("Step 5: Saving model to database...");
    let next_version = database::get_next_model_version(&config.model_name).await?;
    let final_checkpoint_path = format!("{}/final_v{}", checkpoint_dir, next_version);

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
        "training_samples": training_data.len(),
        "evaluation_samples": current_counts.evaluation,
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

    info!(
        model_name = %config.model_name,
        version = next_version,
        checkpoint_path = %final_checkpoint_path,
        "Model saved to database"
    );

    // Step 6: Run evaluation on held-out test set
    info!("Step 6: Running evaluation on test set...");
    let eval_replays = database::list_replays_by_split(DatasetSplit::Evaluation).await?;

    if !eval_replays.is_empty() {
        let eval_data = load_training_data_from_replays(&eval_replays).await?;
        if !eval_data.is_empty() {
            let eval_loss = evaluate_model(&model, &eval_data, &training_config, &device)?;
            let eval_rmse = eval_loss.sqrt();
            info!(
                eval_samples = eval_data.len(),
                eval_loss,
                eval_rmse,
                "Evaluation completed"
            );
        } else {
            warn!("No valid evaluation samples extracted");
        }
    } else {
        warn!("No evaluation replays available");
    }

    info!("=== Training pipeline completed successfully ===");
    Ok(())
}

/// Loads training data from a list of replays.
async fn load_training_data_from_replays(replays: &[Replay]) -> Result<SequenceTrainingData> {
    let mut data = SequenceTrainingData::new();

    let mut skipped_no_players = 0;
    let mut skipped_parse_error = 0;
    let mut skipped_read_error = 0;

    for replay in replays {
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

        // Extract game sequence (one sample per replay)
        let game_sequence = extract_game_sequence(&parsed.frames, &player_ratings);

        // Convert to SequenceSample for ml_model
        data.add_sample(SequenceSample {
            frames: game_sequence.frames,
            target_mmr: game_sequence.target_mmr,
        });
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

/// Finds the latest checkpoint file for a given prefix.
fn find_latest_checkpoint(prefix: &str) -> Result<Option<String>> {
    let parent = std::path::Path::new(prefix)
        .parent()
        .unwrap_or(std::path::Path::new("."));

    if !parent.exists() {
        return Ok(None);
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(parent)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let name = path.file_name()?.to_str()?;

            // Look for checkpoint files (ending in .mpk)
            if name.starts_with("checkpoint_") && name.ends_with(".mpk") {
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
    let filename = std::path::Path::new(path)
        .file_name()?
        .to_str()?;

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
) -> Result<f32> {
    use burn::nn::loss::MseLoss;
    use ml_model::{SequenceBatcher, SequenceDataset};
    use burn::data::dataset::Dataset;

    let dataset = SequenceDataset::new(&data.samples, config.sequence_length);
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
        Ok((total_loss / batch_count as f64) as f32)
    } else {
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_epoch_from_checkpoint() {
        assert_eq!(
            extract_epoch_from_checkpoint("checkpoint_epoch5"),
            Some(5)
        );
        assert_eq!(
            extract_epoch_from_checkpoint("checkpoint_epoch10.mpk"),
            Some(10)
        );
        assert_eq!(
            extract_epoch_from_checkpoint("/path/to/checkpoint_epoch25.mpk"),
            Some(25)
        );
        assert_eq!(
            extract_epoch_from_checkpoint("checkpoint_best"),
            None
        );
        assert_eq!(
            extract_epoch_from_checkpoint("checkpoint_final"),
            None
        );
    }

    #[test]
    fn test_full_train_config_default() {
        let config = FullTrainConfig::default();
        assert_eq!(config.model_name, "lstm_v1");
        assert!((config.train_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert!(!config.resume);
    }
}

