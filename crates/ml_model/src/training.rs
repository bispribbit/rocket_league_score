//! Training logic for the sequence model.

use burn::data::dataset::Dataset;
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::dataset::{SequenceBatcher, SequenceDataset};
use crate::{SequenceModel, SequenceTrainingData, TrainingConfig, save_checkpoint};

/// Tracks the current state of training for resumption.
#[derive(Debug, Clone, Default)]
pub struct TrainingState {
    /// Current epoch number (0-indexed).
    pub current_epoch: usize,
    /// Best validation loss seen so far.
    pub best_valid_loss: f32,
    /// Number of epochs since the best validation loss was achieved.
    pub epochs_without_improvement: usize,
    /// Current training loss.
    pub current_train_loss: f32,
    /// Current validation loss.
    pub current_valid_loss: Option<f32>,
}

impl TrainingState {
    /// Creates a new training state starting from a given epoch.
    #[must_use]
    pub const fn new(start_epoch: usize) -> Self {
        Self {
            current_epoch: start_epoch,
            best_valid_loss: f32::MAX,
            epochs_without_improvement: 0,
            current_train_loss: 0.0,
            current_valid_loss: None,
        }
    }
}

/// Configuration for checkpoint saving during training.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Path prefix for checkpoint files.
    pub path_prefix: String,
    /// Save checkpoint every N epochs (0 to disable).
    pub save_every_n_epochs: usize,
    /// Also save on validation improvement.
    pub save_on_improvement: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            path_prefix: String::from("models/checkpoint"),
            save_every_n_epochs: 5,
            save_on_improvement: true,
        }
    }
}

impl CheckpointConfig {
    /// Creates a new checkpoint config with default settings.
    #[must_use]
    pub fn new(path_prefix: &str) -> Self {
        Self {
            path_prefix: path_prefix.to_string(),
            ..Default::default()
        }
    }
}

/// Output from training.
#[derive(Debug, Clone)]
pub struct TrainingOutput {
    /// Final training loss.
    pub final_train_loss: f32,
    /// Final validation loss (if validation data was used).
    pub final_valid_loss: Option<f32>,
    /// Number of epochs completed.
    pub epochs_completed: usize,
    /// Paths to checkpoints saved during training.
    pub checkpoint_paths: Vec<String>,
}

/// Trains the sequence model on the provided data.
///
/// Uses Adam optimizer with MSE loss. The model learns to predict
/// per-player MMR from sequences of game frames.
///
/// # Arguments
///
/// * `model` - The model to train (will be modified in place).
/// * `data` - The training data (game/segment samples).
/// * `config` - Training configuration.
///
/// # Errors
///
/// Returns an error if training fails.
pub fn train<B: AutodiffBackend>(
    model: &mut SequenceModel<B>,
    data: &SequenceTrainingData,
    config: &TrainingConfig,
) -> anyhow::Result<TrainingOutput>
where
    B::FloatElem: From<f32>,
{
    train_with_checkpoints(model, data, config, None, None)
}

/// Trains the sequence model with checkpoint support.
///
/// Uses Adam optimizer with MSE loss. The model learns to predict
/// per-player MMR from sequences of game frames. Supports:
/// - Resumption from a previous epoch
/// - Checkpoint saving every N epochs
/// - Early stopping based on validation loss
///
/// # Arguments
///
/// * `model` - The model to train (will be modified in place).
/// * `data` - The training data (game/segment samples).
/// * `config` - Training configuration.
/// * `checkpoint_config` - Optional checkpoint configuration for saving during training.
/// * `start_state` - Optional starting state for resumption.
///
/// # Errors
///
/// Returns an error if training fails.
pub fn train_with_checkpoints<B: AutodiffBackend>(
    model: &mut SequenceModel<B>,
    data: &SequenceTrainingData,
    config: &TrainingConfig,
    checkpoint_config: Option<CheckpointConfig>,
    start_state: Option<TrainingState>,
) -> anyhow::Result<TrainingOutput>
where
    B::FloatElem: From<f32>,
{
    if data.is_empty() {
        return Err(anyhow::anyhow!("No training data provided"));
    }

    let device = model.linear1.weight.device();

    // Split data into training and validation sets
    let (train_samples, valid_samples) = data.split(config.validation_split);

    if train_samples.is_empty() {
        return Err(anyhow::anyhow!("No training samples after split"));
    }

    // Create dataset and batcher
    let dataset = SequenceDataset::new(&train_samples, config.sequence_length);
    let batcher = SequenceBatcher::<B>::new(device.clone(), config.sequence_length);

    // Create validation dataset if we have validation data
    let valid_dataset = if !valid_samples.is_empty() {
        Some(SequenceDataset::new(&valid_samples, config.sequence_length))
    } else {
        None
    };

    // Create optimizer with weight decay for regularization
    let mut optimizer = AdamConfig::new().init();

    let loss_fn = MseLoss::new();
    let mut checkpoint_paths = Vec::new();

    // Initialize training state (from provided state or fresh)
    let mut state = start_state.unwrap_or_else(|| TrainingState::new(0));
    let start_epoch = state.current_epoch;

    const EARLY_STOPPING_PATIENCE: usize = 10;

    println!(
        "Starting training with {} samples ({} train, {} valid)",
        data.len(),
        train_samples.len(),
        valid_samples.len()
    );
    println!(
        "Sequence length: {}, Batch size: {}, Learning rate: {}",
        config.sequence_length, config.batch_size, config.learning_rate
    );
    if start_epoch > 0 {
        println!("Resuming from epoch {}", start_epoch + 1);
    }
    if let Some(ref ckpt_cfg) = checkpoint_config {
        println!(
            "Checkpoints will be saved every {} epochs to {}",
            ckpt_cfg.save_every_n_epochs, ckpt_cfg.path_prefix
        );
    }

    for epoch in start_epoch..config.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Simple batching with shuffling
        let num_samples = dataset.len();
        let mut indices: Vec<usize> = (0..num_samples).collect();

        // Shuffle indices using epoch as seed
        shuffle_indices(&mut indices, epoch as u64);

        // Process batches
        for batch_start in (0..num_samples).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(num_samples);
            let Some(batch_indices) = indices.get(batch_start..batch_end) else {
                continue;
            };

            // Collect batch items
            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items);

            // Forward pass
            let predictions = model.forward(batch.inputs);
            let loss = loss_fn.forward(predictions, batch.targets, burn::nn::loss::Reduction::Mean);

            // Extract loss value for epoch average
            // Note: This GPU->CPU sync is expensive, but necessary for accurate epoch metrics
            // With batch_size=128, we now have ~4x fewer batches, reducing sync overhead significantly
            let loss_value: f32 = loss
                .clone()
                .into_data()
                .to_vec()
                .unwrap_or_else(|_| vec![0.0])
                .first()
                .copied()
                .unwrap_or(0.0);

            epoch_loss += loss_value as f64;
            batch_count += 1;

            // Log progress every 20 batches to show training is progressing
            // (fewer batches per epoch now with larger batch size)
            if batch_count % 20 == 0 {
                let avg_loss_so_far = epoch_loss / batch_count as f64;
                let total_batches = (num_samples + config.batch_size - 1) / config.batch_size;
                println!(
                    "  Batch {batch_count}/{total_batches}, avg_loss = {:.6}",
                    avg_loss_so_far
                );
            }

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Update weights
            *model = optimizer.step(config.learning_rate, model.clone(), grads);
        }

        state.current_train_loss = if batch_count > 0 {
            (epoch_loss / batch_count as f64) as f32
        } else {
            0.0
        };
        state.current_epoch = epoch;

        // Validation
        let mut improved = false;
        if let Some(valid_ds) = &valid_dataset {
            let valid_batcher = SequenceBatcher::<B>::new(device.clone(), config.sequence_length);
            let valid_loss = compute_validation_loss(model, valid_ds, &valid_batcher, &loss_fn);
            state.current_valid_loss = Some(valid_loss);

            // Early stopping check
            if valid_loss < state.best_valid_loss {
                state.best_valid_loss = valid_loss;
                state.epochs_without_improvement = 0;
                improved = true;
            } else {
                state.epochs_without_improvement += 1;
                if state.epochs_without_improvement >= EARLY_STOPPING_PATIENCE {
                    log_progress(
                        epoch + 1,
                        state.current_train_loss,
                        state.current_valid_loss,
                    );
                    println!(
                        "Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement"
                    );

                    // Save final checkpoint
                    if let Some(ref ckpt_cfg) = checkpoint_config {
                        let path = format!("{}_final", ckpt_cfg.path_prefix);
                        if let Ok(_ckpt) = save_checkpoint(model, &path, config) {
                            println!("Saved final checkpoint: {path}");
                            checkpoint_paths.push(path);
                        }
                    }

                    return Ok(TrainingOutput {
                        final_train_loss: state.current_train_loss,
                        final_valid_loss: state.current_valid_loss,
                        epochs_completed: epoch + 1,
                        checkpoint_paths,
                    });
                }
            }
        }

        // Log progress every epoch for sequence models (fewer samples)
        log_progress(
            epoch + 1,
            state.current_train_loss,
            state.current_valid_loss,
        );

        // Checkpoint saving
        if let Some(ref ckpt_cfg) = checkpoint_config {
            let should_save_periodic =
                ckpt_cfg.save_every_n_epochs > 0 && (epoch + 1) % ckpt_cfg.save_every_n_epochs == 0;
            let should_save_improvement = ckpt_cfg.save_on_improvement && improved;

            if should_save_periodic || should_save_improvement {
                let suffix = if should_save_improvement && !should_save_periodic {
                    "best".to_string()
                } else {
                    format!("epoch{}", epoch + 1)
                };
                let path = format!("{}_{}", ckpt_cfg.path_prefix, suffix);

                match save_checkpoint(model, &path, config) {
                    Ok(_ckpt) => {
                        println!("Saved checkpoint: {path}");
                        checkpoint_paths.push(path);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to save checkpoint: {e}");
                    }
                }
            }
        }
    }

    // Save final checkpoint
    if let Some(ref ckpt_cfg) = checkpoint_config {
        let path = format!("{}_final", ckpt_cfg.path_prefix);
        if let Ok(_ckpt) = save_checkpoint(model, &path, config) {
            println!("Saved final checkpoint: {path}");
            checkpoint_paths.push(path);
        }
    }

    Ok(TrainingOutput {
        final_train_loss: state.current_train_loss,
        final_valid_loss: state.current_valid_loss,
        epochs_completed: config.epochs,
        checkpoint_paths,
    })
}

/// Computes the validation loss on a dataset.
fn compute_validation_loss<B: Backend>(
    model: &SequenceModel<B>,
    dataset: &SequenceDataset,
    batcher: &SequenceBatcher<B>,
    loss_fn: &MseLoss,
) -> f32 {
    let num_samples = dataset.len();
    if num_samples == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;
    let mut batch_count = 0;

    // Process in batches (use same size as training for consistency)
    const BATCH_SIZE: usize = 128;
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

/// Shuffles indices using a simple LCG-based shuffle.
fn shuffle_indices(indices: &mut [usize], seed: u64) {
    // Simple Fisher-Yates shuffle with LCG random
    let mut rng_state = seed.wrapping_add(12345);

    for i in (1..indices.len()).rev() {
        // LCG: state = (a * state + c) mod m
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = ((rng_state >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Logs training progress.
fn log_progress(epoch: usize, train_loss: f32, valid_loss: Option<f32>) {
    // Convert MSE to RMSE for more intuitive interpretation
    let train_rmse = train_loss.sqrt();
    if let Some(vl) = valid_loss {
        let valid_rmse = vl.sqrt();
        println!(
            "Epoch {epoch}: train_loss = {train_loss:.6} (RMSE: {train_rmse:.1} MMR), valid_loss = {vl:.6} (RMSE: {valid_rmse:.1} MMR)"
        );
    } else {
        println!("Epoch {epoch}: train_loss = {train_loss:.6} (RMSE: {train_rmse:.1} MMR)");
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use feature_extractor::FrameFeatures;

    use super::*;
    use crate::{ModelConfig, SequenceSample};

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_training() {
        let device = NdArrayDevice::default();
        let model_config = ModelConfig::new();
        let mut model: SequenceModel<TestBackend> = SequenceModel::new(&device, &model_config);

        // Create some training data - 20 game samples
        let mut data = SequenceTrainingData::new();
        for i in 0..20 {
            let mmr = (i as f32).mul_add(50.0, 1000.0);
            let frames: Vec<FrameFeatures> = (0..100).map(|_| FrameFeatures::default()).collect();

            data.add_sample(SequenceSample {
                frames,
                target_mmr: [mmr; 6],
            });
        }

        let config = TrainingConfig::new(model_config)
            .with_epochs(2)
            .with_batch_size(4)
            .with_sequence_length(50);

        let result = train(&mut model, &data, &config);
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let output = result.expect("Training should succeed");
        assert_eq!(output.epochs_completed, 2);
    }

    #[test]
    fn test_training_with_checkpoints() {
        let device = NdArrayDevice::default();
        let model_config = ModelConfig::new();
        let mut model: SequenceModel<TestBackend> = SequenceModel::new(&device, &model_config);

        // Create some training data - 20 game samples
        let mut data = SequenceTrainingData::new();
        for i in 0..20 {
            let mmr = (i as f32).mul_add(50.0, 1000.0);
            let frames: Vec<FrameFeatures> = (0..100).map(|_| FrameFeatures::default()).collect();

            data.add_sample(SequenceSample {
                frames,
                target_mmr: [mmr; 6],
            });
        }

        let config = TrainingConfig::new(model_config)
            .with_epochs(10)
            .with_batch_size(4)
            .with_sequence_length(50);

        // Create temp dir for checkpoints
        let temp_dir = std::env::temp_dir().join("ml_test_ckpt");
        let _ = std::fs::create_dir_all(&temp_dir);
        let path_prefix = temp_dir.join("test_model").to_string_lossy().to_string();

        let checkpoint_config = CheckpointConfig {
            path_prefix,
            save_every_n_epochs: 5,
            save_on_improvement: false,
        };

        let result =
            train_with_checkpoints(&mut model, &data, &config, Some(checkpoint_config), None);
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let output = result.expect("Training should succeed");
        assert_eq!(output.epochs_completed, 10);
        // Should have saved at epoch 5, 10 (final)
        assert!(
            !output.checkpoint_paths.is_empty(),
            "Should have saved checkpoints"
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_training_resume() {
        let device = NdArrayDevice::default();
        let model_config = ModelConfig::new();
        let mut model: SequenceModel<TestBackend> = SequenceModel::new(&device, &model_config);

        // Create some training data - 20 game samples
        let mut data = SequenceTrainingData::new();
        for i in 0..20 {
            let mmr = (i as f32).mul_add(50.0, 1000.0);
            let frames: Vec<FrameFeatures> = (0..100).map(|_| FrameFeatures::default()).collect();

            data.add_sample(SequenceSample {
                frames,
                target_mmr: [mmr; 6],
            });
        }

        let config = TrainingConfig::new(model_config)
            .with_epochs(10)
            .with_batch_size(4)
            .with_sequence_length(50);

        // Start from epoch 5
        let start_state = TrainingState::new(5);

        let result = train_with_checkpoints(&mut model, &data, &config, None, Some(start_state));
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let output = result.expect("Training should succeed");
        // Should complete epochs 5-9 (5 epochs total)
        assert_eq!(output.epochs_completed, 10);
    }

    #[test]
    fn test_shuffle_indices() {
        let mut indices: Vec<usize> = (0..10).collect();
        let original = indices.clone();

        shuffle_indices(&mut indices, 42);

        // Should be permuted (very unlikely to be the same)
        assert_ne!(indices, original, "Shuffle should change order");

        // Should contain the same elements
        indices.sort_unstable();
        assert_eq!(indices, original, "Shuffle should preserve elements");
    }
}
