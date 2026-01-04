//! Training logic for the sequence model.

use std::time::Instant;

use burn::module::AutodiffModule;
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tracing::info;

use crate::dataset::{MmapSegmentDataset, SequenceBatcher};
use crate::{SequenceModel, TrainingConfig, save_checkpoint};

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

/// Trains the sequence model using memory-mapped segment data.
///
/// This training method uses zero-copy access to cached feature segments.
/// The data must already be split into train/valid datasets before calling.
///
/// # Arguments
///
/// * `model` - The model to train (will be modified in place).
/// * `train_dataset` - Training data as memory-mapped segments.
/// * `valid_dataset` - Optional validation data as memory-mapped segments.
/// * `config` - Training configuration.
/// * `checkpoint_config` - Optional checkpoint configuration for saving during training.
/// * `start_state` - Optional starting state for resumption.
///
/// # Errors
///
/// Returns an error if training fails.
pub fn train<B: AutodiffBackend>(
    model: &mut SequenceModel<B>,
    train_dataset: &MmapSegmentDataset,
    valid_dataset: Option<&MmapSegmentDataset>,
    config: &TrainingConfig,
    checkpoint_config: Option<CheckpointConfig>,
    start_state: Option<TrainingState>,
) -> anyhow::Result<TrainingOutput>
where
    B::FloatElem: From<f32>,
{
    if train_dataset.is_empty() {
        return Err(anyhow::anyhow!("No training data provided"));
    }

    let device = model.linear1.weight.device();
    let batcher = SequenceBatcher::<B>::new(device, config.sequence_length);

    // Create optimizer
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();
    let mut checkpoint_paths = Vec::new();

    // Initialize training state
    let mut state = start_state.unwrap_or_else(|| TrainingState::new(0));
    let start_epoch = state.current_epoch;

    const EARLY_STOPPING_PATIENCE: usize = 10;

    let valid_segments = valid_dataset.map_or(0, MmapSegmentDataset::len);
    info!(
        "Starting training with {} train segments, {} valid segments",
        train_dataset.len(),
        valid_segments
    );
    info!(
        "Batch size: {}, Learning rate: {}, Segment length: {}",
        config.batch_size, config.learning_rate, config.sequence_length
    );
    if start_epoch > 0 {
        info!("Resuming from epoch {}", start_epoch + 1);
    }

    let num_samples = train_dataset.len();

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Create shuffled indices for this epoch
        let mut indices: Vec<usize> = (0..num_samples).collect();
        shuffle_indices(&mut indices, epoch as u64);

        // Process batches
        let total_batches = num_samples.div_ceil(config.batch_size);

        for batch_idx in 0..total_batches {
            let batch_start = batch_idx * config.batch_size;
            let batch_end = (batch_start + config.batch_size).min(num_samples);

            let Some(batch_indices) = indices.get(batch_start..batch_end) else {
                continue;
            };

            // Create batch directly from mmap indices
            let Some(batch) = batcher.batch_from_indices(train_dataset, batch_indices) else {
                continue;
            };

            // Forward pass
            let predictions = model.forward(batch.inputs);
            let loss = loss_fn.forward(predictions, batch.targets, burn::nn::loss::Reduction::Mean);

            // Extract loss value
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

            // Log progress every 20 batches
            if batch_count % 20 == 0 {
                let avg_loss_so_far = epoch_loss / batch_count as f64;
                info!("  Batch {batch_count}/{total_batches}, avg_loss = {avg_loss_so_far:.6}");
            }

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Update weights
            *model = optimizer.step(config.learning_rate, model.clone(), grads);
        }

        let epoch_duration = epoch_start.elapsed();
        state.current_train_loss = if batch_count > 0 {
            (epoch_loss / batch_count as f64) as f32
        } else {
            0.0
        };
        state.current_epoch = epoch;

        info!(
            "Epoch {epoch} completed in {:.2}s, avg_loss = {:.6}",
            epoch_duration.as_secs_f64(),
            state.current_train_loss
        );

        // Validation
        let mut improved = false;
        if let Some(valid_ds) = valid_dataset {
            let valid_start = Instant::now();
            let inner_model = model.clone().valid();
            let inner_device = inner_model.linear1.weight.device();
            let valid_batcher =
                SequenceBatcher::<B::InnerBackend>::new(inner_device, config.sequence_length);

            let valid_loss =
                compute_validation_loss(&inner_model, valid_ds, &valid_batcher, config.batch_size);
            let validation_time = valid_start.elapsed();
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
                    info!(
                        "Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement"
                    );

                    if let Some(ckpt_cfg) = &checkpoint_config {
                        let path = format!("{}_final", ckpt_cfg.path_prefix);
                        if let Ok(_ckpt) = save_checkpoint(model, &path, config) {
                            info!("Saved final checkpoint: {path}");
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

            info!("  Validation time: {:.2}s", validation_time.as_secs_f64());
        }

        log_progress(
            epoch + 1,
            state.current_train_loss,
            state.current_valid_loss,
        );

        // Checkpoint saving
        if let Some(ckpt_cfg) = &checkpoint_config {
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
                        info!("Saved checkpoint: {path}");
                        checkpoint_paths.push(path);
                    }
                    Err(e) => {
                        info!("Warning: Failed to save checkpoint: {e}");
                    }
                }
            }
        }
    }

    // Save final checkpoint
    if let Some(ckpt_cfg) = &checkpoint_config {
        let path = format!("{}_final", ckpt_cfg.path_prefix);
        if let Ok(_ckpt) = save_checkpoint(model, &path, config) {
            info!("Saved final checkpoint: {path}");
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

/// Computes validation loss on an mmap dataset.
fn compute_validation_loss<B: Backend>(
    model: &SequenceModel<B>,
    dataset: &MmapSegmentDataset,
    batcher: &SequenceBatcher<B>,
    batch_size: usize,
) -> f32 {
    let num_segments = dataset.len();
    if num_segments == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;
    let mut total_samples = 0;

    let indices: Vec<usize> = (0..num_segments).collect();

    for batch_start in (0..num_segments).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_segments);

        let Some(batch_indices) = indices.get(batch_start..batch_end) else {
            continue;
        };

        let Some(batch) = batcher.batch_from_indices(dataset, batch_indices) else {
            continue;
        };

        let num_items = batch_indices.len();
        let predictions = model.forward(batch.inputs);

        // Compute MSE manually
        let diff = predictions - batch.targets;
        let squared = diff.clone() * diff;
        let mse = squared.mean();

        let loss_value: f32 = mse
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![0.0])
            .first()
            .copied()
            .unwrap_or(0.0);

        total_loss += loss_value as f64 * num_items as f64;
        total_samples += num_items;
    }

    if total_samples > 0 {
        (total_loss / total_samples as f64) as f32
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
        info!(
            "Epoch {epoch}: train_loss = {train_loss:.6} (RMSE: {train_rmse:.1} MMR), valid_loss = {vl:.6} (RMSE: {valid_rmse:.1} MMR)"
        );
    } else {
        info!("Epoch {epoch}: train_loss = {train_loss:.6} (RMSE: {train_rmse:.1} MMR)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
