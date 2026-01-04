//! Training logic for the sequence model.

use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Instant;

use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use tracing::{error, info};

use crate::dataset::{SegmentDataset, SequenceBatcher, SequenceDatasetItem};
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

    // Split data into training and validation sets BY GAME (not by segment)
    // This prevents data leakage where segments from the same game appear in both sets
    let (train_samples, valid_samples) = data.split(config.validation_split);

    if train_samples.is_empty() {
        return Err(anyhow::anyhow!("No training samples after split"));
    }

    // Create segment datasets - each game is split into multiple consecutive-frame segments
    // This generates segments on-the-fly to avoid massive memory usage
    let dataset = SegmentDataset::new(train_samples, config.sequence_length);
    let batcher = SequenceBatcher::<B>::new(device, config.sequence_length);

    // Create Arc for dataset (used for prefetching across epochs)
    let dataset_arc = Arc::new(dataset);

    // Create validation dataset if we have validation data
    let valid_dataset = if !valid_samples.is_empty() {
        Some(SegmentDataset::new(valid_samples, config.sequence_length))
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

    let valid_segments = valid_dataset.as_ref().map_or(0, Dataset::len);
    info!(
        "Starting training with {} games ({} train, {} valid)",
        data.len(),
        dataset_arc.game_count(),
        valid_dataset.as_ref().map_or(0, SegmentDataset::game_count)
    );
    info!(
        "Total segments: {} train, {} valid (segment_length={})",
        dataset_arc.len(),
        valid_segments,
        config.sequence_length
    );
    info!(
        "Batch size: {}, Learning rate: {}",
        config.batch_size, config.learning_rate
    );
    if start_epoch > 0 {
        info!("Resuming from epoch {}", start_epoch + 1);
    }
    if let Some(ckpt_cfg) = &checkpoint_config {
        info!(
            "Checkpoints will be saved every {} epochs to {}",
            ckpt_cfg.save_every_n_epochs, ckpt_cfg.path_prefix
        );
    }

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        let mut batch_processing_time = std::time::Duration::ZERO;
        let mut data_loading_time = std::time::Duration::ZERO;

        // Set up prefetching: load next batch while GPU processes current one
        let num_samples = dataset_arc.len();
        let mut indices: Vec<usize> = (0..num_samples).collect();

        // Shuffle indices using epoch as seed
        shuffle_indices(&mut indices, epoch as u64);

        // Clone config values needed in thread (they're Copy or cheap to clone)
        let batch_size = config.batch_size;

        let (prefetch_tx, prefetch_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();

        // Clone indices for prefetch thread (just Vec<usize>, cheap to clone)
        let indices_for_prefetch = indices.clone();

        // Spawn prefetch thread
        let dataset_for_thread = Arc::clone(&dataset_arc);
        let prefetch_thread = thread::spawn(move || {
            let mut next_batch_idx = 0;
            let total_batches = num_samples.div_ceil(batch_size);

            loop {
                // Wait for signal to prefetch next batch
                if prefetch_rx.recv().is_err() {
                    break; // Channel closed, exit thread
                }

                if next_batch_idx >= total_batches {
                    // Signal that we're done
                    _ = ready_tx.send(None);
                    break;
                }

                let batch_start = next_batch_idx * batch_size;
                let batch_end = (batch_start + batch_size).min(num_samples);
                next_batch_idx += 1;

                let Some(batch_indices) = indices_for_prefetch.get(batch_start..batch_end) else {
                    _ = ready_tx.send(None);
                    continue;
                };

                // Load batch items (CPU work)
                let items: Vec<SequenceDatasetItem> = batch_indices
                    .iter()
                    .filter_map(|&i| dataset_for_thread.get(i))
                    .collect();

                if items.is_empty() {
                    _ = ready_tx.send(None);
                    continue;
                }

                // Send loaded items to main thread
                _ = ready_tx.send(Some(items));
            }
        });

        // Process batches with prefetching
        let mut next_batch_items: Option<Vec<SequenceDatasetItem>> = None;
        let mut batch_idx = 0;
        let total_batches = num_samples.div_ceil(config.batch_size);

        // Start prefetching the first batch immediately
        if total_batches > 0
            && let Err(err) = prefetch_tx.send(())
        {
            error!("Failed to prefetch first batch: {err}");
        }

        loop {
            let batch_start_time = Instant::now();

            // Get current batch (prefetched if available, otherwise wait for it or load)
            let items = if let Some(items) = next_batch_items.take() {
                // Use prefetched batch (ideal case - no waiting!)
                items
            } else if batch_idx < total_batches {
                // Prefetch not ready yet - wait for it (should be rare after first batch)
                if let Ok(Some(items)) = ready_rx.recv() {
                    items
                } else {
                    // Prefetch failed or thread ended, load synchronously
                    let batch_start = batch_idx * batch_size;
                    let batch_end = (batch_start + batch_size).min(num_samples);
                    let Some(batch_indices) = indices.get(batch_start..batch_end) else {
                        break;
                    };

                    let data_load_start = Instant::now();
                    let items: Vec<_> = batch_indices
                        .iter()
                        .filter_map(|&i| dataset_arc.get(i))
                        .collect();
                    data_loading_time += data_load_start.elapsed();

                    if items.is_empty() {
                        break;
                    }
                    items
                }
            } else {
                // No more batches
                break;
            };

            // Start prefetching next batch (if not the last one)
            if batch_idx + 1 < total_batches {
                _ = prefetch_tx.send(());
            }

            // Update data loading time for prefetched batches (they were loaded in parallel)
            // Note: We don't track this separately for prefetched vs sync loads,
            // but the overall epoch time will show the improvement

            // Create batch on GPU
            let batch = batcher.batch(items);

            // Forward pass
            let predictions = model.forward(batch.inputs);
            let loss = loss_fn.forward(predictions, batch.targets, burn::nn::loss::Reduction::Mean);

            // Extract loss value for epoch average
            // Note: This GPU->CPU sync is expensive, but necessary for accurate epoch metrics
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
                info!("  Batch {batch_count}/{total_batches}, avg_loss = {avg_loss_so_far:.6}");
            }

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Update weights
            *model = optimizer.step(config.learning_rate, model.clone(), grads);
            batch_processing_time += batch_start_time.elapsed();

            batch_idx += 1;

            // Try to receive prefetched next batch (non-blocking)
            // This happens while GPU is processing, so if it's ready we save time
            if batch_idx < total_batches {
                match ready_rx.try_recv() {
                    Ok(Some(items)) => {
                        next_batch_items = Some(items);
                    }
                    Ok(None)
                    | Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {
                        // Prefetch returned empty, will load synchronously next iteration
                    }
                }
            }
        }

        // Clean up prefetch thread
        drop(prefetch_tx);
        _ = prefetch_thread.join();

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

        // Validation - use inner (non-autodiff) backend for faster inference
        let mut improved = false;
        let mut validation_time = std::time::Duration::ZERO;
        if let Some(valid_ds) = &valid_dataset {
            let valid_start = Instant::now();
            // Convert to inner backend to avoid gradient tracking overhead
            let inner_model = model.clone().valid();
            let inner_device = inner_model.linear1.weight.device();
            let valid_batcher =
                SequenceBatcher::<B::InnerBackend>::new(inner_device, config.sequence_length);
            let valid_loss =
                compute_validation_loss(&inner_model, valid_ds, &valid_batcher, config.batch_size);
            validation_time = valid_start.elapsed();
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

                    // Save final checkpoint
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
        }

        // Log progress every epoch for sequence models (fewer samples)
        log_progress(
            epoch + 1,
            state.current_train_loss,
            state.current_valid_loss,
        );

        // Log timing information
        info!(
            "  Timing: epoch={:.2}s, data_loading={:.2}s, batch_processing={:.2}s, validation={:.2}s",
            epoch_duration.as_secs_f64(),
            data_loading_time.as_secs_f64(),
            batch_processing_time.as_secs_f64(),
            validation_time.as_secs_f64()
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

/// Computes the validation loss on a segment dataset using the inner (non-autodiff) backend.
///
/// This is faster than using the autodiff backend since no computation graph is built.
fn compute_validation_loss<B: Backend>(
    model: &SequenceModel<B>,
    dataset: &SegmentDataset,
    batcher: &SequenceBatcher<B>,
    batch_size: usize,
) -> f32 {
    let num_segments = dataset.len();
    if num_segments == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;
    let mut total_samples = 0;

    // Use same batch size as training for consistent GPU utilization
    for batch_start in (0..num_segments).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_segments);

        let items: Vec<_> = (batch_start..batch_end)
            .filter_map(|i| dataset.get(i))
            .collect();

        if items.is_empty() {
            continue;
        }

        let num_items = items.len();
        let batch = batcher.batch(items);
        let predictions = model.forward(batch.inputs);

        // Compute MSE manually to avoid autodiff overhead
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

        // Weight by number of samples for accurate overall average
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
            .with_epochs(3)
            .with_batch_size(4)
            .with_sequence_length(50);

        // Create temp dir for checkpoints
        let temp_dir = std::env::temp_dir().join("ml_test_ckpt");
        let _create_dir = std::fs::create_dir_all(&temp_dir);
        let path_prefix = temp_dir.join("test_model").to_string_lossy().to_string();

        let checkpoint_config = CheckpointConfig {
            path_prefix,
            save_every_n_epochs: 1,
            save_on_improvement: false,
        };

        let result =
            train_with_checkpoints(&mut model, &data, &config, Some(checkpoint_config), None);
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let output = result.expect("Training should succeed");
        assert_eq!(output.epochs_completed, 3);
        // Should have saved at epoch 5, 10 (final)
        assert!(
            !output.checkpoint_paths.is_empty(),
            "Should have saved checkpoints"
        );

        // Cleanup
        let _cleanup = std::fs::remove_dir_all(&temp_dir);
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
            .with_epochs(3)
            .with_batch_size(4)
            .with_sequence_length(50);

        // Start from epoch 5
        let start_state = TrainingState::new(2);

        let result = train_with_checkpoints(&mut model, &data, &config, None, Some(start_state));
        assert!(result.is_ok(), "Training failed: {:?}", result.err());

        let output = result.expect("Training should succeed");
        // Should complete epochs 5-9 (5 epochs total)
        assert_eq!(output.epochs_completed, 3);
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
