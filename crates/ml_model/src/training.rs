//! Training logic for the sequence model.

use burn::data::dataset::Dataset;
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::dataset::{SequenceBatcher, SequenceDataset};
use crate::{SequenceModel, SequenceTrainingData, TrainingConfig};

/// Output from training.
#[derive(Debug, Clone)]
pub struct TrainingOutput {
    /// Final training loss.
    pub final_train_loss: f32,
    /// Final validation loss (if validation data was used).
    pub final_valid_loss: Option<f32>,
    /// Number of epochs completed.
    pub epochs_completed: usize,
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
        Some(SequenceDataset::new(
            &valid_samples,
            config.sequence_length,
        ))
    } else {
        None
    };

    // Create optimizer with weight decay for regularization
    let mut optimizer = AdamConfig::new().init();

    let loss_fn = MseLoss::new();
    let mut final_train_loss = 0.0;
    let mut final_valid_loss: Option<f32> = None;
    let mut best_valid_loss = f32::MAX;
    let mut epochs_without_improvement = 0;
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

    for epoch in 0..config.epochs {
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

            // Get loss value for logging
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

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);

            // Update weights
            *model = optimizer.step(config.learning_rate, model.clone(), grads);
        }

        final_train_loss = if batch_count > 0 {
            (epoch_loss / batch_count as f64) as f32
        } else {
            0.0
        };

        // Validation
        if let Some(valid_ds) = &valid_dataset {
            let valid_batcher = SequenceBatcher::<B>::new(device.clone(), config.sequence_length);
            let valid_loss = compute_validation_loss(model, valid_ds, &valid_batcher, &loss_fn);
            final_valid_loss = Some(valid_loss);

            // Early stopping check
            if valid_loss < best_valid_loss {
                best_valid_loss = valid_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE {
                    log_progress(epoch + 1, final_train_loss, final_valid_loss);
                    println!(
                        "Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement"
                    );
                    return Ok(TrainingOutput {
                        final_train_loss,
                        final_valid_loss,
                        epochs_completed: epoch + 1,
                    });
                }
            }
        }

        // Log progress every epoch for sequence models (fewer samples)
        log_progress(epoch + 1, final_train_loss, final_valid_loss);
    }

    Ok(TrainingOutput {
        final_train_loss,
        final_valid_loss,
        epochs_completed: config.epochs,
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

    // Process in batches of 32
    const BATCH_SIZE: usize = 32;
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
            let frames: Vec<FrameFeatures> = (0..100)
                .map(|_| FrameFeatures::default())
                .collect();

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
