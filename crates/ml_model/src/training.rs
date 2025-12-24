//! Training logic for the impact model.

use burn::data::dataset::Dataset;
use burn::nn::loss::MseLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::dataset::{ImpactBatcher, ImpactDataset};
use crate::{ImpactModel, TrainingConfig, TrainingData};

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

/// Trains the model on the provided data.
///
/// Uses a simple training loop with Adam optimizer and MSE loss.
///
/// # Arguments
///
/// * `model` - The model to train (will be modified in place).
/// * `data` - The training data.
/// * `config` - Training configuration.
///
/// # Errors
///
/// Returns an error if training fails.
pub fn train<B: AutodiffBackend>(
    model: &mut ImpactModel<B>,
    data: &TrainingData,
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
    let dataset = ImpactDataset::from_slice(&train_samples);
    let batcher = ImpactBatcher::<B>::new(device);

    // Create validation batcher if we have validation data
    let valid_dataset = if !valid_samples.is_empty() {
        Some(ImpactDataset::from_slice(&valid_samples))
    } else {
        None
    };

    // Create optimizer
    let mut optimizer = AdamConfig::new().init();

    let loss_fn = MseLoss::new();
    let mut final_train_loss = 0.0;
    let mut final_valid_loss: Option<f32> = None;
    let mut best_valid_loss = f32::MAX;
    let mut epochs_without_improvement = 0;
    const EARLY_STOPPING_PATIENCE: usize = 10;

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
            let valid_loss = compute_validation_loss(model, valid_ds, &batcher, &loss_fn);
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

        // Log progress every 10 epochs or at the end
        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            log_progress(epoch + 1, final_train_loss, final_valid_loss);
        }
    }

    Ok(TrainingOutput {
        final_train_loss,
        final_valid_loss,
        epochs_completed: config.epochs,
    })
}

/// Computes the validation loss on a dataset.
fn compute_validation_loss<B: Backend>(
    model: &ImpactModel<B>,
    dataset: &ImpactDataset,
    batcher: &ImpactBatcher<B>,
    loss_fn: &MseLoss,
) -> f32 {
    let num_samples = dataset.len();
    if num_samples == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;
    let mut batch_count = 0;

    // Process in batches of 64
    const BATCH_SIZE: usize = 64;
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
    if let Some(vl) = valid_loss {
        println!("Epoch {epoch}: train_loss = {train_loss:.6}, valid_loss = {vl:.6}");
    } else {
        println!("Epoch {epoch}: train_loss = {train_loss:.6}");
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use feature_extractor::FrameFeatures;

    use super::*;
    use crate::ModelConfig;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_training() {
        let device = NdArrayDevice::default();
        let model_config = ModelConfig::new();
        let mut model: ImpactModel<TestBackend> = ImpactModel::new(&device, &model_config);

        // Create some training data
        let mut data = TrainingData::new();
        for i in 0..100 {
            let mmr = (i as f32).mul_add(10.0, 1000.0);
            data.add_samples(vec![feature_extractor::TrainingSample {
                features: FrameFeatures::default(),
                target_mmr: vec![mmr; 6],
            }]);
        }

        let config = TrainingConfig::new(model_config)
            .with_epochs(2)
            .with_batch_size(16);

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
