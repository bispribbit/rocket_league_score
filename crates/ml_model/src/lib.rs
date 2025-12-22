//! ML model crate for Rocket League impact score prediction.
//!
//! This crate uses the Burn deep learning framework to define, train,
//! and run inference with a neural network that predicts player impact
//! scores based on frame features.

mod dataset;
mod training;

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
pub use dataset::{ImpactBatcher, ImpactDataset, ImpactDatasetItem};
use feature_extractor::{FEATURE_COUNT, FrameFeatures, TrainingSample};
pub use training::{TrainingOutput, train};

/// Configuration for the impact score model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Number of hidden units in the first layer.
    #[config(default = 256)]
    pub hidden_size_1: usize,
    /// Number of hidden units in the second layer.
    #[config(default = 128)]
    pub hidden_size_2: usize,
    /// Dropout rate for regularization.
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Configuration for training the model.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer.
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    /// Number of training epochs.
    #[config(default = 100)]
    pub epochs: usize,
    /// Batch size for training.
    #[config(default = 64)]
    pub batch_size: usize,
    /// Model architecture configuration.
    pub model: ModelConfig,
    /// Validation split ratio (0.0 to 1.0).
    #[config(default = 0.1)]
    pub validation_split: f64,
}

/// The impact score prediction model.
///
/// A simple feedforward neural network that takes frame features
/// as input and outputs a predicted MMR/impact score.
#[derive(Module, Debug)]
pub struct ImpactModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear_out: Linear<B>,
    activation: Relu,
}

impl<B: Backend> ImpactModel<B> {
    /// Creates a new impact model with the given configuration.
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let linear1 = LinearConfig::new(FEATURE_COUNT, config.hidden_size_1).init(device);
        let linear2 = LinearConfig::new(config.hidden_size_1, config.hidden_size_2).init(device);
        let linear_out = LinearConfig::new(config.hidden_size_2, 1).init(device);
        let activation = Relu::new();

        Self {
            linear1,
            linear2,
            linear_out,
            activation,
        }
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [`batch_size`, `FEATURE_COUNT`]
    ///
    /// # Returns
    ///
    /// Tensor of shape [`batch_size`, 1] containing predicted impact scores.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        self.linear_out.forward(x)
    }
}

/// Reference to a saved model checkpoint.
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    pub path: String,
    pub version: u32,
    pub training_config: TrainingConfig,
}

/// Training data container.
#[derive(Debug, Clone, Default)]
pub struct TrainingData {
    pub samples: Vec<TrainingSample>,
}

impl TrainingData {
    /// Creates a new empty training data container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds samples to the training data.
    pub fn add_samples(&mut self, samples: Vec<TrainingSample>) {
        self.samples.extend(samples);
    }

    /// Returns the number of samples.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if there are no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Splits the data into training and validation sets.
    ///
    /// # Arguments
    ///
    /// * `validation_ratio` - Fraction of data to use for validation (0.0 to 1.0)
    #[must_use]
    pub fn split(&self, validation_ratio: f64) -> (Vec<TrainingSample>, Vec<TrainingSample>) {
        let validation_count = (self.samples.len() as f64 * validation_ratio) as usize;
        let train_count = self.samples.len() - validation_count;

        let training = self.samples.iter().take(train_count).cloned().collect();
        let validation = self.samples.iter().skip(train_count).cloned().collect();

        (training, validation)
    }
}

/// Creates a new model with the given configuration.
///
/// # Arguments
///
/// * `device` - The device to create the model on.
/// * `config` - The model configuration.
///
/// # Returns
///
/// A new `ImpactModel` instance.
pub fn create_model<B: Backend>(device: &B::Device, config: &ModelConfig) -> ImpactModel<B> {
    ImpactModel::new(device, config)
}

/// Predicts the impact score for a single frame.
///
/// # Arguments
///
/// * `model` - The trained model.
/// * `features` - The frame features to predict on.
/// * `device` - The device to run inference on.
///
/// # Returns
///
/// The predicted impact score (MMR-like value).
pub fn predict<B: Backend>(
    model: &ImpactModel<B>,
    features: &FrameFeatures,
    device: &B::Device,
) -> f32 {
    // Convert features to tensor [1, FEATURE_COUNT]
    let input_data: Vec<f32> = features.features.to_vec();
    let input = Tensor::<B, 1>::from_floats(input_data.as_slice(), device).unsqueeze();

    // Forward pass - output is [1, 1]
    let output = model.forward(input);

    // Extract scalar value using into_data
    let output_data = output.into_data();
    let values = output_data.to_vec::<f32>().unwrap_or_else(|_| vec![1000.0]);

    values.first().copied().unwrap_or(1000.0)
}

/// Predicts impact scores for a batch of frames.
///
/// # Arguments
///
/// * `model` - The trained model.
/// * `features` - Vector of frame features to predict on.
/// * `device` - The device to run inference on.
///
/// # Returns
///
/// Vector of predicted impact scores.
pub fn predict_batch<B: Backend>(
    model: &ImpactModel<B>,
    features: &[FrameFeatures],
    device: &B::Device,
) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }

    // Build input tensor [batch_size, FEATURE_COUNT]
    let batch_size = features.len();
    let mut input_data = Vec::with_capacity(batch_size * FEATURE_COUNT);

    for frame in features {
        input_data.extend_from_slice(&frame.features);
    }

    let input = Tensor::<B, 1>::from_floats(input_data.as_slice(), device)
        .reshape([batch_size, FEATURE_COUNT]);

    // Forward pass
    let output = model.forward(input);

    // Extract values
    let output_data = output.into_data();
    output_data
        .to_vec::<f32>()
        .unwrap_or_else(|_| vec![1000.0; batch_size])
}

/// Saves the model checkpoint to disk.
///
/// # Arguments
///
/// * `model` - The model to save.
/// * `path` - The path to save to (without extension).
/// * `config` - The training configuration used.
///
/// # Errors
///
/// Returns an error if saving fails.
pub fn save_checkpoint<B: Backend>(
    model: &ImpactModel<B>,
    path: &str,
    config: &TrainingConfig,
) -> anyhow::Result<ModelCheckpoint> {
    // Create parent directory if needed
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Use MessagePack recorder for efficient serialization
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {e}"))?;

    // Also save the config
    let config_path = format!("{path}.config.json");
    config
        .save(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to save config: {e}"))?;

    Ok(ModelCheckpoint {
        path: path.to_string(),
        version: 1,
        training_config: config.clone(),
    })
}

/// Loads a model checkpoint from disk.
///
/// # Arguments
///
/// * `path` - The path to load from (without extension).
/// * `device` - The device to load the model to.
///
/// # Errors
///
/// Returns an error if loading fails.
pub fn load_checkpoint<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<ImpactModel<B>> {
    // Try to load config first, fall back to defaults
    let config_path = format!("{path}.config.json");
    let model_config = if Path::new(&config_path).exists() {
        let training_config = TrainingConfig::load(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config: {e}"))?;
        training_config.model
    } else {
        ModelConfig::new()
    };

    // Create model with config
    let model = ImpactModel::new(device, &model_config);

    // Load weights
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(path, &recorder, device)
        .map_err(|e| anyhow::anyhow!("Failed to load model weights: {e}"))?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn test_model_creation() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let _model: ImpactModel<TestBackend> = create_model(&device, &config);
    }

    #[test]
    fn test_model_forward() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: ImpactModel<TestBackend> = create_model(&device, &config);

        // Create a batch of 2 samples
        let input = Tensor::<TestBackend, 2>::zeros([2, FEATURE_COUNT], &device);
        let output = model.forward(input);

        // Output should be [2, 1]
        assert_eq!(output.dims(), [2, 1]);
    }

    #[test]
    fn test_predict_single() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: ImpactModel<TestBackend> = create_model(&device, &config);

        let features = FrameFeatures::default();
        let score = predict(&model, &features, &device);

        // Score should be a reasonable number (model is untrained, so just check it's finite)
        assert!(score.is_finite());
    }

    #[test]
    fn test_predict_batch() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: ImpactModel<TestBackend> = create_model(&device, &config);

        let features = vec![FrameFeatures::default(), FrameFeatures::default()];
        let scores = predict_batch(&model, &features, &device);

        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_training_data() {
        let mut data = TrainingData::new();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);

        data.add_samples(vec![]);
        assert!(data.is_empty());
    }

    #[test]
    fn test_training_data_split() {
        let mut data = TrainingData::new();

        // Add 10 samples
        for i in 0..10 {
            data.add_samples(vec![TrainingSample {
                features: FrameFeatures::default(),
                player_ratings: vec![],
                target_mmr: i as f32 * 100.0,
            }]);
        }

        let (train, val) = data.split(0.2);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::new(ModelConfig::new());
        assert!(config.learning_rate > 0.0);
        assert!(config.epochs > 0);
        assert!(config.batch_size > 0);
    }
}
