//! ML model crate for Rocket League player skill prediction.
//!
//! This crate uses the Burn deep learning framework with an LSTM-based
//! sequence model that analyzes temporal patterns in gameplay to predict
//! player MMR/skill level.
//!
//! ## Key Concepts
//!
//! Unlike frame-by-frame prediction, this model processes **sequences of frames**
//! from entire games or segments, learning temporal patterns like:
//! - Reaction speed and recovery times
//! - Consistency of mechanical execution
//! - Decision making patterns over time
//! - Rotation and positioning habits
//!
//! The model outputs one MMR prediction per player per game/segment.

mod dataset;
mod lazy_dataset;
mod training;

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
pub use dataset::{SegmentDataset, SequenceBatcher, SequenceDatasetItem};
use feature_extractor::{FEATURE_COUNT, FrameFeatures, TOTAL_PLAYERS};
pub use lazy_dataset::{GameLoader, GameMetadata, LazySegmentDataset};
pub use training::{
    CheckpointConfig, TrainingOutput, TrainingState, train, train_with_checkpoints,
    train_with_dataset,
};

/// Configuration for the sequence model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Hidden size for the first LSTM layer.
    #[config(default = 128)]
    pub lstm_hidden_1: usize,
    /// Hidden size for the second LSTM layer.
    #[config(default = 64)]
    pub lstm_hidden_2: usize,
    /// Hidden size for the feedforward layer after LSTM.
    #[config(default = 32)]
    pub feedforward_hidden: usize,
    /// Dropout rate for regularization.
    #[config(default = 0.5)]
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
    #[config(default = 2048)]
    pub batch_size: usize,
    /// Model architecture configuration.
    pub model: ModelConfig,
    /// Validation split ratio (0.0 to 1.0).
    #[config(default = 0.1)]
    pub validation_split: f64,
    /// Sequence length (number of frames per segment).
    /// At 30fps, 90 frames = 3 seconds of gameplay.
    /// Each game is split into non-overlapping segments of this length.
    #[config(default = 90)]
    pub sequence_length: usize,
}

/// LSTM-based sequence model for player skill prediction.
///
/// This model processes sequences of game frames to predict player MMR.
/// It uses stacked LSTM layers to capture temporal patterns in gameplay,
/// then projects the final hidden state to per-player MMR predictions.
#[derive(Module, Debug)]
pub struct SequenceModel<B: Backend> {
    /// First LSTM layer processes raw frame features.
    lstm1: Lstm<B>,
    /// Second LSTM layer for deeper temporal patterns.
    lstm2: Lstm<B>,
    /// Dropout for regularization.
    dropout: Dropout,
    /// First feedforward layer after LSTM.
    linear1: Linear<B>,
    /// Output layer predicting MMR for each player.
    linear_out: Linear<B>,
    /// `ReLU` activation.
    activation: Relu,
    /// Hidden size of first LSTM (needed for inference).
    lstm1_hidden: usize,
    /// Hidden size of second LSTM (needed for inference).
    lstm2_hidden: usize,
}

impl<B: Backend> SequenceModel<B> {
    /// Creates a new sequence model with the given configuration.
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        // LSTM layers
        let lstm1 = LstmConfig::new(FEATURE_COUNT, config.lstm_hidden_1, true).init(device);
        let lstm2 = LstmConfig::new(config.lstm_hidden_1, config.lstm_hidden_2, true).init(device);

        // Dropout
        let dropout = DropoutConfig::new(config.dropout).init();

        // Feedforward layers
        let linear1 =
            LinearConfig::new(config.lstm_hidden_2, config.feedforward_hidden).init(device);
        let linear_out = LinearConfig::new(config.feedforward_hidden, TOTAL_PLAYERS).init(device);

        let activation = Relu::new();

        Self {
            lstm1,
            lstm2,
            dropout,
            linear1,
            linear_out,
            activation,
            lstm1_hidden: config.lstm_hidden_1,
            lstm2_hidden: config.lstm_hidden_2,
        }
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape `[batch_size, seq_len, FEATURE_COUNT]`
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_size, 6]` containing predicted MMR for each player.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // LSTM1: [batch, seq, features] -> [batch, seq, lstm1_hidden]
        let (lstm1_out, _state1) = self.lstm1.forward(input, None);

        // LSTM2: [batch, seq, lstm1_hidden] -> [batch, seq, lstm2_hidden]
        let (lstm2_out, _state2) = self.lstm2.forward(lstm1_out, None);

        // Take the last timestep's hidden state: [batch, lstm2_hidden]
        // Use narrow to select the last timestep, then reshape to remove the seq dimension
        let [batch_size, seq_len, hidden_size] = lstm2_out.dims();
        let last_hidden = lstm2_out
            .narrow(1, seq_len - 1, 1)
            .reshape([batch_size, hidden_size]);

        // Dropout for regularization
        let x = self.dropout.forward(last_hidden);

        // Feedforward layers
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        // Output: [batch, 6]
        self.linear_out.forward(x)
    }

    /// Forward pass for inference (without dropout).
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape `[batch_size, seq_len, FEATURE_COUNT]`
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_size, 6]` containing predicted MMR for each player.
    pub fn forward_inference(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Same as forward but dropout is automatically disabled in eval mode
        // For Burn, dropout is controlled by the tensor's require_grad flag
        self.forward(input)
    }
}

/// A training sample representing one game/segment.
#[derive(Debug, Clone)]
pub struct SequenceSample {
    /// Sequence of frame features from the game.
    pub frames: Vec<FrameFeatures>,
    /// Target MMR for each of the 6 players.
    /// Order: blue team (3 players sorted by `actor_id`), then orange team (3 players).
    pub target_mmr: [f32; TOTAL_PLAYERS],
}

/// Training data container for sequence samples.
#[derive(Debug, Clone, Default)]
pub struct SequenceTrainingData {
    /// Collection of game/segment samples.
    pub samples: Vec<SequenceSample>,
}

impl SequenceTrainingData {
    /// Creates a new empty training data container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds a sample to the training data.
    pub fn add_sample(&mut self, sample: SequenceSample) {
        self.samples.push(sample);
    }

    /// Adds multiple samples to the training data.
    pub fn add_samples(&mut self, samples: Vec<SequenceSample>) {
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
    pub fn split(&self, validation_ratio: f64) -> (Vec<SequenceSample>, Vec<SequenceSample>) {
        let validation_count = (self.samples.len() as f64 * validation_ratio) as usize;
        let train_count = self.samples.len() - validation_count;

        let training = self.samples.iter().take(train_count).cloned().collect();
        let validation = self.samples.iter().skip(train_count).cloned().collect();

        (training, validation)
    }
}

/// Reference to a saved model checkpoint.
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// Path to the saved model.
    pub path: String,
    /// Model version.
    pub version: u32,
    /// Training configuration used.
    pub training_config: TrainingConfig,
}

/// Creates a new sequence model with the given configuration.
pub fn create_model<B: Backend>(device: &B::Device, config: &ModelConfig) -> SequenceModel<B> {
    SequenceModel::new(device, config)
}

/// Predicts MMR for all players from a sequence of frames.
///
/// This function splits the game into non-overlapping segments, predicts MMR
/// for each segment, and averages the predictions across all segments.
///
/// # Arguments
///
/// * `model` - The trained model.
/// * `frames` - All frame features from a game.
/// * `device` - The device to run inference on.
/// * `segment_length` - Number of consecutive frames per segment.
///
/// # Returns
///
/// Array of predicted MMR values for each of the 6 players (averaged across segments).
pub fn predict<B: Backend>(
    model: &SequenceModel<B>,
    frames: &[FrameFeatures],
    device: &B::Device,
    segment_length: usize,
) -> [f32; TOTAL_PLAYERS] {
    if frames.is_empty() {
        return [1000.0; TOTAL_PLAYERS];
    }

    // Calculate number of segments
    let num_segments = if frames.len() >= segment_length {
        frames.len() / segment_length
    } else {
        1
    };

    // Accumulate predictions across all segments
    let mut sum_predictions = [0.0f32; TOTAL_PLAYERS];
    let mut segment_count = 0;

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_length;
        let segment_frames = get_segment_frames(frames, start, segment_length);

        // Build input tensor [1, seg_len, FEATURE_COUNT]
        let mut input_data = Vec::with_capacity(segment_length * FEATURE_COUNT);
        for frame in &segment_frames {
            input_data.extend_from_slice(&frame.features);
        }

        let input = Tensor::<B, 1>::from_floats(input_data.as_slice(), device).reshape([
            1,
            segment_length,
            FEATURE_COUNT,
        ]);

        // Forward pass
        let output = model.forward_inference(input);

        // Extract values
        let output_data = output.into_data();
        if let Ok(values) = output_data.to_vec::<f32>() {
            for (i, val) in values.iter().take(TOTAL_PLAYERS).enumerate() {
                let Some(sum_prediction) = sum_predictions.get_mut(i) else {
                    tracing::error!("Sum predictions has less than {TOTAL_PLAYERS} players");
                    continue;
                };
                *sum_prediction += *val;
            }
            segment_count += 1;
        }
    }

    // Average predictions
    if segment_count > 0 {
        for pred in &mut sum_predictions {
            *pred /= segment_count as f32;
        }
        sum_predictions
    } else {
        [1000.0; TOTAL_PLAYERS]
    }
}

/// Gets a segment of consecutive frames, padding if necessary.
fn get_segment_frames(
    frames: &[FrameFeatures],
    start: usize,
    segment_length: usize,
) -> Vec<FrameFeatures> {
    let end = (start + segment_length).min(frames.len());
    let Some(segment) = frames.get(start..end) else {
        tracing::error!("No segment found for start: {start} and segment_length: {segment_length}");
        return vec![];
    };

    let mut segment = segment.to_vec();

    // Pad with last frame if needed
    if let Some(last) = segment.last().cloned() {
        while segment.len() < segment_length {
            segment.push(last.clone());
        }
    }

    segment
}

/// Predicts MMR with a sliding window, returning how predictions evolve over time.
///
/// This provides an "evolving score" that shows how the model's confidence
/// changes as more of the game is observed.
///
/// # Arguments
///
/// * `model` - The trained model.
/// * `frames` - All frame features from a game.
/// * `device` - The device to run inference on.
/// * `sequence_length` - The window size for predictions.
/// * `step_size` - How many frames to advance between predictions.
///
/// # Returns
///
/// Vector of (timestamp, `mmr_predictions`) tuples showing evolution over time.
pub fn predict_evolving<B: Backend>(
    model: &SequenceModel<B>,
    frames: &[FrameFeatures],
    device: &B::Device,
    sequence_length: usize,
    step_size: usize,
) -> Vec<EvolvingPrediction> {
    let mut predictions = Vec::new();

    if frames.len() < sequence_length {
        // Not enough frames for even one prediction
        let pred = predict(model, frames, device, sequence_length);
        let timestamp = frames.last().map_or(0.0, |f| f.time);
        predictions.push(EvolvingPrediction {
            timestamp,
            player_mmr: pred,
        });
        return predictions;
    }

    // Slide window through the game
    let mut start = 0;
    while start + sequence_length <= frames.len() {
        let Some(window) = frames.get(start..start + sequence_length) else {
            tracing::error!(
                "No window found for start: {start} and sequence_length: {sequence_length}"
            );
            break;
        };
        let pred = predict(model, window, device, sequence_length);
        let timestamp = window.last().map_or(0.0, |f| f.time);

        predictions.push(EvolvingPrediction {
            timestamp,
            player_mmr: pred,
        });

        start += step_size;
    }

    predictions
}

/// A prediction at a specific point in time during a game.
#[derive(Debug, Clone)]
pub struct EvolvingPrediction {
    /// Timestamp in the game (seconds).
    pub timestamp: f32,
    /// Predicted MMR for each player.
    pub player_mmr: [f32; TOTAL_PLAYERS],
}

/// Saves the model checkpoint to disk.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be saved.
pub fn save_checkpoint<B: Backend>(
    model: &SequenceModel<B>,
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
/// * `path` - The path to the checkpoint.
/// * `device` - The device to load the model on.
///
/// # Returns
///
/// The loaded model.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be loaded.
pub fn load_checkpoint<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<SequenceModel<B>> {
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
    let model = SequenceModel::new(device, &model_config);

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
        let _model: SequenceModel<TestBackend> = create_model(&device, &config);
    }

    #[test]
    fn test_model_forward() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: SequenceModel<TestBackend> = create_model(&device, &config);

        // Create a batch of 2 sequences, each with 100 frames
        let batch_size = 2;
        let seq_len = 100;
        let input = Tensor::<TestBackend, 3>::zeros([batch_size, seq_len, FEATURE_COUNT], &device);
        let output = model.forward(input);

        // Output should be [2, 6] (6 players per game)
        assert_eq!(output.dims(), [batch_size, TOTAL_PLAYERS]);
    }

    #[test]
    fn test_predict_single_game() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: SequenceModel<TestBackend> = create_model(&device, &config);

        // Create 500 frames of fake game data
        let frames: Vec<FrameFeatures> = (0..500)
            .map(|i| FrameFeatures {
                features: [0.0; FEATURE_COUNT],
                time: i as f32 * 0.033, // ~30fps
            })
            .collect();

        let scores = predict(&model, &frames, &device, 300);

        // Should get 6 scores (one per player)
        assert_eq!(scores.len(), TOTAL_PLAYERS);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn test_sequence_training_data() {
        let mut data = SequenceTrainingData::new();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);

        data.add_sample(SequenceSample {
            frames: vec![FrameFeatures::default()],
            target_mmr: [1000.0; TOTAL_PLAYERS],
        });

        assert!(!data.is_empty());
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_training_data_split() {
        let mut data = SequenceTrainingData::new();

        // Add 10 samples
        for i in 0..10 {
            data.add_sample(SequenceSample {
                frames: vec![FrameFeatures::default()],
                target_mmr: [i as f32 * 100.0; TOTAL_PLAYERS],
            });
        }

        let (train, val) = data.split(0.2);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    #[test]
    fn test_get_segment_frames() {
        let frames: Vec<FrameFeatures> = (0..100)
            .map(|i| FrameFeatures {
                features: [i as f32; FEATURE_COUNT],
                time: i as f32,
            })
            .collect();

        // Full segment
        let segment = get_segment_frames(&frames, 0, 30);
        assert_eq!(segment.len(), 30);
        assert!((segment[0].time - 0.0).abs() < f32::EPSILON);
        assert!((segment[29].time - 29.0).abs() < f32::EPSILON);

        // Second segment
        let segment = get_segment_frames(&frames, 30, 30);
        assert_eq!(segment.len(), 30);
        assert!((segment[0].time - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_segment_frames_padding() {
        let frames: Vec<FrameFeatures> = (0..50)
            .map(|i| FrameFeatures {
                features: [i as f32; FEATURE_COUNT],
                time: i as f32,
            })
            .collect();

        // Segment that needs padding
        let segment = get_segment_frames(&frames, 30, 30);
        assert_eq!(segment.len(), 30);

        // First 20 should be original frames (30-49)
        assert!((segment[0].time - 30.0).abs() < f32::EPSILON);
        assert!((segment[19].time - 49.0).abs() < f32::EPSILON);

        // Last 10 should be padded with frame 49
        assert!((segment[29].time - 49.0).abs() < f32::EPSILON);
    }
}
