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
//!
//! ## Player-Centric Architecture
//!
//! Each player gets their own sequence of features centered on their perspective.
//! This enables the model to learn individual skill patterns and produce
//! different predictions for each player.

mod dataset;
pub mod segment_cache;
mod training;

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
pub use dataset::{BatchPrefetcher, PreloadedBatchData, SequenceBatch, SequenceBatcher};
use feature_extractor::{
    PLAYER_CENTRIC_FEATURE_COUNT, PlayerCentricFrameFeatures, TOTAL_PLAYERS,
    extract_all_player_centric_features,
};
pub use training::{CheckpointConfig, TrainingOutput, TrainingState, train};

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
    /// At 30fps, 1800 frames = 60 seconds of gameplay.
    /// Each game is split into non-overlapping segments of this length.
    #[config(default = 150)]
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
    ///
    /// This model uses player-centric features and predicts one MMR value per forward pass.
    /// For batch processing of all 6 players, reshape input to [batch*6, seq, features].
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        // LSTM layers - use player-centric feature count
        let lstm1 =
            LstmConfig::new(PLAYER_CENTRIC_FEATURE_COUNT, config.lstm_hidden_1, true).init(device);
        let lstm2 = LstmConfig::new(config.lstm_hidden_1, config.lstm_hidden_2, true).init(device);

        // Dropout
        let dropout = DropoutConfig::new(config.dropout).init();

        // Feedforward layers
        let linear1 =
            LinearConfig::new(config.lstm_hidden_2, config.feedforward_hidden).init(device);
        // Output layer predicts 1 value (will be reshaped to [batch, 6] later)
        let linear_out = LinearConfig::new(config.feedforward_hidden, 1).init(device);

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

    /// Forward pass through the network with player-centric features.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape `[batch_size * 6_players, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`
    ///   Each of the 6 players has their own sequence of player-centric features.
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_size * 6, 1]` containing predicted MMR.
    /// Reshape to `[batch_size, 6]` after calling this function.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Input: [batch*6, seq, player_features]
        // LSTM1: [batch*6, seq, features] -> [batch*6, seq, lstm1_hidden]
        let (lstm1_out, _state1) = self.lstm1.forward(input, None);

        // LSTM2: [batch*6, seq, lstm1_hidden] -> [batch*6, seq, lstm2_hidden]
        let (lstm2_out, _state2) = self.lstm2.forward(lstm1_out, None);

        // Take the last timestep's hidden state: [batch*6, lstm2_hidden]
        let [batch_times_players, seq_len, hidden_size] = lstm2_out.dims();
        let last_hidden = lstm2_out
            .narrow(1, seq_len - 1, 1)
            .reshape([batch_times_players, hidden_size]);

        // Dropout for regularization
        let x = self.dropout.forward(last_hidden);

        // Feedforward layers
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        // Output: [batch*6, 1]
        self.linear_out.forward(x)
    }

    /// Forward pass for inference (without dropout).
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape `[batch_size * 6_players, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_size * 6, 1]` containing predicted MMR.
    /// Reshape to `[batch_size, 6]` after calling this function.
    pub fn forward_inference(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Same as forward but dropout is automatically disabled in eval mode
        // For Burn, dropout is controlled by the tensor's require_grad flag
        self.forward(input)
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

// ============================================================================
// Player-Centric Data Structures
// ============================================================================

/// A training sample with player-centric sequences.
#[derive(Debug, Clone)]
pub struct PlayerCentricSequenceSample {
    /// Player-centric frame sequences: [num_frames, 6_players]
    /// Each frame contains features for all 6 players from their perspectives
    pub player_frames: Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]>,
    /// Target MMR for each of the 6 players.
    pub target_mmr: [f32; TOTAL_PLAYERS],
}

/// Training data container for player-centric samples.
#[derive(Debug, Clone, Default)]
pub struct PlayerCentricSequenceTrainingData {
    /// Collection of game/segment samples with player-centric features.
    pub samples: Vec<PlayerCentricSequenceSample>,
}

impl PlayerCentricSequenceTrainingData {
    /// Creates a new empty training data container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds a sample to the training data.
    pub fn add_sample(&mut self, sample: PlayerCentricSequenceSample) {
        self.samples.push(sample);
    }

    /// Adds multiple samples to the training data.
    pub fn add_samples(&mut self, samples: Vec<PlayerCentricSequenceSample>) {
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
    pub fn split(
        &self,
        validation_ratio: f64,
    ) -> (
        Vec<PlayerCentricSequenceSample>,
        Vec<PlayerCentricSequenceSample>,
    ) {
        let validation_count = (self.samples.len() as f64 * validation_ratio) as usize;
        let train_count = self.samples.len() - validation_count;

        let training = self.samples.iter().take(train_count).cloned().collect();
        let validation = self.samples.iter().skip(train_count).cloned().collect();

        (training, validation)
    }
}

/// Creates a new sequence model with the given configuration.
pub fn create_model<B: Backend>(device: &B::Device, config: &ModelConfig) -> SequenceModel<B> {
    SequenceModel::new(device, config)
}

/// Predicts MMR for all players using player-centric features.
///
/// This function extracts player-centric features, splits the game into segments,
/// predicts MMR for each player in each segment, and averages across segments.
///
/// # Arguments
///
/// * `model` - The trained model.
/// * `frames` - All game frames (will be converted to player-centric features).
/// * `device` - The device to run inference on.
/// * `segment_length` - Number of consecutive frames per segment.
///
/// # Returns
///
/// Array of predicted MMR values for each of the 6 players (averaged across segments).
pub fn predict_player_centric<B: Backend>(
    model: &SequenceModel<B>,
    frames: &[replay_structs::GameFrame],
    device: &B::Device,
    segment_length: usize,
) -> [f32; TOTAL_PLAYERS] {
    if frames.is_empty() {
        return [1000.0; TOTAL_PLAYERS];
    }

    // Extract player-centric features for all frames
    let player_centric_frames: Vec<[PlayerCentricFrameFeatures; 6]> = frames
        .iter()
        .map(extract_all_player_centric_features)
        .collect();

    let num_segments = if player_centric_frames.len() >= segment_length {
        player_centric_frames.len() / segment_length
    } else {
        1
    };

    // Accumulate predictions per player
    let mut sum_predictions = [0.0f32; TOTAL_PLAYERS];
    let mut segment_count = 0;

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_length;
        let segment_frames =
            get_segment_player_frames(&player_centric_frames, start, segment_length);

        // Build input tensor [6_players, seg_len, PLAYER_CENTRIC_FEATURE_COUNT]
        let mut input_data = Vec::with_capacity(6 * segment_length * PLAYER_CENTRIC_FEATURE_COUNT);

        for player_idx in 0..6 {
            for frame_features in &segment_frames {
                if let Some(player_features) = frame_features.get(player_idx) {
                    input_data.extend_from_slice(&player_features.features);
                }
            }
        }

        let input = Tensor::<B, 1>::from_floats(input_data.as_slice(), device).reshape([
            6,
            segment_length,
            PLAYER_CENTRIC_FEATURE_COUNT,
        ]);

        // Forward pass - output is [6, 1]
        let output = model.forward_inference(input);

        // Extract predictions
        let output_data = output.into_data();
        if let Ok(values) = output_data.to_vec::<f32>() {
            for (i, val) in values.iter().enumerate().take(TOTAL_PLAYERS) {
                if let Some(pred) = sum_predictions.get_mut(i) {
                    *pred += *val;
                }
            }
            segment_count += 1;
        }
    }

    // Average across segments
    if segment_count > 0 {
        for pred in &mut sum_predictions {
            *pred /= segment_count as f32;
        }
    }

    sum_predictions
}

/// Gets a segment of consecutive player-centric frames, padding if necessary.
fn get_segment_player_frames(
    frames: &[[PlayerCentricFrameFeatures; 6]],
    start: usize,
    segment_length: usize,
) -> Vec<[PlayerCentricFrameFeatures; 6]> {
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

    // Save model weights
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {e}"))?;

    // Save config alongside the model
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
/// * `path` - Path to the model checkpoint file (without extension)
/// * `device` - The device to load the model onto
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be loaded.
pub fn load_checkpoint<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<SequenceModel<B>> {
    // Try to load config first (for model dimensions)
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

        // Create a batch of 2 games × 6 players, each with 100 frames
        // Input shape: [batch*6, seq_len, player_features]
        let batch_size = 2;
        let seq_len = 100;
        let input = Tensor::<TestBackend, 3>::zeros(
            [batch_size * 6, seq_len, PLAYER_CENTRIC_FEATURE_COUNT],
            &device,
        );
        let output = model.forward(input);

        // Output should be [batch*6, 1] = [12, 1]
        assert_eq!(output.dims(), [batch_size * 6, 1]);

        // After reshaping to [batch, 6], we'd have [2, 6]
        let reshaped = output.reshape([batch_size, TOTAL_PLAYERS]);
        assert_eq!(reshaped.dims(), [batch_size, TOTAL_PLAYERS]);
    }

    #[test]
    fn test_predict_with_player_frames() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let config = ModelConfig::new();
        let model: SequenceModel<TestBackend> = create_model(&device, &config);

        // Create 500 frames of fake player-centric data
        let player_frames: Vec<[PlayerCentricFrameFeatures; 6]> = (0..500)
            .map(|i| {
                [
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                    PlayerCentricFrameFeatures {
                        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
                        time: i as f32 * 0.033,
                    },
                ]
            })
            .collect();

        // Test using internal get_segment_player_frames
        let segment_frames = get_segment_player_frames(&player_frames, 0, 300);

        // Build input tensor manually
        let segment_length = 300;
        let mut input_data = Vec::with_capacity(6 * segment_length * PLAYER_CENTRIC_FEATURE_COUNT);

        for player_idx in 0..6 {
            for frame_features in &segment_frames {
                if let Some(player_features) = frame_features.get(player_idx) {
                    input_data.extend_from_slice(&player_features.features);
                }
            }
        }

        let input = Tensor::<TestBackend, 1>::from_floats(input_data.as_slice(), &device)
            .reshape([6, segment_length, PLAYER_CENTRIC_FEATURE_COUNT]);

        let output = model.forward_inference(input);

        // Should get [6, 1] output
        assert_eq!(output.dims(), [6, 1]);

        // Verify all predictions are finite
        let values = output.into_data().to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 6);
        assert!(values.iter().all(|s| s.is_finite()));
    }

    fn create_default_player_frame() -> [PlayerCentricFrameFeatures; 6] {
        [
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
        ]
    }

    #[test]
    fn test_player_centric_training_data() {
        let mut data = PlayerCentricSequenceTrainingData::new();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);

        data.add_sample(PlayerCentricSequenceSample {
            player_frames: vec![create_default_player_frame()],
            target_mmr: [1000.0; TOTAL_PLAYERS],
        });

        assert!(!data.is_empty());
        assert_eq!(data.len(), 1);
    }

    #[test]
    fn test_training_data_split() {
        let mut data = PlayerCentricSequenceTrainingData::new();

        // Add 10 samples
        for i in 0..10 {
            data.add_sample(PlayerCentricSequenceSample {
                player_frames: vec![create_default_player_frame()],
                target_mmr: [i as f32 * 100.0; TOTAL_PLAYERS],
            });
        }

        let (train, val) = data.split(0.2);
        assert_eq!(train.len(), 8);
        assert_eq!(val.len(), 2);
    }

    fn create_player_frame_with_time(time: f32) -> [PlayerCentricFrameFeatures; 6] {
        [
            PlayerCentricFrameFeatures {
                features: [time; PLAYER_CENTRIC_FEATURE_COUNT],
                time,
            },
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
            PlayerCentricFrameFeatures::default(),
        ]
    }

    #[test]
    fn test_get_segment_player_frames() {
        let frames: Vec<[PlayerCentricFrameFeatures; 6]> = (0..100)
            .map(|i| create_player_frame_with_time(i as f32))
            .collect();

        // Full segment
        let segment = get_segment_player_frames(&frames, 0, 30);
        assert_eq!(segment.len(), 30);
        assert!((segment[0][0].time - 0.0).abs() < f32::EPSILON);
        assert!((segment[29][0].time - 29.0).abs() < f32::EPSILON);

        // Second segment
        let segment = get_segment_player_frames(&frames, 30, 30);
        assert_eq!(segment.len(), 30);
        assert!((segment[0][0].time - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_segment_player_frames_padding() {
        let frames: Vec<[PlayerCentricFrameFeatures; 6]> = (0..50)
            .map(|i| create_player_frame_with_time(i as f32))
            .collect();

        // Segment that needs padding
        let segment = get_segment_player_frames(&frames, 30, 30);
        assert_eq!(segment.len(), 30);

        // First 20 should be original frames (30-49)
        assert!((segment[0][0].time - 30.0).abs() < f32::EPSILON);
        assert!((segment[19][0].time - 49.0).abs() < f32::EPSILON);

        // Last 10 should be padded with frame 49
        assert!((segment[29][0].time - 49.0).abs() < f32::EPSILON);
    }
}
