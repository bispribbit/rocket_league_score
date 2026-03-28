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

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Lstm, LstmConfig, Relu};
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use feature_extractor::{
    FRAME_SUBSAMPLE_RATE, PLAYER_CENTRIC_FEATURE_COUNT, PlayerCentricFrameFeatures, PlayerRoster,
    TOTAL_PLAYERS, extract_all_player_centric_features,
};

/// Scale factor to normalise MMR values to [0, 1] range.
/// Raw MMR range is approximately 0 – 2000, so dividing by this constant
/// produces normalised values suitable for the model output layer.
pub const MMR_SCALE: f32 = 2000.0;
/// Configuration for the sequence model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Hidden size for the first LSTM layer.
    ///
    /// Reduced from 256 to 128 to better match the dataset size (~234K segments).
    /// At 256 the model had ~1.25M params vs ~1.27M effective training examples
    /// (ratio ~1:1).  At 128 the model has ~350K params, giving a healthier ~3.6:1
    /// ratio that reduces overfitting risk.
    #[config(default = 128)]
    pub lstm_hidden_1: usize,
    /// Hidden size for the second LSTM layer.
    ///
    /// Scaled down from 128 to 64 proportionally with `lstm_hidden_1`.
    #[config(default = 64)]
    pub lstm_hidden_2: usize,
    /// Hidden size for the per-player feedforward layer.
    /// Input to this layer is `lstm_hidden_2 * 2` (last hidden + mean pooling concatenated).
    #[config(default = 64)]
    pub feedforward_hidden: usize,
    /// Hidden size for the per-player prediction head MLP.
    /// Each player's feedforward representation passes through this head
    /// independently to produce one MMR scalar per player.
    #[config(default = 32)]
    pub player_head_hidden: usize,
    /// Dropout rate for regularization.
    #[config(default = 0.2)]
    pub dropout: f64,
}

/// Configuration for training the model.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Learning rate for the optimizer.
    ///
    /// Overfit sweep (100 segments, 30 epochs, MSE + Huber grid):
    ///   lr=1e-4 → FLAT (0.8% improvement)
    ///   lr=1e-3 → FLAT (0.8%)
    ///   lr=1e-2 → OK   (64.9%, 1226→430 MMR RMSE)  <-- chosen
    ///   lr=5e-2 → OK   (64.7%, similar final RMSE but less stable)
    ///   lr=1e-1 → OK   (64.6%, oscillates more)
    #[config(default = 1e-2)]
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

/// LSTM-based sequence model with a per-player prediction head.
///
/// Architecture:
/// 1. Per-player LSTM stack processes each player's sequence independently.
/// 2. Temporal pooling: concatenates last hidden state + mean over sequence.
/// 3. Per-player feedforward reduces the pooled representation.
/// 4. Per-player head: shared-weight MLP applied independently to each player's
///    representation, producing one MMR scalar per player (no cross-lobby mixing).
///
/// The final output is `[batch_size, 6]` MMR predictions.
#[derive(Module, Debug)]
pub struct SequenceModel<B: Backend> {
    /// First LSTM layer processes raw frame features.
    lstm1: Lstm<B>,
    /// Second LSTM layer for deeper temporal patterns.
    lstm2: Lstm<B>,
    /// Dropout for regularization.
    dropout: Dropout,
    /// Per-player feedforward: maps temporal pooling output to player representation.
    /// Input size: `lstm2_hidden * 2` (last + mean concatenated).
    player_fc1: Linear<B>,
    /// Per-player prediction head layer 1: `feedforward_hidden -> player_head_hidden`.
    /// Applied independently to each player (shared weights across all 6 slots).
    player_head_fc: Linear<B>,
    /// Per-player prediction head output: `player_head_hidden -> 1`.
    player_head_out: Linear<B>,
    /// `ReLU` activation.
    activation: Relu,
    /// Hidden size of second LSTM (used for dimension tracking in forward).
    lstm2_hidden: usize,
}

impl<B: Backend> SequenceModel<B> {
    /// Creates a new sequence model with the given configuration.
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let lstm1 =
            LstmConfig::new(PLAYER_CENTRIC_FEATURE_COUNT, config.lstm_hidden_1, true).init(device);
        let lstm2 = LstmConfig::new(config.lstm_hidden_1, config.lstm_hidden_2, true).init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        // Temporal pooling concatenates last hidden state + mean → 2 * lstm2_hidden input.
        let player_fc1 =
            LinearConfig::new(config.lstm_hidden_2 * 2, config.feedforward_hidden).init(device);

        // Per-player prediction head: produces one MMR scalar per player.
        let player_head_fc =
            LinearConfig::new(config.feedforward_hidden, config.player_head_hidden).init(device);
        let player_head_out = LinearConfig::new(config.player_head_hidden, 1).init(device);

        Self {
            lstm1,
            lstm2,
            dropout,
            player_fc1,
            player_head_fc,
            player_head_out,
            activation: Relu::new(),
            lstm2_hidden: config.lstm_hidden_2,
        }
    }

    /// Forward pass through the network with player-centric features.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape `[batch_size * 6, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`.
    ///   Players from the same game must be contiguous in the batch dimension
    ///   (i.e. indices `[game*6 .. game*6+6]` belong to the same game).
    ///
    /// # Returns
    ///
    /// Tensor of shape `[batch_size, 6]` with one MMR prediction per player (raw units).
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_times_players, seq_len, _] = input.dims();
        let batch_size = batch_times_players / TOTAL_PLAYERS;

        // LSTM stack: each player's sequence is processed independently.
        // [batch*6, seq, features] → [batch*6, seq, lstm1_hidden]
        let (lstm1_out, _) = self.lstm1.forward(input, None);
        // [batch*6, seq, lstm1_hidden] → [batch*6, seq, lstm2_hidden]
        let (lstm2_out, _) = self.lstm2.forward(lstm1_out, None);

        // Temporal pooling: concat last timestep + mean over the whole sequence.
        // Both carry different signal: last captures the final game state,
        // mean captures the average behaviour across the segment.
        let last_hidden = lstm2_out
            .clone()
            .narrow(1, seq_len - 1, 1)
            .reshape([batch_times_players, self.lstm2_hidden]);
        let mean_hidden = lstm2_out
            .mean_dim(1)
            .reshape([batch_times_players, self.lstm2_hidden]);
        // [batch*6, lstm2_hidden*2]
        let pooled = Tensor::cat(vec![last_hidden, mean_hidden], 1);

        // Per-player feedforward: compress pooled representation.
        let x = self.dropout.forward(pooled);
        let x = self.player_fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        // x: [batch*6, feedforward_hidden]

        // Per-player head (independent per player, shared weights).
        // [batch*6, feedforward_hidden] → [batch*6, player_head_hidden] → [batch*6, 1]
        let player_pred = self.player_head_fc.forward(x);
        let player_pred = self.activation.forward(player_pred);
        let player_pred = self.player_head_out.forward(player_pred);
        // [batch*6, 1] → [batch, 6]
        player_pred.reshape([batch_size, TOTAL_PLAYERS])
    }

    /// Forward pass for inference (dropout disabled in eval mode).
    pub fn forward_inference(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward(input)
    }

    /// Returns the device this model is on.
    #[must_use]
    pub fn device(&self) -> B::Device {
        self.player_fc1.weight.device()
    }
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

/// Maps half-open subsampled feature indices to half-open indices in the original replay.
///
/// Training segments are built from frames taken with [`FRAME_SUBSAMPLE_RATE`] (see
/// [`feature_extractor::extract_player_centric_game_sequence`]).
fn subsampled_bounds_to_original_frames(
    start_subsampled: usize,
    end_subsampled_exclusive: usize,
    original_frame_count: usize,
) -> (usize, usize) {
    if original_frame_count == 0 {
        return (0, 0);
    }
    let step = FRAME_SUBSAMPLE_RATE;
    let start_original = (start_subsampled.saturating_mul(step)).min(original_frame_count - 1);
    let mut end_original_exclusive = end_subsampled_exclusive
        .saturating_mul(step)
        .min(original_frame_count);
    if end_original_exclusive <= start_original {
        end_original_exclusive = (start_original + 1).min(original_frame_count);
    }
    (start_original, end_original_exclusive)
}

/// Result of a single segment prediction.
#[derive(Debug, Clone)]
pub struct SegmentPrediction {
    /// Zero-based segment index.
    pub segment_index: usize,
    /// Start index in the **original** replay frame list (same convention as training cache).
    pub start_frame: usize,
    /// Exclusive end index in the **original** replay frame list.
    pub end_frame: usize,
    /// Predicted MMR for each of the 6 players in this segment.
    pub player_predictions: [f32; TOTAL_PLAYERS],
}

/// Extracted player-centric features for incremental segment prediction.
#[derive(Clone)]
pub struct ExtractedSegmentFeatures {
    /// One entry per **subsampled** replay frame (same step as training: [`FRAME_SUBSAMPLE_RATE`]).
    pub(crate) player_centric_frames: Vec<[PlayerCentricFrameFeatures; 6]>,
    /// Length of the original replay frame list (for mapping segment bounds to wall-clock times).
    pub(crate) original_replay_frame_count: usize,
}

impl ExtractedSegmentFeatures {
    /// Builds features from game frames (call once, then use for all segments).
    ///
    /// Uses the same frame subsampling as the training pipeline (`1` out of every
    /// [`FRAME_SUBSAMPLE_RATE`] replay frames).
    ///
    /// The roster is derived from all frames so that canonical slot assignments
    /// remain stable even when a player disconnects mid-game.
    pub fn from_frames(frames: &[replay_structs::GameFrame]) -> Self {
        let roster = PlayerRoster::from_frames(frames);
        let player_centric_frames = frames
            .iter()
            .step_by(FRAME_SUBSAMPLE_RATE)
            .map(|frame| extract_all_player_centric_features(frame, &roster))
            .collect();
        Self {
            player_centric_frames,
            original_replay_frame_count: frames.len(),
        }
    }

    /// Number of segments for the given segment length.
    pub const fn segment_count(&self, segment_length: usize) -> usize {
        if self.player_centric_frames.len() >= segment_length {
            self.player_centric_frames.len() / segment_length
        } else {
            1
        }
    }

    /// Predicts a single segment by index. Returns `None` if index is out of range.
    ///
    /// This method is `async` so GPU backends (for example WGPU) can read outputs with
    /// [`Tensor::into_data_async`] on targets such as WASM where synchronous reads are not supported.
    #[expect(clippy::future_not_send)]
    pub async fn predict_single_segment<B: Backend>(
        &self,
        model: &SequenceModel<B>,
        device: &B::Device,
        segment_length: usize,
        segment_index: usize,
    ) -> Option<SegmentPrediction> {
        let num_segments = self.segment_count(segment_length);
        if segment_index >= num_segments {
            return None;
        }
        let start = segment_index * segment_length;
        let end = (start + segment_length).min(self.player_centric_frames.len());
        let segment_frames =
            get_segment_player_frames(&self.player_centric_frames, start, segment_length);

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

        // output: [1, 6] (batch_size=1, 6 players) — raw MMR scale (see training loss).
        let output = model.forward_inference(input);
        let output_data = output.into_data_async().await.ok()?;
        let values = output_data.to_vec::<f32>().ok()?;
        let mut player_predictions = [1000.0f32; TOTAL_PLAYERS];
        for (i, val) in values.iter().enumerate().take(TOTAL_PLAYERS) {
            if let Some(pred) = player_predictions.get_mut(i) {
                *pred = *val;
            }
        }
        let (start_original, end_original_exclusive) =
            subsampled_bounds_to_original_frames(start, end, self.original_replay_frame_count);
        Some(SegmentPrediction {
            segment_index,
            start_frame: start_original,
            end_frame: end_original_exclusive,
            player_predictions,
        })
    }
}

/// Predicts MMR for all players per segment using player-centric features.
///
/// Returns a list of per-segment predictions. Use [`predict_player_centric`] if you
/// only need the averaged result.
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
/// A vector of per-segment predictions.
pub fn predict_player_centric_per_segment<B: Backend>(
    model: &SequenceModel<B>,
    frames: &[replay_structs::GameFrame],
    device: &B::Device,
    segment_length: usize,
) -> Vec<SegmentPrediction> {
    if frames.is_empty() {
        return vec![];
    }

    // Build the canonical roster once from all frames, then extract features.
    // Using a single roster for the whole game prevents slot shifting when a
    // player disconnects mid-match.
    let roster = PlayerRoster::from_frames(frames);
    let original_frame_count = frames.len();
    let player_centric_frames: Vec<[PlayerCentricFrameFeatures; 6]> = frames
        .iter()
        .step_by(FRAME_SUBSAMPLE_RATE)
        .map(|frame| extract_all_player_centric_features(frame, &roster))
        .collect();

    let num_segments = if player_centric_frames.len() >= segment_length {
        player_centric_frames.len() / segment_length
    } else {
        1
    };

    let mut results = Vec::with_capacity(num_segments);

    for seg_idx in 0..num_segments {
        let start = seg_idx * segment_length;
        let end = (start + segment_length).min(player_centric_frames.len());
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

        // Forward pass - output is [1, 6] (batch_size=1, 6 players), raw MMR scale.
        let output = model.forward_inference(input);

        // Extract predictions
        let output_data = output.into_data();
        if let Ok(values) = output_data.to_vec::<f32>() {
            let mut player_predictions = [1000.0f32; TOTAL_PLAYERS];
            for (i, val) in values.iter().enumerate().take(TOTAL_PLAYERS) {
                if let Some(pred) = player_predictions.get_mut(i) {
                    *pred = *val;
                }
            }
            let (start_original, end_original_exclusive) =
                subsampled_bounds_to_original_frames(start, end, original_frame_count);
            results.push(SegmentPrediction {
                segment_index: seg_idx,
                start_frame: start_original,
                end_frame: end_original_exclusive,
                player_predictions,
            });
        }
    }

    results
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
    let segments = predict_player_centric_per_segment(model, frames, device, segment_length);

    if segments.is_empty() {
        return [1000.0; TOTAL_PLAYERS];
    }

    let mut sum_predictions = [0.0f32; TOTAL_PLAYERS];
    for segment in &segments {
        for (i, pred) in segment.player_predictions.iter().enumerate() {
            if let Some(sum) = sum_predictions.get_mut(i) {
                *sum += *pred;
            }
        }
    }

    let count = segments.len() as f32;
    for pred in &mut sum_predictions {
        *pred /= count;
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

/// Loads a model checkpoint from in-memory bytes (binary format).
///
/// This function is designed for WASM / embedded model use cases where the model
/// weights are included via `include_bytes!`.
///
/// It tries **NamedMpk** format first (the default training format produced by
/// `save_checkpoint`), then falls back to **Bin** format (produced by
/// `save_checkpoint_bin`).
///
/// # Arguments
///
/// * `model_bytes` - Raw bytes of the model file
/// * `config_json` - JSON string of the training config (can be empty for defaults)
/// * `device` - The device to load the model onto
///
/// # Errors
///
/// Returns an error if the model cannot be loaded from bytes.
pub fn load_checkpoint_from_bytes<B: Backend>(
    model_bytes: &[u8],
    config_json: &str,
    device: &B::Device,
) -> anyhow::Result<SequenceModel<B>> {
    use burn::record::{BinBytesRecorder, NamedMpkBytesRecorder, Recorder};

    let model_config = if config_json.is_empty() {
        ModelConfig::new()
    } else {
        let training_config: TrainingConfig = serde_json::from_str(config_json)
            .map_err(|error| anyhow::anyhow!("Failed to parse model config: {error}"))?;
        training_config.model
    };

    // Create model with config
    let model = SequenceModel::new(device, &model_config);

    // Try NamedMpk format first (default training checkpoint format), then Bin format.
    let mpk_recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    if let Ok(record) = mpk_recorder.load(model_bytes.to_vec(), device) {
        return Ok(model.load_record(record));
    }

    let bin_recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let record = bin_recorder
        .load(model_bytes.to_vec(), device)
        .map_err(|error| {
            anyhow::anyhow!(
                "Failed to load model from bytes (tried NamedMpk and Bin formats): {error}"
            )
        })?;

    Ok(model.load_record(record))
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

        // Create a batch of 2 games × 6 players, each with 100 frames.
        // Input shape: [batch*6, seq_len, player_features]
        let batch_size = 2;
        let seq_len = 100;
        let input = Tensor::<TestBackend, 3>::zeros(
            [batch_size * 6, seq_len, PLAYER_CENTRIC_FEATURE_COUNT],
            &device,
        );
        let output = model.forward(input);

        assert_eq!(output.dims(), [batch_size, TOTAL_PLAYERS]);
        let values = output.into_data().to_vec::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
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

        // Input [6, seq, 95] → batch_size=1 → output [1, 6]
        assert_eq!(output.dims(), [1, TOTAL_PLAYERS]);

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
