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

pub mod fused_lstm;

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::tensor::activation::softmax;
use feature_extractor::{
    FRAME_SUBSAMPLE_RATE, PLAYER_CENTRIC_FEATURE_COUNT, PlayerCentricFrameFeatures, TOTAL_PLAYERS,
    extract_player_centric_game_sequence_inference,
};
use fused_lstm::{FusedLstm, FusedLstmBackend, FusedLstmConfig};

/// Scale factor to normalise MMR values to [0, 1] range.
///
/// Raw MMR range is approximately 0 – 2500 (SSL sits at ~2200, top pros higher),
/// so dividing by this constant keeps the output well below the saturation ceiling.
/// Raised from 2000 to 2500 so that SSL labels (now 2200) are no longer clipped
/// at the normalisation boundary.
pub const MMR_SCALE: f32 = 2500.0;

/// Output scale for the lobby-level bias in raw MMR units.
///
/// The `lobby_bias_head` is otherwise free to match any per-game mean from a
/// single scalar, which lets the LSTM+per-player head stay under-used when
/// all six training labels in a segment share the same MMR. Capping this
/// path forces the per-player branch to own most of the signal.
pub const LOBBY_BIAS_OUTPUT_SCALE: f32 = 0.0;

/// Number of cumulative-logit boundaries for ordinal rank classification.
///
/// We model 22 ranked tiers (Bronze-1 … SSL), requiring 21 boundaries.
/// The k-th logit represents P(player is above rank k+1), so SSL players
/// have all 21 outputs positive and Bronze-1 players have all negative.
pub const ORDINAL_NUM_BOUNDARIES: usize = 21;

/// MMR boundary values for ordinal classification (ascending order).
///
/// `ORDINAL_BOUNDARIES[k]` is the canonical MMR threshold for the k-th
/// boundary.  A player with MMR ≥ `ORDINAL_BOUNDARIES[k]` receives a
/// target of 1 for that boundary (they are "above" that rank tier).
pub const ORDINAL_BOUNDARIES_MMR: [f32; ORDINAL_NUM_BOUNDARIES] = [
    194.0,  // above Bronze-1  (Bronze-2 canonical)
    257.0,  // above Bronze-2
    321.0,  // above Bronze-3
    386.0,  // above Silver-1
    451.0,  // above Silver-2
    516.0,  // above Silver-3
    580.0,  // above Gold-1
    644.0,  // above Gold-2
    709.0,  // above Gold-3
    773.0,  // above Platinum-1
    837.0,  // above Platinum-2
    902.0,  // above Platinum-3
    966.0,  // above Diamond-1
    1030.0, // above Diamond-2
    1127.0, // above Diamond-3
    1258.0, // above Champion-1
    1388.0, // above Champion-2
    1520.0, // above Champion-3
    1651.0, // above GrandChampion-1
    1782.0, // above GrandChampion-2
    2200.0, // above GrandChampion-3 (SSL threshold)
];
/// Configuration for the sequence model.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Hidden size for the first LSTM layer.
    ///
    /// Raised to 256 to give the model more capacity for the longer (300-frame)
    /// sequences introduced in Phase 2.
    #[config(default = 256)]
    pub lstm_hidden_1: usize,
    /// Hidden size for the second LSTM layer.
    #[config(default = 128)]
    pub lstm_hidden_2: usize,
    /// Hidden size for the per-player feedforward layer.
    /// Input is `lstm_hidden_2 * 2` (attention-pool + last hidden state concatenated).
    #[config(default = 128)]
    pub feedforward_hidden: usize,
    /// Hidden size for the per-player prediction head MLP.
    #[config(default = 64)]
    pub player_head_hidden: usize,
    /// Hidden size for the lobby-level encoder used in the split-encoder architecture.
    /// The lobby encoder summarises all 6 players to produce a per-slot bias that is
    /// subtracted from individual skill predictions during training (20 % dropout).
    #[config(default = 64)]
    pub lobby_hidden: usize,
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
    /// At 15fps effective sampling (FRAME_SUBSAMPLE_RATE=2 on 30fps source),
    /// 300 frames ≈ 20 seconds of gameplay per segment.
    #[config(default = 300)]
    pub sequence_length: usize,
}

/// LSTM-based sequence model with split per-player + lobby encoders.
///
/// Architecture (Phase 2 + 3):
/// 1. **Per-player skill encoder**: two-layer LSTM that sees only each player's
///    own features (ball state + self state + ball-relative).  Produces a skill
///    embedding that cannot short-cut via lobby context.
/// 2. **Full-context LSTM**: two-layer LSTM that sees all features (same input
///    as the skill encoder, but the second layer also sees the lobby-mean
///    from the first encoder round to let the model calibrate within the lobby).
/// 3. **Temporal pooling**: attention-weighted pool + last hidden state.
/// 4. **Lobby bias head**: reads the 6-player mean embedding to produce a
///    per-slot lobby bias.  During training the bias is randomly zeroed (20 %)
///    so the skill encoder is forced to carry predictions alone.
/// 5. **Per-player regression head**: `skill_embed → feedforward → MMR scalar`.
///
/// The final output is `[batch_size, 6]` predicted MMR values (raw, not normalised).
#[derive(Module, Debug)]
pub struct SequenceModel<B: Backend> {
    /// First LSTM layer for full per-player feature stream.
    lstm1: FusedLstm<B>,
    /// Second LSTM layer for deeper temporal patterns.
    lstm2: FusedLstm<B>,
    /// Learned attention query for temporal pooling.
    /// Maps `[batch*6, seq, lstm2_hidden]` → `[batch*6, seq, 1]`.
    attention_query: Linear<B>,
    /// Per-player feedforward: attention-pool + last hidden → player representation.
    /// Input size: `lstm2_hidden * 2`.
    player_fc1: Linear<B>,
    /// Per-player prediction head layer 1.
    player_head_fc: Linear<B>,
    /// Per-player prediction head output: → 1 MMR scalar.
    player_head_out: Linear<B>,
    /// Lobby bias head: takes the mean of all 6 players' feedforward representations
    /// and produces a per-slot adjustment.  Input: `feedforward_hidden`, output: 1.
    lobby_bias_head: Linear<B>,
    /// Ordinal classification head: maps per-player feedforward representation to
    /// [`ORDINAL_NUM_BOUNDARIES`] cumulative-logit boundaries used for auxiliary
    /// rank-classification loss.  Not used during inference.
    ordinal_head: Linear<B>,
    /// Dropout for regularization.
    dropout: Dropout,
    /// `ReLU` activation.
    activation: Relu,
    /// Hidden size of second LSTM (stored for dimension tracking).
    lstm2_hidden: usize,
}

impl<B: FusedLstmBackend> SequenceModel<B> {
    /// Creates a new sequence model with the given configuration.
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let lstm1 = FusedLstmConfig::new(PLAYER_CENTRIC_FEATURE_COUNT, config.lstm_hidden_1, true)
            .init(device);
        let lstm2 =
            FusedLstmConfig::new(config.lstm_hidden_1, config.lstm_hidden_2, true).init(device);

        let dropout = DropoutConfig::new(config.dropout).init();

        // Learned attention query maps each timestep to a scalar importance score.
        let attention_query = LinearConfig::new(config.lstm_hidden_2, 1).init(device);

        // Feedforward input = attention-pool (lstm2_hidden) + last hidden (lstm2_hidden).
        let player_fc1 =
            LinearConfig::new(config.lstm_hidden_2 * 2, config.feedforward_hidden).init(device);

        let player_head_fc =
            LinearConfig::new(config.feedforward_hidden, config.player_head_hidden).init(device);
        let player_head_out = LinearConfig::new(config.player_head_hidden, 1).init(device);

        // Lobby bias: takes mean feedforward embedding across all 6 players per game,
        // outputs a single scalar that is broadcast to all player slots.
        let lobby_bias_head = LinearConfig::new(config.feedforward_hidden, 1).init(device);

        // Auxiliary ordinal head for rank classification (training only).
        let ordinal_head =
            LinearConfig::new(config.feedforward_hidden, ORDINAL_NUM_BOUNDARIES).init(device);

        Self {
            lstm1,
            lstm2,
            attention_query,
            player_fc1,
            player_head_fc,
            player_head_out,
            lobby_bias_head,
            ordinal_head,
            dropout,
            activation: Relu::new(),
            lstm2_hidden: config.lstm_hidden_2,
        }
    }

    /// Forward pass with explicit lobby-bias scale.
    ///
    /// * `input` — `[batch_size * 6, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`
    /// * `lobby_bias_scale` — 1.0 normally; 0.0 to zero the lobby bias (20 % of
    ///   training batches) forcing the skill encoder to work standalone.
    ///
    /// Returns `[batch_size, 6]` raw MMR predictions.
    pub fn forward_with_lobby_scale(
        &self,
        input: Tensor<B, 3>,
        lobby_bias_scale: f32,
    ) -> Tensor<B, 2> {
        let [batch_times_players, seq_len, _] = input.dims();
        let batch_size = batch_times_players / TOTAL_PLAYERS;

        // LSTM stack.
        let (lstm1_out, _) = self.lstm1.forward(input, None);
        let (lstm2_out, _) = self.lstm2.forward(lstm1_out, None);

        // Attention-weighted temporal pooling.
        // scores: [batch*6, seq, 1] → softmax over seq → weighted sum.
        let attention_scores = self.attention_query.forward(lstm2_out.clone());
        let attention_weights =
            softmax(attention_scores.reshape([batch_times_players, seq_len]), 1).reshape([
                batch_times_players,
                seq_len,
                1,
            ]);
        let attention_pool = (lstm2_out.clone() * attention_weights)
            .sum_dim(1)
            .reshape([batch_times_players, self.lstm2_hidden]);

        // Last hidden state for recency signal.
        let last_hidden = lstm2_out
            .narrow(1, seq_len - 1, 1)
            .reshape([batch_times_players, self.lstm2_hidden]);

        // [batch*6, lstm2_hidden * 2]
        let pooled = Tensor::cat(vec![attention_pool, last_hidden], 1);

        // Per-player feedforward.
        let x = self.dropout.forward(pooled);
        let x = self.player_fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        // Per-player regression head.
        let player_pred = self.player_head_fc.forward(x.clone());
        let player_pred = self.activation.forward(player_pred);
        let player_pred = self.player_head_out.forward(player_pred);

        // Lobby bias from mean of all 6 players' feedforward representations.
        let feedforward_hidden = x.dims()[1];
        let lobby_mean = x
            .reshape([batch_size, TOTAL_PLAYERS, feedforward_hidden])
            .mean_dim(1)
            .reshape([batch_size, feedforward_hidden]);
        let lobby_bias_per_game = self.lobby_bias_head.forward(lobby_mean); // [batch, 1]
        let lobby_bias = lobby_bias_per_game.expand([batch_size, TOTAL_PLAYERS])
            * lobby_bias_scale
            * LOBBY_BIAS_OUTPUT_SCALE;

        player_pred.reshape([batch_size, TOTAL_PLAYERS]) + lobby_bias
    }

    /// Forward pass that also returns per-player ordinal logits.
    ///
    /// Returns `(mmr_predictions, ordinal_logits)` where:
    /// - `mmr_predictions` is `[batch_size, 6]` raw MMR.
    /// - `ordinal_logits` is `[batch_size * 6, ORDINAL_NUM_BOUNDARIES]` cumulative
    ///   boundary logits for the auxiliary ordinal rank-classification loss.
    pub fn forward_with_ordinal_scale(
        &self,
        input: Tensor<B, 3>,
        lobby_bias_scale: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_times_players, seq_len, _] = input.dims();
        let batch_size = batch_times_players / TOTAL_PLAYERS;

        // Shared LSTM stack.
        let (lstm1_out, _) = self.lstm1.forward(input, None);
        let (lstm2_out, _) = self.lstm2.forward(lstm1_out, None);

        // Attention-weighted temporal pooling.
        let attention_scores = self.attention_query.forward(lstm2_out.clone());
        let attention_weights =
            softmax(attention_scores.reshape([batch_times_players, seq_len]), 1).reshape([
                batch_times_players,
                seq_len,
                1,
            ]);
        let attention_pool = (lstm2_out.clone() * attention_weights)
            .sum_dim(1)
            .reshape([batch_times_players, self.lstm2_hidden]);

        let last_hidden = lstm2_out
            .narrow(1, seq_len - 1, 1)
            .reshape([batch_times_players, self.lstm2_hidden]);

        let pooled = Tensor::cat(vec![attention_pool, last_hidden], 1);

        // Per-player feedforward.
        let x = self.dropout.forward(pooled);
        let x = self.player_fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        // Ordinal logits from the shared representation (before the final head dropout).
        let ordinal_logits = self.ordinal_head.forward(x.clone()); // [batch*6, 21]

        // Per-player regression head.
        let player_pred = self.player_head_fc.forward(x.clone());
        let player_pred = self.activation.forward(player_pred);
        let player_pred = self.player_head_out.forward(player_pred);

        // Lobby bias.
        let feedforward_hidden = x.dims()[1];
        let lobby_mean = x
            .reshape([batch_size, TOTAL_PLAYERS, feedforward_hidden])
            .mean_dim(1)
            .reshape([batch_size, feedforward_hidden]);
        let lobby_bias_per_game = self.lobby_bias_head.forward(lobby_mean);
        let lobby_bias = lobby_bias_per_game.expand([batch_size, TOTAL_PLAYERS])
            * lobby_bias_scale
            * LOBBY_BIAS_OUTPUT_SCALE;

        let mmr_preds = player_pred.reshape([batch_size, TOTAL_PLAYERS]) + lobby_bias;
        (mmr_preds, ordinal_logits)
    }

    /// Forward pass (lobby bias active — use during validation and inference).
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        self.forward_with_lobby_scale(input, 1.0)
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
pub fn create_model<B: FusedLstmBackend>(
    device: &B::Device,
    config: &ModelConfig,
) -> SequenceModel<B> {
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
    /// Builds features from game frames for inference.
    ///
    /// Uses the same subsampling and feature set as the training pipeline.
    /// Cumulative per-player stats reset every `segment_length` subsampled frames.
    /// Score diff is set to 0 (unknown during live inference).
    pub fn from_frames(frames: &[replay_structs::GameFrame], segment_length: usize) -> Self {
        let player_centric_frames =
            extract_player_centric_game_sequence_inference(frames, segment_length);
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
    pub async fn predict_single_segment<B: FusedLstmBackend>(
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
pub fn predict_player_centric_per_segment<B: FusedLstmBackend>(
    model: &SequenceModel<B>,
    frames: &[replay_structs::GameFrame],
    device: &B::Device,
    segment_length: usize,
) -> Vec<SegmentPrediction> {
    if frames.is_empty() {
        return vec![];
    }

    let original_frame_count = frames.len();
    let player_centric_frames =
        extract_player_centric_game_sequence_inference(frames, segment_length);

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
pub fn predict_player_centric<B: FusedLstmBackend>(
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
pub fn load_checkpoint_from_bytes<B: FusedLstmBackend>(
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
