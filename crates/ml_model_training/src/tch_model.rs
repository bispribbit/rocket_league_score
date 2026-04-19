//! cuDNN-accelerated sequence model for training.
//!
//! This module provides [`TchSequenceModel`], which mirrors the architecture of
//! [`ml_model::SequenceModel`] but is implemented in pure `tch-rs` (LibTorch)
//! so that both LSTM layers execute as a **single cuDNN kernel call** rather
//! than burn-nn's per-timestep Rust loop.
//!
//! Measured impact (GTX 1080 Ti, T1 overfit test, seq_len=300, batch=16):
//!   - burn-cuda + fusion :  ~8 s / epoch (forward dominates)
//!   - tch cuDNN LSTM     :  target < 0.1 s / epoch
//!
//! ## Architecture (mirrors SequenceModel exactly)
//! 1. LSTM layer 1: `[input_features → lstm_hidden_1]`
//! 2. LSTM layer 2: `[lstm_hidden_1  → lstm_hidden_2]`
//! 3. Attention-weighted temporal pool + last hidden → `[lstm_hidden_2 × 2]`
//! 4. Per-player feedforward: `[lstm_hidden_2*2 → feedforward_hidden]` + ReLU
//! 5. Player head: `[feedforward_hidden → player_head_hidden → 1]`
//! 6. Lobby-bias head: mean of all 6 players' embeddings → scalar broadcast
//! 7. Ordinal head: `[feedforward_hidden → 21]` (training auxiliary loss)
//!
//! ## Weight transfer to burn (for inference)
//! After training, call [`TchSequenceModel::export_weights`] to get a flat
//! `HashMap<String, Vec<f32>>` keyed by burn-compatible record field names,
//! then load them into a `SequenceModel<NdArray>` for checkpoint serialisation.

use std::collections::HashMap;

use tch::{Device, Kind, Tensor, nn, nn::RNN};

/// Architecture hyper-parameters.  Defaults match `ml_model::ModelConfig`.
#[derive(Clone, Debug)]
pub struct TchModelConfig {
    pub input_size: i64,             // PLAYER_CENTRIC_FEATURE_COUNT (106)
    pub lstm_hidden_1: i64,          // 256
    pub lstm_hidden_2: i64,          // 128
    pub feedforward_hidden: i64,     // 128
    pub player_head_hidden: i64,     // 64
    pub ordinal_num_boundaries: i64, // 21  (ORDINAL_NUM_BOUNDARIES)
    pub total_players: i64,          // 6   (TOTAL_PLAYERS)
}

impl Default for TchModelConfig {
    fn default() -> Self {
        Self {
            input_size: 106,
            lstm_hidden_1: 256,
            lstm_hidden_2: 128,
            feedforward_hidden: 128,
            player_head_hidden: 64,
            ordinal_num_boundaries: 21,
            total_players: 6,
        }
    }
}

/// LibTorch sequence model with cuDNN-fused LSTM.
///
/// The `VarStore` is owned *outside* this struct (standard tch-rs idiom) so
/// the optimizer can reach all parameters:
/// ```ignore
/// let vs   = nn::VarStore::new(Device::Cuda(0));
/// let model = TchSequenceModel::new(&vs.root(), &config);
/// let mut opt = nn::Adam::default().build(&vs, lr)?;
/// ```
pub struct TchSequenceModel {
    lstm1: nn::LSTM,
    lstm2: nn::LSTM,
    attention_query: nn::Linear,
    player_fc1: nn::Linear,
    player_head_fc: nn::Linear,
    player_head_out: nn::Linear,
    lobby_bias_head: nn::Linear,
    ordinal_head: nn::Linear,
    #[allow(dead_code)]
    lstm2_hidden: i64,
    feedforward_hidden: i64,
    total_players: i64,
    #[allow(dead_code)]
    config: TchModelConfig,
}

impl TchSequenceModel {
    pub fn new(vs: &nn::Path<'_>, config: &TchModelConfig) -> Self {
        // batch_first=true: input/output are [batch, seq, features].
        // This matches the shape we receive from SequenceBatcher.
        let lstm_cfg = nn::RNNConfig {
            batch_first: true,
            ..Default::default()
        };

        let lstm1 = nn::lstm(vs / "lstm1", config.input_size, config.lstm_hidden_1, lstm_cfg);
        let lstm2 = nn::lstm(
            vs / "lstm2",
            config.lstm_hidden_1,
            config.lstm_hidden_2,
            lstm_cfg,
        );

        let attention_query =
            nn::linear(vs / "attention_query", config.lstm_hidden_2, 1, Default::default());

        // Input to player_fc1 is attention-pool ++ last-hidden  = lstm_hidden_2 * 2.
        let player_fc1 = nn::linear(
            vs / "player_fc1",
            config.lstm_hidden_2 * 2,
            config.feedforward_hidden,
            Default::default(),
        );

        let player_head_fc = nn::linear(
            vs / "player_head_fc",
            config.feedforward_hidden,
            config.player_head_hidden,
            Default::default(),
        );

        let player_head_out =
            nn::linear(vs / "player_head_out", config.player_head_hidden, 1, Default::default());

        let lobby_bias_head =
            nn::linear(vs / "lobby_bias_head", config.feedforward_hidden, 1, Default::default());

        let ordinal_head = nn::linear(
            vs / "ordinal_head",
            config.feedforward_hidden,
            config.ordinal_num_boundaries,
            Default::default(),
        );

        Self {
            lstm1,
            lstm2,
            attention_query,
            player_fc1,
            player_head_fc,
            player_head_out,
            lobby_bias_head,
            ordinal_head,
            lstm2_hidden: config.lstm_hidden_2,
            feedforward_hidden: config.feedforward_hidden,
            total_players: config.total_players,
            config: config.clone(),
        }
    }

    /// Forward pass (training mode: gradients tracked by tch autograd).
    ///
    /// * `input` — `[batch_size * 6, seq_len, input_features]`
    /// * `lobby_bias_scale` — 1.0 normally; 0.0 to zero the lobby term
    ///
    /// Returns `([batch_size, 6] mmr_preds, [batch*6, 21] ordinal_logits)`.
    pub fn forward_train(
        &self,
        input: &Tensor,
        lobby_bias_scale: f64,
    ) -> (Tensor, Tensor) {
        self.forward_impl(input, lobby_bias_scale, true)
    }

    /// Forward pass (evaluation: no dropout, no gradient tracking needed).
    pub fn forward_eval(&self, input: &Tensor, lobby_bias_scale: f64) -> Tensor {
        let (preds, _) = tch::no_grad(|| self.forward_impl(input, lobby_bias_scale, false));
        preds
    }

    fn forward_impl(
        &self,
        input: &Tensor,
        lobby_bias_scale: f64,
        _train: bool,
    ) -> (Tensor, Tensor) {
        let dims = input.size();
        let batch_times_players = dims[0]; // batch * 6
        let seq_len = dims[1];
        let batch_size = batch_times_players / self.total_players;

        // ── cuDNN LSTM (entire sequence in ONE kernel call each) ─────────────
        let (lstm1_out, _) = self.lstm1.seq(input);
        // lstm1_out: [batch*6, seq, lstm_hidden_1]

        let (lstm2_out, _) = self.lstm2.seq(&lstm1_out);
        // lstm2_out: [batch*6, seq, lstm_hidden_2]

        // ── Attention-weighted temporal pooling ───────────────────────────────
        // scores:  [batch*6, seq, 1] → squeeze → [batch*6, seq]
        // weights: softmax over seq dim
        let attn_scores = lstm2_out.apply(&self.attention_query).squeeze_dim(-1);
        let attn_weights = attn_scores.softmax(-1, Kind::Float); // [batch*6, seq]

        // Weighted sum over seq: [batch*6, lstm_hidden_2]
        let attention_pool = (&lstm2_out * attn_weights.unsqueeze(-1))
            .sum_dim_intlist(Some([1_i64].as_slice()), false, Kind::Float);

        // Last hidden state: [batch*6, lstm_hidden_2]
        let last_hidden = lstm2_out.narrow(1, seq_len - 1, 1).squeeze_dim(1);

        // Concatenate pool + last: [batch*6, lstm_hidden_2 * 2]
        let pooled = Tensor::cat(&[attention_pool, last_hidden], 1);

        // ── Per-player feedforward (no dropout for overfit harness) ───────────
        let x = pooled.apply(&self.player_fc1).relu();
        // x: [batch*6, feedforward_hidden]

        // Ordinal auxiliary head (training only; caller may ignore).
        let ordinal_logits = x.apply(&self.ordinal_head);
        // ordinal_logits: [batch*6, 21]

        // ── Player regression head ────────────────────────────────────────────
        let player_pred = x.apply(&self.player_head_fc).relu().apply(&self.player_head_out);
        // player_pred: [batch*6, 1]

        // ── Lobby bias ────────────────────────────────────────────────────────
        // Mean of all 6 players' feedforward repr per game: [batch, feedforward_hidden]
        let lobby_mean = x
            .view([batch_size, self.total_players, self.feedforward_hidden])
            .mean_dim(Some([1_i64].as_slice()), false, Kind::Float);

        let lobby_bias = lobby_mean.apply(&self.lobby_bias_head) * lobby_bias_scale;
        // [batch, 1] → expand → [batch*6, 1]
        let lobby_bias = lobby_bias
            .expand([batch_size, self.total_players], false)
            .reshape([batch_times_players, 1]);

        // Raw MMR predictions: [batch, 6]
        let mmr_preds = (player_pred + lobby_bias).view([batch_size, self.total_players]);

        (mmr_preds, ordinal_logits)
    }

    /// Export all parameters keyed by burn-compatible record path.
    ///
    /// Keys follow burn's derive-Module naming convention so a conversion
    /// utility can reconstruct a `SequenceModel::Record<NdArray>` from them.
    pub fn export_weights(&self, vs: &nn::VarStore) -> HashMap<String, Vec<f32>> {
        let mut out = HashMap::new();
        for (name, tensor) in vs.variables() {
            let values: Vec<f32> = Vec::try_from(tensor.to(Device::Cpu).to_kind(Kind::Float))
                .unwrap_or_default();
            out.insert(name, values);
        }
        out
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Tensor conversion helpers (burn NdArray → tch CUDA)
// ═════════════════════════════════════════════════════════════════════════════

/// Convert a rank-3 burn NdArray tensor → tch CUDA tensor.
///
/// Data is already on CPU (NdArray is always CPU), so we just reinterpret the
/// f32 buffer and upload to the tch device in one `to()` call.
pub fn burn_3d_to_tch(
    t: burn::tensor::Tensor<burn::backend::NdArray, 3>,
    device: Device,
) -> Tensor {
    let dims = t.dims(); // [usize; 3]
    let shape = [dims[0] as i64, dims[1] as i64, dims[2] as i64];
    let values: Vec<f32> = t.into_data().to_vec().unwrap_or_default();
    Tensor::from_slice(&values).view(shape).to(device)
}

/// Convert a rank-2 burn NdArray tensor → tch CUDA tensor.
pub fn burn_2d_to_tch(
    t: burn::tensor::Tensor<burn::backend::NdArray, 2>,
    device: Device,
) -> Tensor {
    let dims = t.dims(); // [usize; 2]
    let shape = [dims[0] as i64, dims[1] as i64];
    let values: Vec<f32> = t.into_data().to_vec().unwrap_or_default();
    Tensor::from_slice(&values).view(shape).to(device)
}
