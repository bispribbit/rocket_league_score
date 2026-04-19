//! Training logic for the sequence model.

use std::sync::Arc;
use std::time::Instant;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{
    MMR_SCALE, ORDINAL_BOUNDARIES_MMR, ORDINAL_NUM_BOUNDARIES, SequenceModel, TrainingConfig,
};
use replay_structs::{Rank, RankDivision};
use tracing::{info, warn};

use crate::dataset::{BatchPrefetcher, SequenceBatcher};
use crate::segment_cache::SegmentStore;
use crate::{CheckpointValidationMetrics, ValidationRankRmseEntry, save_checkpoint};

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
    /// Per-rank validation RMSE from the last validation pass (MMR units).
    pub last_validation_rank_rmse: Option<Vec<ValidationRankRmseEntry>>,
}

fn checkpoint_validation_metrics_from_state(
    state: &TrainingState,
) -> Option<CheckpointValidationMetrics> {
    let loss = state.current_valid_loss?;
    let rank_entries = state.last_validation_rank_rmse.clone().unwrap_or_default();
    Some(CheckpointValidationMetrics::from_validation_loss_with_rank_breakdown(loss, rank_entries))
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
            last_validation_rank_rmse: None,
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

/// Number of batches to prefetch ahead (keep GPU fed while loading next batches).
const PREFETCH_COUNT: usize = 8;

/// How often to sync with GPU to extract loss value (every N batches).
const LOSS_SYNC_INTERVAL: usize = 10;

/// Standard deviation of Gaussian label jitter applied to training targets (MMR units).
/// Breaking exact same-rank-lobby ties regularises the model against using lobby
/// MMR as a shortcut and encourages it to discriminate within a lobby.
const LABEL_JITTER_STD: f64 = 75.0;

/// Asymmetric quantile (pinball) loss weight for under-predictions at high rank.
/// Adds to the Huber loss to push the model to predict SSL rather than regressing
/// toward the lobby mean.
const PINBALL_WEIGHT: f32 = 0.3;
/// Quantile for the pinball term: 0.9 means under-prediction is penalised 9×.
const PINBALL_TAU: f32 = 0.9;
/// MMR threshold above which the pinball term is activated.
const PINBALL_THRESHOLD_MMR: f32 = 1400.0;

/// Epoch at which EMA-based smurf masking kicks in.
const SMURF_MASK_START_EPOCH: usize = 5;
/// Number of consecutive epochs a segment must be in the top-1 % before masking.
const SMURF_MASK_SUSTAIN_EPOCHS: usize = 3;
/// EMA decay factor for per-segment loss tracking (α for new value).
const SMURF_EMA_ALPHA: f32 = 0.3;

/// Weight of the auxiliary ordinal (cumulative-logit) classification loss.
const ORDINAL_LOSS_WEIGHT: f32 = 0.2;

/// Weight of the within-batch pairwise ranking loss.
const PAIRWISE_LOSS_WEIGHT: f32 = 0.1;

/// Per-segment smurf-masking state.
#[derive(Default)]
struct SmurfMaskState {
    /// Exponential moving average of per-segment loss.
    ema_loss: Vec<f32>,
    /// Number of consecutive epochs each segment has been in the top-1 % EMA.
    high_ema_epochs: Vec<u32>,
    /// Final boolean mask: true = include in training, false = masked out.
    active: Vec<bool>,
}

impl SmurfMaskState {
    fn new(num_segments: usize) -> Self {
        Self {
            ema_loss: vec![0.0; num_segments],
            high_ema_epochs: vec![0; num_segments],
            active: vec![true; num_segments],
        }
    }

    /// Updates EMA from a batch of (segment_index, loss) pairs.
    fn update_ema(&mut self, pairs: &[(usize, f32)]) {
        for &(idx, loss) in pairs {
            if let Some(ema) = self.ema_loss.get_mut(idx) {
                *ema = SMURF_EMA_ALPHA.mul_add(loss, (1.0 - SMURF_EMA_ALPHA) * *ema);
            }
        }
    }

    /// Refreshes the per-epoch high-EMA counters and active mask.
    ///
    /// Called once per epoch after the EMA has been updated for all batches.
    fn refresh_masks(&mut self) {
        let n = self.ema_loss.len();
        if n == 0 {
            return;
        }
        let mut sorted_ema = self.ema_loss.clone();
        sorted_ema.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (n as f32 * 0.99) as usize;
        let threshold = sorted_ema.get(threshold_idx).copied().unwrap_or(f32::MAX);

        let mut masked_count = 0usize;
        for ((ema, high_epochs), active) in self
            .ema_loss
            .iter()
            .zip(self.high_ema_epochs.iter_mut())
            .zip(self.active.iter_mut())
        {
            if *ema >= threshold {
                *high_epochs += 1;
            } else {
                *high_epochs = 0;
            }
            if *high_epochs >= SMURF_MASK_SUSTAIN_EPOCHS as u32 {
                *active = false;
                masked_count += 1;
            }
        }
        if masked_count > 0 {
            warn!(
                smurf_masked_count = masked_count,
                total_segments = n,
                "Smurf masking: permanently masked high-EMA-loss segments this epoch"
            );
        }
    }
}

/// Precomputes inverse-frequency per-rank weights from the training segment store.
///
/// Returns a 23-element vec indexed by `Rank::as_numeric_index()`.
/// Values are clipped to `[0.5, 5.0]` and normalised so the mean is 1.0.
fn compute_inverse_frequency_weights(dataset: &SegmentStore) -> Vec<f32> {
    let mut rank_counts = [0u32; 23];
    for idx in 0..dataset.len() {
        let rank_idx = dataset.get_primary_rank_index(idx).unwrap_or(0) as usize;
        if let Some(count) = rank_counts.get_mut(rank_idx) {
            *count += 1;
        }
    }
    let max_count = rank_counts.iter().copied().max().unwrap_or(1).max(1) as f32;
    let weights: Vec<f32> = rank_counts
        .iter()
        .map(|&count| {
            if count == 0 {
                1.0f32
            } else {
                (max_count / count as f32).clamp(0.5, 5.0)
            }
        })
        .collect();
    let mean_weight = weights.iter().sum::<f32>() / weights.len() as f32;
    weights.iter().map(|w| w / mean_weight).collect()
}

/// Looks up per-sample inverse-frequency weights on CPU and returns them as a Vec.
///
/// `mean_target_mmr` is in raw MMR units (not normalised).
fn lookup_rank_weights(mean_target_mmr_slice: &[f32], rank_weights: &[f32]) -> Vec<f32> {
    mean_target_mmr_slice
        .iter()
        .map(|&mmr| {
            if mmr <= 0.0 {
                return 1.0f32;
            }
            let rank = Rank::from(RankDivision::from(mmr as i32));
            let idx = rank.as_numeric_index() as usize;
            rank_weights.get(idx).copied().unwrap_or(1.0)
        })
        .collect()
}

/// Trains the sequence model using cached segment data.
///
/// This training method loads segments from disk on-demand.
/// The data must already be split into train/valid datasets before calling.
///
/// # Arguments
///
/// * `model` - The model to train (will be modified in place).
/// * `train_dataset` - Training data as cached segments.
/// * `valid_dataset` - Optional validation data as cached segments.
/// * `config` - Training configuration.
/// * `checkpoint_config` - Optional checkpoint configuration for saving during training.
/// * `start_state` - Optional starting state for resumption.
///
/// # Errors
///
/// Returns an error if training fails.
pub fn train<B: AutodiffBackend>(
    model: &mut SequenceModel<B>,
    train_dataset: Arc<SegmentStore>,
    valid_dataset: Option<&Arc<SegmentStore>>,
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

    let device = model.device();

    // Adam with gradient clipping to stabilise LSTM training.
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // Huber loss (delta=1.0 in normalised space ≈ 2500 MMR → all errors quadratic).
    let huber_delta = 1.0_f32;

    // Precompute inverse-frequency per-rank weights once.
    let rank_weights = compute_inverse_frequency_weights(&train_dataset);
    info!(
        "Inverse-frequency rank weights: {:?}",
        rank_weights
            .iter()
            .enumerate()
            .map(|(i, w)| format!("rank{i}={w:.2}"))
            .collect::<Vec<_>>()
    );

    let mut checkpoint_paths = Vec::new();
    let mut state = start_state.unwrap_or_else(|| TrainingState::new(0));
    let start_epoch = state.current_epoch;
    let num_samples = train_dataset.len();

    // Smurf-masking state (activated after SMURF_MASK_START_EPOCH).
    let mut smurf_state = SmurfMaskState::new(num_samples);

    const EARLY_STOPPING_PATIENCE: usize = 10;

    let valid_segments = valid_dataset.map_or(0, |ds| ds.len());
    info!(
        "Starting training: {} train segs, {} valid segs, lr={}, seq_len={}, prefetch={}",
        num_samples, valid_segments, config.learning_rate, config.sequence_length, PREFETCH_COUNT
    );
    if start_epoch > 0 {
        info!("Resuming from epoch {}", start_epoch + 1);
    }

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let mut batch_count = 0;

        // Epoch indices: oversampled for rare ranks.
        let oversampled_indices = train_dataset.build_oversampled_indices(epoch as u64);
        // Apply smurf mask: drop segments that are permanently excluded.
        let indices: Vec<usize> = oversampled_indices
            .into_iter()
            .filter(|&idx| smurf_state.active.get(idx).copied().unwrap_or(true))
            .collect();

        let mut prefetcher = BatchPrefetcher::new(
            train_dataset.clone(),
            indices,
            config.batch_size,
            config.sequence_length,
            PREFETCH_COUNT,
        );

        let total_batches = prefetcher.total_batches();

        // Cosine learning-rate decay.
        let lr = cosine_lr(config.learning_rate, epoch, start_epoch, config.epochs);

        // Randomly zero lobby bias for 20 % of batches (Phase 3 training).
        // We decide per-batch using a deterministic hash of epoch+batch_idx.
        let lobby_zero_fraction = 0.2_f32;

        let mut accumulated_loss: Option<Tensor<B, 1>> = None;
        let mut accumulated_count = 0;
        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_loss_count = 0usize;

        // EMA updates for smurf masking (batch_idx → (segment_idx, loss_scalar) pairs).
        let mut smurf_ema_updates: Vec<(usize, f32)> = Vec::new();

        let mut time_prefetch_wait_us = 0u64;
        let mut time_to_gpu_us = 0u64;
        let mut time_forward_us = 0u64;
        let mut time_backward_us = 0u64;
        let mut time_optimizer_us = 0u64;

        loop {
            let t_prefetch_start = Instant::now();
            let Some(preloaded_batch) = prefetcher.next_batch() else {
                break;
            };
            time_prefetch_wait_us += t_prefetch_start.elapsed().as_micros() as u64;

            let t_to_gpu_start = Instant::now();
            let batch = preloaded_batch.to_batch::<B>(&device);
            time_to_gpu_us += t_to_gpu_start.elapsed().as_micros() as u64;

            let t_forward_start = Instant::now();

            // 20 % lobby-bias dropout.
            let lobby_scale =
                if pseudo_random_f32(epoch as u64, batch_count as u64) < lobby_zero_fraction {
                    0.0_f32
                } else {
                    1.0_f32
                };
            let (predictions, ordinal_logits) =
                model.forward_with_ordinal_scale(batch.inputs, lobby_scale);

            // Known-rank mask: 1.0 where target > 0.
            let mask = batch.targets.clone().greater_elem(0.0).float();
            let known_count = mask
                .clone()
                .sum()
                .into_data()
                .to_vec()
                .unwrap_or_else(|_| vec![1.0f32])
                .first()
                .copied()
                .unwrap_or(1.0f32)
                .max(1.0f32);

            // Per-sample inverse-frequency weights (computed on CPU).
            let known_per_row = mask.clone().sum_dim(1).clamp_min(1.0); // [batch, 1]
            let masked_targets_sum = (batch.targets.clone() * mask.clone()).sum_dim(1); // [batch, 1]
            let mean_target_mmr_vec: Vec<f32> = (masked_targets_sum.clone()
                / known_per_row.clone())
            .into_data()
            .to_vec()
            .unwrap_or_default();

            let weights_vec = lookup_rank_weights(&mean_target_mmr_vec, &rank_weights);
            let weights = Tensor::<B, 1>::from_floats(weights_vec.as_slice(), &device)
                .reshape([mean_target_mmr_vec.len(), 1]); // [batch, 1]

            // ±75 MMR Gaussian label jitter (train only, not validation).
            // Applied in normalised space; only shifts known-rank slots.
            let jitter_norm = Tensor::<B, 2>::random(
                [weights_vec.len(), TOTAL_PLAYERS],
                Distribution::Normal(0.0, LABEL_JITTER_STD / MMR_SCALE as f64),
                &device,
            );
            let raw_targets = batch.targets.clone();
            let raw_predictions = predictions.clone();
            let targets_norm = batch.targets / MMR_SCALE + jitter_norm * mask.clone();
            let predictions_norm = predictions / MMR_SCALE;

            let diff = predictions_norm.clone() - targets_norm.clone();
            let abs_diff = diff.clone().abs();
            let clamped = abs_diff.clone().clamp_min(0.0).clamp_max(huber_delta);
            let huber_loss =
                clamped.clone().powf_scalar(2.0) * 0.5 + (abs_diff.clone() - clamped) * huber_delta;

            // Asymmetric pinball term for high-rank targets (τ = 0.9).
            // Penalises under-prediction more than over-prediction above the threshold.
            let high_rank_mask = targets_norm
                .clone()
                .greater_elem(PINBALL_THRESHOLD_MMR / MMR_SCALE)
                .float();
            let pinball =
                (diff.clone() * PINBALL_TAU - diff.clone().clamp_max(0.0)) * high_rank_mask;

            let element_loss = huber_loss + pinball * PINBALL_WEIGHT;

            let regression_loss = (element_loss * mask.clone() * weights).sum() / known_count;

            // --- Auxiliary ordinal classification loss ---
            // For each player slot (batch*6 rows), build binary targets: 1 if true MMR
            // is above boundary k, 0 otherwise.  Use BCE with logits, masked to known slots.
            let flat_mask = mask
                .clone()
                .reshape([raw_targets.dims()[0] * TOTAL_PLAYERS]);
            let flat_targets_mmr =
                (raw_targets.clone() * MMR_SCALE).reshape([raw_targets.dims()[0] * TOTAL_PLAYERS]);

            let boundaries_vec: Vec<f32> = ORDINAL_BOUNDARIES_MMR.to_vec();
            let boundaries = Tensor::<B, 1>::from_floats(boundaries_vec.as_slice(), &device)
                .unsqueeze::<2>() // [1, 21]
                .transpose(); // [21, 1] — will broadcast

            // ordinal_targets: [batch*6, 21] — 1.0 if player_mmr > boundary_k
            let flat_targets_mmr_col = flat_targets_mmr.clone().unsqueeze::<2>(); // [batch*6, 1]
            let ordinal_targets = flat_targets_mmr_col
                .expand([flat_targets_mmr.dims()[0], ORDINAL_NUM_BOUNDARIES])
                .greater(
                    boundaries
                        .transpose() // [1, 21]
                        .expand([flat_targets_mmr.dims()[0], ORDINAL_NUM_BOUNDARIES]),
                )
                .float(); // [batch*6, 21]

            // BCE with logits: L = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
            let sigmoid_logits = activation::sigmoid(ordinal_logits.clone());
            let bce = ordinal_targets.clone() * (sigmoid_logits.clone().clamp_min(1e-6).log())
                + (ordinal_targets.clone().neg() + 1.0)
                    * ((sigmoid_logits.clone().neg() + 1.0).clamp_min(1e-6).log());
            let bce = bce.neg(); // [batch*6, 21]

            // Mask out unknown-rank slots.
            let ordinal_mask = flat_mask
                .clone()
                .unsqueeze::<2>()
                .expand([flat_mask.dims()[0], ORDINAL_NUM_BOUNDARIES]);
            let ordinal_known_count = ordinal_mask
                .clone()
                .sum()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            let ordinal_known_count = ordinal_known_count.first().copied().unwrap_or(1.0).max(1.0);
            let ordinal_loss =
                (bce * ordinal_mask).sum() / (ordinal_known_count * ORDINAL_NUM_BOUNDARIES as f32);

            // --- Within-batch pairwise ranking loss ---
            // For pairs of players in the same game where both ranks are known,
            // penalise cases where the model inverts the correct ordering.
            // Hinge: max(0, margin - (pred_i - pred_j)) when target_i > target_j.
            let batch_size_local = raw_targets.dims()[0];
            let preds_flat = raw_predictions.reshape([batch_size_local * TOTAL_PLAYERS]);
            let targets_flat = raw_targets.reshape([batch_size_local * TOTAL_PLAYERS]);
            // Reshape to [batch, 6] for per-lobby comparisons.
            let preds_lobby = preds_flat.reshape([batch_size_local, TOTAL_PLAYERS]);
            let targets_lobby = targets_flat.reshape([batch_size_local, TOTAL_PLAYERS]);
            let mask_lobby = mask.clone().reshape([batch_size_local, TOTAL_PLAYERS]);

            // For each lobby, compute outer difference: pred_i - pred_j and target_i - target_j.
            let preds_i = preds_lobby.clone().unsqueeze_dim::<3>(2); // [B, 6, 1]
            let preds_j = preds_lobby.clone().unsqueeze_dim::<3>(1); // [B, 1, 6]
            let pred_diff = preds_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
                - preds_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS]); // [B, 6, 6]

            let targets_i = targets_lobby.clone().unsqueeze_dim::<3>(2);
            let targets_j = targets_lobby.clone().unsqueeze_dim::<3>(1);
            let target_diff = targets_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
                - targets_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS]);

            // Pair mask: both slots known and i strictly outranks j (in MMR units).
            let mask_i = mask_lobby.clone().unsqueeze_dim::<3>(2);
            let mask_j = mask_lobby.clone().unsqueeze_dim::<3>(1);
            let pair_mask = mask_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
                * mask_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
                * target_diff
                    .clone()
                    .greater_elem(MMR_SCALE * 0.01) // at least 25 MMR difference
                    .float();

            let pairwise_hinge_margin: f32 = 50.0 / MMR_SCALE;
            let pairwise_loss_elements =
                (pair_mask.clone() * (pairwise_hinge_margin - pred_diff).clamp_min(0.0)).sum();
            let pair_count = pair_mask
                .sum()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_default();
            let pair_count = pair_count.first().copied().unwrap_or(1.0).max(1.0);
            let pairwise_loss = pairwise_loss_elements / pair_count;

            let loss = regression_loss
                + ordinal_loss * ORDINAL_LOSS_WEIGHT
                + pairwise_loss * PAIRWISE_LOSS_WEIGHT;
            time_forward_us += t_forward_start.elapsed().as_micros() as u64;

            // Collect per-segment loss for smurf masking (after SMURF_MASK_START_EPOCH).
            if epoch >= SMURF_MASK_START_EPOCH {
                let per_sample_loss = (diff.clone().powf_scalar(2.0) * mask.clone())
                    .sum_dim(1)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap_or_default();
                for (batch_pos, loss_val) in per_sample_loss.iter().enumerate() {
                    let seg_idx = preloaded_batch
                        .segment_indices
                        .get(batch_pos)
                        .copied()
                        .unwrap_or(0);
                    smurf_ema_updates.push((seg_idx, *loss_val));
                }
            }

            let loss_unsqueezed = loss.clone().unsqueeze::<1>();
            accumulated_loss = Some(match accumulated_loss {
                Some(acc) => acc + loss_unsqueezed,
                None => loss_unsqueezed,
            });
            accumulated_count += 1;
            batch_count += 1;

            if (accumulated_count >= LOSS_SYNC_INTERVAL || batch_count == total_batches)
                && let Some(acc_loss) = accumulated_loss.take()
            {
                let avg_loss = acc_loss / (accumulated_count as f32);
                let loss_value: f32 = avg_loss
                    .into_data()
                    .to_vec()
                    .unwrap_or_else(|_| vec![0.0])
                    .first()
                    .copied()
                    .unwrap_or(0.0);

                epoch_loss_sum += loss_value as f64 * accumulated_count as f64;
                epoch_loss_count += accumulated_count;
                accumulated_count = 0;

                let avg_epoch_loss = epoch_loss_sum / epoch_loss_count as f64;
                let approx_rmse = (avg_epoch_loss as f32).sqrt() * MMR_SCALE;
                let n = batch_count as f64;
                info!(
                    "  Batch {batch_count}/{total_batches}, avg_loss={avg_epoch_loss:.6} (~{approx_rmse:.1} MMR RMSE) | \
                     prefetch={:.1}ms, to_gpu={:.1}ms, forward={:.1}ms, backward={:.1}ms, optim={:.1}ms (per batch avg)",
                    time_prefetch_wait_us as f64 / 1000.0 / n,
                    time_to_gpu_us as f64 / 1000.0 / n,
                    time_forward_us as f64 / 1000.0 / n,
                    time_backward_us as f64 / 1000.0 / n,
                    time_optimizer_us as f64 / 1000.0 / n,
                );
            }

            let t_backward_start = Instant::now();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            time_backward_us += t_backward_start.elapsed().as_micros() as u64;

            let t_optimizer_start = Instant::now();
            *model = optimizer.step(lr, model.clone(), grads);
            time_optimizer_us += t_optimizer_start.elapsed().as_micros() as u64;
        }

        // Update smurf EMA and refresh masks.
        if epoch >= SMURF_MASK_START_EPOCH {
            smurf_state.update_ema(&smurf_ema_updates);
            smurf_state.refresh_masks();
        }

        // Log epoch timing summary
        let batches = batch_count as f64;
        if batches > 0.0 {
            info!(
                "  Epoch timing breakdown (avg per batch): prefetch={:.2}ms, to_gpu={:.2}ms, forward={:.2}ms, backward={:.2}ms, optim={:.2}ms",
                time_prefetch_wait_us as f64 / 1000.0 / batches,
                time_to_gpu_us as f64 / 1000.0 / batches,
                time_forward_us as f64 / 1000.0 / batches,
                time_backward_us as f64 / 1000.0 / batches,
                time_optimizer_us as f64 / 1000.0 / batches,
            );
        }

        let epoch_duration = epoch_start.elapsed();
        state.current_train_loss = if epoch_loss_count > 0 {
            (epoch_loss_sum / epoch_loss_count as f64) as f32
        } else {
            0.0
        };
        state.current_epoch = epoch;

        // Huber loss is on normalised [0,1] targets; multiply sqrt by MMR_SCALE for
        // an approximate RMSE in MMR units (exact only in the quadratic regime).
        let approx_train_rmse = state.current_train_loss.sqrt() * MMR_SCALE;
        info!(
            "Epoch {epoch} completed in {:.2}s, loss={:.6} (~{:.1} MMR RMSE), lr={lr:.2e}",
            epoch_duration.as_secs_f64(),
            state.current_train_loss,
            approx_train_rmse,
        );

        // Validation
        let mut improved = false;
        if let Some(valid_ds) = valid_dataset {
            let valid_start = Instant::now();
            let inner_model = model.valid();
            let inner_device = inner_model.device();
            let valid_batcher =
                SequenceBatcher::<B::InnerBackend>::new(inner_device, config.sequence_length);

            let validation_result =
                compute_validation_loss(&inner_model, valid_ds, &valid_batcher, config.batch_size);
            let validation_time = valid_start.elapsed();
            state.current_valid_loss = Some(validation_result.loss);
            state.last_validation_rank_rmse = Some(validation_result.rank_rmse_entries);

            // Early stopping check
            if validation_result.loss < state.best_valid_loss {
                state.best_valid_loss = validation_result.loss;
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
                        let validation_metrics = checkpoint_validation_metrics_from_state(&state);
                        if let Ok(_ckpt) = save_checkpoint(model, &path, config, validation_metrics)
                        {
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

                let validation_metrics = checkpoint_validation_metrics_from_state(&state);
                match save_checkpoint(model, &path, config, validation_metrics) {
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
        let validation_metrics = checkpoint_validation_metrics_from_state(&state);
        if let Ok(_ckpt) = save_checkpoint(model, &path, config, validation_metrics) {
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

/// Per-rank error accumulator for validation diagnostics.
#[derive(Default)]
struct PerRankErrors {
    squared_error_sum: f64,
    count: usize,
}

impl PerRankErrors {
    fn rmse(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        ((self.squared_error_sum / self.count as f64) as f32).sqrt()
    }
}

/// Collects squared errors per [`Rank`] from raw prediction/target vectors.
///
/// Targets with value `0.0` (sentinel for unknown rank) are skipped.
fn accumulate_per_rank_errors(
    stats: &mut std::collections::HashMap<Rank, PerRankErrors>,
    predictions: &[f32],
    targets: &[f32],
) {
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        if *target <= 0.0 {
            continue;
        }
        let rank = Rank::from(RankDivision::from(*target));
        let entry = stats.entry(rank).or_default();
        let err = (*pred - *target) as f64;
        entry.squared_error_sum += err * err;
        entry.count += 1;
    }
}

fn log_per_rank_breakdown(stats: &std::collections::HashMap<Rank, PerRankErrors>) {
    info!("  Validation RMSE by rank:");
    for rank in Rank::all_ranked() {
        if let Some(entry) = stats.get(&rank)
            && entry.count > 0
        {
            info!(
                "    {:<20} {:>7.1} MMR RMSE  (n={:>6})",
                rank.as_api_string(),
                entry.rmse(),
                entry.count
            );
        }
    }
}

/// Builds ordered per-rank RMSE entries for checkpoint JSON (same order as logs).
fn validation_rank_rmse_entries_from_stats(
    stats: &std::collections::HashMap<Rank, PerRankErrors>,
) -> Vec<ValidationRankRmseEntry> {
    let mut entries = Vec::new();
    for rank in Rank::all_ranked() {
        if let Some(per_rank) = stats.get(&rank)
            && per_rank.count > 0
        {
            entries.push(ValidationRankRmseEntry {
                rank: rank.as_api_string().to_string(),
                rmse_mmr: f64::from(per_rank.rmse()),
                sample_count: per_rank.count as u64,
            });
        }
    }
    entries
}

/// Result of one full validation pass: aggregate loss and per-rank RMSE.
struct ValidationLossResult {
    /// Mean validation loss (normalized Huber / MSE-style aggregate).
    pub loss: f32,
    /// Per-rank RMSE in MMR units, ladder order.
    pub rank_rmse_entries: Vec<ValidationRankRmseEntry>,
}

/// Computes validation loss on a segment dataset.
///
/// Also logs a per-rank-tier RMSE breakdown for diagnostics.
fn compute_validation_loss<B: Backend>(
    model: &SequenceModel<B>,
    dataset: &Arc<SegmentStore>,
    batcher: &SequenceBatcher<B>,
    batch_size: usize,
) -> ValidationLossResult {
    let num_segments = dataset.len();
    if num_segments == 0 {
        return ValidationLossResult {
            loss: 0.0,
            rank_rmse_entries: Vec::new(),
        };
    }

    let mut total_loss = 0.0;
    let mut total_samples = 0;
    let mut rank_stats: std::collections::HashMap<Rank, PerRankErrors> =
        std::collections::HashMap::new();

    let indices: Vec<usize> = (0..num_segments).collect();

    for batch_start in (0..num_segments).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_segments);

        let Some(batch_indices) = indices.get(batch_start..batch_end) else {
            continue;
        };

        let Some(batch) = batcher.batch_from_indices(dataset, batch_indices) else {
            continue;
        };

        let predictions = model.forward(batch.inputs);

        // Extract raw values for per-rank breakdown before normalizing.
        let raw_preds: Vec<f32> = predictions.clone().into_data().to_vec().unwrap_or_default();
        let raw_targets: Vec<f32> = batch
            .targets
            .clone()
            .into_data()
            .to_vec()
            .unwrap_or_default();
        accumulate_per_rank_errors(&mut rank_stats, &raw_preds, &raw_targets);

        // Mask unknown-rank slots (target == 0.0 sentinel).
        let mask = batch.targets.clone().greater_elem(0.0).float();
        let known_count = mask
            .clone()
            .sum()
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![1.0f32])
            .first()
            .copied()
            .unwrap_or(1.0f32)
            .max(1.0f32);

        let targets_norm = batch.targets / MMR_SCALE;
        let predictions_norm = predictions / MMR_SCALE;
        let diff = predictions_norm - targets_norm;
        let squared = diff.powf_scalar(2.0) * mask;
        let mse = squared.sum() / known_count;

        let loss_value: f32 = mse
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![0.0])
            .first()
            .copied()
            .unwrap_or(0.0);

        total_loss += loss_value as f64;
        total_samples += 1;
    }

    log_per_rank_breakdown(&rank_stats);

    let rank_rmse_entries = validation_rank_rmse_entries_from_stats(&rank_stats);

    let loss = if total_samples > 0 {
        (total_loss / total_samples as f64) as f32
    } else {
        0.0
    };

    ValidationLossResult {
        loss,
        rank_rmse_entries,
    }
}

/// Computes a cosine-decayed learning rate.
///
/// Starts at `base_lr` and decays smoothly to near zero over `total_epochs` epochs.
/// The `start_epoch` parameter supports resuming from a checkpoint.
fn cosine_lr(base_lr: f64, current_epoch: usize, start_epoch: usize, total_epochs: usize) -> f64 {
    if total_epochs <= start_epoch {
        return base_lr;
    }
    let progress = (current_epoch - start_epoch) as f64 / (total_epochs - start_epoch) as f64;
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Returns a deterministic pseudo-random float in [0, 1) from two seeds.
///
/// Used for per-batch lobby-bias dropout decisions (reproducible across runs).
fn pseudo_random_f32(seed_a: u64, seed_b: u64) -> f32 {
    let mut state = seed_a.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(
        seed_b
            .wrapping_mul(1_442_695_040_888_963_407)
            .wrapping_add(1),
    );
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51afd7ed558ccd);
    state ^= state >> 33;
    ((state >> 11) as f32) / (1u64 << 53) as f32
}

/// Logs training progress.
///
/// Loss values are on normalised [0, 1] targets (Huber loss).
/// The approximate RMSE in MMR is `sqrt(loss) * MMR_SCALE`; this is exact only
/// in the quadratic regime of the Huber loss (errors < delta = 0.1 normalised).
fn log_progress(epoch: usize, train_loss: f32, valid_loss: Option<f32>) {
    let approx_train_rmse = train_loss.sqrt() * MMR_SCALE;
    if let Some(vl) = valid_loss {
        let approx_valid_rmse = vl.sqrt() * MMR_SCALE;
        info!(
            "Epoch {epoch}: train_loss={train_loss:.6} (~{approx_train_rmse:.1} MMR), \
             valid_loss={vl:.6} (~{approx_valid_rmse:.1} MMR)"
        );
    } else {
        info!("Epoch {epoch}: train_loss={train_loss:.6} (~{approx_train_rmse:.1} MMR)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_random_f32_range() {
        for a in 0u64..5 {
            for b in 0u64..5 {
                let v = pseudo_random_f32(a, b);
                assert!(
                    (0.0..1.0).contains(&v),
                    "pseudo_random_f32 out of [0,1): {v}"
                );
            }
        }
        // Different seeds should produce different values.
        let v0 = pseudo_random_f32(0, 0);
        let v1 = pseudo_random_f32(1, 0);
        assert!((v0 - v1).abs() > f32::EPSILON);
    }
}
