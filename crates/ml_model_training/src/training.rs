//! Training logic for the sequence model.

use std::sync::Arc;
use std::time::Instant;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use ml_model::{MMR_SCALE, SequenceModel, TrainingConfig};
use replay_structs::{Rank, RankDivision};
use tracing::{info, warn};

use crate::dataset::{BatchPrefetcher, SequenceBatcher};
use crate::minibatch_loss::{LabelJitterStep, production_training_minibatch_loss};
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
    /// Standard deviation of predictions from the last validation pass (raw MMR units).
    ///
    /// Propagated into [`TrainingOutput::final_validation_pred_std_mmr`] so the
    /// warm-start basin-escape gate can use the real computed value instead of a heuristic.
    pub last_validation_pred_std_mmr: Option<f32>,
    /// Reference per-rank validation RMSE (MMR units) captured at the end of the
    /// warm-start phase, used as the **tail-rank guard** baseline.
    ///
    /// When this is set, each validation pass during main training compares the
    /// tail-rank RMSEs (Bronze-1, GC-3, SSL) against this baseline and logs a warning
    /// if they degrade by more than [`TAIL_RANK_GUARD_DEGRADATION_FRACTION`]. The guard
    /// is monitoring-only: it does **not** stop training or revert the model.
    pub tail_rank_baseline: Option<Vec<ValidationRankRmseEntry>>,
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
            last_validation_pred_std_mmr: None,
            tail_rank_baseline: None,
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
    /// Per-rank validation RMSE from the last validation pass (MMR units).
    ///
    /// Used by the full-pipeline warm-start phase to seed
    /// [`TrainingState::tail_rank_baseline`] for the subsequent main-training run.
    pub final_validation_rank_rmse: Option<Vec<ValidationRankRmseEntry>>,
    /// Standard deviation of model predictions over all known validation slots (raw MMR).
    ///
    /// Near-zero values indicate the model is collapsed to a constant. Used by the
    /// warm-start basin-escape gate in `full_pipeline.rs`. `None` when no validation
    /// set is available.
    pub final_validation_pred_std_mmr: Option<f32>,
}

/// Number of batches to prefetch ahead (keep GPU fed while loading next batches).
const PREFETCH_COUNT: usize = 8;

/// How often to sync with GPU to extract loss value (every N batches).
const LOSS_SYNC_INTERVAL: usize = 10;

/// Tail ranks watched by the **tail-rank guard** (see [`TrainingState::tail_rank_baseline`]).
///
/// These are the ranks where mean-collapse hurts the most because they sit at the
/// extremes of the MMR distribution and have very few labelled examples relative to
/// the centre of the ladder. A warm-start baseline that has any signal here is the
/// minimum bar; if main training erodes it past
/// [`TAIL_RANK_GUARD_DEGRADATION_FRACTION`] we want a loud signal in the logs.
const TAIL_RANK_GUARD_API_STRINGS: &[&str] = &["bronze-1", "grand-champion-3", "supersonic-legend"];

/// Fractional degradation (relative to the warm-start baseline RMSE) above which the
/// tail-rank guard logs a warning. `0.25` = "current tail RMSE is more than 125 % of
/// the baseline tail RMSE".
const TAIL_RANK_GUARD_DEGRADATION_FRACTION: f64 = 0.25;

/// Epoch at which EMA-based smurf masking kicks in, or [`None`] to disable masking.
///
/// Currently [`None`]: smurf masking preferentially drops the high-EMA-loss segments
/// at the distribution tails (Bronze-1 and SSL) when the model is in a mean-prediction
/// basin, which reinforces the collapse instead of breaking it. See
/// `docs/experiment.md` for the analysis. Re-enable by setting to `Some(start_epoch)`
/// once the per-rank predicted-distribution log shows the model is no longer collapsed.
const SMURF_MASK_START_EPOCH: Option<usize> = None;
/// Number of consecutive epochs a segment must be in the top-1 % before masking.
const SMURF_MASK_SUSTAIN_EPOCHS: usize = 3;
/// EMA decay factor for per-segment loss tracking (α for new value).
const SMURF_EMA_ALPHA: f32 = 0.3;

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
pub fn compute_inverse_frequency_weights(dataset: &SegmentStore) -> Vec<f32> {
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

/// Returns up to `target_per_rank` segment indices for each of the 23 numeric ranks.
///
/// Pick is deterministic (linear scan in entry order) and skips ranks with zero
/// segments. Used by the warm-start phase to build a tiny balanced mini-set for the
/// pre-training pass without touching the on-disk cache.
#[must_use]
pub fn balanced_segment_indices(dataset: &SegmentStore, target_per_rank: usize) -> Vec<usize> {
    if target_per_rank == 0 || dataset.is_empty() {
        return Vec::new();
    }
    let mut per_rank_counts = [0usize; 23];
    let mut picked = Vec::new();
    for index in 0..dataset.len() {
        let rank_index = dataset.get_primary_rank_index(index).unwrap_or(0) as usize;
        let Some(count) = per_rank_counts.get_mut(rank_index) else {
            continue;
        };
        if *count >= target_per_rank {
            continue;
        }
        *count += 1;
        picked.push(index);
    }
    picked
}

/// Looks up per-sample inverse-frequency weights on CPU and returns them as a Vec.
///
/// `mean_target_mmr` is in raw MMR units (not normalised).
pub fn lookup_rank_weights(mean_target_mmr_slice: &[f32], rank_weights: &[f32]) -> Vec<f32> {
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
pub fn train<B: AutodiffBackend + ml_model::fused_lstm::FusedLstmBackend>(
    model: &mut SequenceModel<B>,
    train_dataset: Arc<SegmentStore>,
    valid_dataset: Option<&Arc<SegmentStore>>,
    config: &TrainingConfig,
    checkpoint_config: Option<CheckpointConfig>,
    start_state: Option<TrainingState>,
) -> anyhow::Result<TrainingOutput>
where
    B::FloatElem: From<f32>,
    B::InnerBackend: ml_model::fused_lstm::FusedLstmBackend,
{
    if train_dataset.is_empty() {
        return Err(anyhow::anyhow!("No training data provided"));
    }

    let device = model.device();

    // Adam with gradient clipping to stabilise LSTM training.
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

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

    // Smurf-masking state, only allocated when [`SMURF_MASK_START_EPOCH`] is enabled.
    // The allocation scales with the dataset size (one bool/float/u32 per segment), so
    // the [`None`] path skips it to keep training memory minimal.
    let mut smurf_state = SMURF_MASK_START_EPOCH.map(|_| SmurfMaskState::new(num_samples));

    const EARLY_STOPPING_PATIENCE: usize = 1000;

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
        let indices: Vec<usize> = if let Some(smurf_state) = smurf_state.as_ref() {
            oversampled_indices
                .into_iter()
                .filter(|&idx| smurf_state.active.get(idx).copied().unwrap_or(true))
                .collect()
        } else {
            oversampled_indices
        };

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
        // Stays empty (and the per-row tensor read below is skipped) while smurf masking is disabled.
        let mut smurf_ema_updates: Vec<(usize, f32)> = Vec::new();
        let smurf_masking_active_this_epoch =
            SMURF_MASK_START_EPOCH.is_some_and(|start| epoch >= start);

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
            let minibatch_out = production_training_minibatch_loss(
                model,
                &batch,
                &device,
                &rank_weights,
                lobby_scale,
                LabelJitterStep {
                    epoch: epoch as u64,
                    batch_in_epoch: batch_count as u64,
                },
            );
            time_forward_us += t_forward_start.elapsed().as_micros() as u64;

            let loss = minibatch_out.loss;

            if smurf_masking_active_this_epoch {
                let per_sample_loss = minibatch_out
                    .per_row_mse_for_smurf
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

            accumulated_loss = Some(match accumulated_loss {
                Some(acc) => acc + loss.clone(),
                None => loss.clone(),
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
        if smurf_masking_active_this_epoch && let Some(smurf_state) = smurf_state.as_mut() {
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
            state.last_validation_pred_std_mmr = Some(validation_result.pred_std_mmr);
            if let Some(baseline) = state.tail_rank_baseline.as_deref() {
                log_tail_rank_guard(baseline, &validation_result.rank_rmse_entries);
            }
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
                        final_validation_rank_rmse: state.last_validation_rank_rmse,
                        final_validation_pred_std_mmr: state.last_validation_pred_std_mmr,
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
        final_validation_rank_rmse: state.last_validation_rank_rmse,
        final_validation_pred_std_mmr: state.last_validation_pred_std_mmr,
    })
}

/// Per-rank error accumulator for validation diagnostics.
///
/// `squared_error_sum` and `target_count` are keyed by **true rank** so that `rmse()`
/// is the RMSE for that rank's labelled slots.
/// `predicted_count` / `correct_count` are keyed by **regression-predicted rank** (rank
/// of the raw MMR output) — they show the regression head's predicted distribution and
/// hit count per bucket, making mean-collapse immediately visible.
/// `ordinal_predicted_count` / `ordinal_correct_count` are keyed by **ordinal-predicted
/// rank** (rank inferred from the auxiliary ordinal head's logits, independent of the
/// regression head). When the regression head is stuck in a constant-predictor saddle,
/// the ordinal columns reveal whether the shared LSTM is still learning to differentiate
/// ranks via the ordinal classification objective.
#[derive(Default)]
struct PerRankErrors {
    squared_error_sum: f64,
    /// Number of labelled slots whose **true** rank is this bucket.
    target_count: usize,
    /// Number of labelled slots whose **regression-predicted** rank lands in this bucket.
    predicted_count: usize,
    /// Number of labelled slots in `predicted_count` where the true rank also matches.
    correct_count: usize,
    /// Number of labelled slots whose **ordinal-predicted** rank lands in this bucket.
    ordinal_predicted_count: usize,
    /// Number of labelled slots in `ordinal_predicted_count` where the true rank matches.
    ordinal_correct_count: usize,
}

impl PerRankErrors {
    fn rmse(&self) -> f32 {
        if self.target_count == 0 {
            return 0.0;
        }
        ((self.squared_error_sum / self.target_count as f64) as f32).sqrt()
    }
}

/// Maps a count of positive ordinal logits (`0..=ORDINAL_NUM_BOUNDARIES`) to a [`Rank`].
///
/// The ordinal head emits 21 boundary logits; with `n` of them positive, the model
/// predicts the player is above `n` boundaries, i.e. the `n`-th rank in `all_ranked()`
/// order (Bronze-1 at `n=0`, SSL at `n=21`).
fn ordinal_rank_from_positive_count(positive_count: usize) -> Rank {
    let ranks: Vec<Rank> = Rank::all_ranked().collect();
    let idx = positive_count.min(ranks.len().saturating_sub(1));
    ranks.get(idx).copied().unwrap_or(Rank::Bronze1)
}

/// Collects squared errors per **true** [`Rank`] and prediction/correct counts per
/// **predicted** [`Rank`] from raw prediction/target vectors.
///
/// Targets with value `0.0` (sentinel for unknown rank) are skipped: those slots have
/// no ground truth so they would distort both the RMSE and the predicted histograms.
/// Skipping them keeps `sum(predicted_count) == sum(target_count)` and similarly for
/// the ordinal counters.
///
/// `ordinal_logits` is the flattened `[num_slots, ORDINAL_NUM_BOUNDARIES]` ordinal-head
/// output for the same slots as `predictions` / `targets`. When [`None`], the ordinal
/// diagnostic is skipped (used by callers that haven't routed the auxiliary head).
fn accumulate_per_rank_errors(
    stats: &mut std::collections::HashMap<Rank, PerRankErrors>,
    predictions: &[f32],
    targets: &[f32],
    ordinal_logits: Option<&[f32]>,
) {
    for (slot_idx, (pred, target)) in predictions.iter().zip(targets.iter()).enumerate() {
        if *target <= 0.0 {
            continue;
        }
        let true_rank = Rank::from(RankDivision::from(*target));
        let true_entry = stats.entry(true_rank).or_default();
        let err = (*pred - *target) as f64;
        true_entry.squared_error_sum += err * err;
        true_entry.target_count += 1;

        let predicted_rank = Rank::from(RankDivision::from(*pred));
        let predicted_entry = stats.entry(predicted_rank).or_default();
        predicted_entry.predicted_count += 1;
        if predicted_rank == true_rank {
            predicted_entry.correct_count += 1;
        }

        if let Some(logits) = ordinal_logits {
            let start = slot_idx * ml_model::ORDINAL_NUM_BOUNDARIES;
            let end = start + ml_model::ORDINAL_NUM_BOUNDARIES;
            if let Some(slot_logits) = logits.get(start..end) {
                let positive = slot_logits.iter().filter(|&&l| l > 0.0).count();
                let ordinal_rank = ordinal_rank_from_positive_count(positive);
                let ordinal_entry = stats.entry(ordinal_rank).or_default();
                ordinal_entry.ordinal_predicted_count += 1;
                if ordinal_rank == true_rank {
                    ordinal_entry.ordinal_correct_count += 1;
                }
            }
        }
    }
}

fn log_per_rank_breakdown(stats: &std::collections::HashMap<Rank, PerRankErrors>) {
    info!(
        "  Validation RMSE by rank (reg = regression head; ord = ordinal head; \
         p=predicted count, v=valid count):"
    );
    for rank in Rank::all_ranked() {
        let Some(entry) = stats.get(&rank) else {
            continue;
        };
        if entry.target_count == 0
            && entry.predicted_count == 0
            && entry.ordinal_predicted_count == 0
        {
            continue;
        }
        info!(
            "    {:<20} {:>7.1} MMR RMSE  reg(p={:>6}, v={:>6})  ord(p={:>6}, v={:>6})",
            rank.as_api_string(),
            entry.rmse(),
            entry.predicted_count,
            entry.correct_count,
            entry.ordinal_predicted_count,
            entry.ordinal_correct_count,
        );
    }
}

/// Builds ordered per-rank RMSE entries for checkpoint JSON (same order as logs).
///
/// `sample_count` records the number of **labelled slots** whose true rank is this
/// bucket — i.e. the denominator of `rmse_mmr`. The predicted-distribution count is
/// a diagnostic that lives only in the training logs, not in the checkpoint payload.
fn validation_rank_rmse_entries_from_stats(
    stats: &std::collections::HashMap<Rank, PerRankErrors>,
) -> Vec<ValidationRankRmseEntry> {
    let mut entries = Vec::new();
    for rank in Rank::all_ranked() {
        if let Some(per_rank) = stats.get(&rank)
            && per_rank.target_count > 0
        {
            entries.push(ValidationRankRmseEntry {
                rank: rank.as_api_string().to_string(),
                rmse_mmr: f64::from(per_rank.rmse()),
                sample_count: per_rank.target_count as u64,
            });
        }
    }
    entries
}

/// Compares the current validation per-rank RMSEs against the warm-start baseline for
/// the watched tail ranks and logs a warning per rank that has degraded past
/// [`TAIL_RANK_GUARD_DEGRADATION_FRACTION`].
///
/// Monitoring only: never modifies the model or stops training. Intended to flag
/// "warm-start broke us out of the basin and then main training pushed us back in"
/// so we can correlate the regression with epoch / LR / loss changes.
fn log_tail_rank_guard(baseline: &[ValidationRankRmseEntry], current: &[ValidationRankRmseEntry]) {
    for rank_label in TAIL_RANK_GUARD_API_STRINGS {
        let baseline_entry = baseline.iter().find(|e| e.rank == *rank_label);
        let current_entry = current.iter().find(|e| e.rank == *rank_label);
        let (Some(baseline_entry), Some(current_entry)) = (baseline_entry, current_entry) else {
            continue;
        };
        if baseline_entry.rmse_mmr <= 0.0 {
            continue;
        }
        let degradation =
            (current_entry.rmse_mmr - baseline_entry.rmse_mmr) / baseline_entry.rmse_mmr;
        if degradation > TAIL_RANK_GUARD_DEGRADATION_FRACTION {
            warn!(
                rank = rank_label,
                baseline_rmse_mmr = baseline_entry.rmse_mmr,
                current_rmse_mmr = current_entry.rmse_mmr,
                degradation_fraction = degradation,
                threshold_fraction = TAIL_RANK_GUARD_DEGRADATION_FRACTION,
                "Tail-rank guard: rank RMSE has degraded past threshold relative to warm-start baseline"
            );
        }
    }
}

/// Result of one full validation pass: aggregate loss, per-rank RMSE, and collapse metrics.
struct ValidationLossResult {
    /// Mean validation loss (normalized Huber / MSE-style aggregate).
    pub loss: f32,
    /// Per-rank RMSE in MMR units, ladder order.
    pub rank_rmse_entries: Vec<ValidationRankRmseEntry>,
    /// Standard deviation of predictions over all known slots (raw MMR).
    /// `0.0` when fewer than two known slots are available.
    pub pred_std_mmr: f32,
}

/// Collapse-detection metrics computed over the full validation set.
///
/// A healthy model shows `pred_std_mmr` well above zero, `pearson_r` approaching 1,
/// and both baseline RMSE values substantially above the model's own RMSE. When all
/// three collapse indicators converge (pred_std ≈ 0, pearson_r ≈ 0, model RMSE ≈
/// constant-predictor RMSE) the model is stuck at the mean.
struct CollapseMetrics {
    /// Standard deviation of predictions over all known slots (raw MMR).
    pred_std_mmr: f32,
    /// Pearson correlation between predictions and targets (known slots only).
    pearson_r: f32,
    /// RMSE of the naive constant predictor (predict the global target mean for every slot).
    constant_predictor_rmse_mmr: f32,
    /// RMSE of the per-lobby mean predictor (predict the mean of each segment's known
    /// targets for every player in that segment). This is the toughest "dumb" baseline
    /// the model must beat to prove it is reading per-player features.
    lobby_mean_predictor_rmse_mmr: f32,
}

/// Gathers per-slot (pred, target) pairs and per-segment lobby data so that
/// [`CollapseMetrics`] can be derived without an extra pass over the dataset.
#[derive(Default)]
struct CollapseAccumulator {
    /// (pred_mmr, target_mmr) for every **known** slot seen during validation.
    known_slots: Vec<(f32, f32)>,
    /// For every segment: the mean target MMR of known slots (used for lobby-mean baseline).
    /// Stored as (sum_sq_error_lobby_mean, count) to allow incremental accumulation.
    lobby_sse: f64,
    lobby_count: u64,
}

impl CollapseAccumulator {
    fn push_batch(&mut self, raw_preds: &[f32], raw_targets: &[f32], batch_size: usize) {
        let slots_per_segment = feature_extractor::TOTAL_PLAYERS;
        for seg in 0..batch_size {
            let start = seg * slots_per_segment;
            let end = start + slots_per_segment;
            let Some(seg_preds) = raw_preds.get(start..end) else {
                continue;
            };
            let Some(seg_targets) = raw_targets.get(start..end) else {
                continue;
            };

            // Known slots in this segment.
            let known: Vec<(f32, f32)> = seg_preds
                .iter()
                .zip(seg_targets.iter())
                .filter(|&(_, t)| *t > 0.0)
                .map(|(&p, &t)| (p, t))
                .collect();

            // Per-segment lobby-mean target (used for the lobby-mean baseline).
            let lobby_target_sum: f32 = known.iter().map(|(_, t)| t).sum();
            let lobby_known_count = known.len();
            if lobby_known_count > 0 {
                let lobby_mean = lobby_target_sum / lobby_known_count as f32;
                for (_, target) in &known {
                    let err = lobby_mean - target;
                    self.lobby_sse += (err * err) as f64;
                }
                self.lobby_count += lobby_known_count as u64;
            }

            self.known_slots.extend_from_slice(&known);
        }
    }

    fn compute_metrics(&self) -> Option<CollapseMetrics> {
        let n = self.known_slots.len();
        if n < 2 {
            return None;
        }
        let n_f = n as f64;

        let sum_p: f64 = self.known_slots.iter().map(|(p, _)| *p as f64).sum();
        let sum_t: f64 = self.known_slots.iter().map(|(_, t)| *t as f64).sum();
        let mean_p = sum_p / n_f;
        let mean_t = sum_t / n_f;

        let var_p: f64 = self
            .known_slots
            .iter()
            .map(|(p, _)| (*p as f64 - mean_p).powi(2))
            .sum::<f64>()
            / (n_f - 1.0);
        let std_p = var_p.sqrt();

        let cov: f64 = self
            .known_slots
            .iter()
            .map(|(p, t)| (*p as f64 - mean_p) * (*t as f64 - mean_t))
            .sum::<f64>()
            / (n_f - 1.0);
        let var_t: f64 = self
            .known_slots
            .iter()
            .map(|(_, t)| (*t as f64 - mean_t).powi(2))
            .sum::<f64>()
            / (n_f - 1.0);
        let std_t = var_t.sqrt();
        let pearson_r = if std_p > 1e-8 && std_t > 1e-8 {
            (cov / (std_p * std_t)).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let constant_sse: f64 = self
            .known_slots
            .iter()
            .map(|(_, t)| (*t as f64 - mean_t).powi(2))
            .sum();
        let constant_predictor_rmse_mmr = (constant_sse / n_f).sqrt() as f32;

        let lobby_mean_predictor_rmse_mmr = if self.lobby_count > 0 {
            (self.lobby_sse / self.lobby_count as f64).sqrt() as f32
        } else {
            0.0
        };

        Some(CollapseMetrics {
            pred_std_mmr: std_p as f32,
            pearson_r: pearson_r as f32,
            constant_predictor_rmse_mmr,
            lobby_mean_predictor_rmse_mmr,
        })
    }
}

/// Computes validation loss on a segment dataset.
///
/// Also logs a per-rank-tier RMSE breakdown and collapse-detection diagnostics.
fn compute_validation_loss<B: Backend + ml_model::fused_lstm::FusedLstmBackend>(
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
            pred_std_mmr: 0.0,
        };
    }

    let mut total_loss = 0.0;
    let mut total_samples = 0;
    let mut rank_stats: std::collections::HashMap<Rank, PerRankErrors> =
        std::collections::HashMap::new();
    let mut collapse = CollapseAccumulator::default();

    let indices: Vec<usize> = (0..num_segments).collect();

    for batch_start in (0..num_segments).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_segments);

        let Some(batch_indices) = indices.get(batch_start..batch_end) else {
            continue;
        };

        let Some(batch) = batcher.batch_from_indices(dataset, batch_indices) else {
            continue;
        };

        let actual_batch_size = batch.targets.dims()[0];

        let (predictions, ordinal_logits) = model.forward_with_ordinal_scale(batch.inputs, 1.0);

        // Extract raw values for per-rank breakdown before normalizing.
        let raw_preds: Vec<f32> = predictions.clone().into_data().to_vec().unwrap_or_default();
        let raw_targets: Vec<f32> = batch
            .targets
            .clone()
            .into_data()
            .to_vec()
            .unwrap_or_default();
        let raw_ordinal_logits: Vec<f32> = ordinal_logits.into_data().to_vec().unwrap_or_default();
        accumulate_per_rank_errors(
            &mut rank_stats,
            &raw_preds,
            &raw_targets,
            Some(&raw_ordinal_logits),
        );
        collapse.push_batch(&raw_preds, &raw_targets, actual_batch_size);

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

    // Log collapse-detection diagnostics after the per-rank table and extract pred_std.
    let pred_std_mmr = if let Some(metrics) = collapse.compute_metrics() {
        info!(
            "  Collapse diagnostics: pred_std={:.1} MMR  pearson_r={:.4}  \
             constant_baseline={:.1} MMR  lobby_mean_baseline={:.1} MMR",
            metrics.pred_std_mmr,
            metrics.pearson_r,
            metrics.constant_predictor_rmse_mmr,
            metrics.lobby_mean_predictor_rmse_mmr,
        );
        if metrics.pred_std_mmr < 50.0 {
            warn!(
                pred_std_mmr = metrics.pred_std_mmr,
                "Collapse alert: prediction std < 50 MMR — model may be stuck at the population mean"
            );
        }
        metrics.pred_std_mmr
    } else {
        0.0
    };

    let rank_rmse_entries = validation_rank_rmse_entries_from_stats(&rank_stats);

    let loss = if total_samples > 0 {
        (total_loss / total_samples as f64) as f32
    } else {
        0.0
    };

    ValidationLossResult {
        loss,
        rank_rmse_entries,
        pred_std_mmr,
    }
}

/// Linear warmup fraction of the total schedule (capped at [`COSINE_LR_WARMUP_MAX_EPOCHS`]).
///
/// Matches the `overfit_wgpu` harness so production and the harness share the same
/// effective LR trajectory. The first epoch starts at [`COSINE_LR_WARMUP_FLOOR`] × `base_lr`
/// and ramps linearly to `base_lr` by the end of warmup.
const COSINE_LR_WARMUP_FRACTION: f64 = 0.25;
const COSINE_LR_WARMUP_MAX_EPOCHS: usize = 20;
/// LR multiplier at the very start of warmup (epoch 0).
const COSINE_LR_WARMUP_FLOOR: f64 = 0.1;
/// Minimum LR multiplier the cosine tail decays to (fraction of `base_lr`).
///
/// Without a floor the tail epochs effectively stop training; the harness uses 0.10
/// (`--lr-floor 0.10` in the validated CLIs) and we mirror that here.
const COSINE_LR_TAIL_FLOOR: f64 = 0.1;

/// Computes a learning rate with linear warmup followed by cosine decay to a floor.
///
/// * Epochs `[start_epoch, start_epoch + warmup_epochs)`: linear ramp from
///   [`COSINE_LR_WARMUP_FLOOR`] × `base_lr` up to `base_lr`.
/// * Epochs `[start_epoch + warmup_epochs, total_epochs)`: cosine from `base_lr` down to
///   [`COSINE_LR_TAIL_FLOOR`] × `base_lr`.
///
/// `start_epoch` supports resuming from a checkpoint — the warmup re-applies after
/// every resume, which is intentional: it gently re-engages the optimizer state.
fn cosine_lr(base_lr: f64, current_epoch: usize, start_epoch: usize, total_epochs: usize) -> f64 {
    if total_epochs <= start_epoch {
        return base_lr;
    }
    let span = total_epochs - start_epoch;
    let warmup_epochs =
        ((span as f64 * COSINE_LR_WARMUP_FRACTION) as usize).min(COSINE_LR_WARMUP_MAX_EPOCHS);
    let local_epoch = current_epoch.saturating_sub(start_epoch);

    if local_epoch < warmup_epochs {
        let ramp = local_epoch as f64 / warmup_epochs.max(1) as f64;
        return base_lr * ramp.mul_add(1.0 - COSINE_LR_WARMUP_FLOOR, COSINE_LR_WARMUP_FLOOR);
    }

    let cosine_span = span.saturating_sub(warmup_epochs).max(1) as f64;
    let progress = (local_epoch - warmup_epochs) as f64 / cosine_span;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    base_lr * cosine.max(COSINE_LR_TAIL_FLOOR)
}

/// Returns a deterministic pseudo-random float in [0, 1) from two seeds.
///
/// Used for per-batch lobby-bias dropout decisions (reproducible across runs).
pub fn pseudo_random_f32(seed_a: u64, seed_b: u64) -> f32 {
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
