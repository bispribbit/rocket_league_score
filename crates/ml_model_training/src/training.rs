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
use tracing::info;

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
/// Increased to allow more time for parallel I/O loading.
const PREFETCH_COUNT: usize = 8;

/// How often to sync with GPU to extract loss value (every N batches).
/// Higher values = better GPU utilization but less frequent progress updates.
const LOSS_SYNC_INTERVAL: usize = 10;

/// Approximate center of the MMR distribution (roughly Gold 1 / low Plat).
///
/// This is where the per-rank RMSE is consistently lowest, so the model already
/// handles this region well without extra weighting.
const WEIGHT_CENTER_MMR: f32 = 800.0;

/// Maximum extra weight applied at each tail (Bronze 1 and SSL).
///
/// With these values:
///   Bronze 1 (~200 MMR)  → distance ≈ 0.75 → raw weight ≈ 2.69×
///   Gold 1   (~800 MMR)  → distance  = 0.0  → raw weight  = 1.0× (min)
///   SSL      (~1941 MMR) → distance ≈ 1.0   → raw weight ≈ 3.5×
const WEIGHT_BOOST: f32 = 3.5;

/// Computes per-sample loss weights from mean target MMR using a symmetric
/// U-shaped curve centred on [`WEIGHT_CENTER_MMR`].
///
/// Both tails (low Bronze and SSL) receive elevated weights so the model is
/// pushed to represent the full rank ladder rather than collapsing toward the
/// centre of the distribution.  Weights are normalised to mean 1.0 so the
/// effective learning rate is unchanged.
fn compute_mmr_weights<B: Backend>(mean_mmr: Tensor<B, 2>) -> Tensor<B, 2> {
    // Normalised distance from centre: 0 at WEIGHT_CENTER_MMR, 1 at the farther tail.
    let max_distance = (MMR_SCALE - WEIGHT_CENTER_MMR).max(WEIGHT_CENTER_MMR);
    let distance = (mean_mmr - WEIGHT_CENTER_MMR).abs() / max_distance;
    let raw_weights = distance.powf_scalar(2.0) * WEIGHT_BOOST + 1.0;

    let mean_weight: f32 = raw_weights
        .clone()
        .mean()
        .into_data()
        .to_vec()
        .unwrap_or_else(|_| vec![1.0])
        .first()
        .copied()
        .unwrap_or(1.0);

    raw_weights / mean_weight
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

    // Huber loss on normalised [0,1] targets (raw MMR / 2000).
    //
    // Overfit sweep (100 segments, 30 epochs, lr=1e-2, no clip):
    //   delta=0.1 → FLAT  (+1.7 MMR, gradient ±0.1 too weak for 1.25M params)
    //   delta=0.5 → OK    (820→309 MMR RMSE)
    //   delta=1.0 → BEST  (867→307 MMR RMSE, fastest convergence)   <-- chosen
    //
    // delta=1.0 in normalised space = 2000 MMR, so virtually all errors stay in
    // the quadratic regime.  This makes the loss behave like MSE for typical
    // predictions while still capping the gradient for extreme outliers (e.g.
    // corrupt replays with wildly wrong MMR labels).
    //
    // Computed element-wise (not via HuberLoss struct) so we can apply
    // per-sample MMR-based weights before reduction.
    let huber_delta = 1.0_f32;

    let mut checkpoint_paths = Vec::new();

    // Initialize training state
    let mut state = start_state.unwrap_or_else(|| TrainingState::new(0));
    let start_epoch = state.current_epoch;

    const EARLY_STOPPING_PATIENCE: usize = 10;

    let valid_segments = valid_dataset.map_or(0, |ds| ds.len());
    info!(
        "Starting training with {} train segments, {} valid segments",
        train_dataset.len(),
        valid_segments
    );
    info!(
        "Batch size: {}, Learning rate: {}, Segment length: {}, Prefetch: {}",
        config.batch_size, config.learning_rate, config.sequence_length, PREFETCH_COUNT
    );
    info!(
        "MMR loss weighting enabled: center={WEIGHT_CENTER_MMR}, boost={WEIGHT_BOOST} (symmetric U-shape)"
    );
    if start_epoch > 0 {
        info!("Resuming from epoch {}", start_epoch + 1);
    }

    let num_samples = train_dataset.len();

    for epoch in start_epoch..config.epochs {
        let epoch_start = Instant::now();
        let mut batch_count = 0;

        // Create shuffled indices for this epoch
        let mut indices: Vec<usize> = (0..num_samples).collect();
        shuffle_indices(&mut indices, epoch as u64);

        // Create prefetcher for this epoch - batches will be loaded in background
        let mut prefetcher = BatchPrefetcher::new(
            train_dataset.clone(),
            indices,
            config.batch_size,
            config.sequence_length,
            PREFETCH_COUNT,
        );

        let total_batches = prefetcher.total_batches();

        // Cosine learning rate decay: starts at base_lr and decays to ~0 by the last epoch.
        let lr = cosine_lr(config.learning_rate, epoch, start_epoch, config.epochs);

        // Accumulate loss on GPU to avoid sync on every batch
        let mut accumulated_loss: Option<Tensor<B, 1>> = None;
        let mut accumulated_count = 0;
        let mut epoch_loss_sum = 0.0f64;
        let mut epoch_loss_count = 0usize;

        // Timing accumulators (in microseconds)
        let mut time_prefetch_wait_us = 0u64;
        let mut time_to_gpu_us = 0u64;
        let mut time_forward_us = 0u64;
        let mut time_backward_us = 0u64;
        let mut time_optimizer_us = 0u64;

        // Process batches from prefetcher
        loop {
            // Time waiting for prefetcher
            let t_prefetch_start = Instant::now();
            let Some(preloaded_batch) = prefetcher.next_batch() else {
                break;
            };
            time_prefetch_wait_us += t_prefetch_start.elapsed().as_micros() as u64;

            // Time converting to GPU tensors
            let t_to_gpu_start = Instant::now();
            let batch = preloaded_batch.to_batch::<B>(&device);
            time_to_gpu_us += t_to_gpu_start.elapsed().as_micros() as u64;

            // Time for forward pass
            let t_forward_start = Instant::now();
            let predictions = model.forward(batch.inputs);

            // Mask: 1.0 where the target is known (> 0), 0.0 for sentinel slots.
            // This prevents unknown-rank players from contributing any gradient.
            let mask = batch.targets.clone().greater_elem(0.0).float(); // [batch, 6]
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

            // Per-sample weights based on mean MMR of *known* slots only.
            let known_per_row = mask.clone().sum_dim(1).clamp_min(1.0); // [batch, 1]
            let masked_targets = batch.targets.clone() * mask.clone();
            let mean_target_mmr = masked_targets.clone().sum_dim(1) / known_per_row; // [batch, 1]
            let weights = compute_mmr_weights::<B>(mean_target_mmr); // [batch, 1]

            let targets_norm = batch.targets / MMR_SCALE;
            let predictions_norm = predictions / MMR_SCALE;

            let diff = predictions_norm - targets_norm;
            let abs_diff = diff.clone().abs();
            let clamped = abs_diff.clone().clamp_min(0.0).clamp_max(huber_delta);
            let element_loss =
                clamped.clone().powf_scalar(2.0) * 0.5 + (abs_diff - clamped) * huber_delta;

            // Zero out loss for unknown-rank slots, apply per-sample weights, then
            // average over the number of *known* elements (not total elements).
            let loss = (element_loss * mask * weights).sum() / known_count;
            time_forward_us += t_forward_start.elapsed().as_micros() as u64;

            // Accumulate loss on GPU (no sync)
            let loss_unsqueezed = loss.clone().unsqueeze::<1>();
            accumulated_loss = Some(match accumulated_loss {
                Some(acc) => acc + loss_unsqueezed,
                None => loss_unsqueezed,
            });
            accumulated_count += 1;
            batch_count += 1;

            // Only sync with GPU periodically to get loss value
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

                // Log progress with timing breakdown
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

            // Time for backward pass
            let t_backward_start = Instant::now();
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            time_backward_us += t_backward_start.elapsed().as_micros() as u64;

            // Time for optimizer step
            let t_optimizer_start = Instant::now();
            *model = optimizer.step(lr, model.clone(), grads);
            time_optimizer_us += t_optimizer_start.elapsed().as_micros() as u64;
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
