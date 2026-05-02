//! Shared minibatch loss for production training and the `overfit_wgpu` harness.
//!
//! Production path: Huber, τ-averaged pinball, rank weights, ordinal, pairwise
//! ([`production_training_minibatch_loss`]).
//! Label jitter is **deterministic** per [`LabelJitterStep`] (epoch + batch index), not Wgpu `Tensor::random`.
//! `--mse-only` ablation: [`mse_ablation_minibatch_loss`] (MSE + minibatch spread; not used by
//! [`super::training::train`]).

use burn::prelude::*;
use burn::tensor::activation;
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{MMR_SCALE, ORDINAL_BOUNDARIES_MMR, ORDINAL_NUM_BOUNDARIES, SequenceModel};
use replay_structs::{Rank, RankDivision};

use crate::dataset::SequenceBatch;

/// Same logic as [`crate::training::lookup_rank_weights`], kept here to avoid a
/// `training` ↔ `minibatch_loss` module cycle.
fn lookup_rank_weights_slice(mean_target_mmr_slice: &[f32], rank_weights: &[f32]) -> Vec<f32> {
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

/// Primary pinball term: average of [`PINBALL_QUANTILES`] on the high-rank mask only (same total
/// scale as the previous single-τ pinball multiplied by this weight).
const PINBALL_WEIGHT: f32 = 0.3;
/// Weaker secondary pinball: same τ average on **all** known slots (Bronze through SSL).
const PINBALL_FULL_BATCH_WEIGHT: f32 = 0.06;
/// Quantiles for τ-averaged pinball: symmetric around 0.5, excluding 0.5 (pure MAE / median pull).
const PINBALL_QUANTILES: [f32; 4] = [0.1, 0.25, 0.75, 0.9];
const PINBALL_THRESHOLD_MMR: f32 = 1400.0;

const ORDINAL_LOSS_WEIGHT: f32 = 0.2;
const PAIRWISE_LOSS_WEIGHT: f32 = 0.1;

/// Penalises minibatches whose **predicted** MMR spread (std of known slots, normalised space) is
/// much **smaller** than the **target** spread. That discourages the shortcut where the model
/// outputs a narrow “bell” around the batch mean instead of matching the width of the label
/// distribution across players in the batch.
const DISTRIBUTION_SPREAD_LOSS_WEIGHT: f32 = 0.2;
/// Distribution spread uses `sqrt(variance)`; keep a **normalized** variance floor high enough that
/// `1 / (2 * sqrt(floor))` does not explode autodiff on tiny fitted variances.
const DISTRIBUTION_SPREAD_EPSILON_NORM: f32 = 1.0e-5;
/// [`mse_ablation_minibatch_loss`] only: no pinball or pairwise, so a stronger spread penalty keeps
/// the harness aligned with full training anti-collapse behaviour.
const MSE_ABLATION_DISTRIBUTION_SPREAD_WEIGHT: f32 = 0.35;

/// Label jitter in MMR, applied in normalised space with mean zero per row.
/// Must match [super::training] `LABEL_JITTER_STD`.
const LABEL_JITTER_STD: f64 = 40.0;

/// Keeps ordinal logits in a range where `log_sigmoid` and gradients stay finite during mixed-rank batches.
const ORDINAL_LOGITS_CLAMP: f32 = 40.0;

/// Row-level boost for the harness T3 MSE ablation: rows whose mean MMR (clean) is
/// above `threshold_mmr` multiply per-element loss by `extra_multiplier`.
#[derive(Debug, Clone)]
pub struct MseExtremeMmrRowBoost {
    pub threshold_mmr: f32,
    pub extra_multiplier: f32,
}

/// Identifies one training minibatch for **deterministic** mean-zero label jitter.
///
/// Jitter values are a pure function of [`LabelJitterStep`] and matrix cell `(row, slot)`, so the
/// same step always yields the same noise on any machine (unlike [`Tensor::random`] on Wgpu).
#[derive(Debug, Clone, Copy)]
pub struct LabelJitterStep {
    pub epoch: u64,
    pub batch_in_epoch: u64,
}

const fn jitter_splitmix64(value: u64) -> u64 {
    let mut accumulator = value.wrapping_add(0x9E3779B97F4A7C15);
    accumulator = (accumulator ^ (accumulator >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    accumulator = (accumulator ^ (accumulator >> 27)).wrapping_mul(0x94D049BB133111EB);
    accumulator ^ (accumulator >> 31)
}

/// Open subinterval `(0, 1)` for Box–Muller (avoids `ln(0)`).
fn jitter_open_unit_interval(seed: u64) -> f64 {
    let word = jitter_splitmix64(seed);
    let mantissa = ((word >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    mantissa.clamp(1.0e-12, 1.0 - 1.0e-12)
}

fn jitter_standard_normal_from_seed(seed: u64) -> f64 {
    let uniform_first = jitter_open_unit_interval(seed);
    let uniform_second = jitter_open_unit_interval(seed ^ 0x1234_5678_9ABC_DEF0);
    (-2.0_f64 * uniform_first.ln()).sqrt() * (std::f64::consts::TAU * uniform_second).cos()
}

fn gaussian_noise_for_label_cell(step: LabelJitterStep, row: usize, slot: usize) -> f32 {
    let key = step
        .epoch
        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
        .wrapping_add(step.batch_in_epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((row as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((slot as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    jitter_standard_normal_from_seed(key) as f32
}

fn mean_zero_label_jitter_normalized<B: burn::tensor::backend::Backend>(
    device: &B::Device,
    batch_size: usize,
    mask: Tensor<B, 2>,
    jitter_step: LabelJitterStep,
) -> Tensor<B, 2> {
    let sigma = (LABEL_JITTER_STD / f64::from(MMR_SCALE)) as f32;
    let mut flat = Vec::with_capacity(batch_size * TOTAL_PLAYERS);
    for row in 0..batch_size {
        for slot in 0..TOTAL_PLAYERS {
            flat.push(gaussian_noise_for_label_cell(jitter_step, row, slot) * sigma);
        }
    }
    let jitter =
        Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch_size, TOTAL_PLAYERS]);
    let row_sum = (jitter.clone() * mask.clone()).sum_dim(1);
    let known = mask.clone().sum_dim(1).clamp_min(1.0);
    let row_mean = row_sum / known;
    (jitter - row_mean) * mask
}

/// Mean pinball loss over several quantiles. `diff` = prediction − target (normalised). For each
/// `quantile`, uses `quantile · relu(−diff) + (1 − quantile) · relu(diff)` (standard check loss).
fn multi_quantile_pinball<B: burn::tensor::backend::Backend>(
    diff: Tensor<B, 2>,
    quantiles: &[f32],
) -> Tensor<B, 2> {
    let target_above_prediction = activation::relu(diff.clone().neg());
    let prediction_above_target = activation::relu(diff);
    let mut quantile_iter = quantiles.iter().copied();
    let first = quantile_iter
        .next()
        .expect("pinball quantiles must not be empty");
    let mut sum = target_above_prediction.clone() * first
        + prediction_above_target.clone() * (1.0f32 - first);
    for quantile in quantile_iter {
        sum = sum
            + target_above_prediction.clone() * quantile
            + prediction_above_target.clone() * (1.0f32 - quantile);
    }
    sum / quantiles.len() as f32
}

/// Masked slots only. Compares standard deviation of **clean** targets vs predictions in
/// normalised space; returns squared `relu(std_target − std_prediction)`.
fn masked_minibatch_spread_loss_squared<B: burn::tensor::backend::Backend>(
    predictions_norm: Tensor<B, 2>,
    targets_norm_clean: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let known_tensor_spread = mask.clone().sum().clamp_min(1.0);
    let mean_slot_target =
        (targets_norm_clean.clone() * mask.clone()).sum() / known_tensor_spread.clone();
    let mean_slot_target_sq =
        (targets_norm_clean.powf_scalar(2.0) * mask.clone()).sum() / known_tensor_spread.clone();
    let var_slot_target = (mean_slot_target_sq - mean_slot_target.powf_scalar(2.0))
        .clamp_min(DISTRIBUTION_SPREAD_EPSILON_NORM);
    let std_slot_target = var_slot_target.sqrt();

    let mean_slot_pred =
        (predictions_norm.clone() * mask.clone()).sum() / known_tensor_spread.clone();
    let mean_slot_pred_sq = (predictions_norm.powf_scalar(2.0) * mask).sum() / known_tensor_spread;
    let var_slot_pred = (mean_slot_pred_sq - mean_slot_pred.powf_scalar(2.0))
        .clamp_min(DISTRIBUTION_SPREAD_EPSILON_NORM);
    let std_slot_pred = var_slot_pred.sqrt();

    let spread_gap = (std_slot_target - std_slot_pred).clamp_min(0.0);
    spread_gap.powf_scalar(2.0)
}

/// Output of the production (full training) loss: combined scalar loss and per-row
/// squared error sums in normalised space for smurf EMA in [`super::training::train`].
pub struct ProductionMinibatchLossOutput<B: burn::tensor::backend::Backend> {
    /// Scalar loss tensor (shape `[1]`, ready for `backward()`).
    pub loss: Tensor<B, 1>,
    /// For each batch row, sum of squared (pred−target) on normalised, **jittered** targets
    /// (same definition as the previous inline training loop for smurf masking).
    pub per_row_mse_for_smurf: Tensor<B, 2>,
    /// Sum of `(pred_norm − target_norm_clean)²` over masked elements (overfit harness metrics;
    /// [`super::training::train`] ignores these when destructuring).
    pub harness_sum_sq_error_norm: f32,
    /// Mask sum (known player slots) for harness RMSE.
    pub harness_known_slots: f32,
    pub harness_pred_sum_norm: f32,
    pub harness_target_sum_norm: f32,
}

/// Full production training loss: Huber + τ-averaged pinball, inverse-frequency row weights,
/// ordinal BCE, pairwise hinge, using [`SequenceModel::forward_with_ordinal_scale`].
pub fn production_training_minibatch_loss<
    B: burn::tensor::backend::AutodiffBackend + ml_model::fused_lstm::FusedLstmBackend,
>(
    model: &SequenceModel<B>,
    batch: &SequenceBatch<B>,
    device: &B::Device,
    rank_weights: &[f32],
    lobby_scale: f32,
    jitter_step: LabelJitterStep,
) -> ProductionMinibatchLossOutput<B>
where
    B::FloatElem: From<f32>,
    B::InnerBackend: ml_model::fused_lstm::FusedLstmBackend,
{
    let huber_delta = 1.0_f32;
    let (predictions, ordinal_logits) =
        model.forward_with_ordinal_scale(batch.inputs.clone(), lobby_scale);

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

    let known_per_row = mask.clone().sum_dim(1).clamp_min(1.0);
    let masked_targets_sum = (batch.targets.clone() * mask.clone()).sum_dim(1);
    let mean_target_mmr_vec: Vec<f32> = (masked_targets_sum / known_per_row)
        .into_data()
        .to_vec()
        .unwrap_or_default();

    let weights_vec = lookup_rank_weights_slice(&mean_target_mmr_vec, rank_weights);
    let weights = Tensor::<B, 1>::from_floats(weights_vec.as_slice(), device)
        .reshape([mean_target_mmr_vec.len(), 1]);

    let jitter_norm = mean_zero_label_jitter_normalized::<B>(
        device,
        mean_target_mmr_vec.len(),
        mask.clone(),
        jitter_step,
    );
    let raw_targets = batch.targets.clone();
    let raw_predictions = predictions.clone();
    let targets_norm = batch.targets.clone() / MMR_SCALE + jitter_norm;
    let predictions_norm = predictions / MMR_SCALE;
    let diff = predictions_norm.clone() - targets_norm.clone();
    let per_row_mse_for_smurf = (diff.clone().powf_scalar(2.0) * mask.clone()).sum_dim(1);
    let abs_diff = diff.clone().abs();
    let clamped = abs_diff.clone().clamp_min(0.0).clamp_max(huber_delta);
    let huber_loss = clamped.clone().powf_scalar(2.0) * 0.5 + (abs_diff - clamped) * huber_delta;

    let high_rank_mask = targets_norm
        .greater_elem(PINBALL_THRESHOLD_MMR / MMR_SCALE)
        .float();
    let pinball_mixed = multi_quantile_pinball(diff, &PINBALL_QUANTILES);
    let pinball_high_rank = pinball_mixed.clone() * high_rank_mask;
    let pinball_full_batch = pinball_mixed * mask.clone();
    let element_loss = huber_loss
        + pinball_high_rank * PINBALL_WEIGHT
        + pinball_full_batch * PINBALL_FULL_BATCH_WEIGHT;
    let regression_loss = (element_loss * mask.clone() * weights).sum() / known_count;

    let flat_row_count = raw_targets.dims()[0] * TOTAL_PLAYERS;
    let flat_mask = mask.clone().reshape([flat_row_count]);
    let flat_targets_mmr = (raw_targets.clone() * MMR_SCALE).reshape([flat_row_count, 1]);

    let boundaries_vec: Vec<f32> = ORDINAL_BOUNDARIES_MMR.to_vec();
    let boundaries = Tensor::<B, 1>::from_floats(boundaries_vec.as_slice(), device)
        .unsqueeze::<2>()
        .transpose();

    let ordinal_targets = flat_targets_mmr
        .expand([flat_row_count, ORDINAL_NUM_BOUNDARIES])
        .greater(
            boundaries
                .transpose()
                .expand([flat_row_count, ORDINAL_NUM_BOUNDARIES]),
        )
        .float();

    let ordinal_logits_stable = ordinal_logits
        .clamp_max(ORDINAL_LOGITS_CLAMP)
        .clamp_min(-ORDINAL_LOGITS_CLAMP);
    let negative_log_likelihood = ordinal_targets.clone()
        * activation::log_sigmoid(ordinal_logits_stable.clone())
        + (ordinal_targets.neg() + 1.0) * activation::log_sigmoid(ordinal_logits_stable.neg());
    let bce = negative_log_likelihood.neg();

    let ordinal_mask = flat_mask
        .reshape([flat_row_count, 1])
        .expand([flat_row_count, ORDINAL_NUM_BOUNDARIES]);
    let ordinal_known_count = ordinal_mask
        .clone()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    let ordinal_known_count = ordinal_known_count.first().copied().unwrap_or(1.0).max(1.0);
    let ordinal_loss =
        (bce * ordinal_mask).sum() / (ordinal_known_count * ORDINAL_NUM_BOUNDARIES as f32);

    let batch_size_local = raw_targets.dims()[0];
    let preds_flat = raw_predictions.reshape([batch_size_local * TOTAL_PLAYERS]);
    let targets_flat = raw_targets.reshape([batch_size_local * TOTAL_PLAYERS]);
    let preds_lobby = preds_flat.reshape([batch_size_local, TOTAL_PLAYERS]);
    let targets_lobby = targets_flat.reshape([batch_size_local, TOTAL_PLAYERS]);
    let mask_lobby = mask.clone().reshape([batch_size_local, TOTAL_PLAYERS]);

    let preds_i = preds_lobby.clone().unsqueeze_dim::<3>(2);
    let preds_j = preds_lobby.unsqueeze_dim::<3>(1);
    let pred_diff = preds_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
        - preds_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS]);

    let targets_i = targets_lobby.clone().unsqueeze_dim::<3>(2);
    let targets_j = targets_lobby.unsqueeze_dim::<3>(1);
    let target_diff = targets_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
        - targets_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS]);

    let mask_i = mask_lobby.clone().unsqueeze_dim::<3>(2);
    let mask_j = mask_lobby.unsqueeze_dim::<3>(1);
    let pair_mask = mask_i.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
        * mask_j.expand([batch_size_local, TOTAL_PLAYERS, TOTAL_PLAYERS])
        * target_diff.greater_elem(MMR_SCALE * 0.01).float();

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

    // Minibatch spread: punish predictions that are too tight vs clean targets (global mean shortcut).
    let targets_norm_clean = batch.targets.clone() / MMR_SCALE;
    let distribution_spread_loss = masked_minibatch_spread_loss_squared(
        predictions_norm.clone(),
        targets_norm_clean.clone(),
        mask.clone(),
    );

    // Clean-target diagnostics for the overfit harness (same normalised `predictions` as the loss).
    let diff_harness = predictions_norm.clone() - targets_norm_clean.clone();
    let harness_sum_sq = (diff_harness.powf_scalar(2.0) * mask.clone()).sum();
    let known_slots = mask.clone().sum();
    let harness_sum_sq_error_norm: f32 = harness_sum_sq
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_known_slots: f32 = known_slots
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_pred_sum_norm: f32 = (predictions_norm * mask.clone())
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_target_sum_norm: f32 = (targets_norm_clean * mask)
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);

    let total = regression_loss
        + ordinal_loss * ORDINAL_LOSS_WEIGHT
        + pairwise_loss * PAIRWISE_LOSS_WEIGHT
        + distribution_spread_loss * DISTRIBUTION_SPREAD_LOSS_WEIGHT;
    let loss = total.unsqueeze::<1>();
    ProductionMinibatchLossOutput {
        loss,
        per_row_mse_for_smurf,
        harness_sum_sq_error_norm,
        harness_known_slots,
        harness_pred_sum_norm,
        harness_target_sum_norm,
    }
}

/// MSE ablation: scalar loss and harness diagnostics (clean-target RMSE stats).
pub struct MseAblationOutput<B: burn::tensor::backend::Backend> {
    pub loss: Tensor<B, 1>,
    pub harness_sum_sq_error_norm: f32,
    pub harness_known_slots: f32,
    pub harness_pred_sum_norm: f32,
    pub harness_target_sum_norm: f32,
}

/// Harness `--mse-only` loss: MSE on mean-zero jitter and optional T3 high-MMR row boost.
///
/// Adds masked minibatch spread ([`MSE_ABLATION_DISTRIBUTION_SPREAD_WEIGHT`]; same statistic as production).
///
/// No Huber, pinball, ordinal, or pairwise. Uses [`SequenceModel::forward_with_lobby_scale`] (same
/// MMR head path as [`SequenceModel::forward`]) so training matches `eval_per_rank_rmse` and
/// leaves the ordinal head out of the autodiff graph.
pub fn mse_ablation_minibatch_loss<
    B: burn::tensor::backend::AutodiffBackend + ml_model::fused_lstm::FusedLstmBackend,
>(
    model: &SequenceModel<B>,
    batch: &SequenceBatch<B>,
    device: &B::Device,
    lobby_scale: f32,
    extreme_mmr: Option<MseExtremeMmrRowBoost>,
    jitter_step: LabelJitterStep,
) -> MseAblationOutput<B>
where
    B::FloatElem: From<f32>,
    B::InnerBackend: ml_model::fused_lstm::FusedLstmBackend,
{
    let predictions = model.forward_with_lobby_scale(batch.inputs.clone(), lobby_scale);
    let mask = batch.targets.clone().greater_elem(0.0).float();
    let actual_batch_size = batch.targets.dims()[0];

    let weights: Tensor<B, 2> = {
        let unit_row: Vec<f32> = vec![1.0f32; actual_batch_size];
        let base = Tensor::<B, 1>::from_floats(unit_row.as_slice(), device)
            .reshape([actual_batch_size, 1]);
        if let Some(boost) = extreme_mmr.as_ref() {
            let known_per_row = mask.clone().sum_dim(1).clamp_min(1.0);
            let mean_target_mmr = (batch.targets.clone() * mask.clone()).sum_dim(1) / known_per_row;
            let high = mean_target_mmr.greater_elem(boost.threshold_mmr).float();
            let row_scale = high * (boost.extra_multiplier - 1.0) + 1.0;
            base * row_scale
        } else {
            base
        }
    };

    let targets_norm_clean = batch.targets.clone() / MMR_SCALE;
    let jitter_norm = mean_zero_label_jitter_normalized::<B>(
        device,
        actual_batch_size,
        mask.clone(),
        jitter_step,
    );
    let targets_norm_train = targets_norm_clean.clone() + jitter_norm;
    let preds_norm = predictions / MMR_SCALE;
    let diff_loss = preds_norm.clone() - targets_norm_train;
    // Plain MSE (no 0.5): standard gradient scale so T3 high-MMR row weighting matches the historical harness.
    let mse_element = diff_loss.powf_scalar(2.0);
    let known_count_tensor = mask.clone().sum();
    let mean_squared_error_loss =
        (mse_element * mask.clone() * weights).sum() / known_count_tensor.clone().clamp_min(1.0);
    let distribution_spread_loss = masked_minibatch_spread_loss_squared(
        preds_norm.clone(),
        targets_norm_clean.clone(),
        mask.clone(),
    );
    let loss = (mean_squared_error_loss
        + distribution_spread_loss * MSE_ABLATION_DISTRIBUTION_SPREAD_WEIGHT)
        .unsqueeze::<1>();

    let diff_harness = preds_norm.clone() - targets_norm_clean.clone();
    let sum_sq = (diff_harness.powf_scalar(2.0) * mask.clone()).sum();
    let harness_sum_sq_error_norm: f32 = sum_sq
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_known_slots: f32 = known_count_tensor
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_pred_sum_norm: f32 = (preds_norm * mask.clone())
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);
    let harness_target_sum_norm: f32 = (targets_norm_clean * mask)
        .sum()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default()
        .first()
        .copied()
        .unwrap_or(0.0);

    MseAblationOutput {
        loss,
        harness_sum_sq_error_norm,
        harness_known_slots,
        harness_pred_sum_norm,
        harness_target_sum_norm,
    }
}
