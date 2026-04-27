//! Shared minibatch loss for production training and the `overfit_wgpu` harness.
//!
//! Production path: Huber, pinball, rank weights, ordinal, pairwise ([`production_training_minibatch_loss`]).
//! `--mse-only` ablation: [`mse_ablation_minibatch_loss`] (not used by [`super::training::train`]).

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

/// Asymmetric quantile (pinball) loss weight (same as [super::training]).
const PINBALL_WEIGHT: f32 = 0.3;
const PINBALL_TAU: f32 = 0.9;
const PINBALL_THRESHOLD_MMR: f32 = 1400.0;

const ORDINAL_LOSS_WEIGHT: f32 = 0.2;
const PAIRWISE_LOSS_WEIGHT: f32 = 0.1;

/// Label jitter in MMR, applied in normalised space with mean zero per row.
/// Must match [super::training] `LABEL_JITTER_STD`.
const LABEL_JITTER_STD: f64 = 40.0;

/// Row-level boost for the harness T3 MSE ablation: rows whose mean MMR (clean) is
/// above `threshold_mmr` multiply per-element loss by `extra_multiplier`.
#[derive(Debug, Clone)]
pub struct MseExtremeMmrRowBoost {
    pub threshold_mmr: f32,
    pub extra_multiplier: f32,
}

fn mean_zero_label_jitter_normalized<B: burn::tensor::backend::Backend>(
    device: &B::Device,
    batch_size: usize,
    mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let jitter = Tensor::<B, 2>::random(
        [batch_size, TOTAL_PLAYERS],
        burn::tensor::Distribution::Normal(0.0, LABEL_JITTER_STD / f64::from(MMR_SCALE)),
        device,
    );
    let row_sum = (jitter.clone() * mask.clone()).sum_dim(1);
    let known = mask.clone().sum_dim(1).clamp_min(1.0);
    let row_mean = row_sum / known;
    (jitter - row_mean) * mask
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

/// Full production training loss: Huber + pinball, inverse-frequency row weights,
/// ordinal BCE, pairwise hinge, using [`SequenceModel::forward_with_ordinal_scale`].
pub fn production_training_minibatch_loss<
    B: burn::tensor::backend::AutodiffBackend + ml_model::fused_lstm::FusedLstmBackend,
>(
    model: &SequenceModel<B>,
    batch: &SequenceBatch<B>,
    device: &B::Device,
    rank_weights: &[f32],
    lobby_scale: f32,
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

    let jitter_norm =
        mean_zero_label_jitter_normalized::<B>(device, mean_target_mmr_vec.len(), mask.clone());
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
    let pinball = (diff.clone() * PINBALL_TAU - diff.clamp_max(0.0)) * high_rank_mask;
    let element_loss = huber_loss + pinball * PINBALL_WEIGHT;
    let regression_loss = (element_loss * mask.clone() * weights).sum() / known_count;

    let flat_mask = mask
        .clone()
        .reshape([raw_targets.dims()[0] * TOTAL_PLAYERS]);
    let flat_targets_mmr =
        (raw_targets.clone() * MMR_SCALE).reshape([raw_targets.dims()[0] * TOTAL_PLAYERS]);

    let boundaries_vec: Vec<f32> = ORDINAL_BOUNDARIES_MMR.to_vec();
    let boundaries = Tensor::<B, 1>::from_floats(boundaries_vec.as_slice(), device)
        .unsqueeze::<2>()
        .transpose();

    let flat_targets_mmr_col = flat_targets_mmr.clone().unsqueeze::<2>();
    let ordinal_targets = flat_targets_mmr_col
        .expand([flat_targets_mmr.dims()[0], ORDINAL_NUM_BOUNDARIES])
        .greater(
            boundaries
                .transpose()
                .expand([flat_targets_mmr.dims()[0], ORDINAL_NUM_BOUNDARIES]),
        )
        .float();

    let sigmoid_logits = activation::sigmoid(ordinal_logits);
    let bce = ordinal_targets.clone() * (sigmoid_logits.clone().clamp_min(1e-6).log())
        + (ordinal_targets.neg() + 1.0) * ((sigmoid_logits.neg() + 1.0).clamp_min(1e-6).log());
    let bce = bce.neg();

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

    // Clean-target diagnostics for the overfit harness (same normalised `predictions` as the loss).
    let targets_norm_clean = batch.targets.clone() / MMR_SCALE;
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

    let total =
        regression_loss + ordinal_loss * ORDINAL_LOSS_WEIGHT + pairwise_loss * PAIRWISE_LOSS_WEIGHT;
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

/// Harness `--mse-only` loss: MSE on mean-zero jitter, optional T3 high-MMR row boost.
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
    let jitter_norm =
        mean_zero_label_jitter_normalized::<B>(device, actual_batch_size, mask.clone());
    let targets_norm_train = targets_norm_clean.clone() + jitter_norm;
    let preds_norm = predictions / MMR_SCALE;
    let diff_loss = preds_norm.clone() - targets_norm_train;
    // Plain MSE (no 0.5): standard gradient scale so T3 high-MMR row weighting matches the historical harness.
    let mse_element = diff_loss.powf_scalar(2.0);
    let known_count_tensor = mask.clone().sum();
    let loss =
        (mse_element * mask.clone() * weights).sum() / known_count_tensor.clone().clamp_min(1.0);
    let loss = loss.unsqueeze::<1>();

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
