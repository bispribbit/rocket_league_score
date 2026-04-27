#![allow(clippy::indexing_slicing)]

//! Overfit regression-test harness that exercises the `FusedLstm` CubeCL kernel
//! through the `burn-wgpu` backend (Vulkan via Mesa's dzn driver inside the
//! WSL2 devcontainer, or native DX12/Metal/Vulkan on the host).
//!
//! Runs the full Burn stack (model + Adam + Autodiff) on GPU through the
//! CubeCL fused-LSTM kernel across three tiers (T1/T2/T3) to validate that the
//! fused kernel path is numerically sound and to measure its per-epoch speed.
//!
//! Usage:
//!   cargo run --bin overfit_wgpu --release \
//!       -- [SEGMENTS_DIR] [--epochs E] [--t1-epochs E1] [--t2-epochs E2] \
//!          [--t3-epochs E3] [--seq-len L] [--lr LR] [--lr-floor F] [--mse-only]
//!
//! Per-tier epoch flags override `T1`/`T2`/`T3` after `--epochs` if they appear
//! **later** on the command line (avoid `--epochs` after `--t2-epochs` or it
//! resets all three).
//!
//! `--mse-only`: ablation (experiment 4 in `experiment.md`) — MSE on mean-zero
//! jittered targets, **unit** rank weights, **no** pinball. Oversampling, lobby
//! alternation, and learning-rate schedule stay on.
//!
//! In the default (non `--mse-only`) mode, the harness uses
//! [`ml_model_training::minibatch_loss::production_training_minibatch_loss`], the same
//! Huber+pinball+rank-weight+ordinal+pairwise path as [`ml_model_training::train`].
//! With `--mse-only`, the harness uses [`mse_ablation_minibatch_loss`] (MSE on jitter,
//! optional T3 high-MMR row boost; [`SequenceModel::forward_with_lobby_scale`] to match
//! evaluation).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use feature_extractor::{PLAYER_CENTRIC_FEATURE_COUNT, TOTAL_PLAYERS};
use ml_model::{MMR_SCALE, ModelConfig, SequenceModel, create_model};
use ml_model_training::minibatch_loss::{
    mse_ablation_minibatch_loss, production_training_minibatch_loss,
};
use ml_model_training::segment_cache::{SegmentFileInfo, SegmentStore};
use ml_model_training::{
    MseExtremeMmrRowBoost, SequenceBatcher, compute_inverse_frequency_weights, pseudo_random_f32,
};
use replay_structs::Rank;

type TrainBackend = Autodiff<Wgpu>;
type TrainDevice = WgpuDevice;

// =============================================================================
// Args + config
// =============================================================================

struct HarnessConfig {
    segments_dir: PathBuf,
    t1_epochs: usize,
    t2_epochs: usize,
    t3_epochs: usize,
    sequence_length: usize,
    batch_size: usize,
    learning_rate: f64,
    /// Minimum multiplier on `learning_rate` after warmup (cosine tail). Default `0.05`.
    cosine_lr_floor: f64,
    /// When true: train with MSE only (no pinball, no inverse-frequency rank weights).
    mse_only: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            segments_dir: PathBuf::from("ballchasing/segments/v6"),
            t1_epochs: 80,
            t2_epochs: 80,
            t3_epochs: 150,
            sequence_length: 300,
            batch_size: 32,
            learning_rate: 1e-2,
            cosine_lr_floor: 0.05,
            mse_only: false,
        }
    }
}

fn parse_args() -> HarnessConfig {
    let mut config = HarnessConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--epochs") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(e) = val.parse::<usize>()
                {
                    config.t1_epochs = e;
                    config.t2_epochs = e;
                    config.t3_epochs = e;
                }
            }
            Some("--t1-epochs") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(e) = val.parse::<usize>()
                {
                    config.t1_epochs = e;
                }
            }
            Some("--t2-epochs") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(e) = val.parse::<usize>()
                {
                    config.t2_epochs = e;
                }
            }
            Some("--t3-epochs") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(e) = val.parse::<usize>()
                {
                    config.t3_epochs = e;
                }
            }
            Some("--seq-len") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(v) = val.parse::<usize>()
                {
                    config.sequence_length = v;
                }
            }
            Some("--lr") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(v) = val.parse::<f64>()
                {
                    config.learning_rate = v;
                }
            }
            Some("--lr-floor") => {
                i += 1;
                if let Some(val) = args.get(i)
                    && let Ok(v) = val.parse::<f64>()
                {
                    config.cosine_lr_floor = v.clamp(0.0, 1.0);
                }
            }
            Some("--mse-only") => {
                config.mse_only = true;
            }
            Some(path) if !path.starts_with("--") => {
                config.segments_dir = PathBuf::from(path);
            }
            _ => {}
        }
        i += 1;
    }
    config
}

// =============================================================================
// Rank inference + canonical MMR (same as overfit_test)
// =============================================================================

fn rank_from_path(path: &Path) -> Option<Rank> {
    for component in path.components() {
        if let Some(s) = component.as_os_str().to_str()
            && let Ok(rank) = Rank::from_str(s)
        {
            return Some(rank);
        }
    }
    None
}

const fn canonical_mmr_for_rank(rank: Rank) -> f32 {
    match rank {
        Rank::Unranked => 100.0,
        Rank::Bronze1 => 130.0,
        Rank::Bronze2 => 194.0,
        Rank::Bronze3 => 257.0,
        Rank::Silver1 => 321.0,
        Rank::Silver2 => 386.0,
        Rank::Silver3 => 451.0,
        Rank::Gold1 => 516.0,
        Rank::Gold2 => 580.0,
        Rank::Gold3 => 644.0,
        Rank::Platinum1 => 709.0,
        Rank::Platinum2 => 773.0,
        Rank::Platinum3 => 837.0,
        Rank::Diamond1 => 902.0,
        Rank::Diamond2 => 966.0,
        Rank::Diamond3 => 1030.0,
        Rank::Champion1 => 1127.0,
        Rank::Champion2 => 1258.0,
        Rank::Champion3 => 1388.0,
        Rank::GrandChampion1 => 1520.0,
        Rank::GrandChampion2 => 1651.0,
        Rank::GrandChampion3 => 1782.0,
        Rank::SupersonicLegend => 2200.0,
    }
}

// =============================================================================
// Segment discovery
// =============================================================================

struct LabelledSegment {
    path: PathBuf,
    rank: Rank,
    replay_id_str: String,
}

fn collect_feature_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_feature_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "features") {
            out.push(path);
        }
    }
}

fn discover_all_segments(dir: &Path, sequence_length: usize) -> Vec<LabelledSegment> {
    let expected_size =
        6 * sequence_length * PLAYER_CENTRIC_FEATURE_COUNT * std::mem::size_of::<f32>();

    let mut files = Vec::new();
    collect_feature_files(dir, &mut files);

    let mut labelled = Vec::new();
    for path in files {
        if std::fs::metadata(&path).map_or(true, |m| m.len() != expected_size as u64) {
            continue;
        }
        let Some(rank) = rank_from_path(&path) else {
            continue;
        };
        let replay_id_str = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        labelled.push(LabelledSegment {
            path,
            rank,
            replay_id_str,
        });
    }

    labelled
}

fn group_by_replay(segments: &[LabelledSegment]) -> HashMap<String, (Rank, Vec<usize>)> {
    let mut map: HashMap<String, (Rank, Vec<usize>)> = HashMap::new();
    for (idx, seg) in segments.iter().enumerate() {
        map.entry(seg.replay_id_str.clone())
            .or_insert_with(|| (seg.rank, Vec::new()))
            .1
            .push(idx);
    }
    map
}

fn build_store_from_indices(
    all_segments: &[LabelledSegment],
    indices: &[usize],
    sequence_length: usize,
) -> SegmentStore {
    let mut store = SegmentStore::new("overfit-wgpu".to_string(), sequence_length);
    for &idx in indices {
        let Some(seg) = all_segments.get(idx) else {
            continue;
        };
        let mmr = canonical_mmr_for_rank(seg.rank);
        let target_mmr = [mmr; TOTAL_PLAYERS];
        let info = SegmentFileInfo {
            path: seg.path.clone(),
            start_frame: 0,
            end_frame: sequence_length,
            replay_id: uuid::Uuid::nil(),
        };
        store.add_segments(std::slice::from_ref(&info), target_mmr);
    }
    store
}

// =============================================================================
// Training loop — same minibatch loss as `train()` (or MSE ablation for `--mse-only`)
// =============================================================================

fn run_training(
    model: &mut SequenceModel<TrainBackend>,
    store: &Arc<SegmentStore>,
    epochs: usize,
    batch_size: usize,
    sequence_length: usize,
    learning_rate: f64,
    cosine_lr_floor: f64,
    label: &str,
    early_stop_target: Option<f32>,
    mse_only: bool,
    mse_extreme_mmr_boost: Option<MseExtremeMmrRowBoost>,
) -> Vec<f32> {
    let device = model.device();
    // Match `train`: gradient clip 1.0; keep 5.0 for the historical `--mse-only` ablation.
    let grad_clip = if mse_only { 5.0 } else { 1.0 };
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(grad_clip)))
        .init();
    let batcher = SequenceBatcher::<TrainBackend>::new(device.clone(), sequence_length);

    let rank_weights: Vec<f32> = if mse_only {
        Vec::new()
    } else {
        compute_inverse_frequency_weights(store)
    };

    let mut rmse_history = Vec::with_capacity(epochs);
    let print_every = (epochs / 10).max(10);
    let warmup_epochs = (epochs / 4).min(20);
    let mut consecutive_below_target = 0_usize;

    let wall_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        let lr_scale = if epoch < warmup_epochs {
            (epoch as f64 / warmup_epochs.max(1) as f64).mul_add(0.9, 0.1)
        } else {
            let progress = (epoch - warmup_epochs) as f64 / (epochs - warmup_epochs).max(1) as f64;
            (0.5 * (1.0 + (std::f64::consts::PI * progress).cos())).max(cosine_lr_floor)
        };
        let effective_lr = learning_rate * lr_scale;

        let indices = store.build_oversampled_indices(epoch as u64);

        let mut epoch_sq_err_sum = 0.0f64;
        let mut epoch_known_count: f64 = 0.0;
        let mut batch_prep_total = Duration::ZERO;
        let mut gpu_plus_sync_total = Duration::ZERO;

        let mut diag_pred_sum_mmr = 0.0f64;
        let mut diag_target_sum_mmr = 0.0f64;

        let mut batch_count = 0_usize;

        for chunk in indices.chunks(batch_size) {
            let t_prep = Instant::now();
            let Some(batch) = batcher.batch_from_indices(store, chunk) else {
                continue;
            };
            batch_prep_total += t_prep.elapsed();

            let t_gpu = Instant::now();

            let lobby_scale = if mse_only {
                if batch_count.is_multiple_of(2) {
                    1.0_f32
                } else {
                    0.0_f32
                }
            } else if pseudo_random_f32(epoch as u64, batch_count as u64) < 0.2 {
                0.0_f32
            } else {
                1.0_f32
            };

            let loss: Tensor<TrainBackend, 1> = if mse_only {
                let out = mse_ablation_minibatch_loss(
                    model,
                    &batch,
                    &device,
                    lobby_scale,
                    mse_extreme_mmr_boost.clone(),
                );
                epoch_sq_err_sum += f64::from(out.harness_sum_sq_error_norm)
                    * f64::from(MMR_SCALE)
                    * f64::from(MMR_SCALE);
                epoch_known_count += f64::from(out.harness_known_slots);
                diag_pred_sum_mmr += f64::from(out.harness_pred_sum_norm) * f64::from(MMR_SCALE);
                diag_target_sum_mmr +=
                    f64::from(out.harness_target_sum_norm) * f64::from(MMR_SCALE);
                out.loss
            } else {
                let out = production_training_minibatch_loss(
                    model,
                    &batch,
                    &device,
                    &rank_weights,
                    lobby_scale,
                );
                epoch_sq_err_sum += f64::from(out.harness_sum_sq_error_norm)
                    * f64::from(MMR_SCALE)
                    * f64::from(MMR_SCALE);
                epoch_known_count += f64::from(out.harness_known_slots);
                diag_pred_sum_mmr += f64::from(out.harness_pred_sum_norm) * f64::from(MMR_SCALE);
                diag_target_sum_mmr +=
                    f64::from(out.harness_target_sum_norm) * f64::from(MMR_SCALE);
                out.loss
            };

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            let model_snapshot = model.clone();
            *model = optimizer.step(effective_lr, model_snapshot, grads);

            gpu_plus_sync_total += t_gpu.elapsed();
            batch_count = batch_count.saturating_add(1);
        }

        let known_usize = epoch_known_count as usize;
        let rmse = if known_usize > 0 {
            (epoch_sq_err_sum / epoch_known_count).sqrt() as f32
        } else {
            f32::MAX
        };
        rmse_history.push(rmse);

        let epoch_secs = epoch_start.elapsed().as_secs_f64();

        if (epoch + 1) % print_every == 0 || epoch == 0 || epoch + 1 == epochs {
            println!(
                "  [{label}] epoch {:>4}/{} — RMSE {:7.1} MMR  lr={:.2e}  ({:.2}s/epoch: prep={:.2}s gpu={:.2}s)",
                epoch + 1,
                epochs,
                rmse,
                effective_lr,
                epoch_secs,
                batch_prep_total.as_secs_f64(),
                gpu_plus_sync_total.as_secs_f64(),
            );
            if known_usize > 0 {
                let pred_mean = (diag_pred_sum_mmr / epoch_known_count) as f32;
                let target_mean = (diag_target_sum_mmr / epoch_known_count) as f32;
                println!(
                    "  [{label}] collapse-diag: pred_mean={pred_mean:.1}  target_mean={target_mean:.1} MMR"
                );
            }
        }

        if let Some(target) = early_stop_target {
            if rmse < target {
                consecutive_below_target = consecutive_below_target.saturating_add(1);
                if consecutive_below_target >= 2 {
                    println!(
                        "  [{label}] early stop at epoch {}/{} — RMSE {:.1} < target {:.1} MMR",
                        epoch + 1,
                        epochs,
                        rmse,
                        target
                    );
                    break;
                }
            } else {
                consecutive_below_target = 0;
            }
        }
    }

    let total_secs = wall_start.elapsed().as_secs_f64();
    let completed = rmse_history.len();
    println!(
        "  [{label}] total wall-clock: {:.1}s for {} epoch(s) ({:.2}s/epoch avg)",
        total_secs,
        completed,
        total_secs / completed.max(1) as f64,
    );

    rmse_history
}

// =============================================================================
// Per-rank RMSE evaluation
// =============================================================================

fn eval_per_rank_rmse(
    model: &SequenceModel<TrainBackend>,
    store: &Arc<SegmentStore>,
    sequence_length: usize,
    batch_size: usize,
) -> HashMap<Rank, f32> {
    let batcher = SequenceBatcher::<TrainBackend>::new(model.device(), sequence_length);
    let mut rank_sq_err: HashMap<Rank, f64> = HashMap::new();
    let mut rank_count: HashMap<Rank, usize> = HashMap::new();

    let indices: Vec<usize> = (0..store.len()).collect();
    for chunk in indices.chunks(batch_size) {
        let Some(batch) = batcher.batch_from_indices(store, chunk) else {
            continue;
        };

        let preds: Vec<f32> = model
            .forward(batch.inputs)
            .into_data()
            .to_vec()
            .unwrap_or_default();
        let targets: Vec<f32> = batch.targets.into_data().to_vec().unwrap_or_default();

        for (pred, target) in preds.iter().zip(targets.iter()) {
            if *target <= 0.0 {
                continue;
            }
            let rank = Rank::from(replay_structs::RankDivision::from(*target as i32));
            let err = f64::from(*pred - *target);
            *rank_sq_err.entry(rank).or_default() += err * err;
            *rank_count.entry(rank).or_default() += 1;
        }
    }

    rank_sq_err
        .iter()
        .map(|(rank, sq)| {
            let n = rank_count.get(rank).copied().unwrap_or(1).max(1);
            (*rank, (sq / n as f64).sqrt() as f32)
        })
        .collect()
}

// =============================================================================
// T1 / T2 / T3 runners — copied verbatim from overfit_test but typed on
// `Autodiff<Wgpu>` so `SequenceModel` uses the FusedLstm CubeCL kernel.
// =============================================================================

fn run_t1(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    device: &TrainDevice,
    model_config: &ModelConfig,
) -> bool {
    println!("\n=== T1: Single-SSL replay overfit (target RMSE < 50 MMR) ===");

    let by_replay = group_by_replay(all_segments);
    let ssl_replays: Vec<(&String, &(Rank, Vec<usize>))> = by_replay
        .iter()
        .filter(|(_, (rank, _))| *rank == Rank::SupersonicLegend)
        .collect();

    if ssl_replays.is_empty() {
        println!(
            "  SKIP: no SSL segments found in {}",
            config.segments_dir.display()
        );
        return true;
    }

    let (_, (_, best_indices)) = ssl_replays
        .iter()
        .max_by_key(|(_, (_, segs))| segs.len())
        .expect("ssl_replays is non-empty: checked immediately above");

    let mut store = build_store_from_indices(all_segments, best_indices, config.sequence_length);
    if let Err(e) = store.preload_all_segments() {
        eprintln!("  ERROR: preload failed: {e}");
        return false;
    }
    let store = Arc::new(store);
    println!(
        "  {} segments from 1 SSL replay  ({} epochs, lr={:.0e})",
        store.len(),
        config.t1_epochs,
        config.learning_rate
    );

    let mut no_dropout_cfg = model_config.clone();
    no_dropout_cfg.dropout = 0.0;
    let mut model: SequenceModel<TrainBackend> = create_model(device, &no_dropout_cfg);
    let history = run_training(
        &mut model,
        &store,
        config.t1_epochs,
        config.batch_size,
        config.sequence_length,
        config.learning_rate,
        config.cosine_lr_floor,
        "T1",
        Some(50.0),
        config.mse_only,
        None,
    );

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = end < 50.0;
    println!(
        "  RMSE: {start:.1} → {end:.1} MMR  {}",
        if pass { "PASS" } else { "FAIL (need < 50)" }
    );
    pass
}

fn run_t2(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    device: &TrainDevice,
    model_config: &ModelConfig,
) -> bool {
    println!("\n=== T2: Bronze-1 vs SSL pair (RMSE < 300 MMR) ===");

    let by_replay = group_by_replay(all_segments);

    let pick_replay = |target_rank: Rank| -> Option<Vec<usize>> {
        by_replay
            .iter()
            .filter(|(_, (rank, _))| *rank == target_rank)
            .max_by_key(|(_, (_, segs))| segs.len())
            .map(|(_, (_, segs))| segs.clone())
    };

    let Some(ssl_indices) = pick_replay(Rank::SupersonicLegend) else {
        println!("  SKIP: no SSL segments.");
        return true;
    };
    let Some(bronze_indices) = pick_replay(Rank::Bronze1) else {
        println!("  SKIP: no Bronze-1 segments.");
        return true;
    };

    let combined: Vec<usize> = ssl_indices
        .iter()
        .chain(bronze_indices.iter())
        .copied()
        .collect();

    let mut store = build_store_from_indices(all_segments, &combined, config.sequence_length);
    if let Err(e) = store.preload_all_segments() {
        eprintln!("  ERROR: preload failed: {e}");
        return false;
    }
    let store = Arc::new(store);
    println!(
        "  {} segs ({} SSL + {} Bronze-1)  ({} epochs)",
        store.len(),
        ssl_indices.len(),
        bronze_indices.len(),
        config.t2_epochs
    );

    let mut no_dropout_cfg = model_config.clone();
    no_dropout_cfg.dropout = 0.0;
    let mut model: SequenceModel<TrainBackend> = create_model(device, &no_dropout_cfg);
    let history = run_training(
        &mut model,
        &store,
        config.t2_epochs,
        config.batch_size,
        config.sequence_length,
        config.learning_rate,
        config.cosine_lr_floor,
        "T2",
        Some(300.0),
        config.mse_only,
        None,
    );

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = end < 300.0;
    println!(
        "  RMSE: {start:.1} → {end:.1} MMR  {}",
        if pass { "PASS" } else { "FAIL (need < 300)" }
    );
    pass
}

fn run_t3(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    device: &TrainDevice,
    model_config: &ModelConfig,
) -> bool {
    println!("\n=== T3: Balanced mini-set per rank (SSL RMSE < 350 MMR) ===");

    let by_replay = group_by_replay(all_segments);

    let mut per_rank: HashMap<Rank, Vec<usize>> = HashMap::new();
    for (rank, segs) in by_replay.values() {
        let entry = per_rank.entry(*rank).or_default();
        if entry.len() < 5 * 3 {
            for &idx in segs.iter().take(3) {
                entry.push(idx);
            }
        }
    }

    if per_rank.is_empty() {
        println!("  SKIP: no labelled segments.");
        return true;
    }

    let combined: Vec<usize> = per_rank.values().flatten().copied().collect();
    let mut store = build_store_from_indices(all_segments, &combined, config.sequence_length);
    if let Err(e) = store.preload_all_segments() {
        eprintln!("  ERROR: preload failed: {e}");
        return false;
    }
    let store = Arc::new(store);

    println!(
        "  {} segs across {} ranks  ({} epochs)",
        store.len(),
        per_rank.len(),
        config.t3_epochs
    );

    let mut no_dropout_cfg = model_config.clone();
    no_dropout_cfg.dropout = 0.0;
    let mut model: SequenceModel<TrainBackend> = create_model(device, &no_dropout_cfg);
    let t3_learning_rate = config.learning_rate * 1.5;
    let t3_cosine_lr_floor = config.cosine_lr_floor.max(0.15);
    let history = run_training(
        &mut model,
        &store,
        config.t3_epochs,
        config.batch_size,
        config.sequence_length,
        t3_learning_rate,
        t3_cosine_lr_floor,
        "T3",
        None,
        config.mse_only,
        Some(MseExtremeMmrRowBoost {
            threshold_mmr: 1800.0,
            extra_multiplier: 10.0,
        }),
    );

    let per_rank_rmse =
        eval_per_rank_rmse(&model, &store, config.sequence_length, config.batch_size);

    println!("\n  Per-rank RMSE:");
    let mut ssl_rmse = f32::MAX;
    for rank in Rank::all_ranked() {
        if let Some(rmse) = per_rank_rmse.get(&rank) {
            let marker = if rank == Rank::SupersonicLegend {
                " <- key"
            } else {
                ""
            };
            println!("    {:>22}: {:>7.1} MMR{marker}", format!("{rank:?}"), rmse);
            if rank == Rank::SupersonicLegend {
                ssl_rmse = *rmse;
            }
        }
    }

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = ssl_rmse < 350.0;
    println!("\n  Overall RMSE: {start:.1} → {end:.1} MMR");
    println!(
        "  SSL RMSE: {ssl_rmse:.1} (target < 350)  {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = parse_args();

    println!("=== Overfit Regression-Test Harness — WGPU + FusedLstm ===");
    println!("  Segments dir : {}", config.segments_dir.display());
    println!("  Sequence len : {}", config.sequence_length);
    println!("  Learning rate: {:.0e}", config.learning_rate);
    let default_cosine_lr_floor = HarnessConfig::default().cosine_lr_floor;
    if (config.cosine_lr_floor - default_cosine_lr_floor).abs() > 1e-9 {
        println!(
            "  Cosine lr floor (fraction of base lr): {:.4}",
            config.cosine_lr_floor
        );
    }
    println!("  T1/T2 epochs : {}/{}", config.t1_epochs, config.t2_epochs);
    println!("  T3 epochs    : {}", config.t3_epochs);
    if config.mse_only {
        println!("  Loss mode    : MSE ablation (no pinball, unit rank weights)");
    } else {
        println!("  Loss mode    : MSE + pinball + inverse-frequency rank weights");
    }

    let wall_start = Instant::now();

    println!("\nDiscovering and labelling segments...");
    let all_segments = discover_all_segments(&config.segments_dir, config.sequence_length);

    if all_segments.is_empty() {
        eprintln!(
            "No valid segments found in '{}'.",
            config.segments_dir.display()
        );
        eprintln!(
            "Expected segment size: 6 x {} x {} x 4 bytes",
            config.sequence_length, PLAYER_CENTRIC_FEATURE_COUNT
        );
        std::process::exit(1);
    }

    let rank_summary: Vec<(Rank, usize)> = {
        let mut map: HashMap<Rank, usize> = HashMap::new();
        for seg in &all_segments {
            *map.entry(seg.rank).or_default() += 1;
        }
        let mut v: Vec<(Rank, usize)> = map.into_iter().collect();
        v.sort_by_key(|(r, _)| r.as_numeric_index());
        v
    };
    println!(
        "Found {} segments across {} ranks:",
        all_segments.len(),
        rank_summary.len()
    );
    for (rank, count) in &rank_summary {
        println!("  {:>22}: {count}", format!("{rank:?}"));
    }

    // `WgpuDevice::default()` picks the first adapter wgpu enumerates. Inside
    // the WSL2 devcontainer this is the dzn adapter pointing at the host GPU
    // (provided `WGPU_BACKEND=vulkan` is set; see `.devcontainer/docker-
    // compose.yml`). On Windows natively, this is the DX12 NVIDIA adapter.
    let device = WgpuDevice::default();
    Wgpu::<f32>::seed(&device, 42);
    println!("\nBackend: burn-wgpu (Vulkan / DX12 via wgpu) with FusedLstm CubeCL kernel");

    let model_config = ModelConfig::new();

    let t1_pass = run_t1(&all_segments, &config, &device, &model_config);
    let t2_pass = run_t2(&all_segments, &config, &device, &model_config);
    let t3_pass = run_t3(&all_segments, &config, &device, &model_config);

    println!(
        "\n=== HARNESS SUMMARY (elapsed: {:.1}s) ===",
        wall_start.elapsed().as_secs_f64()
    );
    println!(
        "  T1 (single SSL overfit)  : {}",
        if t1_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  T2 (Bronze-1 vs SSL)     : {}",
        if t2_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  T3 (balanced mini-set)   : {}",
        if t3_pass { "PASS" } else { "FAIL" }
    );

    let all_pass = t1_pass && t2_pass && t3_pass;
    println!("\n  Overall: {}", if all_pass { "PASS" } else { "FAIL" });

    if !all_pass {
        std::process::exit(1);
    }
}
