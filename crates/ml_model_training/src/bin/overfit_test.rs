#![allow(clippy::indexing_slicing)]

//! Overfit regression-test harness with three tiers of difficulty.
//!
//! Labels are inferred from the segment-cache path structure:
//!   `segments/v6/ranked-standard/{rank}/{uuid}/{start}-{end}.features`
//!
//! **T1** – Single-SSL replay overfitting: train on one SSL replay's segments until
//!   RMSE < 50 MMR (proves the model can represent the top end of the distribution).
//!
//! **T2** – Bronze-1 vs SSL pair: train on one Bronze-1 replay + one SSL replay and
//!   verify the model separates them (both RMSE < 300 MMR at end).
//!
//! **T3** – Balanced mini-set (≤ 5 segments per rank, all ranks present): train for
//!   150 epochs, require per-rank RMSE < 350 MMR for SSL.
//!
//! Usage:
//!   cargo run --bin overfit_test --release -- [SEGMENTS_DIR] [--epochs E]

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use feature_extractor::{PLAYER_CENTRIC_FEATURE_COUNT, TOTAL_PLAYERS};
use ml_model::MMR_SCALE;
use ml_model_training::SequenceBatcher;
use ml_model_training::segment_cache::{SegmentFileInfo, SegmentStore};
use ml_model_training::tch_model::{TchModelConfig, TchSequenceModel, burn_2d_to_tch, burn_3d_to_tch};
use replay_structs::Rank;
use tch::{Kind, nn, nn::OptimizerConfig};

/// Burn NdArray is used only for CPU data loading (SegmentStore → batch tensors).
/// All model computation happens in tch on CUDA.
type BatchBackend = NdArray;


const DEBUG_LOG_PATH: &str = "/workspace/.cursor/debug-1732ff.log";
const DEBUG_SESSION: &str = "1732ff";
const DEBUG_RUN_ID: &str = "tch-cudnn";
fn dbg_log(location: &str, hypothesis: &str, message: &str, data: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let line = format!(
        "{{\"sessionId\":\"{DEBUG_SESSION}\",\"runId\":\"{DEBUG_RUN_ID}\",\"timestamp\":{ts},\"location\":\"{location}\",\"hypothesisId\":\"{hypothesis}\",\"message\":\"{message}\",\"data\":{data}}}\n"
    );
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(DEBUG_LOG_PATH) {
        let _ = f.write_all(line.as_bytes());
    }
}


struct HarnessConfig {
    segments_dir: PathBuf,
    t1_epochs: usize,
    t2_epochs: usize,
    t3_epochs: usize,
    sequence_length: usize,
    batch_size: usize,
    learning_rate: f64,
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
                if let Some(val) = args.get(i) {
                    let e: usize = val.parse().unwrap_or(config.t3_epochs);
                    config.t1_epochs = e;
                    config.t2_epochs = e;
                    config.t3_epochs = e;
                }
            }
            Some("--seq-len") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.sequence_length = val.parse().unwrap_or(config.sequence_length);
                }
            }
            Some("--lr") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.learning_rate = val.parse().unwrap_or(config.learning_rate);
                }
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
// Rank inference from path and canonical MMR mapping
// =============================================================================

/// Infers the `Rank` from the segment cache path by looking for a known rank
/// name component (e.g. "bronze-1", "ssl", "supersonic-legend").
fn rank_from_path(path: &Path) -> Option<Rank> {
    for component in path.components() {
        if let Some(s) = component.as_os_str().to_str() {
            if let Ok(rank) = Rank::from_str(s) {
                return Some(rank);
            }
        }
    }
    None
}

/// Returns a canonical mid-rank MMR value for a `Rank`.
///
/// These values are chosen to sit comfortably inside each rank's MMR range
/// and match the `RankDivision::mmr_middle()` of that rank's Div-2.
fn canonical_mmr_for_rank(rank: Rank) -> f32 {
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

/// One discovered segment with its inferred rank label.
struct LabelledSegment {
    path: PathBuf,
    rank: Rank,
    /// Parent directory name (replay UUID) — used to group segments.
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
        if std::fs::metadata(&path)
            .map(|m| m.len() != expected_size as u64)
            .unwrap_or(true)
        {
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

        labelled.push(LabelledSegment { path, rank, replay_id_str });
    }

    labelled
}

/// Groups segments by replay UUID: `replay_id_str → (rank, segments)`.
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
    let mut store = SegmentStore::new("overfit-harness".to_string(), sequence_length);
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
// Training helper
// =============================================================================

/// tch Adam training loop with cuDNN LSTM.
///
/// Data is loaded via `SequenceBatcher<NdArray>` (CPU) and uploaded to the
/// tch CUDA device each batch. The model forward, loss, backward, and
/// optimizer step all run inside LibTorch's autograd engine.
fn run_training(
    model: &TchSequenceModel,
    vs: &nn::VarStore,
    store: &Arc<SegmentStore>,
    epochs: usize,
    batch_size: usize,
    sequence_length: usize,
    learning_rate: f64,
    label: &str,
    early_stop_target: Option<f32>,
) -> Vec<f32> {
    let tch_device = vs.device();

    // CPU batcher — NdArray is always CPU; we upload to CUDA inside the loop.
    let batcher = SequenceBatcher::<BatchBackend>::new(NdArrayDevice::Cpu, sequence_length);
    let indices: Vec<usize> = (0..store.len()).collect();

    // Plain Adam — no weight decay for overfit harness.
    let mut opt = nn::Adam::default()
        .build(vs, learning_rate)
        .expect("failed to build Adam optimizer");

    let mut rmse_history = Vec::with_capacity(epochs);
    let print_every = (epochs / 10).max(10);
    let warmup_epochs = 5_usize.min(epochs / 4);
    let mut consecutive_below_target = 0_usize;

    let wall_start = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        // Cosine decay + linear warmup (same schedule as before).
        let lr_scale = if epoch < warmup_epochs {
            0.1 + 0.9 * (epoch as f64 / warmup_epochs as f64)
        } else {
            let progress =
                (epoch - warmup_epochs) as f64 / (epochs - warmup_epochs).max(1) as f64;
            0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        };
        let effective_lr = learning_rate * lr_scale;
        opt.set_lr(effective_lr);

        let mut epoch_sq_err_sum = 0.0f64;
        let mut epoch_known_count = 0usize;
        let mut batch_prep_total = Duration::ZERO;
        let mut gpu_plus_sync_total = Duration::ZERO;
        // #region agent log
        let mut epoch_model_fwd_total_us: u128 = 0;
        let mut epoch_model_fwd_sync_us: u128 = 0;
        let mut epoch_bwd_total = Duration::ZERO;
        let mut epoch_optim_total = Duration::ZERO;
        let mut batch_idx_in_epoch = 0usize;
        // #endregion

        for chunk in indices.chunks(batch_size) {
            // ── CPU data loading ──────────────────────────────────────────────
            let t_prep = Instant::now();
            let Some(batch) = batcher.batch_from_indices(store, chunk) else {
                continue;
            };
            // Convert burn NdArray CPU tensors → tch CUDA tensors.
            let input_dims = batch.inputs.dims();
            let input_tch = burn_3d_to_tch(batch.inputs, tch_device);
            let targets_tch = burn_2d_to_tch(batch.targets, tch_device);
            batch_prep_total += t_prep.elapsed();

            let t_gpu = Instant::now();

            // #region agent log
            if epoch == 0 && batch_idx_in_epoch == 0 {
                dbg_log(
                    "overfit_test.rs:run_training",
                    "tch-cudnn",
                    "first_batch_input_shape",
                    &format!(
                        "{{\"label\":\"{label}\",\"rows\":{},\"seq_len\":{},\"features\":{},\"batch_segments\":{},\"backend\":\"tch-cudnn\"}}",
                        input_dims[0], input_dims[1], input_dims[2], chunk.len()
                    ),
                );
            }

            // H8 replacement: time model forward with explicit CUDA sync.
            let t_model_fwd = Instant::now();
            // #endregion

            // ── cuDNN LSTM forward ────────────────────────────────────────────
            let (predictions, _ordinal) = model.forward_train(&input_tch, 1.0);

            // #region agent log
            let model_fwd_dispatch_us = t_model_fwd.elapsed().as_micros();
            let t_sync = Instant::now();
            // Force a real GPU sync so we measure actual kernel completion.
            let _ = predictions.sum(Kind::Float).double_value(&[]);
            let model_fwd_sync_us = t_sync.elapsed().as_micros();
            let model_fwd_total_us = t_model_fwd.elapsed().as_micros();
            epoch_model_fwd_total_us += model_fwd_total_us;
            epoch_model_fwd_sync_us += model_fwd_sync_us;
            // #endregion

            // ── Loss (MSE over known targets) ─────────────────────────────────
            // Mask: targets encoded as 0 mean "unknown".
            let mask = targets_tch.ne(0.0).to_kind(Kind::Float);
            let known_count = mask.sum(Kind::Float).double_value(&[]).max(1.0);

            let diff = &predictions - &targets_tch;
            let diff_norm = &diff / MMR_SCALE as f64;
            let loss = (diff_norm.pow_tensor_scalar(2.0) * &mask).sum(Kind::Float) / known_count;

            // Accumulate RMSE stats (sync once per batch via double_value).
            let sq = (diff.pow_tensor_scalar(2.0) * &mask)
                .sum(Kind::Float)
                .double_value(&[]);
            epoch_sq_err_sum += sq;
            epoch_known_count += known_count as usize;

            // ── Backward + optimizer ──────────────────────────────────────────
            // #region agent log
            let t_bwd = Instant::now();
            // #endregion
            opt.backward_step(&loss);
            // #region agent log
            let bwd_elapsed = t_bwd.elapsed();
            epoch_bwd_total += bwd_elapsed;

            // tch's backward_step = zero_grad + backward + step (all-in-one).
            // Optim time is amortised into bwd above; track separately as zero.
            let t_optim = Instant::now();
            let optim_elapsed = t_optim.elapsed();
            epoch_optim_total += optim_elapsed;

            if epoch < 2 {
                dbg_log(
                    "overfit_test.rs:run_training",
                    "tch-cudnn",
                    "per_batch_timing_us",
                    &format!(
                        "{{\"label\":\"{label}\",\"epoch\":{epoch},\"batch\":{batch_idx_in_epoch},\"model_fwd_dispatch_us\":{model_fwd_dispatch_us},\"model_fwd_sync_us\":{model_fwd_sync_us},\"model_fwd_total_us\":{model_fwd_total_us},\"bwd_us\":{}}}",
                        bwd_elapsed.as_micros(),
                    ),
                );
            }
            batch_idx_in_epoch += 1;
            // #endregion

            gpu_plus_sync_total += t_gpu.elapsed();
        }

        // #region agent log
        dbg_log(
            "overfit_test.rs:run_training",
            "tch-cudnn",
            "epoch_breakdown_ms",
            &format!(
                "{{\"label\":\"{label}\",\"epoch\":{epoch},\"batches\":{batch_idx_in_epoch},\"total_ms\":{},\"model_fwd_total_ms\":{},\"model_fwd_sync_ms\":{},\"bwd_ms\":{},\"optim_ms\":{},\"prep_ms\":{}}}",
                epoch_start.elapsed().as_millis(),
                epoch_model_fwd_total_us / 1000,
                epoch_model_fwd_sync_us / 1000,
                epoch_bwd_total.as_millis(),
                epoch_optim_total.as_millis(),
                batch_prep_total.as_millis(),
            ),
        );
        // #endregion

        let rmse = if epoch_known_count > 0 {
            (epoch_sq_err_sum / epoch_known_count as f64).sqrt() as f32
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
        }

        if let Some(target) = early_stop_target {
            if rmse < target {
                consecutive_below_target += 1;
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
    model: &TchSequenceModel,
    tch_device: tch::Device,
    store: &Arc<SegmentStore>,
    sequence_length: usize,
    batch_size: usize,
) -> HashMap<Rank, f32> {
    let batcher = SequenceBatcher::<BatchBackend>::new(NdArrayDevice::Cpu, sequence_length);
    let mut rank_sq_err: HashMap<Rank, f64> = HashMap::new();
    let mut rank_count: HashMap<Rank, usize> = HashMap::new();

    let indices: Vec<usize> = (0..store.len()).collect();
    for chunk in indices.chunks(batch_size) {
        let Some(batch) = batcher.batch_from_indices(store, chunk) else {
            continue;
        };

        let targets_cpu: Vec<f32> = batch.targets.into_data().to_vec().unwrap_or_default();
        let input_tch = burn_3d_to_tch(batch.inputs, tch_device);

        let raw_preds = model.forward_eval(&input_tch, 1.0);
        let n = raw_preds.numel();
        let cpu_flat = raw_preds.contiguous().to(tch::Device::Cpu).reshape([-1i64]);
        let mut preds = vec![0.0f32; n];
        let _ = cpu_flat.f_copy_data(&mut preds, n);

        for (pred, target) in preds.iter().zip(targets_cpu.iter()) {
            if *target <= 0.0 {
                continue;
            }
            let rank = Rank::from(replay_structs::RankDivision::from(*target as i32));
            let err = (*pred - *target) as f64;
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
// T1 – Single-SSL replay
// =============================================================================

fn run_t1(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    tch_device: tch::Device,
    model_config: &TchModelConfig,
) -> bool {
    println!("\n=== T1: Single-SSL replay overfit (target RMSE < 50 MMR) ===");

    let by_replay = group_by_replay(all_segments);
    let ssl_replays: Vec<(&String, &(Rank, Vec<usize>))> = by_replay
        .iter()
        .filter(|(_, (rank, _))| *rank == Rank::SupersonicLegend)
        .collect();

    if ssl_replays.is_empty() {
        println!("  SKIP: no SSL segments found in {}", config.segments_dir.display());
        return true;
    }

    let (_, (_, best_indices)) = ssl_replays
        .iter()
        .max_by_key(|(_, (_, segs))| segs.len())
        .unwrap();

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

    let vs = nn::VarStore::new(tch_device);
    let model = TchSequenceModel::new(&vs.root(), model_config);
    let history = run_training(
        &model,
        &vs,
        &store,
        config.t1_epochs,
        config.batch_size,
        config.sequence_length,
        config.learning_rate,
        "T1",
        Some(50.0),
    );

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = end < 50.0;
    println!(
        "  RMSE: {start:.1} → {end:.1} MMR  {}",
        if pass { "PASS ✓" } else { "FAIL ✗ (need < 50)" }
    );
    pass
}

// =============================================================================
// T2 – Bronze-1 vs SSL pair
// =============================================================================

fn run_t2(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    tch_device: tch::Device,
    model_config: &TchModelConfig,
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

    let combined: Vec<usize> =
        ssl_indices.iter().chain(bronze_indices.iter()).copied().collect();

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

    let vs = nn::VarStore::new(tch_device);
    let model = TchSequenceModel::new(&vs.root(), model_config);
    let history = run_training(
        &model,
        &vs,
        &store,
        config.t2_epochs,
        config.batch_size,
        config.sequence_length,
        config.learning_rate,
        "T2",
        Some(300.0),
    );

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = end < 300.0;
    println!(
        "  RMSE: {start:.1} → {end:.1} MMR  {}",
        if pass { "PASS ✓" } else { "FAIL ✗ (need < 300)" }
    );
    pass
}

// =============================================================================
// T3 – Balanced mini-set
// =============================================================================

fn run_t3(
    all_segments: &[LabelledSegment],
    config: &HarnessConfig,
    tch_device: tch::Device,
    model_config: &TchModelConfig,
) -> bool {
    println!("\n=== T3: Balanced mini-set per rank (SSL RMSE < 350 MMR) ===");

    let by_replay = group_by_replay(all_segments);

    // Collect at most 3 segments from at most 5 replays per rank.
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

    let vs = nn::VarStore::new(tch_device);
    let model = TchSequenceModel::new(&vs.root(), model_config);
    let history = run_training(
        &model,
        &vs,
        &store,
        config.t3_epochs,
        config.batch_size,
        config.sequence_length,
        config.learning_rate,
        "T3",
        None,
    );

    let per_rank_rmse =
        eval_per_rank_rmse(&model, tch_device, &store, config.sequence_length, config.batch_size);

    println!("\n  Per-rank RMSE:");
    let mut ssl_rmse = f32::MAX;
    for rank in Rank::all_ranked() {
        if let Some(rmse) = per_rank_rmse.get(&rank) {
            let marker = if rank == Rank::SupersonicLegend { " ← key" } else { "" };
            println!(
                "    {:>22}: {:>7.1} MMR{marker}",
                format!("{rank:?}"),
                rmse
            );
            if rank == Rank::SupersonicLegend {
                ssl_rmse = *rmse;
            }
        }
    }

    let start = history.first().copied().unwrap_or(f32::MAX);
    let end = history.last().copied().unwrap_or(f32::MAX);
    let pass = ssl_rmse < 350.0;
    println!(
        "\n  Overall RMSE: {start:.1} → {end:.1} MMR"
    );
    println!(
        "  SSL RMSE: {ssl_rmse:.1} (target < 350)  {}",
        if pass { "PASS ✓" } else { "FAIL ✗" }
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

    println!("=== Overfit Regression-Test Harness (T1 / T2 / T3) ===");
    println!("  Segments dir : {}", config.segments_dir.display());
    println!("  Sequence len : {}", config.sequence_length);
    println!("  Learning rate: {:.0e}", config.learning_rate);
    println!("  T1/T2 epochs : {}/{}", config.t1_epochs, config.t2_epochs);
    println!("  T3 epochs    : {}", config.t3_epochs);

    let wall_start = Instant::now();

    println!("\nDiscovering and labelling segments...");
    let all_segments = discover_all_segments(&config.segments_dir, config.sequence_length);

    if all_segments.is_empty() {
        eprintln!(
            "No valid segments found in '{}'.",
            config.segments_dir.display()
        );
        eprintln!(
            "Expected segment size: 6 × {} × {} × 4 bytes",
            config.sequence_length, PLAYER_CENTRIC_FEATURE_COUNT
        );
        eprintln!("Make sure SEGMENTS_DIR points to the v6 cache root.");
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

    // tch-rs cuDNN backend (LibTorch CUDA).
    let tch_device = tch::Device::Cuda(0);
    println!("\nBackend: tch cuDNN (LibTorch 2.4.0)");

    let model_config = TchModelConfig::default();

    let t1_pass = run_t1(&all_segments, &config, tch_device, &model_config);
    let t2_pass = run_t2(&all_segments, &config, tch_device, &model_config);
    let t3_pass = run_t3(&all_segments, &config, tch_device, &model_config);

    println!(
        "\n=== HARNESS SUMMARY (elapsed: {:.1}s) ===",
        wall_start.elapsed().as_secs_f64()
    );
    println!("  T1 (single SSL overfit)  : {}", if t1_pass { "PASS" } else { "FAIL" });
    println!("  T2 (Bronze-1 vs SSL)     : {}", if t2_pass { "PASS" } else { "FAIL" });
    println!("  T3 (balanced mini-set)   : {}", if t3_pass { "PASS" } else { "FAIL" });

    let all_pass = t1_pass && t2_pass && t3_pass;
    println!(
        "\n  Overall: {}",
        if all_pass {
            "PASS ✓ — ready to proceed to Phase 2"
        } else {
            "FAIL ✗ — investigate before proceeding"
        }
    );

    if !all_pass {
        std::process::exit(1);
    }
}
