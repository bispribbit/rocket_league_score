#![allow(clippy::indexing_slicing)]

//! Quick overfit sanity-check with automatic learning rate sweep.
//!
//! Scans the existing segment cache directory and picks N segments, then runs
//! multiple independent training loops with different learning rates and loss
//! functions to find a configuration where the model can learn.
//!
//! Usage:
//!   cargo run --bin overfit_test --release -- [SEGMENTS_DIR] [--segments N] [--epochs E]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use burn::backend::{Autodiff, Wgpu};
use burn::grad_clipping::GradientClippingConfig;
use burn::nn::loss::{HuberLoss, MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{MMR_SCALE, ModelConfig, SequenceModel, create_model};
use ml_model_training::SequenceBatcher;
use ml_model_training::segment_cache::SegmentStore;

type TrainBackend = Autodiff<Wgpu>;

struct OverfitConfig {
    segments_dir: PathBuf,
    max_segments: usize,
    epochs: usize,
    batch_size: usize,
    sequence_length: usize,
}

impl Default for OverfitConfig {
    fn default() -> Self {
        Self {
            segments_dir: PathBuf::from("data/segments/v3"),
            max_segments: 64,
            epochs: 30,
            batch_size: 64,
            sequence_length: 150,
        }
    }
}

fn parse_args() -> OverfitConfig {
    let mut config = OverfitConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--segments") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.max_segments = val.parse().unwrap_or(config.max_segments);
                }
            }
            Some("--epochs") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.epochs = val.parse().unwrap_or(config.epochs);
                }
            }
            Some("--batch-size") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.batch_size = val.parse().unwrap_or(config.batch_size);
                }
            }
            Some("--seq-len") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.sequence_length = val.parse().unwrap_or(config.sequence_length);
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

fn discover_segments(dir: &Path, max: usize, segment_length: usize) -> SegmentStore {
    use feature_extractor::PLAYER_CENTRIC_FEATURE_COUNT;

    let expected_size =
        6 * segment_length * PLAYER_CENTRIC_FEATURE_COUNT * std::mem::size_of::<f32>();
    let mut store = SegmentStore::new("overfit-test".to_string(), segment_length);

    let mut feature_files: Vec<PathBuf> = Vec::new();
    collect_feature_files(dir, &mut feature_files);

    feature_files.sort();

    let step = if feature_files.len() > max {
        feature_files.len() / max
    } else {
        1
    };

    let mut picked: Vec<PathBuf> = feature_files
        .iter()
        .step_by(step)
        .take(max)
        .cloned()
        .collect();

    picked.retain(|p| {
        std::fs::metadata(p)
            .map(|m| m.len() == expected_size as u64)
            .unwrap_or(false)
    });

    let segment_infos: Vec<ml_model_training::segment_cache::SegmentFileInfo> = picked
        .into_iter()
        .map(|path| ml_model_training::segment_cache::SegmentFileInfo {
            path,
            start_frame: 0,
            end_frame: segment_length,
            replay_id: uuid::Uuid::nil(),
        })
        .collect();

    let varied_mmr: Vec<[f32; TOTAL_PLAYERS]> = segment_infos
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let base = ((i % 10) as f32).mul_add(150.0, 400.0);
            [
                base,
                base + 50.0,
                base + 100.0,
                base + 25.0,
                base + 75.0,
                base + 125.0,
            ]
        })
        .collect();

    for (info, mmr) in segment_infos.iter().zip(varied_mmr.iter()) {
        store.add_segments(std::slice::from_ref(info), *mmr);
    }

    store
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

#[derive(Clone)]
struct EpochResult {
    epoch: usize,
    loss: f32,
    approx_rmse: f32,
    elapsed_ms: u128,
}

#[derive(Clone, Copy)]
enum LossType {
    Mse,
    Huber { delta: f32 },
}

impl std::fmt::Display for LossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mse => write!(f, "MSE"),
            Self::Huber { delta } => write!(f, "Huber(d={delta})"),
        }
    }
}

struct RunConfig {
    lr: f64,
    loss_type: LossType,
    grad_clip: bool,
}

impl std::fmt::Display for RunConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} lr={:.0e} clip={}",
            self.loss_type,
            self.lr,
            if self.grad_clip { "yes" } else { "no" }
        )
    }
}

fn train_run(
    model: &mut SequenceModel<TrainBackend>,
    store: &Arc<SegmentStore>,
    base_config: &OverfitConfig,
    run: &RunConfig,
) -> Vec<EpochResult> {
    let device = model.device();

    let optimizer_config = if run.grad_clip {
        AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    } else {
        AdamConfig::new()
    };
    let mut optimizer = optimizer_config.init();

    let batcher = SequenceBatcher::<TrainBackend>::new(device, base_config.sequence_length);
    let indices: Vec<usize> = (0..store.len()).collect();
    let mut results = Vec::with_capacity(base_config.epochs);

    for epoch in 0..base_config.epochs {
        let start = Instant::now();
        let mut epoch_loss = 0.0f64;
        let mut batch_count = 0usize;

        for batch_start in (0..store.len()).step_by(base_config.batch_size) {
            let batch_end = (batch_start + base_config.batch_size).min(store.len());
            let Some(batch_idx) = indices.get(batch_start..batch_end) else {
                continue;
            };
            let Some(batch) = batcher.batch_from_indices(store, batch_idx) else {
                continue;
            };

            let predictions = model.forward(batch.inputs);
            let targets_norm = batch.targets.clone() / MMR_SCALE;
            let preds_norm = predictions / MMR_SCALE;

            let loss = match run.loss_type {
                LossType::Mse => MseLoss::new().forward(preds_norm, targets_norm, Reduction::Mean),
                LossType::Huber { delta } => {
                    let loss_fn = HuberLoss {
                        delta,
                        lin_bias: delta * delta / 2.0,
                    };
                    loss_fn.forward(preds_norm, targets_norm, Reduction::Mean)
                }
            };

            let loss_val: f32 = loss
                .clone()
                .into_data()
                .to_vec()
                .unwrap_or_else(|_| vec![0.0])
                .first()
                .copied()
                .unwrap_or(0.0);

            epoch_loss += loss_val as f64;
            batch_count += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, model);
            *model = optimizer.step(run.lr, model.clone(), grads);
        }

        let avg_loss = if batch_count > 0 {
            (epoch_loss / batch_count as f64) as f32
        } else {
            0.0
        };
        let rmse = avg_loss.sqrt() * MMR_SCALE;

        results.push(EpochResult {
            epoch,
            loss: avg_loss,
            approx_rmse: rmse,
            elapsed_ms: start.elapsed().as_millis(),
        });
    }

    results
}

struct RunSummary {
    label: String,
    start_rmse: f32,
    end_rmse: f32,
    improvement: f32,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = parse_args();

    println!("=== Overfit Sanity Check (LR Sweep) ===");
    println!("  Segments dir  : {}", config.segments_dir.display());
    println!("  Max segments  : {}", config.max_segments);
    println!("  Epochs        : {}", config.epochs);
    println!("  Batch size    : {}", config.batch_size);
    println!("  Sequence len  : {}", config.sequence_length);

    println!("\nDiscovering segments...");
    let mut store = discover_segments(
        &config.segments_dir,
        config.max_segments,
        config.sequence_length,
    );

    if store.is_empty() {
        eprintln!(
            "No valid segments found in {}. Check the path.",
            config.segments_dir.display()
        );
        std::process::exit(1);
    }

    println!("Found {} segments. Preloading into memory...", store.len());
    if let Err(e) = store.preload_all_segments() {
        eprintln!("Failed to preload segments: {e}");
        std::process::exit(1);
    }
    let store = Arc::new(store);

    let device = burn::backend::wgpu::WgpuDevice::default();
    let model_config = ModelConfig::new();

    // Define the experiment grid: sweep LR for MSE, then test Huber variants
    let runs: Vec<RunConfig> = vec![
        // MSE with increasing LR, no gradient clipping
        RunConfig {
            lr: 1e-3,
            loss_type: LossType::Mse,
            grad_clip: false,
        },
        RunConfig {
            lr: 1e-2,
            loss_type: LossType::Mse,
            grad_clip: false,
        },
        RunConfig {
            lr: 5e-2,
            loss_type: LossType::Mse,
            grad_clip: false,
        },
        RunConfig {
            lr: 1e-1,
            loss_type: LossType::Mse,
            grad_clip: false,
        },
        // MSE with grad clip for comparison
        RunConfig {
            lr: 1e-2,
            loss_type: LossType::Mse,
            grad_clip: true,
        },
        // Huber with different deltas at best MSE LR
        RunConfig {
            lr: 1e-2,
            loss_type: LossType::Huber { delta: 0.1 },
            grad_clip: false,
        },
        RunConfig {
            lr: 1e-2,
            loss_type: LossType::Huber { delta: 0.5 },
            grad_clip: false,
        },
        RunConfig {
            lr: 1e-2,
            loss_type: LossType::Huber { delta: 1.0 },
            grad_clip: false,
        },
    ];

    let mut summaries: Vec<RunSummary> = Vec::new();

    for (i, run) in runs.iter().enumerate() {
        let label = format!("[{}/{}] {run}", i + 1, runs.len());
        println!("\n--- {label} ---");

        let mut model: SequenceModel<TrainBackend> = create_model(&device, &model_config);
        let results = train_run(&mut model, &store, &config, run);

        // Print compact progress (first, every 5th, and last epoch)
        for r in &results {
            if r.epoch == 0 || r.epoch % 5 == 0 || r.epoch == config.epochs - 1 {
                println!(
                    "  epoch {:>3}: loss={:.6} ~{:.1} MMR  ({:.1}s)",
                    r.epoch,
                    r.loss,
                    r.approx_rmse,
                    r.elapsed_ms as f64 / 1000.0,
                );
            }
        }

        if let (Some(first), Some(last)) = (results.first(), results.last()) {
            let improvement = first.approx_rmse - last.approx_rmse;
            let pct = improvement / first.approx_rmse * 100.0;
            let verdict = if pct > 5.0 { "LEARNING" } else { "FLAT" };
            println!(
                "  => {verdict}: {:.1} -> {:.1} MMR ({improvement:+.1}, {pct:+.1}%)",
                first.approx_rmse, last.approx_rmse,
            );
            summaries.push(RunSummary {
                label: format!("{run}"),
                start_rmse: first.approx_rmse,
                end_rmse: last.approx_rmse,
                improvement,
            });
        }
    }

    // Final comparison table
    println!("\n=== COMPARISON TABLE ===");
    println!(
        "{:<35} {:>10} {:>10} {:>12} {:>8}",
        "Configuration", "Start", "End", "Improvement", "Status"
    );
    println!("{}", "-".repeat(79));

    let mut best_idx = 0;
    let mut best_improvement = f32::MIN;

    for (i, s) in summaries.iter().enumerate() {
        let status = if s.improvement > 50.0 {
            "OK"
        } else if s.improvement > 5.0 {
            "SLOW"
        } else {
            "FLAT"
        };
        println!(
            "{:<35} {:>8.1} {:>8.1} {:>+10.1} {:>8}",
            s.label, s.start_rmse, s.end_rmse, s.improvement, status,
        );
        if s.improvement > best_improvement {
            best_improvement = s.improvement;
            best_idx = i;
        }
    }

    // Recommendations
    println!("\n=== RECOMMENDATIONS ===");
    if best_improvement < 5.0 {
        println!("  No configuration showed significant learning.");
        println!("  Possible issues:");
        println!("    1. Features might be all zeros -- check segment file contents");
        println!(
            "    2. Model might be too large for this tiny dataset -- try reducing hidden sizes"
        );
        println!("    3. LSTM gradient flow might be blocked -- try shorter --seq-len (e.g. 30)");
    } else if let Some(best) = summaries.get(best_idx) {
        println!("  Best config: {}", best.label);
        println!(
            "  Achieved {:.1} MMR improvement in {} epochs",
            best.improvement, config.epochs
        );

        let any_mse_works = summaries
            .iter()
            .any(|s| matches!(s.label.as_str(), l if l.starts_with("MSE")) && s.improvement > 20.0);
        let any_huber_works = summaries
            .iter()
            .any(|s| s.label.starts_with("Huber") && s.improvement > 20.0);

        if any_mse_works && !any_huber_works {
            println!(
                "  MSE works but Huber does not -- increase Huber delta or use MSE for training"
            );
        } else if any_mse_works && any_huber_works {
            println!("  Both MSE and Huber can learn -- safe to use either for full training");
        }
    }
}
