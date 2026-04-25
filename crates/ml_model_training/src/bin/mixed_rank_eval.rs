#![allow(clippy::indexing_slicing)]

//! Mixed-rank lobby evaluation.
//!
//! Assembles "synthetic lobbies" by drawing one player slot from each of six different
//! rank buckets (e.g. Bronze-1, Silver-1, Gold-1, Platinum-1, Diamond-1, SSL) and
//! stacking their individual feature sequences into a single 6-player input batch.
//!
//! This tests whether the model can assign different MMR predictions to players in the
//! same lobby when their individual skill signals differ — the core capability needed
//! to detect smurfs (a high-skill player in a low-rank lobby).
//!
//! # Design
//!
//! A synthetic lobby is constructed from the segment cache:
//! - Pick one segment from each of six distinct ranks.
//! - Stack their `[seq_len, features]` tensors as the six player slots: `[6, seq, feat]`.
//! - The per-slot target is each segment's canonical MMR.
//! - Record per-slot RMSE across all synthetic lobbies.
//!
//! Goal: per-slot RMSE < 400 MMR on synthetic mixed lobbies.
//!
//! Usage:
//!   cargo run --bin mixed_rank_eval --release -- MODEL_PATH [SEGMENTS_DIR] [--lobbies N]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use feature_extractor::PLAYER_CENTRIC_FEATURE_COUNT;
use ml_model::SequenceModel;
use ml_model_training::load_checkpoint;
use replay_structs::Rank;

// Eval-only path: NdArray on CPU is plenty for ~500 lobbies and avoids
// pulling the conflicting LibTorch dep into the crate.
type Backend = NdArray;
type BackendDevice = NdArrayDevice;

struct EvalConfig {
    model_path: PathBuf,
    segments_dir: PathBuf,
    num_lobbies: usize,
    sequence_length: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/lstm_v13/checkpoint_best"),
            segments_dir: PathBuf::from("ballchasing/segments/v6"),
            num_lobbies: 500,
            sequence_length: 300,
        }
    }
}

fn parse_args() -> EvalConfig {
    let mut config = EvalConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--lobbies") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.num_lobbies = val.parse().unwrap_or(config.num_lobbies);
                }
            }
            Some("--seq-len") => {
                i += 1;
                if let Some(val) = args.get(i) {
                    config.sequence_length = val.parse().unwrap_or(config.sequence_length);
                }
            }
            Some(path) if !path.starts_with("--") => {
                if i == 1 {
                    config.model_path = PathBuf::from(path);
                } else {
                    config.segments_dir = PathBuf::from(path);
                }
            }
            _ => {}
        }
        i += 1;
    }
    config
}

// =============================================================================
// Rank inference from path (same as overfit_test.rs)
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
// Segment discovery and grouping by rank
// =============================================================================

struct LabelledSegment {
    path: PathBuf,
    rank: Rank,
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
    let mut result = Vec::new();
    for path in files {
        if std::fs::metadata(&path).map_or(true, |m| m.len() != expected_size as u64) {
            continue;
        }
        let Some(rank) = rank_from_path(&path) else {
            continue;
        };
        result.push(LabelledSegment { path, rank });
    }
    result
}

// =============================================================================
// Mixed-rank lobby evaluation
// =============================================================================

/// Reads one segment's features from disk and returns the per-player feature slice
/// for a *single* player slot (player 0, which is the focal player in player-centric layout).
///
/// Player-centric layout: `[6 players, seq_len, features]`.
/// This function reads the whole blob and extracts the first player's sub-tensor.
fn read_player0_features(path: &Path, sequence_length: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
        .collect();
    // Player 0 occupies floats[0 .. seq_len * features].
    let player_size = sequence_length * PLAYER_CENTRIC_FEATURE_COUNT;
    if floats.len() < 6 * player_size {
        return None;
    }
    Some(floats[..player_size].to_vec())
}

/// Runs mixed-rank lobby evaluation.
///
/// Constructs `num_lobbies` synthetic 6-player lobbies where each slot comes
/// from a different rank bucket.  Runs inference and records per-slot RMSE.
///
/// Returns `true` if the per-slot RMSE for SSL is < 400 MMR.
fn evaluate_mixed_lobbies(
    model: &SequenceModel<Backend>,
    by_rank: &HashMap<Rank, Vec<&LabelledSegment>>,
    sequence_length: usize,
    num_lobbies: usize,
    device: BackendDevice,
) -> bool {
    // Pick 6 ranks with available segments to form a diverse lobby.
    let lobby_ranks: Vec<Rank> = {
        let preferred = [
            Rank::Bronze1,
            Rank::Silver1,
            Rank::Gold1,
            Rank::Platinum1,
            Rank::Diamond1,
            Rank::SupersonicLegend,
        ];
        let available: Vec<Rank> = preferred
            .iter()
            .filter(|r| by_rank.contains_key(r))
            .copied()
            .collect();
        if available.len() < 2 {
            println!("  Not enough ranks available for mixed lobbies — skipping.");
            return true;
        }
        available
    };

    let num_slots = lobby_ranks.len();
    println!("  Lobby composition ({num_slots} slots):");
    for rank in &lobby_ranks {
        println!("    slot: {rank:?}");
    }

    let target_mmrs: Vec<f32> = lobby_ranks
        .iter()
        .map(|r| canonical_mmr_for_rank(*r))
        .collect();

    // Per-slot accumulators.
    let mut slot_sq_err = vec![0.0f64; num_slots];
    let mut slot_count = vec![0usize; num_slots];

    let player_feat_size = sequence_length * PLAYER_CENTRIC_FEATURE_COUNT;

    let mut lobby_idx = 0usize;
    let mut seed_state: u64 = 42;

    while lobby_idx < num_lobbies {
        // Build one mixed lobby: pick one segment per rank slot.
        let mut lobby_input: Vec<f32> = Vec::with_capacity(num_slots * player_feat_size);
        let mut valid = true;

        for rank in &lobby_ranks {
            let segs = &by_rank[rank];
            // Simple LCG pick.
            seed_state = seed_state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let seg_idx = ((seed_state >> 33) as usize) % segs.len();
            let seg = segs[seg_idx];

            let Some(features) = read_player0_features(&seg.path, sequence_length) else {
                valid = false;
                break;
            };
            lobby_input.extend_from_slice(&features);
        }

        if !valid || lobby_input.len() != num_slots * player_feat_size {
            continue;
        }

        // Input shape: [num_slots, seq_len, features]
        let input = Tensor::<Backend, 1>::from_floats(lobby_input.as_slice(), &device).reshape([
            num_slots,
            sequence_length,
            PLAYER_CENTRIC_FEATURE_COUNT,
        ]);

        // Forward pass: output [1, num_slots] → reshape [num_slots].
        // We need batch_size=1, but our model expects [batch*6, seq, feat].
        // The lobby is exactly the 6-player input, so batch_size = 1 and
        // the input is already [6, seq, feat].
        let output = model.forward(input); // [1, num_slots]
        let preds: Vec<f32> = output.into_data().to_vec().unwrap_or_default();

        for (slot, (pred, target)) in preds.iter().zip(target_mmrs.iter()).enumerate() {
            let err = (*pred - *target) as f64;
            if let Some(sq) = slot_sq_err.get_mut(slot) {
                *sq += err * err;
            }
            if let Some(cnt) = slot_count.get_mut(slot) {
                *cnt += 1;
            }
        }

        lobby_idx += 1;
    }

    println!("\n  Per-slot RMSE across {num_lobbies} mixed lobbies:");
    let mut ssl_rmse = f32::MAX;
    for (slot, rank) in lobby_ranks.iter().enumerate() {
        let n = slot_count.get(slot).copied().unwrap_or(0);
        if n == 0 {
            continue;
        }
        let sq = slot_sq_err.get(slot).copied().unwrap_or(0.0);
        let rmse = (sq / n as f64).sqrt() as f32;
        let marker = if *rank == Rank::SupersonicLegend {
            " ← key"
        } else {
            ""
        };
        println!(
            "    {:>22}: {:>7.1} MMR  (n={n}){marker}",
            format!("{rank:?}"),
            rmse
        );
        if *rank == Rank::SupersonicLegend {
            ssl_rmse = rmse;
        }
    }

    let pass = ssl_rmse < 400.0;
    println!(
        "\n  SSL slot RMSE: {ssl_rmse:.1} MMR  {}",
        if pass {
            "PASS ✓ (< 400)"
        } else {
            "FAIL ✗ (need < 400)"
        }
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

    println!("=== Mixed-Rank Lobby Evaluation ===");
    println!("  Model      : {}", config.model_path.display());
    println!("  Segments   : {}", config.segments_dir.display());
    println!("  Num lobbies: {}", config.num_lobbies);
    println!("  Seq len    : {}", config.sequence_length);

    let device = BackendDevice::Cpu;

    println!("\nLoading model...");
    let model_path_str = config.model_path.to_string_lossy();
    let model: SequenceModel<Backend> = match load_checkpoint(&model_path_str, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model checkpoint: {e}");
            std::process::exit(1);
        }
    };

    println!("Discovering segments...");
    let all_segments = discover_all_segments(&config.segments_dir, config.sequence_length);
    if all_segments.is_empty() {
        eprintln!("No segments found in {}", config.segments_dir.display());
        std::process::exit(1);
    }

    // Group by rank.
    let mut by_rank: HashMap<Rank, Vec<&LabelledSegment>> = HashMap::new();
    for seg in &all_segments {
        by_rank.entry(seg.rank).or_default().push(seg);
    }

    println!(
        "Found {} segments across {} ranks.",
        all_segments.len(),
        by_rank.len()
    );

    let pass = evaluate_mixed_lobbies(
        &model,
        &by_rank,
        config.sequence_length,
        config.num_lobbies,
        device,
    );

    println!("\n=== RESULT: {} ===", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }
}
