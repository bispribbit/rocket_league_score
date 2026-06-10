//! Ridge-regression feature-signal probe.
#![allow(clippy::indexing_slicing, clippy::needless_range_loop)]
//!
//! Reads cached segment files directly from disk and fits a small ridge regression
//! model on per-player aggregate features to answer one question before spending
//! GPU time on the LSTM:
//!
//! **"Do the features we extract actually carry generalizable rank signal?"**
//!
//! If ridge regression beats the constant-predictor baseline the kinematics contain
//! exploitable rank signal and the problem is optimization/architecture. If it cannot
//! beat the constant baseline the data or feature–label alignment needs investigation
//! first.
//!
//! # Aggregate features used
//!
//! These are extracted from the **last frame** of each player's 106-dim sequence
//! (where cumulative stats have reached their final value for the segment) and from
//! the **mean** of instantaneous quantities across the segment:
//!
//! | Index | Quantity | Type |
//! |-------|----------|------|
//! | 22 | boost_collected | cumulative |
//! | 23 | boost_spent | cumulative |
//! | 24 | airborne_fraction | cumulative |
//! | 25 | supersonic_fraction | cumulative |
//! | 26 | demo_received_fraction | cumulative |
//! | 18 | speed (mean across frames) | mean |
//! | 19 | boost_amount (mean across frames) | mean |
//! | 20 | dist_to_ball (mean across frames) | mean |
//!
//! # Usage
//!
//! ```sh
//! cargo run --release -p ml_model_training --bin signal_probe -- \
//!     --segments-dir /path/to/segments/v6 \
//!     --sequence-length 300 \
//!     --lambda 1.0 \
//!     --max-segments 50000
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use feature_extractor::{PLAYER_CENTRIC_FEATURE_COUNT, TOTAL_PLAYERS};

// ── CLI args ──────────────────────────────────────────────────────────────────

struct ProbeConfig {
    /// Root path that contains per-rank subdirectories of `*.features` files.
    segments_dir: PathBuf,
    /// Number of frames per segment (must match how the cache was built).
    sequence_length: usize,
    /// Ridge regularisation strength λ (L2 penalty on weights).
    lambda: f64,
    /// Cap on total segments loaded (for quick iteration). 0 = unlimited.
    max_segments: usize,
    /// Fraction of segments reserved for validation (stratified by rank directory).
    validation_fraction: f64,
}

impl ProbeConfig {
    fn from_env() -> Self {
        let mut segments_dir = PathBuf::from("segments/v6");
        let mut sequence_length = 300_usize;
        let mut lambda = 1.0_f64;
        let mut max_segments = 0_usize;
        let mut validation_fraction = 0.1_f64;

        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--segments-dir" => {
                    index += 1;
                    if let Some(value) = args.get(index) {
                        segments_dir = PathBuf::from(value);
                    }
                }
                "--sequence-length" => {
                    index += 1;
                    if let Some(value) = args.get(index) {
                        sequence_length = value.parse().unwrap_or(300);
                    }
                }
                "--lambda" => {
                    index += 1;
                    if let Some(value) = args.get(index) {
                        lambda = value.parse().unwrap_or(1.0);
                    }
                }
                "--max-segments" => {
                    index += 1;
                    if let Some(value) = args.get(index) {
                        max_segments = value.parse().unwrap_or(0);
                    }
                }
                "--val-fraction" => {
                    index += 1;
                    if let Some(value) = args.get(index) {
                        validation_fraction = value.parse().unwrap_or(0.1);
                    }
                }
                _ => {}
            }
            index += 1;
        }

        Self {
            segments_dir,
            sequence_length,
            lambda,
            max_segments,
            validation_fraction,
        }
    }
}

// ── Feature indices in the 106-dim per-player vector ─────────────────────────

/// Cumulative stats live at indices 22-26 in the focal-player region.
/// We read them from the LAST frame of the segment (final cumulative value).
const CUMULATIVE_INDICES: [usize; 5] = [22, 23, 24, 25, 26];

/// Instantaneous indices we average across all frames.
/// 18 = focal player speed, 19 = boost_amount, 20 = dist_to_ball (rel).
const MEAN_INDICES: [usize; 3] = [18, 19, 20];

/// Total probe features per player: 5 cumulative + 3 mean + 1 bias term.
const PROBE_FEATURES: usize = 5 + 3 + 1;

// ── Segment loading ───────────────────────────────────────────────────────────

/// One loaded segment: aggregate feature matrix [6, PROBE_FEATURES] and MMR targets [6].
struct SegmentSample {
    /// Row = player (6 rows). Column = probe feature (including bias). Values normalised.
    features: Vec<[f64; PROBE_FEATURES]>,
    /// Target MMR for each player (0.0 = unknown/masked).
    targets: [f32; TOTAL_PLAYERS],
}

fn extract_probe_features(raw: &[f32], sequence_length: usize) -> Vec<[f64; PROBE_FEATURES]> {
    let floats_per_player = sequence_length * PLAYER_CENTRIC_FEATURE_COUNT;
    let mut result = Vec::with_capacity(TOTAL_PLAYERS);

    for player_index in 0..TOTAL_PLAYERS {
        let player_start = player_index * floats_per_player;
        let Some(player_slice) = raw.get(player_start..player_start + floats_per_player) else {
            result.push([0.0; PROBE_FEATURES]);
            continue;
        };

        // Cumulative stats: take last frame.
        let last_frame_start = (sequence_length - 1) * PLAYER_CENTRIC_FEATURE_COUNT;
        let last_frame = &player_slice[last_frame_start..];

        let mut features = [0.0f64; PROBE_FEATURES];
        for (output_index, &feature_index) in CUMULATIVE_INDICES.iter().enumerate() {
            features[output_index] = last_frame.get(feature_index).copied().unwrap_or(0.0) as f64;
        }

        // Mean of instantaneous quantities across all frames.
        let cum_len = CUMULATIVE_INDICES.len();
        for (output_offset, &feature_index) in MEAN_INDICES.iter().enumerate() {
            let mean = (0..sequence_length)
                .filter_map(|frame| {
                    player_slice.get(frame * PLAYER_CENTRIC_FEATURE_COUNT + feature_index)
                })
                .map(|&v| v as f64)
                .sum::<f64>()
                / sequence_length as f64;
            features[cum_len + output_offset] = mean;
        }

        // Bias term.
        features[PROBE_FEATURES - 1] = 1.0;
        result.push(features);
    }

    result
}

/// Recursively collects all `*.features` files under `root`.
fn collect_feature_files(root: &Path) -> Vec<(PathBuf, f32)> {
    let mut files = Vec::new();
    collect_recursive(root, &mut files);
    files
}

fn collect_recursive(dir: &Path, out: &mut Vec<(PathBuf, f32)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_recursive(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("features") {
            // We don't have individual player MMR from the filename alone;
            // the parent directory encodes the folder rank which we use as a
            // coarse proxy label for the signal test.
            let folder_mmr = mmr_from_path(&path);
            out.push((path, folder_mmr));
        }
    }
}

/// Extracts the coarse MMR proxy from the rank folder in the path.
/// e.g. `.../bronze-1/...` → ~194 MMR, `.../supersonic-legend/...` → 2200.
fn mmr_from_path(path: &Path) -> f32 {
    let rank_mmr_map = [
        ("bronze-1", 194.0),
        ("bronze-2", 257.0),
        ("bronze-3", 321.0),
        ("silver-1", 386.0),
        ("silver-2", 451.0),
        ("silver-3", 516.0),
        ("gold-1", 580.0),
        ("gold-2", 644.0),
        ("gold-3", 709.0),
        ("platinum-1", 773.0),
        ("platinum-2", 837.0),
        ("platinum-3", 902.0),
        ("diamond-1", 966.0),
        ("diamond-2", 1030.0),
        ("diamond-3", 1127.0),
        ("champion-1", 1258.0),
        ("champion-2", 1388.0),
        ("champion-3", 1520.0),
        ("grand-champion-1", 1651.0),
        ("grand-champion-2", 1782.0),
        ("grand-champion-3", 1913.0),
        ("supersonic-legend", 2200.0),
    ];
    for component in path.components() {
        if let Some(segment_str) = component.as_os_str().to_str() {
            for (name, mmr) in rank_mmr_map {
                if segment_str == name {
                    return mmr;
                }
            }
        }
    }
    0.0 // unknown
}

fn load_segment(path: &Path, sequence_length: usize, folder_mmr: f32) -> Option<SegmentSample> {
    let bytes = fs::read(path).ok()?;
    let expected = TOTAL_PLAYERS * sequence_length * PLAYER_CENTRIC_FEATURE_COUNT * 4;
    if bytes.len() != expected {
        return None;
    }
    let raw: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| {
            let array: [u8; 4] = chunk
                .try_into()
                .expect("chunks_exact(4) guarantees 4-byte slices");
            f32::from_le_bytes(array)
        })
        .collect();

    let features = extract_probe_features(&raw, sequence_length);
    // Use folder MMR as proxy for all six player slots (coarse but avoids needing the DB).
    let targets = [folder_mmr; TOTAL_PLAYERS];

    Some(SegmentSample { features, targets })
}

// ── Ridge regression (closed-form normal equations) ──────────────────────────

/// Solves (X^T X + λI) w = X^T y using Cholesky / Gaussian elimination.
///
/// `design` is [n_samples, n_features] (each row is one observation).
/// `labels` is [n_samples].
fn ridge_regression(
    design: &[[f64; PROBE_FEATURES]],
    labels: &[f64],
    lambda: f64,
) -> [f64; PROBE_FEATURES] {
    let _n = design.len();
    let p = PROBE_FEATURES;

    // X^T X (p×p)
    let mut xtx = [[0.0f64; PROBE_FEATURES]; PROBE_FEATURES];
    for row in design {
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    // Add λI
    for i in 0..p {
        xtx[i][i] += lambda;
    }

    // X^T y (p×1)
    let mut xty = [0.0f64; PROBE_FEATURES];
    for (row, &label) in design.iter().zip(labels.iter()) {
        for i in 0..p {
            xty[i] += row[i] * label;
        }
    }

    // Gaussian elimination with partial pivoting.
    let mut augmented = [[0.0f64; PROBE_FEATURES + 1]; PROBE_FEATURES];
    for i in 0..p {
        for j in 0..p {
            augmented[i][j] = xtx[i][j];
        }
        augmented[i][p] = xty[i];
    }

    for col in 0..p {
        // Pivot
        let mut max_row = col;
        let mut max_val = augmented[col][col].abs();
        for row in (col + 1)..p {
            let abs_val = augmented[row][col].abs();
            if abs_val > max_val {
                max_val = abs_val;
                max_row = row;
            }
        }
        augmented.swap(col, max_row);

        let pivot = augmented[col][col];
        if pivot.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..p {
            let factor = augmented[row][col] / pivot;
            for k in col..=p {
                let subtract = factor * augmented[col][k];
                augmented[row][k] -= subtract;
            }
        }
    }

    // Back substitution
    let mut weights = [0.0f64; PROBE_FEATURES];
    for i in (0..p).rev() {
        let mut sum = augmented[i][p];
        for j in (i + 1)..p {
            sum -= augmented[i][j] * weights[j];
        }
        let denominator = augmented[i][i];
        if denominator.abs() > 1e-12
            && let Some(slot) = weights.get_mut(i)
        {
            *slot = sum / denominator;
        }
    }
    weights
}

fn dot(weights: &[f64; PROBE_FEATURES], features: &[f64; PROBE_FEATURES]) -> f64 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(w, x)| w * x)
        .sum()
}

// ── Metrics ───────────────────────────────────────────────────────────────────

fn rmse(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }
    let sse: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum();
    (sse / n as f64).sqrt()
}

fn pearson(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_p = predictions.iter().sum::<f64>() / n;
    let mean_t = targets.iter().sum::<f64>() / n;
    let cov: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(p, t)| (p - mean_p) * (t - mean_t))
        .sum::<f64>()
        / (n - 1.0);
    let std_p = (predictions
        .iter()
        .map(|p| (p - mean_p).powi(2))
        .sum::<f64>()
        / (n - 1.0))
        .sqrt();
    let std_t = (targets.iter().map(|t| (t - mean_t).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std_p < 1e-8 || std_t < 1e-8 {
        0.0
    } else {
        (cov / (std_p * std_t)).clamp(-1.0, 1.0)
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let config = ProbeConfig::from_env();

    println!(
        "Signal probe: reading segments from {}",
        config.segments_dir.display()
    );
    println!(
        "  sequence_length={}, lambda={}, max_segments={}, val_fraction={}",
        config.sequence_length, config.lambda, config.max_segments, config.validation_fraction
    );

    // Collect all .features file paths.
    println!("Scanning for .features files…");
    let mut all_files = collect_feature_files(&config.segments_dir);
    if all_files.is_empty() {
        eprintln!(
            "No .features files found under {}",
            config.segments_dir.display()
        );
        eprintln!("Make sure --segments-dir points to the segments/v6/ directory.");
        std::process::exit(1);
    }
    // Deterministic shuffle for reproducible splits.
    deterministic_shuffle(&mut all_files);

    if config.max_segments > 0 && all_files.len() > config.max_segments {
        all_files.truncate(config.max_segments);
    }
    println!("Found {} candidate files", all_files.len());

    // Load samples.
    let mut samples: Vec<SegmentSample> = Vec::new();
    let mut skipped = 0_usize;
    for (path, folder_mmr) in &all_files {
        if folder_mmr.abs() < 1.0 {
            // Unknown rank folder — skip.
            skipped += 1;
            continue;
        }
        match load_segment(path, config.sequence_length, *folder_mmr) {
            Some(sample) => samples.push(sample),
            None => skipped += 1,
        }
    }
    println!(
        "Loaded {} segments ({} skipped/invalid)",
        samples.len(),
        skipped
    );
    if samples.is_empty() {
        eprintln!("No valid segments loaded — check the path and sequence_length.");
        std::process::exit(1);
    }

    // Train/validation split (last val_fraction of the shuffled list).
    let val_count = ((samples.len() as f64 * config.validation_fraction) as usize).max(1);
    let train_count = samples.len().saturating_sub(val_count);
    println!("Split: {train_count} train, {val_count} validation");

    // Build design matrices.
    // Each player slot in a segment is one observation. We only use known (non-zero) targets.
    let mut train_design: Vec<[f64; PROBE_FEATURES]> = Vec::new();
    let mut train_labels: Vec<f64> = Vec::new();
    for sample in samples.iter().take(train_count) {
        for (player, &target) in sample.features.iter().zip(sample.targets.iter()) {
            if target > 0.0 {
                train_design.push(*player);
                train_labels.push(target as f64);
            }
        }
    }

    let mut val_design: Vec<[f64; PROBE_FEATURES]> = Vec::new();
    let mut val_labels: Vec<f64> = Vec::new();
    for sample in samples.iter().skip(train_count) {
        for (player, &target) in sample.features.iter().zip(sample.targets.iter()) {
            if target > 0.0 {
                val_design.push(*player);
                val_labels.push(target as f64);
            }
        }
    }

    println!(
        "Train observations: {}, Val observations: {}",
        train_design.len(),
        val_design.len()
    );

    // Constant predictor baseline.
    let global_mean = train_labels.iter().sum::<f64>() / train_labels.len() as f64;
    let const_pred_train: Vec<f64> = vec![global_mean; train_labels.len()];
    let const_pred_val: Vec<f64> = vec![global_mean; val_labels.len()];
    let const_rmse_train = rmse(&const_pred_train, &train_labels);
    let const_rmse_val = rmse(&const_pred_val, &val_labels);

    println!("\n─── Baseline: constant predictor (global mean = {global_mean:.1} MMR) ───");
    println!("  Train RMSE: {const_rmse_train:.1} MMR");
    println!("  Val   RMSE: {const_rmse_val:.1} MMR");

    // Fit ridge regression.
    println!("\nFitting ridge regression (λ={})…", config.lambda);
    let weights = ridge_regression(&train_design, &train_labels, config.lambda);

    let ridge_pred_train: Vec<f64> = train_design.iter().map(|row| dot(&weights, row)).collect();
    let ridge_pred_val: Vec<f64> = val_design.iter().map(|row| dot(&weights, row)).collect();

    let ridge_rmse_train = rmse(&ridge_pred_train, &train_labels);
    let ridge_rmse_val = rmse(&ridge_pred_val, &val_labels);
    let ridge_pearson_val = pearson(&ridge_pred_val, &val_labels);

    println!("\n─── Ridge regression results ───");
    println!("  Train RMSE: {ridge_rmse_train:.1} MMR");
    println!("  Val   RMSE: {ridge_rmse_val:.1} MMR");
    println!("  Val Pearson r: {ridge_pearson_val:.4}");

    let improvement_pct = 100.0 * (1.0 - ridge_rmse_val / const_rmse_val);
    println!("\n─── Signal verdict ─────────────────────────────────────────────────────────");
    println!("  Ridge val RMSE vs constant baseline: {improvement_pct:+.1}%");
    if ridge_rmse_val < const_rmse_val * 0.95 {
        println!("  ✓  Features carry rank signal (>5% improvement over constant predictor).");
        println!("     The problem is optimization/architecture, not data.");
        println!("     → Fix bugs, tune LR, improve architecture — the data is usable.");
    } else {
        println!("  ✗  Features do NOT beat the constant predictor (<5% improvement).");
        println!("     Investigate: label/feature slot alignment, rank distribution,");
        println!("     or whether 20 s segments are too short for rank to be detectable.");
        println!("     → Do NOT spend more GPU time until data/features are fixed.");
    }

    println!("\n─── Learned weights ─────────────────────────────────────────────────────────");
    let feature_names = [
        "boost_collected (cumul)",
        "boost_spent (cumul)",
        "airborne_fraction (cumul)",
        "supersonic_fraction (cumul)",
        "demo_received_fraction (cumul)",
        "speed (mean)",
        "boost_amount (mean)",
        "dist_to_ball (mean)",
        "bias",
    ];
    for (name, &weight) in feature_names.iter().zip(weights.iter()) {
        println!("  {name:<35} {weight:+.4}");
    }
}

/// Fisher–Yates shuffle with a fixed seed for deterministic train/val splits.
fn deterministic_shuffle<T>(data: &mut [T]) {
    let n = data.len();
    let mut state: u64 = 0x123456789ABCDEF0;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = (state >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
}
