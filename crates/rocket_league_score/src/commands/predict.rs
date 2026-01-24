//! Predict command - runs inference on a replay file using the LSTM sequence model.

use std::path::Path;

use anyhow::Result;
use burn::backend::Wgpu;
use feature_extractor::extract_frame_features;
use ml_model::{SequenceModel, load_checkpoint, predict};
use replay_parser::parse_replay;
use replay_structs::RankDivision;
use tracing::{info, warn};

use super::init_device;

type Backend = Wgpu;

/// Default sequence length for inference (should match training config).
const DEFAULT_SEQUENCE_LENGTH: usize = 300;

/// Runs the predict command.
///
/// # Arguments
///
/// * `replay_path` - Path to the replay file to analyze
/// * `model_path` - Path to model checkpoint file
///
/// # Errors
///
/// Returns an error if prediction fails.
pub fn run(replay_path: &Path, model_path: &Path) -> Result<()> {
    info!(
        replay = %replay_path.display(),
        model_path = %model_path.display(),
        "Predicting player MMR with LSTM model"
    );

    // Try to load sequence length from config file if available
    let config_path = format!("{}.config.json", model_path.to_string_lossy());
    let sequence_length = if std::path::Path::new(&config_path).exists() {
        if let Ok(config_data) = std::fs::read_to_string(&config_path) {
            if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_data) {
                config_json
                    .get("sequence_length")
                    .and_then(serde_json::Value::as_u64)
                    .map_or(DEFAULT_SEQUENCE_LENGTH, |v| v as usize)
            } else {
                DEFAULT_SEQUENCE_LENGTH
            }
        } else {
            DEFAULT_SEQUENCE_LENGTH
        }
    } else {
        DEFAULT_SEQUENCE_LENGTH
    };

    info!(sequence_length, "Using sequence length");

    // Load model weights
    let device = init_device();
    let model: SequenceModel<Backend> = load_checkpoint(&model_path.to_string_lossy(), &device)?;

    // Parse the replay
    let parsed = parse_replay(replay_path)?;

    if parsed.frames.is_empty() {
        warn!("No frames found in replay");
        return Ok(());
    }

    info!(frames = parsed.frames.len(), "Analyzing gameplay sequence");

    // Extract player names from the first frame (players are sorted by team then actor_id)
    let player_names: Vec<String> = parsed
        .frames
        .first()
        .map(|frame| {
            frame
                .players
                .iter()
                .map(|p| p.name.as_ref().clone())
                .collect()
        })
        .unwrap_or_default();

    // Extract features from all frames (no score context available in predict mode)
    let frame_features: Vec<_> = parsed.frames.iter().map(extract_frame_features).collect();

    // Show predictions per segment
    let num_segments = frame_features.len() / sequence_length;
    info!("=== MMR Per Segment ({} segments) ===", num_segments.max(1));

    let mut blue_avg_segments = Vec::with_capacity(num_segments);
    let mut orange_avg_segments = Vec::with_capacity(num_segments);
    // Track per-player predictions across all segments (6 players total)
    let mut player_predictions: Vec<Vec<f32>> =
        (0..6).map(|_| Vec::with_capacity(num_segments)).collect();

    for segment_idx in 0..num_segments.max(1) {
        let start = segment_idx * sequence_length;
        let end = (start + sequence_length).min(frame_features.len());
        let Some(segment_frames) = frame_features.get(start..end) else {
            continue;
        };

        let segment_prediction = predict(&model, segment_frames, &device, sequence_length);

        let start_time = segment_frames.first().map_or(0.0, |f| f.time);
        let end_time = segment_frames.last().map_or(0.0, |f| f.time);

        info!(
            "  Segment {} ({:.1}s - {:.1}s)",
            segment_idx + 1,
            start_time,
            end_time
        );

        // Store per-player predictions
        for (player_idx, &mmr) in segment_prediction.iter().enumerate() {
            if let Some(predictions) = player_predictions.get_mut(player_idx) {
                predictions.push(mmr);
            }
        }

        // Blue Team
        let blue_avg = segment_prediction.iter().take(3).sum::<f32>() / 3.0;
        blue_avg_segments.push(blue_avg);
        info!("    Blue Team (avg: {:.0}):", blue_avg);
        for (player_idx, mmr) in segment_prediction.iter().take(3).enumerate() {
            let player_name = player_names
                .get(player_idx)
                .map_or("Unknown", String::as_str);
            info!("      {} - MMR: {:.0}", player_name, mmr);
        }

        // Orange Team
        let orange_avg = segment_prediction.iter().skip(3).sum::<f32>() / 3.0;
        orange_avg_segments.push(orange_avg);
        info!("    Orange Team (avg: {:.0}):", orange_avg);
        for (player_idx, mmr) in segment_prediction.iter().skip(3).enumerate() {
            let player_name = player_names
                .get(player_idx + 3)
                .map_or("Unknown", String::as_str);
            info!("      {} - MMR: {:.0}", player_name, mmr);
        }
        info!("");
    }

    let blue_avg_final = blue_avg_segments.iter().sum::<f32>() / blue_avg_segments.len() as f32;
    let orange_avg_final =
        orange_avg_segments.iter().sum::<f32>() / orange_avg_segments.len() as f32;

    // Calculate per-player averages
    let player_averages: Vec<f32> = player_predictions
        .iter()
        .map(|predictions| {
            if predictions.is_empty() {
                0.0
            } else {
                predictions.iter().sum::<f32>() / predictions.len() as f32
            }
        })
        .collect();

    info!("=== Game Summary ===");
    info!("");
    info!("Blue Team Average MMR: {:.0}", blue_avg_final);
    for (player_idx, &avg_mmr) in player_averages.iter().take(3).enumerate() {
        let player_name = player_names
            .get(player_idx)
            .map_or("Unknown", String::as_str);
        let rank = RankDivision::from(avg_mmr);
        info!("  {} - MMR: {:.0} ({})", player_name, avg_mmr, rank);
    }

    info!("Orange Team Average MMR: {:.0}", orange_avg_final);
    for (player_idx, &avg_mmr) in player_averages.iter().skip(3).enumerate() {
        let player_name = player_names
            .get(player_idx + 3)
            .map_or("Unknown", String::as_str);
        let rank = RankDivision::from(avg_mmr);
        info!("  {} - MMR: {:.0} ({})", player_name, avg_mmr, rank);
    }

    Ok(())
}
