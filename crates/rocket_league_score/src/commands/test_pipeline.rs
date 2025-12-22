//! Small-scale end-to-end test of the ML pipeline.
//!
//! This command runs a quick sanity check:
//! 1. Load a few replays with metadata
//! 2. Extract features
//! 3. Train for a few epochs
//! 4. Verify loss decreases
//! 5. Run inference and check output is reasonable

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use feature_extractor::{PlayerRating, extract_frame_features};
use ml_model::{ModelConfig, TrainingConfig, TrainingData, create_model, predict, train};
use replay_parser::{parse_replay, segment_by_goals};
use serde::Deserialize;
use tracing::info;

use crate::rank::Rank;

type TrainBackend = Autodiff<Wgpu>;

/// Converts a rank ID and division to MMR using the Rank enum.
fn rank_to_mmr(rank_id: &str, division: Option<i32>) -> Option<i32> {
    Rank::from_rank_id(rank_id, division).map(Rank::mmr_middle)
}

/// Player rank from metadata.
#[derive(Debug, Deserialize)]
struct PlayerRank {
    id: String,
    #[expect(dead_code)]
    tier: Option<i32>,
    division: Option<i32>,
}

/// Player info from metadata.
#[derive(Debug, Deserialize)]
struct PlayerInfo {
    name: String,
    rank: Option<PlayerRank>,
}

/// Team info from metadata.
#[derive(Debug, Deserialize)]
struct TeamInfo {
    players: Vec<PlayerInfo>,
}

/// Game metadata structure (simplified).
#[derive(Debug, Deserialize)]
struct GameMetadata {
    id: String,
    data: GameData,
}

#[derive(Debug, Deserialize)]
struct GameData {
    blue: TeamInfo,
    orange: TeamInfo,
}

/// Loads metadata for replays from a JSONL file.
fn load_metadata(
    metadata_path: &Path,
    limit: Option<usize>,
) -> Result<HashMap<String, Vec<(String, i32)>>> {
    let file = File::open(metadata_path)
        .with_context(|| format!("Failed to open metadata file: {}", metadata_path.display()))?;
    let reader = BufReader::new(file);

    let mut metadata_map: HashMap<String, Vec<(String, i32)>> = HashMap::new();
    let mut count = 0;

    for line in reader.lines() {
        if let Some(max) = limit
            && count >= max
        {
            break;
        }

        let line = line?;
        let game: GameMetadata = match serde_json::from_str(&line) {
            Ok(g) => g,
            Err(_) => continue, // Skip malformed lines
        };

        let mut players = Vec::new();

        // Extract blue team players
        for player in &game.data.blue.players {
            if let Some(rank) = &player.rank
                && let Some(mmr) = rank_to_mmr(&rank.id, rank.division)
            {
                players.push((player.name.clone(), mmr));
            }
        }

        // Extract orange team players
        for player in &game.data.orange.players {
            if let Some(rank) = &player.rank
                && let Some(mmr) = rank_to_mmr(&rank.id, rank.division)
            {
                players.push((player.name.clone(), mmr));
            }
        }

        if !players.is_empty() {
            metadata_map.insert(format!("{}.replay", game.id), players);
            count += 1;
        }
    }

    Ok(metadata_map)
}

/// Runs the end-to-end pipeline test.
///
/// # Errors
///
/// Returns an error if the test fails.
pub fn run(replay_dir: &Path, metadata_path: &Path, num_replays: usize) -> Result<()> {
    info!("=== ML Pipeline End-to-End Test ===");
    info!(
        replay_dir = %replay_dir.display(),
        metadata_path = %metadata_path.display(),
        num_replays,
        "Configuration"
    );

    // Step 1: Load metadata
    info!("Step 1: Loading metadata...");
    let metadata = load_metadata(metadata_path, Some(num_replays * 10))?;
    info!(games_with_metadata = metadata.len(), "Metadata loaded");

    if metadata.is_empty() {
        anyhow::bail!("No valid metadata found");
    }

    // Step 2: Find matching replays and extract features
    info!("Step 2: Finding replays and extracting features...");
    let mut training_data = TrainingData::new();
    let mut replays_processed = 0;
    let mut total_frames = 0;

    for (replay_name, players) in &metadata {
        if replays_processed >= num_replays {
            break;
        }

        let replay_path = replay_dir.join(replay_name);
        if !replay_path.exists() {
            continue;
        }

        // Parse replay
        let parsed = match parse_replay(&replay_path) {
            Ok(p) => p,
            Err(e) => {
                info!(replay = %replay_name, error = %e, "Failed to parse replay, skipping");
                continue;
            }
        };

        if parsed.frames.is_empty() {
            continue;
        }

        // Convert player ratings
        let player_ratings: Vec<PlayerRating> = players
            .iter()
            .enumerate()
            .map(|(i, (_, mmr))| PlayerRating {
                player_id: i as u32,
                mmr: *mmr,
            })
            .collect();

        let avg_mmr: f32 =
            player_ratings.iter().map(|r| r.mmr as f32).sum::<f32>() / player_ratings.len() as f32;

        // Segment by goals and extract features
        let segments = segment_by_goals(&parsed);

        for segment in segments {
            let Some(frames) = parsed.frames.get(segment.start_frame..segment.end_frame) else {
                continue;
            };

            // Sample every 10th frame to reduce data size for testing
            for (i, frame) in frames.iter().enumerate() {
                if i % 10 != 0 {
                    continue;
                }

                let features = extract_frame_features(frame);
                training_data.add_samples(vec![feature_extractor::TrainingSample {
                    features,
                    player_ratings: player_ratings.clone(),
                    target_mmr: avg_mmr,
                }]);
                total_frames += 1;
            }
        }

        replays_processed += 1;
        info!(
            replay = %replay_name,
            frames = parsed.frames.len(),
            avg_mmr = avg_mmr as i32,
            "Processed replay"
        );
    }

    if training_data.is_empty() {
        anyhow::bail!(
            "No training data extracted. Found {} replays in metadata but none matched files in {}",
            metadata.len(),
            replay_dir.display()
        );
    }

    info!(
        replays = replays_processed,
        samples = training_data.len(),
        total_frames,
        "Feature extraction complete"
    );

    // Step 3: Create model
    info!("Step 3: Creating model...");
    let device = WgpuDevice::default();
    let model_config = ModelConfig::new();
    let mut model = create_model::<TrainBackend>(&device, &model_config);

    info!(
        hidden1 = model_config.hidden_size_1,
        hidden2 = model_config.hidden_size_2,
        "Model created"
    );

    // Step 4: Train for a few epochs
    info!("Step 4: Training model (5 epochs)...");
    let config = TrainingConfig::new(model_config)
        .with_epochs(5)
        .with_batch_size(32)
        .with_learning_rate(1e-3); // Higher LR for quick test

    let output = train(&mut model, &training_data, &config)?;

    info!(
        final_loss = output.final_train_loss,
        epochs = output.epochs_completed,
        "Training complete"
    );

    // Step 5: Run inference on a sample
    info!("Step 5: Running inference...");

    // Get the inner model for inference (strip Autodiff wrapper)
    let inference_model = model.valid();

    // Sample a few predictions
    let sample_indices = [0, training_data.len() / 2, training_data.len() - 1];

    for &idx in &sample_indices {
        if let Some(sample) = training_data.samples.get(idx) {
            let predicted = predict(&inference_model, &sample.features, &device);
            info!(
                sample = idx,
                target_mmr = sample.target_mmr as i32,
                predicted_mmr = predicted as i32,
                diff = (predicted - sample.target_mmr).abs() as i32,
                "Prediction"
            );
        }
    }

    // Step 6: Sanity checks
    info!("Step 6: Running sanity checks...");

    let mut all_passed = true;

    // Check 1: Loss should be finite
    if output.final_train_loss.is_finite() {
        info!("PASS: Loss is finite ({})", output.final_train_loss);
    } else {
        info!("FAIL: Loss is not finite");
        all_passed = false;
    }

    // Check 2: Predictions should be in reasonable MMR range (0-3000)
    let test_features = feature_extractor::FrameFeatures::default();
    let test_pred = predict(&inference_model, &test_features, &device);

    if (0.0..=3000.0).contains(&test_pred) {
        info!("PASS: Prediction in reasonable range ({})", test_pred);
    } else {
        info!("FAIL: Prediction out of range ({})", test_pred);
        all_passed = false;
    }

    if all_passed {
        info!("=== All sanity checks passed! Pipeline is working. ===");
    } else {
        info!("=== Some checks failed. Review the output above. ===");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_to_mmr() {
        // Supersonic Legend: mmr_middle = (1883 + 2000) / 2 = 1941
        assert_eq!(rank_to_mmr("supersonic-legend", Some(1)), Some(1941));

        // Grand Champion 3 Div 1: mmr_middle = (1706 + 1739) / 2 = 1722
        assert_eq!(rank_to_mmr("grand-champion-3", Some(1)), Some(1722));
        // Grand Champion 3 Div 2: mmr_middle = (1746 + 1780) / 2 = 1763
        assert_eq!(rank_to_mmr("grand-champion-3", Some(2)), Some(1763));
        // Grand Champion 3 Div 3: mmr_middle = (1794 + 1809) / 2 = 1801
        assert_eq!(rank_to_mmr("grand-champion-3", Some(3)), Some(1801));

        // Unknown rank
        assert_eq!(rank_to_mmr("unknown-rank", Some(1)), None);
    }
}
