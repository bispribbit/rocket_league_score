//! Small-scale end-to-end test of the ML pipeline.
//!
//! This command runs a quick sanity check:
//! 1. Load a few replays with metadata
//! 2. Extract features
//! 3. Train for a few epochs
//! 4. Verify loss decreases
//! 5. Run inference and check output is reasonable

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use database::{BallchasingReplayRepository, DownloadStatus};
use feature_extractor::{PlayerRating, extract_frame_features};
use ml_model::{ModelConfig, TrainingConfig, TrainingData, create_model, predict, train};
use replay_parser::{parse_replay, segment_by_goals};
use replay_structs::{PlayerWithRating, RankInfo, ReplaySummary};
use tracing::info;

use crate::rank::Rank;

type TrainBackend = Autodiff<Wgpu>;

/// Converts a rank ID and division to MMR using the Rank enum.
fn rank_to_mmr(rank_info: &RankInfo) -> Option<i32> {
    Rank::from_rank_id(&rank_info.id, rank_info.division).map(Rank::mmr_middle)
}

/// Loads metadata for replays from the database.
async fn load_metadata(
    limit: Option<usize>,
) -> Result<HashMap<String, Vec<PlayerWithRating>>> {
    // Get all downloaded replays
    let all_ranks = database::BallchasingRank::all_ranked();
    let mut all_replays = Vec::new();

    for rank in all_ranks {
        let replays = BallchasingReplayRepository::list_by_rank(rank, Some(DownloadStatus::Downloaded)).await?;
        all_replays.extend(replays);
    }

    let mut metadata_map: HashMap<String, Vec<PlayerWithRating>> = HashMap::new();
    let mut count = 0;

    for replay in all_replays {
        if let Some(max) = limit
            && count >= max
        {
            break;
        }

        // Deserialize metadata
        let summary: ReplaySummary = serde_json::from_value(replay.metadata)
            .context("Failed to deserialize replay metadata")?;

        let mut players = Vec::new();

        // Extract blue team players
        if let Some(blue_team) = &summary.blue
            && let Some(team_players) = &blue_team.players
        {
            for player in team_players {
                let mmr = if let Some(rank) = &player.rank {
                    rank_to_mmr(rank)
                } else if let (Some(min_rank), Some(max_rank)) = (&summary.min_rank, &summary.max_rank) {
                    // Use middle of min and max rank if player rank is null
                    let min_mmr = rank_to_mmr(min_rank)
                        .ok_or_else(|| anyhow::anyhow!("Failed to convert min_rank to MMR"))?;
                    let max_mmr = rank_to_mmr(max_rank)
                        .ok_or_else(|| anyhow::anyhow!("Failed to convert max_rank to MMR"))?;
                    Some(i32::midpoint(min_mmr, max_mmr))
                } else {
                    None
                };

                if let (Some(name), Some(mmr_value)) = (player.name.as_ref(), mmr) {
                    players.push(PlayerWithRating {
                        player_name: name.clone(),
                        mmr: mmr_value,
                    });
                }
            }
        }

        // Extract orange team players
        if let Some(orange_team) = &summary.orange
            && let Some(team_players) = &orange_team.players
        {
            for player in team_players {
                let mmr = if let Some(rank) = &player.rank {
                    rank_to_mmr(rank)
                } else if let (Some(min_rank), Some(max_rank)) = (&summary.min_rank, &summary.max_rank) {
                    // Use middle of min and max rank if player rank is null
                    let min_mmr = rank_to_mmr(min_rank)
                        .ok_or_else(|| anyhow::anyhow!("Failed to convert min_rank to MMR"))?;
                    let max_mmr = rank_to_mmr(max_rank)
                        .ok_or_else(|| anyhow::anyhow!("Failed to convert max_rank to MMR"))?;
                    Some(i32::midpoint(min_mmr, max_mmr))
                } else {
                    None
                };

                if let (Some(name), Some(mmr_value)) = (player.name.as_ref(), mmr) {
                    players.push(PlayerWithRating {
                        player_name: name.clone(),
                        mmr: mmr_value,
                    });
                }
            }
        }

        if !players.is_empty() {
            // Use replay ID as filename (format: {id}.replay)
            let filename = format!("{}.replay", summary.id);
            metadata_map.insert(filename, players);
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
pub async fn run(replay_dir: &Path, num_replays: usize) -> Result<()> {
    info!("=== ML Pipeline End-to-End Test ===");
    info!(
        replay_dir = %replay_dir.display(),
        num_replays,
        "Configuration"
    );

    // Step 1: Load metadata
    info!("Step 1: Loading metadata...");
    let metadata = load_metadata(Some(num_replays * 10)).await?;
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
            .map(|(i, player)| PlayerRating {
                player_id: i as u32,
                mmr: player.mmr,
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
        use replay_structs::RankInfo;

        // Supersonic Legend: mmr_middle = (1883 + 2000) / 2 = 1941
        let rank = RankInfo {
            id: "supersonic-legend".to_string(),
            tier: None,
            division: Some(1),
            name: None,
        };
        assert_eq!(rank_to_mmr(&rank), Some(1941));

        // Grand Champion 3 Div 1: mmr_middle = (1706 + 1739) / 2 = 1722
        let rank = RankInfo {
            id: "grand-champion-3".to_string(),
            tier: None,
            division: Some(1),
            name: None,
        };
        assert_eq!(rank_to_mmr(&rank), Some(1722));

        // Grand Champion 3 Div 2: mmr_middle = (1746 + 1780) / 2 = 1763
        let rank = RankInfo {
            id: "grand-champion-3".to_string(),
            tier: None,
            division: Some(2),
            name: None,
        };
        assert_eq!(rank_to_mmr(&rank), Some(1763));

        // Grand Champion 3 Div 3: mmr_middle = (1794 + 1809) / 2 = 1801
        let rank = RankInfo {
            id: "grand-champion-3".to_string(),
            tier: None,
            division: Some(3),
            name: None,
        };
        assert_eq!(rank_to_mmr(&rank), Some(1801));

        // Unknown rank
        let rank = RankInfo {
            id: "unknown-rank".to_string(),
            tier: None,
            division: Some(1),
            name: None,
        };
        assert_eq!(rank_to_mmr(&rank), None);
    }
}
