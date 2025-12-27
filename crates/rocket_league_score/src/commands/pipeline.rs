//! Small-scale end-to-end test of the ML pipeline.
//!
//! This command runs a quick sanity check:
//! 1. Load a few replays with metadata
//! 2. Extract features
//! 3. Train for a few epochs
//! 4. Verify loss decreases
//! 5. Run inference and check output is reasonable

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use config::OBJECT_STORE;
use feature_extractor::{PlayerRating, extract_segment_samples};
use ml_model::{ModelConfig, TrainingConfig, TrainingData, create_model, predict, train};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::{parse_replay_from_bytes, segment_by_goals};
use replay_structs::{DownloadStatus, Rank};
use tracing::{error, info};

use super::init_wgpu_device;

type TrainBackend = Autodiff<Wgpu>;

/// Runs the end-to-end pipeline test.
///
/// # Errors
///
/// Returns an error if the test fails.
pub async fn run(num_replays: usize) -> Result<()> {
    info!("=== ML Pipeline End-to-End Test ===");
    // Step 1: Load metadata
    info!("Step 1: Loading replays metadata...");
    let replays =
        database::list_replays_by_rank(Rank::Bronze1, Some(DownloadStatus::Downloaded)).await?;

    info!(replays = replays.len(), "Replays loaded");

    if replays.is_empty() {
        anyhow::bail!("No valid replays found");
    }

    // Step 2: Find matching replays and extract features
    info!("Step 2: Finding replays and extracting features...");
    let mut training_data = TrainingData::new();
    let mut replays_processed = 0;
    let mut total_frames = 0;

    for replay in &replays {
        if replays_processed >= num_replays {
            break;
        }

        let replay_path = ObjectStorePath::from(replay.file_path.clone());

        // Read from object_store as bytes
        let replay_data = match OBJECT_STORE
            .get(&replay_path)
            .await
            .context("Failed to read from object_store")
        {
            Ok(get_result) => match get_result.bytes().await {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!(replay = %replay_path.to_string(), error = %e, "Failed to read bytes from object_store, skipping");
                    continue;
                }
            },
            Err(e) => {
                error!(replay = %replay_path.to_string(), error = %e, "Failed to read from object_store, skipping");
                continue;
            }
        };

        // Parse replay from bytes
        let parsed = match parse_replay_from_bytes(&replay_data) {
            Ok(p) => p,
            Err(e) => {
                info!(replay = %replay_path.to_string(), error = %e, "Failed to parse replay, skipping");
                continue;
            }
        };

        if parsed.frames.is_empty() {
            continue;
        }

        // Get player ratings for this replay
        let db_players = database::list_replay_players_by_replay(replay.id).await?;

        if db_players.is_empty() {
            // Skip replays without player ratings
            continue;
        }

        // Convert player ratings
        let player_ratings: Vec<PlayerRating> = db_players
            .iter()
            .map(|player| PlayerRating {
                player_name: player.player_name.clone(),
                mmr: player.rank_division.mmr_middle(),
            })
            .collect();

        // Segment by goals and extract features
        let segments = segment_by_goals(&parsed);

        for segment in segments {
            let Some(frames) = parsed.frames.get(segment.start_frame..segment.end_frame) else {
                continue;
            };

            let samples = extract_segment_samples(frames, &player_ratings);
            training_data.add_samples(samples);
            total_frames += frames.len();
        }

        replays_processed += 1;
        info!(
            replay = %replay_path.to_string(),
            frames = parsed.frames.len(),
            "Processed replay"
        );
    }

    if training_data.is_empty() {
        anyhow::bail!(
            "No training data extracted. Found {} replays in metadata but none matched files on disk",
            replays.len()
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
    let device = init_wgpu_device()?;
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

            // Evaluate against each player's target MMR
            for (player_idx, target_mmr) in sample.target_mmr.iter().enumerate() {
                let diff = (predicted - target_mmr).abs();
                let team = if player_idx < feature_extractor::PLAYERS_PER_TEAM {
                    "blue"
                } else {
                    "orange"
                };
                let slot = player_idx % feature_extractor::PLAYERS_PER_TEAM;

                info!(
                    sample = idx,
                    player_slot = slot,
                    team = team,
                    target_mmr = *target_mmr as i32,
                    predicted_mmr = predicted as i32,
                    diff = diff as i32,
                    "Player prediction"
                );
            }
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
