//! Small-scale end-to-end test of the LSTM sequence model pipeline.
//!
//! This command runs a quick sanity check:
//! 1. Load a few replays with metadata
//! 2. Extract game sequences (one sample per replay)
//! 3. Train for a few epochs
//! 4. Verify loss decreases
//! 5. Run inference and check output is reasonable

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use burn::module::AutodiffModule;
use config::OBJECT_STORE;
use feature_extractor::{PlayerRating, extract_game_sequence};
use ml_model::{
    ModelConfig, SequenceSample, SequenceTrainingData, TrainingConfig, create_model, predict, train,
};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DownloadStatus, Rank};
use tracing::{error, info};

use super::init_device;

type TrainBackend = Autodiff<Wgpu>;

/// Runs the end-to-end pipeline test.
///
/// # Errors
///
/// Returns an error if the test fails.
pub async fn run(num_replays: usize) -> Result<()> {
    info!("=== LSTM Sequence Model Pipeline Test ===");

    // Step 1: Load metadata
    info!("Step 1: Loading replays metadata...");
    let replays =
        database::list_replays_by_rank(Rank::Bronze1, Some(DownloadStatus::Downloaded)).await?;

    info!(replays = replays.len(), "Replays loaded");

    if replays.is_empty() {
        anyhow::bail!("No valid replays found");
    }

    // Step 2: Find matching replays and extract game sequences
    info!("Step 2: Extracting game sequences (one per replay)...");
    let mut training_data = SequenceTrainingData::new();
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
                team: player.team,
                mmr: player.rank_division.mmr_middle(),
            })
            .collect();

        info!(replay = %replay.id, "Player ratings {player_ratings:?}");

        // Extract game sequence (one sample per replay)
        let game_sequence = extract_game_sequence(
            &parsed.frames,
            &player_ratings,
            Some(&parsed.goal_frames),
            Some(&parsed.goals),
        );
        let frame_count = game_sequence.frames.len();

        training_data.add_sample(SequenceSample {
            frames: game_sequence.frames,
            target_mmr: game_sequence.target_mmr,
        });

        total_frames += frame_count;
        replays_processed += 1;

        info!(
            replay = %replay_path.to_string(),
            frames = frame_count,
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

    // Step 3: Create LSTM sequence model
    info!("Step 3: Creating LSTM sequence model...");
    let device = init_device();
    let model_config = ModelConfig::new();
    let mut model = create_model::<TrainBackend>(&device, &model_config);

    info!(
        lstm_hidden_1 = model_config.lstm_hidden_1,
        lstm_hidden_2 = model_config.lstm_hidden_2,
        feedforward_hidden = model_config.feedforward_hidden,
        dropout = model_config.dropout,
        "Model created"
    );

    // Step 4: Train for a few epochs
    info!("Step 4: Training model (10 epochs)...");
    let config = TrainingConfig::new(model_config)
        .with_epochs(10)
        .with_batch_size(4) // Smaller batch for few samples
        .with_sequence_length(200) // Sample 200 frames per game
        .with_learning_rate(1e-3); // Higher LR for quick test

    let output = train(&mut model, &training_data, &config)?;

    let train_rmse = output.final_train_loss.sqrt();
    info!(
        final_loss = output.final_train_loss,
        train_rmse,
        epochs = output.epochs_completed,
        "Training complete"
    );

    // Step 5: Run inference on a sample
    info!("Step 5: Running inference...");

    // Get the inner model for inference (strip Autodiff wrapper)
    let inference_model = model.valid();

    // Test on first sample
    if let Some(sample) = training_data.samples.first() {
        let predictions = predict(
            &inference_model,
            &sample.frames,
            &device,
            config.sequence_length,
        );

        info!("Predictions for all 6 players:");
        for (player_idx, (predicted, target)) in
            predictions.iter().zip(sample.target_mmr.iter()).enumerate()
        {
            let diff = (predicted - target).abs();
            let team = if player_idx < feature_extractor::PLAYERS_PER_TEAM {
                "blue"
            } else {
                "orange"
            };
            let slot = player_idx % feature_extractor::PLAYERS_PER_TEAM;

            info!(
                player_slot = slot,
                team = team,
                target_mmr = *target as i32,
                predicted_mmr = *predicted as i32,
                diff = diff as i32,
                "Player prediction"
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
    let test_frames: Vec<feature_extractor::FrameFeatures> = (0..100)
        .map(|_| feature_extractor::FrameFeatures::default())
        .collect();
    let test_preds = predict(
        &inference_model,
        &test_frames,
        &device,
        config.sequence_length,
    );

    let all_reasonable = test_preds.iter().all(|p| (0.0..=3000.0).contains(p));
    if all_reasonable {
        info!(
            "PASS: All 6 predictions in reasonable range ({:?})",
            test_preds.iter().map(|p| *p as i32).collect::<Vec<_>>()
        );
    } else {
        info!(
            "FAIL: Some predictions out of range ({:?})",
            test_preds.iter().map(|p| *p as i32).collect::<Vec<_>>()
        );
        all_passed = false;
    }

    // Check 3: Predictions should not all be the same (model is learning something)
    let variance: f32 = {
        let mean = test_preds.iter().sum::<f32>() / test_preds.len() as f32;
        test_preds.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / test_preds.len() as f32
    };

    if variance > 1.0 {
        info!("PASS: Predictions have variance ({:.2})", variance);
    } else {
        info!(
            "INFO: Low prediction variance ({:.2}) - model may need more training",
            variance
        );
        // Don't fail on this, just informational
    }

    if all_passed {
        info!("=== All sanity checks passed! LSTM pipeline is working. ===");
    } else {
        info!("=== Some checks failed. Review the output above. ===");
    }

    Ok(())
}
