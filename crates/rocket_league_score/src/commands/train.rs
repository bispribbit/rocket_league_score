//! Train command - trains the LSTM sequence model on ingested replays.
//!
//! This command loads replays from the database, extracts game sequences,
//! and trains the sequence model to predict player MMR from temporal patterns.

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use config::OBJECT_STORE;
use feature_extractor::{PlayerRating, extract_game_sequence};
use ml_model::{
    ModelConfig, SequenceSample, SequenceTrainingData, TrainingConfig, create_model,
    save_checkpoint, train,
};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use tracing::{error, info, warn};

use super::init_device;

// Training requires Autodiff wrapper for automatic differentiation
type TrainBackend = Autodiff<Wgpu>;

/// Runs the train command.
///
/// # Errors
///
/// Returns an error if training fails.
pub async fn run(
    model_name: &str,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> Result<()> {
    info!(model_name, "Starting LSTM sequence model training");

    let model_config = ModelConfig::new();
    let config = TrainingConfig::new(model_config.clone())
        .with_learning_rate(learning_rate)
        .with_epochs(epochs)
        .with_batch_size(batch_size);

    // Load training data from database
    info!("Loading training data (one sample per replay)...");
    let training_data = load_training_data().await?;

    if training_data.is_empty() {
        anyhow::bail!("No training data found. Please ingest replays first.");
    }

    info!(
        samples = training_data.len(),
        sequence_length = config.sequence_length,
        "Loaded game sequence samples"
    );

    // Create model with Autodiff backend for training
    let device = init_device();
    let mut model = create_model::<TrainBackend>(&device, &model_config);

    info!(
        lstm_hidden_1 = model_config.lstm_hidden_1,
        lstm_hidden_2 = model_config.lstm_hidden_2,
        feedforward_hidden = model_config.feedforward_hidden,
        dropout = model_config.dropout,
        "Model architecture"
    );

    // Train model
    info!(epochs, batch_size, learning_rate, "Training model");
    let output = train(&mut model, &training_data, &config)?;

    // Save model checkpoint
    let next_version = database::get_next_model_version(model_name).await?;
    let checkpoint_path = format!("models/{model_name}_{next_version}");

    save_checkpoint(&model, &checkpoint_path, &config)?;

    let train_rmse = output.final_train_loss.sqrt();
    let valid_rmse = output.final_valid_loss.map(f32::sqrt);
    info!(
        final_train_loss = output.final_train_loss,
        train_rmse,
        final_valid_loss = output.final_valid_loss,
        valid_rmse,
        epochs_completed = output.epochs_completed,
        "Training completed"
    );

    // Create model record in database
    let training_config_json = serde_json::json!({
        "model_type": "lstm_sequence",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "sequence_length": config.sequence_length,
        "lstm_hidden_1": model_config.lstm_hidden_1,
        "lstm_hidden_2": model_config.lstm_hidden_2,
        "feedforward_hidden": model_config.feedforward_hidden,
        "dropout": model_config.dropout,
    });

    let training_metrics_json = serde_json::json!({
        "final_train_loss": output.final_train_loss,
        "final_train_rmse": train_rmse,
        "final_valid_loss": output.final_valid_loss,
        "final_valid_rmse": valid_rmse,
        "epochs_completed": output.epochs_completed,
    });

    database::insert_model(
        model_name,
        next_version,
        &checkpoint_path,
        Some(training_config_json),
        Some(training_metrics_json),
    )
    .await?;

    info!(
        model_name,
        version = next_version,
        checkpoint_path,
        "Model saved"
    );

    Ok(())
}

/// Loads training data from the database.
///
/// Each replay becomes one training sample - the model learns to predict
/// player MMR from the entire sequence of gameplay.
async fn load_training_data() -> Result<SequenceTrainingData> {
    let mut data = SequenceTrainingData::new();

    // Get all ranked replays
    let mut all_replays = Vec::new();
    for rank in replay_structs::Rank::all_ranked() {
        let rank_replays =
            database::list_replays_by_rank(rank, Some(replay_structs::DownloadStatus::Downloaded))
                .await?;
        all_replays.extend(rank_replays);
    }

    info!(
        total_replays = all_replays.len(),
        "Found replays to process"
    );

    let mut skipped_no_players = 0;
    let mut skipped_parse_error = 0;
    let mut skipped_read_error = 0;

    for replay in all_replays {
        // Get player ratings for this replay
        let db_players = database::list_replay_players_by_replay(replay.id).await?;

        if db_players.is_empty() {
            skipped_no_players += 1;
            continue;
        }

        // Convert to feature extractor format
        let player_ratings: Vec<PlayerRating> = db_players
            .iter()
            .map(|p| PlayerRating {
                player_name: p.player_name.clone(),
                team: p.team,
                mmr: p.rank_division.mmr_middle(),
            })
            .collect();

        // Read from object_store as bytes
        let object_path = ObjectStorePath::from(replay.file_path.clone());
        let replay_data = match OBJECT_STORE
            .get(&object_path)
            .await
            .context("Failed to read from object_store")
        {
            Ok(get_result) => match get_result.bytes().await {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!(replay = %object_path.to_string(), error = %e, "Failed to read bytes");
                    skipped_read_error += 1;
                    continue;
                }
            },
            Err(e) => {
                error!(replay = %object_path.to_string(), error = %e, "Failed to get from object_store");
                skipped_read_error += 1;
                continue;
            }
        };

        // Parse the replay from bytes
        let parsed = match parse_replay_from_bytes(&replay_data) {
            Ok(p) => p,
            Err(e) => {
                warn!(replay = %object_path.to_string(), error = %e, "Failed to parse replay");
                skipped_parse_error += 1;
                continue;
            }
        };

        if parsed.frames.is_empty() {
            warn!(replay = %object_path.to_string(), "Replay has no frames");
            skipped_parse_error += 1;
            continue;
        }

        // Extract game sequence (one sample per replay)
        let game_sequence = extract_game_sequence(&parsed.frames, &player_ratings);

        // Convert to SequenceSample for ml_model
        data.add_sample(SequenceSample {
            frames: game_sequence.frames,
            target_mmr: game_sequence.target_mmr,
        });
    }

    if skipped_no_players > 0 || skipped_parse_error > 0 || skipped_read_error > 0 {
        info!(
            skipped_no_players,
            skipped_parse_error, skipped_read_error, "Skipped some replays"
        );
    }

    Ok(data)
}
