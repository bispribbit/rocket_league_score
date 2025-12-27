//! Train command - trains the ML model on ingested replays.

use anyhow::{Context, Result};
use burn::backend::{Autodiff, Wgpu};
use config::OBJECT_STORE;
use feature_extractor::{PlayerRating, extract_segment_samples};
use ml_model::{ModelConfig, TrainingConfig, TrainingData, create_model, save_checkpoint, train};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::{parse_replay_from_bytes, segment_by_goals};
use tracing::{error, info};

use super::init_wgpu_device;

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
    info!(model_name, "Starting training");

    let model_config = ModelConfig::new();
    let config = TrainingConfig::new(model_config.clone())
        .with_learning_rate(learning_rate)
        .with_epochs(epochs)
        .with_batch_size(batch_size);

    // Load training data from database
    info!("Loading training data...");
    let training_data = load_training_data().await?;

    if training_data.is_empty() {
        anyhow::bail!("No training data found. Please ingest replays first.");
    }

    info!(samples = training_data.len(), "Loaded training samples");

    // Create model with Autodiff backend for training
    let device = init_wgpu_device()?;
    let mut model = create_model::<TrainBackend>(&device, &model_config);

    // Train model
    info!(epochs, "Training model");
    let output = train(&mut model, &training_data, &config)?;

    // Save model checkpoint
    let next_version = database::get_next_model_version(model_name).await?;
    let checkpoint_path = format!("models/{model_name}_{next_version}");

    save_checkpoint(&model, &checkpoint_path, &config)?;
    info!(final_loss = output.final_train_loss, "Training completed");

    // Create model record in database
    let training_config_json = serde_json::json!({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size_1": model_config.hidden_size_1,
        "hidden_size_2": model_config.hidden_size_2,
    });

    database::insert_model(
        model_name,
        next_version,
        &checkpoint_path,
        Some(training_config_json),
        None, // TODO: Add training metrics
    )
    .await?;

    info!(
        model_name,
        version = next_version,
        checkpoint_path,
        "Training complete"
    );

    Ok(())
}

/// Loads training data from the database.
async fn load_training_data() -> Result<TrainingData> {
    let mut data = TrainingData::new();

    // Get all 3v3 replays (our primary training data)
    // Note: Since we unified to ballchasing replays, we get all downloaded replays
    // The metadata should contain game mode information if needed
    let mut all_replays = Vec::new();
    for rank in replay_structs::Rank::all_ranked() {
        let rank_replays =
            database::list_replays_by_rank(rank, Some(replay_structs::DownloadStatus::Downloaded))
                .await?;
        all_replays.extend(rank_replays);
    }
    let replays = all_replays;

    for replay in replays {
        // Get player ratings for this replay
        let db_players = database::list_replay_players_by_replay(replay.id).await?;

        if db_players.is_empty() {
            // Skip replays without player ratings
            continue;
        }

        // Convert to feature extractor format
        let player_ratings: Vec<PlayerRating> = db_players
            .iter()
            .map(|p| PlayerRating {
                player_name: p.player_name.clone(),
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
                    error!(replay = %object_path.to_string(), error = %e, "Failed to read bytes from object_store, skipping");
                    continue;
                }
            },
            Err(e) => {
                error!(replay = %object_path.to_string(), error = %e, "Failed to read from object_store, skipping");
                continue;
            }
        };

        // Parse the replay from bytes
        let Ok(parsed) = parse_replay_from_bytes(&replay_data) else {
            continue;
        };

        // Segment by goals and extract features
        let segments = segment_by_goals(&parsed);

        for segment in segments {
            let Some(frames) = parsed.frames.get(segment.start_frame..segment.end_frame) else {
                continue;
            };

            let samples = extract_segment_samples(frames, &player_ratings);
            data.add_samples(samples);
        }
    }

    Ok(data)
}
