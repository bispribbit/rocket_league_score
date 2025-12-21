//! Train command - trains the ML model on ingested replays.

use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use database::{
    CreateModel, GameMode, ModelRepository, ReplayPlayerRepository, ReplayRepository,
};
use feature_extractor::{extract_segment_samples, PlayerRating};
use ml_model::{create_model, train, ModelConfig, TrainingConfig, TrainingData};
use replay_parser::{parse_replay, segment_by_goals};
use sqlx::PgPool;
use tracing::info;

type Backend = Wgpu;

/// Runs the train command.
///
/// # Errors
///
/// Returns an error if training fails.
pub async fn run(
    pool: &PgPool,
    model_name: &str,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) -> Result<()> {
    info!(model_name, "Starting training");

    let config = TrainingConfig {
        learning_rate,
        epochs,
        batch_size,
        model: ModelConfig::default(),
    };

    // Load training data from database
    info!("Loading training data...");
    let training_data = load_training_data(pool).await?;

    if training_data.is_empty() {
        anyhow::bail!("No training data found. Please ingest replays first.");
    }

    info!(samples = training_data.len(), "Loaded training samples");

    // Create model
    let device = WgpuDevice::default();
    let mut model = create_model::<Backend>(&device, &config.model);

    // Train model
    info!(epochs, "Training model");
    train(&mut model, &training_data, &config)?;

    // Save model checkpoint
    let next_version = ModelRepository::next_version(pool, model_name).await?;
    let checkpoint_path = format!("models/{model_name}_{next_version}.bin");

    // TODO: Actually save the model weights
    // ml_model::save_checkpoint(&model, &checkpoint_path)?;

    // Create model record in database
    let training_config_json = serde_json::json!({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size_1": config.model.hidden_size_1,
        "hidden_size_2": config.model.hidden_size_2,
    });

    ModelRepository::create(
        pool,
        CreateModel {
            name: model_name.to_string(),
            version: next_version,
            checkpoint_path: checkpoint_path.clone(),
            training_config: Some(training_config_json),
            metrics: None, // TODO: Add training metrics
        },
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
async fn load_training_data(pool: &PgPool) -> Result<TrainingData> {
    let mut data = TrainingData::new();

    // Get all 3v3 replays (our primary training data)
    let replays = ReplayRepository::list_by_game_mode(pool, GameMode::Soccar3v3).await?;

    for replay in replays {
        // Get player ratings for this replay
        let db_players = ReplayPlayerRepository::list_by_replay(pool, replay.id).await?;

        if db_players.is_empty() {
            // Skip replays without player ratings
            continue;
        }

        // Convert to feature extractor format
        let player_ratings: Vec<PlayerRating> = db_players
            .iter()
            .enumerate()
            .map(|(i, p)| PlayerRating {
                player_id: i as u32,
                mmr: p.skill_rating,
            })
            .collect();

        // Parse the replay file
        let Ok(parsed) = parse_replay(std::path::Path::new(&replay.file_path)) else {
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

