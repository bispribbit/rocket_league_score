//! Predict command - runs inference on a replay file using the LSTM sequence model.

use std::path::Path;

use anyhow::{Context, Result};
use burn::backend::Wgpu;
use feature_extractor::extract_frame_features;
use ml_model::{SequenceModel, load_checkpoint, predict, predict_evolving};
use replay_parser::parse_replay;
use tracing::{info, warn};

use super::init_device;

type Backend = Wgpu;

/// Default sequence length for inference (should match training config).
const DEFAULT_SEQUENCE_LENGTH: usize = 300;

/// Runs the predict command.
///
/// # Errors
///
/// Returns an error if prediction fails.
pub async fn run(replay_path: &Path, model_name: &str, version: Option<i32>) -> Result<()> {
    info!(replay = %replay_path.display(), "Predicting player MMR with LSTM model");

    // Load the model
    let model_record = if let Some(v) = version {
        database::find_model_by_name_version(model_name, v).await?
    } else {
        database::find_latest_model(model_name).await?
    };

    let model_record = model_record.context(format!(
        "Model '{model_name}' not found. Train a model first."
    ))?;

    // Get sequence length from training config if available
    let sequence_length = model_record
        .training_config
        .as_ref()
        .and_then(|c| c.get("sequence_length"))
        .and_then(serde_json::Value::as_u64)
        .map_or(DEFAULT_SEQUENCE_LENGTH, |v| v as usize);

    info!(
        model = %model_record.name,
        version = model_record.version,
        sequence_length,
        "Using model"
    );

    // Load model weights
    let device = init_device();
    let model: SequenceModel<Backend> = load_checkpoint(&model_record.checkpoint_path, &device)?;

    // Parse the replay
    let parsed = parse_replay(replay_path)?;

    if parsed.frames.is_empty() {
        warn!("No frames found in replay");
        return Ok(());
    }

    info!(frames = parsed.frames.len(), "Analyzing gameplay sequence");

    // Extract features from all frames
    let frame_features: Vec<_> = parsed.frames.iter().map(extract_frame_features).collect();

    // Get final prediction for the entire game
    let final_predictions = predict(&model, &frame_features, &device, sequence_length);

    info!("=== Final Player MMR Predictions ===");
    info!("Blue Team:");
    for (i, mmr) in final_predictions.iter().take(3).enumerate() {
        let skill_level = interpret_mmr(*mmr);
        info!(
            "  Player {} - MMR: {:.0} ({})",
            i + 1,
            mmr,
            skill_level
        );
    }
    info!("Orange Team:");
    for (i, mmr) in final_predictions.iter().skip(3).enumerate() {
        let skill_level = interpret_mmr(*mmr);
        info!(
            "  Player {} - MMR: {:.0} ({})",
            i + 1,
            mmr,
            skill_level
        );
    }

    // Show how predictions evolved over the game
    info!("\n=== MMR Evolution During Game ===");

    // Use 10% of frames as step size for evolution
    let step_size = (frame_features.len() / 10).max(1);
    let evolution = predict_evolving(&model, &frame_features, &device, sequence_length, step_size);

    for (idx, pred) in evolution.iter().enumerate() {
        let progress_pct = (idx as f32 / evolution.len() as f32) * 100.0;
        let blue_avg = pred.player_mmr.iter().take(3).sum::<f32>() / 3.0;
        let orange_avg = pred.player_mmr.iter().skip(3).sum::<f32>() / 3.0;

        info!(
            "  {:5.1}% | Time: {:6.1}s | Blue avg: {:6.0} | Orange avg: {:6.0}",
            progress_pct, pred.timestamp, blue_avg, orange_avg
        );
    }

    // Summary
    let blue_avg_final = final_predictions.iter().take(3).sum::<f32>() / 3.0;
    let orange_avg_final = final_predictions.iter().skip(3).sum::<f32>() / 3.0;

    info!("\n=== Game Summary ===");
    info!("Blue Team Average MMR: {:.0} ({})", blue_avg_final, interpret_mmr(blue_avg_final));
    info!("Orange Team Average MMR: {:.0} ({})", orange_avg_final, interpret_mmr(orange_avg_final));

    Ok(())
}

/// Interprets an MMR value as a rank name.
const fn interpret_mmr(mmr: f32) -> &'static str {
    match mmr as i32 {
        ..=174 => "Bronze I",
        175..=234 => "Bronze II",
        235..=294 => "Bronze III",
        295..=354 => "Silver I",
        355..=414 => "Silver II",
        415..=474 => "Silver III",
        475..=534 => "Gold I",
        535..=594 => "Gold II",
        595..=654 => "Gold III",
        655..=714 => "Platinum I",
        715..=774 => "Platinum II",
        775..=834 => "Platinum III",
        835..=894 => "Diamond I",
        895..=954 => "Diamond II",
        955..=1074 => "Diamond III",
        1075..=1174 => "Champion I",
        1175..=1274 => "Champion II",
        1275..=1434 => "Champion III",
        1435..=1574 => "Grand Champion I",
        1575..=1714 => "Grand Champion II",
        1715..=1882 => "Grand Champion III",
        _ => "Supersonic Legend",
    }
}
