//! Predict command - runs inference on a replay file.

use std::path::Path;

use anyhow::{Context, Result};
use burn::backend::Wgpu;
use database;
use feature_extractor::extract_frame_features;
use ml_model::{ImpactModel, load_checkpoint, predict};
use replay_parser::parse_replay;
use tracing::{info, warn};

use super::init_wgpu_device;

type Backend = Wgpu;

/// Runs the predict command.
///
/// # Errors
///
/// Returns an error if prediction fails.
pub async fn run(replay_path: &Path, model_name: &str, version: Option<i32>) -> Result<()> {
    info!(replay = %replay_path.display(), "Predicting impact scores");

    // Load the model
    let model_record = if let Some(v) = version {
        database::find_model_by_name_version(model_name, v).await?
    } else {
        database::find_latest_model(model_name).await?
    };

    let model_record = model_record.context(format!(
        "Model '{model_name}' not found. Train a model first."
    ))?;

    info!(
        model = %model_record.name,
        version = model_record.version,
        "Using model"
    );

    // Load model weights
    let device = init_wgpu_device()?;
    let model: ImpactModel<Backend> = load_checkpoint(&model_record.checkpoint_path, &device)?;

    // Parse the replay
    let parsed = parse_replay(replay_path)?;

    if parsed.frames.is_empty() {
        warn!("No frames found in replay");
        return Ok(());
    }

    info!(frames = parsed.frames.len(), "Analyzing frames");

    // Run prediction on each frame and aggregate
    let mut scores: Vec<f32> = Vec::new();

    for frame in &parsed.frames {
        let features = extract_frame_features(frame);
        let score = predict(&model, &features, &device);
        scores.push(score);
    }

    // Calculate statistics
    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;
    let min_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    info!("=== Impact Score Results ===");
    info!(avg_score, "Average Score");
    info!(min_score, "Min Score");
    info!(max_score, "Max Score");

    // Interpret the score
    let skill_level = interpret_score(avg_score);
    info!(skill_level, "Estimated Skill Level");

    Ok(())
}

/// Interprets an impact score as a skill level.
const fn interpret_score(score: f32) -> &'static str {
    // These thresholds are based on Rocket League's MMR system
    // and would be refined through actual training data
    match score as i32 {
        0..=299 => "Bronze",
        300..=499 => "Silver",
        500..=699 => "Gold",
        700..=899 => "Platinum",
        900..=1099 => "Diamond",
        1100..=1299 => "Champion",
        1300..=1499 => "Grand Champion",
        _ => "Supersonic Legend",
    }
}
