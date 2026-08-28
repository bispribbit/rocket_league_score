//! Example: Run full training pipeline with database integration.
//!
//! This example runs the production training pipeline which:
//! - Assigns train/test splits to replays in the database
//! - Trains with checkpoint saving every 5 epochs
//! - Supports resumption from checkpoints
//! - Evaluates on held-out test set
//!
//! Usage:
//!   cargo run --example pipeline
//!
//! Environment Variables:
//!   `DATABASE_URL`     - `PostgreSQL` connection string (required)
//!   `MODEL_NAME`       - Model name for saving (default: `lstm_v3`)
//!   `TRAIN_RATIO`      - Training set ratio (default: 0.9)
//!   EPOCHS           - Number of epochs (default: 100)
//!   `BATCH_SIZE`       - Batch size (default: 144, the value `lstm_v20` was trained at;
//!                       may be reduced automatically to keep the fused LSTM projection
//!                       inside its VRAM budget)
//!   `LEARNING_RATE`    - Learning rate (default: 0.03 — matches the `overfit_wgpu`
//!                       harness; combined with the warmup + 0.10 floor in
//!                       `training::cosine_lr`, this is the schedule validated end-to-end
//!                       on T1/T2/T3.)
//!   RESUME           - Resume from checkpoint (default: false)
//!   `MAX_REPLAYS`      - Limit number of replays (default: None, uses all)
//!   `SELF_ONLY_FEATURES` - Train on the self-only 27-feature view, zeroing the other
//!                       five cars (default: false). This is the step-3 go/no-go ablation
//!                       in `docs/smurf-detection-handoff.md`. Score it on within-lobby
//!                       concordance / top-1, **not** RMSE: the lobby shortcut it removes
//!                       is worth 98.4 % of the RMSE objective, so RMSE getting worse is
//!                       the expected outcome and says nothing about the question.
//!
//! Step-3 ablation pair (fixed dev subset, same seed and subset for both):
//!   `MODEL_NAME=lstm_v22_full DEV_SUBSET_REPLAYS=3000 EPOCHS=60 cargo run --release --example pipeline`
//!   `MODEL_NAME=lstm_v22_self SELF_ONLY_FEATURES=true DEV_SUBSET_REPLAYS=3000 EPOCHS=60 cargo run --release --example pipeline`
//!
//! Self-only re-baseline at full data (step-3 follow-up, `lstm_v23_self`):
//!   `MODEL_NAME=lstm_v23_self SELF_ONLY_FEATURES=true EPOCHS=100 cargo run --release --example pipeline`
//!
//! **Score a self-only run on `checkpoint_best_ordinal`, not `checkpoint_best`.** The
//! latter is selected on per-segment validation loss, which is ~98.4 % lobby-level
//! accuracy — precisely the objective this feature view gives up. `_best_ordinal` is
//! selected on within-lobby concordance instead.
//!
//! Example:
//!   `DATABASE_URL=postgres`://... EPOCHS=50 cargo run --example pipeline

use std::fs::File;

use anyhow::Result;
use chrono::Local;
use database::initialize_pool;
use rocket_league_score::commands;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::{Layer, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;

fn get_env_or_default<T: core::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create log file with current date and time
    let now = Local::now();
    let filename = format!("models/{}.txt", now.format("%Y%m%d_%H%M%S"));
    let log_file = File::create(&filename)?;

    // Initialize tracing with both stdout and file output
    let env_filter = EnvFilter::new("info");

    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(env_filter.clone());

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(log_file)
        .with_ansi(false)
        .with_filter(env_filter);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .init();

    // Read configuration from environment
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "lstm_v15".to_string());
    let train_ratio: f64 = get_env_or_default("TRAIN_RATIO", 0.9);
    let epochs: usize = get_env_or_default("EPOCHS", 500);
    let batch_size: usize = get_env_or_default("BATCH_SIZE", 144);
    let learning_rate: f64 = get_env_or_default("LEARNING_RATE", 0.03);
    let resume: bool = get_env_or_default("RESUME", true);
    let max_replays: Option<usize> = std::env::var("MAX_REPLAYS")
        .ok()
        .and_then(|s| s.parse().ok());
    let self_only_features: bool = get_env_or_default("SELF_ONLY_FEATURES", false);
    let dev_subset_replays: Option<usize> = std::env::var("DEV_SUBSET_REPLAYS")
        .ok()
        .and_then(|s| s.parse().ok());

    info!("=== Full Training Pipeline ===");
    info!("Model name:    {model_name}");
    info!(
        "Train ratio:   {train_ratio:.1}%",
        train_ratio = train_ratio * 100.0
    );
    info!("Epochs:        {epochs}");
    info!("Batch size:    {batch_size}");
    info!("Learning rate: {learning_rate}");
    info!("Resume:        {resume}");
    info!(
        "Max replays:   {}",
        max_replays.map_or_else(|| "all".to_string(), |n| n.to_string())
    );
    info!(
        "Feature view:  {}",
        if self_only_features {
            "self-only-27 (ABLATION — context features zeroed)"
        } else {
            "full-106"
        }
    );

    // Initialize database
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    // Run the full training pipeline. Built as a config rather than through
    // `full_pipeline::run` so the ablation knobs are reachable from here.
    let config = commands::full_pipeline::FullTrainConfig {
        model_name,
        train_ratio,
        epochs,
        batch_size,
        learning_rate,
        resume,
        checkpoint_every_n_epochs: 5,
        max_replays,
        dev_subset_replays,
        self_only_features,
    };
    commands::full_pipeline::run_with_config(&config).await?;

    Ok(())
}
