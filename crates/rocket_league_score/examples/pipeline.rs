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
//!   `MODEL_NAME`       - Model name for saving (default: `lstm_v1`)
//!   `TRAIN_RATIO`      - Training set ratio (default: 0.9)
//!   EPOCHS           - Number of epochs (default: 100)
//!   `BATCH_SIZE`       - Batch size (default: 32)
//!   `LEARNING_RATE`    - Learning rate (default: 0.001)
//!   RESUME           - Resume from checkpoint (default: false)
//!
//! Example:
//!   `DATABASE_URL=postgres`://... EPOCHS=50 cargo run --example pipeline

use anyhow::Result;
use database::initialize_pool;
use rocket_league_score::commands;
use tracing_subscriber::EnvFilter;

fn get_env_or_default<T: core::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    // Read configuration from environment
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "lstm_v1".to_string());
    let train_ratio: f64 = get_env_or_default("TRAIN_RATIO", 0.9);
    let epochs: usize = get_env_or_default("EPOCHS", 100);
    let batch_size: usize = get_env_or_default("BATCH_SIZE", 32);
    let learning_rate: f64 = get_env_or_default("LEARNING_RATE", 0.001);
    let resume: bool = get_env_or_default("RESUME", false);

    println!("=== Full Training Pipeline ===");
    println!("Model name:    {model_name}");
    println!(
        "Train ratio:   {train_ratio:.1}%",
        train_ratio = train_ratio * 100.0
    );
    println!("Epochs:        {epochs}");
    println!("Batch size:    {batch_size}");
    println!("Learning rate: {learning_rate}");
    println!("Resume:        {resume}");
    println!();

    // Initialize database
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    // Run the full training pipeline
    commands::full_train::run(
        &model_name,
        train_ratio,
        epochs,
        batch_size,
        learning_rate,
        resume,
    )
    .await?;

    Ok(())
}
