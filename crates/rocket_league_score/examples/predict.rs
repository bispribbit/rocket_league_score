//! Example: Run prediction on a replay file using a model.
//!
//! This example demonstrates how to use the predict command to analyze a replay
//! and predict player MMR using a model loaded from a file path.
//!
//! Usage:
//!   cargo run --example predict -- --replay-path replay.replay --model-path models/lstm_v5/checkpoint-epoch-50
//!
//! Examples:
//!   # Using model from file path
//!   cargo run --example predict -- --replay-path replay.replay --model-path models/lstm_v5/checkpoint-epoch-50

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rocket_league_score::commands;
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Predict player MMR from a replay file using a trained model.
#[derive(Parser, Debug)]
#[command(name = "predict")]
#[command(about = "Predict player MMR from a replay file", long_about = None)]
struct Args {
    /// Path to the replay file to analyze
    #[arg(short = 'r', long, value_name = "PATH")]
    replay_path: PathBuf,

    /// Path to model checkpoint file
    #[arg(short = 'm', long, value_name = "PATH")]
    model_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    // Parse command line arguments
    let args = Args::parse();

    info!("=== Predict Player MMR ===");
    info!("Replay path:   {}", args.replay_path.display());
    info!("Model path:    {}", args.model_path.display());

    // Run prediction
    commands::predict::run(&args.replay_path, &args.model_path)?;

    info!("=== Prediction completed successfully ===");

    Ok(())
}
