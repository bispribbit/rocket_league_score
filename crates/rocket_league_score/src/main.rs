//! Rocket League Impact Score Calculator
//!
//! A machine learning-based tool for evaluating player performance
//! in Rocket League replays.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use database::{create_pool, run_migrations};
use tracing::info;
use tracing_subscriber::EnvFilter;

mod commands;

/// Rocket League Impact Score Calculator
#[derive(Parser)]
#[command(name = "rl-score")]
#[command(about = "ML-based impact score calculator for Rocket League replays")]
#[command(version)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest replay files into the database
    Ingest {
        /// Path to the folder containing replay files
        #[arg(short, long)]
        folder: PathBuf,

        /// Game mode for the replays (e.g., "3v3", "2v2", "1v1")
        #[arg(short, long, default_value = "3v3")]
        game_mode: String,

        /// Path to a CSV file containing player ratings
        /// Format: `replay_filename,player_name,team,skill_rating`
        #[arg(short, long)]
        ratings_file: Option<PathBuf>,
    },

    /// Train the ML model on ingested replays
    Train {
        /// Name for the model
        #[arg(short, long, default_value = "impact_model")]
        name: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "64")]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.0001")]
        learning_rate: f64,
    },

    /// Predict impact scores for a replay
    Predict {
        /// Path to the replay file
        #[arg(short, long)]
        replay: PathBuf,

        /// Model name to use for prediction
        #[arg(short, long, default_value = "impact_model")]
        model: String,

        /// Model version (uses latest if not specified)
        #[arg(short, long)]
        version: Option<i32>,
    },

    /// Run database migrations
    Migrate,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing subscriber
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::fmt().with_env_filter(filter).init();

    let database_url = std::env::var("DATABASE_URL")?;
    let pool = create_pool(&database_url).await?;

    match cli.command {
        Commands::Ingest {
            folder,
            game_mode,
            ratings_file,
        } => {
            commands::ingest::run(&pool, &folder, &game_mode, ratings_file.as_deref()).await?;
        }
        Commands::Train {
            name,
            epochs,
            batch_size,
            learning_rate,
        } => {
            commands::train::run(&pool, &name, epochs, batch_size, learning_rate).await?;
        }
        Commands::Predict {
            replay,
            model,
            version,
        } => {
            commands::predict::run(&pool, &replay, &model, version).await?;
        }
        Commands::Migrate => {
            run_migrations(&pool).await?;
            info!("Migrations completed successfully");
        }
    }

    Ok(())
}
