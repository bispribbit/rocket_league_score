//! Example: Quick end-to-end test of the full training pipeline.
//!
//! This example runs the full training pipeline with a limited number of replays
//! to quickly test the entire system without running a full training session.
//!
//! Usage:
//!   cargo run --example full_pipeline
//!
//! Environment Variables:
//!   `DATABASE_URL` - `PostgreSQL` connection string (required)
//!   `NUM_REPLAYS`  - Number of replays to use (default: 100)
//!
//! This is useful for:
//! - Verifying the pipeline works end-to-end
//! - Testing changes without waiting for a full training run
//! - Identifying bottlenecks or issues at a smaller scale

use anyhow::Result;
use database::initialize_pool;
use rocket_league_score::commands;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    // Configuration for quick testing
    let num_replays: usize = 200;

    info!("=== Full Pipeline Test ===");
    info!("Testing with {num_replays} replays");

    // Initialize database
    let database_url =
        std::env::var("DATABASE_URL").expect("DATABASE_URL environment variable is required");
    initialize_pool(&database_url).await?;

    // Run the full training pipeline with limited replays
    commands::full_pipeline::run(
        "test_model", // Test model name
        0.9,          // Train ratio
        5,            // Few epochs for quick testing
        2048,         // Batch size
        0.001,        // Learning rate
        false,        // Don't resume
        Some(num_replays),
    )
    .await?;

    info!("=== Pipeline test completed successfully ===");

    Ok(())
}
