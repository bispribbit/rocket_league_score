//! Ingest command - imports replay files into the database.

use std::path::Path;

use anyhow::{Context, Result};
use database::{CreateReplay, CreateReplayPlayer, GameMode, ReplayPlayerRepository, ReplayRepository};
use replay_parser::parse_replay;
use sqlx::PgPool;
use tracing::{info, warn};

/// Runs the ingest command.
///
/// # Errors
///
/// Returns an error if ingestion fails.
pub async fn run(
    pool: &PgPool,
    folder: &Path,
    game_mode_str: &str,
    ratings_file: Option<&Path>,
) -> Result<()> {
    info!(folder = %folder.display(), "Ingesting replays");

    let game_mode = GameMode::from_str(game_mode_str)
        .context("Invalid game mode. Use: 3v3, 2v2, 1v1, hoops, rumble, dropshot, snowday")?;

    // TODO: Parse ratings file if provided
    let ratings = if let Some(path) = ratings_file {
        info!(path = %path.display(), "Loading ratings file");
        load_ratings_file(path)?
    } else {
        info!("No ratings file provided, using default ratings");
        Vec::new()
    };

    // Find all .replay files in the folder
    let replay_files = find_replay_files(folder)?;
    info!(count = replay_files.len(), "Found replay files");

    let mut ingested = 0;
    let mut skipped = 0;

    for replay_path in &replay_files {
        let file_path_str = replay_path.to_string_lossy().to_string();

        // Check if already ingested
        if ReplayRepository::find_by_path(pool, &file_path_str).await?.is_some() {
            skipped += 1;
            continue;
        }

        // Parse the replay to validate it
        match parse_replay(replay_path) {
            Ok(_parsed) => {
                // TODO: Extract metadata from parsed replay

                // Create replay record
                let replay = ReplayRepository::create(
                    pool,
                    CreateReplay {
                        file_path: file_path_str.clone(),
                        game_mode,
                    },
                )
                .await?;

                // Add player ratings if available
                let file_name = replay_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                let players_for_replay: Vec<_> = ratings
                    .iter()
                    .filter(|r| r.replay_filename == file_name)
                    .map(|r| CreateReplayPlayer {
                        replay_id: replay.id,
                        player_name: r.player_name.clone(),
                        team: r.team,
                        skill_rating: r.skill_rating,
                    })
                    .collect();

                if !players_for_replay.is_empty() {
                    ReplayPlayerRepository::create_many(pool, players_for_replay).await?;
                }

                ingested += 1;
            }
            Err(e) => {
                warn!(path = %replay_path.display(), error = %e, "Failed to parse replay");
            }
        }
    }

    info!(ingested, skipped, "Ingestion complete");

    Ok(())
}

/// Rating entry from a CSV file.
#[derive(Debug, Clone)]
struct RatingEntry {
    replay_filename: String,
    player_name: String,
    team: i16,
    skill_rating: i32,
}

/// Loads player ratings from a CSV file.
fn load_ratings_file(path: &Path) -> Result<Vec<RatingEntry>> {
    // TODO: Implement actual CSV parsing
    // Expected format: replay_filename,player_name,team,skill_rating
    // Example: "abc123.replay,PlayerOne,0,1547"

    let _content = std::fs::read_to_string(path)?;

    // Mock implementation - return empty vec
    Ok(Vec::new())
}

/// Finds all .replay files in a directory (recursively).
fn find_replay_files(folder: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    if !folder.is_dir() {
        anyhow::bail!("Not a directory: {}", folder.display());
    }

    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension()
                && ext == "replay" {
                    files.push(path);
                }
        } else if path.is_dir() {
            // Recursively search subdirectories
            files.extend(find_replay_files(&path)?);
        }
    }

    Ok(files)
}

