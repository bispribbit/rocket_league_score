//! Ingest command - imports replay files into the database.

use core::str::FromStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use database::{CreateReplay, CreateReplayPlayer, DownloadStatus, GameMode};
use replay_parser::{ExtractedPlayer, extract_players_from_metadata, parse_replay};
use tracing::{info, warn};
use uuid::Uuid;

use crate::rank::Rank;

/// Gets the base replay path from environment variable or uses default.
fn get_replay_base_path() -> PathBuf {
    std::env::var("REPLAY_BASE_PATH")
        .map_or_else(|_| PathBuf::from("/workspace/ballchasing"), PathBuf::from)
}

/// Runs the ingest command.
///
/// # Errors
///
/// Returns an error if ingestion fails.
pub async fn run(folder: &Path, game_mode_str: &str, ratings_file: Option<&Path>) -> Result<()> {
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

    // Get base replay path for computing relative paths
    let base_path = get_replay_base_path();
    info!(base_path = %base_path.display(), "Using base replay path");

    // Find all .replay files in the folder
    let replay_files = find_replay_files(folder)?;
    info!(count = replay_files.len(), "Found replay files");

    let mut ingested = 0;
    let mut skipped = 0;

    for replay_path in &replay_files {
        // Check if already ingested
        if database::find_replay_by_path(&replay_path).await?.is_some() {
            skipped += 1;
            continue;
        }

        // Parse the replay to validate it
        match parse_replay(replay_path) {
            Ok(_parsed) => {
                // Create replay record with relative path
                let replay = database::insert_replay(CreateReplay {
                    file_path: relative_path.clone(),
                    game_mode,
                })
                .await?;

                let file_name = replay_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Try to find ballchasing metadata
                let mut players_for_replay: Vec<CreateReplayPlayer> = Vec::new();

                // Try to match by file_path first (using relative path)
                let ballchasing_replay = find_ballchasing_replay_by_path(&relative_path).await?;

                if let Some(ballchasing) = ballchasing_replay {
                    // Extract players from ballchasing metadata
                    match extract_players_from_metadata(&ballchasing.metadata, |rank_info| {
                        Rank::from_rank_id(&rank_info.id, rank_info.division).map(Rank::mmr_middle)
                    }) {
                        Ok(extracted_players) => {
                            players_for_replay = extracted_players
                                .into_iter()
                                .map(|p: ExtractedPlayer| CreateReplayPlayer {
                                    replay_id: replay.id,
                                    player_name: p.player_name,
                                    team: p.team,
                                    skill_rating: p.skill_rating,
                                })
                                .collect();
                            info!(
                                replay_id = %replay.id,
                                players = players_for_replay.len(),
                                "Extracted players from ballchasing metadata"
                            );
                        }
                        Err(e) => {
                            warn!(
                                replay_id = %replay.id,
                                error = %e,
                                "Failed to extract players from ballchasing metadata"
                            );
                        }
                    }
                }

                // Fall back to CSV ratings if no ballchasing metadata found
                if players_for_replay.is_empty() {
                    players_for_replay = ratings
                        .iter()
                        .filter(|r| r.replay_filename == file_name)
                        .map(|r| CreateReplayPlayer {
                            replay_id: replay.id,
                            player_name: r.player_name.clone(),
                            team: r.team,
                            skill_rating: r.skill_rating,
                        })
                        .collect();
                }

                if !players_for_replay.is_empty() {
                    database::insert_replay_players(players_for_replay).await?;
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
                && ext == "replay"
            {
                files.push(path);
            }
        } else if path.is_dir() {
            // Recursively search subdirectories
            files.extend(find_replay_files(&path)?);
        }
    }

    Ok(files)
}

/// Finds a ballchasing replay by file path.
///
/// Tries to match by `file_path` field first, then by extracting UUID from filename.
async fn find_ballchasing_replay_by_path(
    file_path: &str,
) -> Result<Option<database::BallchasingReplay>> {
    // Try to find by file_path field
    let all_ranks = database::BallchasingRank::all_ranked();
    for rank in all_ranks {
        let replays =
            database::list_ballchasing_replays_by_rank(rank, Some(DownloadStatus::Downloaded))
                .await?;

        for replay in replays {
            if let Some(replay_file_path) = &replay.file_path
                && replay_file_path == file_path
            {
                return Ok(Some(replay));
            }
        }
    }

    // Try to extract UUID from filename and match by ID
    if let Some(file_name) = std::path::Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        && let Ok(replay_id) = Uuid::parse_str(file_name)
        && let Some(replay) = database::find_ballchasing_replay_by_id(replay_id).await?
    {
        return Ok(Some(replay));
    }

    Ok(None)
}
