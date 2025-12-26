//! Ingest command - imports replay files into the database.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use replay_parser::{extract_players_from_metadata, parse_replay};
use replay_structs::{BallchasingRank, BallchasingReplay, DownloadStatus, GameMode};
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
pub async fn run(folder: &Path, game_mode: GameMode, ratings_file: Option<&Path>) -> Result<()> {
    info!(folder = %folder.display(), "Ingesting replays");

    // Load ratings from CSV if provided
    let ratings = if let Some(ratings_path) = ratings_file {
        load_ratings_from_csv(ratings_path)?
    } else {
        Vec::new()
    };

    // Find all .replay files in the folder
    let replay_files = find_replay_files(folder)?;
    info!(count = replay_files.len(), "Found replay files");

    let base_path = get_replay_base_path();
    let mut ingested = 0;
    let mut skipped = 0;

    for replay_path in &replay_files {
        // Check if already ingested
        if database::find_replay_by_path(replay_path).await?.is_some() {
            skipped += 1;
            continue;
        }

        // Compute relative path from base
        let relative_path = replay_path.strip_prefix(&base_path).unwrap_or(replay_path);

        // Parse the replay to validate it
        match parse_replay(replay_path) {
            Ok(_parsed) => {
                // Create replay record with relative path
                let replay = database::insert_replay(relative_path, game_mode).await?;

                let file_name = replay_path
                    .file_name()
                    .and_then(|os_str| os_str.to_str())
                    .unwrap_or("");

                // Try to find ballchasing metadata
                let mut replay_ids = Vec::new();
                let mut player_names = Vec::new();
                let mut teams = Vec::new();
                let mut skill_ratings = Vec::new();

                // Try to match by file_path first (using relative path)
                let relative_path_str = relative_path.to_string_lossy().to_string();
                let ballchasing_replay =
                    find_ballchasing_replay_by_path(&relative_path_str).await?;

                if let Some(ballchasing) = ballchasing_replay {
                    // Extract players from ballchasing metadata
                    match extract_players_from_metadata(&ballchasing.metadata, |rank_info| {
                        Rank::from_rank_id(&rank_info.id, rank_info.division).map(Rank::mmr_middle)
                    }) {
                        Ok(extracted_players) => {
                            for p in extracted_players {
                                replay_ids.push(replay.id);
                                player_names.push(p.player_name);
                                teams.push(p.team);
                                skill_ratings.push(p.skill_rating);
                            }
                            info!(
                                replay_id = %replay.id,
                                players = replay_ids.len(),
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
                if replay_ids.is_empty() {
                    for rating in ratings.iter().filter(|r| r.replay_filename == file_name) {
                        replay_ids.push(replay.id);
                        player_names.push(rating.player_name.clone());
                        teams.push(rating.team);
                        skill_ratings.push(rating.skill_rating);
                    }
                }

                if !replay_ids.is_empty() {
                    database::insert_replay_players(
                        &replay_ids,
                        &player_names,
                        &teams,
                        &skill_ratings,
                    )
                    .await?;
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
#[derive(Debug, Clone, serde::Deserialize)]
struct RatingEntry {
    replay_filename: String,
    player_name: String,
    team: i16,
    skill_rating: i32,
}

/// Loads ratings from a CSV file.
fn load_ratings_from_csv(path: &Path) -> Result<Vec<RatingEntry>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Failed to open ratings file: {}", path.display()))?;

    let mut ratings = Vec::new();
    for result in reader.deserialize() {
        let record: RatingEntry = result.with_context(|| "Failed to parse CSV record")?;
        ratings.push(record);
    }

    Ok(ratings)
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
async fn find_ballchasing_replay_by_path(file_path: &str) -> Result<Option<BallchasingReplay>> {
    // Try to find by file_path field
    let all_ranks = BallchasingRank::all_ranked();
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
