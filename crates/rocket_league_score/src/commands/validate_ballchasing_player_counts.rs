//! Scan downloaded replays and compare database player rows to replay-file player-count diagnostics.
//!
//! Run: `cargo run --bin validate_ballchasing_player_counts` with `DATABASE_URL` and optional
//! `REPLAY_BASE_PATH` (same as the ballchasing object store).

use anyhow::{Context, Result};
use config::{OBJECT_STORE, get_base_path};
use database::{initialize_pool, list_downloaded_replays, list_replay_players_by_replay};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::match_validation::diagnose_human_player_count;
use replay_parser::parse_boxcars_replay_without_policy;
use tracing::info;

/// Loads each downloaded replay from the object store and prints rows where the file suggests a
/// player count other than six, or where the database has a number of `replay_players` rows other
/// than six.
pub async fn run() -> Result<()> {
    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;

    initialize_pool(&database_url).await?;

    let base_path = get_base_path();
    info!(
        "Scanning downloaded replays (object store base: {})",
        base_path.display()
    );

    let replays = list_downloaded_replays().await?;

    let mut mismatch_count = 0usize;
    let mut read_error_count = 0usize;
    let mut parse_error_count = 0usize;

    for replay in &replays {
        let object_path = ObjectStorePath::from(replay.file_path.clone());
        let replay_data = match OBJECT_STORE
            .get(&object_path)
            .await
            .context("Failed to read from object_store")
        {
            Ok(get_result) => match get_result.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    read_error_count += 1;
                    tracing::warn!(
                        replay_id = %replay.id,
                        path = %object_path,
                        %error,
                        "Failed to read replay bytes"
                    );
                    continue;
                }
            },
            Err(error) => {
                read_error_count += 1;
                tracing::warn!(
                    replay_id = %replay.id,
                    path = %object_path,
                    %error,
                    "Failed to get from object_store"
                );
                continue;
            }
        };

        let boxcars_replay = match parse_boxcars_replay_without_policy(&replay_data) {
            Ok(parsed) => parsed,
            Err(error) => {
                parse_error_count += 1;
                tracing::warn!(
                    replay_id = %replay.id,
                    path = %object_path,
                    %error,
                    "Failed to parse replay"
                );
                continue;
            }
        };

        let breakdown = diagnose_human_player_count(&boxcars_replay);
        let db_players = list_replay_players_by_replay(replay.id).await?;
        let database_player_row_count = db_players.len();

        let has_issue = breakdown.effective_count != 6 || database_player_row_count != 6;

        if has_issue {
            mismatch_count += 1;
            info!(
                replay_id = %replay.id,
                file_path = %replay.file_path,
                database_player_row_count,
                player_stats_rows = breakdown.player_stats_rows,
                named_pri_player_count = breakdown.named_pri_player_count,
                unique_id_pri_player_count = breakdown.unique_id_pri_player_count,
                team_size_header = ?breakdown.team_size_header,
                effective_count = breakdown.effective_count,
                "Player count mismatch (expected six humans for 3v3)"
            );
        }
    }

    info!(
        total_downloaded_replays = replays.len(),
        mismatch_or_diagnostic_rows = mismatch_count,
        read_error_count,
        parse_error_count,
        "Finished scanning downloaded replays"
    );

    Ok(())
}
