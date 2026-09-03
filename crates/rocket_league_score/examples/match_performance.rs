//! Extracts per-player **match performance** for the evaluation split.
//!
//! # Why this exists
//!
//! Every metric in `docs/experiment.md` up to row 29 scores the model against a player's
//! **account rank** — "is this player's rank at least 150 MMR above their lobby's median
//! rank". The product question is different: *"this silver lobby has one guy playing like a
//! platinum and exterminating the competition"*. That is a claim about how someone **played**,
//! not about what their account says.
//!
//! The two disagree in both directions, and both disagreements are invisible to the current
//! metrics:
//!
//! * A high-ranked player who had a bad game is a **positive** by account rank with nothing
//!   to detect.
//! * A lower-ranked player who genuinely dominated is a **negative** by account rank, even
//!   though they are exactly what the product wants surfaced.
//!
//! This binary produces the missing half: model-independent, per-player performance for each
//! evaluation lobby, keyed by the **same slot convention** the prediction dump uses so the two
//! can be joined. Nothing here touches a model, so the output can be used to build labels
//! without circularity.
//!
//! # Slot convention
//!
//! Blue team sorted by player name occupies slots 0–2, orange sorted by name slots 3–5 —
//! identical to `revalidate`'s `target_mmr_from_players` and to
//! `PlayerRoster::from_player_ratings`. Joining on `(replay_id, slot)` is therefore exact.
//!
//! # What is measured
//!
//! * `goals` — attributed from the replay header's goal events.
//! * `closest_to_ball_frames` — frames where this player was the closest of all six to the
//!   ball. A possession proxy: the player driving the game holds it for longer. This is the
//!   more useful of the two, because goals are heavily zero-inflated (a dominant defensive or
//!   playmaking game scores none).
//! * `demos_received` — transitions into the demolished state; being repeatedly demolished is
//!   the opposite of dominating.
//! * `mean_speed_uu` — mean speed in unreal units, a coarse mechanical-activity measure.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example match_performance -- \
//!     --split evaluation --out data/eval_match_performance.csv
//! ```

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use config::OBJECT_STORE;
use database::{initialize_pool, list_replay_players_by_replay, list_replays_by_split};
use feature_extractor::TOTAL_PLAYERS;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, ParsedReplay};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

/// Players per team, mirroring the slot layout.
const PLAYERS_PER_TEAM: usize = TOTAL_PLAYERS / 2;

#[derive(Parser, Debug)]
#[command(name = "match_performance")]
#[command(about = "Extract per-player match performance for a dataset split", long_about = None)]
struct Args {
    /// Dataset split to extract (`training` or `evaluation`).
    #[arg(long, default_value = "evaluation")]
    split: String,

    /// Destination CSV.
    #[arg(long, default_value = "data/eval_match_performance.csv")]
    out: PathBuf,

    /// Stop after this many replays (for a quick smoke run).
    #[arg(long)]
    limit: Option<usize>,

    /// Log progress every N replays.
    #[arg(long, default_value_t = 100)]
    progress_every: usize,
}

/// One player's performance in one match.
#[derive(Debug, Default, Clone)]
struct SlotPerformance {
    player_name: String,
    goals: u32,
    closest_to_ball_frames: u32,
    demos_received: u32,
    speed_sum_uu: f64,
    speed_samples: u64,
}

fn parse_split(raw: &str) -> Result<DatasetSplit> {
    match raw.to_ascii_lowercase().as_str() {
        "training" | "train" => Ok(DatasetSplit::Training),
        "evaluation" | "eval" | "test" => Ok(DatasetSplit::Evaluation),
        other => anyhow::bail!("unknown split {other:?} (expected training or evaluation)"),
    }
}

/// Builds the canonical slot → player-name mapping.
///
/// Must stay identical to `revalidate::target_mmr_from_players`, otherwise performance rows
/// would be joined onto the wrong player's prediction — a silent and total corruption of the
/// analysis, since every row would still look well-formed.
fn slot_names(players: &[replay_structs::ReplayPlayer]) -> [String; TOTAL_PLAYERS] {
    let mut blue: Vec<_> = players.iter().filter(|p| p.team == 0).collect();
    let mut orange: Vec<_> = players.iter().filter(|p| p.team == 1).collect();

    blue.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    core::array::from_fn(|slot| {
        if slot < PLAYERS_PER_TEAM {
            blue.get(slot)
                .map_or_else(String::new, |p| p.player_name.clone())
        } else {
            orange
                .get(slot - PLAYERS_PER_TEAM)
                .map_or_else(String::new, |p| p.player_name.clone())
        }
    })
}

/// Accumulates per-slot performance over a parsed replay.
fn measure(parsed: &ParsedReplay, names: &[String; TOTAL_PLAYERS]) -> [SlotPerformance; TOTAL_PLAYERS] {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (slot, name) in names.iter().enumerate() {
        if !name.is_empty() {
            by_name.insert(name.as_str(), slot);
        }
    }

    let mut performance: [SlotPerformance; TOTAL_PLAYERS] = core::array::from_fn(|slot| {
        SlotPerformance {
            player_name: names[slot].clone(),
            ..SlotPerformance::default()
        }
    });

    for goal in &parsed.goals {
        if let Some(&slot) = by_name.get(goal.player_name.as_str())
            && let Some(entry) = performance.get_mut(slot)
        {
            entry.goals += 1;
        }
    }

    let mut was_demolished = [false; TOTAL_PLAYERS];

    for frame in &parsed.frames {
        let ball = &frame.ball.position;

        let mut closest: Option<(usize, f32)> = None;
        for player in &frame.players {
            let Some(&slot) = by_name.get(player.name.as_str()) else {
                continue;
            };

            let position = &player.actor_state.position;
            let dx = position.x - ball.x;
            let dy = position.y - ball.y;
            let dz = position.z - ball.z;
            let distance_sq = dz.mul_add(dz, dx.mul_add(dx, dy * dy));

            if closest.is_none_or(|(_, best)| distance_sq < best) {
                closest = Some((slot, distance_sq));
            }

            let velocity = &player.actor_state.velocity;
            let speed = f64::from(
                velocity
                    .z
                    .mul_add(velocity.z, velocity.x.mul_add(velocity.x, velocity.y * velocity.y))
                    .sqrt(),
            );
            if let Some(entry) = performance.get_mut(slot) {
                entry.speed_sum_uu += speed;
                entry.speed_samples += 1;
            }

            // Count the rising edge only: `is_demolished` stays true for the whole respawn
            // window, so summing it would measure respawn duration rather than demo count.
            let demolished = player.actor_state.is_demolished;
            if demolished
                && !was_demolished[slot]
                && let Some(entry) = performance.get_mut(slot)
            {
                entry.demos_received += 1;
            }
            was_demolished[slot] = demolished;
        }

        if let Some((slot, _)) = closest
            && let Some(entry) = performance.get_mut(slot)
        {
            entry.closest_to_ball_frames += 1;
        }
    }

    performance
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .init();

    let args = Args::parse();
    let split = parse_split(&args.split)?;

    let database_url =
        std::env::var("DATABASE_URL").context("DATABASE_URL environment variable is required")?;
    initialize_pool(&database_url).await?;

    let mut replays = list_replays_by_split(split).await?;
    if let Some(limit) = args.limit {
        replays.truncate(limit);
    }
    anyhow::ensure!(!replays.is_empty(), "no replays for split {split:?}");

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.out)
            .with_context(|| format!("creating {}", args.out.display()))?,
    );
    writeln!(
        out,
        "replay_id,slot,player_name,matched,goals,closest_to_ball_frames,total_frames,demos_received,mean_speed_uu"
    )?;

    info!(
        replays = replays.len(),
        split = %args.split,
        "Extracting match performance"
    );

    let mut written = 0usize;
    let mut skipped_no_players = 0usize;
    let mut skipped_fetch = 0usize;
    let mut skipped_parse = 0usize;

    for (index, replay) in replays.iter().enumerate() {
        if index > 0 && args.progress_every > 0 && index.is_multiple_of(args.progress_every) {
            info!(
                processed = index,
                total = replays.len(),
                rows = written,
                "progress"
            );
        }

        let players = list_replay_players_by_replay(replay.id).await?;
        if players.is_empty() {
            skipped_no_players += 1;
            continue;
        }
        let names = slot_names(&players);

        let object_path = ObjectStorePath::from(replay.file_path.clone());
        let bytes = match OBJECT_STORE.get(&object_path).await {
            Ok(result) => match result.bytes().await {
                Ok(bytes) => bytes,
                Err(error) => {
                    warn!(replay_id = %replay.id, %error, "read failed");
                    skipped_fetch += 1;
                    continue;
                }
            },
            Err(error) => {
                warn!(replay_id = %replay.id, %error, "fetch failed");
                skipped_fetch += 1;
                continue;
            }
        };

        let parsed = match parse_replay_from_bytes(&bytes) {
            Ok(parsed) if !parsed.frames.is_empty() => parsed,
            _ => {
                skipped_parse += 1;
                continue;
            }
        };

        let total_frames = parsed.frames.len();
        let performance = measure(&parsed, &names);

        for (slot, entry) in performance.iter().enumerate() {
            if entry.player_name.is_empty() {
                continue;
            }
            let mean_speed = if entry.speed_samples > 0 {
                entry.speed_sum_uu / entry.speed_samples as f64
            } else {
                0.0
            };
            writeln!(
                out,
                "{},{},{},{},{},{},{},{},{:.2}",
                replay.id,
                slot,
                entry.player_name.replace(',', ";"),
                u8::from(entry.speed_samples > 0),
                entry.goals,
                entry.closest_to_ball_frames,
                total_frames,
                entry.demos_received,
                mean_speed
            )?;
            written += 1;
        }
    }

    out.flush()?;

    info!(
        rows = written,
        replays_processed = replays.len() - skipped_no_players - skipped_fetch - skipped_parse,
        skipped_no_players,
        skipped_fetch,
        skipped_parse,
        out = %args.out.display(),
        "Done"
    );
    println!(
        "match_performance: wrote {written} rows to {} (skipped: no_players={skipped_no_players} fetch={skipped_fetch} parse={skipped_parse})",
        args.out.display()
    );

    Ok(())
}
