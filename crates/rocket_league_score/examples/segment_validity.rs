//! Does "this 20 seconds you played like a Champion" actually mean anything?
//!
//! # The claim under test
//!
//! The app shows a per-segment rank readout: each ~20-second slice of a replay gets its own
//! predicted level, and the game average becomes the player's score. That interface asserts
//! something specific — **a player's level varies across a game, and the model can see it**.
//!
//! Nothing in the project has ever tested that assertion. Every metric through row 30 scores
//! either a per-player whole-match aggregate or a within-lobby comparison. The per-segment
//! numbers, which are the thing users actually look at, have never been validated at all.
//!
//! They may not survive scrutiny, because training works against them: a Champion's replay
//! yields ~15 segments and **every one is labelled "Champion"**, including the stretches where
//! they did nothing. Each such segment is a gradient telling the model "this looked weak, call
//! it Champion anyway" — i.e. training actively flattens the variation the UI displays.
//!
//! # How it is tested
//!
//! For each player in each replay, take that player's own segments and ask whether the ones
//! the model scored higher were also the ones they actually played better in. Performance is
//! measured model-independently (ball-possession share over the segment's own frame window,
//! plus speed and demos), so this is a genuine external check rather than the model grading
//! its own homework.
//!
//! The comparison is **within one player within one game**, which controls for everything
//! that separates players and games: account rank, lobby level, playstyle, match tempo. All
//! that varies is which 20 seconds it is.
//!
//! [`ml_model::SegmentPrediction`] reports `start_frame` / `end_frame` in original replay frame
//! indices, so performance is computed over exactly the window the model scored — no
//! approximate alignment.
//!
//! # Reading the output
//!
//! `within_player_concordance` is the fraction of same-player segment pairs where the higher
//! prediction also had the higher possession. **0.500 means the per-segment readout is noise
//! around a per-player average** — the UI would be decorative. Above 0.500 means it tracks real
//! variation in play.
//!
//! A shuffled control is reported alongside: predictions permuted within each player, which
//! must land at 0.500. If it does not, the measurement itself is broken.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example segment_validity -- \
//!     --model models/lstm_v20/checkpoint_best --limit 400 \
//!     --out data/segment_validity.csv
//! ```
//!
//! The feature view defaults to whatever the checkpoint recorded in its `.config.json`, so a
//! self-only checkpoint is scored with its context columns zeroed. Note that a full-context
//! model is expected to look flat here for an uninteresting reason — the lobby shortcut makes
//! it predict the lobby, which does not change within a game — so a self-only arm is the
//! informative one.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use clap::Parser;
use config::OBJECT_STORE;
use database::{initialize_pool, list_replay_players_by_replay, list_replays_by_split};
use feature_extractor::TOTAL_PLAYERS;
use ml_model::SequenceModel;
use ml_model_training::load_checkpoint;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;
use replay_parser::parse_replay_from_bytes;
use replay_structs::{DatasetSplit, ParsedReplay};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

type InferenceBackend = NdArray;

const PLAYERS_PER_TEAM: usize = TOTAL_PLAYERS / 2;

#[derive(Parser, Debug)]
#[command(name = "segment_validity")]
#[command(about = "Test whether per-segment rank readouts track actual play", long_about = None)]
struct Args {
    /// Checkpoint to score with. Must be a **full-context** model (see module docs).
    #[arg(short = 'm', long, default_value = "models/lstm_v20/checkpoint_best")]
    model: String,

    /// Dataset split.
    #[arg(long, default_value = "evaluation")]
    split: String,

    /// Per-segment rows are written here.
    #[arg(long, default_value = "data/segment_validity.csv")]
    out: PathBuf,

    /// Replays to score. CPU inference, so this dominates runtime.
    #[arg(long, default_value_t = 400)]
    limit: usize,

    /// Sequence length; must match the checkpoint.
    #[arg(long, default_value_t = 300)]
    seq_len: usize,

    /// Score on the self-only 27-feature view, zeroing the other five cars.
    ///
    /// Defaults to whatever the checkpoint recorded in its `.config.json`, matching
    /// `revalidate`. Without this, a self-only checkpoint would be fed context features it
    /// never trained on.
    #[arg(long)]
    self_only: Option<bool>,
}

/// Reads `self_only_features` back out of a checkpoint config, defaulting to `false` for
/// checkpoints written before the field existed.
fn checkpoint_self_only(model_path: &str) -> bool {
    let Ok(raw) = std::fs::read_to_string(format!("{model_path}.config.json")) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&raw)
        .ok()
        .and_then(|value| value.get("self_only_features")?.as_bool())
        .unwrap_or(false)
}

/// One player's measured performance over one segment's frame window.
#[derive(Debug, Default, Clone, Copy)]
struct SegmentPerformance {
    possession_frames: u32,
    window_frames: u32,
    demos_received: u32,
    speed_sum_uu: f64,
    speed_samples: u64,
    goals: u32,
}

impl SegmentPerformance {
    fn possession_share(self) -> f64 {
        if self.window_frames == 0 {
            0.0
        } else {
            f64::from(self.possession_frames) / f64::from(self.window_frames)
        }
    }

    fn mean_speed(self) -> f64 {
        if self.speed_samples == 0 {
            0.0
        } else {
            self.speed_sum_uu / self.speed_samples as f64
        }
    }
}

fn parse_split(raw: &str) -> Result<DatasetSplit> {
    match raw.to_ascii_lowercase().as_str() {
        "training" | "train" => Ok(DatasetSplit::Training),
        "evaluation" | "eval" | "test" => Ok(DatasetSplit::Evaluation),
        other => anyhow::bail!("unknown split {other:?}"),
    }
}

/// Slot → name and slot → target MMR, matching `revalidate` exactly.
fn slot_roster(
    players: &[replay_structs::ReplayPlayer],
) -> ([String; TOTAL_PLAYERS], [f32; TOTAL_PLAYERS]) {
    let mut blue: Vec<_> = players.iter().filter(|p| p.team == 0).collect();
    let mut orange: Vec<_> = players.iter().filter(|p| p.team == 1).collect();
    blue.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    let pick = |slot: usize| -> Option<&&replay_structs::ReplayPlayer> {
        if slot < PLAYERS_PER_TEAM {
            blue.get(slot)
        } else {
            orange.get(slot - PLAYERS_PER_TEAM)
        }
    };

    let names = core::array::from_fn(|slot| {
        pick(slot).map_or_else(String::new, |p| p.player_name.clone())
    });
    let targets = core::array::from_fn(|slot| {
        pick(slot).map_or(0.0, |p| {
            if p.rank_known {
                p.rank_division.mmr_middle() as f32
            } else {
                0.0
            }
        })
    });

    (names, targets)
}

/// Measures every slot's performance over `[start_frame, end_frame)`.
fn measure_window(
    parsed: &ParsedReplay,
    by_name: &HashMap<&str, usize>,
    start_frame: usize,
    end_frame: usize,
) -> [SegmentPerformance; TOTAL_PLAYERS] {
    let mut performance = [SegmentPerformance::default(); TOTAL_PLAYERS];
    let mut was_demolished = [false; TOTAL_PLAYERS];

    let end = end_frame.min(parsed.frames.len());
    let Some(window) = parsed.frames.get(start_frame..end) else {
        return performance;
    };

    for frame in window {
        let ball = &frame.ball.position;
        let mut closest: Option<(usize, f32)> = None;

        for player in &frame.players {
            let Some(&slot) = by_name.get(player.name.as_str()) else {
                continue;
            };
            let Some(entry) = performance.get_mut(slot) else {
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
            entry.speed_sum_uu += f64::from(
                velocity
                    .z
                    .mul_add(
                        velocity.z,
                        velocity.x.mul_add(velocity.x, velocity.y * velocity.y),
                    )
                    .sqrt(),
            );
            entry.speed_samples += 1;
            entry.window_frames += 1;

            let demolished = player.actor_state.is_demolished;
            if demolished && !was_demolished[slot] {
                entry.demos_received += 1;
            }
            was_demolished[slot] = demolished;
        }

        if let Some((slot, _)) = closest
            && let Some(entry) = performance.get_mut(slot)
        {
            entry.possession_frames += 1;
        }
    }

    for goal in &parsed.goals {
        if goal.frame >= start_frame
            && goal.frame < end
            && let Some(&slot) = by_name.get(goal.player_name.as_str())
            && let Some(entry) = performance.get_mut(slot)
        {
            entry.goals += 1;
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

    let device = NdArrayDevice::Cpu;
    info!(model = %args.model, "Loading checkpoint");
    let model: SequenceModel<InferenceBackend> =
        load_checkpoint(&args.model, &device).context("Failed to load checkpoint")?;

    let self_only = args.self_only.unwrap_or_else(|| checkpoint_self_only(&args.model));
    let feature_view = ml_model_training::FeatureView::from(self_only);
    info!(feature_view = feature_view.label(), "Feature view");

    let mut replays = list_replays_by_split(split).await?;
    replays.truncate(args.limit);
    anyhow::ensure!(!replays.is_empty(), "no replays for split {split:?}");

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    writeln!(
        out,
        "replay_id,slot,player_name,segment_index,start_frame,end_frame,prediction_mmr,target_mmr,possession_share,mean_speed_uu,demos_received,goals"
    )?;

    info!(replays = replays.len(), "Scoring segments");

    let mut rows = 0usize;
    let mut skipped = 0usize;

    for (index, replay) in replays.iter().enumerate() {
        if index > 0 && index.is_multiple_of(50) {
            info!(processed = index, total = replays.len(), rows, "progress");
        }

        let players = list_replay_players_by_replay(replay.id).await?;
        if players.is_empty() {
            skipped += 1;
            continue;
        }
        let (names, targets) = slot_roster(&players);

        let object_path = ObjectStorePath::from(replay.file_path.clone());
        let Ok(result) = OBJECT_STORE.get(&object_path).await else {
            skipped += 1;
            continue;
        };
        let Ok(bytes) = result.bytes().await else {
            skipped += 1;
            continue;
        };
        let parsed = match parse_replay_from_bytes(&bytes) {
            Ok(parsed) if !parsed.frames.is_empty() => parsed,
            _ => {
                skipped += 1;
                continue;
            }
        };

        let mut frames = feature_extractor::extract_player_centric_game_sequence_inference_with_context(
            &parsed,
            args.seq_len,
        );
        if feature_view == ml_model_training::FeatureView::SelfOnly {
            for frame in &mut frames {
                for player in frame.iter_mut() {
                    feature_view.mask_in_place(&mut player.features);
                }
            }
        }
        let segments = ml_model::predict_from_player_centric_frames(
            &model,
            &frames,
            parsed.frames.len(),
            &device,
            args.seq_len,
        );
        if segments.len() < 3 {
            // Fewer than three segments cannot show within-game variation.
            skipped += 1;
            continue;
        }

        let mut by_name: HashMap<&str, usize> = HashMap::new();
        for (slot, name) in names.iter().enumerate() {
            if !name.is_empty() {
                by_name.insert(name.as_str(), slot);
            }
        }

        for segment in &segments {
            let performance =
                measure_window(&parsed, &by_name, segment.start_frame, segment.end_frame);

            for slot in 0..TOTAL_PLAYERS {
                let Some(name) = names.get(slot).filter(|n| !n.is_empty()) else {
                    continue;
                };
                let Some(entry) = performance.get(slot) else {
                    continue;
                };
                // Slots the parser never produced would otherwise be recorded as a
                // flawless zero-possession game.
                if entry.speed_samples == 0 {
                    continue;
                }
                let Some(prediction) = segment.player_predictions.get(slot) else {
                    continue;
                };

                writeln!(
                    out,
                    "{},{},{},{},{},{},{:.4},{:.1},{:.6},{:.2},{},{}",
                    replay.id,
                    slot,
                    name.replace(',', ";"),
                    segment.segment_index,
                    segment.start_frame,
                    segment.end_frame,
                    prediction,
                    targets.get(slot).copied().unwrap_or(0.0),
                    entry.possession_share(),
                    entry.mean_speed(),
                    entry.demos_received,
                    entry.goals
                )?;
                rows += 1;
            }
        }
    }

    out.flush()?;
    warn!(skipped, "replays skipped");
    println!(
        "segment_validity: wrote {rows} rows to {} (skipped {skipped} replays)",
        args.out.display()
    );

    Ok(())
}
