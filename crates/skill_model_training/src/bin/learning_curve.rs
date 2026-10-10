#![expect(
    clippy::indexing_slicing,
    reason = "slot arrays and lobby indices bounded by their own lengths"
)]

//! How much data the match model needs: trains on growing subsets of the training replays
//! and scores each on the full evaluation split.
//!
//! `--keep-players-per-team` drops players beyond the first N of each team before training
//! and scoring, which approximates how much a duels (1) or doubles (2) replay carries
//! compared with a standard one. `--players-per-team` keeps only replays of that playlist
//! size once several playlists are in the CSV.
//!
//! Usage:
//!   cargo run --release -p skill_model_training --bin learning_curve -- --sizes 1000,2000,4000

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use feature_extractor::{MATCH_STAT_COUNT, SLOTS_PER_TEAM};
use skill_model::TabularLayout;
use skill_model_training::dataset::{Lobby, Partition, evaluate, read_lobbies, train_model};
use skill_model_training::evaluation::{
    PlayerPrediction, compute_lobby_metrics, stable_replay_hash,
};
use skill_model_training::gradient_boosting::GradientBoostingConfig;

#[derive(Parser, Debug)]
#[command(about = "Learning curve of the whole-match model")]
struct Args {
    #[arg(long, default_value = "data/match_stats.csv")]
    match_stats: PathBuf,

    /// Training replay counts to try (the early-stopping share is taken from the rest).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1000,2000,4000,8000,16000,1000000"
    )]
    sizes: Vec<usize>,

    /// Keep only the first N players of each team (simulates smaller playlists).
    #[arg(long)]
    keep_players_per_team: Option<usize>,

    /// Keep only replays whose playlist has this many players per team.
    #[arg(long)]
    players_per_team: Option<usize>,
}

/// Drops every slot past the first `keep` of each team.
fn keep_first_players(lobby: &mut Lobby, keep: usize) {
    for slot in 0..lobby.stats.len() {
        if slot % SLOTS_PER_TEAM >= keep {
            lobby.stats[slot] = None;
            lobby.targets[slot] = 0.0;
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut lobbies = read_lobbies(&args.match_stats)?;
    if let Some(players_per_team) = args.players_per_team {
        lobbies.retain(|lobby| lobby.players_per_team() == Some(players_per_team));
    }
    if let Some(keep) = args.keep_players_per_team {
        for lobby in &mut lobbies {
            keep_first_players(lobby, keep);
        }
    }
    let layout = TabularLayout {
        own: true,
        deviation: true,
        lobby_mean: true,
        team_context: true,
        stats: (0..MATCH_STAT_COUNT).collect(),
    };
    let config = GradientBoostingConfig::default();

    // A fixed random order of the training replays: each size is a prefix of it, so larger
    // runs contain the smaller ones.
    let mut training_order: Vec<usize> = (0..lobbies.len())
        .filter(|&index| lobbies[index].partition == Partition::Training)
        .collect();
    training_order.sort_by_key(|&index| stable_replay_hash(lobbies[index].replay_id));
    let early_stopping_count = lobbies
        .iter()
        .filter(|lobby| lobby.partition == Partition::EarlyStopping)
        .count();
    println!(
        "{} training, {early_stopping_count} early-stopping, {} evaluation replays",
        training_order.len(),
        lobbies
            .iter()
            .filter(|lobby| lobby.partition == Partition::Evaluation)
            .count()
    );

    for &size in &args.sizes {
        let size = size.min(training_order.len());
        let mut subset = lobbies.clone();
        for &index in training_order.iter().skip(size) {
            subset[index].partition = Partition::Evaluation;
            subset[index].stats = Default::default();
        }
        // Keep early stopping proportional to the training set, as in the full run.
        let early_stopping_keep = (size / 9).max(200);
        let mut early_stopping_seen = 0;
        for lobby in &mut subset {
            if lobby.partition == Partition::EarlyStopping {
                early_stopping_seen += 1;
                if early_stopping_seen > early_stopping_keep {
                    lobby.partition = Partition::Evaluation;
                    lobby.stats = Default::default();
                }
            }
        }
        let model = train_model(&subset, &layout, &config);
        let predictions: Vec<PlayerPrediction> = evaluate(&subset, &model)
            .iter()
            .map(|scored| scored.prediction)
            .collect();
        let metrics = compute_lobby_metrics(&predictions);
        let training_players: usize = subset
            .iter()
            .filter(|lobby| lobby.partition == Partition::Training)
            .map(|lobby| lobby.labelled_slots().count())
            .sum();
        println!(
            "  training replays {size:>6} (players {training_players:>6})  player_rmse={:.1}  lobby_rmse={:.1}  within_r={:.3} ±{:.3}  concordance={:.3}",
            metrics.player_rmse,
            metrics.lobby_rmse,
            metrics.within_r,
            metrics.within_r_standard_error,
            metrics.concordance,
        );
    }
    Ok(())
}
