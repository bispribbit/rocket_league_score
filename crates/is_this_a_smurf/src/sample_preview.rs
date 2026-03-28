//! Synthetic [`PredictionResults`] for local UI preview (smurf badge, tables).
//!
//! Only wired from the upload screen in debug builds; the data is tiny and harmless in release
//! binaries if linked, but the button is not shown there.

use feature_extractor::TOTAL_PLAYERS;
use replay_structs::{RankDivision, Team};

use crate::app_state::{PlayerAverage, PredictionResults, SegmentDisplayData};

const SAMPLE_REPLAY_DISPLAY_NAME: &str = "preview_match.replay";

#[must_use]
pub(crate) fn sample_replay_display_name() -> String {
    SAMPLE_REPLAY_DISPLAY_NAME.to_string()
}

/// Returns results where the third player (last slot on blue) has a mean MMR well above the lobby
/// median, so the smurf badge appears next to their name and column header.
#[must_use]
pub(crate) fn prediction_results_with_obvious_smurf() -> PredictionResults {
    let player_names: Vec<String> = vec![
        "Citrus".to_string(),
        "Nova".to_string(),
        "DefinitelyNotSmurfing".to_string(),
        "Moss".to_string(),
        "Pebble".to_string(),
        "Granite".to_string(),
    ];
    let player_teams: Vec<Team> = vec![
        Team::Blue,
        Team::Blue,
        Team::Blue,
        Team::Orange,
        Team::Orange,
        Team::Orange,
    ];

    let segment_one_mmr: [f32; TOTAL_PLAYERS] = [1000.0, 1040.0, 1650.0, 1070.0, 1110.0, 1160.0];
    let segment_two_mmr: [f32; TOTAL_PLAYERS] = [1020.0, 1080.0, 1750.0, 1110.0, 1150.0, 1200.0];

    let averages: [f32; TOTAL_PLAYERS] = std::array::from_fn(|player_index| {
        let first = segment_one_mmr.get(player_index).copied().unwrap_or(0.0);
        let second = segment_two_mmr.get(player_index).copied().unwrap_or(0.0);
        f32::midpoint(first, second)
    });

    let player_averages: Vec<PlayerAverage> = (0..TOTAL_PLAYERS)
        .map(|player_index| {
            let name = player_names
                .get(player_index)
                .cloned()
                .unwrap_or_else(|| format!("Player {}", player_index + 1));
            let team = player_teams
                .get(player_index)
                .copied()
                .unwrap_or(Team::Blue);
            let average_mmr = averages.get(player_index).copied().unwrap_or(0.0);
            PlayerAverage {
                name,
                team,
                average_mmr,
                rank: RankDivision::from(average_mmr),
            }
        })
        .collect();

    let segments: Vec<SegmentDisplayData> = vec![
        SegmentDisplayData {
            segment_number: 1,
            start_time: 0.0,
            end_time: 120.0,
            player_mmr: segment_one_mmr,
        },
        SegmentDisplayData {
            segment_number: 2,
            start_time: 120.0,
            end_time: 240.0,
            player_mmr: segment_two_mmr,
        },
    ];

    PredictionResults {
        player_names,
        player_teams,
        segments,
        player_averages,
    }
}
