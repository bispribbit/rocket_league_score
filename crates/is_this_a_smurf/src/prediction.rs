//! Replay parsing helpers, segment timeline data, and final result building.

use feature_extractor::{FRAME_SUBSAMPLE_RATE, TOTAL_PLAYERS};
use ml_model::SegmentPrediction;
use replay_structs::{GameFrame, ParsedReplay, RankDivision, Team};

use crate::app_state::{
    GoalMarkerDisplay, PlayerAverage, PredictionResults, SegmentDisplayData, SegmentStepInfo,
    StepStatus,
};

/// MMR above the lobby median before we treat a player as a smurf suspect (approximate).
pub(crate) const SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN: f32 = 200.0;

/// Builds segment step infos (time ranges) from parsed frames and sequence length.
///
/// `sequence_length` counts **subsampled** frames (same step as training:
/// [`FRAME_SUBSAMPLE_RATE`]). Wall-clock bounds use the corresponding span in the original replay.
pub(crate) fn segment_step_infos(
    frames: &[GameFrame],
    sequence_length: usize,
    num_segments: usize,
) -> Vec<SegmentStepInfo> {
    let step = FRAME_SUBSAMPLE_RATE;
    (0..num_segments)
        .map(|seg_idx| {
            let start_frame = seg_idx * sequence_length * step;
            let end_frame = ((seg_idx + 1) * sequence_length * step).min(frames.len());
            let start_time = frames.get(start_frame).map_or(0.0, |frame| frame.time);
            let end_time = frames
                .get(end_frame.saturating_sub(1))
                .map_or(0.0, |frame| frame.time);
            SegmentStepInfo {
                start_time,
                end_time,
                status: StepStatus::Pending,
                player_segment_ranks: None,
            }
        })
        .collect()
}

/// Converts per-player predicted MMR to rank divisions for one segment.
pub(crate) fn ranks_from_player_predictions(
    player_predictions: &[f32; TOTAL_PLAYERS],
) -> [RankDivision; TOTAL_PLAYERS] {
    let mut result = [RankDivision::BronzeIDivision1; TOTAL_PLAYERS];
    for (player_index, slot) in result.iter_mut().enumerate() {
        if let Some(mmr_value) = player_predictions.get(player_index) {
            *slot = RankDivision::from(*mmr_value);
        }
    }
    result
}

/// Times (seconds) at each segment boundary: start of segment 0, then starts of 1..n, then end of last segment.
pub(crate) fn compute_segment_boundary_times(segment_steps: &[SegmentStepInfo]) -> Vec<f32> {
    let Some(first_step) = segment_steps.first() else {
        return Vec::new();
    };
    let mut boundary_times_seconds = Vec::with_capacity(segment_steps.len() + 1);
    boundary_times_seconds.push(first_step.start_time);
    for step in segment_steps.iter().skip(1) {
        boundary_times_seconds.push(step.start_time);
    }
    if let Some(last_step) = segment_steps.last() {
        boundary_times_seconds.push(last_step.end_time);
    }
    boundary_times_seconds
}

/// Pads or truncates to [`TOTAL_PLAYERS`] for stable lane indices in the timeline UI.
pub(crate) fn prepare_players_for_timeline(parsed: &ParsedReplay) -> (Vec<String>, Vec<Team>) {
    let Some(first_frame) = parsed.frames.first() else {
        return (Vec::new(), Vec::new());
    };
    let mut player_names: Vec<String> = first_frame
        .players
        .iter()
        .map(|player| player.name.as_ref().clone())
        .collect();
    let mut player_teams: Vec<Team> = first_frame
        .players
        .iter()
        .map(|player| player.team)
        .collect();
    while player_names.len() < TOTAL_PLAYERS {
        player_names.push(format!("Player {}", player_names.len() + 1));
        player_teams.push(Team::Blue);
    }
    player_names.truncate(TOTAL_PLAYERS);
    player_teams.truncate(TOTAL_PLAYERS);
    (player_names, player_teams)
}

/// Goal markers for the timeline, derived from replay header goals and frame times.
pub(crate) fn build_goal_markers(
    parsed: &ParsedReplay,
    player_names: &[String],
) -> Vec<GoalMarkerDisplay> {
    let mut markers = Vec::new();
    for goal in &parsed.goals {
        let Some(time_seconds) = parsed.frames.get(goal.frame).map(|frame| frame.time) else {
            continue;
        };
        let player_lane_index = player_names
            .iter()
            .position(|name| name == &goal.player_name);
        markers.push(GoalMarkerDisplay {
            time_seconds,
            scorer_name: goal.player_name.clone(),
            team: goal.player_team,
            player_lane_index,
        });
    }
    markers
}

/// Median of a pre-sorted slice of `f32` values. Returns `0.0` for an empty slice.
///
/// Even-length slices return the midpoint of the two central values.
fn median_f32(sorted_values: &[f32]) -> f32 {
    let len = sorted_values.len();
    if len == 0 {
        return 0.0;
    }
    let mid = len / 2;
    if len % 2 == 1 {
        sorted_values.get(mid).copied().unwrap_or(0.0)
    } else {
        let lower = sorted_values
            .get(mid.saturating_sub(1))
            .copied()
            .unwrap_or(0.0);
        let upper = sorted_values.get(mid).copied().unwrap_or(0.0);
        f32::midpoint(lower, upper)
    }
}

/// Median of per-player median MMR values across the lobby.
///
/// Used as the lobby baseline when deciding whether a player looks suspiciously high.
pub(crate) fn lobby_median_mmr(player_averages: &[crate::app_state::PlayerAverage]) -> f32 {
    let mut values: Vec<f32> = player_averages
        .iter()
        .map(|player| player.median_mmr)
        .collect();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    median_f32(&values)
}

/// Median MMR across segments per player, mapped to global rank badges.
///
/// The median is used instead of the mean so that a few low segments (e.g. an SSL
/// player having an off-moment) do not pull the final rank estimate down.
pub(crate) fn global_ranks_from_predictions(
    segment_predictions: &[SegmentPrediction],
) -> [RankDivision; TOTAL_PLAYERS] {
    let mut result = [RankDivision::BronzeIDivision1; TOTAL_PLAYERS];
    if segment_predictions.is_empty() {
        return result;
    }
    for (player_index, slot) in result.iter_mut().enumerate() {
        let mut mmr_values: Vec<f32> = segment_predictions
            .iter()
            .filter_map(|segment| segment.player_predictions.get(player_index).copied())
            .collect();
        mmr_values
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        *slot = RankDivision::from(median_f32(&mmr_values));
    }
    result
}

/// Formats seconds as `mm:ss` with zero-padded minutes (e.g. team cards, exports).
#[expect(dead_code)] // Reserved for other UI; timeline uses [`format_timeline_boundary_label`].
pub(crate) fn format_time_mm_ss(seconds: f32) -> String {
    let total_seconds = seconds.max(0.0) as u32;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;
    format!("{minutes:02}:{secs:02}")
}

/// Formats seconds as `m:ss` (no leading zero on minutes) for boundary labels under the timeline.
pub(crate) fn format_timeline_boundary_label(seconds: f32) -> String {
    let total_seconds = seconds.max(0.0) as u32;
    let minutes = total_seconds / 60;
    let secs = total_seconds % 60;
    format!("{minutes}:{secs:02}")
}

/// Builds full prediction results from parsed data and segment predictions.
pub(crate) fn build_prediction_results(
    parsed: &ParsedReplay,
    segment_predictions: Vec<SegmentPrediction>,
) -> Result<PredictionResults, String> {
    let first_frame = parsed.frames.first().ok_or("No frames in the replay")?;
    let player_names: Vec<String> = first_frame
        .players
        .iter()
        .map(|player| player.name.as_ref().clone())
        .collect();
    let player_teams: Vec<Team> = first_frame
        .players
        .iter()
        .map(|player| player.team)
        .collect();

    let segments: Vec<SegmentDisplayData> = segment_predictions
        .iter()
        .map(|segment| {
            let start_time = parsed
                .frames
                .get(segment.start_frame)
                .map_or(0.0, |frame| frame.time);
            let end_time = parsed
                .frames
                .get(segment.end_frame.saturating_sub(1))
                .map_or(0.0, |frame| frame.time);
            SegmentDisplayData {
                segment_number: segment.segment_index + 1,
                start_time,
                end_time,
                player_mmr: segment.player_predictions,
            }
        })
        .collect();

    let player_averages: Vec<PlayerAverage> = (0..TOTAL_PLAYERS)
        .map(|player_index| {
            let mut mmr_values: Vec<f32> = segment_predictions
                .iter()
                .map(|segment| {
                    segment
                        .player_predictions
                        .get(player_index)
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            mmr_values.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            let median_mmr = median_f32(&mmr_values);
            let name = player_names
                .get(player_index)
                .cloned()
                .unwrap_or_else(|| format!("Player {}", player_index + 1));
            let team = player_teams
                .get(player_index)
                .copied()
                .unwrap_or(Team::Blue);
            let rank = RankDivision::from(median_mmr);
            PlayerAverage {
                name,
                team,
                median_mmr,
                rank,
            }
        })
        .collect();

    Ok(PredictionResults {
        player_names,
        player_teams,
        segments,
        player_averages,
    })
}
