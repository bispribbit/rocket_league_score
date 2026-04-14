//! Match timeline animation and early working panel (spinner before the timeline is ready).

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use replay_structs::Team;

use crate::app_state::{GoalMarkerDisplay, ProgressState, StepStatus};
use crate::prediction::format_timeline_boundary_label;
use crate::rank_icon::rank_division_icon_asset;

/// One goal on the timeline after sorting by time, with running score (e.g. `2-1`).
#[derive(Clone)]
struct GoalTimelineRow {
    time_seconds: f32,
    team: Team,
    score_label: String,
    player_lane_index: Option<usize>,
}

#[expect(clippy::missing_const_for_fn)] // Branches inspect `StepStatus::Done(String)`.
fn early_working_subtitle(progress: &ProgressState) -> &'static str {
    if matches!(progress.reading_file, StepStatus::Processing) {
        return "Reading the file from your device…";
    }
    if matches!(progress.copying_into_memory, StepStatus::Processing) {
        return "Copying the replay into app memory (this can be large)…";
    }
    if matches!(progress.parsing, StepStatus::Processing) {
        return "Parsing replay data…";
    }
    if matches!(progress.loading_model, StepStatus::Processing) {
        return "Loading the prediction model…";
    }
    "Running analysis…"
}

/// Large spinner and copy when the match timeline is not ready yet (read, parse, model load).
#[component]
pub(crate) fn EarlyWorkingPanel(progress: ProgressState) -> Element {
    let subtitle = early_working_subtitle(&progress);
    rsx! {
        div { class: "w-full mb-10 rounded-xl border border-gray-800 bg-gradient-to-b from-gray-900/90 to-gray-950/95 p-8 sm:p-12 shadow-lg shadow-black/40 flex flex-col items-center justify-center min-h-[min(52vh,24rem)] text-center",
            div { class: "w-14 h-14 sm:w-16 sm:h-16 border-[3px] border-gray-700 border-t-blue-500 rounded-full animate-spin mb-6" }
            p { class: "text-lg sm:text-xl font-semibold text-gray-100 tracking-tight",
                "Working on your replay"
            }
            p { class: "mt-3 max-w-md text-sm sm:text-base text-gray-400 leading-relaxed",
                "{subtitle}"
            }
            p { class: "mt-6 text-xs text-gray-600",
                "You can leave this tab open — analysis runs in your browser."
            }
        }
    }
}

fn sorted_goals_with_scores(goals: &[GoalMarkerDisplay]) -> Vec<GoalTimelineRow> {
    let mut ordered: Vec<GoalMarkerDisplay> = goals.to_vec();
    ordered.sort_by(|left, right| {
        left.time_seconds
            .partial_cmp(&right.time_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut blue_goals = 0_u32;
    let mut orange_goals = 0_u32;
    ordered
        .into_iter()
        .map(|goal| {
            match goal.team {
                Team::Blue => blue_goals += 1,
                Team::Orange => orange_goals += 1,
            }
            GoalTimelineRow {
                time_seconds: goal.time_seconds,
                team: goal.team,
                score_label: format!("{blue_goals}-{orange_goals}"),
                player_lane_index: goal.player_lane_index,
            }
        })
        .collect()
}

/// Horizontal match timeline with per-player lanes, segment boundaries, goals, and ranks.
#[component]
pub(crate) fn AnalysisTimeline(progress: ProgressState) -> Element {
    let Some(track) = progress.timeline.clone() else {
        return rsx! {};
    };

    let num_segments = track.num_segments.max(1) as f32;

    let goal_index_position = |goal_time: f32| -> f32 {
        let boundaries = &track.boundary_times_seconds;
        for seg_idx in 0..track.num_segments {
            let Some(seg_start) = boundaries.get(seg_idx).copied() else {
                continue;
            };
            let Some(seg_end) = boundaries.get(seg_idx + 1).copied() else {
                continue;
            };
            if goal_time >= seg_start && goal_time <= seg_end {
                let span = (seg_end - seg_start).max(0.001);
                let fraction = (goal_time - seg_start) / span;
                return (seg_idx as f32 + fraction) / num_segments * 100.0;
            }
        }
        0.0
    };

    let goal_entries = sorted_goals_with_scores(&track.goals);

    rsx! {
        div { class: "w-full mb-10 rounded-xl border border-gray-800 bg-gradient-to-b from-gray-900/90 to-gray-950/95 p-5 overflow-x-hidden shadow-lg shadow-black/40",
            div { class: "mb-4",
                h3 { class: "text-lg font-semibold text-gray-100 tracking-tight text-center",
                    "Smelling smurfs"
                }
                div { class: "flex flex-wrap justify-center gap-x-6 gap-y-2 mt-3 text-[11px] text-gray-400",
                    span { class: "flex items-center gap-1.5",
                        span { class: "text-amber-400 text-sm", "▲" }
                        "Goal"
                    }
                    span { class: "flex items-center gap-1.5",
                        span { class: "w-1 h-4 bg-zinc-900 ring-1 ring-zinc-600 inline-block" }
                        "Smurf-scanned segment"
                    }
                }
            }

            div { class: "flex flex-row gap-3 w-full min-w-0 items-start",
                // Player names (team tint + border; slots 0–2 blue, 3–5 orange in standard replays)
                div { class: "flex flex-col w-40 flex-shrink-0",
                    for player_lane_index in 0..TOTAL_PLAYERS {
                        {
                            let player_name = track
                                .player_names
                                .get(player_lane_index)
                                .cloned()
                                .unwrap_or_else(|| format!("Player {}", player_lane_index + 1));
                            let team = track
                                .player_teams
                                .get(player_lane_index)
                                .copied()
                                .unwrap_or(Team::Blue);
                            let lane_styles = if team == Team::Blue {
                                "border-l-4 border-l-blue-500/70 bg-blue-950/30 border-b border-blue-900/30"
                            } else {
                                "border-l-4 border-l-orange-500/70 bg-orange-950/30 border-b border-orange-900/30"
                            };
                            rsx! {
                                div {
                                    key: "tl-name-{player_lane_index}",
                                    class: "h-11 flex items-center text-xs font-medium text-gray-100 truncate pr-2 pl-2 {lane_styles}",
                                    "{player_name}"
                                }
                            }
                        }
                    }
                    // Spacer aligns name rows with the timeline bottom axis (h-11).
                    div { class: "h-11 shrink-0 border-t border-gray-800/40" }
                }

                // Match track (ends at last segment; no separate result column — summary is below)
                div { class: "flex-1 flex flex-col min-h-[336px] min-w-0",
                    div { class: "relative flex-1 flex flex-col rounded-lg bg-gray-950/80 border border-gray-800 overflow-visible min-h-[336px]",
                        // Lanes + overlays (match time uses the left MATCH_TRACK_WIDTH_PERCENT of this area)
                        div { class: "relative h-72 min-h-72 shrink-0",
                            // Full-height overlays: boundaries + goal markers
                            div { class: "absolute inset-0 pointer-events-none z-[1]",
                                for boundary_index in 0..track.boundary_times_seconds.len() {
                                    {
                                        let boundary_left = boundary_index as f32 / num_segments * 100.0;
                                        let boundary_count = track.boundary_times_seconds.len();
                                        let is_last_boundary =
                                            boundary_count > 0 && boundary_index == boundary_count - 1;
                                            && boundary_index == boundary_count - 1;
                                        let boundary_style = if is_last_boundary {
                                            format!(
                                                "left: {boundary_left:.2}%; height: 100%; transform: translateX(-100%);",
                                            )
                                        } else {
                                            format!("left: {boundary_left:.2}%; height: 100%;")
                                        }
                                        rsx! {
                                            div {
                                                key: "boundary-{boundary_index}",
                                                class: "absolute top-0 w-1.5 bg-zinc-950 ring-1 ring-zinc-700 z-[2]",
                                                style: "{boundary_style}",
                                            }
                                        }
                                    }
                                }
                                for (goal_index, entry) in goal_entries.iter().enumerate() {
                                    {
                                        let goal_left = goal_index_position(entry.time_seconds);
                                        let dash_color = if entry.team == Team::Blue {
                                            "border-blue-400"
                                        } else {
                                            "border-orange-400"
                                        };
                                        let goal_column_style =
                                            format!("left: {goal_left:.2}%; height: 100%;");
                                        let score_color = if entry.team == Team::Blue {
                                            "text-blue-300"
                                        } else {
                                            "text-orange-300"
                                        };
                                        let score_label = entry.score_label.clone();
                                        rsx! {
                                            div {
                                                key: "goal-line-{goal_index}",
                                                class: "absolute top-0 z-[3] flex flex-col items-center -translate-x-1/2",
                                                style: "{goal_column_style}",
                                                div { class: "absolute top-0 left-1/2 -translate-x-1/2 -translate-y-full pb-0.5 whitespace-nowrap text-[10px] font-mono font-bold {score_color} drop-shadow-sm text-center",
                                                    "{score_label}"
                                                }
                                                div { class: "flex-1 w-0 border-l-2 border-dashed {dash_color} opacity-80" }
                                            }
                                        }
                                    }
                                }
                            }

                            // Lanes (segment ranks); fixed row height matches name column.
                            div { class: "relative z-[5] flex flex-col h-72 shrink-0",
                                for player_lane_index in 0..TOTAL_PLAYERS {
                                    {
                                        let team = track
                                            .player_teams
                                            .get(player_lane_index)
                                            .copied()
                                            .unwrap_or(Team::Blue);
                                        let lane_bg = if team == Team::Blue {
                                            "bg-blue-500/10"
                                        } else {
                                            "bg-orange-500/10"
                                        };
                                        rsx! {
                                            div {
                                                key: "lane-{player_lane_index}",
                                                class: "relative h-11 shrink-0 overflow-visible border-b border-gray-800/60 {lane_bg}",
                                                for segment_index in 0..track.num_segments {
                                                    {
                                                        let segment_left = segment_index as f32 / num_segments * 100.0;
                                                        let segment_width = 100.0 / num_segments;
                                                        let segment_style = format!(
                                                            "left: {segment_left:.4}%; width: {segment_width:.4}%;",
                                                        );
                                                        let rank_visible = progress
                                                            .segments
                                                            .get(segment_index)
                                                            .and_then(|step| step.player_segment_ranks.as_ref())
                                                            .is_some();
                                                        let segment_rank = progress
                                                            .segments
                                                            .get(segment_index)
                                                            .and_then(|step| {
                                                                step.player_segment_ranks
                                                                    .as_ref()
                                                                    .and_then(|ranks| ranks.get(player_lane_index).copied())
                                                            });
                                                        let opacity_class = if rank_visible { "opacity-100" } else { "opacity-0" };
                                                        rsx! {
                                                            div {
                                                                key: "seg-badge-{player_lane_index}-{segment_index}",
                                                                class: "absolute inset-x-0 top-0 bottom-0 flex items-end justify-center px-0 pb-px pt-0 leading-none transition-opacity duration-500 {opacity_class}",
                                                                style: "{segment_style}",
                                                                if let Some(division) = segment_rank {
                                                                    img {
                                                                        src: rank_division_icon_asset(division),
                                                                        alt: "",
                                                                        title: division.to_string(),
                                                                        class: "block h-10 w-10 max-h-[calc(100%-2px)] shrink-0 object-contain object-bottom drop-shadow-sm",
                                                                    }
                                                                } else {
                                                                    span { class: "block h-10 w-10 max-h-[calc(100%-2px)] shrink-0" }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                for (goal_index, entry) in goal_entries.iter().enumerate() {
                                                    {
                                                        let marker_left = goal_index_position(entry.time_seconds);
                                                        let marker_style = format!(
                                                            "left: {marker_left:.2}%; top: 50%; transform: translate(-50%, -50%);",
                                                        );
                                                        let show_on_this_lane =
                                                            entry.player_lane_index == Some(player_lane_index);
                                                        if show_on_this_lane {
                                                            rsx! {
                                                                div {
                                                                    key: "goal-tri-{player_lane_index}-{goal_index}",
                                                                    class: "absolute z-[8] flex h-8 w-8 shrink-0 items-center justify-center pointer-events-none leading-none",
                                                                    style: "{marker_style}",
                                                                    span { class: "block text-2xl text-yellow-400", "▲" }
                                                                }
                                                            }
                                                        } else {
                                                            rsx! {}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Bottom axis: one time label under each segment boundary.
                        // Position uses index-based equal spacing; label text uses play-time.
                        div { class: "relative shrink-0 h-11 w-full border-t border-gray-800/80 bg-gray-950/60",
                            for boundary_index in 0..track.boundary_times_seconds.len() {
                                {
                                    let boundary_left_percent = boundary_index as f32 / num_segments * 100.0;
                                    let boundary_count = track.boundary_times_seconds.len();
                                    let is_last_boundary =
                                        boundary_count > 0 && boundary_index == boundary_count - 1;
                                        && boundary_index == boundary_count - 1;
                                    let boundary_tick_style = if is_last_boundary {
                                        format!("left: {boundary_left_percent:.2}%; transform: translateX(-100%);")
                                    } else {
                                        format!("left: {boundary_left_percent:.2}%; transform: translateX(-50%);")
                                    };
                                    let fallback_time = track
                                        .boundary_times_seconds
                                        .get(boundary_index)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let play_time = track
                                        .boundary_play_times_seconds
                                        .get(boundary_index)
                                        .copied()
                                        .unwrap_or(fallback_time);
                                    let boundary_tick_label = format_timeline_boundary_label(play_time);
                                    rsx! {
                                        div {
                                            key: "boundary-time-{boundary_index}",
                                            class: "absolute top-0 bottom-0 flex items-start justify-center pt-1.5 pointer-events-none z-[4]",
                                            style: "{boundary_tick_style}",
                                            span { class: "text-[9px] leading-none text-gray-400 font-mono tabular-nums whitespace-nowrap",
                                                "{boundary_tick_label}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
