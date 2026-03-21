//! Processing screen: timeline animation and per-step status list.

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use replay_structs::Team;

use crate::app_state::{AnalysisTimelinePhase, ProgressState, StepStatus};
use crate::prediction::format_time_mm_ss;

/// Horizontal match timeline with per-player lanes, segment boundaries, goals, cars, and ranks.
#[component]
fn AnalysisTimeline(progress: ProgressState) -> Element {
    const MATCH_TRACK_WIDTH_PERCENT: f32 = 85.0;

    let Some(track) = progress.timeline.clone() else {
        return rsx! {};
    };

    let duration_seconds = track.match_duration_seconds.max(0.001);
    let boundary_len = track.boundary_times_seconds.len();
    let safe_boundary_index = if boundary_len == 0 {
        0
    } else {
        track
            .car_at_boundary_index
            .min(boundary_len.saturating_sub(1))
    };
    let car_time_seconds = track
        .boundary_times_seconds
        .get(safe_boundary_index)
        .copied()
        .unwrap_or(0.0);
    let car_left_percent = match track.phase {
        AnalysisTimelinePhase::InferenceInProgress => {
            (car_time_seconds / duration_seconds) * MATCH_TRACK_WIDTH_PERCENT
        }
        AnalysisTimelinePhase::RevealingGlobalRanks => MATCH_TRACK_WIDTH_PERCENT + 5.0_f32,
    };
    let car_position_style =
        format!("left: {car_left_percent:.2}%; transition: left 0.55s ease-out;");

    let end_label = format_time_mm_ss(duration_seconds);
    let show_global_column = matches!(track.phase, AnalysisTimelinePhase::RevealingGlobalRanks)
        && track.global_ranks.is_some();

    rsx! {
        div { class: "w-full mb-10 rounded-xl border border-gray-800 bg-gray-900/60 p-4 overflow-x-auto",
            p { class: "text-gray-500 text-xs mb-3",
                "▲ Goal (scorer lane) — Black bars: segment boundaries — Cars advance after each segment."
            }

            div { class: "flex flex-row gap-2 min-w-[720px]",
                // Player names column
                div { class: "flex flex-col w-36 flex-shrink-0 pt-6",
                    for player_lane_index in 0..TOTAL_PLAYERS {
                        {
                            let player_name = track
                                .player_names
                                .get(player_lane_index)
                                .cloned()
                                .unwrap_or_else(|| format!("Player {}", player_lane_index + 1));
                            rsx! {
                                div {
                                    key: "tl-name-{player_lane_index}",
                                    class: "h-12 flex items-center text-xs font-medium text-gray-300 truncate pr-2 border-b border-gray-800/80",
                                    "{player_name}"
                                }
                            }
                        }
                    }
                }

                // Match track + global column
                div { class: "flex-1 flex flex-row gap-1 min-h-[288px]",
                    div { class: "relative flex-1 rounded-lg bg-gray-950/80 border border-gray-800 overflow-visible",
                        // Axis labels
                        div { class: "absolute -top-5 left-0 text-[10px] text-gray-500", "00:00" }
                        div { class: "absolute -top-5 right-[15%] text-[10px] text-gray-500", "{end_label}" }

                        // Full-height overlays: boundaries + goal markers
                        div { class: "absolute inset-0 pointer-events-none z-[1]",
                            for (boundary_index, boundary_time) in track
                                .boundary_times_seconds
                                .iter()
                                .copied()
                                .enumerate()
                            {
                                {
                                    let boundary_left =
                                        (boundary_time / duration_seconds) * MATCH_TRACK_WIDTH_PERCENT;
                                    let boundary_style =
                                        format!("left: {boundary_left:.2}%; height: 100%;");
                                    rsx! {
                                        div {
                                            key: "boundary-{boundary_index}",
                                            class: "absolute top-0 w-1 bg-black z-[2]",
                                            style: "{boundary_style}",
                                        }
                                    }
                                }
                            }
                            for (goal_index, goal) in track.goals.iter().enumerate() {
                                {
                                    let goal_left = (goal.time_seconds / duration_seconds)
                                        * MATCH_TRACK_WIDTH_PERCENT;
                                    let dash_color = if goal.team == Team::Blue {
                                        "border-blue-400"
                                    } else {
                                        "border-orange-400"
                                    };
                                    let goal_line_style =
                                        format!("left: {goal_left:.2}%; height: 100%;");
                                    rsx! {
                                        div {
                                            key: "goal-line-{goal_index}",
                                            class: "absolute top-0 border-l border-dashed {dash_color} opacity-70 z-[1]",
                                            style: "{goal_line_style}",
                                        }
                                    }
                                }
                            }
                        }

                        // Lanes (segment ranks + cars)
                        div { class: "relative z-[5] flex flex-col h-full pt-1",
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
                                    let car_fill = if team == Team::Blue {
                                        "#60a5fa"
                                    } else {
                                        "#fb923c"
                                    };
                                    rsx! {
                                        div {
                                            key: "lane-{player_lane_index}",
                                            class: "relative flex-1 min-h-[44px] border-b border-gray-800/60 {lane_bg}",
                                            for segment_index in 0..track.num_segments {
                                                {
                                                    let (segment_left, segment_width) = match (
                                                        track.boundary_times_seconds.get(segment_index),
                                                        track.boundary_times_seconds.get(segment_index + 1),
                                                    ) {
                                                        (Some(start_seconds), Some(end_seconds)) => (
                                                            (*start_seconds / duration_seconds)
                                                                * MATCH_TRACK_WIDTH_PERCENT,
                                                            ((*end_seconds - *start_seconds) / duration_seconds)
                                                                * MATCH_TRACK_WIDTH_PERCENT,
                                                        ),
                                                        _ => (0.0_f32, 0.0_f32),
                                                    };
                                                    let segment_style = format!(
                                                        "left: {segment_left:.2}%; width: {segment_width:.2}%;",
                                                    );
                                                    let rank_visible = progress
                                                        .segments
                                                        .get(segment_index)
                                                        .and_then(|step| step.player_segment_ranks.as_ref())
                                                        .is_some();
                                                    let rank_text = progress
                                                        .segments
                                                        .get(segment_index)
                                                        .and_then(|step| {
                                                            step.player_segment_ranks.as_ref().and_then(
                                                                |ranks| {
                                                                    ranks
                                                                        .get(player_lane_index)
                                                                        .map(|rank| format!("{rank}"))
                                                                },
                                                            )
                                                        })
                                                        .unwrap_or_default();
                                                    let opacity_class = if rank_visible {
                                                        "opacity-100"
                                                    } else {
                                                        "opacity-0"
                                                    };
                                                    rsx! {
                                                        div {
                                                            key: "seg-badge-{player_lane_index}-{segment_index}",
                                                            class: "absolute bottom-0.5 flex justify-center pointer-events-none transition-opacity duration-500 {opacity_class}",
                                                            style: "{segment_style}",
                                                            span { class: "text-[9px] leading-tight px-1 py-0.5 rounded bg-gray-800/95 text-gray-200 border border-gray-700 max-w-full truncate",
                                                                "{rank_text}"
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            for (goal_index, goal) in track.goals.iter().enumerate() {
                                                {
                                                    let marker_left = (goal.time_seconds / duration_seconds)
                                                        * MATCH_TRACK_WIDTH_PERCENT;
                                                    let marker_style =
                                                        format!("left: {marker_left:.2}%; bottom: 2px;");
                                                    let show_on_this_lane =
                                                        goal.player_lane_index == Some(player_lane_index);
                                                    if show_on_this_lane {
                                                        rsx! {
                                                            div {
                                                                key: "goal-tri-{player_lane_index}-{goal_index}",
                                                                class: "absolute text-[10px] leading-none -translate-x-1/2 z-[8]",
                                                                style: "{marker_style}",
                                                                span { class: "text-yellow-400", "▲" }
                                                            }
                                                        }
                                                    } else {
                                                        rsx! {}
                                                    }
                                                }
                                            }
                                            div {
                                                class: "absolute bottom-0 z-[10] w-8 h-6 -translate-x-1/2 transition-transform duration-500",
                                                style: "{car_position_style}",
                                                svg {
                                                    view_box: "0 0 32 24",
                                                    class: "w-8 h-6 drop-shadow-md",
                                                    polygon {
                                                        points: "16,2 28,20 4,20",
                                                        fill: "{car_fill}",
                                                        stroke: "#1f2937",
                                                        stroke_width: "1",
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Global rank column
                    div { class: "w-24 flex-shrink-0 flex flex-col justify-center border border-gray-800 rounded-lg bg-gray-900/90 px-1 py-2",
                        if show_global_column {
                            for player_lane_index in 0..TOTAL_PLAYERS {
                                {
                                    let global_label = track
                                        .global_ranks
                                        .and_then(|ranks| {
                                            ranks
                                                .get(player_lane_index)
                                                .map(|rank| format!("{rank}"))
                                        })
                                        .unwrap_or_default();
                                    let reveal_class = "opacity-100 transition-opacity duration-700";
                                    rsx! {
                                        div {
                                            key: "global-rank-{player_lane_index}",
                                            class: "h-12 flex items-center justify-center px-0.5 border-b border-gray-800/60 last:border-b-0",
                                            span { class: "text-[8px] text-center leading-tight text-amber-200 font-medium {reveal_class}",
                                                "{global_label}"
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            div { class: "flex-1 flex items-center justify-center text-[9px] text-gray-600 text-center px-1",
                                "Global rank"
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Loading / processing indicator with step-by-step progression.
#[component]
pub(crate) fn ProcessingPage(filename: String, progress: ProgressState) -> Element {
    rsx! {
        div { class: "flex flex-col min-h-screen px-4 py-8 w-full max-w-6xl mx-auto",
            h2 { class: "text-2xl font-semibold mb-2 text-center", "Analyzing..." }
            p { class: "text-gray-400 mb-6 text-center",
                "File: "
                span { class: "text-gray-200 font-medium", "{filename}" }
            }

            AnalysisTimeline { progress: progress.clone() }

            div { class: "w-full max-w-md mx-auto space-y-2",
                // Parsing
                div { class: "flex items-center gap-3 py-2 px-4 rounded-lg bg-gray-900/80",
                    StepStatusIcon { status: progress.parsing.clone() }
                    span { class: "text-gray-300", "Parsing replay" }
                    StepStatusLabel { status: progress.parsing.clone() }
                }
                // Loading model
                div { class: "flex items-center gap-3 py-2 px-4 rounded-lg bg-gray-900/80",
                    StepStatusIcon { status: progress.loading_model.clone() }
                    span { class: "text-gray-300", "Loading model" }
                    StepStatusLabel { status: progress.loading_model.clone() }
                }
                // Segments
                for (index, segment) in progress.segments.iter().enumerate() {
                    div {
                        key: "seg-{index}",
                        class: "flex items-center gap-3 py-2 px-4 rounded-lg bg-gray-900/80",
                        StepStatusIcon { status: segment.status.clone() }
                        span { class: "text-gray-300",
                            "Segment {segment.start_time:.0}s to {segment.end_time:.0}s"
                        }
                        StepStatusLabel { status: segment.status.clone() }
                    }
                }
            }
        }
    }
}

/// Icon for a step: spinner (processing), check (done), or empty (pending).
#[component]
fn StepStatusIcon(status: StepStatus) -> Element {
    match status {
        StepStatus::Pending => rsx! {
            span { class: "w-5 h-5 rounded-full border-2 border-gray-600 flex-shrink-0" }
        },
        StepStatus::Processing => rsx! {
            div { class: "w-5 h-5 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin flex-shrink-0" }
        },
        StepStatus::Done(_) => rsx! {
            svg {
                class: "w-5 h-5 text-green-500 flex-shrink-0",
                fill: "none",
                stroke: "currentColor",
                stroke_width: "2",
                view_box: "0 0 24 24",
                path {
                    stroke_linecap: "round",
                    stroke_linejoin: "round",
                    d: "M5 13l4 4L19 7",
                }
            }
        },
    }
}

/// Label for a step: "Done", rank name, or "To process" / empty.
#[component]
fn StepStatusLabel(status: StepStatus) -> Element {
    match status {
        StepStatus::Pending => rsx! {
            span { class: "ml-auto text-gray-500 text-sm", "To process" }
        },
        StepStatus::Processing => rsx! {
            span { class: "ml-auto text-blue-400 text-sm font-medium", "Processing..." }
        },
        StepStatus::Done(label) => rsx! {
            span { class: "ml-auto text-green-400 font-medium", "{label}" }
        },
    }
}
