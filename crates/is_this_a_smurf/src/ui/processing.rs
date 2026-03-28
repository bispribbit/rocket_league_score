//! Processing screen: timeline animation and per-step status list.

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use replay_structs::Team;

use crate::app_state::{AnalysisTimelinePhase, GoalMarkerDisplay, ProgressState, StepStatus};
use crate::branding::SMURF_SUSPECT_BADGE;
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
fn EarlyWorkingPanel(progress: ProgressState) -> Element {
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

/// Compact warning triangle for the smurf parking callout at the end of each lane.
#[component]
fn LaneEndAlertTriangleIcon() -> Element {
    rsx! {
        svg {
            class: "w-3.5 h-3.5 text-red-500 shrink-0 drop-shadow",
            fill: "none",
            stroke: "currentColor",
            stroke_width: "1.5",
            view_box: "0 0 24 24",
            path {
                stroke_linecap: "round",
                stroke_linejoin: "round",
                d: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z",
            }
        }
    }
}

fn timeline_lane_car_graphic(car_fill: &'static str) -> Element {
    rsx! {
        svg {
            view_box: "0 0 36 28",
            class: "w-9 h-7 shrink-0 drop-shadow-lg filter",
            rect {
                x: "6",
                y: "8",
                width: "24",
                height: "14",
                rx: "3",
                fill: "{car_fill}",
                stroke: "#0f172a",
                stroke_width: "1.2",
            }
            rect {
                x: "11",
                y: "10",
                width: "14",
                height: "6",
                rx: "1",
                fill: "#1e293b",
                opacity: "0.35",
            }
        }
    }
}

/// Car icon at the end of the lane while segments run, or Yes/No smurf callout after analysis.
#[component]
fn TimelineLaneParkingEnd(
    show_smurf_parking_answers: bool,
    lane_smurf_suspect: Option<bool>,
    car_fill: &'static str,
) -> Element {
    if !show_smurf_parking_answers {
        return timeline_lane_car_graphic(car_fill);
    }

    match lane_smurf_suspect {
        Some(true) => rsx! {
            div { class: "flex items-center justify-center gap-0.5",
                LaneEndAlertTriangleIcon {}
                img {
                    src: SMURF_SUSPECT_BADGE,
                    alt: "Smurf suspect — predicted MMR much higher than lobby median",
                    title: "Predicted MMR is more than 200 above the lobby median (approximate)",
                    class: "h-8 w-auto max-h-9 max-w-[2.1rem] object-contain drop-shadow-md",
                }
                LaneEndAlertTriangleIcon {}
            }
            span { class: "text-[9px] font-bold leading-none text-amber-200/95", "Yes" }
        },
        Some(false) => rsx! {
            {timeline_lane_car_graphic(car_fill)}
            span { class: "text-[9px] font-semibold leading-none text-gray-400", "No" }
        },
        None => timeline_lane_car_graphic(car_fill),
    }
}

/// Horizontal match timeline with per-player lanes, segment boundaries, goals, cars, and ranks.
#[component]
pub(crate) fn AnalysisTimeline(progress: ProgressState) -> Element {
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

    let show_global_column = matches!(track.phase, AnalysisTimelinePhase::RevealingGlobalRanks)
        && track.global_ranks.is_some();
    let show_smurf_parking_answers =
        matches!(track.phase, AnalysisTimelinePhase::RevealingGlobalRanks)
            && track.global_ranks.is_some()
            && track.smurf_suspect_by_player.is_some();

    let smurf_heading_band_style = format!("left: {MATCH_TRACK_WIDTH_PERCENT:.2}%; right: 0;");

    let goal_entries = sorted_goals_with_scores(&track.goals);

    rsx! {
        div { class: "w-full mb-10 rounded-xl border border-gray-800 bg-gradient-to-b from-gray-900/90 to-gray-950/95 p-5 overflow-x-auto shadow-lg shadow-black/40",
            div { class: "mb-4",
                h3 { class: "text-lg font-semibold text-gray-100 tracking-tight text-center",
                    "Match analysis timeline"
                }
                div { class: "flex flex-wrap justify-center gap-x-6 gap-y-2 mt-3 text-[11px] text-gray-400",
                    span { class: "flex items-center gap-1.5",
                        span { class: "text-amber-400 text-sm", "▲" }
                        "Goal (scorer lane)"
                    }
                    span { class: "flex items-center gap-1.5",
                        span { class: "w-1 h-4 bg-zinc-900 ring-1 ring-zinc-600 inline-block" }
                        "Analyzed segment boundary"
                    }
                    span { class: "flex flex-wrap items-center justify-center gap-x-1.5 gap-y-1 max-w-lg text-center",
                        span { class: "text-gray-500", "Lane end —" }
                        span { class: "font-medium text-gray-300", "Is this a smurf?" }
                        span { class: "text-gray-500", ":" }
                        span { class: "text-gray-400",
                            "No "
                            span { class: "text-gray-600", "(car)" }
                        }
                        span { class: "text-gray-500", "·" }
                        span { class: "text-amber-200/90",
                            "Yes "
                            span { class: "text-amber-200/50", "(alerts + badge)" }
                        }
                    }
                }
            }

            div { class: "flex flex-row gap-3 min-w-[760px] items-start",
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
                                    class: "h-12 flex items-center text-xs font-medium text-gray-100 truncate pr-2 pl-2 {lane_styles}",
                                    "{player_name}"
                                }
                            }
                        }
                    }
                    // Spacer aligns name rows with the timeline bottom axis (h-12).
                    div { class: "h-12 shrink-0 border-t border-gray-800/40" }
                }

                // Match track + global column
                div { class: "flex-1 flex flex-row gap-1 min-h-[336px]",
                    div { class: "relative flex-1 flex flex-col rounded-lg bg-gray-950/80 border border-gray-800 overflow-visible min-h-[336px]",
                        // Lane end title (aligned with parking position when ranks are revealed)
                        if show_smurf_parking_answers {
                            div {
                                class: "absolute z-[12] top-1 flex h-5 items-center justify-center pointer-events-none",
                                style: "{smurf_heading_band_style}",
                                span { class: "text-[9px] font-semibold text-gray-400 tracking-wide text-center px-1",
                                    "Is this a smurf?"
                                }
                            }
                        }
                        // Lanes + overlays (match time uses the left MATCH_TRACK_WIDTH_PERCENT of this area)
                        div { class: "relative h-72 min-h-72 shrink-0",
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
                                            class: "absolute top-0 w-1.5 bg-zinc-950 ring-1 ring-zinc-700 z-[2]",
                                            style: "{boundary_style}",
                                        }
                                    }
                                }
                            }
                            for (goal_index, entry) in goal_entries.iter().enumerate() {
                                {
                                    let goal_left = (entry.time_seconds / duration_seconds)
                                        * MATCH_TRACK_WIDTH_PERCENT;
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
                                            div { class: "absolute -top-6 whitespace-nowrap text-[10px] font-mono font-bold {score_color} drop-shadow-sm",
                                                "{score_label}"
                                            }
                                            div {
                                                class: "flex-1 w-0 border-l-2 border-dashed {dash_color} opacity-80",
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Lanes (segment ranks + cars); fixed row height matches name column.
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
                                    let car_fill = if team == Team::Blue {
                                        "#60a5fa"
                                    } else {
                                        "#fb923c"
                                    };
                                    let lane_smurf_suspect: Option<bool> = track
                                        .smurf_suspect_by_player
                                        .as_ref()
                                        .and_then(|flags| flags.get(player_lane_index).copied());
                                    rsx! {
                                        div {
                                            key: "lane-{player_lane_index}",
                                            class: "relative h-12 shrink-0 overflow-visible border-b border-gray-800/60 {lane_bg}",
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
                                                    let segment_rank = progress.segments.get(segment_index).and_then(
                                                        |step| {
                                                            step.player_segment_ranks
                                                                .as_ref()
                                                                .and_then(|ranks| ranks.get(player_lane_index).copied())
                                                        },
                                                    );
                                                    let opacity_class = if rank_visible {
                                                        "opacity-100"
                                                    } else {
                                                        "opacity-0"
                                                    };
                                                    rsx! {
                                                        div {
                                                            key: "seg-badge-{player_lane_index}-{segment_index}",
                                                            class: "absolute inset-y-0 flex items-center justify-center pointer-events-none transition-opacity duration-500 {opacity_class}",
                                                            style: "{segment_style}",
                                                            if let Some(division) = segment_rank {
                                                                img {
                                                                    src: rank_division_icon_asset(division),
                                                                    alt: format!("{division}"),
                                                                    class: "h-7 w-7 max-h-[min(1.75rem,80%)] max-w-[min(1.75rem,90%)] object-contain drop-shadow-sm",
                                                                }
                                                            } else {
                                                                span { class: "h-7 w-7 max-h-[min(1.75rem,80%)] max-w-[min(1.75rem,90%)] inline-block shrink-0" }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            for (goal_index, entry) in goal_entries.iter().enumerate() {
                                                {
                                                    let marker_left = (entry.time_seconds / duration_seconds)
                                                        * MATCH_TRACK_WIDTH_PERCENT;
                                                    let marker_style =
                                                        format!("left: {marker_left:.2}%; bottom: 2px;");
                                                    let show_on_this_lane =
                                                        entry.player_lane_index == Some(player_lane_index);
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
                                                class: "absolute bottom-0 z-[10] flex max-w-[5.75rem] -translate-x-1/2 flex-col items-center justify-end gap-0.5 pb-0.5 transition-all duration-500 ease-out pointer-events-none",
                                                style: "{car_position_style}",
                                                TimelineLaneParkingEnd {
                                                    show_smurf_parking_answers,
                                                    lane_smurf_suspect,
                                                    car_fill,
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        }

                        // Bottom axis: one time label under each segment boundary (same horizontal scale as bars)
                        div { class: "relative shrink-0 h-12 w-full border-t border-gray-800/80 bg-gray-950/60",
                            for (boundary_index, boundary_time_seconds) in track
                                .boundary_times_seconds
                                .iter()
                                .copied()
                                .enumerate()
                            {
                                {
                                    let boundary_left_percent = (boundary_time_seconds / duration_seconds)
                                        * MATCH_TRACK_WIDTH_PERCENT;
                                    let boundary_tick_style = format!(
                                        "left: {boundary_left_percent:.2}%; transform: translateX(-50%);",
                                    );
                                    let boundary_tick_label =
                                        format_timeline_boundary_label(boundary_time_seconds);
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

                    // Global rank column (bottom spacer lines up with the timeline axis row)
                    div { class: "w-[5.5rem] flex-shrink-0 flex flex-col justify-start self-stretch min-h-[336px] border border-amber-900/40 rounded-lg bg-gray-900/95 px-1.5 py-2 shadow-inner",
                        if show_global_column {
                            for player_lane_index in 0..TOTAL_PLAYERS {
                                {
                                    let global_division = track
                                        .global_ranks
                                        .as_ref()
                                        .and_then(|ranks| ranks.get(player_lane_index).copied());
                                    let reveal_class = "opacity-100 transition-opacity duration-700";
                                    rsx! {
                                        div {
                                            key: "global-rank-{player_lane_index}",
                                            class: "h-12 flex shrink-0 items-center justify-center px-0.5 border-b border-gray-800/60 last:border-b-0",
                                            if let Some(division) = global_division {
                                                img {
                                                    src: rank_division_icon_asset(division),
                                                    alt: format!("{division}"),
                                                    class: "h-9 w-9 object-contain drop-shadow-md {reveal_class}",
                                                }
                                            } else {
                                                span { class: "text-[9px] text-center text-amber-100/40 {reveal_class}", "—" }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            div { class: "flex min-h-[288px] flex-1 flex-col items-center justify-center text-[9px] text-gray-600 text-center px-1",
                                "Global rank"
                            }
                        }
                        div { class: "h-12 shrink-0" }
                    }
                }
            }
        }
    }
}

/// Loading / processing indicator with step-by-step progression.
#[component]
pub(crate) fn ProcessingPage(filename: String, progress: ProgressState) -> Element {
    let show_timeline = progress.timeline.is_some();
    rsx! {
        div { class: "min-h-screen px-4 py-8 w-full max-w-7xl mx-auto",
            h2 { class: "text-2xl font-semibold mb-2 text-center", "Processing your replay" }
            p { class: "text-gray-400 mb-8 text-center",
                "File: "
                span { class: "text-gray-200 font-medium", "{filename}" }
            }

            div { class: "grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(240px,280px)] gap-8 xl:gap-10 items-start",
                if show_timeline {
                    AnalysisTimeline { progress: progress.clone() }
                } else {
                    EarlyWorkingPanel { progress: progress.clone() }
                }

                div { class: "space-y-2 xl:sticky xl:top-6 rounded-xl border border-gray-800/80 bg-gray-950/40 p-3",
                    p { class: "text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2", "Pipeline" }
                // Reading file
                div { class: "flex items-center gap-3 py-2 px-4 rounded-lg bg-gray-900/80",
                    StepStatusIcon { status: progress.reading_file.clone() }
                    span { class: "text-gray-300", "Reading file" }
                    StepStatusLabel { status: progress.reading_file.clone() }
                }
                div { class: "flex items-center gap-3 py-2 px-4 rounded-lg bg-gray-900/80",
                    StepStatusIcon { status: progress.copying_into_memory.clone() }
                    span { class: "text-gray-300", "Copying into memory" }
                    StepStatusLabel { status: progress.copying_into_memory.clone() }
                }
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
