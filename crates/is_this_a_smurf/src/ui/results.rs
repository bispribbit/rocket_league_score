//! Results tables and error view.

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use replay_structs::{RankDivision, Team};

use crate::app_state::{AppState, PlayerAverage, PredictionResults, SegmentDisplayData};
use crate::branding::SMURF_SUSPECT_BADGE;
use crate::rank_icon::rank_division_icon_asset;

/// MMR above the lobby median before we show the playful smurf badge (predictions are approximate).
const SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN: f32 = 200.0;

fn lobby_median_average_mmr(player_averages: &[PlayerAverage]) -> f32 {
    let mut values: Vec<f32> = player_averages
        .iter()
        .map(|player| player.average_mmr)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values.get(mid).copied().unwrap_or(0.0)
    } else {
        let lower = values.get(mid.saturating_sub(1)).copied().unwrap_or(0.0);
        let upper = values.get(mid).copied().unwrap_or(0.0);
        f32::midpoint(lower, upper)
    }
}

fn player_looks_high_for_lobby(player_mmr: f32, lobby_median_mmr: f32) -> bool {
    player_mmr > lobby_median_mmr + SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN
}

/// Results page with per-segment and summary tables.
#[component]
pub(crate) fn ResultsPage(
    filename: String,
    results: PredictionResults,
    state: Signal<AppState>,
) -> Element {
    let mut state = state;

    let blue_players: Vec<PlayerAverage> = results
        .player_averages
        .iter()
        .filter(|player| player.team == Team::Blue)
        .cloned()
        .collect();
    let orange_players: Vec<PlayerAverage> = results
        .player_averages
        .iter()
        .filter(|player| player.team == Team::Orange)
        .cloned()
        .collect();

    let lobby_median_mmr = lobby_median_average_mmr(&results.player_averages);
    let player_average_mmrs: Vec<f32> = results
        .player_averages
        .iter()
        .map(|player| player.average_mmr)
        .collect();

    rsx! {
        div { class: "max-w-6xl mx-auto px-4 py-8",
            // Header
            div { class: "flex items-center justify-between mb-8",
                div {
                    h1 { class: "text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                        "Replay estimates"
                    }
                    p { class: "text-gray-400 mt-1",
                        "{filename} — {results.segments.len()} segment(s)"
                    }
                }
                button {
                    class: "px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors",
                    onclick: move |_| {
                        state.set(AppState::WaitingForUpload);
                    },
                    "Analyze another replay"
                }
            }

            // Summary tables - Blue and Orange side by side
            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
                // Blue team summary
                TeamSummaryCard {
                    team_name: "Blue Team",
                    team_color: "blue",
                    players: blue_players,
                    lobby_median_mmr,
                }
                // Orange team summary
                TeamSummaryCard {
                    team_name: "Orange Team",
                    team_color: "orange",
                    players: orange_players,
                    lobby_median_mmr,
                }
            }

            // Per-segment table
            SegmentTable {
                segments: results.segments.clone(),
                player_names: results.player_names,
                lobby_median_mmr,
                player_average_mmrs,
            }
        }
    }
}

/// Card showing team summary with player rank icons and rank names.
#[component]
fn TeamSummaryCard(
    team_name: String,
    team_color: String,
    players: Vec<PlayerAverage>,
    lobby_median_mmr: f32,
) -> Element {
    let border_color = if team_color == "blue" {
        "border-blue-500/50"
    } else {
        "border-orange-500/50"
    };
    let header_gradient = if team_color == "blue" {
        "from-blue-500/20 to-transparent"
    } else {
        "from-orange-500/20 to-transparent"
    };
    let text_accent = if team_color == "blue" {
        "text-blue-400"
    } else {
        "text-orange-400"
    };

    // Team average MMR
    let team_average_mmr = if players.is_empty() {
        0.0
    } else {
        players.iter().map(|player| player.average_mmr).sum::<f32>() / players.len() as f32
    };

    rsx! {
        div { class: "bg-gray-900 rounded-xl border {border_color} overflow-hidden",
            // Header
            div { class: "bg-gradient-to-r {header_gradient} px-6 py-4",
                div { class: "flex items-center justify-between",
                    h2 { class: "text-xl font-bold {text_accent}", "{team_name}" }
                    span { class: "text-gray-400 text-sm",
                        "Average: "
                        span { class: "font-semibold text-gray-200", "{team_average_mmr:.0}" }
                    }
                }
            }
            // Player rows
            div { class: "divide-y divide-gray-800",
                for player in &players {
                    div { class: "px-6 py-4 flex items-center justify-between",
                        div {
                            div { class: "flex items-center gap-2 flex-wrap",
                                p { class: "font-semibold text-gray-100", "{player.name}" }
                                if player_looks_high_for_lobby(player.average_mmr, lobby_median_mmr) {
                                    img {
                                        src: SMURF_SUSPECT_BADGE,
                                        alt: "Predicted MMR much higher than lobby median",
                                        title: "Predicted MMR is more than 200 above the lobby median (approximate)",
                                        class: "w-9 h-9 object-contain shrink-0",
                                    }
                                }
                            }
                            p { class: "text-sm text-gray-500", "{player.rank}" }
                        }
                        div { class: "text-right flex flex-col items-end gap-1",
                            {
                                let mmr_tooltip = format!("Approximate MMR: {:.0}", player.average_mmr);
                                rsx! {
                                    img {
                                        src: rank_division_icon_asset(player.rank),
                                        alt: format!("{}", player.rank),
                                        title: mmr_tooltip,
                                        class: "w-12 h-12 object-contain drop-shadow-md",
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

/// Table showing per-segment predictions for all players.
#[component]
fn SegmentTable(
    segments: Vec<SegmentDisplayData>,
    player_names: Vec<String>,
    lobby_median_mmr: f32,
    player_average_mmrs: Vec<f32>,
) -> Element {
    if segments.is_empty() {
        return rsx! {
            p { class: "text-gray-500 text-center py-8", "No segments available." }
        };
    }

    rsx! {
        div { class: "bg-gray-900 rounded-xl border border-gray-800 overflow-hidden",
            div { class: "px-6 py-4 border-b border-gray-800",
                h2 { class: "text-xl font-bold text-gray-100", "Predictions per segment" }
                p { class: "text-gray-500 text-sm mt-1",
                    "Predicted competitive rank for each player in each time segment (hover for approximate MMR)"
                }
            }

            div { class: "overflow-x-auto",
                table { class: "w-full text-sm",
                    thead {
                        tr { class: "bg-gray-800/50",
                            th { class: "px-4 py-3 text-left text-gray-400 font-medium", "Segment" }
                            th { class: "px-4 py-3 text-left text-gray-400 font-medium", "Time" }
                            // Blue team headers
                            for (player_index, name) in player_names.iter().enumerate().take(3) {
                                {
                                    let average_mmr = player_average_mmrs
                                        .get(player_index)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let show_smurf =
                                        player_looks_high_for_lobby(average_mmr, lobby_median_mmr);
                                    rsx! {
                                        th {
                                            key: "header-blue-{player_index}",
                                            class: "px-4 py-3 text-right text-blue-400 font-medium",
                                            div { class: "flex items-center justify-end gap-2",
                                                if show_smurf {
                                                    img {
                                                        src: SMURF_SUSPECT_BADGE,
                                                        alt: "Smurf suspect",
                                                        title: "Average predicted MMR is more than 200 above the lobby median",
                                                        class: "w-7 h-7 object-contain shrink-0",
                                                    }
                                                }
                                                span { "{name}" }
                                            }
                                        }
                                    }
                                }
                            }
                            // Orange team headers
                            for (player_index, name) in player_names.iter().enumerate().skip(3).take(3) {
                                {
                                    let average_mmr = player_average_mmrs
                                        .get(player_index)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let show_smurf =
                                        player_looks_high_for_lobby(average_mmr, lobby_median_mmr);
                                    rsx! {
                                        th {
                                            key: "header-orange-{player_index}",
                                            class: "px-4 py-3 text-right text-orange-400 font-medium",
                                            div { class: "flex items-center justify-end gap-2",
                                                if show_smurf {
                                                    img {
                                                        src: SMURF_SUSPECT_BADGE,
                                                        alt: "Smurf suspect",
                                                        title: "Average predicted MMR is more than 200 above the lobby median",
                                                        class: "w-7 h-7 object-contain shrink-0",
                                                    }
                                                }
                                                span { "{name}" }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    tbody { class: "divide-y divide-gray-800/50",
                        for segment in &segments {
                            tr {
                                key: "seg-{segment.segment_number}",
                                class: "hover:bg-gray-800/30 transition-colors",
                                td { class: "px-4 py-3 font-medium text-gray-300",
                                    "#{segment.segment_number}"
                                }
                                td { class: "px-4 py-3 text-gray-500",
                                    "{segment.start_time:.1}s - {segment.end_time:.1}s"
                                }
                                // Blue team values
                                for player_index in 0..3 {
                                    {
                                        let predicted_mmr = segment.player_mmr.get(player_index).copied().unwrap_or(0.0);
                                        let division = RankDivision::from(predicted_mmr);
                                        let icon = rank_division_icon_asset(division);
                                        let rank_label = format!("{division}");
                                        let mmr_tooltip = format!("Approximate MMR: {predicted_mmr:.0}");
                                        rsx! {
                                            td {
                                                key: "seg-{segment.segment_number}-blue-{player_index}",
                                                class: "px-4 py-3 text-right text-blue-300",
                                                img {
                                                    src: icon,
                                                    alt: rank_label,
                                                    title: mmr_tooltip,
                                                    class: "w-10 h-10 object-contain inline-block align-middle",
                                                }
                                            }
                                        }
                                    }
                                }
                                // Orange team values
                                for player_index in 3..TOTAL_PLAYERS {
                                    {
                                        let predicted_mmr = segment.player_mmr.get(player_index).copied().unwrap_or(0.0);
                                        let division = RankDivision::from(predicted_mmr);
                                        let icon = rank_division_icon_asset(division);
                                        let rank_label = format!("{division}");
                                        let mmr_tooltip = format!("Approximate MMR: {predicted_mmr:.0}");
                                        rsx! {
                                            td {
                                                key: "seg-{segment.segment_number}-orange-{player_index}",
                                                class: "px-4 py-3 text-right text-orange-300",
                                                img {
                                                    src: icon,
                                                    alt: rank_label,
                                                    title: mmr_tooltip,
                                                    class: "w-10 h-10 object-contain inline-block align-middle",
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
}

/// Error page with a message and a retry button.
#[component]
pub(crate) fn ErrorPage(message: String, state: Signal<AppState>) -> Element {
    let mut state = state;

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4",
            // Error icon
            svg {
                class: "w-20 h-20 text-red-500 mb-6",
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
            h2 { class: "text-2xl font-bold text-red-400 mb-4", "Error" }
            p { class: "text-gray-400 text-center max-w-md mb-8", "{message}" }
            button {
                class: "px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors",
                onclick: move |_| {
                    state.set(AppState::WaitingForUpload);
                },
                "Try again"
            }
        }
    }
}
