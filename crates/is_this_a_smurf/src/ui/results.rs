//! Results tables and error view.

use dioxus::prelude::*;
use replay_structs::Team;

use crate::app_state::{AppState, PlayerAverage, PredictionResults, ProgressState};
use crate::branding::SMURF_SUSPECT_BADGE;
use crate::prediction::SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN;
use crate::rank_icon::rank_division_icon_asset;

use super::processing::AnalysisTimeline;

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

/// Results page with team summary cards and the same match timeline as the processing screen.
#[component]
pub(crate) fn ResultsPage(
    filename: String,
    results: PredictionResults,
    timeline_progress: Option<ProgressState>,
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

    rsx! {
        div { class: "max-w-7xl mx-auto px-4 py-8",
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

            if let Some(progress) = timeline_progress {
                AnalysisTimeline { progress }
            }
        }
    }
}

/// Warning triangle used to frame the smurf suspect illustration (`alert` — car — `alert`).
#[component]
fn LobbyAlertTriangleIcon() -> Element {
    rsx! {
        svg {
            class: "w-9 h-9 sm:w-10 sm:h-10 text-red-500 shrink-0 drop-shadow-md",
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
            // Player rows: name (left), smurf callout centered between name and rank icon (right).
            div { class: "divide-y divide-gray-800",
                for player in &players {
                    div { class: "px-6 py-4 flex items-center gap-3 min-h-[5.5rem]",
                        div { class: "min-w-0 flex-1 flex flex-col justify-center gap-1",
                            p { class: "font-semibold text-gray-100 break-words", "{player.name}" }
                            p { class: "text-sm text-gray-500", "{player.rank}" }
                        }
                        if player_looks_high_for_lobby(player.average_mmr, lobby_median_mmr) {
                            div { class: "flex shrink-0 items-center justify-center gap-2 sm:gap-3",
                                LobbyAlertTriangleIcon {}
                                img {
                                    src: SMURF_SUSPECT_BADGE,
                                    alt: "Predicted MMR much higher than lobby median",
                                    title: "Predicted MMR is more than 200 above the lobby median (approximate)",
                                    class: "h-14 w-auto max-w-[5rem] sm:h-[4.5rem] sm:max-w-[6rem] object-contain",
                                }
                                LobbyAlertTriangleIcon {}
                            }
                        }
                        div { class: "shrink-0 flex flex-col items-end justify-center gap-1",
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
