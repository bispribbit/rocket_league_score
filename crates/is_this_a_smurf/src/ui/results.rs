//! Results tables and error view.

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use replay_structs::{Team, UnsupportedReplayMatch};

use rand::{Rng, rng};

use crate::app_state::{AppState, PlayerAverage, PredictionResults, ProgressState};
use crate::branding::SMURF_SUSPECT_BADGE;
use crate::prediction::SMURF_SUSPICION_MMR_ABOVE_LOBBY_MEDIAN;
use crate::rank_icon::rank_division_icon_asset;

const PREDICTED_RANK_ROAST_LABELS: [&str; 10] = [
    "Estimated Delusion Bracket",
    "Alleged Skill Bracket",
    "Court-Ordered Rank Guessed",
    "Projected Excuse Division",
    "Certified Bad Decision Tier",
    "Approximate Main Character Level",
    "Likely Quick Chat Skill Class",
    "Estimated Blame Allocation Rank",
    "Audited Ball-Chasing Bracket",
    "Professional Opinion Nobody Asked For",
];

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

fn predicted_rank_roast_label() -> &'static str {
    let mut rng = rng();
    let index = rng.random_range(0..PREDICTED_RANK_ROAST_LABELS.len());
    PREDICTED_RANK_ROAST_LABELS
        .get(index)
        .copied()
        .unwrap_or("Estimated Delusion Bracket")
}

/// Blue and orange team summary cards for a completed prediction.
#[component]
pub(crate) fn PlayerSummaryGrid(results: PredictionResults) -> Element {
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
        div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
            TeamSummaryCard {
                team_name: "Blue Team",
                team_color: "blue",
                loading: false,
                players_complete: blue_players,
                player_names_loading: vec![],
                lobby_median_mmr,
            }
            TeamSummaryCard {
                team_name: "Orange Team",
                team_color: "orange",
                loading: false,
                players_complete: orange_players,
                player_names_loading: vec![],
                lobby_median_mmr,
            }
        }
    }
}

/// One-line verdict under the team summary: tier-flavored if no smurf, or names suspects.
#[component]
pub(crate) fn MatchVerdictBanner(results: PredictionResults) -> Element {
    let line = crate::verdict_copy::match_verdict_paragraph(&results);
    rsx! {
        div { class: "rounded-xl border border-gray-800 bg-gray-900/80 px-5 py-4 mb-10",
            p { class: "text-center text-base leading-relaxed text-gray-200",
                "{line}"
            }
        }
    }
}

/// Same layout as [`PlayerSummaryGrid`], but team averages and rank icons stay hidden until results exist.
#[component]
pub(crate) fn PlayerSummaryGridLoading(progress: ProgressState) -> Element {
    let Some(track) = progress.timeline.as_ref() else {
        return rsx! {
            div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
                TeamSummaryCard {
                    team_name: "Blue Team",
                    team_color: "blue",
                    loading: true,
                    players_complete: vec![],
                    player_names_loading: vec![],
                    lobby_median_mmr: 0.0,
                }
                TeamSummaryCard {
                    team_name: "Orange Team",
                    team_color: "orange",
                    loading: true,
                    players_complete: vec![],
                    player_names_loading: vec![],
                    lobby_median_mmr: 0.0,
                }
            }
        };
    };

    let mut blue_names: Vec<String> = Vec::new();
    let mut orange_names: Vec<String> = Vec::new();
    for player_lane_index in 0..TOTAL_PLAYERS {
        let player_name = track
            .player_names
            .get(player_lane_index)
            .cloned()
            .unwrap_or_else(|| format!("Player {}", player_lane_index + 1));
        match track
            .player_teams
            .get(player_lane_index)
            .copied()
            .unwrap_or(Team::Blue)
        {
            Team::Blue => blue_names.push(player_name),
            Team::Orange => orange_names.push(player_name),
        }
    }

    rsx! {
        div { class: "grid grid-cols-1 md:grid-cols-2 gap-6 mb-10",
            TeamSummaryCard {
                team_name: "Blue Team",
                team_color: "blue",
                loading: true,
                players_complete: vec![],
                player_names_loading: blue_names,
                lobby_median_mmr: 0.0,
            }
            TeamSummaryCard {
                team_name: "Orange Team",
                team_color: "orange",
                loading: true,
                players_complete: vec![],
                player_names_loading: orange_names,
                lobby_median_mmr: 0.0,
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
    loading: bool,
    players_complete: Vec<PlayerAverage>,
    player_names_loading: Vec<String>,
    lobby_median_mmr: f32,
) -> Element {
    let roast_label = predicted_rank_roast_label();
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

    let _team_average_mmr = if players_complete.is_empty() {
        0.0
    } else {
        players_complete
            .iter()
            .map(|player| player.average_mmr)
            .sum::<f32>()
            / players_complete.len() as f32
    };

    rsx! {
        div { class: "bg-gray-900 rounded-xl border {border_color} overflow-hidden",
            div { class: "bg-gradient-to-r {header_gradient} px-6 py-4",
                div { class: "flex items-center justify-between gap-3",
                    h2 { class: "text-xl font-bold {text_accent}", "{team_name}" }
                    if loading {
                        span { class: "text-gray-500 text-sm font-medium", "Calculating…" }
                    } else {
                        span { "{roast_label}"
                        }
                    }
                }
            }
            div { class: "divide-y divide-gray-800",
                if loading {
                    if player_names_loading.is_empty() {
                        p { class: "px-6 py-6 text-sm text-gray-500 text-center",
                            "Waiting for replay data…"
                        }
                    } else {
                        for name in player_names_loading {
                            div { class: "px-6 py-4 flex items-center min-h-[5.5rem]",
                                p { class: "font-semibold text-gray-100 break-words", "{name}" }
                            }
                        }
                    }
                } else {
                    for player in &players_complete {
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
                                        alt: "Predicted rank is much higher than lobby",
                                        title: "Predicted rank is much higher than lobby",
                                        class: "h-14 w-auto max-w-[5rem] sm:h-[4.5rem] sm:max-w-[6rem] object-contain",
                                    }
                                    LobbyAlertTriangleIcon {}
                                }
                            }
                            div { class: "shrink-0 flex flex-col items-end justify-center gap-1",
                                img {
                                    src: rank_division_icon_asset(player.rank),
                                    alt: "",
                                    title: player.rank.to_string(),
                                    class: "block w-12 h-11 object-contain drop-shadow-md",
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Shown when the replay is valid but not ranked standard 3v3.
#[component]
pub(crate) fn UnsupportedReplayPage(
    details: UnsupportedReplayMatch,
    state: Signal<AppState>,
) -> Element {
    let mut state = state;
    let message = details.user_message();

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4",
            svg {
                class: "w-20 h-20 text-amber-500 mb-6",
                fill: "none",
                stroke: "currentColor",
                stroke_width: "1.5",
                view_box: "0 0 24 24",
                path {
                    stroke_linecap: "round",
                    stroke_linejoin: "round",
                    d: "M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z",
                }
            }
            h2 { class: "text-2xl font-bold text-amber-400 mb-4", "Unsupported match type" }
            p { class: "text-gray-400 text-center max-w-md mb-8", "{message}" }
            button {
                class: "px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-colors",
                onclick: move |_| {
                    state.set(AppState::WaitingForUpload);
                },
                "Try another replay"
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
