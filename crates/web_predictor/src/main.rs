#![allow(clippy::same_name_method)]
//! Dioxus web application for Rocket League replay MMR prediction.
//!
//! Upload a `.replay` file and get per-player MMR predictions displayed
//! in per-segment and summary tables.

use dioxus::prelude::*;
use feature_extractor::TOTAL_PLAYERS;
use ml_model::{
    ExtractedSegmentFeatures, SegmentPrediction, SequenceModel, load_checkpoint_from_bytes,
};
use replay_parser::parse_replay_from_bytes;
use replay_structs::{RankDivision, Team};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

// ---------------------------------------------------------------------------
// Embedded model
// ---------------------------------------------------------------------------

/// Model weights in binary format (saved via `save_checkpoint_bin`).
///
static MODEL_BYTES: &[u8] = include_bytes!("../../../data/v5.mpk");

/// Model training config JSON (for architecture dimensions).
///
/// Place your config at `crates/web_predictor/model/checkpoint.config.json`.
static MODEL_CONFIG: &str = include_str!("../../../data/v5.config.json");

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default sequence length for inference (should match training config).
const DEFAULT_SEQUENCE_LENGTH: usize = 300;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Prediction results for the entire replay.
#[derive(Debug, Clone, PartialEq)]
struct PredictionResults {
    /// Player names (6 players).
    player_names: Vec<String>,
    /// Team for each player (6 entries).
    player_teams: Vec<Team>,
    /// Per-segment predictions.
    segments: Vec<SegmentDisplayData>,
    /// Final averaged results per player.
    player_averages: Vec<PlayerAverage>,
}

/// Display data for a single segment.
#[derive(Debug, Clone, PartialEq)]
struct SegmentDisplayData {
    /// One-based segment number.
    segment_number: usize,
    /// Start time in seconds.
    start_time: f32,
    /// End time in seconds.
    end_time: f32,
    /// Predicted MMR for each player (6 values).
    player_mmr: [f32; TOTAL_PLAYERS],
}

/// Averaged result for a single player.
#[derive(Debug, Clone, PartialEq)]
struct PlayerAverage {
    /// Player name.
    name: String,
    /// Player team.
    team: Team,
    /// Average predicted MMR across all segments.
    average_mmr: f32,
    /// Rank derived from average MMR.
    rank: RankDivision,
}

// ---------------------------------------------------------------------------
// Prediction logic
// ---------------------------------------------------------------------------

/// Yields to the browser event loop so the UI can repaint.
#[expect(clippy::future_not_send, clippy::let_underscore_untyped)]
async fn yield_to_ui() {
    let _ = document::eval("new Promise(r => setTimeout(r, 0))").await;
}

/// Builds segment step infos (time ranges) from parsed frames and sequence length.
fn segment_step_infos(
    frames: &[replay_structs::GameFrame],
    sequence_length: usize,
    num_segments: usize,
) -> Vec<SegmentStepInfo> {
    (0..num_segments)
        .map(|seg_idx| {
            let start_frame = seg_idx * sequence_length;
            let end_frame = (start_frame + sequence_length).min(frames.len());
            let start_time = frames.get(start_frame).map_or(0.0, |frame| frame.time);
            let end_time = frames
                .get(end_frame.saturating_sub(1))
                .map_or(0.0, |frame| frame.time);
            SegmentStepInfo {
                start_time,
                end_time,
                status: StepStatus::Pending,
            }
        })
        .collect()
}

/// Builds full prediction results from parsed data and segment predictions.
fn build_prediction_results(
    parsed: &replay_structs::ParsedReplay,
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
            let sum: f32 = segment_predictions
                .iter()
                .map(|segment| {
                    segment
                        .player_predictions
                        .get(player_index)
                        .copied()
                        .unwrap_or(0.0)
                })
                .sum();
            let count = segment_predictions.len().max(1) as f32;
            let average_mmr = sum / count;
            let name = player_names
                .get(player_index)
                .cloned()
                .unwrap_or_else(|| format!("Player {}", player_index + 1));
            let team = player_teams
                .get(player_index)
                .copied()
                .unwrap_or(Team::Blue);
            let rank = RankDivision::from(average_mmr);
            PlayerAverage {
                name,
                team,
                average_mmr,
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

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

/// Status of a single step in the pipeline.
#[derive(Debug, Clone, PartialEq)]
enum StepStatus {
    Pending,
    Processing,
    Done(String),
}

/// Progress for one segment (time range + status).
#[derive(Debug, Clone, PartialEq)]
struct SegmentStepInfo {
    start_time: f32,
    end_time: f32,
    status: StepStatus,
}

/// Live progress during analysis (parsing, model load, segments).
#[derive(Debug, Clone, PartialEq)]
struct ProgressState {
    parsing: StepStatus,
    loading_model: StepStatus,
    segments: Vec<SegmentStepInfo>,
}

/// The different states the application can be in.
#[derive(Debug, Clone)]
enum AppState {
    /// Waiting for the user to upload a replay file (processing also
    /// happens while in this state, with progress shown via a local signal
    /// inside `UploadPage`).
    WaitingForUpload,
    /// Displaying prediction results.
    ShowingResults {
        /// Name of the analyzed file.
        filename: String,
        /// Prediction results.
        results: PredictionResults,
    },
    /// An error occurred.
    Error {
        /// Error description.
        message: String,
    },
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

fn main() {
    // Initialize logger so tracing::info! / debug! appear in the browser console (web).
    dioxus::logger::init(tracing::Level::DEBUG).expect("failed to init dioxus logger");
    dioxus::launch(App);
}

/// Root application component.
#[component]
#[expect(clippy::volatile_composites)] // from dioxus asset! macro
fn App() -> Element {
    let state = use_signal(|| AppState::WaitingForUpload);

    rsx! {
        document::Stylesheet { href: asset!("/assets/tailwind.css") }

        div { class: "min-h-screen bg-gray-950 text-gray-100",
            match state() {
                AppState::WaitingForUpload => rsx! {
                    UploadPage { state }
                },
                AppState::ShowingResults { filename, results } => rsx! {
                    ResultsPage { filename, results, state }
                },
                AppState::Error { message } => rsx! {
                    ErrorPage { message, state }
                },
            }
        }
    }
}

/// Local processing state kept inside `UploadPage` so the component stays
/// mounted (and the async future stays alive) during the entire pipeline.
#[derive(Debug, Clone, PartialEq)]
struct LocalProcessing {
    /// Name of the file being processed.
    filename: String,
    /// Current progress.
    progress: ProgressState,
}

/// Upload page with a centered drag-and-drop area.
///
/// Processing runs as a future on **this** component's scope. We avoid
/// changing `AppState` until we have the final results, which keeps
/// `UploadPage` mounted and prevents the future from being cancelled.
#[component]
fn UploadPage(state: Signal<AppState>) -> Element {
    let mut state = state;
    let mut local_processing = use_signal(|| None::<LocalProcessing>);

    // If we are processing, render the progress page from here.
    if let Some(processing) = local_processing() {
        return rsx! {
            ProcessingPage {
                filename: processing.filename,
                progress: processing.progress,
            }
        };
    }

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4",
            // Title
            h1 { class: "text-4xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                "RL Replay Predictor"
            }
            p { class: "text-gray-400 mb-10 text-lg",
                "Predict each player's MMR from a Rocket League replay"
            }

            // Upload zone
            label { class: "relative flex flex-col items-center justify-center w-full max-w-lg h-64 border-2 border-dashed border-gray-600 rounded-2xl cursor-pointer hover:border-blue-500 hover:bg-gray-900/50 transition-all duration-300",
                // Icon
                svg {
                    class: "w-16 h-16 mb-4 text-gray-500",
                    fill: "none",
                    stroke: "currentColor",
                    stroke_width: "1.5",
                    view_box: "0 0 24 24",
                    path {
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        d: "M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5",
                    }
                }
                p { class: "text-gray-400 text-lg font-medium",
                    "Click or drop a "
                    span { class: "text-blue-400", ".replay" }
                }
                p { class: "text-gray-500 text-sm mt-1",
                    "Rocket League replay file"
                }

                input {
                    id: "replay-file-input",
                    r#type: "file",
                    accept: ".replay",
                    class: "absolute inset-0 w-full h-full opacity-0 cursor-pointer",
                    onchange: move |_event: Event<FormData>| {
                        // Grab the File object from the DOM before entering async.
                        let file = web_sys::window()
                            .and_then(|window| window.document())
                            .and_then(|document| document.get_element_by_id("replay-file-input"))
                            .and_then(|element| element.dyn_into::<web_sys::HtmlInputElement>().ok())
                            .and_then(|input| input.files())
                            .and_then(|file_list| file_list.get(0));
                        tracing::info!("[replay] onchange: file selected");

                        async move {
                            // ---- Step 1: read file bytes via web-sys (no JS eval, no base64) --
                            let Some(file) = file else {
                                tracing::info!("[replay] no file found on input element");
                                state.set(AppState::Error {
                                    message: "No file selected.".to_string(),
                                });
                                return;
                            };
                            let filename = file.name();
                            tracing::info!("[replay] reading file: {}, size: {} bytes", filename, file.size());

                            let array_buffer = match JsFuture::from(file.array_buffer()).await {
                                Ok(buffer) => buffer,
                                Err(error) => {
                                    tracing::info!("[replay] array_buffer() error: {:?}", error);
                                    state.set(AppState::Error {
                                        message: format!("Could not read file: {error:?}"),
                                    });
                                    return;
                                }
                            };
                            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
                            let data: Vec<u8> = uint8_array.to_vec();
                            tracing::info!("[replay] file read ok, {} bytes", data.len());
                            tracing::info!("[replay] file read ok, {} bytes", data.len());

                            // ---- Step 2: show progress (local signal, UploadPage stays mounted) ----
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Processing,
                                    loading_model: StepStatus::Pending,
                                    segments: vec![],
                                },
                            }));
                            yield_to_ui().await;

                            // ---- Step 3: parse replay ------------------------------------------
                            tracing::info!("[replay] parse_replay_from_bytes starting");
                            let parsed = match parse_replay_from_bytes(&data) {
                                Ok(parsed) => parsed,
                                Err(error) => {
                                    tracing::info!("[replay] parse error: {}", error);
                                    state.set(AppState::Error {
                                        message: format!("Replay parsing error: {error}"),
                                    });
                                    return;
                                }
                            };
                            tracing::info!("[replay] parse ok, {} frames", parsed.frames.len());
                            if parsed.frames.is_empty() {
                                state.set(AppState::Error {
                                    message: "No frames found in the replay.".to_string(),
                                });
                                return;
                            }
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Done("Done".to_string()),
                                    loading_model: StepStatus::Processing,
                                    segments: vec![],
                                },
                            }));
                            yield_to_ui().await;

                            // ---- Step 4: load model --------------------------------------------
                            tracing::info!("[replay] load_checkpoint_from_bytes starting");
                            let device = burn::backend::wgpu::WgpuDevice::default();
                            type Backend = burn::backend::Wgpu;
                            let model: SequenceModel<Backend> = match load_checkpoint_from_bytes(
                                MODEL_BYTES, MODEL_CONFIG, &device,
                            ) {
                                Ok(model) => {
                                    tracing::info!("[replay] model loaded ok");
                                    model
                                }
                                Err(error) => {
                                    tracing::info!("[replay] model load error: {}", error);
                                    state.set(AppState::Error {
                                        message: format!("Model loading error: {error}"),
                                    });
                                    return;
                                }
                            };
                            let sequence_length = if MODEL_CONFIG.is_empty() {
                                DEFAULT_SEQUENCE_LENGTH
                            } else {
                                serde_json::from_str::<serde_json::Value>(MODEL_CONFIG)
                                    .ok()
                                    .and_then(|config_value| {
                                        config_value.get("sequence_length")?.as_u64()
                                    })
                                    .map_or(DEFAULT_SEQUENCE_LENGTH, |length_value| {
                                        length_value as usize
                                    })
                            };

                            // ---- Step 5: extract features & run inference per segment -----------
                            let extracted = ExtractedSegmentFeatures::from_frames(&parsed.frames);
                            let num_segments = extracted.segment_count(sequence_length);
                            tracing::info!("[replay] {} segments, starting inference", num_segments);
                            let mut segment_steps =
                                segment_step_infos(&parsed.frames, sequence_length, num_segments);
                            if let Some(first) = segment_steps.first_mut() {
                                first.status = StepStatus::Processing;
                            }
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Done("Done".to_string()),
                                    loading_model: StepStatus::Done("Done".to_string()),
                                    segments: segment_steps.clone(),
                                },
                            }));
                            yield_to_ui().await;

                            let mut segment_predictions = Vec::with_capacity(num_segments);
                            for seg_idx in 0..num_segments {
                                let Some(prediction) = extracted.predict_single_segment(
                                    &model,
                                    &device,
                                    sequence_length,
                                    seg_idx,
                                ) else {
                                    break;
                                };
                                let avg_mmr = prediction.player_predictions.iter().sum::<f32>()
                                    / prediction.player_predictions.len() as f32;
                                let rank_label = format!("{}", RankDivision::from(avg_mmr));
                                segment_predictions.push(prediction);
                                if let Some(step) = segment_steps.get_mut(seg_idx) {
                                    step.status = StepStatus::Done(rank_label);
                                }
                                if let Some(next) = segment_steps.get_mut(seg_idx + 1) {
                                    next.status = StepStatus::Processing;
                                }
                                local_processing.set(Some(LocalProcessing {
                                    filename: filename.clone(),
                                    progress: ProgressState {
                                        parsing: StepStatus::Done("Done".to_string()),
                                        loading_model: StepStatus::Done("Done".to_string()),
                                        segments: segment_steps.clone(),
                                    },
                                }));
                                yield_to_ui().await;
                            }

                            // ---- Step 6: done – set app state (UploadPage may now unmount) -----
                            tracing::info!("[replay] all segments done, building results");
                            match build_prediction_results(&parsed, segment_predictions) {
                                Ok(results) => {
                                    state.set(AppState::ShowingResults {
                                        filename,
                                        results,
                                    });
                                }
                                Err(message) => {
                                    state.set(AppState::Error { message });
                                }
                            }
                        }
                    },
                }
            }
        }
    }
}

/// Loading / processing indicator with step-by-step progression.
#[component]
fn ProcessingPage(filename: String, progress: ProgressState) -> Element {
    rsx! {
        div { class: "flex flex-col items-center min-h-screen px-4 py-12",
            h2 { class: "text-2xl font-semibold mb-2", "Analyzing..." }
            p { class: "text-gray-400 mb-8",
                "File: "
                span { class: "text-gray-200 font-medium", "{filename}" }
            }

            div { class: "w-full max-w-md space-y-2",
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

/// Results page with per-segment and summary tables.
#[component]
fn ResultsPage(filename: String, results: PredictionResults, state: Signal<AppState>) -> Element {
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

    rsx! {
        div { class: "max-w-6xl mx-auto px-4 py-8",
            // Header
            div { class: "flex items-center justify-between mb-8",
                div {
                    h1 { class: "text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                        "Prediction results"
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
                TeamSummaryCard { team_name: "Blue Team", team_color: "blue", players: blue_players }
                // Orange team summary
                TeamSummaryCard { team_name: "Orange Team", team_color: "orange", players: orange_players }
            }

            // Per-segment table
            SegmentTable {
                segments: results.segments.clone(),
                player_names: results.player_names,
            }
        }
    }
}

/// Card showing team summary with player MMR and rank.
#[component]
fn TeamSummaryCard(team_name: String, team_color: String, players: Vec<PlayerAverage>) -> Element {
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
                            p { class: "font-semibold text-gray-100", "{player.name}" }
                            p { class: "text-sm text-gray-500", "{player.rank}" }
                        }
                        div { class: "text-right",
                            p { class: "text-2xl font-bold {text_accent}",
                                "{player.average_mmr:.0}"
                            }
                            p { class: "text-xs text-gray-500", "MMR" }
                        }
                    }
                }
            }
        }
    }
}

/// Table showing per-segment predictions for all players.
#[component]
fn SegmentTable(segments: Vec<SegmentDisplayData>, player_names: Vec<String>) -> Element {
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
                    "Predicted MMR for each player in each time segment"
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
                                th {
                                    key: "header-blue-{player_index}",
                                    class: "px-4 py-3 text-right text-blue-400 font-medium",
                                    "{name}"
                                }
                            }
                            // Orange team headers
                            for (player_index, name) in player_names.iter().enumerate().skip(3).take(3) {
                                th {
                                    key: "header-orange-{player_index}",
                                    class: "px-4 py-3 text-right text-orange-400 font-medium",
                                    "{name}"
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
                                        rsx! {
                                            td {
                                                key: "seg-{segment.segment_number}-blue-{player_index}",
                                                class: "px-4 py-3 text-right font-mono text-blue-300",
                                                "{predicted_mmr:.0}"
                                            }
                                        }
                                    }
                                }
                                // Orange team values
                                for player_index in 3..TOTAL_PLAYERS {
                                    {
                                        let predicted_mmr = segment.player_mmr.get(player_index).copied().unwrap_or(0.0);
                                        rsx! {
                                            td {
                                                key: "seg-{segment.segment_number}-orange-{player_index}",
                                                class: "px-4 py-3 text-right font-mono text-orange-300",
                                                "{predicted_mmr:.0}"
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
fn ErrorPage(message: String, state: Signal<AppState>) -> Element {
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
