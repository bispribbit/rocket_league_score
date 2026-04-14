//! Upload drop zone and full inference pipeline (keeps the future alive on this scope).
//!
//! ## Threading and WASM
//!
//! The default browser target `wasm32-unknown-unknown` does **not** support Rust `std::thread`.
//! True parallel threads in WASM need either:
//! - **Web Workers**: a separate JS worker loading another WASM module (or the same bundle with
//!   careful setup), with all inputs and outputs passed through `postMessage` (often large copies),
//!   or
//! - **Atomics + `wasm-bindgen-rayon`**: rebuild with `+atomics,+bulk-memory`, and serve the page
//!   with cross-origin isolation (`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`)
//!   so `SharedArrayBuffer` is available.
//!
//! This crate therefore runs inference on the **same thread** as the UI, but the async loop calls
//! [`crate::browser_async::yield_to_ui`] after each segment so the browser can repaint between
//! steps. Moving compute to a worker would be a larger architectural change (model + tensors in the
//! worker, progress messages back to the main thread).

#[cfg(target_arch = "wasm32")]
use burn::backend::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use burn::backend::Wgpu;
#[cfg(target_arch = "wasm32")]
use burn::backend::ndarray::NdArrayDevice;
#[cfg(not(target_arch = "wasm32"))]
use burn::backend::wgpu::WgpuDevice;
use dioxus::prelude::*;
use ml_model::{ExtractedSegmentFeatures, SequenceModel, load_checkpoint_from_bytes};
use replay_parser::{ReplayAcceptanceError, parse_replay_from_bytes};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::processing::{AnalysisTimeline, EarlyWorkingPanel};
use super::results::{MatchVerdictBanner, PlayerSummaryGrid, PlayerSummaryGridLoading};
use crate::app_state::{
    AnalysisTimelinePhase, AppState, LocalProcessing, ProgressState, StepStatus, TimelineTrackState,
};
use crate::branding::IS_THIS_A_SMURF_HERO;
use crate::browser_async::{sleep_milliseconds, yield_for_dom_paint, yield_to_ui};
use crate::embedded_model::{MODEL_BYTES, MODEL_CONFIG, sequence_length_from_embedded_config};
use crate::prediction::{
    build_goal_markers, build_prediction_results, compute_segment_boundary_play_times,
    compute_segment_boundary_times, global_ranks_from_predictions, prepare_players_for_timeline,
    ranks_from_player_predictions, segment_step_infos,
};

/// Burn backend for inference: WGPU on native hosts, pure CPU ndarray on WASM (WGPU checkpoint
/// load and sync tensor reads panic on WASM — see cubecl-common `reader`).
#[cfg(target_arch = "wasm32")]
type InferenceBackend = NdArray;
#[cfg(target_arch = "wasm32")]
type InferenceDevice = NdArrayDevice;

#[cfg(not(target_arch = "wasm32"))]
type InferenceBackend = Wgpu;
#[cfg(not(target_arch = "wasm32"))]
type InferenceDevice = WgpuDevice;

/// Upload page with a centered drag-and-drop area.
///
/// Processing runs as a future on **this** component's scope. On success we
/// keep `AppState::WaitingForUpload` and store the outcome in
/// [`LocalProcessing`] so the timeline and summary stay on one screen without
/// routing. Errors use [`AppState::Error`]; unsupported match types use
/// [`AppState::UnsupportedReplay`].
#[component]
pub(crate) fn UploadPage(state: Signal<AppState>) -> Element {
    let mut state = state;
    let mut local_processing = use_signal(|| None::<LocalProcessing>);

    if let Some(LocalProcessing {
        filename,
        progress,
        results,
    }) = local_processing()
    {
        let show_timeline = progress.timeline.is_some();
        return rsx! {
            div { class: "flex flex-col min-h-screen w-full bg-gray-950 text-gray-100",
                div { class: "max-w-7xl mx-auto px-4 py-8 w-full flex flex-col gap-8",
                    div { class: "flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4",
                        div {
                            h1 { class: "text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                                "Is there a smurf in this match?"
                            }
                            p { class: "text-gray-400 mt-1", "{filename}" }
                        }
                        if results.is_some() {
                            button {
                                class: "px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors self-start sm:self-center shrink-0",
                                onclick: move |_| {
                                    local_processing.set(None);
                                },
                                "Analyze another replay"
                            }
                        }
                    }
                    if show_timeline {
                        AnalysisTimeline { progress: progress.clone() }
                    } else {
                        EarlyWorkingPanel { progress: progress.clone() }
                    }
                    if let Some(prediction_results) = results {
                        PlayerSummaryGrid { results: prediction_results.clone() }
                        MatchVerdictBanner { results: prediction_results }
                    } else {
                        PlayerSummaryGridLoading { progress }
                    }
                }
            }
        };
    }

    rsx! {
        div { class: "flex flex-col items-center justify-center min-h-screen px-4 py-10 gap-6",
            h1 { class: "w-full max-w-lg text-center text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-orange-400",
                "Is this a smurf?"
            }
            // Hero art + blurb share the same width and surface treatment as the upload zone below.
            div { class: "w-full max-w-lg rounded-2xl border border-gray-700/60 bg-gradient-to-b from-gray-900/80 to-gray-950/90 p-6 shadow-xl shadow-black/40",
                div { class: "flex justify-center rounded-xl bg-gray-950/50 p-3 ring-1 ring-inset ring-gray-800/80",
                    img {
                        src: IS_THIS_A_SMURF_HERO,
                        alt: "Cartoon Rocket League cars and rainbow banner art",
                        class: "max-h-[min(42vh,28rem)] w-auto max-w-full object-contain rounded-lg",
                    }
                }
                p { class: "mt-4 text-center text-base leading-relaxed text-gray-400",
                    "Upload a Rocket League replay. We guess everyone's rank, then expose who's committed to the smurf lifestyle."
                }
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
                p { class: "text-gray-500 text-sm mt-1", "Ranked 3v3 Rocket League replay file" }

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

                        spawn(async move {
                            // ---- Step 1: read file bytes via web-sys (no JS eval, no base64) --
                            let Some(file) = file else {
                                tracing::info!("[replay] no file found on input element");
                                state.set(AppState::Error("No file selected.".to_string()));
                                return;
                            };
                            let filename = file.name();
                            tracing::info!(
                                "[replay] reading file: {}, size: {} bytes", filename, file.size()
                            );
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Processing,
                                            copying_into_memory: StepStatus::Pending,
                                            parsing: StepStatus::Pending,
                                            loading_model: StepStatus::Pending,
                                            segments: vec![],
                                            timeline: None,
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            let array_buffer = match JsFuture::from(file.array_buffer()).await {
                                Ok(buffer) => buffer,
                                Err(error) => {
                                    tracing::info!("[replay] array_buffer() error: {:?}", error);
                                    state
                                        .set(AppState::Error(format!("Could not read file: {error:?}")));
                                    return;
                                }
                            };
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Done("Done".to_string()),
                                            copying_into_memory: StepStatus::Processing,
                                            parsing: StepStatus::Pending,
                                            loading_model: StepStatus::Pending,
                                            segments: vec![],
                                            timeline: None,
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            yield_for_dom_paint().await;
                            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
                            let data: Vec<u8> = uint8_array.to_vec();
                            tracing::info!("[replay] file read ok, {} bytes", data.len());
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Done("Done".to_string()),
                                            copying_into_memory: StepStatus::Done("Done".to_string()),
                                            parsing: StepStatus::Processing,
                                            loading_model: StepStatus::Pending,
                                            segments: vec![],
                                            timeline: None,
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            yield_for_dom_paint().await;
                            tracing::info!("[replay] parse_replay_from_bytes starting");
                            let parsed = match parse_replay_from_bytes(&data) {
                                Ok(parsed) => parsed,
                                Err(ReplayAcceptanceError::Unsupported(details)) => {
                                    tracing::info!(
                                        "[replay] unsupported match: {}", details.detected_mode_label
                                    );
                                    state.set(AppState::UnsupportedReplay(details));
                                    return;
                                }
                                Err(ReplayAcceptanceError::Parse(error)) => {
                                    tracing::info!("[replay] parse error: {}", error);
                                    state.set(AppState::Error(format!("Replay parsing error: {error}")));
                                    return;
                                }
                            };
                            tracing::info!("[replay] parse ok, {} frames", parsed.frames.len());
                            if parsed.frames.is_empty() {
                                state.set(AppState::Error("No frames found in the replay.".to_string()));
                                return;
                            }
                            let sequence_length = sequence_length_from_embedded_config();
                            let extracted = ExtractedSegmentFeatures::from_frames(&parsed.frames);
                            let num_segments = extracted.segment_count(sequence_length);
                            let mut segment_steps = segment_step_infos(
                                &parsed.frames,
                                sequence_length,
                                num_segments,
                            );
                            let (timeline_player_names, timeline_player_teams) = prepare_players_for_timeline(
                                &parsed,
                            );
                            let goal_markers = build_goal_markers(&parsed, &timeline_player_names);
                            let match_duration_seconds = parsed
                                .frames
                                .last()
                                .map_or(1.0_f32, |frame| frame.time)
                                .max(0.001);
                            let boundary_times_seconds = compute_segment_boundary_times(&segment_steps);
                            let boundary_play_times_seconds = compute_segment_boundary_play_times(
                                &parsed.frames,
                                &segment_steps,
                                sequence_length,
                            );
                            let mut timeline_snapshot = if num_segments > 0 {
                                let timeline_snapshot = TimelineTrackState {
                                    match_duration_seconds,
                                    boundary_times_seconds,
                                    boundary_play_times_seconds,
                                    goals: goal_markers,
                                    player_names: timeline_player_names,
                                    player_teams: timeline_player_teams,
                                    phase: AnalysisTimelinePhase::InferenceInProgress,
                                    num_segments,
                                    global_ranks: None,
                                };
                                tracing::info!("[replay] building results");
                                Some(timeline_snapshot)
                            } else {
                                None
                            };
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Done("Done".to_string()),
                                            copying_into_memory: StepStatus::Done("Done".to_string()),
                                            parsing: StepStatus::Done("Done".to_string()),
                                            loading_model: StepStatus::Processing,
                                            segments: segment_steps.clone(),
                                            timeline: timeline_snapshot.clone(),
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            yield_for_dom_paint().await;
                            tracing::info!(
                                "[replay] load_checkpoint_from_bytes starting (backend = {})", if
                                cfg!(target_arch = "wasm32") { "NdArray" } else { "Wgpu" }
                            );
                            let device = InferenceDevice::default();
                            let model: SequenceModel<InferenceBackend> = match load_checkpoint_from_bytes(
                                MODEL_BYTES,
                                MODEL_CONFIG,
                                &device,
                            ) {
                                Ok(model) => {
                                    tracing::info!("[replay] model loaded ok");
                                    model
                                }
                                Err(error) => {
                                    tracing::info!("[replay] model load error: {}", error);
                                    state.set(AppState::Error(format!("Model loading error: {error}")));
                                    return;
                                }
                            };
                            tracing::info!("[replay] {} segments, starting inference", num_segments);
                            if let Some(first_segment_step) = segment_steps.first_mut() {
                                first_segment_step.status = StepStatus::Processing;
                            }
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Done("Done".to_string()),
                                            copying_into_memory: StepStatus::Done("Done".to_string()),
                                            parsing: StepStatus::Done("Done".to_string()),
                                            loading_model: StepStatus::Done("Done".to_string()),
                                            segments: segment_steps.clone(),
                                            timeline: timeline_snapshot.clone(),
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            yield_for_dom_paint().await;
                            let mut segment_predictions = Vec::with_capacity(num_segments);
                            for seg_idx in 0..num_segments {
                                let Some(prediction) = extracted
                                    .predict_single_segment(&model, &device, sequence_length, seg_idx)
                                    .await else {
                                    break;
                                };
                                let segment_player_ranks = ranks_from_player_predictions(
                                    &prediction.player_predictions,
                                );
                                segment_predictions.push(prediction);
                                if let Some(completed_step) = segment_steps.get_mut(seg_idx) {
                                    completed_step.player_segment_ranks = Some(segment_player_ranks);
                                    completed_step.status = StepStatus::Done("Complete".to_string());
                                }
                                if let Some(next_step) = segment_steps.get_mut(seg_idx + 1) {
                                    next_step.status = StepStatus::Processing;
                                }
                                local_processing
                                    .set(
                                        Some(LocalProcessing {
                                            filename: filename.clone(),
                                            progress: ProgressState {
                                                reading_file: StepStatus::Done("Done".to_string()),
                                                copying_into_memory: StepStatus::Done("Done".to_string()),
                                                parsing: StepStatus::Done("Done".to_string()),
                                                loading_model: StepStatus::Done("Done".to_string()),
                                                segments: segment_steps.clone(),
                                                timeline: timeline_snapshot.clone(),
                                            },
                                            results: None,
                                        }),
                                    );
                                yield_to_ui().await;
                            }
                            tracing::info!("[replay] all segments done, revealing global ranks");
                            let global_ranks_array = global_ranks_from_predictions(&segment_predictions);
                            if let Some(track) = timeline_snapshot.as_mut() {
                                track.phase = AnalysisTimelinePhase::RevealingGlobalRanks;
                                track.global_ranks = Some(global_ranks_array);
                            }
                            local_processing
                                .set(
                                    Some(LocalProcessing {
                                        filename: filename.clone(),
                                        progress: ProgressState {
                                            reading_file: StepStatus::Done("Done".to_string()),
                                            copying_into_memory: StepStatus::Done("Done".to_string()),
                                            parsing: StepStatus::Done("Done".to_string()),
                                            loading_model: StepStatus::Done("Done".to_string()),
                                            segments: segment_steps.clone(),
                                            timeline: timeline_snapshot.clone(),
                                        },
                                        results: None,
                                    }),
                                );
                            yield_to_ui().await;
                            if timeline_snapshot.is_some() {
                                sleep_milliseconds(850).await;
                            }
                            tracing::info!("[replay] building results");
                            match build_prediction_results(&parsed, segment_predictions) {
                                Ok(results) => {
                                    local_processing
                                        .set(
                                            Some(LocalProcessing {
                                                filename,
                                                progress: ProgressState {
                                                    reading_file: StepStatus::Done("Done".to_string()),
                                                    copying_into_memory: StepStatus::Done("Done".to_string()),
                                                    parsing: StepStatus::Done("Done".to_string()),
                                                    loading_model: StepStatus::Done("Done".to_string()),
                                                    segments: segment_steps.clone(),
                                                    timeline: timeline_snapshot.clone(),
                                                },
                                                results: Some(results),
                                            }),
                                        );
                                }
                                Err(message) => {
                                    state.set(AppState::Error(message));
                                }
                            }
                        });
                    },
                }
            }
        }
    }
}
