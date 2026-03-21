//! Upload drop zone and full inference pipeline (keeps the future alive on this scope).

use dioxus::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::app_state::{
    AnalysisTimelinePhase, AppState, LocalProcessing, ProgressState, StepStatus, TimelineTrackState,
};
use crate::browser_async::{sleep_milliseconds, yield_to_ui};
use crate::embedded_model::{DEFAULT_SEQUENCE_LENGTH, MODEL_BYTES, MODEL_CONFIG};
use crate::prediction::{
    build_goal_markers, build_prediction_results, compute_segment_boundary_times,
    global_ranks_from_predictions, prepare_players_for_timeline, ranks_from_player_predictions,
    segment_step_infos,
};
use ml_model::{ExtractedSegmentFeatures, SequenceModel, load_checkpoint_from_bytes};
use replay_parser::parse_replay_from_bytes;

use super::processing::ProcessingPage;

/// Upload page with a centered drag-and-drop area.
///
/// Processing runs as a future on **this** component's scope. We avoid
/// changing `AppState` until we have the final results, which keeps
/// `UploadPage` mounted and prevents the future from being cancelled.
#[component]
pub(crate) fn UploadPage(state: Signal<AppState>) -> Element {
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
                                state.set(AppState::Error("No file selected.".to_string()));
                                return;
                            };
                            let filename = file.name();
                            tracing::info!("[replay] reading file: {}, size: {} bytes", filename, file.size());

                            let array_buffer = match JsFuture::from(file.array_buffer()).await {
                                Ok(buffer) => buffer,
                                Err(error) => {
                                    tracing::info!("[replay] array_buffer() error: {:?}", error);
                                    state.set(AppState::Error(format!(
                                        "Could not read file: {error:?}"
                                    )));
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
                                    timeline: None,
                                },
                            }));
                            yield_to_ui().await;

                            // ---- Step 3: parse replay ------------------------------------------
                            tracing::info!("[replay] parse_replay_from_bytes starting");
                            let parsed = match parse_replay_from_bytes(&data) {
                                Ok(parsed) => parsed,
                                Err(error) => {
                                    tracing::info!("[replay] parse error: {}", error);
                                    state.set(AppState::Error(format!(
                                        "Replay parsing error: {error}"
                                    )));
                                    return;
                                }
                            };
                            tracing::info!("[replay] parse ok, {} frames", parsed.frames.len());
                            if parsed.frames.is_empty() {
                                state.set(AppState::Error(
                                    "No frames found in the replay.".to_string(),
                                ));
                                return;
                            }
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Done("Done".to_string()),
                                    loading_model: StepStatus::Processing,
                                    segments: vec![],
                                    timeline: None,
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
                                    state.set(AppState::Error(format!(
                                        "Model loading error: {error}"
                                    )));
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
                            if let Some(first_segment_step) = segment_steps.first_mut() {
                                first_segment_step.status = StepStatus::Processing;
                            }
                            let (timeline_player_names, timeline_player_teams) =
                                prepare_players_for_timeline(&parsed);
                            let goal_markers = build_goal_markers(&parsed, &timeline_player_names);
                            let match_duration_seconds = parsed
                                .frames
                                .last()
                                .map_or(1.0_f32, |frame| frame.time)
                                .max(0.001);
                            let boundary_times_seconds =
                                compute_segment_boundary_times(&segment_steps);
                            let mut timeline_snapshot = if num_segments > 0 {
                                Some(TimelineTrackState {
                                    match_duration_seconds,
                                    boundary_times_seconds,
                                    goals: goal_markers,
                                    player_names: timeline_player_names,
                                    player_teams: timeline_player_teams,
                                    phase: AnalysisTimelinePhase::InferenceInProgress,
                                    car_at_boundary_index: 0,
                                    num_segments,
                                    global_ranks: None,
                                })
                            } else {
                                None
                            };
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Done("Done".to_string()),
                                    loading_model: StepStatus::Done("Done".to_string()),
                                    segments: segment_steps.clone(),
                                    timeline: timeline_snapshot.clone(),
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
                                let segment_player_ranks =
                                    ranks_from_player_predictions(&prediction.player_predictions);
                                segment_predictions.push(prediction);
                                if let Some(completed_step) = segment_steps.get_mut(seg_idx) {
                                    completed_step.player_segment_ranks = Some(segment_player_ranks);
                                    completed_step.status =
                                        StepStatus::Done("Complete".to_string());
                                }
                                if let Some(next_step) = segment_steps.get_mut(seg_idx + 1) {
                                    next_step.status = StepStatus::Processing;
                                }
                                if let Some(track) = timeline_snapshot.as_mut() {
                                    track.car_at_boundary_index = seg_idx + 1;
                                    track.phase = AnalysisTimelinePhase::InferenceInProgress;
                                }
                                local_processing.set(Some(LocalProcessing {
                                    filename: filename.clone(),
                                    progress: ProgressState {
                                        parsing: StepStatus::Done("Done".to_string()),
                                        loading_model: StepStatus::Done("Done".to_string()),
                                        segments: segment_steps.clone(),
                                        timeline: timeline_snapshot.clone(),
                                    },
                                }));
                                yield_to_ui().await;
                            }

                            // ---- Step 6: global rank reveal, then results -----------------------
                            tracing::info!("[replay] all segments done, revealing global ranks");
                            let global_ranks_array =
                                global_ranks_from_predictions(&segment_predictions);
                            if let Some(track) = timeline_snapshot.as_mut() {
                                track.phase = AnalysisTimelinePhase::RevealingGlobalRanks;
                                track.global_ranks = Some(global_ranks_array);
                            }
                            local_processing.set(Some(LocalProcessing {
                                filename: filename.clone(),
                                progress: ProgressState {
                                    parsing: StepStatus::Done("Done".to_string()),
                                    loading_model: StepStatus::Done("Done".to_string()),
                                    segments: segment_steps.clone(),
                                    timeline: timeline_snapshot.clone(),
                                },
                            }));
                            yield_to_ui().await;
                            if timeline_snapshot.is_some() {
                                sleep_milliseconds(850).await;
                            }

                            tracing::info!("[replay] building results");
                            match build_prediction_results(&parsed, segment_predictions) {
                                Ok(results) => {
                                    state.set(AppState::ShowingResults(filename, results));
                                }
                                Err(message) => {
                                    state.set(AppState::Error(message));
                                }
                            }
                        }
                    },
                }
            }
        }
    }
}
