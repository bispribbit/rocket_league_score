//! Application state, progress tracking, and result payloads.

use feature_extractor::TOTAL_PLAYERS;
use replay_structs::{RankDivision, Team, UnsupportedReplayMatch};

/// Prediction results for the entire replay.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PredictionResults {
    /// Player names (6 players).
    pub(crate) player_names: Vec<String>,
    /// Team for each player (6 entries).
    pub(crate) player_teams: Vec<Team>,
    /// Per-segment predictions.
    pub(crate) segments: Vec<SegmentDisplayData>,
    /// Final averaged results per player.
    pub(crate) player_averages: Vec<PlayerAverage>,
}

/// Display data for a single segment.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SegmentDisplayData {
    /// One-based segment number.
    pub(crate) segment_number: usize,
    /// Start time in seconds.
    pub(crate) start_time: f32,
    /// End time in seconds.
    pub(crate) end_time: f32,
    /// Predicted MMR for each player (6 values).
    pub(crate) player_mmr: [f32; TOTAL_PLAYERS],
}

/// Averaged result for a single player.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlayerAverage {
    /// Player name.
    pub(crate) name: String,
    /// Player team.
    pub(crate) team: Team,
    /// Average predicted MMR across all segments.
    pub(crate) average_mmr: f32,
    /// Rank derived from average MMR.
    pub(crate) rank: RankDivision,
}

/// Status of a single step in the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StepStatus {
    Pending,
    Processing,
    Done(String),
}

/// Progress for one segment (time range + status).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SegmentStepInfo {
    pub(crate) start_time: f32,
    pub(crate) end_time: f32,
    pub(crate) status: StepStatus,
    /// Filled when the segment inference step is complete.
    pub(crate) player_segment_ranks: Option<[RankDivision; TOTAL_PLAYERS]>,
}

/// One goal shown on the analysis timeline (replay-derived).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GoalMarkerDisplay {
    pub(crate) time_seconds: f32,
    pub(crate) scorer_name: String,
    pub(crate) team: Team,
    pub(crate) player_lane_index: Option<usize>,
}

/// Whether cars are advancing per segment or moving to the global rank column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnalysisTimelinePhase {
    InferenceInProgress,
    RevealingGlobalRanks,
}

/// State for the animated match timeline during processing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TimelineTrackState {
    pub(crate) match_duration_seconds: f32,
    pub(crate) boundary_times_seconds: Vec<f32>,
    pub(crate) goals: Vec<GoalMarkerDisplay>,
    pub(crate) player_names: Vec<String>,
    pub(crate) player_teams: Vec<Team>,
    pub(crate) phase: AnalysisTimelinePhase,
    pub(crate) num_segments: usize,
    pub(crate) global_ranks: Option<[RankDivision; TOTAL_PLAYERS]>,
}

/// Live progress during analysis (parsing, model load, segments).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProgressState {
    /// Reading file bytes from the browser file API (can take a moment for large replays).
    pub(crate) reading_file: StepStatus,
    /// Copying the `ArrayBuffer` into a Rust `Vec` (synchronous; often the slowest step after the browser read).
    pub(crate) copying_into_memory: StepStatus,
    pub(crate) parsing: StepStatus,
    pub(crate) loading_model: StepStatus,
    pub(crate) segments: Vec<SegmentStepInfo>,
    /// Present while inferring segments (and briefly for the global rank reveal).
    pub(crate) timeline: Option<TimelineTrackState>,
}

/// The different states the application can be in.
#[derive(Debug, Clone)]
pub(crate) enum AppState {
    /// Waiting for the user to upload a replay file (processing also
    /// happens while in this state, with progress shown via a local signal
    /// inside `UploadPage`).
    WaitingForUpload,
    /// An error occurred (error message).
    Error(String),
    /// Replay parsed but the match type is not supported (e.g. not ranked 3v3 standard).
    UnsupportedReplay(UnsupportedReplayMatch),
}

/// Local processing state kept inside `UploadPage` so the component stays
/// mounted (and the async future stays alive) during the entire pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalProcessing {
    /// Name of the file being processed.
    pub(crate) filename: String,
    /// Current progress.
    pub(crate) progress: ProgressState,
    /// Filled when inference completes; keeps the timeline and summary on one screen without routing.
    pub(crate) results: Option<PredictionResults>,
}
