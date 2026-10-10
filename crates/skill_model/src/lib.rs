//! Per-player skill model shipped in the "Is this a smurf?" app.
//!
//! Burn-free on purpose: everything here runs unchanged in the WASM app. A parsed replay is
//! summarised per player with [`feature_extractor::compute_player_match_stats`], gradient
//! boosted trees turn the summaries into MMR estimates ([`TabularSkillModel`]), a second
//! model does the same per 60-second window for the timeline, and [`coaching`] picks one
//! "here's what the next rank does better" roast per player.
//!
//! Everything ships as one [`SkillModelBundle`], stored in a compact binary format
//! ([`SkillModelBundle::to_bytes`]).
//!
//! See `docs/experiment-plan-2026-10.md` for how the model was chosen and measured.

mod analysis;
pub mod coaching;
mod encoding;
mod model;

pub use analysis::{MINIMUM_WINDOW_SECONDS, MatchAnalysis, PlayerTimeline};
pub use encoding::{BundleDecodeError, BundleEncodeError, SkillModelBundle};
pub use model::{FrozenEnsemble, FrozenNode, LobbySummary, TabularLayout, TabularSkillModel};

/// Margin above the lobby's median prediction at which a player is flagged as a smurf.
///
/// The single definition of the shipped rule: the app's badge, its verdict copy and the
/// offline evaluation tools all read it from here.
///
/// Fitted for the tabular model on held-out lobbies (`docs/experiment.md` row 44). Earlier
/// rows (≤ 43) scored the LSTM with a hand-set `+200`, which the tabular model almost never
/// reaches (0.16 % of players).
pub const SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR: f32 = 100.0;
