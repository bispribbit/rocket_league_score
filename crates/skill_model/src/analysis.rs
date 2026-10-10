#![expect(
    clippy::indexing_slicing,
    reason = "per-slot arrays of length TOTAL_PLAYERS indexed by slot < TOTAL_PLAYERS"
)]

//! Turns a parsed replay into everything the app shows.
//!
//! * **Whole-match scores** come from the match model — the verdict and the cards.
//! * **The timeline** shows *form*: each player's whole-match score, moved up or down by how
//!   much better or worse they did in a window than in their own average window, times
//!   [`SkillModelBundle::timeline_emphasis`]. Window-level lobby estimates jump by ±100 MMR
//!   from one minute to the next for the whole lobby at once (see `docs/model.md`),
//!   so the timeline never shows them directly.
//! * **One roast per player** from [`crate::coaching`].

use std::ops::Range;

use feature_extractor::{
    PLAYERS_PER_TEAM, PlayerMatchStats, PlayerRoster, TOTAL_PLAYERS, compute_player_match_stats,
    compute_player_window_stats, time_windows,
};
use replay_structs::{ParsedReplay, Team};

use crate::SkillModelBundle;
use crate::coaching::CoachingAdvice;

/// Players with less live time than this in a window get no timeline score there. Same
/// cut-off the window model was trained with.
pub const MINIMUM_WINDOW_SECONDS: f32 = 10.0;

/// One timeline window: its frame range and each slot's displayed MMR.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerTimeline {
    /// Raw frame range of the window.
    pub frames: Range<usize>,
    /// Displayed MMR per slot, `None` when the player was (nearly) absent.
    pub player_mmr: [Option<f32>; TOTAL_PLAYERS],
}

/// Everything the app needs for one replay.
#[derive(Debug, Clone)]
pub struct MatchAnalysis {
    /// Player names in slot order (blue sorted by name, then orange); empty when unused.
    pub names: [String; TOTAL_PLAYERS],
    /// Whole-match MMR per slot.
    pub match_mmr: [Option<f32>; TOTAL_PLAYERS],
    /// Roast per slot.
    pub coaching: [Option<CoachingAdvice>; TOTAL_PLAYERS],
    /// Timeline windows in order.
    pub timeline: Vec<PlayerTimeline>,
}

impl MatchAnalysis {
    /// Team of a slot (slots `0..3` are blue).
    #[must_use]
    pub const fn team(slot: usize) -> Team {
        if slot < PLAYERS_PER_TEAM {
            Team::Blue
        } else {
            Team::Orange
        }
    }
}

/// Mean over present slots.
fn present_mean(values: &[Option<f32>; TOTAL_PLAYERS]) -> f32 {
    let present: Vec<f32> = values.iter().flatten().copied().collect();
    present.iter().sum::<f32>() / present.len().max(1) as f32
}

impl SkillModelBundle {
    /// Scores one replay.
    #[must_use]
    pub fn analyze(&self, parsed: &ParsedReplay) -> MatchAnalysis {
        let roster = PlayerRoster::from_parsed(parsed);
        let match_stats = compute_player_match_stats(parsed, &roster);
        let match_mmr = self.match_model.predict(&match_stats);
        let coaching = core::array::from_fn(|slot| {
            let stats: &PlayerMatchStats = match_stats[slot].as_ref()?;
            self.coaching.advise(match_mmr[slot]?, &stats.values)
        });
        let timeline = self.timeline(parsed, &roster, &match_mmr);
        MatchAnalysis {
            names: roster.names,
            match_mmr,
            coaching,
            timeline,
        }
    }

    fn timeline(
        &self,
        parsed: &ParsedReplay,
        roster: &PlayerRoster,
        match_mmr: &[Option<f32>; TOTAL_PLAYERS],
    ) -> Vec<PlayerTimeline> {
        let windows = time_windows(parsed, self.window_seconds);
        // Each window: every present player's deviation from that window's lobby, and how
        // long they played in it.
        let mut deviations: Vec<[Option<f32>; TOTAL_PLAYERS]> = Vec::with_capacity(windows.len());
        let mut weights: Vec<[f32; TOTAL_PLAYERS]> = Vec::with_capacity(windows.len());
        for range in &windows {
            let mut stats = compute_player_window_stats(parsed, roster, range.clone());
            for slot_stats in &mut stats {
                if slot_stats
                    .as_ref()
                    .is_some_and(|s| s.live_seconds < MINIMUM_WINDOW_SECONDS)
                {
                    *slot_stats = None;
                }
            }
            let predictions = self.window_model.predict(&stats);
            let lobby = present_mean(&predictions);
            deviations.push(core::array::from_fn(|slot| {
                predictions[slot].map(|value| value - lobby)
            }));
            weights.push(core::array::from_fn(|slot| {
                stats[slot].as_ref().map_or(0.0, |s| s.live_seconds)
            }));
        }

        // A player's average window deviation, so the timeline shows form around their own
        // whole-match level rather than a second, competing estimate of it.
        let average_deviation: [f32; TOTAL_PLAYERS] = core::array::from_fn(|slot| {
            let mut weighted = 0.0;
            let mut total = 0.0;
            for (window, weight) in deviations.iter().zip(&weights) {
                if let Some(deviation) = window[slot] {
                    weighted = deviation.mul_add(weight[slot], weighted);
                    total += weight[slot];
                }
            }
            if total > 0.0 { weighted / total } else { 0.0 }
        });

        windows
            .into_iter()
            .zip(deviations)
            .map(|(frames, window)| PlayerTimeline {
                frames,
                player_mmr: core::array::from_fn(|slot| {
                    let level = match_mmr[slot]?;
                    let form = window[slot]? - average_deviation[slot];
                    Some(self.timeline_emphasis.mul_add(form, level))
                }),
            })
            .collect()
    }
}
