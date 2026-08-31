//! Rank-index label warping — the step-6 "tail compression" intervention.
//!
//! # Why
//!
//! MMR spacing along the ladder is wildly uneven. Champion-1 → Champion-2 is 130 MMR;
//! Grand-Champion-3 → Supersonic-Legend is 418. Squared error on raw MMR therefore spends
//! roughly three times the gradient on one step at the top of the ladder as on one step in
//! the middle, and rewards the model for emitting very large values.
//!
//! Row 28 measured the consequence. Held-out margins (prediction − lobby median
//! prediction) have a p99 of `+434 MMR` among **non**-smurfs against `+271` among smurfs:
//! the most extreme margins belong to ordinary players the model over-rates, so the
//! strictest thresholds score *zero* true positives. Nine times the training data did not
//! thin that tail — it pushed it further out.
//!
//! # What this does
//!
//! Maps raw MMR onto a **rank index**: the position along the ladder measured in ladder
//! steps rather than MMR points, so every rank step is the same width by construction.
//! A model trained against this target has no incentive to emit extreme values, because
//! extremeness buys it nothing the loss rewards.
//!
//! The index is rescaled by [`crate::MMR_SCALE`] / [`RANK_INDEX_SPAN`] so warped values
//! occupy the same numeric range as raw MMR. Every downstream consumer — the `/ MMR_SCALE`
//! normalisation, the collapse gates, the label-jitter σ — keeps working unchanged.
//!
//! # What it deliberately does not touch
//!
//! Only the **regression** target is warped. The ordinal head thresholds *targets* against
//! [`crate::ORDINAL_BOUNDARIES_MMR`] and never sees a prediction, so it is unaffected. Rank
//! weights are looked up from raw targets. The pairwise hinge compares predictions against
//! target *ordering*, and the warp is strictly monotone, so ordering is preserved exactly.

use crate::{MMR_SCALE, ORDINAL_BOUNDARIES_MMR, ORDINAL_NUM_BOUNDARIES};

/// Total width of the rank index: one segment below the first boundary, one between each
/// adjacent pair, and one above the last.
pub const RANK_INDEX_SPAN: f32 = ORDINAL_NUM_BOUNDARIES as f32 + 1.0;

/// Width of the final segment, used for the open-ended region above the SSL boundary.
///
/// Reuses the last inter-boundary width so a player above the top boundary keeps a finite,
/// sensibly-scaled index instead of saturating — saturation would make every SSL-and-above
/// player identical to the model and destroy ordering in exactly the region smurfs occupy.
fn top_segment_width() -> f32 {
    let last = ORDINAL_BOUNDARIES_MMR[ORDINAL_NUM_BOUNDARIES - 1];
    let previous = ORDINAL_BOUNDARIES_MMR[ORDINAL_NUM_BOUNDARIES - 2];
    (last - previous).max(1.0)
}

/// The breakpoints and segment widths defining the piecewise-linear warp.
///
/// Returned as `(start_mmr, width_mmr)` pairs in ascending order, one per index step. The
/// warp is the sum of clamped ramps over these segments, which is what makes it expressible
/// as a handful of elementwise tensor ops (see `ml_model_training::label_warp`).
#[must_use]
pub fn rank_index_segments() -> Vec<(f32, f32)> {
    let mut segments = Vec::with_capacity(ORDINAL_NUM_BOUNDARIES + 1);

    // Below the first boundary: 0 MMR maps to index 0, the first boundary to index 1.
    segments.push((0.0, ORDINAL_BOUNDARIES_MMR[0].max(1.0)));

    for pair in ORDINAL_BOUNDARIES_MMR.windows(2) {
        if let [low, high] = *pair {
            segments.push((low, (high - low).max(1.0)));
        }
    }

    segments.push((
        ORDINAL_BOUNDARIES_MMR[ORDINAL_NUM_BOUNDARIES - 1],
        top_segment_width(),
    ));

    segments
}

/// Maps raw MMR to a rank index in `[0, RANK_INDEX_SPAN]`.
///
/// Strictly monotone increasing, so it never changes the ordering of two players.
#[must_use]
pub fn mmr_to_rank_index(mmr: f32) -> f32 {
    rank_index_segments()
        .into_iter()
        .map(|(start, width)| ((mmr - start) / width).clamp(0.0, 1.0))
        .sum()
}

/// Maps raw MMR onto the warped scale, in the same numeric range as raw MMR.
#[must_use]
pub fn warp_mmr(mmr: f32) -> f32 {
    mmr_to_rank_index(mmr) / RANK_INDEX_SPAN * MMR_SCALE
}

/// Inverse of [`warp_mmr`], for reporting a warped prediction back in MMR units.
///
/// Exact on the segment breakpoints and linear between them. Values at or above the top of
/// the index saturate at the last breakpoint plus one segment width, since the warp is only
/// defined up to there.
#[must_use]
pub fn unwarp_mmr(warped: f32) -> f32 {
    let index = (warped / MMR_SCALE * RANK_INDEX_SPAN).clamp(0.0, RANK_INDEX_SPAN);
    let segments = rank_index_segments();

    let whole = index.floor();
    let fraction = index - whole;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "index is clamped to [0, RANK_INDEX_SPAN], so the cast is in range"
    )]
    let segment_index = (whole as usize).min(segments.len() - 1);

    let (start, width) = segments.get(segment_index).copied().unwrap_or((0.0, 1.0));
    start + fraction * width
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The warp must never reorder two players — every ordinal metric in the project
    /// depends on that, and a non-monotone warp would silently corrupt concordance.
    #[test]
    fn warp_is_strictly_monotone_across_the_ladder() {
        let mut previous = f32::NEG_INFINITY;
        let mut mmr = 0.0_f32;
        while mmr <= 2600.0 {
            let warped = warp_mmr(mmr);
            assert!(
                warped > previous,
                "warp not increasing at {mmr} MMR: {warped} <= {previous}"
            );
            previous = warped;
            mmr += 5.0;
        }
    }

    /// Each ladder boundary must land on consecutive integer indices — that is the whole
    /// point: every rank step becomes the same width.
    #[test]
    fn boundaries_land_on_a_uniform_grid() {
        for (position, boundary) in ORDINAL_BOUNDARIES_MMR.iter().enumerate() {
            let index = mmr_to_rank_index(*boundary);
            let expected = position as f32 + 1.0;
            assert!(
                (index - expected).abs() < 1e-3,
                "boundary {boundary} MMR mapped to index {index}, expected {expected}"
            );
        }
    }

    /// The intervention, stated as a test: the widest ladder step (GC-3 → SSL, 418 MMR)
    /// and one of the narrowest (Bronze-1 → Bronze-2, 63 MMR) must become equal after
    /// warping, so squared error stops over-weighting the top of the ladder.
    #[test]
    fn uneven_mmr_steps_become_equal_rank_steps() {
        let narrow = warp_mmr(ORDINAL_BOUNDARIES_MMR[1]) - warp_mmr(ORDINAL_BOUNDARIES_MMR[0]);
        let wide = warp_mmr(ORDINAL_BOUNDARIES_MMR[20]) - warp_mmr(ORDINAL_BOUNDARIES_MMR[19]);

        let raw_narrow = ORDINAL_BOUNDARIES_MMR[1] - ORDINAL_BOUNDARIES_MMR[0];
        let raw_wide = ORDINAL_BOUNDARIES_MMR[20] - ORDINAL_BOUNDARIES_MMR[19];
        assert!(
            raw_wide > raw_narrow * 5.0,
            "precondition: raw steps should be very uneven ({raw_narrow} vs {raw_wide})"
        );

        assert!(
            (wide - narrow).abs() < 1e-2,
            "warped steps still uneven: {narrow} vs {wide}"
        );
    }

    /// Round-trips on the breakpoints, which is what makes a warped prediction reportable
    /// in MMR units.
    #[test]
    fn unwarp_inverts_warp_on_the_boundaries() {
        for boundary in ORDINAL_BOUNDARIES_MMR {
            let round_tripped = unwarp_mmr(warp_mmr(boundary));
            assert!(
                (round_tripped - boundary).abs() < 1.0,
                "{boundary} MMR round-tripped to {round_tripped}"
            );
        }
    }

    /// Above the top boundary the index must keep increasing rather than saturating —
    /// saturation would flatten every SSL-and-above player into one value, which is exactly
    /// the population smurf detection cares about.
    #[test]
    fn above_the_top_boundary_still_orders() {
        let top = ORDINAL_BOUNDARIES_MMR[ORDINAL_NUM_BOUNDARIES - 1];
        assert!(warp_mmr(top + 100.0) > warp_mmr(top));
        assert!(warp_mmr(top + 300.0) > warp_mmr(top + 100.0));
    }

    /// Warped values must stay inside the raw-MMR numeric range so the `/ MMR_SCALE`
    /// normalisation, collapse gates and jitter σ all keep their calibration.
    #[test]
    fn warped_values_stay_in_the_mmr_numeric_range() {
        for mmr in [0.0_f32, 500.0, 1200.0, 2200.0, 2600.0] {
            let warped = warp_mmr(mmr);
            assert!(
                (0.0..=MMR_SCALE).contains(&warped),
                "{mmr} MMR warped to {warped}, outside [0, {MMR_SCALE}]"
            );
        }
    }
}
