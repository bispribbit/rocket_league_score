//! Fitting the smurf flag threshold, and scoring it honestly.
//!
//! The shipped rule flags a player when their whole-match prediction sits more than
//! [`ml_model::SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR`] (+200) above the median prediction of
//! their lobby. Step 1 of `docs/smurf-detection-handoff.md` measured the achievable margin
//! at `+10` to `+18 MMR`, so that rule essentially never fires: detection is 0–3 % not
//! because the ordering is wrong but because the threshold is out of range.
//!
//! This module re-fits the threshold from the data instead of asserting it, and reports a
//! **precision/recall curve** rather than a single detection rate. Two things make the
//! numbers here mean more than the `mixed_detection_rate` in [`crate::training`]:
//!
//! 1. **Every player is scored, not just the top-labelled player of a mixed lobby.** That
//!    is the only way false positives exist, and without false positives there is no
//!    precision — a threshold of `-inf` would "detect" 100 %. Deployment does not know in
//!    advance which lobbies are mixed, so neither does this.
//! 2. **The threshold is fitted on one set of lobbies and scored on another** (see
//!    [`split_by_replay`]). A threshold chosen and reported on the same players is a
//!    training-set number.
//!
//! The label side of the definition is the one the acquisition pipeline and the training
//! metric already use: a player is a smurf proxy when their *label* sits
//! [`SMURF_LABEL_MARGIN_MMR`] or more above their lobby's median label. This is a proxy,
//! not ground truth — it says "the highest-ranked player in a mixed lobby", which is what
//! this dataset can supply.

use std::collections::HashMap;

use uuid::Uuid;

/// Label margin over the lobby median at which a player counts as a smurf proxy.
///
/// Aliases the acquisition/oversampling threshold so "positive" here means the same thing
/// as "mixed lobby" everywhere else in the tree: `segment_top_gap_mmr` is exactly the
/// largest [`MarginSample::label_margin_mmr`] in a lobby.
pub const SMURF_LABEL_MARGIN_MMR: f32 = crate::segment_cache::MIXED_TOP_GAP_THRESHOLD_MMR;

/// One player's whole-match prediction in one lobby.
///
/// The prediction is the median over that player's segments, matching what
/// `is_this_a_smurf` shows and what [`crate::WithinLobbyMetrics`] scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPrediction {
    /// Replay this player's lobby belongs to.
    pub replay_id: Uuid,
    /// Player slot (blue 0..3, orange 3..6), name-sorted as in the training path.
    pub slot: usize,
    /// Whole-match prediction in MMR.
    pub prediction_mmr: f32,
    /// Label in MMR. Always `> 0`; unknown-rank slots are never emitted.
    pub target_mmr: f32,
    /// Number of segment predictions the median was taken over.
    pub segments: usize,
}

/// A player reduced to the two quantities the flag rule is defined on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarginSample {
    /// Replay this player's lobby belongs to.
    pub replay_id: Uuid,
    /// Player slot within the lobby.
    pub slot: usize,
    /// `prediction - lobby median prediction`. The quantity the rule thresholds.
    pub margin_mmr: f32,
    /// `label - lobby median label`. The quantity that defines a positive.
    pub label_margin_mmr: f32,
}

impl MarginSample {
    /// True when this player is a smurf proxy: labelled [`SMURF_LABEL_MARGIN_MMR`] or more
    /// above their lobby's median label.
    pub fn is_positive(&self) -> bool {
        self.label_margin_mmr >= SMURF_LABEL_MARGIN_MMR
    }
}

/// Median of `values`, sorted in place. Returns `0.0` for an empty slice.
///
/// Mirrors the helper the validation metrics use in `training.rs`, which is private to that
/// module; [`tests::median_matches_the_validation_metric`] pins the two together.
fn median_of(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let lower = values.get(mid - 1).copied().unwrap_or(0.0);
        let upper = values.get(mid).copied().unwrap_or(0.0);
        f32::midpoint(lower, upper)
    } else {
        values.get(mid).copied().unwrap_or(0.0)
    }
}

/// Centres every player on their own lobby's medians.
///
/// Lobbies with fewer than two rank-known players are dropped: a one-player lobby has no
/// median to be above, and the shipped rule cannot fire in one either. Output is sorted by
/// `(replay_id, slot)` so a dump is reproducible across runs.
pub fn margin_samples(predictions: &[PlayerPrediction]) -> Vec<MarginSample> {
    let mut by_replay: HashMap<Uuid, Vec<PlayerPrediction>> = HashMap::new();
    for prediction in predictions {
        by_replay
            .entry(prediction.replay_id)
            .or_default()
            .push(*prediction);
    }

    let mut samples = Vec::with_capacity(predictions.len());
    for (replay_id, players) in by_replay {
        if players.len() < 2 {
            continue;
        }
        let mut preds: Vec<f32> = players.iter().map(|p| p.prediction_mmr).collect();
        let mut targets: Vec<f32> = players.iter().map(|p| p.target_mmr).collect();
        let pred_median = median_of(&mut preds);
        let target_median = median_of(&mut targets);

        for player in players {
            samples.push(MarginSample {
                replay_id,
                slot: player.slot,
                margin_mmr: player.prediction_mmr - pred_median,
                label_margin_mmr: player.target_mmr - target_median,
            });
        }
    }

    samples.sort_by(|a, b| a.replay_id.cmp(&b.replay_id).then(a.slot.cmp(&b.slot)));
    samples
}

/// A threshold that separates `higher` from `lower`, two observed margins with
/// `higher > lower`, under the rule `margin > threshold`.
///
/// The midpoint is preferred because a fitted threshold gets applied to unseen margins, and
/// sitting between the two observed values generalises better than sitting on either. But
/// for adjacent floats `(higher + lower) / 2` can round back up onto `higher`, which would
/// exclude the very group the threshold is meant to include. `lower` always separates them
/// exactly, so fall back to it in that case.
fn separating_threshold(higher: f32, lower: f32) -> f32 {
    let midpoint = f32::midpoint(higher, lower);
    if midpoint < higher { midpoint } else { lower }
}

/// One point on the precision/recall curve: the outcome of flagging at one threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatingPoint {
    /// Flag when `margin_mmr > threshold_mmr`, matching the shipped rule's strict `>`.
    pub threshold_mmr: f32,
    /// Players flagged at this threshold.
    pub flagged: usize,
    /// Flagged players who are smurf proxies.
    pub true_positives: usize,
    /// `true_positives / flagged`. `0.0` when nothing is flagged.
    pub precision: f32,
    /// `true_positives / positives`. `0.0` when there are no positives.
    pub recall: f32,
    /// Harmonic mean of the two. `0.0` when either is zero.
    pub f1: f32,
    /// `flagged / samples` — the share of all players the rule accuses.
    pub flag_rate: f32,
}

/// Scores one threshold over `samples`. Flags when `margin_mmr > threshold_mmr`.
pub fn evaluate_threshold(samples: &[MarginSample], threshold_mmr: f32) -> OperatingPoint {
    let positives = samples.iter().filter(|s| s.is_positive()).count();
    let mut flagged = 0usize;
    let mut true_positives = 0usize;
    for sample in samples {
        if sample.margin_mmr > threshold_mmr {
            flagged += 1;
            if sample.is_positive() {
                true_positives += 1;
            }
        }
    }
    operating_point(
        threshold_mmr,
        flagged,
        true_positives,
        positives,
        samples.len(),
    )
}

fn operating_point(
    threshold_mmr: f32,
    flagged: usize,
    true_positives: usize,
    positives: usize,
    total: usize,
) -> OperatingPoint {
    let precision = if flagged == 0 {
        0.0
    } else {
        true_positives as f32 / flagged as f32
    };
    let recall = if positives == 0 {
        0.0
    } else {
        true_positives as f32 / positives as f32
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    OperatingPoint {
        threshold_mmr,
        flagged,
        true_positives,
        precision,
        recall,
        f1,
        flag_rate: if total == 0 {
            0.0
        } else {
            flagged as f32 / total as f32
        },
    }
}

/// Every distinct operating point, from strictest to loosest.
///
/// One point per distinct margin value: the threshold is placed midway between that margin
/// and the next one down, so `margin > threshold` includes exactly the intended group and
/// no tie can land on the boundary. The final point flags everything.
pub fn precision_recall_curve(samples: &[MarginSample]) -> Vec<OperatingPoint> {
    if samples.is_empty() {
        return Vec::new();
    }
    let positives = samples.iter().filter(|s| s.is_positive()).count();
    let total = samples.len();

    let mut sorted: Vec<&MarginSample> = samples.iter().collect();
    sorted.sort_by(|a, b| {
        b.margin_mmr
            .partial_cmp(&a.margin_mmr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut curve = Vec::new();
    let mut flagged = 0usize;
    let mut true_positives = 0usize;
    for (index, sample) in sorted.iter().enumerate() {
        flagged += 1;
        if sample.is_positive() {
            true_positives += 1;
        }
        // Only emit at the end of a run of equal margins — no threshold can separate two
        // players whose margin is identical.
        let threshold = match sorted.get(index + 1).map(|next| next.margin_mmr) {
            Some(next) if (next - sample.margin_mmr).abs() < f32::EPSILON => continue,
            Some(next) => separating_threshold(sample.margin_mmr, next),
            None => sample.margin_mmr - 1.0,
        };
        curve.push(operating_point(
            threshold,
            flagged,
            true_positives,
            positives,
            total,
        ));
    }
    curve
}

/// Threshold that flags approximately `flag_rate` of `samples` — the percentile fit.
///
/// `flag_rate` is clamped to `[0, 1]`. A rate of `0` returns a threshold above every
/// margin (nothing flagged); `1` returns one below every margin.
///
/// **Ties undershoot, never overshoot.** When many players share one margin the threshold
/// lands on that value and excludes the whole group, so the achieved flag rate can fall
/// below the requested one. That is the conservative direction, and it is not hypothetical:
/// a collapsed model puts every player at margin `0`, and odd-sized lobbies always put
/// their median player there. Read the achieved `flag_rate` off the returned
/// [`OperatingPoint`] rather than assuming the request was met.
pub fn threshold_at_flag_rate(samples: &[MarginSample], flag_rate: f64) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut margins: Vec<f32> = samples.iter().map(|s| s.margin_mmr).collect();
    margins.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let take = (flag_rate.clamp(0.0, 1.0) * margins.len() as f64).round() as usize;
    if take == 0 {
        return margins.first().copied().unwrap_or(0.0);
    }
    if take >= margins.len() {
        return margins.last().copied().unwrap_or(0.0) - 1.0;
    }
    // Flag the top `take`: put the threshold between the last included and the first
    // excluded margin.
    let last_included = margins.get(take - 1).copied().unwrap_or(0.0);
    let first_excluded = margins.get(take).copied().unwrap_or(0.0);
    separating_threshold(last_included, first_excluded)
}

/// Share of `samples` that are smurf proxies — the base rate any precision must beat.
pub fn positive_base_rate(samples: &[MarginSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().filter(|s| s.is_positive()).count() as f64 / samples.len() as f64
}

/// The curve point with the highest F1, ties broken toward the stricter threshold.
pub fn best_f1(curve: &[OperatingPoint]) -> Option<OperatingPoint> {
    curve
        .iter()
        .copied()
        .reduce(|best, point| if point.f1 > best.f1 { point } else { best })
}

/// Area under the precision/recall curve, by the rectangle rule over recall.
///
/// This is average precision, the standard threshold-free summary of a ranker. A perfect
/// ranker scores `1.0`; a useless one scores the positive base rate.
pub fn average_precision(curve: &[OperatingPoint]) -> f32 {
    let mut area = 0.0f64;
    let mut previous_recall = 0.0f64;
    for point in curve {
        let recall = f64::from(point.recall);
        area = (recall - previous_recall).mul_add(f64::from(point.precision), area);
        previous_recall = recall;
    }
    area as f32
}

/// The in-training mixed-lobby metrics, re-derived from a prediction dump.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixedLobbyCrossCheck {
    /// Lobbies whose top label sits [`SMURF_LABEL_MARGIN_MMR`] or more above the median.
    pub mixed_lobbies: usize,
    /// Of those, the ones with a single unambiguous highest-labelled player.
    pub scored_lobbies: usize,
    /// Fraction where that player also received the highest prediction, ties split.
    pub top1_rate: f32,
    /// Mean of (top-labelled player's prediction − lobby median prediction).
    pub mean_margin_mmr: f32,
}

/// Recomputes [`crate::WithinLobbyMetrics`]'s mixed-lobby numbers from a dump.
///
/// This deliberately re-implements the definitions in `training.rs` rather than sharing
/// them, because its purpose is to be a **cross-check**: if these numbers do not reproduce
/// the ones the scoring pass logged, the dump does not faithfully represent what was
/// measured and nothing fitted off it can be trusted.
///
/// It also gives the two disagreeing top-1 figures in `docs/smurf-detection-handoff.md`
/// (34.7 % from the in-training metric, 22.2 % from `smurf_spotcheck`) a common reference
/// point: agreement here isolates the disagreement to `smurf_spotcheck`'s different lobby
/// population rather than to a difference in how the rate is computed.
pub fn mixed_lobby_cross_check(predictions: &[PlayerPrediction]) -> MixedLobbyCrossCheck {
    let mut by_replay: HashMap<Uuid, Vec<PlayerPrediction>> = HashMap::new();
    for prediction in predictions {
        by_replay
            .entry(prediction.replay_id)
            .or_default()
            .push(*prediction);
    }

    let mut mixed_lobbies = 0usize;
    let mut scored = 0usize;
    let mut top1 = 0.0f64;
    let mut margin_sum = 0.0f64;

    for players in by_replay.values() {
        if players.len() < 2 {
            continue;
        }
        let mut targets: Vec<f32> = players.iter().map(|p| p.target_mmr).collect();
        let target_median = median_of(&mut targets);
        let highest_target = players
            .iter()
            .map(|p| p.target_mmr)
            .fold(f32::MIN, f32::max);
        if highest_target - target_median < SMURF_LABEL_MARGIN_MMR {
            continue;
        }
        mixed_lobbies += 1;

        // Only well defined when one player is clearly highest-labelled.
        let at_top = players
            .iter()
            .filter(|p| (p.target_mmr - highest_target).abs() < f32::EPSILON)
            .count();
        if at_top != 1 {
            continue;
        }
        let Some(top_player) = players
            .iter()
            .find(|p| (p.target_mmr - highest_target).abs() < f32::EPSILON)
        else {
            continue;
        };

        scored += 1;
        let mut preds: Vec<f32> = players.iter().map(|p| p.prediction_mmr).collect();
        let pred_median = median_of(&mut preds);
        margin_sum += f64::from(top_player.prediction_mmr - pred_median);

        // Fractional credit on ties, so a collapsed model scores chance rather than 100 %.
        let highest_pred = players
            .iter()
            .map(|p| p.prediction_mmr)
            .fold(f32::MIN, f32::max);
        if (top_player.prediction_mmr - highest_pred).abs() < f32::EPSILON {
            let tied = players
                .iter()
                .filter(|p| (p.prediction_mmr - highest_pred).abs() < f32::EPSILON)
                .count()
                .max(1);
            top1 += 1.0 / tied as f64;
        }
    }

    MixedLobbyCrossCheck {
        mixed_lobbies,
        scored_lobbies: scored,
        top1_rate: if scored == 0 {
            0.0
        } else {
            (top1 / scored as f64) as f32
        },
        mean_margin_mmr: if scored == 0 {
            0.0
        } else {
            (margin_sum / scored as f64) as f32
        },
    }
}

/// Deterministic 64-bit hash of a replay id (FNV-1a plus a splitmix64 finalizer).
///
/// Hand-rolled so that any reproducible subset keyed off a replay — this module's
/// fit/held-out split, the pipeline's fixed dev subset — is stable across runs, machines
/// and toolchains. `DefaultHasher` guarantees none of those: it is explicitly allowed to
/// differ between releases and is randomly seeded per process in some configurations.
///
/// Uniform enough in the top bits to compare directly against a scaled cutoff, which is how
/// both callers select a fraction.
pub fn stable_replay_hash(replay_id: Uuid) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in replay_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // FNV-1a avalanches poorly into the high bits, and the split compares the hash against
    // a cutoff near `u64::MAX / 2` — which is to say it reads the top bit. Without a
    // finalizer, ids differing only in their low bytes land on the same side and the split
    // comes out lopsided. This is splitmix64's finalizer.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// Splits predictions into a fit set and a held-out set, **by lobby**.
///
/// Splitting by replay rather than by player is not a detail: margins are defined against a
/// lobby median, so a lobby broken across the two sides would have its medians computed
/// from a subset of its own players, and the two halves would not be independent.
///
/// `fit_fraction` is the share of *lobbies* in the fit set, clamped to `[0, 1]`.
pub fn split_by_replay(
    predictions: &[PlayerPrediction],
    fit_fraction: f64,
) -> (Vec<PlayerPrediction>, Vec<PlayerPrediction>) {
    let cutoff = (fit_fraction.clamp(0.0, 1.0) * u64::MAX as f64) as u64;
    let mut fit = Vec::new();
    let mut held_out = Vec::new();
    for prediction in predictions {
        if stable_replay_hash(prediction.replay_id) < cutoff {
            fit.push(*prediction);
        } else {
            held_out.push(*prediction);
        }
    }
    (fit, held_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(seed: u128) -> Uuid {
        Uuid::from_u128(seed)
    }

    fn prediction(seed: u128, slot: usize, pred: f32, target: f32) -> PlayerPrediction {
        PlayerPrediction {
            replay_id: replay(seed),
            slot,
            prediction_mmr: pred,
            target_mmr: target,
            segments: 1,
        }
    }

    /// A lobby of three: predictions and labels are each centred on their own median, and
    /// the player 300 MMR above the label median is the only positive.
    #[test]
    fn margins_are_centred_on_the_lobby_median() {
        let predictions = vec![
            prediction(1, 0, 600.0, 600.0),
            prediction(1, 1, 700.0, 650.0),
            prediction(1, 2, 900.0, 950.0),
        ];
        let samples = margin_samples(&predictions);
        assert_eq!(samples.len(), 3);
        // Median prediction 700, median label 650.
        assert!((samples[0].margin_mmr + 100.0).abs() < 1e-4);
        assert!((samples[0].label_margin_mmr + 50.0).abs() < 1e-4);
        assert!((samples[2].margin_mmr - 200.0).abs() < 1e-4);
        assert!((samples[2].label_margin_mmr - 300.0).abs() < 1e-4);
        assert!(!samples[0].is_positive());
        assert!(!samples[1].is_positive());
        assert!(samples[2].is_positive());
    }

    /// A lobby with a single rank-known player has no median to be measured against.
    #[test]
    fn single_player_lobbies_are_dropped() {
        let predictions = vec![
            prediction(1, 0, 600.0, 600.0),
            prediction(2, 0, 900.0, 950.0),
            prediction(2, 1, 600.0, 600.0),
        ];
        let samples = margin_samples(&predictions);
        assert_eq!(samples.len(), 2);
        assert!(samples.iter().all(|s| s.replay_id == replay(2)));
    }

    /// Perfect ranking: every positive margin sits above every negative one, so some
    /// threshold reaches precision 1 at recall 1.
    #[test]
    fn perfect_separation_reaches_f1_one() {
        let mut predictions = Vec::new();
        for lobby in 0..20u128 {
            let smurf = lobby % 2 == 0;
            let high = if smurf { 1400.0 } else { 620.0 };
            predictions.push(prediction(lobby, 0, 600.0, 600.0));
            predictions.push(prediction(lobby, 1, 610.0, 610.0));
            predictions.push(prediction(lobby, 2, high, high));
        }
        let samples = margin_samples(&predictions);
        let curve = precision_recall_curve(&samples);
        let best = best_f1(&curve).expect("non-empty curve");
        assert!((best.f1 - 1.0).abs() < 1e-6, "best F1 was {}", best.f1);
        assert!((average_precision(&curve) - 1.0).abs() < 1e-6);
    }

    /// A collapsed model predicts every slot in a lobby identically, so every margin is
    /// exactly 0 and no threshold can separate anything. Precision degenerates to the base
    /// rate — the honest reading, and the whole reason precision is reported at all.
    #[test]
    fn collapsed_model_scores_the_base_rate() {
        let mut predictions = Vec::new();
        for lobby in 0..20u128 {
            let smurf = lobby % 2 == 0;
            for slot in 0..3usize {
                let target = if smurf && slot == 2 { 1400.0 } else { 600.0 };
                predictions.push(prediction(lobby, slot, 700.0, target));
            }
        }
        let samples = margin_samples(&predictions);
        assert!(samples.iter().all(|s| s.margin_mmr.abs() < 1e-6));

        let curve = precision_recall_curve(&samples);
        // Only one distinct margin exists, so the curve has exactly one point: flag all.
        assert_eq!(curve.len(), 1);
        let base_rate = positive_base_rate(&samples);
        assert!((f64::from(curve[0].precision) - base_rate).abs() < 1e-6);
        assert!((curve[0].recall - 1.0).abs() < 1e-6);
        // And the shipped +200 rule fires on nobody at all.
        let shipped = evaluate_threshold(&samples, ml_model::SMURF_MARGIN_OVER_LOBBY_MEDIAN_MMR);
        assert_eq!(shipped.flagged, 0);
        assert_eq!(shipped.true_positives, 0);
    }

    /// The percentile fit flags the share it was asked for, on margins with no ties.
    ///
    /// Six-slot lobbies with spread-out predictions, which is the shape real data has.
    #[test]
    fn threshold_at_flag_rate_flags_that_fraction() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next_prediction = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            400.0 + (state >> 40) as f32 / 4.0
        };
        let mut predictions = Vec::new();
        for lobby in 0..100u128 {
            for slot in 0..6usize {
                predictions.push(prediction(lobby, slot, next_prediction(), 700.0));
            }
        }
        let samples = margin_samples(&predictions);
        for rate in [0.1, 0.25, 0.5] {
            let threshold = threshold_at_flag_rate(&samples, rate);
            let point = evaluate_threshold(&samples, threshold);
            let actual = f64::from(point.flag_rate);
            assert!(
                (actual - rate).abs() < 0.01,
                "asked for {rate}, flagged {actual}"
            );
        }
    }

    /// With a mass of tied margins the fit excludes the whole tied group rather than an
    /// arbitrary part of it, so the achieved rate lands below the request — never above.
    ///
    /// Odd-sized lobbies guarantee this case: the median player sits at margin exactly 0.
    #[test]
    fn threshold_at_flag_rate_undershoots_on_ties() {
        let mut predictions = Vec::new();
        for lobby in 0..50u128 {
            predictions.push(prediction(lobby, 0, 600.0, 700.0));
            predictions.push(prediction(lobby, 1, 700.0, 700.0));
            predictions.push(prediction(lobby, 2, 700.0 + lobby as f32 * 10.0, 700.0));
        }
        let samples = margin_samples(&predictions);
        // A third of the players are the lobby median, all at margin 0 exactly.
        let tied_at_zero = samples.iter().filter(|s| s.margin_mmr.abs() < 1e-6).count();
        assert!(tied_at_zero >= samples.len() / 3);

        let point = evaluate_threshold(&samples, threshold_at_flag_rate(&samples, 0.5));
        assert!(
            f64::from(point.flag_rate) < 0.5,
            "expected an undershoot, flagged {}",
            point.flag_rate
        );
        // Nobody tied at the boundary got flagged.
        assert!(
            samples
                .iter()
                .filter(|s| s.margin_mmr > point.threshold_mmr)
                .all(|s| s.margin_mmr.abs() > 1e-6)
        );
    }

    /// The split keeps every lobby whole, is deterministic, and loses nobody.
    #[test]
    fn split_keeps_lobbies_intact() {
        let mut predictions = Vec::new();
        for lobby in 0..200u128 {
            for slot in 0..3usize {
                predictions.push(prediction(lobby, slot, 600.0, 600.0));
            }
        }
        let (fit, held_out) = split_by_replay(&predictions, 0.5);
        assert_eq!(fit.len() + held_out.len(), predictions.len());
        assert_eq!(split_by_replay(&predictions, 0.5).0, fit);

        let fit_replays: std::collections::HashSet<Uuid> =
            fit.iter().map(|p| p.replay_id).collect();
        assert!(
            held_out.iter().all(|p| !fit_replays.contains(&p.replay_id)),
            "a lobby was split across fit and held-out"
        );
        // A 50/50 ask should land near 50/50. Sequential UUIDs differ only in their low
        // bytes, which is exactly the input an unfinalized hash splits lopsidedly.
        let fit_share = fit.len() as f64 / predictions.len() as f64;
        assert!(
            (0.4..0.6).contains(&fit_share),
            "50/50 split came out {fit_share:.3} — check the hash finalizer"
        );
    }

    /// Recall is monotone non-decreasing as the threshold loosens, the curve ends by
    /// flagging everyone, and every emitted threshold reproduces its own point when
    /// re-evaluated. Guards the midpoint and tie handling in the sweep.
    #[test]
    fn curve_is_monotone_and_complete() {
        let mut predictions = Vec::new();
        for lobby in 0..30u128 {
            predictions.push(prediction(lobby, 0, 600.0, 600.0));
            predictions.push(prediction(lobby, 1, 650.0, 640.0));
            let high = lobby % 3 == 0;
            predictions.push(prediction(
                lobby,
                2,
                650.0 + lobby as f32,
                if high { 900.0 } else { 660.0 },
            ));
        }
        let samples = margin_samples(&predictions);
        let curve = precision_recall_curve(&samples);
        assert!(!curve.is_empty());
        for pair in curve.windows(2) {
            assert!(pair[1].threshold_mmr < pair[0].threshold_mmr);
            assert!(pair[1].recall >= pair[0].recall);
            assert!(pair[1].flagged > pair[0].flagged);
        }
        let last = curve.last().expect("non-empty");
        assert_eq!(last.flagged, samples.len());
        assert!((last.recall - 1.0).abs() < 1e-6);
        for point in &curve {
            let recomputed = evaluate_threshold(&samples, point.threshold_mmr);
            assert_eq!(recomputed.flagged, point.flagged);
            assert_eq!(recomputed.true_positives, point.true_positives);
        }
    }

    /// A collapsed model scores exactly chance on the cross-check, matching the assertion
    /// the in-training metric makes about itself. Six equal predictions tie at the top, so
    /// fractional credit gives 1/6, and the margin over the median is 0.
    #[test]
    fn cross_check_collapsed_model_scores_chance() {
        let mut predictions = Vec::new();
        for lobby in 0..10u128 {
            for slot in 0..6usize {
                let target = if slot == 5 { 1400.0 } else { 600.0 };
                predictions.push(prediction(lobby, slot, 700.0, target));
            }
        }
        let check = mixed_lobby_cross_check(&predictions);
        assert_eq!(check.mixed_lobbies, 10);
        assert_eq!(check.scored_lobbies, 10);
        assert!((check.top1_rate - 1.0 / 6.0).abs() < 1e-6);
        assert!(check.mean_margin_mmr.abs() < 1e-3);
    }

    /// Perfect ordering scores 1.0, and homogeneous lobbies are not counted as mixed.
    #[test]
    fn cross_check_counts_only_mixed_lobbies() {
        let mut predictions = Vec::new();
        // Five mixed lobbies, top player predicted highest.
        for lobby in 0..5u128 {
            for slot in 0..6usize {
                let smurf = slot == 5;
                let value = if smurf { 1400.0 } else { 600.0 };
                predictions.push(prediction(lobby, slot, value, value));
            }
        }
        // Five homogeneous lobbies — no top gap, so out of scope entirely.
        for lobby in 100..105u128 {
            for slot in 0..6usize {
                predictions.push(prediction(lobby, slot, 600.0, 600.0));
            }
        }
        let check = mixed_lobby_cross_check(&predictions);
        assert_eq!(check.mixed_lobbies, 5);
        assert_eq!(check.scored_lobbies, 5);
        assert!((check.top1_rate - 1.0).abs() < 1e-6);
        assert!(check.mean_margin_mmr > 700.0);
    }

    /// The separator must sit strictly below the value it is separating, even for adjacent
    /// floats where the midpoint rounds back up. Without this the curve would report a
    /// threshold that excludes the group it just counted as flagged.
    #[test]
    fn separating_threshold_always_admits_the_higher_value() {
        // Ordinary case: strictly between.
        let threshold = separating_threshold(200.0, 100.0);
        assert!(threshold > 100.0 && threshold < 200.0);

        // Adjacent floats, where the midpoint has nowhere to land.
        let higher = 1234.5678f32;
        let lower = f32::from_bits(higher.to_bits() - 1);
        assert!(lower < higher);
        let tight = separating_threshold(higher, lower);
        assert!(tight < higher, "threshold {tight} would exclude {higher}");
        assert!(tight >= lower);
    }

    /// The lobby median used here must be the one the validation metrics use, or the
    /// margins fitted in this module would describe a different rule than the one measured
    /// in `training.rs`. Even/odd lengths are the case that actually differs between
    /// median conventions.
    #[test]
    fn median_matches_the_validation_metric() {
        assert!((median_of(&mut [3.0, 1.0, 2.0]) - 2.0).abs() < 1e-6);
        assert!((median_of(&mut [4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-6);
        assert!(median_of(&mut []).abs() < 1e-6);
    }
}
