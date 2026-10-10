//! Evaluation metrics for the skill model.
//!
//! The headline numbers use **every** player in every evaluation lobby (~7–8k players), so
//! they resolve small differences; see `docs/model.md` for what each one means.
//!
//! The central quantity is the within-lobby deviation: a player's value minus their lobby's
//! mean. [`LobbyMetrics::within_r`] correlates predicted and labelled deviations; the
//! accompanying slope and shrunk RMSE say whether the model's deviations have the right
//! *size* as well as the right *direction*.

use std::collections::HashMap;

use uuid::Uuid;

/// One player's whole-match prediction in one lobby.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPrediction {
    /// Replay this player's lobby belongs to.
    pub replay_id: Uuid,
    /// Player slot (blue 0..3, orange 3..6).
    pub slot: usize,
    /// Prediction in MMR.
    pub prediction_mmr: f32,
    /// Label in MMR (`0` = rank unknown).
    pub target_mmr: f32,
}

/// Stable, well-mixed hash of a replay id, for train / early-stopping splits that never
/// change between runs or machines.
#[must_use]
pub fn stable_replay_hash(replay_id: Uuid) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in replay_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // FNV-1a mixes poorly into the high bits; finish with splitmix64's finalizer.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

/// Minimum rank-known players for a lobby to contribute to within-lobby metrics.
///
/// With fewer, the lobby mean is dominated by one or two players and the deviations are
/// mostly the mirror image of each other.
pub const MINIMUM_PLAYERS_PER_LOBBY: usize = 4;

/// Pairs whose labels sit closer than this are skipped by [`LobbyMetrics::concordance`].
///
/// Same value as the in-training metric: adjacent divisions sit ~19 MMR apart, so closer
/// pairs are label noise rather than a skill difference.
pub const CONCORDANCE_MINIMUM_LABEL_GAP_MMR: f32 = 25.0;

/// Number of lobby-level bootstrap resamples behind [`LobbyMetrics::within_r_standard_error`].
const BOOTSTRAP_RESAMPLES: usize = 200;

/// Benchmark metrics for one prediction dump.
#[derive(Debug, Clone, Copy, Default)]
pub struct LobbyMetrics {
    /// Lobbies with at least [`MINIMUM_PLAYERS_PER_LOBBY`] rank-known players.
    pub lobby_count: usize,
    /// Players in those lobbies.
    pub player_count: usize,
    /// Pearson r between predicted and labelled within-lobby deviations. **Primary metric.**
    pub within_r: f64,
    /// Standard error of [`Self::within_r`] from a bootstrap over lobbies.
    pub within_r_standard_error: f64,
    /// OLS slope of labelled deviation on predicted deviation. `< 1` means the predicted
    /// deviations are too large for the signal they carry; `> 1` means too small.
    pub within_slope: f64,
    /// RMS of the labelled within-lobby deviations: the within-RMSE of predicting the lobby
    /// value for every player.
    pub label_within_rms: f64,
    /// Within-lobby RMSE of the raw predicted deviations.
    pub within_rmse: f64,
    /// Within-lobby RMSE after scaling predicted deviations by [`Self::within_slope`].
    pub within_rmse_shrunk: f64,
    /// RMSE of lobby-mean prediction against lobby-mean label.
    pub lobby_rmse: f64,
    /// Per-player whole-match RMSE.
    pub player_rmse: f64,
    /// Within-lobby pairwise concordance over pairs more than
    /// [`CONCORDANCE_MINIMUM_LABEL_GAP_MMR`] apart. Ties count half.
    pub concordance: f64,
    /// Number of pairs behind [`Self::concordance`].
    pub concordance_pairs: usize,
}

/// One player centred on their lobby.
#[derive(Debug, Clone, Copy)]
struct CentredPlayer {
    /// Prediction minus the lobby's mean prediction.
    prediction: f64,
    /// Label minus the lobby's mean label.
    label: f64,
}

/// One lobby's players, centred, plus the lobby-level means.
#[derive(Debug, Clone)]
struct CentredLobby {
    players: Vec<CentredPlayer>,
    raw: Vec<PlayerPrediction>,
    mean_prediction: f64,
    mean_label: f64,
}

/// Running sums for a Pearson correlation and an OLS slope.
#[derive(Debug, Clone, Copy, Default)]
struct PairSums {
    count: f64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl PairSums {
    fn add(&mut self, x: f64, y: f64) {
        self.count += 1.0;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx = x.mul_add(x, self.sum_xx);
        self.sum_yy = y.mul_add(y, self.sum_yy);
        self.sum_xy = x.mul_add(y, self.sum_xy);
    }

    fn covariance(&self) -> f64 {
        (self.sum_x / self.count).mul_add(-(self.sum_y / self.count), self.sum_xy / self.count)
    }

    fn variance_x(&self) -> f64 {
        (self.sum_x / self.count).mul_add(-(self.sum_x / self.count), self.sum_xx / self.count)
    }

    fn variance_y(&self) -> f64 {
        (self.sum_y / self.count).mul_add(-(self.sum_y / self.count), self.sum_yy / self.count)
    }

    fn pearson(&self) -> f64 {
        if self.count < 2.0 {
            return f64::NAN;
        }
        let denominator = (self.variance_x() * self.variance_y()).sqrt();
        if denominator <= 0.0 {
            return 0.0;
        }
        self.covariance() / denominator
    }

    /// Slope of y on x.
    fn slope(&self) -> f64 {
        let variance = self.variance_x();
        if variance <= 0.0 {
            return 0.0;
        }
        self.covariance() / variance
    }
}

fn centre_lobbies(predictions: &[PlayerPrediction]) -> Vec<CentredLobby> {
    let mut by_replay: HashMap<Uuid, Vec<PlayerPrediction>> = HashMap::new();
    for prediction in predictions {
        if prediction.target_mmr > 0.0 {
            by_replay
                .entry(prediction.replay_id)
                .or_default()
                .push(*prediction);
        }
    }

    let mut replay_ids: Vec<Uuid> = by_replay.keys().copied().collect();
    replay_ids.sort();

    let mut lobbies = Vec::with_capacity(replay_ids.len());
    for replay_id in replay_ids {
        let Some(mut players) = by_replay.remove(&replay_id) else {
            continue;
        };
        if players.len() < MINIMUM_PLAYERS_PER_LOBBY {
            continue;
        }
        players.sort_by_key(|player| player.slot);
        let count = players.len() as f64;
        let mean_prediction = players
            .iter()
            .map(|p| f64::from(p.prediction_mmr))
            .sum::<f64>()
            / count;
        let mean_label = players.iter().map(|p| f64::from(p.target_mmr)).sum::<f64>() / count;

        let centred = players
            .iter()
            .map(|player| CentredPlayer {
                prediction: f64::from(player.prediction_mmr) - mean_prediction,
                label: f64::from(player.target_mmr) - mean_label,
            })
            .collect();

        lobbies.push(CentredLobby {
            players: centred,
            raw: players,
            mean_prediction,
            mean_label,
        });
    }
    lobbies
}

fn within_sums<'a>(lobbies: impl Iterator<Item = &'a CentredLobby>) -> PairSums {
    let mut sums = PairSums::default();
    for lobby in lobbies {
        for player in &lobby.players {
            sums.add(player.prediction, player.label);
        }
    }
    sums
}

/// Deterministic 64-bit generator for the bootstrap, so a dump always scores identically.
const fn split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn bootstrap_standard_error(lobbies: &[CentredLobby]) -> f64 {
    if lobbies.len() < 2 {
        return f64::NAN;
    }
    let mut state = 0x5EED_u64;
    let mut estimates = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for _ in 0..BOOTSTRAP_RESAMPLES {
        let resample = (0..lobbies.len()).filter_map(|_| {
            let index = (split_mix(&mut state) % lobbies.len() as u64) as usize;
            lobbies.get(index)
        });
        estimates.push(within_sums(resample).pearson());
    }
    let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let variance =
        estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (estimates.len() as f64 - 1.0);
    variance.sqrt()
}

/// Scores a set of whole-match predictions.
#[must_use]
pub fn compute_lobby_metrics(predictions: &[PlayerPrediction]) -> LobbyMetrics {
    let lobbies = centre_lobbies(predictions);
    if lobbies.is_empty() {
        return LobbyMetrics::default();
    }

    let sums = within_sums(lobbies.iter());
    let slope = sums.slope();

    let mut label_square_sum = 0.0;
    let mut raw_square_sum = 0.0;
    let mut shrunk_square_sum = 0.0;
    let mut player_square_sum = 0.0;
    let mut lobby_square_sum = 0.0;
    let mut concordant = 0.0;
    let mut concordance_pairs = 0usize;

    for lobby in &lobbies {
        lobby_square_sum += (lobby.mean_prediction - lobby.mean_label).powi(2);
        for player in &lobby.players {
            label_square_sum += player.label.powi(2);
            raw_square_sum += (player.label - player.prediction).powi(2);
            shrunk_square_sum += slope.mul_add(-player.prediction, player.label).powi(2);
        }
        for raw in &lobby.raw {
            player_square_sum += f64::from(raw.prediction_mmr - raw.target_mmr).powi(2);
        }
        for (index, first) in lobby.raw.iter().enumerate() {
            for second in lobby.raw.iter().skip(index + 1) {
                let label_gap = first.target_mmr - second.target_mmr;
                if label_gap.abs() <= CONCORDANCE_MINIMUM_LABEL_GAP_MMR {
                    continue;
                }
                let prediction_gap = first.prediction_mmr - second.prediction_mmr;
                concordance_pairs += 1;
                if prediction_gap == 0.0 {
                    concordant += 0.5;
                } else if (prediction_gap > 0.0) == (label_gap > 0.0) {
                    concordant += 1.0;
                }
            }
        }
    }

    let player_count = sums.count as usize;
    let players = sums.count;
    LobbyMetrics {
        lobby_count: lobbies.len(),
        player_count,
        within_r: sums.pearson(),
        within_r_standard_error: bootstrap_standard_error(&lobbies),
        within_slope: slope,
        label_within_rms: (label_square_sum / players).sqrt(),
        within_rmse: (raw_square_sum / players).sqrt(),
        within_rmse_shrunk: (shrunk_square_sum / players).sqrt(),
        lobby_rmse: (lobby_square_sum / lobbies.len() as f64).sqrt(),
        player_rmse: (player_square_sum / players).sqrt(),
        concordance: if concordance_pairs > 0 {
            concordant / concordance_pairs as f64
        } else {
            f64::NAN
        },
        concordance_pairs,
    }
}

/// Label margin over the lobby median that makes a player a smurf-proxy positive.
///
/// Mostly parties with a stronger friend: smurf-like lobbies where the true rank is known.
pub const PROXY_POSITIVE_MARGIN_MMR: f32 = 150.0;

/// How the smurf flag performs at one margin.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlagMetrics {
    /// Margin over the lobby's median prediction.
    pub margin_mmr: f32,
    /// Players flagged.
    pub flagged: usize,
    /// Flagged players who are proxy positives.
    pub correct: usize,
    /// Proxy positives in the set.
    pub positives: usize,
    /// Players scored.
    pub players: usize,
}

impl FlagMetrics {
    /// Share of flags that are correct.
    #[must_use]
    pub fn precision(&self) -> f64 {
        self.correct as f64 / self.flagged.max(1) as f64
    }

    /// Share of proxy positives that get flagged.
    #[must_use]
    pub fn recall(&self) -> f64 {
        self.correct as f64 / self.positives.max(1) as f64
    }

    /// Share of all players flagged.
    #[must_use]
    pub fn flag_rate(&self) -> f64 {
        self.flagged as f64 / self.players.max(1) as f64
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    let upper = values.get(middle).copied().unwrap_or(0.0);
    if values.len().is_multiple_of(2) {
        let lower = values.get(middle.wrapping_sub(1)).copied().unwrap_or(upper);
        f32::midpoint(lower, upper)
    } else {
        upper
    }
}

/// Scores the app's rule — flag when prediction > lobby median prediction + `margin_mmr` —
/// against the proxy positives, over every labelled player in lobbies with ≥ 2 labels.
#[must_use]
pub fn flag_metrics(predictions: &[PlayerPrediction], margin_mmr: f32) -> FlagMetrics {
    let mut by_replay: HashMap<Uuid, Vec<PlayerPrediction>> = HashMap::new();
    for prediction in predictions.iter().filter(|p| p.target_mmr > 0.0) {
        by_replay
            .entry(prediction.replay_id)
            .or_default()
            .push(*prediction);
    }
    let mut metrics = FlagMetrics {
        margin_mmr,
        ..FlagMetrics::default()
    };
    for players in by_replay.values().filter(|players| players.len() >= 2) {
        let prediction_median =
            median(&mut players.iter().map(|p| p.prediction_mmr).collect::<Vec<_>>());
        let label_median = median(&mut players.iter().map(|p| p.target_mmr).collect::<Vec<_>>());
        for player in players {
            let positive = player.target_mmr - label_median >= PROXY_POSITIVE_MARGIN_MMR;
            let flagged = player.prediction_mmr > prediction_median + margin_mmr;
            metrics.players += 1;
            metrics.positives += usize::from(positive);
            metrics.flagged += usize::from(flagged);
            metrics.correct += usize::from(flagged && positive);
        }
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lobby(replay: u128, rows: &[[f32; 2]]) -> Vec<PlayerPrediction> {
        rows.iter()
            .enumerate()
            .map(|(slot, row)| PlayerPrediction {
                replay_id: Uuid::from_u128(replay),
                slot,
                prediction_mmr: row[0],
                target_mmr: row[1],
            })
            .collect()
    }

    /// Predictions that equal the labels plus a per-lobby offset are perfect within-lobby.
    #[test]
    fn perfect_within_lobby_ordering_scores_one() {
        let mut predictions = lobby(
            1,
            &[
                [1100.0, 1000.0],
                [1200.0, 1100.0],
                [1300.0, 1200.0],
                [1400.0, 1300.0],
            ],
        );
        predictions.extend(lobby(
            2,
            &[
                [500.0, 600.0],
                [520.0, 620.0],
                [700.0, 800.0],
                [650.0, 750.0],
            ],
        ));
        let metrics = compute_lobby_metrics(&predictions);
        assert_eq!(metrics.lobby_count, 2);
        assert!((metrics.within_r - 1.0).abs() < 1e-9);
        assert!((metrics.within_slope - 1.0).abs() < 1e-9);
        assert!(metrics.within_rmse < 1e-6);
        assert!((metrics.concordance - 1.0).abs() < 1e-9);
        assert!((metrics.lobby_rmse - 100.0).abs() < 1e-6);
    }

    /// A model that predicts the same value for every player in a lobby has no within-lobby
    /// signal: r is 0, concordance is exactly chance, and its within-RMSE equals the label
    /// spread.
    #[test]
    fn constant_per_lobby_scores_chance() {
        let predictions = lobby(
            1,
            &[
                [900.0, 800.0],
                [900.0, 900.0],
                [900.0, 1000.0],
                [900.0, 1100.0],
            ],
        );
        let metrics = compute_lobby_metrics(&predictions);
        assert!(metrics.within_r.abs() < 1e-9);
        assert!((metrics.concordance - 0.5).abs() < 1e-9);
        assert!((metrics.within_rmse - metrics.label_within_rms).abs() < 1e-6);
    }

    /// Deviations twice as large as the labelled ones keep r = 1 but halve the slope, and the
    /// shrunk RMSE removes the excess.
    #[test]
    fn over_dispersed_deviations_have_half_slope() {
        let predictions = lobby(
            1,
            &[
                [800.0, 900.0],
                [1000.0, 1000.0],
                [1200.0, 1100.0],
                [1000.0, 1000.0],
            ],
        );
        let metrics = compute_lobby_metrics(&predictions);
        assert!((metrics.within_r - 1.0).abs() < 1e-9);
        assert!((metrics.within_slope - 0.5).abs() < 1e-9);
        assert!(metrics.within_rmse > 1.0);
        assert!(metrics.within_rmse_shrunk < 1e-6);
    }

    /// Lobbies with too few rank-known players and unknown labels are excluded.
    #[test]
    fn small_lobbies_and_unknown_labels_are_excluded() {
        let mut predictions = lobby(1, &[[900.0, 800.0], [900.0, 0.0], [900.0, 1000.0]]);
        predictions.extend(lobby(
            2,
            &[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 0.0], [5.0, 5.0]],
        ));
        let metrics = compute_lobby_metrics(&predictions);
        assert_eq!(metrics.lobby_count, 1);
        assert_eq!(metrics.player_count, 4);
    }
}
