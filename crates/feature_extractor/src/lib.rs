#![expect(clippy::indexing_slicing)]

//! Feature extractor crate for Rocket League ML model.
//!
//! This crate transforms raw replay frame data into player-centric ML-ready
//! feature vectors that can be used for training and inference.
//!
//! Each player gets their own feature vector centered on their perspective,
//! enabling the model to learn individual skill patterns.

use std::collections::{BTreeSet, HashMap};

use replay_structs::{GameFrame, GoalEvent, ParsedReplay, PlayerState, Quaternion, Team, Vector3};

/// Feature count for player-centric representation.
///
/// Breakdown:
/// - Ball state: 7
/// - This player state: 13
/// - This player ball relationship: 2
/// - Per-player cumulative segment stats: 5
///   (boost_collected, boost_spent, airborne_fraction, supersonic_fraction, demo_received_fraction)
/// - Teammate 1 state: 13
/// - Teammate 2 state: 13
/// - Teammate relationships: 4
/// - Opponent 1 state: 12 (no boost)
/// - Opponent 2 state: 12 (no boost)
/// - Opponent 3 state: 12 (no boost)
/// - Opponent relationships: 6
/// - Game context: 1 (ball-to-goal distance)
/// - Score + time context: 3 (score_diff_normalized, seconds_remaining_normalized, is_overtime)
/// - Role indicators: 3 (closest/second/third to ball on team)
/// - Total: 106
pub const PLAYER_CENTRIC_FEATURE_COUNT: usize = 106;

/// Number of features that belong solely to the focal player (ball + self + ball-relative + cumulative).
///
/// These are the features used by the per-player skill encoder in the split-encoder architecture.
/// Indices 0 .. SELF_PLAYER_FEATURE_COUNT.
pub const SELF_PLAYER_FEATURE_COUNT: usize = 7 + 13 + 2 + 5; // = 27

/// Number of players expected in a 3v3 match.
pub const PLAYERS_PER_TEAM: usize = 3;
pub const TOTAL_PLAYERS: usize = PLAYERS_PER_TEAM * 2;

/// Field dimensions for normalization (Rocket League standard field).
pub mod field {
    /// Half-length of the field (X axis).
    pub const HALF_LENGTH: f32 = 5120.0;
    /// Half-width of the field (Y axis) - also goal Y position.
    pub const HALF_WIDTH: f32 = 5120.0;
    /// Maximum height of the field (Z axis).
    pub const MAX_HEIGHT: f32 = 2044.0;
    /// Field diagonal for distance normalization.
    pub const DIAGONAL: f32 = 7245.0; // sqrt(5120^2 + 5120^2)

    /// Maximum car speed (supersonic).
    pub const MAX_CAR_SPEED: f32 = 2300.0;
    /// Maximum ball speed (approximate).
    pub const MAX_BALL_SPEED: f32 = 6000.0;

    /// Blue goal Y position.
    pub const BLUE_GOAL_Y: f32 = -5120.0;
    /// Orange goal Y position.
    pub const ORANGE_GOAL_Y: f32 = 5120.0;
}

/// A player's actual MMR rating used as training label.
#[derive(Debug, Clone, Default)]
pub struct PlayerRating {
    pub player_name: String,
    pub mmr: i32,
    pub team: i16,
}

/// Cumulative per-player state tracked within a segment.
///
/// All values are reset to zero at each segment boundary.
#[derive(Debug, Clone, Default)]
struct CumulativePlayerState {
    /// Total boost gained (in boost units, 0-100 scale, clamped).
    boost_collected: f32,
    /// Total boost spent (boost units).
    boost_spent: f32,
    /// Number of subsampled frames spent airborne (z > `AIRBORNE_Z_THRESHOLD`).
    frames_airborne: u32,
    /// Number of subsampled frames at supersonic speed (> `SUPERSONIC_SPEED`).
    frames_supersonic: u32,
    /// Number of subsampled frames where this player was demolished.
    frames_demolished: u32,
    /// Total subsampled frames in segment so far (for computing fractions).
    total_frames: u32,
    /// Previous frame's boost amount (for detecting changes).
    prev_boost: f32,
}

/// Minimum player height above the ground to count as airborne (Rocket League units).
const AIRBORNE_Z_THRESHOLD: f32 = 25.0;

/// Minimum speed to count as supersonic (Rocket League units per second).
const SUPERSONIC_SPEED: f32 = 2200.0;

/// Index of the first of the three score/time context features in the feature vector.
///
/// Layout: `[score_diff_normalized, seconds_remaining_normalized, is_overtime]`. These are
/// the only features that depend on match metadata rather than on the frame itself, which
/// makes them the ones at risk of diverging between the training and inference paths.
/// Named so tests can assert on them; a `debug_assert_eq!` at the write site keeps it
/// honest if the preceding feature blocks ever change size.
pub const SCORE_TIME_CONTEXT_FEATURE_OFFSET: usize = 100;

/// Game context extracted for a single frame (shared across all 6 player views).
#[derive(Debug, Clone, Default)]
struct FrameGameContext {
    /// Normalised score differential from the focal player's perspective:
    /// positive = focal player's team is winning, negative = losing.
    /// Value = (own_team_score - opponent_score) / 5.0, clamped to [-1, 1].
    score_diff_normalized: f32,
    /// Remaining match time normalised to [0, 1]: `seconds_remaining / 300`.
    /// Saturates at 0.0 during overtime (clock does not go negative).
    seconds_remaining_normalized: f32,
    /// 1.0 if the game is in overtime (`seconds_remaining <= 0`), else 0.0.
    is_overtime: f32,
}

/// Player-centric feature vector focused on one specific player.
#[derive(Debug, Clone)]
pub struct PlayerCentricFrameFeatures {
    /// The player-centric feature vector.
    pub features: [f32; PLAYER_CENTRIC_FEATURE_COUNT],
    /// Timestamp of the frame.
    pub time: f32,
}

impl Default for PlayerCentricFrameFeatures {
    fn default() -> Self {
        Self {
            features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
            time: 0.0,
        }
    }
}

/// Canonical ordered roster for a 3v3 match.
///
/// Slots 0–2 hold the blue team sorted alphabetically by player name;
/// slots 3–5 hold the orange team sorted alphabetically by player name.
///
/// This ordering is identical to the one produced by `build_target_mmr_array`,
/// so slot `i` in the feature vector always maps to slot `i` in the target
/// MMR array.  The roster is built **once per game** and then used for every
/// frame, which means the slot assignment stays stable even after a player
/// disconnects mid-match: the disconnected slot emits all-zero features while
/// all remaining slots keep their canonical positions.
#[derive(Debug, Clone)]
pub struct PlayerRoster {
    /// Six canonical player names: [blue_0, blue_1, blue_2, orange_0, orange_1, orange_2].
    pub names: [String; TOTAL_PLAYERS],
}

impl PlayerRoster {
    /// Builds the roster from player ratings (training path).
    ///
    /// The sort order mirrors `build_target_mmr_array` so that every feature
    /// slot corresponds to the correct target MMR entry.
    #[must_use]
    pub fn from_player_ratings(player_ratings: &[PlayerRating]) -> Self {
        let mut blue: Vec<&PlayerRating> = player_ratings.iter().filter(|r| r.team == 0).collect();
        let mut orange: Vec<&PlayerRating> =
            player_ratings.iter().filter(|r| r.team == 1).collect();

        blue.sort_by(|a, b| a.player_name.cmp(&b.player_name));
        orange.sort_by(|a, b| a.player_name.cmp(&b.player_name));

        let names = core::array::from_fn(|i| {
            if i < PLAYERS_PER_TEAM {
                blue.get(i)
                    .map_or_else(String::new, |r| r.player_name.clone())
            } else {
                orange
                    .get(i - PLAYERS_PER_TEAM)
                    .map_or_else(String::new, |r| r.player_name.clone())
            }
        });

        Self { names }
    }

    /// Builds the roster by scanning all frames in a replay (inference path).
    ///
    /// Collects every unique player name per team observed across all frames,
    /// then sorts them alphabetically — reproducing the same canonical order as
    /// the training path without requiring external metadata.
    #[must_use]
    pub fn from_frames(frames: &[GameFrame]) -> Self {
        let mut blue: BTreeSet<String> = BTreeSet::new();
        let mut orange: BTreeSet<String> = BTreeSet::new();

        for frame in frames {
            for player in &frame.players {
                let name = (*player.name).clone();
                match player.team {
                    Team::Blue => {
                        blue.insert(name);
                    }
                    Team::Orange => {
                        orange.insert(name);
                    }
                }
            }
        }

        let names = core::array::from_fn(|i| {
            if i < PLAYERS_PER_TEAM {
                blue.iter().nth(i).cloned().unwrap_or_default()
            } else {
                orange
                    .iter()
                    .nth(i - PLAYERS_PER_TEAM)
                    .cloned()
                    .unwrap_or_default()
            }
        });

        Self { names }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Normalizes X position to [-1, 1].
fn normalize_x(x: f32) -> f32 {
    (x / field::HALF_LENGTH).clamp(-1.0, 1.0)
}

/// Normalizes Y position to [-1, 1].
fn normalize_y(y: f32) -> f32 {
    (y / field::HALF_WIDTH).clamp(-1.0, 1.0)
}

/// Normalizes Z position to [0, 1].
fn normalize_z(z: f32) -> f32 {
    (z / field::MAX_HEIGHT).clamp(0.0, 1.0)
}

/// Normalizes ball velocity component to [-1, 1].
fn normalize_ball_velocity(v: f32) -> f32 {
    (v / field::MAX_BALL_SPEED).clamp(-1.0, 1.0)
}

/// Normalizes car velocity component to [-1, 1].
fn normalize_car_velocity(v: f32) -> f32 {
    (v / field::MAX_CAR_SPEED).clamp(-1.0, 1.0)
}

/// Computes the magnitude of a 3D vector.
fn vector_magnitude(v: &Vector3) -> f32 {
    v.z.mul_add(v.z, v.x.mul_add(v.x, v.y * v.y)).sqrt()
}

/// Computes the distance between two 3D points.
fn distance(a: &Vector3, b: &Vector3) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

/// Computes the normalized direction from point a to point b.
fn direction(from: &Vector3, to: &Vector3) -> Vector3 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let dz = to.z - from.z;
    let mag = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

    if mag < 0.0001 {
        Vector3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    } else {
        Vector3 {
            x: dx / mag,
            y: dy / mag,
            z: dz / mag,
        }
    }
}

/// Dot product of two 3D vectors.
fn dot(a: &Vector3, b: &Vector3) -> f32 {
    a.z.mul_add(b.z, a.x.mul_add(b.x, a.y * b.y))
}

/// Extracts the forward vector from a quaternion rotation.
/// In Rocket League, the default forward is along the positive X axis.
fn quaternion_forward(q: &Quaternion) -> Vector3 {
    // Rotate the unit X vector by the quaternion
    // forward = q * (1, 0, 0) * q^(-1)
    // Simplified formula for rotating (1, 0, 0):
    let x = 2.0f32.mul_add(-q.y.mul_add(q.y, q.z * q.z), 1.0);
    let y = 2.0 * q.x.mul_add(q.y, q.w * q.z);
    let z = 2.0 * q.x.mul_add(q.z, -(q.w * q.y));

    Vector3 { x, y, z }
}

/// Builds a fixed-size target MMR array from player ratings.
fn build_target_mmr_array(player_ratings: &[PlayerRating]) -> [f32; TOTAL_PLAYERS] {
    let mut target_mmr = [1000.0f32; TOTAL_PLAYERS];

    // Separate by team and sort
    let mut blue_ratings: Vec<_> = player_ratings.iter().filter(|r| r.team == 0).collect();
    let mut orange_ratings: Vec<_> = player_ratings.iter().filter(|r| r.team == 1).collect();

    // Sort by name for consistent ordering (since we don't have actor_id here)
    blue_ratings.sort_by(|a, b| a.player_name.cmp(&b.player_name));
    orange_ratings.sort_by(|a, b| a.player_name.cmp(&b.player_name));

    // Fill blue team (first 3 slots)
    for (i, rating) in blue_ratings.iter().take(PLAYERS_PER_TEAM).enumerate() {
        target_mmr[i] = rating.mmr as f32;
    }

    // Fill orange team (next 3 slots)
    for (i, rating) in orange_ratings.iter().take(PLAYERS_PER_TEAM).enumerate() {
        target_mmr[PLAYERS_PER_TEAM + i] = rating.mmr as f32;
    }

    target_mmr
}

// ============================================================================
// Player-Centric Feature Extraction
// ============================================================================

/// Extracts player-centric features for one canonical player slot from a frame.
///
/// Uses a pre-built name → state map so that each slot is looked up by the
/// player's canonical name rather than by their current position in the frame's
/// player list.  This makes slot assignments stable across car respawns and
/// disconnects: if the player named by `roster.names[player_index]` is absent
/// from this frame, an all-zero feature vector is returned for that slot while
/// all other slots keep their canonical positions.
///
/// # Arguments
///
/// * `frame` - The game frame to extract features from.
/// * `player_index` - Canonical slot (0-2 blue, 3-5 orange).
/// * `roster` - The fixed canonical name list for this match.
/// * `player_map` - Pre-built `name → &PlayerState` map for the current frame.
/// * `cumulative` - Cumulative per-player stats accumulated since segment start.
/// * `game_ctx` - Per-frame game context (score diff, time remaining).
/// * `role_closest` - Whether this player is closest/second/third to ball on team.
pub(crate) fn extract_player_centric_frame_features(
    frame: &GameFrame,
    player_index: usize,
    roster: &PlayerRoster,
    player_map: &HashMap<&str, &PlayerState>,
    cumulative: &CumulativePlayerState,
    game_ctx: &FrameGameContext,
    role_rank: u8,
) -> PlayerCentricFrameFeatures {
    let mut features = PlayerCentricFrameFeatures {
        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
        time: frame.time,
    };

    let canonical_name = match roster.names.get(player_index) {
        Some(name) => name.as_str(),
        None => return features,
    };

    let Some(&this_player) = player_map.get(canonical_name) else {
        return features;
    };

    let (team_start, opp_start) = if player_index < PLAYERS_PER_TEAM {
        (0, PLAYERS_PER_TEAM)
    } else {
        (PLAYERS_PER_TEAM, 0)
    };

    // Canonical teammate slots (excluding self), each may be None if disconnected.
    let teammates: Vec<Option<&PlayerState>> = (team_start..team_start + PLAYERS_PER_TEAM)
        .filter(|&i| i != player_index)
        .map(|i| {
            roster
                .names
                .get(i)
                .and_then(|name| player_map.get(name.as_str()).copied())
        })
        .collect();

    // Canonical opponent slots in alphabetical order, each may be None if disconnected.
    let opponents: Vec<Option<&PlayerState>> = (opp_start..opp_start + PLAYERS_PER_TEAM)
        .map(|i| {
            roster
                .names
                .get(i)
                .and_then(|name| player_map.get(name.as_str()).copied())
        })
        .collect();

    let mut idx = 0;

    // 1. Ball state (7 features)
    features.features[idx] = normalize_x(frame.ball.position.x);
    features.features[idx + 1] = normalize_y(frame.ball.position.y);
    features.features[idx + 2] = normalize_z(frame.ball.position.z);
    features.features[idx + 3] = normalize_ball_velocity(frame.ball.velocity.x);
    features.features[idx + 4] = normalize_ball_velocity(frame.ball.velocity.y);
    features.features[idx + 5] = normalize_ball_velocity(frame.ball.velocity.z);
    let ball_speed = vector_magnitude(&frame.ball.velocity);
    features.features[idx + 6] = (ball_speed / field::MAX_BALL_SPEED).min(1.0);
    idx += 7;

    // 2. This player's state (13 features)
    write_player_state_to_slice(&mut features.features[idx..idx + 13], this_player);
    idx += 13;

    // 3. This player's ball relationship (2 features)
    if !this_player.actor_state.is_demolished {
        let dist_to_ball = distance(&this_player.actor_state.position, &frame.ball.position);
        features.features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);
        let forward = quaternion_forward(&this_player.actor_state.rotation);
        let to_ball = direction(&this_player.actor_state.position, &frame.ball.position);
        features.features[idx + 1] = dot(&forward, &to_ball);
    }
    idx += 2;

    // 4. Per-player cumulative segment stats (5 features)
    //    These reset at each segment boundary so the model sees within-segment effort.
    let seg_max_boost = 300.0_f32; // rough normalisation ceiling per segment
    features.features[idx] = (cumulative.boost_collected / seg_max_boost).min(1.0);
    features.features[idx + 1] = (cumulative.boost_spent / seg_max_boost).min(1.0);
    let total = cumulative.total_frames.max(1) as f32;
    features.features[idx + 2] = cumulative.frames_airborne as f32 / total;
    features.features[idx + 3] = cumulative.frames_supersonic as f32 / total;
    features.features[idx + 4] = cumulative.frames_demolished as f32 / total;
    idx += 5;

    // 5. Teammate 1 state (13 features) — zeros if absent
    if let Some(t1) = teammates.first().and_then(|t| *t) {
        write_player_state_to_slice(&mut features.features[idx..idx + 13], t1);
    }
    idx += 13;

    // 6. Teammate 2 state (13 features) — zeros if absent
    if let Some(t2) = teammates.get(1).and_then(|t| *t) {
        write_player_state_to_slice(&mut features.features[idx..idx + 13], t2);
    }
    idx += 13;

    // 7. Teammate relationships (4 features)
    if let Some(t1) = teammates.first().and_then(|t| *t)
        && !t1.actor_state.is_demolished
    {
        let dist_to_ball = distance(&t1.actor_state.position, &frame.ball.position);
        features.features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);
        let dist_to_this = distance(&t1.actor_state.position, &this_player.actor_state.position);
        features.features[idx + 1] = (dist_to_this / field::DIAGONAL).min(1.0);
    }
    if let Some(t2) = teammates.get(1).and_then(|t| *t)
        && !t2.actor_state.is_demolished
    {
        let dist_to_ball = distance(&t2.actor_state.position, &frame.ball.position);
        features.features[idx + 2] = (dist_to_ball / field::DIAGONAL).min(1.0);
        let dist_to_this = distance(&t2.actor_state.position, &this_player.actor_state.position);
        features.features[idx + 3] = (dist_to_this / field::DIAGONAL).min(1.0);
    }
    idx += 4;

    // 8–10. Opponent states (12 features each, NO BOOST)
    for i in 0..3 {
        if let Some(opp) = opponents.get(i).and_then(|t| *t) {
            write_opponent_state_to_slice(&mut features.features[idx..idx + 12], opp);
        }
        idx += 12;
    }

    // 11. Opponent relationships (6 features)
    for i in 0..3 {
        if let Some(opp) = opponents.get(i).and_then(|t| *t)
            && !opp.actor_state.is_demolished
        {
            let dist_to_ball = distance(&opp.actor_state.position, &frame.ball.position);
            features.features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);
            let dist_to_this =
                distance(&opp.actor_state.position, &this_player.actor_state.position);
            features.features[idx + 1] = (dist_to_this / field::DIAGONAL).min(1.0);
        }
        idx += 2;
    }

    // 12. Ball-to-blue-goal distance (1 feature)
    let blue_goal = Vector3 {
        x: 0.0,
        y: field::BLUE_GOAL_Y,
        z: 0.0,
    };
    features.features[idx] =
        (distance(&frame.ball.position, &blue_goal) / field::DIAGONAL).min(1.0);
    idx += 1;

    // 13. Score + time context (3 features)
    debug_assert_eq!(
        idx, SCORE_TIME_CONTEXT_FEATURE_OFFSET,
        "score/time context moved; update SCORE_TIME_CONTEXT_FEATURE_OFFSET"
    );
    features.features[idx] = game_ctx.score_diff_normalized;
    features.features[idx + 1] = game_ctx.seconds_remaining_normalized;
    features.features[idx + 2] = game_ctx.is_overtime;
    idx += 3;

    // 14. Role indicators (3 features): closest/second/third to ball on own team.
    //     role_rank 0 = closest, 1 = second, 2 = third.
    features.features[idx] = if role_rank == 0 { 1.0 } else { 0.0 };
    features.features[idx + 1] = if role_rank == 1 { 1.0 } else { 0.0 };
    features.features[idx + 2] = if role_rank == 2 { 1.0 } else { 0.0 };

    features
}

/// Writes player state to a slice (13 features).
fn write_player_state_to_slice(slice: &mut [f32], player: &PlayerState) {
    if player.actor_state.is_demolished {
        // All zeros except demolished flag
        for val in slice.iter_mut().take(12) {
            *val = 0.0;
        }
        slice[12] = 1.0;
    } else {
        // Position
        slice[0] = normalize_x(player.actor_state.position.x);
        slice[1] = normalize_y(player.actor_state.position.y);
        slice[2] = normalize_z(player.actor_state.position.z);

        // Velocity
        slice[3] = normalize_car_velocity(player.actor_state.velocity.x);
        slice[4] = normalize_car_velocity(player.actor_state.velocity.y);
        slice[5] = normalize_car_velocity(player.actor_state.velocity.z);

        // Rotation
        slice[6] = player.actor_state.rotation.x;
        slice[7] = player.actor_state.rotation.y;
        slice[8] = player.actor_state.rotation.z;
        slice[9] = player.actor_state.rotation.w;

        // Speed
        let speed = vector_magnitude(&player.actor_state.velocity);
        slice[10] = (speed / field::MAX_CAR_SPEED).min(1.0);

        // Boost
        slice[11] = player.actor_state.boost;

        // Demolished flag
        slice[12] = 0.0;
    }
}

/// Writes opponent state to a slice (12 features, NO BOOST).
/// Same as player state but without boost feature.
fn write_opponent_state_to_slice(slice: &mut [f32], player: &PlayerState) {
    if player.actor_state.is_demolished {
        // All zeros except demolished flag
        for val in slice.iter_mut().take(11) {
            *val = 0.0;
        }
        slice[11] = 1.0;
    } else {
        // Position
        slice[0] = normalize_x(player.actor_state.position.x);
        slice[1] = normalize_y(player.actor_state.position.y);
        slice[2] = normalize_z(player.actor_state.position.z);

        // Velocity
        slice[3] = normalize_car_velocity(player.actor_state.velocity.x);
        slice[4] = normalize_car_velocity(player.actor_state.velocity.y);
        slice[5] = normalize_car_velocity(player.actor_state.velocity.z);

        // Rotation
        slice[6] = player.actor_state.rotation.x;
        slice[7] = player.actor_state.rotation.y;
        slice[8] = player.actor_state.rotation.z;
        slice[9] = player.actor_state.rotation.w;

        // Speed (NO BOOST for opponents)
        let speed = vector_magnitude(&player.actor_state.velocity);
        slice[10] = (speed / field::MAX_CAR_SPEED).min(1.0);

        // Demolished flag
        slice[11] = 0.0;
    }
}

/// Extracts player-centric features for all 6 canonical player slots from a frame.
///
/// Builds the name → state map once, then extracts each slot using roster-based
/// lookup.  Players absent from the frame (disconnected) produce all-zero feature
/// vectors; all other slots keep their canonical positions unchanged.
#[cfg(test)]
fn extract_all_player_centric_features(
    frame: &GameFrame,
    roster: &PlayerRoster,
    cumulatives: &[CumulativePlayerState; TOTAL_PLAYERS],
    game_ctx: &FrameGameContext,
) -> [PlayerCentricFrameFeatures; TOTAL_PLAYERS] {
    let player_map: HashMap<&str, &PlayerState> =
        frame.players.iter().map(|p| (&**p.name, p)).collect();

    // Compute role rank (distance rank to ball) per player within each team.
    // role_rank[i] = 0 if player i is closest to ball on their team, 1 second, 2 third.
    let mut role_ranks = [2u8; TOTAL_PLAYERS];
    for team_offset in [0usize, PLAYERS_PER_TEAM] {
        let mut dists: Vec<(usize, f32)> = (team_offset..team_offset + PLAYERS_PER_TEAM)
            .map(|slot| {
                let dist = roster
                    .names
                    .get(slot)
                    .and_then(|name| player_map.get(name.as_str()))
                    .filter(|ps| !ps.actor_state.is_demolished)
                    .map_or(f32::MAX, |ps| {
                        distance(&ps.actor_state.position, &frame.ball.position)
                    });
                (slot, dist)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (slot, _)) in dists.iter().enumerate() {
            role_ranks[*slot] = rank as u8;
        }
    }

    core::array::from_fn(|i| {
        extract_player_centric_frame_features(
            frame,
            i,
            roster,
            &player_map,
            &cumulatives[i],
            game_ctx,
            role_ranks[i],
        )
    })
}

/// A sequence sample with player-centric features.
#[derive(Debug, Clone)]
pub struct PlayerCentricGameSequence {
    /// Frame features for all 6 players: [num_frames, 6_players, features]
    /// Organized as: for each frame, array of 6 player feature vectors
    pub player_frames: Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]>,
    /// Target MMR for each player [6]
    pub target_mmr: [f32; TOTAL_PLAYERS],
}

/// Frame subsampling rate: take 1 frame out of every N frames.
///
/// At 30fps, taking 1/2 frames gives ~15fps effective sampling rate,
/// preserving finer mechanical timing while keeping sequences manageable.
/// Raised from 3 to 2 (Phase 2) to give 20 s of context at 300 frames.
pub const FRAME_SUBSAMPLE_RATE: usize = 2;

/// Extracts a game sequence sample with player-centric features.
///
/// Trims goal-replay windows (frames between a goal event and the next kickoff)
/// so the model never sees the celebration / teleportation state.  Uses
/// `FRAME_SUBSAMPLE_RATE` subsampling and tracks per-player cumulative stats
/// that reset at every `segment_length` subsampled frames.
///
/// # Arguments
///
/// * `parsed` - The full parsed replay (frames + goal/kickoff metadata).
/// * `player_ratings` - Player ratings with team assignments.
/// * `segment_length` - Number of subsampled frames per training segment
///   (used to reset cumulative stats at each segment boundary).
///
/// # Returns
///
/// A `PlayerCentricGameSequence` with features for all 6 players across live frames.
pub fn extract_player_centric_game_sequence(
    parsed: &ParsedReplay,
    player_ratings: &[PlayerRating],
    segment_length: usize,
) -> PlayerCentricGameSequence {
    let target_mmr = build_target_mmr_array(player_ratings);
    let roster = PlayerRoster::from_player_ratings(player_ratings);

    let player_frames = extract_player_centric_frames_with_context(
        &parsed.frames,
        &roster,
        segment_length,
        Some(SequenceContext::from_parsed(parsed)),
    );

    PlayerCentricGameSequence {
        player_frames,
        target_mmr,
    }
}

/// Match-level metadata that is not recoverable from the frame list alone.
///
/// Everything here comes off a [`ParsedReplay`]. It is optional at the call site so a
/// caller that genuinely only holds frames can still extract features, but every
/// in-tree caller has a `ParsedReplay` and should pass it: omitting the context changes
/// the feature distribution the model sees (see
/// [`extract_player_centric_frames_with_context`]).
#[derive(Debug, Clone, Copy)]
pub struct SequenceContext<'a> {
    /// Scored goals, used to reconstruct the running score at each frame.
    pub goals: &'a [GoalEvent],
    /// Frame indices at which a goal was scored.
    pub goal_frames: &'a [usize],
    /// Frame indices at which play resumed after a kickoff.
    pub kickoff_frames: &'a [usize],
}

impl<'a> SequenceContext<'a> {
    /// Borrows the match metadata out of a parsed replay.
    #[must_use]
    pub fn from_parsed(parsed: &'a ParsedReplay) -> Self {
        Self {
            goals: &parsed.goals,
            goal_frames: &parsed.goal_frames,
            kickoff_frames: &parsed.kickoff_frames,
        }
    }
}

/// Shared feature extraction used by both the training and the inference paths.
///
/// Having exactly one implementation is the point: when training and inference each had
/// their own loop, the inference loop silently disagreed with the training loop on three
/// features, so the model was asked at serving time to score vectors of a kind it had
/// never been trained on.
///
/// `context` controls the two things that need match metadata:
///
/// * **Goal-replay exclusion** — frames inside a goal celebration are dropped, as in
///   training. Without context they are kept.
/// * **Score differential** — reconstructed from the goal list. Without context it is
///   `0.0`, which is in-distribution (a tied game) even though it is not always correct.
///
/// The clock features (`seconds_remaining_normalized`, `is_overtime`) are always derived
/// from `frame.seconds_remaining`, with or without context. They read straight off the
/// frame, so there was never a reason to zero them, and zeroing both at once produced the
/// pair `seconds_remaining_normalized = 0, is_overtime = 0` — a combination that cannot
/// occur in training, where a zero clock always means overtime.
#[must_use]
pub fn extract_player_centric_frames_with_context(
    frames: &[GameFrame],
    roster: &PlayerRoster,
    segment_length: usize,
    context: Option<SequenceContext<'_>>,
) -> Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]> {
    // Only meaningful with context; without it nothing is excluded.
    let excluded = context.map(|ctx| {
        build_goal_replay_excluded_set(ctx.goal_frames, ctx.kickoff_frames)
    });

    // Only meaningful with context; without it the score stays 0-0 for every frame.
    let cumulative_score =
        context.map(|ctx| precompute_cumulative_score(ctx.goals, frames.len()));

    let mut cumulatives = core::array::from_fn(|_| CumulativePlayerState::default());
    let mut player_frames: Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]> = Vec::new();
    let mut subsampled_count = 0usize;

    for (frame_idx, frame) in frames.iter().enumerate() {
        if frame_idx % FRAME_SUBSAMPLE_RATE != 0 {
            continue;
        }
        if excluded
            .as_ref()
            .is_some_and(|set| set.contains(&frame_idx))
        {
            continue;
        }

        // Reset cumulative state at segment boundaries.
        if subsampled_count > 0 && subsampled_count.is_multiple_of(segment_length) {
            for state in &mut cumulatives {
                *state = CumulativePlayerState::default();
            }
        }

        // Build per-frame player map for fast lookup.
        let player_map: HashMap<&str, &PlayerState> =
            frame.players.iter().map(|p| (&**p.name, p)).collect();

        // Update cumulative state for each player slot.
        update_cumulative_states(&mut cumulatives, roster, &player_map);

        // Compute score diff from the focal perspective (team-0 = blue).
        let (blue_score, orange_score) = cumulative_score
            .as_ref()
            .and_then(|scores| scores.get(frame_idx).copied())
            .unwrap_or((0, 0));

        let seconds_remaining_normalized = (frame.seconds_remaining as f32 / 300.0).clamp(0.0, 1.0);
        let is_overtime = if frame.seconds_remaining <= 0 {
            1.0
        } else {
            0.0
        };
        let game_ctx_blue = FrameGameContext {
            score_diff_normalized: ((blue_score as f32 - orange_score as f32) / 5.0)
                .clamp(-1.0, 1.0),
            seconds_remaining_normalized,
            is_overtime,
        };
        let game_ctx_orange = FrameGameContext {
            score_diff_normalized: ((orange_score as f32 - blue_score as f32) / 5.0)
                .clamp(-1.0, 1.0),
            seconds_remaining_normalized,
            is_overtime,
        };

        // Extract all 6 player feature vectors.
        // Blue players (slots 0-2) use game_ctx_blue; orange (slots 3-5) use game_ctx_orange.
        let role_ranks = compute_role_ranks(frame, roster, &player_map);
        let mut frame_features = core::array::from_fn(|_| PlayerCentricFrameFeatures::default());
        for slot in 0..TOTAL_PLAYERS {
            let ctx = if slot < PLAYERS_PER_TEAM {
                &game_ctx_blue
            } else {
                &game_ctx_orange
            };
            frame_features[slot] = extract_player_centric_frame_features(
                frame,
                slot,
                roster,
                &player_map,
                &cumulatives[slot],
                ctx,
                role_ranks[slot],
            );
        }

        player_frames.push(frame_features);
        subsampled_count += 1;
    }

    player_frames
}


/// Builds the set of original frame indices that fall inside goal-replay windows.
///
/// A goal-replay window runs from the goal frame (exclusive) to the next kickoff
/// frame (exclusive).  Frames inside these windows show the celebration camera,
/// car teleportation, and paused clock — they are pure noise for skill prediction.
fn build_goal_replay_excluded_set(
    goal_frames: &[usize],
    kickoff_frames: &[usize],
) -> std::collections::HashSet<usize> {
    let mut excluded = std::collections::HashSet::new();
    for &goal in goal_frames {
        // Find the first kickoff that comes AFTER this goal.
        let next_kickoff = kickoff_frames
            .iter()
            .find(|&&kf| kf > goal)
            .copied()
            .unwrap_or(usize::MAX);
        // Exclude [goal+1 .. next_kickoff).
        for frame_idx in (goal + 1)..next_kickoff.min(goal + 600) {
            excluded.insert(frame_idx);
        }
    }
    excluded
}

/// Precomputes (blue_score, orange_score) for every original frame index.
fn precompute_cumulative_score(goals: &[GoalEvent], num_frames: usize) -> Vec<(u32, u32)> {
    let mut scores = vec![(0u32, 0u32); num_frames];
    let mut blue = 0u32;
    let mut orange = 0u32;
    let mut goal_idx = 0;
    let mut sorted_goals: Vec<&GoalEvent> = goals.iter().collect();
    sorted_goals.sort_by_key(|g| g.frame);

    for (frame_idx, score_slot) in scores.iter_mut().enumerate() {
        while goal_idx < sorted_goals.len() && sorted_goals[goal_idx].frame <= frame_idx {
            match sorted_goals[goal_idx].player_team {
                Team::Blue => blue += 1,
                Team::Orange => orange += 1,
            }
            goal_idx += 1;
        }
        *score_slot = (blue, orange);
    }
    scores
}

/// Updates cumulative per-player state from the current frame.
fn update_cumulative_states(
    cumulatives: &mut [CumulativePlayerState; TOTAL_PLAYERS],
    roster: &PlayerRoster,
    player_map: &HashMap<&str, &PlayerState>,
) {
    for (slot, state) in cumulatives.iter_mut().enumerate() {
        let Some(name) = roster.names.get(slot) else {
            continue;
        };
        let Some(&player) = player_map.get(name.as_str()) else {
            continue;
        };

        state.total_frames += 1;

        // Boost tracking: compare to previous frame's boost.
        let current_boost = player.actor_state.boost * 100.0; // normalise to 0-100
        if current_boost > state.prev_boost + 5.0 {
            // Boost increased significantly → player collected a pad.
            state.boost_collected += current_boost - state.prev_boost;
        } else if current_boost < state.prev_boost - 1.0 {
            // Boost decreased → player used boost.
            state.boost_spent += state.prev_boost - current_boost;
        }
        state.prev_boost = current_boost;

        // Airborne.
        if player.actor_state.position.z > AIRBORNE_Z_THRESHOLD && !player.actor_state.is_demolished
        {
            state.frames_airborne += 1;
        }

        // Supersonic.
        let speed = vector_magnitude(&player.actor_state.velocity);
        if speed > SUPERSONIC_SPEED && !player.actor_state.is_demolished {
            state.frames_supersonic += 1;
        }

        // Demolished.
        if player.actor_state.is_demolished {
            state.frames_demolished += 1;
        }
    }
}

/// Returns role-rank per slot (0 = closest to ball within own team).
fn compute_role_ranks(
    frame: &GameFrame,
    roster: &PlayerRoster,
    player_map: &HashMap<&str, &PlayerState>,
) -> [u8; TOTAL_PLAYERS] {
    let mut role_ranks = [2u8; TOTAL_PLAYERS];
    for team_offset in [0usize, PLAYERS_PER_TEAM] {
        let mut dists: Vec<(usize, f32)> = (team_offset..team_offset + PLAYERS_PER_TEAM)
            .map(|slot| {
                let dist = roster
                    .names
                    .get(slot)
                    .and_then(|name| player_map.get(name.as_str()))
                    .filter(|ps| !ps.actor_state.is_demolished)
                    .map_or(f32::MAX, |ps| {
                        distance(&ps.actor_state.position, &frame.ball.position)
                    });
                (slot, dist)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, (slot, _)) in dists.iter().enumerate() {
            role_ranks[*slot] = rank as u8;
        }
    }
    role_ranks
}

/// Extraction for inference when the caller only holds the frame list.
///
/// Prefer [`extract_player_centric_game_sequence_inference_with_context`]: without match
/// metadata the score differential is pinned to `0.0` and goal-replay frames are kept, so
/// the features drift from what the model was trained on. This entry point exists for
/// callers that genuinely have nothing but frames.
///
/// The roster is recovered from the frames themselves (union of names across all frames,
/// blue then orange, each sorted by name) rather than from database ratings.
#[must_use]
pub fn extract_player_centric_game_sequence_inference(
    frames: &[GameFrame],
    segment_length: usize,
) -> Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]> {
    let roster = PlayerRoster::from_frames(frames);
    extract_player_centric_frames_with_context(frames, &roster, segment_length, None)
}

/// Extraction for inference that reproduces the training feature distribution exactly.
///
/// Same code path, same goal-replay exclusion, and the same reconstructed score
/// differential as [`extract_player_centric_game_sequence`]. The only remaining
/// difference is that the roster comes from the frames instead of from database ratings,
/// which affects slot *labelling*, not the feature values in each slot.
#[must_use]
pub fn extract_player_centric_game_sequence_inference_with_context(
    parsed: &ParsedReplay,
    segment_length: usize,
) -> Vec<[PlayerCentricFrameFeatures; TOTAL_PLAYERS]> {
    let roster = PlayerRoster::from_frames(&parsed.frames);
    extract_player_centric_frames_with_context(
        &parsed.frames,
        &roster,
        segment_length,
        Some(SequenceContext::from_parsed(parsed)),
    )
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, Team};

    use super::*;

    #[test]
    fn test_player_centric_features_differ() {
        // Create a frame with distinct players
        let empty_cumulative: [CumulativePlayerState; TOTAL_PLAYERS] =
            core::array::from_fn(|_| CumulativePlayerState::default());
        let game_ctx = FrameGameContext::default();
        let frame = GameFrame {
            time: 100.0,
            delta: 0.03,
            seconds_remaining: 250,
            ball: BallState {
                position: Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 100.0,
                },
                velocity: Vector3 {
                    x: 1000.0,
                    y: 500.0,
                    z: 0.0,
                },
            },
            players: vec![
                // Blue team
                PlayerState {
                    actor_id: 1,
                    name: Arc::new("Player1".to_string()),
                    team: Team::Blue,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: -1000.0,
                            y: -2000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: 500.0,
                            y: 200.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        boost: 0.3,
                        is_demolished: false,
                    },
                },
                PlayerState {
                    actor_id: 2,
                    name: Arc::new("Player2".to_string()),
                    team: Team::Blue,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: -500.0,
                            y: -1000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: 800.0,
                            y: 400.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        boost: 0.7,
                        is_demolished: false,
                    },
                },
                PlayerState {
                    actor_id: 3,
                    name: Arc::new("Player3".to_string()),
                    team: Team::Blue,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: -2000.0,
                            y: -3000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: 200.0,
                            y: 100.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        boost: 0.5,
                        is_demolished: false,
                    },
                },
                // Orange team
                PlayerState {
                    actor_id: 4,
                    name: Arc::new("Player4".to_string()),
                    team: Team::Orange,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: 1000.0,
                            y: 2000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: -500.0,
                            y: -200.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: core::f32::consts::FRAC_1_SQRT_2,
                            w: core::f32::consts::FRAC_1_SQRT_2,
                        },
                        boost: 0.9,
                        is_demolished: false,
                    },
                },
                PlayerState {
                    actor_id: 5,
                    name: Arc::new("Player5".to_string()),
                    team: Team::Orange,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: 500.0,
                            y: 1000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: -800.0,
                            y: -400.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        boost: 0.1,
                        is_demolished: false,
                    },
                },
                PlayerState {
                    actor_id: 6,
                    name: Arc::new("Player6".to_string()),
                    team: Team::Orange,
                    actor_state: ActorState {
                        position: Vector3 {
                            x: 2000.0,
                            y: 3000.0,
                            z: 17.0,
                        },
                        velocity: Vector3 {
                            x: -200.0,
                            y: -100.0,
                            z: 0.0,
                        },
                        rotation: Quaternion {
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            w: 1.0,
                        },
                        boost: 0.6,
                        is_demolished: false,
                    },
                },
            ],
        };

        // Build roster from the single test frame, then extract features.
        let roster = PlayerRoster::from_frames(std::slice::from_ref(&frame));
        let all_features =
            extract_all_player_centric_features(&frame, &roster, &empty_cumulative, &game_ctx);

        // Verify we get 6 feature vectors
        assert_eq!(all_features.len(), 6);

        // Verify each player's features are different
        // Player 1's "this player" boost should be 0.3
        assert!((all_features[0].features[18] - 0.3).abs() < 0.01);

        // Player 2's "this player" boost should be 0.7
        assert!((all_features[1].features[18] - 0.7).abs() < 0.01);

        // Player 4's "this player" boost should be 0.9
        assert!((all_features[3].features[18] - 0.9).abs() < 0.01);

        // Verify all feature vectors have correct size
        for player_features in &all_features {
            assert_eq!(player_features.features.len(), PLAYER_CENTRIC_FEATURE_COUNT);
        }

        // Verify ball features are the same for all players (indices 0-6)
        for i in 0..7 {
            let first_ball_feature = all_features[0].features[i];
            for player_features in &all_features {
                assert!((player_features.features[i] - first_ball_feature).abs() < 0.0001);
            }
        }
    }

    #[test]
    fn test_player_centric_feature_count() {
        // Ball: 7, This player: 13, Ball relationship: 2
        // Cumulative self stats: 5
        // Teammate 1: 13, Teammate 2: 13, Teammate relationships: 4
        // Opponent 1-3: 12×3, Opponent relationships: 6
        // Ball-to-goal: 1, Score+time: 3, Role indicators: 3
        let expected = 7 + 13 + 2 + 5 + 13 + 13 + 4 + 12 + 12 + 12 + 6 + 1 + 3 + 3;
        assert_eq!(expected, PLAYER_CENTRIC_FEATURE_COUNT);
    }

    #[test]
    fn test_normalization_bounds() {
        // Position at field boundary should normalize to 1.0
        assert!((normalize_x(field::HALF_LENGTH) - 1.0).abs() < f32::EPSILON);
        assert!((normalize_y(field::HALF_WIDTH) - 1.0).abs() < f32::EPSILON);
        assert!((normalize_z(field::MAX_HEIGHT) - 1.0).abs() < f32::EPSILON);

        // Beyond bounds should clamp
        assert!((normalize_x(10000.0) - 1.0).abs() < f32::EPSILON);
        assert!((normalize_x(-10000.0) - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quaternion_forward() {
        // Identity quaternion should give forward along X
        let identity = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let forward = quaternion_forward(&identity);
        assert!((forward.x - 1.0).abs() < 0.001);
        assert!(forward.y.abs() < 0.001);
        assert!(forward.z.abs() < 0.001);
    }

    // =========================================================================
    // F4 regression: the training and inference feature paths must agree.
    //
    // These guard the fix for the train/inference skew. Before it, inference ran a
    // separate loop that pinned all three score/time context features to zero and kept
    // goal-replay frames, so the model was served vectors unlike anything in its
    // training set — including `seconds_remaining_normalized == 0` together with
    // `is_overtime == 0`, which training can never produce.
    // =========================================================================

    /// Builds a player in a fixed slot, offset so no two players coincide.
    fn f4_player(name: &str, team: Team, offset: f32) -> PlayerState {
        PlayerState {
            actor_id: offset as i32,
            name: Arc::new(name.to_string()),
            team,
            actor_state: ActorState {
                position: Vector3 {
                    x: offset * 100.0,
                    y: offset * -80.0,
                    z: 17.0,
                },
                velocity: Vector3 {
                    x: offset * 30.0,
                    y: offset * 10.0,
                    z: 0.0,
                },
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                boost: (offset * 0.1).min(1.0),
                is_demolished: false,
            },
        }
    }

    /// A six-player replay whose clock runs into overtime, with one goal and a kickoff.
    ///
    /// The clock crossing zero is the point: it forces frames where `is_overtime == 1`,
    /// which is exactly the combination the old inference path could not represent.
    fn f4_parsed_replay(frame_count: usize) -> ParsedReplay {
        let names = ["Ayla", "Bex", "Cyd", "Dov", "Esk", "Fen"];
        let frames: Vec<GameFrame> = (0..frame_count)
            .map(|i| {
                let players = names
                    .iter()
                    .enumerate()
                    .map(|(slot, name)| {
                        let team = if slot < PLAYERS_PER_TEAM {
                            Team::Blue
                        } else {
                            Team::Orange
                        };
                        f4_player(name, team, (i as f32).mul_add(0.25, (slot + 1) as f32))
                    })
                    .collect();
                GameFrame {
                    time: i as f32 * 0.033,
                    delta: 0.033,
                    // Starts positive, crosses zero into overtime partway through.
                    seconds_remaining: 20 - i as i32,
                    ball: BallState {
                        position: Vector3 {
                            x: i as f32 * 5.0,
                            y: i as f32 * -3.0,
                            z: 100.0,
                        },
                        velocity: Vector3 {
                            x: 300.0,
                            y: -100.0,
                            z: 20.0,
                        },
                    },
                    players,
                }
            })
            .collect();

        ParsedReplay {
            goals: vec![GoalEvent {
                frame: 12,
                player_name: "Ayla".to_string(),
                player_team: Team::Blue,
            }],
            goal_frames: vec![12],
            kickoff_frames: vec![24],
            frames,
        }
    }

    #[test]
    fn test_inference_with_context_matches_training_features() {
        let parsed = f4_parsed_replay(60);
        let ratings: Vec<PlayerRating> = ["Ayla", "Bex", "Cyd", "Dov", "Esk", "Fen"]
            .iter()
            .enumerate()
            .map(|(slot, name)| PlayerRating {
                player_name: (*name).to_string(),
                mmr: 800,
                team: i16::from(slot >= PLAYERS_PER_TEAM),
            })
            .collect();

        let training = extract_player_centric_game_sequence(&parsed, &ratings, 10);
        let inference = extract_player_centric_game_sequence_inference_with_context(&parsed, 10);

        // Same frames survive goal-replay exclusion on both paths.
        assert_eq!(
            training.player_frames.len(),
            inference.len(),
            "training and inference kept a different number of frames"
        );
        assert!(
            !inference.is_empty(),
            "fixture produced no frames; the test would pass vacuously"
        );

        // And every feature of every slot of every frame is bit-for-bit identical.
        // Exact equality is deliberate: the two paths run the same arithmetic on the
        // same inputs, so any difference at all is the skew reappearing.
        #[expect(clippy::float_cmp, reason = "bit-identical output is the property under test")]
        for (frame_idx, (train_frame, inf_frame)) in
            training.player_frames.iter().zip(inference.iter()).enumerate()
        {
            for (slot, (train_slot, inf_slot)) in
                train_frame.iter().zip(inf_frame.iter()).enumerate()
            {
                assert_eq!(
                    train_slot.features, inf_slot.features,
                    "features diverged at frame {frame_idx}, slot {slot}"
                );
            }
        }
    }

    #[test]
    fn test_training_path_excludes_goal_replay_frames() {
        // Guards the fixture itself: if exclusion stopped happening, the equality test
        // above would still pass while silently comparing two unexcluded paths.
        let parsed = f4_parsed_replay(60);
        let roster = PlayerRoster::from_frames(&parsed.frames);

        let with_context = extract_player_centric_frames_with_context(
            &parsed.frames,
            &roster,
            10,
            Some(SequenceContext::from_parsed(&parsed)),
        );
        let without_context =
            extract_player_centric_frames_with_context(&parsed.frames, &roster, 10, None);

        assert!(
            with_context.len() < without_context.len(),
            "goal-replay frames were not excluded when context was supplied"
        );
    }

    #[test]
    fn test_clock_features_never_take_the_impossible_combination() {
        // `seconds_remaining_normalized == 0` means the clock has run out, which always
        // means overtime. The pair (0, 0) is unreachable in training, so no inference path
        // may emit it — that was the concrete symptom of the skew.
        let parsed = f4_parsed_replay(60);
        let seconds_idx = SCORE_TIME_CONTEXT_FEATURE_OFFSET + 1;
        let overtime_idx = SCORE_TIME_CONTEXT_FEATURE_OFFSET + 2;

        let with_context = extract_player_centric_game_sequence_inference_with_context(&parsed, 10);
        let without_context =
            extract_player_centric_game_sequence_inference(&parsed.frames, 10);

        let mut saw_overtime = false;
        for (label, frames) in [
            ("with context", &with_context),
            ("without context", &without_context),
        ] {
            for (frame_idx, frame) in frames.iter().enumerate() {
                for (slot, player) in frame.iter().enumerate() {
                    // These features are written as literal 0.0 / 1.0, never computed,
                    // so exact comparison is the correct test.
                    #[expect(clippy::float_cmp, reason = "features are literal 0.0/1.0 flags")]
                    {
                        let seconds = player.features[seconds_idx];
                        let overtime = player.features[overtime_idx];
                        if overtime == 1.0 {
                            saw_overtime = true;
                        }
                        assert!(
                            !(seconds == 0.0 && overtime == 0.0),
                            "{label}: impossible clock pair (0, 0) at frame {frame_idx}, slot {slot}"
                        );
                    }
                }
            }
        }

        assert!(
            saw_overtime,
            "fixture never reached overtime, so the assertion above proved nothing"
        );
    }

    #[test]
    fn test_score_context_is_zero_without_metadata_and_nonzero_with_it() {
        // The one feature that genuinely cannot be recovered from frames alone. This
        // documents the remaining, deliberate difference between the two entry points.
        let parsed = f4_parsed_replay(60);
        let score_idx = SCORE_TIME_CONTEXT_FEATURE_OFFSET;

        let with_context = extract_player_centric_game_sequence_inference_with_context(&parsed, 10);
        let without_context =
            extract_player_centric_game_sequence_inference(&parsed.frames, 10);

        assert!(
            without_context
                .iter()
                .all(|frame| frame.iter().all(|p| p.features[score_idx] == 0.0)),
            "score differential should be zero when no goal list is supplied"
        );
        assert!(
            with_context
                .iter()
                .any(|frame| frame.iter().any(|p| p.features[score_idx] != 0.0)),
            "score differential should be reconstructed from the goal list"
        );
    }

}
