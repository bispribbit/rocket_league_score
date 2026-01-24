#![expect(clippy::indexing_slicing)]

//! Feature extractor crate for Rocket League ML model.
//!
//! This crate transforms raw replay frame data into player-centric ML-ready
//! feature vectors that can be used for training and inference.
//!
//! Each player gets their own feature vector centered on their perspective,
//! enabling the model to learn individual skill patterns.

use replay_structs::{GameFrame, PlayerState, Quaternion, Team, Vector3};

/// Feature count for player-centric representation.
///
/// Breakdown:
/// - Ball state: 7
/// - This player state: 13
/// - This player ball relationship: 2
/// - Teammate 1 state: 13
/// - Teammate 2 state: 13
/// - Teammate relationships: 4
/// - Opponent 1 state: 12 (no boost)
/// - Opponent 2 state: 12 (no boost)
/// - Opponent 3 state: 12 (no boost)
/// - Opponent relationships: 6
/// - Game context: 1
/// - Total: 95
pub const PLAYER_CENTRIC_FEATURE_COUNT: usize = 95;

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

// ============================================================================
// Helper Functions
// ============================================================================

/// Sorts players into blue (team 0) and orange (team 1) teams.
/// Pads with default players if fewer than 3 per team.
fn sort_players_by_team(players: &[PlayerState]) -> (Vec<&PlayerState>, Vec<&PlayerState>) {
    let mut blue: Vec<&PlayerState> = players.iter().filter(|p| p.team == Team::Blue).collect();
    let mut orange: Vec<&PlayerState> = players.iter().filter(|p| p.team == Team::Orange).collect();

    // Sort by actor_id for consistent ordering
    blue.sort_by_key(|p| p.actor_id);
    orange.sort_by_key(|p| p.actor_id);

    (blue, orange)
}

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

/// Extracts player-centric features for a specific player from a game frame.
///
/// This creates a feature vector centered on one player's perspective, including:
/// - Ball state
/// - This player's full state
/// - Teammates' full states (with boost)
/// - Opponents' states (WITHOUT boost - information leakage)
/// - Relationships (distances)
/// - Game context
///
/// # Arguments
///
/// * `frame` - The game frame
/// * `player_index` - Which player (0-5: 0-2 blue, 3-5 orange)
///
/// # Returns
///
/// Features centered on this player's perspective
pub fn extract_player_centric_frame_features(
    frame: &GameFrame,
    player_index: usize,
) -> PlayerCentricFrameFeatures {
    let mut features = PlayerCentricFrameFeatures {
        features: [0.0; PLAYER_CENTRIC_FEATURE_COUNT],
        time: frame.time,
    };

    // Get sorted players
    let (blue_players, orange_players) = sort_players_by_team(&frame.players);

    // Determine which player we're focusing on
    let (this_player, teammates, opponents) = if player_index < 3 {
        // Blue team player
        if player_index >= blue_players.len() {
            // Player doesn't exist, return zeros
            return features;
        }
        let this = blue_players[player_index];
        let teammates: Vec<&PlayerState> = blue_players
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != player_index)
            .map(|(_, p)| *p)
            .collect();
        (this, teammates, orange_players)
    } else {
        // Orange team player
        let orange_idx = player_index - 3;
        if orange_idx >= orange_players.len() {
            // Player doesn't exist, return zeros
            return features;
        }
        let this = orange_players[orange_idx];
        let teammates: Vec<&PlayerState> = orange_players
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != orange_idx)
            .map(|(_, p)| *p)
            .collect();
        (this, teammates, blue_players)
    };

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

    // 4. Teammate 1 state (13 features)
    if let Some(&teammate1) = teammates.first() {
        write_player_state_to_slice(&mut features.features[idx..idx + 13], teammate1);
    }
    idx += 13;

    // 5. Teammate 2 state (13 features)
    if let Some(&teammate2) = teammates.get(1) {
        write_player_state_to_slice(&mut features.features[idx..idx + 13], teammate2);
    }
    idx += 13;

    // 6. Teammate relationships (4 features)
    if let Some(&teammate1) = teammates.first() {
        if !teammate1.actor_state.is_demolished {
            let dist_to_ball = distance(&teammate1.actor_state.position, &frame.ball.position);
            features.features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);
            let dist_to_this = distance(
                &teammate1.actor_state.position,
                &this_player.actor_state.position,
            );
            features.features[idx + 1] = (dist_to_this / field::DIAGONAL).min(1.0);
        }
    }
    if let Some(&teammate2) = teammates.get(1) {
        if !teammate2.actor_state.is_demolished {
            let dist_to_ball = distance(&teammate2.actor_state.position, &frame.ball.position);
            features.features[idx + 2] = (dist_to_ball / field::DIAGONAL).min(1.0);
            let dist_to_this = distance(
                &teammate2.actor_state.position,
                &this_player.actor_state.position,
            );
            features.features[idx + 3] = (dist_to_this / field::DIAGONAL).min(1.0);
        }
    }
    idx += 4;

    // 7-9. Opponent states (12 features each, NO BOOST)
    for i in 0..3 {
        if let Some(&opponent) = opponents.get(i) {
            write_opponent_state_to_slice(&mut features.features[idx..idx + 12], opponent);
        }
        idx += 12;
    }

    // 10. Opponent relationships (6 features)
    for i in 0..3 {
        if let Some(&opponent) = opponents.get(i) {
            if !opponent.actor_state.is_demolished {
                let dist_to_ball = distance(&opponent.actor_state.position, &frame.ball.position);
                features.features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);
                let dist_to_this = distance(
                    &opponent.actor_state.position,
                    &this_player.actor_state.position,
                );
                features.features[idx + 1] = (dist_to_this / field::DIAGONAL).min(1.0);
            }
        }
        idx += 2;
    }

    // 11. Game context (1 feature)
    let blue_goal = Vector3 {
        x: 0.0,
        y: field::BLUE_GOAL_Y,
        z: 0.0,
    };
    features.features[idx] =
        (distance(&frame.ball.position, &blue_goal) / field::DIAGONAL).min(1.0);

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

/// Extracts player-centric features for ALL 6 players from a game frame.
///
/// Returns array of 6 feature vectors, one per player.
pub fn extract_all_player_centric_features(
    frame: &GameFrame,
) -> [PlayerCentricFrameFeatures; TOTAL_PLAYERS] {
    [
        extract_player_centric_frame_features(frame, 0),
        extract_player_centric_frame_features(frame, 1),
        extract_player_centric_frame_features(frame, 2),
        extract_player_centric_frame_features(frame, 3),
        extract_player_centric_frame_features(frame, 4),
        extract_player_centric_frame_features(frame, 5),
    ]
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
/// This reduces data size while still capturing gameplay patterns.
/// At 30fps, taking 1/6 frames gives ~5fps effective sampling rate.
pub const FRAME_SUBSAMPLE_RATE: usize = 6;

/// Extracts a game sequence sample with player-centric features.
///
/// Uses frame subsampling (1 frame out of every `FRAME_SUBSAMPLE_RATE` frames)
/// to reduce data size while preserving gameplay patterns.
///
/// # Arguments
///
/// * `frames` - All frames from the replay
/// * `player_ratings` - Player ratings with team assignments
///
/// # Returns
///
/// A `PlayerCentricGameSequence` with features for all 6 players across subsampled frames
pub fn extract_player_centric_game_sequence(
    frames: &[GameFrame],
    player_ratings: &[PlayerRating],
) -> PlayerCentricGameSequence {
    let target_mmr = build_target_mmr_array(player_ratings);

    // Extract features for subsampled frames (1 out of every FRAME_SUBSAMPLE_RATE), all players
    let player_frames: Vec<[PlayerCentricFrameFeatures; 6]> = frames
        .iter()
        .step_by(FRAME_SUBSAMPLE_RATE)
        .map(extract_all_player_centric_features)
        .collect();

    PlayerCentricGameSequence {
        player_frames,
        target_mmr,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, Team};

    use super::*;

    #[test]
    fn test_player_centric_features_differ() {
        // Create a frame with distinct players
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

        // Extract player-centric features for all players
        let all_features = extract_all_player_centric_features(&frame);

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
        // Verify the feature count calculation
        // Ball: 7, This player: 13, Ball relationship: 2
        // Teammate 1: 13, Teammate 2: 13, Teammate relationships: 4
        // Opponent 1-3: 12×3, Opponent relationships: 6
        // Game context: 1
        let expected = 7 + 13 + 2 + 13 + 13 + 4 + 12 + 12 + 12 + 6 + 1;
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
}
