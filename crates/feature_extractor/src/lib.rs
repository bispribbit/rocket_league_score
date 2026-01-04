#![expect(clippy::indexing_slicing)]

//! Feature extractor crate for Rocket League ML model.
//!
//! This crate transforms raw replay frame data into ML-ready feature vectors
//! that can be used for training and inference.
//!
//! See `FEATURES.md` for the complete feature specification.

use replay_structs::{GameFrame, PlayerState, Quaternion, Team, Vector3};

/// Total number of features extracted per frame.
///
/// Breakdown:
/// - Ball state: 7
/// - Player state: 13 × 6 = 78
/// - Player geometry: 7 × 6 = 42
/// - Game context: 1
/// - Total: 128
pub const FEATURE_COUNT: usize = 128;

/// Number of players expected in a 3v3 match.
pub const PLAYERS_PER_TEAM: usize = 3;
pub const TOTAL_PLAYERS: usize = PLAYERS_PER_TEAM * 2;

/// Feature indices for named access.
pub mod indices {
    // Ball state (0-6)
    pub const BALL_START: usize = 0;
    pub const BALL_COUNT: usize = 7;

    // Player state (7-84): 13 features × 6 players
    pub const PLAYER_STATE_START: usize = 7;
    pub const PLAYER_STATE_FEATURES: usize = 13;

    // Player geometry (85-126): 7 features × 6 players
    pub const PLAYER_GEOM_START: usize = 85;
    pub const PLAYER_GEOM_FEATURES: usize = 7;

    // Game context (127): 1 feature
    pub const GAME_CONTEXT_START: usize = 127;
    pub const GAME_CONTEXT_COUNT: usize = 1;
}

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

/// Feature vector extracted from a single game frame.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// The raw feature vector.
    pub features: [f32; FEATURE_COUNT],
    /// Timestamp of the frame.
    pub time: f32,
}

impl Default for FrameFeatures {
    fn default() -> Self {
        Self {
            features: [0.0; FEATURE_COUNT],
            time: 0.0,
        }
    }
}

/// Training sample combining features with ground truth labels.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub features: FrameFeatures,
    /// Target MMR per player slot (6 players: blue team sorted by `actor_id`, then orange team sorted by `actor_id`).
    /// Matches the feature order in the feature vector.
    pub target_mmr: Vec<f32>,
}

/// Extracts ML features from a single game frame.
///
/// # Arguments
///
/// * `frame` - The game frame to extract features from.
///
/// # Returns
///
/// A `FrameFeatures` struct containing the normalized feature vector.
pub fn extract_frame_features(frame: &GameFrame) -> FrameFeatures {
    let mut features = FrameFeatures {
        features: [0.0; FEATURE_COUNT],
        time: frame.time,
    };

    // Sort players into teams
    let (blue_players, orange_players) = sort_players_by_team(&frame.players);

    // 1. Extract ball features (indices 0-6)
    extract_ball_features(
        &mut features.features,
        &frame.ball.position,
        &frame.ball.velocity,
    );

    // 2. Extract player state features (indices 7-84)
    extract_player_state_features(&mut features.features, &blue_players, &orange_players);

    // 3. Extract player geometry features (indices 85-126)
    extract_player_geometry_features(
        &mut features.features,
        &blue_players,
        &orange_players,
        &frame.ball.position,
    );

    // 4. Extract game context features (ball distance to blue goal) (index 127)
    extract_game_context_features(&mut features.features, &frame.ball.position);

    features
}

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

/// Extracts ball state features (7 features).
fn extract_ball_features(
    features: &mut [f32; FEATURE_COUNT],
    position: &Vector3,
    velocity: &Vector3,
) {
    let idx = indices::BALL_START;

    // Position (normalized)
    features[idx] = normalize_x(position.x);
    features[idx + 1] = normalize_y(position.y);
    features[idx + 2] = normalize_z(position.z);

    // Velocity (normalized)
    // Note: ball_vel_y already indicates direction toward goals
    // (positive = toward orange, negative = toward blue)
    features[idx + 3] = normalize_ball_velocity(velocity.x);
    features[idx + 4] = normalize_ball_velocity(velocity.y);
    features[idx + 5] = normalize_ball_velocity(velocity.z);

    // Speed magnitude
    let speed = vector_magnitude(velocity);
    features[idx + 6] = (speed / field::MAX_BALL_SPEED).min(1.0);
}

/// Extracts player state features (13 features × 6 players = 78 features).
fn extract_player_state_features(
    features: &mut [f32; FEATURE_COUNT],
    blue_players: &[&PlayerState],
    orange_players: &[&PlayerState],
) {
    // Process blue team (first 3 player slots)
    for (i, player) in blue_players.iter().take(PLAYERS_PER_TEAM).enumerate() {
        let idx = indices::PLAYER_STATE_START + i * indices::PLAYER_STATE_FEATURES;
        write_player_state(features, idx, player);
    }
    // Pad with zeros if fewer than 3 blue players
    for i in blue_players.len()..PLAYERS_PER_TEAM {
        let idx = indices::PLAYER_STATE_START + i * indices::PLAYER_STATE_FEATURES;
        for j in 0..indices::PLAYER_STATE_FEATURES {
            features[idx + j] = 0.0;
        }
    }

    // Process orange team (next 3 player slots)
    for (i, player) in orange_players.iter().take(PLAYERS_PER_TEAM).enumerate() {
        let idx =
            indices::PLAYER_STATE_START + (PLAYERS_PER_TEAM + i) * indices::PLAYER_STATE_FEATURES;
        write_player_state(features, idx, player);
    }
    // Pad with zeros if fewer than 3 orange players
    for i in orange_players.len()..PLAYERS_PER_TEAM {
        let idx =
            indices::PLAYER_STATE_START + (PLAYERS_PER_TEAM + i) * indices::PLAYER_STATE_FEATURES;
        for j in 0..indices::PLAYER_STATE_FEATURES {
            features[idx + j] = 0.0;
        }
    }
}

/// Writes a single player's state features to the feature vector.
fn write_player_state(features: &mut [f32; FEATURE_COUNT], idx: usize, player: &PlayerState) {
    // If demolished, zero out all features except the demolished flag
    if player.actor_state.is_demolished {
        // Position (zeroed)
        features[idx] = 0.0;
        features[idx + 1] = 0.0;
        features[idx + 2] = 0.0;

        // Velocity (zeroed)
        features[idx + 3] = 0.0;
        features[idx + 4] = 0.0;
        features[idx + 5] = 0.0;

        // Rotation (zeroed)
        features[idx + 6] = 0.0;
        features[idx + 7] = 0.0;
        features[idx + 8] = 0.0;
        features[idx + 9] = 0.0;

        // Speed (zeroed)
        features[idx + 10] = 0.0;

        // Boost (zeroed)
        features[idx + 11] = 0.0;

        // Demolished flag
        features[idx + 12] = 1.0;
    } else {
        // Position
        features[idx] = normalize_x(player.actor_state.position.x);
        features[idx + 1] = normalize_y(player.actor_state.position.y);
        features[idx + 2] = normalize_z(player.actor_state.position.z);

        // Velocity
        features[idx + 3] = normalize_car_velocity(player.actor_state.velocity.x);
        features[idx + 4] = normalize_car_velocity(player.actor_state.velocity.y);
        features[idx + 5] = normalize_car_velocity(player.actor_state.velocity.z);

        // Rotation (quaternion - already in [-1, 1] range)
        features[idx + 6] = player.actor_state.rotation.x;
        features[idx + 7] = player.actor_state.rotation.y;
        features[idx + 8] = player.actor_state.rotation.z;
        features[idx + 9] = player.actor_state.rotation.w;

        // Speed magnitude (normalized to supersonic = 1.0)
        let speed = vector_magnitude(&player.actor_state.velocity);
        features[idx + 10] = (speed / field::MAX_CAR_SPEED).min(1.0);

        // Boost (already 0-1)
        features[idx + 11] = player.actor_state.boost;

        // Demolished flag
        features[idx + 12] = 0.0;
    }
}

/// Extracts player geometry features (7 features × 6 players = 42 features).
fn extract_player_geometry_features(
    features: &mut [f32; FEATURE_COUNT],
    blue_players: &[&PlayerState],
    orange_players: &[&PlayerState],
    ball_pos: &Vector3,
) {
    // Blue team geometry
    for (i, player) in blue_players.iter().take(PLAYERS_PER_TEAM).enumerate() {
        let idx = indices::PLAYER_GEOM_START + i * indices::PLAYER_GEOM_FEATURES;
        write_player_geometry(
            features,
            idx,
            player,
            ball_pos,
            blue_players,
            orange_players,
        );
    }
    for i in blue_players.len()..PLAYERS_PER_TEAM {
        let idx = indices::PLAYER_GEOM_START + i * indices::PLAYER_GEOM_FEATURES;
        for j in 0..indices::PLAYER_GEOM_FEATURES {
            features[idx + j] = 0.0;
        }
    }

    // Orange team geometry
    for (i, player) in orange_players.iter().take(PLAYERS_PER_TEAM).enumerate() {
        let idx =
            indices::PLAYER_GEOM_START + (PLAYERS_PER_TEAM + i) * indices::PLAYER_GEOM_FEATURES;
        write_player_geometry(
            features,
            idx,
            player,
            ball_pos,
            orange_players,
            blue_players,
        );
    }
    for i in orange_players.len()..PLAYERS_PER_TEAM {
        let idx =
            indices::PLAYER_GEOM_START + (PLAYERS_PER_TEAM + i) * indices::PLAYER_GEOM_FEATURES;
        for j in 0..indices::PLAYER_GEOM_FEATURES {
            features[idx + j] = 0.0;
        }
    }
}

/// Writes a single player's geometry features (7 features).
fn write_player_geometry(
    features: &mut [f32; FEATURE_COUNT],
    idx: usize,
    player: &PlayerState,
    ball_pos: &Vector3,
    teammates: &[&PlayerState],
    opponents: &[&PlayerState],
) {
    // If demolished, zero out all geometry features
    if player.actor_state.is_demolished {
        for j in 0..indices::PLAYER_GEOM_FEATURES {
            features[idx + j] = 0.0;
        }
        return;
    }

    // 0: Distance to ball (normalized by field diagonal)
    let dist_to_ball = distance(&player.actor_state.position, ball_pos);
    features[idx] = (dist_to_ball / field::DIAGONAL).min(1.0);

    // 1: Facing ball (dot product of forward vector with direction to ball)
    let forward = quaternion_forward(&player.actor_state.rotation);
    let to_ball = direction(&player.actor_state.position, ball_pos);
    features[idx + 1] = dot(&forward, &to_ball);

    // 2-3: Distance to each teammate (2 teammates in 3v3)
    let teammate_distances =
        get_distances_to_others(&player.actor_state.position, player.actor_id, teammates);
    features[idx + 2] = teammate_distances[0];
    features[idx + 3] = teammate_distances[1];

    // 4-6: Distance to each opponent (3 opponents in 3v3)
    let opponent_distances =
        get_distances_to_others(&player.actor_state.position, player.actor_id, opponents);
    features[idx + 4] = opponent_distances[0];
    features[idx + 5] = opponent_distances[1];
    features[idx + 6] = opponent_distances[2];
}

/// Gets distances to all other players (excluding self), sorted by `actor_id`.
/// Returns a fixed-size array with 1.0 for missing players.
fn get_distances_to_others(
    position: &Vector3,
    self_id: i32,
    players: &[&PlayerState],
) -> [f32; PLAYERS_PER_TEAM] {
    let mut distances = [1.0f32; PLAYERS_PER_TEAM]; // Default to max distance

    // Filter out self and collect distances
    let mut other_players: Vec<_> = players.iter().filter(|p| p.actor_id != self_id).collect();

    // Sort by actor_id for consistent ordering
    other_players.sort_by_key(|p| p.actor_id);

    // Compute distances
    for (i, other) in other_players.iter().take(PLAYERS_PER_TEAM).enumerate() {
        let dist = distance(position, &other.actor_state.position);
        distances[i] = (dist / field::DIAGONAL).min(1.0);
    }

    distances
}

/// Extracts game context features (1 feature).
fn extract_game_context_features(features: &mut [f32; FEATURE_COUNT], ball_pos: &Vector3) {
    let idx = indices::GAME_CONTEXT_START;

    // Ball distance to blue goal (orange goal distance is redundant - highly correlated)
    let blue_goal = Vector3 {
        x: 0.0,
        y: field::BLUE_GOAL_Y,
        z: 0.0,
    };

    features[idx] = (distance(ball_pos, &blue_goal) / field::DIAGONAL).min(1.0);
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

/// A sequence sample representing an entire game/replay for LSTM training.
#[derive(Debug, Clone)]
pub struct GameSequenceSample {
    /// All frame features from the game.
    pub frames: Vec<FrameFeatures>,
    /// Target MMR for each of the 6 players.
    /// Order: blue team (3 players sorted by `actor_id`), then orange team (3 players).
    pub target_mmr: [f32; TOTAL_PLAYERS],
}

/// Extracts a single game sequence sample from all frames in a replay.
///
/// This creates one training sample per replay, where the model learns
/// to predict player MMR from the entire sequence of gameplay.
///
/// # Arguments
///
/// * `frames` - All frames from the replay.
/// * `player_ratings` - Player ratings with team assignments.
///
/// # Returns
///
/// A `GameSequenceSample` containing all frame features and target MMR values.
pub fn extract_game_sequence(
    frames: &[GameFrame],
    player_ratings: &[PlayerRating],
) -> GameSequenceSample {
    // Build target MMR array
    let target_mmr = build_target_mmr_array(player_ratings);

    // Extract features from ALL frames
    let frame_features: Vec<FrameFeatures> = frames.iter().map(extract_frame_features).collect();

    GameSequenceSample {
        frames: frame_features,
        target_mmr,
    }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, Team};

    use super::*;

    #[test]
    fn test_feature_count() {
        // Verify our constants add up correctly
        let expected = indices::BALL_COUNT
            + (indices::PLAYER_STATE_FEATURES * TOTAL_PLAYERS)
            + (indices::PLAYER_GEOM_FEATURES * TOTAL_PLAYERS)
            + indices::GAME_CONTEXT_COUNT;

        assert_eq!(expected, FEATURE_COUNT);
    }

    #[test]
    fn test_extract_empty_frame() {
        let frame = GameFrame::default();
        let features = extract_frame_features(&frame);
        assert_eq!(features.features.len(), FEATURE_COUNT);
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

    #[test]
    fn test_extract_with_players() {
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
                        boost: 0.5,
                        is_demolished: false,
                    },
                },
                PlayerState {
                    actor_id: 2,
                    name: Arc::new("Player2".to_string()),
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
                        boost: 1.0,
                        is_demolished: false,
                    },
                },
            ],
        };

        let features = extract_frame_features(&frame);

        // Check ball features
        assert!((features.features[0] - 0.0).abs() < 0.001); // ball_pos_x
        assert!((features.features[2]).abs() < 0.1); // ball_pos_z (low)
        assert!(features.features[6] > 0.0); // ball_speed > 0

        // Check that player features are populated
        assert!(features.features[indices::PLAYER_STATE_START + 11] > 0.0); // blue player 1 boost
    }
}
