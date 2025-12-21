//! Feature extractor crate for Rocket League ML model.
//!
//! This crate transforms raw replay frame data into ML-ready feature vectors
//! that can be used for training and inference.

use replay_parser::{GameFrame, Vector3};

/// The number of features extracted per frame.
/// This includes:
/// - Ball position (3) and velocity (3) = 6
/// - Per player (6 players):
///   - Position relative to ball (3)
///   - Position relative to own goal (3)
///   - Velocity (3)
///   - Boost level (1)
///   - Is demolished (1)
///   = 11 per player × 6 = 66
/// - Team possession indicator (1)
/// - Ball distance to each goal (2)
/// Total: ~75 features
pub const FEATURE_COUNT: usize = 75;

/// A player's actual MMR rating used as training label.
///
/// Using continuous MMR values instead of categorical labels allows the
/// model to learn the full skill spectrum and capture subtle differences
/// between players of similar skill levels.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerRating {
    pub player_id: u32,
    pub mmr: i32, // Actual MMR value (e.g., 1547)
}

/// Feature vector extracted from a single game frame.
///
/// Contains normalized features suitable for neural network input.
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
    /// MMR ratings for all players in this frame.
    pub player_ratings: Vec<PlayerRating>,
    /// Average MMR of the team we're evaluating (used as label).
    pub target_mmr: f32,
}

/// Field dimensions for normalization (Rocket League standard field).
pub mod field {
    /// Half-length of the field (X axis).
    pub const HALF_LENGTH: f32 = 5120.0;
    /// Half-width of the field (Y axis).
    pub const HALF_WIDTH: f32 = 4096.0;
    /// Maximum height of the field (Z axis).
    pub const MAX_HEIGHT: f32 = 2044.0;
    /// Maximum expected velocity.
    pub const MAX_VELOCITY: f32 = 6000.0;
    /// Goal Y position (positive for team 1, negative for team 0).
    pub const GOAL_Y: f32 = 5120.0;
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
    // TODO: Implement actual feature extraction
    // For now, return mock features

    let mut features = FrameFeatures {
        features: [0.0; FEATURE_COUNT],
        time: frame.time,
    };

    // Mock: Extract ball features (normalized to [-1, 1] range)
    let idx = 0;
    features.features[idx] = normalize_position_x(frame.ball.position.x);
    features.features[idx + 1] = normalize_position_y(frame.ball.position.y);
    features.features[idx + 2] = normalize_position_z(frame.ball.position.z);
    features.features[idx + 3] = normalize_velocity(frame.ball.velocity.x);
    features.features[idx + 4] = normalize_velocity(frame.ball.velocity.y);
    features.features[idx + 5] = normalize_velocity(frame.ball.velocity.z);

    // TODO: Extract player features for each of the 6 players
    // - Position relative to ball
    // - Position relative to own goal
    // - Velocity
    // - Boost level
    // - Is demolished flag

    // TODO: Extract team-level features
    // - Which team has possession
    // - Ball distance to each goal

    features
}

/// Normalizes features in-place to have zero mean and unit variance.
///
/// This is important for neural network training stability.
pub const fn normalize_features(features: &mut FrameFeatures) {
    // TODO: Implement proper normalization based on training data statistics
    // For now, features are already normalized during extraction

    // This function would typically:
    // 1. Subtract the mean of each feature (computed from training data)
    // 2. Divide by the standard deviation of each feature
    let _ = features;
}

/// Extracts features from all frames in a segment and pairs with ratings.
///
/// # Arguments
///
/// * `frames` - Slice of game frames to process.
/// * `player_ratings` - MMR ratings for each player.
///
/// # Returns
///
/// Vector of training samples.
pub fn extract_segment_samples(
    frames: &[GameFrame],
    player_ratings: &[PlayerRating],
) -> Vec<TrainingSample> {
    // TODO: Implement actual extraction
    // For now, return mock samples

    let avg_mmr = if player_ratings.is_empty() {
        1000.0
    } else {
        player_ratings.iter().map(|r| r.mmr as f32).sum::<f32>() / player_ratings.len() as f32
    };

    frames
        .iter()
        .map(|frame| TrainingSample {
            features: extract_frame_features(frame),
            player_ratings: player_ratings.to_vec(),
            target_mmr: avg_mmr,
        })
        .collect()
}

// Helper normalization functions

fn normalize_position_x(x: f32) -> f32 {
    x / field::HALF_LENGTH
}

fn normalize_position_y(y: f32) -> f32 {
    y / field::HALF_WIDTH
}

fn normalize_position_z(z: f32) -> f32 {
    z / field::MAX_HEIGHT
}

fn normalize_velocity(v: f32) -> f32 {
    v / field::MAX_VELOCITY
}

/// Computes the distance between two 3D points.
#[expect(dead_code)]
fn distance(a: &Vector3, b: &Vector3) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_features_default() {
        let features = FrameFeatures::default();
        assert_eq!(features.features.len(), FEATURE_COUNT);
        assert!(features.features.iter().all(|&f| f == 0.0));
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
        assert!((normalize_position_x(field::HALF_LENGTH) - 1.0).abs() < f32::EPSILON);
        assert!((normalize_position_y(field::HALF_WIDTH) - 1.0).abs() < f32::EPSILON);
    }
}
