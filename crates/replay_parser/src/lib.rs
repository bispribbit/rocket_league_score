//! Replay parser crate for Rocket League replays.
//!
//! This crate wraps the `boxcars` library to parse `.replay` files
//! into structured Rust types suitable for ML feature extraction.

use std::path::Path;

/// Represents a 3D vector (position or velocity).
#[derive(Debug, Clone, Copy, Default)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// State of the ball at a given frame.
#[derive(Debug, Clone, Default)]
pub struct BallState {
    pub position: Vector3,
    pub velocity: Vector3,
}

/// State of a player at a given frame.
#[derive(Debug, Clone, Default)]
pub struct PlayerState {
    pub player_id: u32,
    pub name: String,
    pub team: u8, // 0 or 1
    pub position: Vector3,
    pub velocity: Vector3,
    pub rotation: Vector3,
    pub boost: f32, // 0.0 to 1.0
    pub is_demolished: bool,
}

/// A single frame of game state.
#[derive(Debug, Clone, Default)]
pub struct GameFrame {
    pub time: f32,
    pub ball: BallState,
    pub players: Vec<PlayerState>,
}

/// Metadata about the replay.
#[derive(Debug, Clone, Default)]
pub struct ReplayMetadata {
    pub game_mode: String,
    pub map_name: String,
    pub match_length: f32,
    pub team_0_score: u32,
    pub team_1_score: u32,
}

/// A fully parsed replay containing metadata and all frames.
#[derive(Debug, Clone, Default)]
pub struct ParsedReplay {
    pub metadata: ReplayMetadata,
    pub frames: Vec<GameFrame>,
    pub goal_frames: Vec<usize>, // Indices of frames where goals occurred
    pub kickoff_frames: Vec<usize>, // Indices of frames where kickoffs started
}

/// A segment of gameplay between a kickoff and a goal (or end of game).
#[derive(Debug, Clone)]
pub struct GameSegment {
    pub start_frame: usize,
    pub end_frame: usize,
    pub ended_with_goal: bool,
    pub scoring_team: Option<u8>,
}

/// Parses a replay file from the given path.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_replay(path: &Path) -> anyhow::Result<ParsedReplay> {
    // TODO: Implement actual parsing using boxcars
    // For now, return a mock replay
    let _raw_data = std::fs::read(path)?;

    // Mock implementation - will be replaced with actual boxcars parsing
    // The boxcars crate provides:
    // - ParserBuilder::new(&data).parse() -> Replay
    // - Replay contains network_frames which have actor updates
    // - We need to extract ball and player positions from these

    Ok(ParsedReplay {
        metadata: ReplayMetadata {
            game_mode: "3v3".to_string(),
            map_name: "DFH Stadium".to_string(),
            match_length: 300.0,
            team_0_score: 0,
            team_1_score: 0,
        },
        frames: Vec::new(),
        goal_frames: Vec::new(),
        kickoff_frames: Vec::new(),
    })
}

/// Segments a replay into chunks between kickoffs and goals.
///
/// Each segment represents a continuous period of play that can be
/// used for training the ML model.
pub fn segment_by_goals(replay: &ParsedReplay) -> Vec<GameSegment> {
    // TODO: Implement actual segmentation logic
    // For now, return mock segments

    let mut segments = Vec::new();

    if replay.kickoff_frames.is_empty() {
        return segments;
    }

    // Create segments between each kickoff and the next goal/kickoff
    for (i, &kickoff_frame) in replay.kickoff_frames.iter().enumerate() {
        // Find the next goal or kickoff
        let next_kickoff = replay
            .kickoff_frames
            .get(i + 1)
            .copied()
            .unwrap_or(replay.frames.len());

        // Check if there's a goal in this segment
        let goal_in_segment = replay
            .goal_frames
            .iter()
            .find(|&&g| g > kickoff_frame && g < next_kickoff);

        let (end_frame, ended_with_goal, scoring_team) = if let Some(&goal_frame) = goal_in_segment
        {
            // TODO: Determine which team scored
            (goal_frame, true, Some(0u8))
        } else {
            (next_kickoff, false, None)
        };

        segments.push(GameSegment {
            start_frame: kickoff_frame,
            end_frame,
            ended_with_goal,
            scoring_team,
        });
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3_default() {
        let v = Vector3::default();
        assert!((v.x - 0.0).abs() < f32::EPSILON);
        assert!((v.y - 0.0).abs() < f32::EPSILON);
        assert!((v.z - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_segment_empty_replay() {
        let replay = ParsedReplay::default();
        let segments = segment_by_goals(&replay);
        assert!(segments.is_empty());
    }
}
