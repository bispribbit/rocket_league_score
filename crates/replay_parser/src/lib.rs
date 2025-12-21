//! Replay parser crate for Rocket League replays.
//!
//! This crate wraps the `boxcars` library to parse `.replay` files
//! into structured Rust types suitable for ML feature extraction.

use std::collections::HashMap;
use std::path::Path;

use boxcars::{Attribute, HeaderProp, ParserBuilder, Replay};

/// Represents a 3D vector (position or velocity).
#[derive(Debug, Clone, Copy, Default)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<boxcars::Vector3f> for Vector3 {
    fn from(v: boxcars::Vector3f) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

/// Represents a quaternion rotation.
///
/// Quaternions are preferred over Euler angles for ML because:
/// - They are continuous (no sudden jumps at angle boundaries)
/// - No gimbal lock issues
/// - Neural networks handle them well
#[derive(Debug, Clone, Copy, Default)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<boxcars::Quaternion> for Quaternion {
    fn from(q: boxcars::Quaternion) -> Self {
        Self {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
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
    pub actor_id: i32,
    pub name: String,
    pub team: u8, // 0 or 1
    pub position: Vector3,
    pub velocity: Vector3,
    pub rotation: Quaternion,
    pub boost: f32, // 0.0 to 1.0
    pub is_demolished: bool,
}

/// A single frame of game state.
#[derive(Debug, Clone, Default)]
pub struct GameFrame {
    pub time: f32,
    pub delta: f32,
    pub seconds_remaining: i32,
    pub ball: BallState,
    pub players: Vec<PlayerState>,
}

/// Metadata about the replay.
#[derive(Debug, Clone, Default)]
pub struct ReplayMetadata {
    pub replay_id: String,
    pub replay_name: String,
    pub game_mode: String,
    pub map_name: String,
    pub team_size: u32,
    pub team_0_score: u32,
    pub team_1_score: u32,
    pub num_frames: u32,
}

/// Goal event extracted from replay header.
#[derive(Debug, Clone)]
pub struct GoalEvent {
    pub frame: usize,
    pub player_name: String,
    pub player_team: u8,
}

/// A fully parsed replay containing metadata and all frames.
#[derive(Debug, Clone, Default)]
pub struct ParsedReplay {
    pub metadata: ReplayMetadata,
    pub frames: Vec<GameFrame>,
    pub goals: Vec<GoalEvent>,
    pub goal_frames: Vec<usize>,
    pub kickoff_frames: Vec<usize>,
}

/// A segment of gameplay between a kickoff and a goal (or end of game).
#[derive(Debug, Clone)]
pub struct GameSegment {
    pub start_frame: usize,
    pub end_frame: usize,
    pub ended_with_goal: bool,
    pub scoring_team: Option<u8>,
}

/// Internal state for tracking actors during parsing.
#[derive(Debug, Default)]
struct ActorState {
    position: Vector3,
    velocity: Vector3,
    rotation: Quaternion,
    boost: f32,
    is_demolished: bool,
}

/// Internal state for tracking player info.
#[derive(Debug, Default, Clone)]
struct PlayerInfo {
    name: String,
    team: i32, // -1 = unknown, 0 or 1
}

/// Parses a replay file from the given path.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn parse_replay(path: &Path) -> anyhow::Result<ParsedReplay> {
    let raw_data = std::fs::read(path)?;
    let replay = ParserBuilder::new(&raw_data)
        .must_parse_network_data()
        .parse()?;

    parse_boxcars_replay(&replay)
}

/// Parses a replay from raw bytes.
///
/// # Errors
///
/// Returns an error if parsing fails.
pub fn parse_replay_bytes(data: &[u8]) -> anyhow::Result<ParsedReplay> {
    let replay = ParserBuilder::new(data).must_parse_network_data().parse()?;

    parse_boxcars_replay(&replay)
}

fn parse_boxcars_replay(replay: &Replay) -> anyhow::Result<ParsedReplay> {
    let metadata = extract_metadata(replay);
    let goals = extract_goals(replay);
    let goal_frames: Vec<usize> = goals.iter().map(|goal| goal.frame).collect();

    // Find object IDs for important types
    let ball_object_id = find_object_id(&replay.objects, "Archetypes.Ball.Ball_Default");
    let car_object_id = find_object_id(&replay.objects, "Archetypes.Car.Car_Default");
    let boost_object_id = find_object_id(
        &replay.objects,
        "Archetypes.CarComponents.CarComponent_Boost",
    );

    // Find attribute IDs for important properties
    let player_name_attr =
        find_object_id(&replay.objects, "Engine.PlayerReplicationInfo:PlayerName");
    let team_attr = find_object_id(&replay.objects, "Engine.PlayerReplicationInfo:Team");

    let network_frames = replay
        .network_frames
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No network frames in replay"))?;

    // Track active actors
    let mut ball_actors: HashMap<i32, ActorState> = HashMap::new();
    let mut car_actors: HashMap<i32, ActorState> = HashMap::new();
    let mut boost_actors: HashMap<i32, i32> = HashMap::new(); // boost_actor_id -> car_actor_id
    let mut boost_amounts: HashMap<i32, f32> = HashMap::new(); // car_actor_id -> boost (0-1)
    let mut player_infos: HashMap<i32, PlayerInfo> = HashMap::new(); // PRI actor_id -> PlayerInfo
    let mut car_to_pri: HashMap<i32, i32> = HashMap::new(); // car_actor_id -> PRI actor_id

    // State tracking for frame reconstruction
    let mut current_seconds_remaining: i32 = 300;
    let mut frames = Vec::with_capacity(network_frames.frames.len());
    let mut kickoff_frames = Vec::new();
    let mut last_kickoff_detected = false;

    for (frame_idx, frame) in network_frames.frames.iter().enumerate() {
        // Handle new actors
        for new_actor in &frame.new_actors {
            let actor_id = new_actor.actor_id.0;
            let object_id = new_actor.object_id.0 as usize;

            if Some(object_id) == ball_object_id {
                ball_actors.insert(actor_id, ActorState::default());
            } else if Some(object_id) == car_object_id {
                car_actors.insert(actor_id, ActorState::default());
            } else if Some(object_id) == boost_object_id {
                // Boost component - we'll link it to a car via Vehicle attribute
                boost_actors.insert(actor_id, 0);
            }

            // Check if this is a PRI (player replication info) based on object name
            if let Some(obj_name) = replay.objects.get(object_id)
                && obj_name.contains("PRI_TA")
                && !obj_name.contains(':')
            {
                player_infos.insert(
                    actor_id,
                    PlayerInfo {
                        team: -1,
                        ..Default::default()
                    },
                );
            }
        }

        // Handle deleted actors
        for deleted_actor in &frame.deleted_actors {
            let actor_id = deleted_actor.0;
            ball_actors.remove(&actor_id);
            car_actors.remove(&actor_id);
            boost_actors.remove(&actor_id);
            player_infos.remove(&actor_id);
            car_to_pri.retain(|_, v| *v != actor_id);
        }

        // Handle updated actors
        for update in &frame.updated_actors {
            let actor_id = update.actor_id.0;

            match &update.attribute {
                Attribute::RigidBody(rigid_body) => {
                    let state = ActorState {
                        position: rigid_body.location.into(),
                        velocity: rigid_body
                            .linear_velocity
                            .map(Into::into)
                            .unwrap_or_default(),
                        rotation: rigid_body.rotation.into(),
                        ..Default::default()
                    };

                    if let std::collections::hash_map::Entry::Occupied(mut entry) =
                        ball_actors.entry(actor_id)
                    {
                        entry.insert(state);
                    } else if car_actors.contains_key(&actor_id)
                        && let Some(existing) = car_actors.get_mut(&actor_id)
                    {
                        existing.position = state.position;
                        existing.velocity = state.velocity;
                        existing.rotation = state.rotation;
                    }
                }
                Attribute::ReplicatedBoost(boost) => {
                    // This is a boost component actor
                    if let Some(&car_id) = boost_actors.get(&actor_id)
                        && car_id != 0
                    {
                        let boost_normalized = f32::from(boost.boost_amount) / 255.0;
                        boost_amounts.insert(car_id, boost_normalized);
                        if let Some(car) = car_actors.get_mut(&car_id) {
                            car.boost = boost_normalized;
                        }
                    }
                }
                Attribute::String(name) => {
                    // Player name - check if this is the PlayerName attribute
                    let stream_id = update.stream_id.0 as usize;
                    if Some(stream_id) == player_name_attr
                        && let Some(info) = player_infos.get_mut(&actor_id)
                        && !name.is_empty()
                    {
                        info.name.clone_from(name);
                    }
                }
                Attribute::TeamPaint(team_paint) => {
                    // Also check if this is a player info
                    if let Some(info) = player_infos.get_mut(&actor_id) {
                        info.team = i32::from(team_paint.team);
                    }
                }
                Attribute::FlaggedByte(flagged, team_id) => {
                    // Team assignment via flagged byte (Team attribute)
                    let stream_id = update.stream_id.0 as usize;
                    if Some(stream_id) == team_attr
                        && *flagged
                        && let Some(info) = player_infos.get_mut(&actor_id)
                    {
                        // team_id typically encodes the team (0 or 1)
                        info.team = i32::from(*team_id % 2);
                    }
                }
                Attribute::ActiveActor(active_actor) => {
                    // This links various components together
                    // For boost components: boost_actor -> car_actor
                    if boost_actors.contains_key(&actor_id) {
                        let car_id = active_actor.actor.0;
                        boost_actors.insert(actor_id, car_id);
                    }
                    // For cars: car_actor -> PRI_actor (PlayerReplicationInfo link)
                    if car_actors.contains_key(&actor_id) {
                        let pri_id = active_actor.actor.0;
                        car_to_pri.insert(actor_id, pri_id);
                    }
                }
                Attribute::Int(seconds) => {
                    // Check if this is SecondsRemaining
                    let object_name = replay.objects.get(update.stream_id.0 as usize);
                    if object_name.is_some_and(|n| n.contains("SecondsRemaining")) {
                        current_seconds_remaining = *seconds;
                    }
                }
                Attribute::Demolish(demo) => {
                    // Mark car as demolished
                    let victim_id = demo.victim.0;
                    if let Some(car) = car_actors.get_mut(&victim_id) {
                        car.is_demolished = true;
                    }
                }
                Attribute::DemolishFx(demo) => {
                    // Mark car as demolished (extended version)
                    let victim_id = demo.victim.0;
                    if let Some(car) = car_actors.get_mut(&victim_id) {
                        car.is_demolished = true;
                    }
                }
                Attribute::Reservation(res) => {
                    // Link player to team via reservation
                    if let Some(info) = player_infos.get_mut(&actor_id)
                        && let Some(name) = &res.name
                    {
                        info.name.clone_from(name);
                    }
                }
                _ => {}
            }
        }

        // Detect kickoff (ball at center position with low velocity)
        let ball_at_center = ball_actors.values().any(|b| {
            b.position.x.abs() < 10.0
                && b.position.y.abs() < 10.0
                && b.velocity.x.abs() < 1.0
                && b.velocity.y.abs() < 1.0
        });

        if ball_at_center && !last_kickoff_detected && frame_idx > 0 {
            kickoff_frames.push(frame_idx);
        }
        last_kickoff_detected = ball_at_center;

        // Build frame state
        let ball_state = ball_actors
            .values()
            .next()
            .map(|b| BallState {
                position: b.position,
                velocity: b.velocity,
            })
            .unwrap_or_default();

        let mut players: Vec<PlayerState> = car_actors
            .iter()
            .map(|(&car_id, car)| {
                // Try to find player info for this car
                let pri_id = car_to_pri.get(&car_id);
                let player_info = pri_id.and_then(|id| player_infos.get(id));

                let boost = boost_amounts.get(&car_id).copied().unwrap_or(car.boost);

                // Determine team - try player info first, otherwise use car position heuristic
                let team = player_info
                    .and_then(|p| {
                        if p.team >= 0 {
                            Some(p.team as u8)
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| {
                        // Heuristic: cars on positive Y side at start are team 1
                        u8::from(car.position.y > 0.0)
                    });

                PlayerState {
                    actor_id: car_id,
                    name: player_info
                        .filter(|p| !p.name.is_empty())
                        .map_or_else(|| format!("Player_{car_id}"), |p| p.name.clone()),
                    team,
                    position: car.position,
                    velocity: car.velocity,
                    rotation: car.rotation,
                    boost,
                    is_demolished: car.is_demolished,
                }
            })
            .collect();

        // Sort players by team then actor_id for consistent ordering
        players.sort_by(|a, b| a.team.cmp(&b.team).then(a.actor_id.cmp(&b.actor_id)));

        // Reset demolished state after capturing it
        for car in car_actors.values_mut() {
            car.is_demolished = false;
        }

        frames.push(GameFrame {
            time: frame.time,
            delta: frame.delta,
            seconds_remaining: current_seconds_remaining,
            ball: ball_state,
            players,
        });
    }

    // Add initial kickoff if we have frames
    if !frames.is_empty() && (kickoff_frames.is_empty() || kickoff_frames.first() != Some(&0)) {
        kickoff_frames.insert(0, 0);
    }

    Ok(ParsedReplay {
        metadata,
        frames,
        goals,
        goal_frames,
        kickoff_frames,
    })
}

fn extract_metadata(replay: &Replay) -> ReplayMetadata {
    let mut metadata = ReplayMetadata::default();

    for (key, value) in &replay.properties {
        match (key.as_str(), value) {
            ("Id", HeaderProp::Str(id)) => metadata.replay_id.clone_from(id),
            ("ReplayName", HeaderProp::Str(replay_name)) => {
                metadata.replay_name.clone_from(replay_name);
            }
            ("MapName", HeaderProp::Name(map_name)) => metadata.map_name.clone_from(map_name),
            ("TeamSize", HeaderProp::Int(team_size)) => metadata.team_size = *team_size as u32,
            ("Team0Score", HeaderProp::Int(team_0_score)) => {
                metadata.team_0_score = *team_0_score as u32;
            }
            ("Team1Score", HeaderProp::Int(team_1_score)) => {
                metadata.team_1_score = *team_1_score as u32;
            }
            ("NumFrames", HeaderProp::Int(num_frames)) => metadata.num_frames = *num_frames as u32,
            ("MatchType", HeaderProp::Name(match_type)) => {
                metadata.game_mode.clone_from(match_type);
            }
            _ => {}
        }
    }

    metadata
}

fn extract_goals(replay: &Replay) -> Vec<GoalEvent> {
    let mut goals = Vec::new();

    for (key, value) in &replay.properties {
        if key == "Goals"
            && let HeaderProp::Array(goal_array) = value
        {
            for goal_props in goal_array {
                let mut frame = 0;
                let mut player_name = String::new();
                let mut player_team = 0u8;

                for (prop_key, prop_value) in goal_props {
                    match (prop_key.as_str(), prop_value) {
                        ("frame", HeaderProp::Int(f)) => frame = *f as usize,
                        ("PlayerName", HeaderProp::Str(name)) => player_name.clone_from(name),
                        ("PlayerTeam", HeaderProp::Int(t)) => player_team = *t as u8,
                        _ => {}
                    }
                }

                goals.push(GoalEvent {
                    frame,
                    player_name,
                    player_team,
                });
            }
        }
    }

    goals
}

fn find_object_id(objects: &[String], name: &str) -> Option<usize> {
    objects.iter().position(|o| o == name)
}

/// Segments a replay into chunks between kickoffs and goals.
///
/// Each segment represents a continuous period of play that can be
/// used for training the ML model.
pub fn segment_by_goals(replay: &ParsedReplay) -> Vec<GameSegment> {
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
            .goals
            .iter()
            .find(|g| g.frame > kickoff_frame && g.frame < next_kickoff);

        let (end_frame, ended_with_goal, scoring_team) = goal_in_segment
            .map_or((next_kickoff, false, None), |goal| {
                (goal.frame, true, Some(goal.player_team))
            });

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
    use tracing::{info, warn};

    use super::*;

    #[test]
    fn test_segment_empty_replay() {
        let replay = ParsedReplay::default();
        let segments = segment_by_goals(&replay);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_parse_real_replay() {
        let _tracing = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let replay_path = Path::new("data/000f8951-ae9e-419a-8fed-0a22af68cabd.replay");
        if !replay_path.exists() {
            warn!("Test replay not found, skipping test");
            return;
        }

        let parsed = parse_replay(replay_path).expect("Failed to parse replay");

        // Check metadata
        info!("=== METADATA ===");
        info!(replay_id = %parsed.metadata.replay_id, "Replay ID");
        info!(map = %parsed.metadata.map_name, "Map");
        info!(team_size = parsed.metadata.team_size, "Team size");
        info!(
            team_0_score = parsed.metadata.team_0_score,
            team_1_score = parsed.metadata.team_1_score,
            "Score"
        );
        info!(
            header_frames = parsed.metadata.num_frames,
            "Num frames (header)"
        );
        info!(parsed_frames = parsed.frames.len(), "Parsed frames");

        assert!(parsed.frames.len() > 1000, "Expected many frames");

        // Check goals
        info!("=== GOALS ===");
        for goal in &parsed.goals {
            info!(
                frame = goal.frame,
                player = %goal.player_name,
                team = goal.player_team,
                "Goal scored"
            );
        }
        assert!(!parsed.goals.is_empty(), "Expected goals in replay");

        // Check kickoffs
        info!("=== KICKOFFS ===");
        info!(
            kickoff_frames = ?&parsed.kickoff_frames[..parsed.kickoff_frames.len().min(10)],
            "Kickoff frames"
        );

        // Check a sample frame in the middle
        let mid_idx = parsed.frames.len() / 2;
        if let Some(frame) = parsed.frames.get(mid_idx) {
            info!("=== SAMPLE FRAME {} ===", mid_idx);
            info!(time = frame.time, "Time (seconds)");
            info!(
                seconds_remaining = frame.seconds_remaining,
                "Seconds remaining"
            );
            info!(
                ball_x = frame.ball.position.x,
                ball_y = frame.ball.position.y,
                ball_z = frame.ball.position.z,
                "Ball position"
            );
            info!(player_count = frame.players.len(), "Players");
            for player in &frame.players {
                info!(
                    name = %player.name,
                    team = player.team,
                    pos_x = player.position.x,
                    pos_y = player.position.y,
                    pos_z = player.position.z,
                    velocity_x = player.velocity.x,
                    velocity_y = player.velocity.y,
                    velocity_z = player.velocity.z,
                    boost = player.boost * 100.0,
                    "Player state"
                );
            }
        }

        // Check segments
        let segments = segment_by_goals(&parsed);
        info!("=== SEGMENTS ===");
        info!(total_segments = segments.len(), "Total segments");
        for (i, seg) in segments.iter().enumerate().take(5) {
            info!(
                segment = i,
                start_frame = seg.start_frame,
                end_frame = seg.end_frame,
                ended_with_goal = seg.ended_with_goal,
                "Segment"
            );
        }
    }
}
