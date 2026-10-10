//! Per-player, whole-match summary statistics.
//!
//! Experiment A2 in `docs/experiment-plan-2026-10.md`: a tabular baseline that scores each
//! player from ~40 hand-crafted match-level stats. It answers, in minutes rather than GPU
//! hours, how much within-lobby signal per-player play carries and which families of
//! behaviour carry it.
//!
//! Everything positional is measured in the **team-canonical frame** (own goal at `−y`),
//! so a blue and an orange player doing the same thing get the same numbers. Time-based
//! stats are weighted by frame duration and exclude goal-replay frames, exactly as the
//! sequence features do.

use std::collections::HashMap;
use std::ops::Range;

use replay_structs::{ParsedReplay, Quaternion, Vector3};

use crate::{PLAYERS_PER_TEAM, PlayerRoster, TOTAL_PLAYERS, build_goal_replay_excluded_set};

/// Names of the stats in [`PlayerMatchStats::values`] order.
pub const MATCH_STAT_NAMES: [&str; MATCH_STAT_COUNT] = [
    "mean_speed",
    "supersonic_fraction",
    "slow_fraction",
    "ground_fraction",
    "low_air_fraction",
    "high_air_fraction",
    "wall_fraction",
    "mean_height",
    "jumps_per_minute",
    "mean_boost",
    "boost_empty_fraction",
    "boost_full_fraction",
    "boost_collected_per_minute",
    "boost_used_per_minute",
    "big_pads_per_minute",
    "small_pads_per_minute",
    "boost_while_supersonic_fraction",
    "defensive_third_fraction",
    "offensive_third_fraction",
    "mean_canonical_y",
    "behind_ball_fraction",
    "mean_distance_to_ball",
    "closest_on_team_fraction",
    "last_back_on_team_fraction",
    "most_forward_on_team_fraction",
    "mean_distance_to_nearest_teammate",
    "mean_distance_to_own_goal",
    "possession_fraction",
    "touches_per_minute",
    "mean_ball_speed_after_touch",
    "aerial_touch_fraction",
    "forward_touch_fraction",
    "mean_touch_speed_gain",
    "goals",
    "demos_received",
    "airborne_upright_fraction",
    "slow_with_boost_fraction",
    "supersonic_per_boost",
    "jumps_pressed_per_minute",
    "double_jumps_per_minute",
    "dodges_per_minute",
    "wavedash_like_dodges_per_minute",
    "aerial_dodges_per_minute",
    "pitch_positive_dodge_fraction",
    "pitch_negative_dodge_fraction",
    "roll_dodge_fraction",
    "diagonal_dodge_fraction",
    "fast_dodge_fraction",
    "boosting_fraction",
    "powerslide_fraction",
    "air_roll_fraction",
    "reverse_fraction",
    "mean_absolute_steer",
    "mean_air_angular_speed",
    "fast_aerial_fraction",
    "scoreboard_score_per_minute",
    "scoreboard_assists",
    "scoreboard_saves",
    "scoreboard_shots",
    "scoreboard_goals_per_shot",
    "demos_inflicted",
    "kickoff_first_touch_fraction",
    "kickoff_win_fraction",
    "double_commit_fraction",
    "last_back_when_defending_fraction",
    "goal_side_when_defending_fraction",
    "last_back_goal_distance_when_defending",
    "overcommit_fraction",
    "shadow_defense_fraction",
    "support_spacing_fraction",
    "goal_side_after_touch_fraction",
];

/// Number of stats per player.
pub const MATCH_STAT_COUNT: usize = 71;

/// One player's stats over one match, in [`MATCH_STAT_NAMES`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerMatchStats {
    /// Stat values.
    pub values: [f32; MATCH_STAT_COUNT],
    /// Seconds of live play this player was present for.
    pub live_seconds: f32,
}

/// Field thirds boundary in Unreal units (field half-width / 3).
const THIRD_BOUNDARY_Y: f32 = 5120.0 / 3.0;
/// Car height below which a car counts as grounded.
const GROUND_HEIGHT: f32 = 40.0;
/// Car height above which an airborne car counts as high in the air.
const HIGH_AIR_HEIGHT: f32 = 300.0;
/// A car beyond these coordinates and above [`WALL_MINIMUM_HEIGHT`] is on (or next to) a wall.
const WALL_X: f32 = 3900.0;
const WALL_Y: f32 = 4900.0;
const WALL_MINIMUM_HEIGHT: f32 = 100.0;
const SUPERSONIC_SPEED: f32 = 2200.0;
const SLOW_SPEED: f32 = 500.0;
/// Boost increments above this are big pads (100), below are small pads (12).
const BIG_PAD_INCREMENT: f32 = 0.25;
const SMALL_PAD_MINIMUM_INCREMENT: f32 = 0.04;
/// Ball velocity change that counts as a touch when a car is close enough.
const TOUCH_MINIMUM_VELOCITY_CHANGE: f32 = 250.0;
/// Maximum centre-to-centre car/ball distance for a touch (ball radius 93 + car reach).
const TOUCH_MAXIMUM_DISTANCE: f32 = 260.0;
/// Frame durations are clamped so a stalled network frame cannot dominate a time average.
const MAXIMUM_FRAME_SECONDS: f32 = 0.1;

#[derive(Debug, Clone, Copy, Default)]
struct PreviousCarState {
    boost: f32,
    height: f32,
    was_demolished: bool,
    present: bool,
    jumping: bool,
    double_jumping: bool,
    dodging: bool,
    /// Seconds since the last jump press, for flip timing.
    seconds_since_jump: f32,
}

/// Time-weighted and count accumulators for one player.
#[derive(Debug, Clone, Default)]
struct Accumulator {
    seconds: f64,
    speed: f64,
    supersonic: f64,
    slow: f64,
    ground: f64,
    low_air: f64,
    high_air: f64,
    wall: f64,
    height: f64,
    jumps: u32,
    boost: f64,
    boost_empty: f64,
    boost_full: f64,
    boost_collected: f64,
    boost_used: f64,
    big_pads: u32,
    small_pads: u32,
    boost_while_supersonic: f64,
    defensive_third: f64,
    offensive_third: f64,
    canonical_y: f64,
    behind_ball: f64,
    distance_to_ball: f64,
    closest_on_team: f64,
    last_back_on_team: f64,
    most_forward_on_team: f64,
    nearest_teammate_distance: f64,
    nearest_teammate_seconds: f64,
    distance_to_own_goal: f64,
    possession: f64,
    touches: u32,
    ball_speed_after_touch: f64,
    aerial_touches: u32,
    forward_touches: u32,
    touch_speed_gain: f64,
    goals: u32,
    demos_received: u32,
    airborne_seconds: f64,
    airborne_upright: f64,
    slow_with_boost: f64,
    jump_presses: u32,
    double_jumps: u32,
    fast_aerials: u32,
    dodges: u32,
    wavedash_like_dodges: u32,
    aerial_dodges: u32,
    forward_dodges: u32,
    backward_dodges: u32,
    sideways_dodges: u32,
    diagonal_dodges: u32,
    fast_dodges: u32,
    boosting: f64,
    powerslide: f64,
    air_roll: f64,
    reverse: f64,
    absolute_steer: f64,
    air_angular_speed: f64,
    demos_inflicted: u32,
    kickoff_first_touches: u32,
    kickoff_wins: u32,
    double_commit: f64,
    defending_seconds: f64,
    last_back_defending: f64,
    goal_side_defending: f64,
    last_back_goal_distance_defending: f64,
    overcommit: f64,
    opponent_possession_own_half: f64,
    shadow_defense: f64,
    teammate_possession: f64,
    support_spacing: f64,
    recovery_touches: u32,
    recovered_goal_side: u32,
}

fn length(vector: &Vector3) -> f32 {
    vector
        .z
        .mul_add(vector.z, vector.x.mul_add(vector.x, vector.y * vector.y))
        .sqrt()
}

fn distance(a: &Vector3, b: &Vector3) -> f32 {
    let difference = Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    };
    length(&difference)
}

/// Z component of the car's up vector (1 = wheels down, −1 = upside down).
fn up_z(rotation: &Quaternion) -> f32 {
    2.0f32.mul_add(
        -rotation.x.mul_add(rotation.x, rotation.y * rotation.y),
        1.0,
    )
}

/// One present, live car in the current frame.
#[derive(Debug, Clone, Copy)]
struct LiveCar {
    slot: usize,
    position: Vector3,
    canonical_y: f32,
    distance_to_ball: f32,
}

/// Computes [`PlayerMatchStats`] for every roster slot. Slots whose player never appears
/// are `None`.
#[must_use]
pub fn compute_player_match_stats(
    parsed: &ParsedReplay,
    roster: &PlayerRoster,
) -> [Option<PlayerMatchStats>; TOTAL_PLAYERS] {
    compute_player_window_stats(parsed, roster, 0..parsed.frames.len())
}

/// Same stats as [`compute_player_match_stats`], restricted to the frames in `frames`.
///
/// Goals and kickoffs count only when they happen inside the window. The scoreboard
/// family comes from the end-of-match header, so it is filled only when the window covers
/// the whole replay and is zero otherwise.
#[must_use]
pub fn compute_player_window_stats(
    parsed: &ParsedReplay,
    roster: &PlayerRoster,
    frames: Range<usize>,
) -> [Option<PlayerMatchStats>; TOTAL_PLAYERS] {
    let whole_match = frames.start == 0 && frames.end >= parsed.frames.len();
    let slot_by_name: HashMap<&str, usize> = roster
        .names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty())
        .map(|(slot, name)| (name.as_str(), slot))
        .collect();
    let excluded = build_goal_replay_excluded_set(&parsed.goal_frames, &parsed.kickoff_frames);

    let mut accumulators: [Accumulator; TOTAL_PLAYERS] =
        core::array::from_fn(|_| Accumulator::default());
    let mut previous: [PreviousCarState; TOTAL_PLAYERS] =
        [PreviousCarState::default(); TOTAL_PLAYERS];
    let mut previous_ball_velocity: Option<Vector3> = None;
    let mut last_demolition_time: HashMap<i32, f32> = HashMap::new();
    // Time of each player's last touch while waiting to see them get back goal-side.
    let mut pending_recovery: [Option<f32>; TOTAL_PLAYERS] = [None; TOTAL_PLAYERS];

    for goal in parsed
        .goals
        .iter()
        .filter(|goal| frames.contains(&goal.frame))
    {
        if let Some(&slot) = slot_by_name.get(goal.player_name.as_str())
            && let Some(accumulator) = accumulators.get_mut(slot)
        {
            accumulator.goals += 1;
        }
    }

    for (frame_index, frame) in parsed
        .frames
        .iter()
        .enumerate()
        .skip(frames.start)
        .take(frames.end.saturating_sub(frames.start))
    {
        if excluded.contains(&frame_index) {
            previous_ball_velocity = None;
            previous = [PreviousCarState::default(); TOTAL_PLAYERS];
            continue;
        }
        let seconds = f64::from(frame.delta.clamp(0.0, MAXIMUM_FRAME_SECONDS));
        let ball = &frame.ball;

        let mut live_cars: Vec<LiveCar> = Vec::with_capacity(TOTAL_PLAYERS);
        for player in &frame.players {
            let Some(&slot) = slot_by_name.get(player.name.as_str()) else {
                continue;
            };
            let state = &player.actor_state;
            let Some(accumulator) = accumulators.get_mut(slot) else {
                continue;
            };
            let Some(prior) = previous.get_mut(slot) else {
                continue;
            };

            if state.is_demolished {
                if !prior.was_demolished {
                    accumulator.demos_received += 1;
                }
                *prior = PreviousCarState {
                    was_demolished: true,
                    present: true,
                    ..*prior
                };
                continue;
            }

            // The roster slot decides the team, not the frame: some replays flip a player's
            // team attribute back and forth, which would mirror their positioning mid-match.
            let sign = team_sign(slot);
            let speed = length(&state.velocity);
            let height = state.position.z;
            let on_wall = (state.position.x.abs() > WALL_X || state.position.y.abs() > WALL_Y)
                && height > WALL_MINIMUM_HEIGHT;
            let grounded = height < GROUND_HEIGHT;
            let canonical_y = sign * state.position.y;
            let ball_canonical_y = sign * ball.position.y;
            let distance_to_ball = distance(&state.position, &ball.position);
            let own_goal = Vector3 {
                x: 0.0,
                y: -sign * 5120.0,
                z: 0.0,
            };

            accumulator.seconds += seconds;
            accumulator.speed += seconds * f64::from(speed);
            accumulator.supersonic += seconds * f64::from(u8::from(speed >= SUPERSONIC_SPEED));
            accumulator.slow += seconds * f64::from(u8::from(speed < SLOW_SPEED));
            accumulator.ground += seconds * f64::from(u8::from(grounded));
            accumulator.wall += seconds * f64::from(u8::from(on_wall));
            accumulator.high_air +=
                seconds * f64::from(u8::from(!on_wall && height >= HIGH_AIR_HEIGHT));
            accumulator.low_air +=
                seconds * f64::from(u8::from(!on_wall && !grounded && height < HIGH_AIR_HEIGHT));
            accumulator.height += seconds * f64::from(height);
            if !grounded && !on_wall {
                accumulator.airborne_seconds += seconds;
                accumulator.airborne_upright +=
                    seconds * f64::from(u8::from(up_z(&state.rotation) > 0.7));
            }

            accumulator.boost += seconds * f64::from(state.boost);
            accumulator.boost_empty += seconds * f64::from(u8::from(state.boost < 0.03));
            accumulator.boost_full += seconds * f64::from(u8::from(state.boost > 0.95));
            accumulator.slow_with_boost +=
                seconds * f64::from(u8::from(speed < SLOW_SPEED && state.boost > 0.5));

            if prior.present && !prior.was_demolished {
                let boost_change = state.boost - prior.boost;
                if boost_change > 0.0 {
                    accumulator.boost_collected += f64::from(boost_change);
                    if boost_change > BIG_PAD_INCREMENT {
                        accumulator.big_pads += 1;
                    } else if boost_change > SMALL_PAD_MINIMUM_INCREMENT {
                        accumulator.small_pads += 1;
                    }
                } else if boost_change < 0.0 {
                    accumulator.boost_used += f64::from(-boost_change);
                    if speed >= SUPERSONIC_SPEED {
                        accumulator.boost_while_supersonic += seconds;
                    }
                }
                if prior.height < GROUND_HEIGHT && !grounded && !on_wall {
                    accumulator.jumps += 1;
                }
            }

            accumulator.defensive_third +=
                seconds * f64::from(u8::from(canonical_y < -THIRD_BOUNDARY_Y));
            accumulator.offensive_third +=
                seconds * f64::from(u8::from(canonical_y > THIRD_BOUNDARY_Y));
            accumulator.canonical_y += seconds * f64::from(canonical_y);
            accumulator.behind_ball +=
                seconds * f64::from(u8::from(canonical_y < ball_canonical_y));
            accumulator.distance_to_ball += seconds * f64::from(distance_to_ball);
            accumulator.distance_to_own_goal +=
                seconds * f64::from(distance(&state.position, &own_goal));

            let seconds_since_jump =
                accumulate_mechanics(accumulator, prior, state, grounded, on_wall, seconds);

            *prior = PreviousCarState {
                boost: state.boost,
                height,
                was_demolished: false,
                present: true,
                jumping: state.controls.jumping,
                double_jumping: state.controls.double_jumping,
                dodging: state.controls.dodging,
                seconds_since_jump,
            };
            live_cars.push(LiveCar {
                slot,
                position: state.position,
                canonical_y,
                distance_to_ball,
            });
        }

        accumulate_team_roles(&mut accumulators, &live_cars, seconds);
        accumulate_double_commits(&mut accumulators, &live_cars, seconds);
        let possessor = live_cars
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
            .copied();
        accumulate_situational(
            &mut accumulators,
            &live_cars,
            ball.position.y,
            possessor.as_ref(),
            seconds,
        );
        for car in &live_cars {
            let Some(touched_at) = pending_recovery.get(car.slot).copied().flatten() else {
                continue;
            };
            let ball_canonical_y = team_sign(car.slot) * ball.position.y;
            if car.canonical_y < ball_canonical_y {
                if let Some(accumulator) = accumulators.get_mut(car.slot) {
                    accumulator.recovered_goal_side += 1;
                }
                pending_recovery[car.slot] = None;
            } else if frame.time - touched_at > RECOVERY_SECONDS {
                pending_recovery[car.slot] = None;
            }
        }
        for demolition in &frame.demolitions {
            let recent = last_demolition_time
                .get(&demolition.victim_actor_id)
                .is_some_and(|&time| frame.time - time < DEMOLITION_REPEAT_SECONDS);
            last_demolition_time.insert(demolition.victim_actor_id, frame.time);
            if recent {
                continue;
            }
            let attacker_slot = frame
                .players
                .iter()
                .find(|player| player.actor_id == demolition.attacker_actor_id)
                .and_then(|player| slot_by_name.get(player.name.as_str()));
            if let Some(&slot) = attacker_slot
                && let Some(accumulator) = accumulators.get_mut(slot)
            {
                accumulator.demos_inflicted += 1;
            }
        }

        if let Some(closest) = live_cars
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
        {
            if let Some(accumulator) = accumulators.get_mut(closest.slot) {
                accumulator.possession += seconds;
            }
            if let Some(before) = previous_ball_velocity {
                let change = Vector3 {
                    x: ball.velocity.x - before.x,
                    y: ball.velocity.y - before.y,
                    z: ball.velocity.z - before.z,
                };
                if length(&change) > TOUCH_MINIMUM_VELOCITY_CHANGE
                    && closest.distance_to_ball < TOUCH_MAXIMUM_DISTANCE
                    && let Some(accumulator) = accumulators.get_mut(closest.slot)
                {
                    let sign = if closest.slot < PLAYERS_PER_TEAM {
                        1.0
                    } else {
                        -1.0
                    };
                    let speed_after = length(&ball.velocity);
                    accumulator.touches += 1;
                    accumulator.recovery_touches += 1;
                    if let Some(pending) = pending_recovery.get_mut(closest.slot) {
                        *pending = Some(frame.time);
                    }
                    accumulator.ball_speed_after_touch += f64::from(speed_after);
                    accumulator.touch_speed_gain += f64::from(speed_after - length(&before));
                    accumulator.aerial_touches += u32::from(ball.position.z > HIGH_AIR_HEIGHT);
                    accumulator.forward_touches += u32::from(sign * ball.velocity.y > 0.0);
                }
            }
        }
        previous_ball_velocity = Some(ball.velocity);
    }

    let kickoff_count = accumulate_kickoffs(&mut accumulators, parsed, &slot_by_name, &frames);
    let scoreboard_by_name: HashMap<&str, &replay_structs::HeaderPlayerStats> = parsed
        .header_player_stats
        .iter()
        .filter(|_| whole_match)
        .map(|stats| (stats.name.as_str(), stats))
        .collect();
    core::array::from_fn(|slot| {
        let scoreboard = roster
            .names
            .get(slot)
            .and_then(|name| scoreboard_by_name.get(name.as_str()))
            .copied();
        accumulators
            .get(slot)
            .and_then(|accumulator| finish(accumulator, scoreboard, kickoff_count))
    })
}

/// Height below which a dodge on a descending car counts as a wavedash-like landing dodge.
const WAVEDASH_MAXIMUM_HEIGHT: f32 = 50.0;
/// Jump-to-dodge or jump-to-double-jump delay below which the move counts as "fast".
const FAST_FOLLOW_UP_SECONDS: f32 = 0.25;
/// Dodge torque component ratio above which the flip counts as pure pitch or pure roll.
const DOMINANT_TORQUE_RATIO: f32 = 2.0;

/// Input-derived mechanics for one car in one frame. Returns the updated time since the
/// last jump press.
fn accumulate_mechanics(
    accumulator: &mut Accumulator,
    prior: &PreviousCarState,
    state: &replay_structs::ActorState,
    grounded: bool,
    on_wall: bool,
    seconds: f64,
) -> f32 {
    let controls = state.controls;
    let airborne = !grounded && !on_wall;
    accumulator.boosting =
        seconds.mul_add(f64::from(u8::from(controls.boosting)), accumulator.boosting);
    accumulator.powerslide = seconds.mul_add(
        f64::from(u8::from(controls.handbrake && grounded)),
        accumulator.powerslide,
    );
    accumulator.air_roll = seconds.mul_add(
        f64::from(u8::from(controls.handbrake && airborne)),
        accumulator.air_roll,
    );
    accumulator.reverse = seconds.mul_add(
        f64::from(u8::from(controls.throttle < -0.1 && grounded)),
        accumulator.reverse,
    );
    accumulator.absolute_steer =
        seconds.mul_add(f64::from(controls.steer.abs()), accumulator.absolute_steer);
    if airborne {
        accumulator.air_angular_speed = seconds.mul_add(
            f64::from(length(&state.angular_velocity)),
            accumulator.air_angular_speed,
        );
    }

    let mut seconds_since_jump = if prior.present {
        prior.seconds_since_jump + seconds as f32
    } else {
        f32::MAX
    };
    if controls.jumping && !prior.jumping {
        accumulator.jump_presses += 1;
        seconds_since_jump = 0.0;
    }
    if controls.double_jumping && !prior.double_jumping {
        accumulator.double_jumps += 1;
        accumulator.fast_aerials += u32::from(seconds_since_jump < FAST_FOLLOW_UP_SECONDS);
    }
    if controls.dodging && !prior.dodging {
        accumulator.dodges += 1;
        let height = state.position.z;
        accumulator.wavedash_like_dodges +=
            u32::from(height < WAVEDASH_MAXIMUM_HEIGHT && state.velocity.z < 0.0);
        accumulator.aerial_dodges += u32::from(height > HIGH_AIR_HEIGHT);
        accumulator.fast_dodges += u32::from(seconds_since_jump < FAST_FOLLOW_UP_SECONDS);
        let pitch = controls.dodge_torque.y.abs();
        let roll = controls.dodge_torque.x.abs();
        if pitch + roll > f32::EPSILON {
            if pitch > DOMINANT_TORQUE_RATIO * roll {
                if controls.dodge_torque.y > 0.0 {
                    accumulator.forward_dodges += 1;
                } else {
                    accumulator.backward_dodges += 1;
                }
            } else if roll > DOMINANT_TORQUE_RATIO * pitch {
                accumulator.sideways_dodges += 1;
            } else {
                accumulator.diagonal_dodges += 1;
            }
        }
    }
    seconds_since_jump
}

/// Demolition events of the same victim closer together than this are one demolition.
const DEMOLITION_REPEAT_SECONDS: f32 = 1.0;
/// Ball speed that marks the first touch after a kickoff.
const KICKOFF_TOUCH_BALL_SPEED: f32 = 100.0;
/// Seconds after the first kickoff touch at which the ball side decides who won it.
const KICKOFF_OUTCOME_SECONDS: f32 = 3.0;
/// Two teammates closer than this to the ball at once are double-committing.
const DOUBLE_COMMIT_DISTANCE: f32 = 600.0;

/// Seconds after a touch within which getting back goal-side counts as a recovery.
const RECOVERY_SECONDS: f32 = 3.0;
/// Maximum distance to the ball for goal-side defending to count as shadowing.
const SHADOW_DISTANCE: f32 = 1500.0;
/// Distance band to the ball that counts as useful support while a teammate has it.
const SUPPORT_MINIMUM_DISTANCE: f32 = 800.0;
const SUPPORT_MAXIMUM_DISTANCE: f32 = 3000.0;

/// `+1` for blue slots, `−1` for orange: multiplies y into the team-canonical frame.
const fn team_sign(slot: usize) -> f32 {
    if slot < PLAYERS_PER_TEAM { 1.0 } else { -1.0 }
}

/// Positioning judged against the situation rather than as raw occupancy.
///
/// "Defending" means the ball is in the player's own half. The possessor is the car
/// closest to the ball over all six players.
fn accumulate_situational(
    accumulators: &mut [Accumulator; TOTAL_PLAYERS],
    live_cars: &[LiveCar],
    ball_y: f32,
    possessor: Option<&LiveCar>,
    seconds: f64,
) {
    for team_start in [0, PLAYERS_PER_TEAM] {
        let team: Vec<&LiveCar> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + PLAYERS_PER_TEAM).contains(&car.slot))
            .collect();
        let Some(first) = team.first() else {
            continue;
        };
        let sign = team_sign(first.slot);
        let ball_canonical_y = sign * ball_y;
        let defending = ball_canonical_y < 0.0;
        let last_back = team
            .iter()
            .min_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y))
            .map(|car| car.slot);
        let closest_on_team = team
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball))
            .map(|car| car.slot);
        let possessor_slot = possessor.map(|car| car.slot);
        let opponents_have_ball = possessor_slot
            .is_some_and(|slot| !(team_start..team_start + PLAYERS_PER_TEAM).contains(&slot));
        let own_goal = Vector3 {
            x: 0.0,
            y: -sign * 5120.0,
            z: 0.0,
        };

        for car in &team {
            let Some(accumulator) = accumulators.get_mut(car.slot) else {
                continue;
            };
            let goal_side = car.canonical_y < ball_canonical_y;
            if defending {
                accumulator.defending_seconds += seconds;
                if goal_side {
                    accumulator.goal_side_defending += seconds;
                }
                if last_back == Some(car.slot) {
                    accumulator.last_back_defending += seconds;
                    accumulator.last_back_goal_distance_defending = seconds.mul_add(
                        f64::from(distance(&car.position, &own_goal)),
                        accumulator.last_back_goal_distance_defending,
                    );
                }
                if opponents_have_ball {
                    accumulator.opponent_possession_own_half += seconds;
                    if goal_side && car.distance_to_ball < SHADOW_DISTANCE {
                        accumulator.shadow_defense += seconds;
                    }
                }
            }
            if !goal_side && closest_on_team != Some(car.slot) {
                accumulator.overcommit += seconds;
            }
            let teammate_has_ball = possessor_slot.is_some_and(|slot| {
                slot != car.slot && (team_start..team_start + PLAYERS_PER_TEAM).contains(&slot)
            });
            if teammate_has_ball {
                accumulator.teammate_possession += seconds;
                if (SUPPORT_MINIMUM_DISTANCE..=SUPPORT_MAXIMUM_DISTANCE)
                    .contains(&car.distance_to_ball)
                {
                    accumulator.support_spacing += seconds;
                }
            }
        }
    }
}

/// Credits time spent double-committing: two or more teammates near the ball at once.
fn accumulate_double_commits(
    accumulators: &mut [Accumulator; TOTAL_PLAYERS],
    live_cars: &[LiveCar],
    seconds: f64,
) {
    for team_start in [0, PLAYERS_PER_TEAM] {
        let committed: Vec<usize> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + PLAYERS_PER_TEAM).contains(&car.slot))
            .filter(|car| car.distance_to_ball < DOUBLE_COMMIT_DISTANCE)
            .map(|car| car.slot)
            .collect();
        if committed.len() >= 2 {
            for slot in committed {
                if let Some(accumulator) = accumulators.get_mut(slot) {
                    accumulator.double_commit += seconds;
                }
            }
        }
    }
}

/// For every kickoff: who touched the ball first, and whether the ball was on the
/// opponents' side [`KICKOFF_OUTCOME_SECONDS`] later. Returns the number of kickoffs seen.
fn accumulate_kickoffs(
    accumulators: &mut [Accumulator; TOTAL_PLAYERS],
    parsed: &ParsedReplay,
    slot_by_name: &HashMap<&str, usize>,
    frames: &Range<usize>,
) -> u32 {
    let mut kickoffs = 0;
    for &kickoff in parsed
        .kickoff_frames
        .iter()
        .filter(|kickoff| frames.contains(kickoff))
    {
        let Some(touch_index) = parsed
            .frames
            .iter()
            .enumerate()
            .skip(kickoff)
            .find(|(_, frame)| length(&frame.ball.velocity) > KICKOFF_TOUCH_BALL_SPEED)
            .map(|(index, _)| index)
        else {
            continue;
        };
        let Some(touch_frame) = parsed.frames.get(touch_index) else {
            continue;
        };
        let toucher = touch_frame
            .players
            .iter()
            .filter(|player| !player.actor_state.is_demolished)
            .min_by(|a, b| {
                distance(&a.actor_state.position, &touch_frame.ball.position).total_cmp(&distance(
                    &b.actor_state.position,
                    &touch_frame.ball.position,
                ))
            });
        let Some(toucher) = toucher else {
            continue;
        };
        kickoffs += 1;
        let Some(&slot) = slot_by_name.get(toucher.name.as_str()) else {
            continue;
        };
        let outcome_frame = parsed
            .frames
            .iter()
            .skip(touch_index)
            .find(|frame| frame.time - touch_frame.time >= KICKOFF_OUTCOME_SECONDS);
        if let Some(accumulator) = accumulators.get_mut(slot) {
            accumulator.kickoff_first_touches += 1;
            if let Some(outcome) = outcome_frame {
                let won = team_sign(slot) * outcome.ball.position.y > 0.0;
                accumulator.kickoff_wins += u32::from(won);
            }
        }
    }
    kickoffs
}

/// Per-frame team roles: closest to ball, last back, most forward, nearest-teammate spacing.
fn accumulate_team_roles(
    accumulators: &mut [Accumulator; TOTAL_PLAYERS],
    live_cars: &[LiveCar],
    seconds: f64,
) {
    for team_start in [0, PLAYERS_PER_TEAM] {
        let team: Vec<&LiveCar> = live_cars
            .iter()
            .filter(|car| (team_start..team_start + PLAYERS_PER_TEAM).contains(&car.slot))
            .collect();
        if team.len() < 2 {
            continue;
        }
        let closest = team
            .iter()
            .min_by(|a, b| a.distance_to_ball.total_cmp(&b.distance_to_ball));
        let last_back = team
            .iter()
            .min_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y));
        let most_forward = team
            .iter()
            .max_by(|a, b| a.canonical_y.total_cmp(&b.canonical_y));
        if let Some(car) = closest
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.closest_on_team += seconds;
        }
        if let Some(car) = last_back
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.last_back_on_team += seconds;
        }
        if let Some(car) = most_forward
            && let Some(accumulator) = accumulators.get_mut(car.slot)
        {
            accumulator.most_forward_on_team += seconds;
        }
        for car in &team {
            let nearest = team
                .iter()
                .filter(|other| other.slot != car.slot)
                .map(|other| distance(&car.position, &other.position))
                .fold(f32::MAX, f32::min);
            if let Some(accumulator) = accumulators.get_mut(car.slot) {
                accumulator.nearest_teammate_distance =
                    seconds.mul_add(f64::from(nearest), accumulator.nearest_teammate_distance);
                accumulator.nearest_teammate_seconds += seconds;
            }
        }
    }
}

fn finish(
    accumulator: &Accumulator,
    scoreboard: Option<&replay_structs::HeaderPlayerStats>,
    kickoff_count: u32,
) -> Option<PlayerMatchStats> {
    if accumulator.seconds <= 0.0 {
        return None;
    }
    let seconds = accumulator.seconds;
    let minutes = seconds / 60.0;
    let fraction = |value: f64| (value / seconds) as f32;
    let per_minute = |value: f64| (value / minutes) as f32;
    let ratio = |numerator: f64, denominator: f64| {
        if denominator > 0.0 {
            (numerator / denominator) as f32
        } else {
            0.0
        }
    };
    let per_dodge = |count: u32| {
        if accumulator.dodges == 0 {
            0.0
        } else {
            count as f32 / accumulator.dodges as f32
        }
    };
    let per_touch = |value: f64| {
        if accumulator.touches == 0 {
            0.0
        } else {
            (value / f64::from(accumulator.touches)) as f32
        }
    };
    let values = [
        fraction(accumulator.speed) / 2300.0,
        fraction(accumulator.supersonic),
        fraction(accumulator.slow),
        fraction(accumulator.ground),
        fraction(accumulator.low_air),
        fraction(accumulator.high_air),
        fraction(accumulator.wall),
        fraction(accumulator.height) / 2044.0,
        per_minute(f64::from(accumulator.jumps)),
        fraction(accumulator.boost),
        fraction(accumulator.boost_empty),
        fraction(accumulator.boost_full),
        per_minute(accumulator.boost_collected * 100.0),
        per_minute(accumulator.boost_used * 100.0),
        per_minute(f64::from(accumulator.big_pads)),
        per_minute(f64::from(accumulator.small_pads)),
        fraction(accumulator.boost_while_supersonic),
        fraction(accumulator.defensive_third),
        fraction(accumulator.offensive_third),
        fraction(accumulator.canonical_y) / 5120.0,
        fraction(accumulator.behind_ball),
        fraction(accumulator.distance_to_ball) / 7245.0,
        fraction(accumulator.closest_on_team),
        fraction(accumulator.last_back_on_team),
        fraction(accumulator.most_forward_on_team),
        if accumulator.nearest_teammate_seconds > 0.0 {
            (accumulator.nearest_teammate_distance / accumulator.nearest_teammate_seconds) as f32
                / 7245.0
        } else {
            0.0
        },
        fraction(accumulator.distance_to_own_goal) / 7245.0,
        fraction(accumulator.possession),
        per_minute(f64::from(accumulator.touches)),
        per_touch(accumulator.ball_speed_after_touch) / 6000.0,
        per_touch(f64::from(accumulator.aerial_touches)),
        per_touch(f64::from(accumulator.forward_touches)),
        per_touch(accumulator.touch_speed_gain) / 6000.0,
        accumulator.goals as f32,
        accumulator.demos_received as f32,
        if accumulator.airborne_seconds > 0.0 {
            (accumulator.airborne_upright / accumulator.airborne_seconds) as f32
        } else {
            0.0
        },
        fraction(accumulator.slow_with_boost),
        if accumulator.boost_used > 0.0 {
            (accumulator.supersonic / (accumulator.boost_used * 100.0)) as f32
        } else {
            0.0
        },
        per_minute(f64::from(accumulator.jump_presses)),
        per_minute(f64::from(accumulator.double_jumps)),
        per_minute(f64::from(accumulator.dodges)),
        per_minute(f64::from(accumulator.wavedash_like_dodges)),
        per_minute(f64::from(accumulator.aerial_dodges)),
        per_dodge(accumulator.forward_dodges),
        per_dodge(accumulator.backward_dodges),
        per_dodge(accumulator.sideways_dodges),
        per_dodge(accumulator.diagonal_dodges),
        per_dodge(accumulator.fast_dodges),
        fraction(accumulator.boosting),
        fraction(accumulator.powerslide),
        fraction(accumulator.air_roll),
        fraction(accumulator.reverse),
        fraction(accumulator.absolute_steer),
        if accumulator.airborne_seconds > 0.0 {
            (accumulator.air_angular_speed / accumulator.airborne_seconds) as f32
        } else {
            0.0
        },
        if accumulator.double_jumps > 0 {
            accumulator.fast_aerials as f32 / accumulator.double_jumps as f32
        } else {
            0.0
        },
        scoreboard.map_or(0.0, |s| s.score as f32 / minutes as f32),
        scoreboard.map_or(0.0, |s| s.assists as f32),
        scoreboard.map_or(0.0, |s| s.saves as f32),
        scoreboard.map_or(0.0, |s| s.shots as f32),
        scoreboard.map_or(0.0, |s| {
            if s.shots > 0 {
                s.goals as f32 / s.shots as f32
            } else {
                0.0
            }
        }),
        accumulator.demos_inflicted as f32,
        if kickoff_count > 0 {
            accumulator.kickoff_first_touches as f32 / kickoff_count as f32
        } else {
            0.0
        },
        if accumulator.kickoff_first_touches > 0 {
            accumulator.kickoff_wins as f32 / accumulator.kickoff_first_touches as f32
        } else {
            0.0
        },
        fraction(accumulator.double_commit),
        ratio(
            accumulator.last_back_defending,
            accumulator.defending_seconds,
        ),
        ratio(
            accumulator.goal_side_defending,
            accumulator.defending_seconds,
        ),
        ratio(
            accumulator.last_back_goal_distance_defending,
            accumulator.last_back_defending,
        ) / 7245.0,
        fraction(accumulator.overcommit),
        ratio(
            accumulator.shadow_defense,
            accumulator.opponent_possession_own_half,
        ),
        ratio(accumulator.support_spacing, accumulator.teammate_possession),
        ratio(
            f64::from(accumulator.recovered_goal_side),
            f64::from(accumulator.recovery_touches),
        ),
    ];
    Some(PlayerMatchStats {
        values,
        live_seconds: seconds as f32,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, GameFrame, PlayerState, Team};

    use super::*;

    fn car(name: &str, team: Team, x: f32, y: f32, boost: f32) -> PlayerState {
        PlayerState {
            actor_id: 0,
            name: Arc::new(name.to_string()),
            team,
            actor_state: ActorState {
                position: Vector3 { x, y, z: 17.0 },
                velocity: Vector3 {
                    x: 0.0,
                    y: 1000.0,
                    z: 0.0,
                },
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                boost,
                is_demolished: false,
                ..ActorState::default()
            },
        }
    }

    fn roster() -> PlayerRoster {
        PlayerRoster {
            names: core::array::from_fn(|slot| {
                ["b0", "b1", "b2", "o0", "o1", "o2"][slot].to_string()
            }),
        }
    }

    fn frame(blue_y: f32, orange_y: f32) -> GameFrame {
        GameFrame {
            time: 0.0,
            delta: 1.0 / 30.0,
            seconds_remaining: 200,
            ball: BallState::default(),
            demolitions: Vec::new(),
            players: vec![
                car("b0", Team::Blue, 0.0, blue_y, 0.5),
                car("b1", Team::Blue, 500.0, blue_y + 100.0, 0.5),
                car("b2", Team::Blue, -500.0, blue_y + 200.0, 0.5),
                car("o0", Team::Orange, 0.0, orange_y, 0.5),
                car("o1", Team::Orange, 500.0, orange_y - 100.0, 0.5),
                car("o2", Team::Orange, -500.0, orange_y - 200.0, 0.5),
            ],
        }
    }

    fn index_of(name: &str) -> usize {
        MATCH_STAT_NAMES
            .iter()
            .position(|stat| *stat == name)
            .unwrap_or(usize::MAX)
    }

    /// Mirrored positions produce identical positional stats for the two teams.
    #[test]
    fn positions_are_team_canonical() {
        let parsed = ParsedReplay {
            frames: (0..60).map(|_| frame(-4000.0, 4000.0)).collect(),
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let blue = stats[0]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let orange = stats[3]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let defensive = index_of("defensive_third_fraction");
        assert!((blue[defensive] - 1.0).abs() < 1e-6);
        assert!((orange[defensive] - 1.0).abs() < 1e-6);
        let last_back = index_of("last_back_on_team_fraction");
        assert!((blue[last_back] - 1.0).abs() < 1e-6);
        assert!((orange[last_back] - 1.0).abs() < 1e-6);
        assert!(
            (blue[index_of("mean_canonical_y")] - orange[index_of("mean_canonical_y")]).abs()
                < 1e-6
        );
    }

    #[test]
    fn names_and_values_have_the_same_length() {
        assert_eq!(MATCH_STAT_NAMES.len(), MATCH_STAT_COUNT);
        let parsed = ParsedReplay {
            frames: vec![frame(0.0, 0.0), frame(0.0, 0.0)],
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        assert!(stats.iter().all(Option::is_some));
    }

    /// A big pad pickup is counted once and the boost gain is recorded.
    #[test]
    fn big_pad_pickup_is_counted() {
        let mut frames: Vec<GameFrame> = (0..30).map(|_| frame(0.0, 0.0)).collect();
        for later in frames.iter_mut().skip(15) {
            later.players[0].actor_state.boost = 1.0;
        }
        let parsed = ParsedReplay {
            frames,
            ..ParsedReplay::default()
        };
        let stats = compute_player_match_stats(&parsed, &roster());
        let values = stats[0]
            .as_ref()
            .map_or([0.0; MATCH_STAT_COUNT], |s| s.values);
        let seconds = 30.0 / 30.0;
        assert!((values[index_of("big_pads_per_minute")] - 60.0 / seconds).abs() < 1e-3);
    }
}

/// Splits a replay into windows that end at each goal: kickoff → goal, kickoff → goal, …,
/// last kickoff → end. Goal-replay frames fall inside the window after the goal and are
/// skipped by the stats themselves.
#[must_use]
pub fn goal_windows(parsed: &ParsedReplay) -> Vec<Range<usize>> {
    let mut boundaries: Vec<usize> = parsed
        .goal_frames
        .iter()
        .map(|&frame| frame + 1)
        .filter(|&frame| frame < parsed.frames.len())
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut windows = Vec::with_capacity(boundaries.len() + 1);
    let mut start = 0;
    for boundary in boundaries {
        if boundary > start {
            windows.push(start..boundary);
            start = boundary;
        }
    }
    if start < parsed.frames.len() {
        windows.push(start..parsed.frames.len());
    }
    windows
}

/// Splits a replay into consecutive windows of about `seconds` of **live** play (goal
/// replays excluded). A trailing remainder shorter than half a window joins the previous
/// window.
#[must_use]
pub fn time_windows(parsed: &ParsedReplay, seconds: f32) -> Vec<Range<usize>> {
    let excluded = build_goal_replay_excluded_set(&parsed.goal_frames, &parsed.kickoff_frames);
    let mut windows = Vec::new();
    let mut start = 0;
    let mut live = 0.0;
    for (index, frame) in parsed.frames.iter().enumerate() {
        if !excluded.contains(&index) {
            live += frame.delta.clamp(0.0, MAXIMUM_FRAME_SECONDS);
        }
        if live >= seconds {
            windows.push(start..index + 1);
            start = index + 1;
            live = 0.0;
        }
    }
    if start < parsed.frames.len() {
        if live < seconds / 2.0
            && let Some(last) = windows.last_mut()
        {
            last.end = parsed.frames.len();
        } else {
            windows.push(start..parsed.frames.len());
        }
    }
    windows
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn frames(count: usize) -> Vec<replay_structs::GameFrame> {
        (0..count)
            .map(|index| replay_structs::GameFrame {
                time: index as f32 / 30.0,
                delta: 1.0 / 30.0,
                ..replay_structs::GameFrame::default()
            })
            .collect()
    }

    /// Goal windows tile the replay exactly, cutting just after each goal.
    #[test]
    fn goal_windows_tile_the_replay() {
        let parsed = ParsedReplay {
            frames: frames(1000),
            goal_frames: vec![300, 700],
            ..ParsedReplay::default()
        };
        let windows = goal_windows(&parsed);
        assert_eq!(windows, vec![0..301, 301..701, 701..1000]);
    }

    /// Time windows tile the replay; a short remainder joins the last window.
    #[test]
    fn time_windows_tile_the_replay() {
        let parsed = ParsedReplay {
            frames: frames(30 * 130),
            ..ParsedReplay::default()
        };
        let windows = time_windows(&parsed, 60.0);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows.first().map(|w| w.start), Some(0));
        assert_eq!(windows.last().map(|w| w.end), Some(30 * 130));
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "windows must be contiguous");
        }
    }
}

#[cfg(test)]
mod situational_tests {
    use std::sync::Arc;

    use replay_structs::{ActorState, BallState, GameFrame, PlayerState, Team};

    use super::*;

    fn car(name: &str, team: Team, x: f32, y: f32) -> PlayerState {
        PlayerState {
            actor_id: 0,
            name: Arc::new(name.to_string()),
            team,
            actor_state: ActorState {
                position: Vector3 { x, y, z: 17.0 },
                rotation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                ..ActorState::default()
            },
        }
    }

    fn stat(values: &[f32; MATCH_STAT_COUNT], name: &str) -> f32 {
        MATCH_STAT_NAMES
            .iter()
            .position(|stat| *stat == name)
            .map_or(f32::NAN, |index| values[index])
    }

    /// Ball deep in blue's half, an orange attacker on it. Blue: b0 in net (last back,
    /// goal-side), b1 shadowing goal-side near the ball, b2 stranded upfield (overcommitted).
    #[test]
    fn defending_roles_are_credited_to_the_right_players() {
        let frame = GameFrame {
            time: 0.0,
            delta: 1.0 / 30.0,
            seconds_remaining: 200,
            ball: BallState {
                position: Vector3 {
                    x: 0.0,
                    y: -3000.0,
                    z: 93.0,
                },
                ..BallState::default()
            },
            players: vec![
                car("b0", Team::Blue, 0.0, -5000.0),
                car("b1", Team::Blue, 300.0, -3800.0),
                car("b2", Team::Blue, 0.0, 2000.0),
                car("o0", Team::Orange, 0.0, -2900.0),
                car("o1", Team::Orange, 1000.0, 0.0),
                car("o2", Team::Orange, 0.0, 4000.0),
            ],
            demolitions: Vec::new(),
        };
        let parsed = ParsedReplay {
            frames: vec![frame; 60],
            ..ParsedReplay::default()
        };
        let roster = PlayerRoster {
            names: core::array::from_fn(|slot| {
                ["b0", "b1", "b2", "o0", "o1", "o2"][slot].to_string()
            }),
        };
        let stats = compute_player_match_stats(&parsed, &roster);
        let values = |slot: usize| {
            stats[slot]
                .as_ref()
                .map_or([0.0; MATCH_STAT_COUNT], |s| s.values)
        };

        assert!((stat(&values(0), "last_back_when_defending_fraction") - 1.0).abs() < 1e-6);
        assert!((stat(&values(1), "last_back_when_defending_fraction")).abs() < 1e-6);
        assert!((stat(&values(0), "goal_side_when_defending_fraction") - 1.0).abs() < 1e-6);
        assert!((stat(&values(2), "goal_side_when_defending_fraction")).abs() < 1e-6);
        assert!((stat(&values(1), "shadow_defense_fraction") - 1.0).abs() < 1e-6);
        assert!((stat(&values(0), "shadow_defense_fraction")).abs() < 1e-6);
        assert!((stat(&values(2), "overcommit_fraction") - 1.0).abs() < 1e-6);
        assert!((stat(&values(0), "overcommit_fraction")).abs() < 1e-6);
        // Orange is attacking, not defending.
        assert!((stat(&values(3), "goal_side_when_defending_fraction")).abs() < 1e-6);
    }
}
