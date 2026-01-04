# Feature Specification for Rocket League Impact Model

This document describes all features extracted from game frames for ML training and inference.

## Design Principles

1. **Normalized to [-1, 1]** - All features are normalized for neural network stability
2. **Team-agnostic ordering** - Players sorted by team (3 blue, 3 orange) for consistent input
3. **Symmetric** - Features work the same for both teams (goal positions flipped as needed)
4. **Per-frame** - Each feature vector represents a single moment in time

## Feature Layout

### Summary

| Category | Features per unit | Units | Total |
|----------|-------------------|-------|-------|
| Ball State | 7 | 1 | 7 |
| Player State | 13 | 6 | 78 |
| Player Geometry | 7 | 6 | 42 |
| Game Context | 1 | 1 | 1 |
| **TOTAL** | | | **128** |

---

## 1. Ball State (7 features)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `ball_pos_x` | [-1, 1] | Ball X position (normalized by field half-length) |
| 1 | `ball_pos_y` | [-1, 1] | Ball Y position (normalized by field half-width) |
| 2 | `ball_pos_z` | [0, 1] | Ball Z position (normalized by max height) |
| 3 | `ball_vel_x` | [-1, 1] | Ball X velocity (normalized by max velocity) |
| 4 | `ball_vel_y` | [-1, 1] | Ball Y velocity (toward orange goal = positive) |
| 5 | `ball_vel_z` | [-1, 1] | Ball Z velocity (normalized by max velocity) |
| 6 | `ball_speed` | [0, 1] | Ball speed magnitude (normalized) |

---

## 2. Player State (13 features × 6 players = 78 features)

Players are ordered: [Blue1, Blue2, Blue3, Orange1, Orange2, Orange3]
Within each team, players are sorted by actor_id for consistency.

For each player (indices relative to player offset):

| Offset | Feature | Range | Description |
|--------|---------|-------|-------------|
| 0 | `pos_x` | [-1, 1] | Player X position |
| 1 | `pos_y` | [-1, 1] | Player Y position |
| 2 | `pos_z` | [0, 1] | Player Z position |
| 3 | `vel_x` | [-1, 1] | Player X velocity |
| 4 | `vel_y` | [-1, 1] | Player Y velocity |
| 5 | `vel_z` | [-1, 1] | Player Z velocity |
| 6 | `rot_x` | [-1, 1] | Quaternion X component |
| 7 | `rot_y` | [-1, 1] | Quaternion Y component |
| 8 | `rot_z` | [-1, 1] | Quaternion Z component |
| 9 | `rot_w` | [-1, 1] | Quaternion W component |
| 10 | `speed` | [0, 1] | Speed magnitude (0=still, 1=supersonic) |
| 11 | `boost` | [0, 1] | Boost amount |
| 12 | `is_demolished` | {0, 1} | 1 if currently demolished |

**Player Index Mapping:**
- Features 7-19: Blue Player 1
- Features 20-32: Blue Player 2
- Features 33-45: Blue Player 3
- Features 46-58: Orange Player 1
- Features 59-71: Orange Player 2
- Features 72-84: Orange Player 3

---

## 3. Player Geometry (7 features × 6 players = 42 features)

Derived geometric relationships for each player:

| Offset | Feature | Range | Description |
|--------|---------|-------|-------------|
| 0 | `dist_to_ball` | [0, 1] | Distance to ball (normalized by field diagonal) |
| 1 | `facing_ball` | [-1, 1] | Dot product of forward vector with direction to ball |
| 2 | `dist_to_teammate_1` | [0, 1] | Distance to 1st teammate (sorted by actor_id) |
| 3 | `dist_to_teammate_2` | [0, 1] | Distance to 2nd teammate (sorted by actor_id) |
| 4 | `dist_to_opponent_1` | [0, 1] | Distance to 1st opponent (sorted by actor_id) |
| 5 | `dist_to_opponent_2` | [0, 1] | Distance to 2nd opponent (sorted by actor_id) |
| 6 | `dist_to_opponent_3` | [0, 1] | Distance to 3rd opponent (sorted by actor_id) |

**Index Mapping:**
- Features 85-91: Blue Player 1
- Features 92-98: Blue Player 2
- Features 99-105: Blue Player 3
- Features 106-112: Orange Player 1
- Features 113-119: Orange Player 2
- Features 120-126: Orange Player 3

---

## 4. Game Context (1 feature)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 127 | `ball_dist_to_blue_goal` | [0, 1] | Distance from ball to blue goal |

Note: `ball_dist_to_orange_goal` is not necessary because it's perfectly negatively correlated with `ball_dist_to_blue_goal`.

---

## Constants

```rust
// Field dimensions (Unreal units)
const FIELD_HALF_LENGTH: f32 = 5120.0;  // X axis
const FIELD_HALF_WIDTH: f32 = 5120.0;   // Y axis (goal line)
const FIELD_MAX_HEIGHT: f32 = 2044.0;   // Z axis
const FIELD_DIAGONAL: f32 = 7245.0;     // sqrt(5120^2 + 5120^2) for distance normalization

// Velocity
const MAX_CAR_SPEED: f32 = 2300.0;      // Supersonic threshold
const MAX_BALL_SPEED: f32 = 6000.0;     // Approximate max

// Goal positions (Y coordinate)
const BLUE_GOAL_Y: f32 = -5120.0;
const ORANGE_GOAL_Y: f32 = 5120.0;
```

---


---

## Usage Notes

### For Training
- Features are extracted per-frame
- Labels are player MMR values
- The model learns to predict skill level from game state

### For Inference (BakkesMod integration)
- Extract features from live game state
- Run model forward pass
- Output is predicted MMR/impact score
- Update at ~30Hz for real-time display
