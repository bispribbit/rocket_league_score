# Rocket League Impact Score Calculator - Architecture

## Overview

An ML-based system to evaluate player performance in Rocket League replays, outputting an "impact score" that reflects skill level (similar to MMR: 100-300 for bronze, 1500+ for elite players).

## Crate Structure

```
crates/
├── database/             # PostgreSQL access via SQLx
├── replay_parser/        # Parse .replay files using boxcars
├── feature_extractor/    # Extract ML features from frame data
├── ml_model/             # Burn neural network for prediction
└── rocket_league_score/  # CLI binary (ingest/train/predict)
```

## Data Flow

```
.replay files
    │
    ▼
┌─────────────────┐
│  replay_parser  │  Parse with boxcars → ParsedReplay, GameFrame, PlayerState
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  feature_extractor  │  Extract per-frame features (147 floats)
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│    ml_model     │  Burn neural network → Impact Score (0-2000+)
└────────┬────────┘
         │
         ▼
    Impact Score
```

## Database Schema

### Tables

- **replays**: Metadata (id, file_path, game_mode, timestamps)
- **replay_players**: Player ratings per replay (player_name, team, skill_rating as MMR)
- **models**: Trained model versions (name, version, checkpoint_path, config, metrics)

### Game Modes

`soccar_3v3`, `soccar_2v2`, `soccar_1v1`, `hoops`, `rumble`, `dropshot`, `snowday`

## CLI Commands

```bash
# Ingest replays with player MMR ratings
rocket_league_score ingest --folder replays/3v3 --game-mode 3v3 --ratings-file ratings.csv

# Train the model
rocket_league_score train --name impact_model --epochs 100 --batch-size 64

# Predict impact scores for a replay
rocket_league_score predict --replay game.replay --model impact_model

# Run database migrations
rocket_league_score migrate
```

## Key Design Decisions

1. **MMR-based labels**: Using actual player MMR (integer) instead of categorical skill labels allows the model to learn the continuous skill spectrum.

2. **Per-frame features**: 147 features including:
   - Ball state (7): position, velocity, speed
   - Per player state (6 × 13 = 78): position, velocity, rotation quaternion, speed, boost, demolished
   - Per player geometry (6 × 9 = 54): distance to ball, distance to own goal, facing ball, goal line position, distance to each teammate (2), distance to each opponent (3)
   - Team context (2 × 3 = 6): team centroid, average boost
   - Game context (2): ball distance to each goal

3. **Segment-based training**: Replays are split into segments between kickoffs and goals for focused learning.

4. **Minimal database storage**: Only metadata and player ratings stored, not full replay data.

---

## Implementation Status

### Completed (Structure in Place)

- [x] Workspace configuration with all crates
- [x] `replay_parser` types: `ParsedReplay`, `GameFrame`, `PlayerState`, `BallState`, `GameSegment`, `Quaternion`
- [x] `feature_extractor` types: `FrameFeatures`, `PlayerRating`, `TrainingSample`
- [x] `ml_model` types: `ImpactModel`, `TrainingConfig`, `ModelConfig`, `TrainingData`
- [x] Database migrations for all tables
- [x] Repository functions (CRUD for replays, players, models)
- [x] CLI commands structure (ingest, train, predict, migrate)

### DONE: replay_parser

- [x] **Implemented `parse_replay()` with boxcars**:
  - Parses network frames from boxcars `Replay`
  - Extracts ball actor positions/velocities per frame
  - Extracts player/car actor positions/velocities/boost per frame
  - Extracts goal events from replay header
  - Detects kickoff frames (ball at center position)
  - Tracks team assignments via TeamPaint and FlaggedByte attributes
  
- [x] **Implemented `segment_by_goals()`**:
  - Creates segments between kickoffs and goals
  - Identifies which team scored from goal events
  
- [ ] **Player names**: Currently showing as `Player_X` - need to properly link PlayerReplicationInfo names to car actors

### DONE: feature_extractor

- [x] **Implemented `extract_frame_features()`**: Full 122-feature extraction:
  - Ball state: position, velocity, speed, height classification, trajectory toward goals
  - Player state: position, velocity, rotation (quaternion), speed magnitude, boost, demolished
  - Player geometry: distance to ball, distance to own goal, facing ball angle, between-ness score
  - Team context: team centroid position, average boost
  - Game context: time remaining, overtime flag, ball-to-goal distances
  
- [x] **Normalization**: All features normalized to [-1, 1] or [0, 1] ranges during extraction

- [x] **FEATURES.md**: Comprehensive documentation of all 122 features with indices

### TODO: ml_model

- [ ] **Implement `train()`**: Currently a no-op. Need to:
  - Convert `TrainingData` to Burn tensors/datasets
  - Create data loaders with batching
  - Implement training loop with Adam optimizer
  - Compute MSE loss between predicted and target MMR
  - Add validation and early stopping

- [ ] **Implement `predict()`**: Currently returns mock score. Need to:
  - Properly convert features to tensor
  - Run forward pass
  - Extract scalar output

- [ ] **Implement `save_checkpoint()` / `load_checkpoint()`**: Need to use Burn's record system for model serialization.

### TODO: CLI / ingest

- [ ] **Implement CSV ratings parsing**: Currently returns empty vec. Need to parse format: `replay_filename,player_name,team,skill_rating`

### Future Enhancements

- [ ] BakkesMod integration for real-time scoring
- [ ] Support for 2v2 and 1v1 game modes
- [ ] Advanced features: 50/50 outcomes, passing sequences, boost steal detection
- [ ] Per-player scoring (not just per-frame)
- [ ] Training dashboard with loss curves

---

## Training Data Requirements

To train the model, you need:

1. **Replay files** from various skill levels (bronze to SSL)
2. **Player ratings CSV** with format:
   ```csv
   replay_filename,player_name,team,skill_rating
   abc123.replay,PlayerOne,0,1547
   abc123.replay,PlayerTwo,0,1623
   abc123.replay,PlayerThree,0,1489
   abc123.replay,OpponentA,1,1512
   ...
   ```

The model learns to predict what MMR-level play "looks like" at any given frame, then outputs an impact score reflecting skill level.

