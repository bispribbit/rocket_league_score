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
│  feature_extractor  │  Extract per-frame features (~75 floats)
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

2. **Per-frame features**: ~75 features including:
   - Ball position and velocity (6 floats)
   - Per player (6 players × 11 features = 66 floats):
     - Position relative to ball and own goal
     - Velocity
     - Boost level
     - Demolished flag
   - Team-level features (possession, ball-to-goal distances)

3. **Segment-based training**: Replays are split into segments between kickoffs and goals for focused learning.

4. **Minimal database storage**: Only metadata and player ratings stored, not full replay data.

---

## Implementation Status

### Completed (Structure in Place)

- [x] Workspace configuration with all crates
- [x] `replay_parser` types: `ParsedReplay`, `GameFrame`, `PlayerState`, `BallState`, `GameSegment`
- [x] `feature_extractor` types: `FrameFeatures`, `PlayerRating`, `TrainingSample`
- [x] `ml_model` types: `ImpactModel`, `TrainingConfig`, `ModelConfig`, `TrainingData`
- [x] Database migrations for all tables
- [x] Repository functions (CRUD for replays, players, models)
- [x] CLI commands structure (ingest, train, predict, migrate)

### TODO: replay_parser

- [ ] **Implement `parse_replay()` with boxcars**: Currently returns mock data. Need to:
  - Parse network frames from boxcars `Replay`
  - Extract ball actor positions/velocities per frame
  - Extract player actor positions/velocities/boost per frame
  - Identify goal and kickoff events from game events
  
- [ ] **Implement `segment_by_goals()`**: Currently returns mock segments. Need to:
  - Detect kickoff frames (ball at center, countdown complete)
  - Detect goal frames from game events
  - Determine which team scored

### TODO: feature_extractor

- [ ] **Implement `extract_frame_features()`**: Currently only extracts ball position. Need to:
  - Extract all 6 player positions relative to ball and goals
  - Calculate distances to nearest opponents/teammates
  - Add boost levels and demolished flags
  - Compute team possession indicator

- [ ] **Implement `normalize_features()`**: Need to compute mean/std from training data and apply normalization.

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

