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
└── rocket_league_score/  # CLI binary (ingest/train/predict/test-pipeline)
```

## Data Flow

```
.replay files + metadata.jsonl
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
# End-to-end pipeline test (no database required)
rocket_league_score test-pipeline \
  --replay-dir replays/3v3 \
  --metadata replays/3v3/metadata.jsonl \
  --num-replays 5

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

5. **Metadata from ballchasing.com**: Player ranks are extracted from `metadata.jsonl` files and converted to MMR using rank-to-MMR mapping tables.

---

## Implementation Status

### ✅ DONE: Core Infrastructure

- [x] Workspace configuration with all crates
- [x] Database migrations for all tables
- [x] Repository functions (CRUD for replays, players, models)
- [x] CLI commands structure (ingest, train, predict, migrate, test-pipeline)

### ✅ DONE: replay_parser

- [x] **Implemented `parse_replay()` with boxcars**:
  - Parses network frames from boxcars `Replay`
  - Extracts ball actor positions/velocities per frame
  - Extracts player/car actor positions/velocities/boost per frame
  - Extracts goal events from replay header
  - Detects kickoff frames (ball at center position)
  - Tracks team assignments via TeamPaint and FlaggedByte attributes
  
- [ ] **Player names**: Currently showing as `Player_X` - need to properly link PlayerReplicationInfo names to car actors

### ✅ DONE: feature_extractor

- [x] **Implemented `extract_frame_features()`**: Full 147-feature extraction:
  - Ball state: position, velocity, speed
  - Player state: position, velocity, rotation (quaternion), speed magnitude, boost, demolished
  - Player geometry: distance to ball, distance to own goal, facing ball angle, goal-line position
  - Team context: team centroid position, average boost
  - Game context: ball-to-goal distances
  
- [x] **Normalization**: All features normalized to [-1, 1] or [0, 1] ranges during extraction
- [x] **FEATURES.md**: Comprehensive documentation of all features with indices

### ✅ DONE: ml_model

- [x] **ImpactModel**: 3-layer feedforward neural network (147 → 256 → 128 → 1)
- [x] **Training pipeline**:
  - `ImpactDataset` and `ImpactBatcher` for data loading
  - Adam optimizer with configurable learning rate
  - MSE loss function
  - Validation split (configurable, default 10%)
  - Early stopping (10 epochs patience)
  - Progress logging
- [x] **Inference**: `predict()` and `predict_batch()` functions
- [x] **Checkpointing**: `save_checkpoint()` and `load_checkpoint()` using Burn's MessagePack format
- [x] **Documentation**: MLModel.md with architecture and usage guide

### ✅ DONE: test_pipeline Command

- [x] **Metadata parsing**: Reads `metadata.jsonl` and extracts player ranks
- [x] **Rank-to-MMR mapping**: Converts rank strings to MMR values
- [x] **End-to-end test**: Parses replays, extracts features, trains model, runs inference
- [x] **Sanity checks**: Verifies loss is finite and predictions are in reasonable range

### 🔄 IN PROGRESS: Full Training

- [ ] Run small-scale test to verify pipeline works
- [ ] Scale up to full dataset with proper train/validation/test splits
- [ ] Evaluate model performance

### Future Enhancements

- [ ] BakkesMod integration for real-time scoring
- [ ] Support for 2v2 and 1v1 game modes (different feature counts)
- [ ] Advanced features: 50/50 outcomes, passing sequences, boost steal detection
- [ ] Per-player scoring (not just per-frame)
- [ ] Training dashboard with loss curves
- [ ] Model hyperparameter tuning

---

## Training Data

### Available Data

| Mode | Replays | Metadata |
|------|---------|----------|
| 3v3  | 36,078  | ✅ metadata.jsonl |
| 2v2  | 42,200  | ✅ metadata.jsonl |
| 1v1  | 41,318  | ✅ metadata.jsonl |

### Metadata Format

Each line in `metadata.jsonl` is a JSON object with:
- `id`: Replay UUID (matches filename without `.replay`)
- `data.blue.players[]`: Blue team players with `rank.id`, `rank.division`
- `data.orange.players[]`: Orange team players with `rank.id`, `rank.division`

### Rank-to-MMR Mapping

Player ranks are converted to approximate MMR values:

| Rank | Base MMR |
|------|----------|
| Supersonic Legend | 1883 |
| Grand Champion III | 1706 |
| Grand Champion II | 1575 |
| Grand Champion I | 1436 |
| Champion III | 1315 |
| Champion II | 1195 |
| Champion I | 1075 |
| Diamond III | 980 |
| Diamond II | 915 |
| Diamond I | 835 |
| Platinum III | 760 |
| ... | ... |
| Bronze I | 0 |

Each division adds ~20-40 MMR to the base value.

---

## Quick Start: Running the Pipeline Test

```bash
# Build the project
cargo build --release

# Run end-to-end test with 5 replays
./target/release/rocket_league_score test-pipeline \
  --replay-dir /workspace/replays/3v3 \
  --metadata /workspace/replays/3v3/metadata.jsonl \
  --num-replays 5
```

Expected output:
1. Loads metadata and matches to replay files
2. Parses replays and extracts features
3. Trains model for 5 epochs
4. Runs inference on samples
5. Reports sanity check results
