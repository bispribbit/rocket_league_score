# ML Model Crate

This crate implements the neural network for predicting Rocket League player skill (MMR) using the [Burn](https://burn.dev) deep learning framework with an LSTM-based sequence model.

## Overview

The `ml_model` crate provides:

1. **Player-Centric Neural Network Architecture** - LSTM sequence model for individual player skill prediction
2. **Training Pipeline** - Complete training loop with Adam optimizer and MSE loss
3. **Inference** - Player-centric batch prediction functions
4. **Segment Caching** - Efficient storage and retrieval of player-centric features
5. **Checkpointing** - Model saving and loading using Burn's record system

## Architecture

### Player-Centric Design

Unlike global feature models, this architecture gives each player their own feature sequence centered on their perspective. This enables the model to:

- Predict **different MMR values** for each player
- Learn **individual skill patterns** rather than team averages
- Detect **smurfs** and **carries** within a match

### Network Structure

```
Input: [batch_size × 6, seq_len, 95]  (player-centric features for each of 6 players)
         │
         ▼
┌─────────────────────────────────┐
│   LSTM Layer 1 (95 → 128)       │  First LSTM layer
│   (processes each player's      │
│    temporal sequence)           │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   LSTM Layer 2 (128 → 64)       │  Second LSTM layer
│   (deeper temporal patterns)    │
└─────────────┬───────────────────┘
              │
              ▼
        [Last hidden state]
              │
              ▼
┌─────────────────────────────────┐
│   Dropout (0.5)                 │
│   Linear (64 → 32) + ReLU       │  Feedforward layers
│   Dropout (0.5)                 │
│   Linear (32 → 1)               │  Output per player
└─────────────┬───────────────────┘
              │
              ▼
Output: [batch_size × 6, 1] → reshape to [batch_size, 6]
        (predicted MMR for each of 6 players)
```

### Feature Layout (95 per player)

| Feature Group | Count | Description |
|--------------|-------|-------------|
| Ball state | 7 | Position, velocity, speed |
| This player | 13 | Position, velocity, rotation, speed, boost, demolished |
| Ball relationship | 2 | Distance to ball, facing ball |
| Teammate 1 | 13 | Full state with boost |
| Teammate 2 | 13 | Full state with boost |
| Teammate relationships | 4 | Distances to ball and this player |
| Opponent 1 | 12 | Full state WITHOUT boost |
| Opponent 2 | 12 | Full state WITHOUT boost |
| Opponent 3 | 12 | Full state WITHOUT boost |
| Opponent relationships | 6 | Distances to ball and this player |
| Game context | 1 | Ball distance to blue goal |

**Note:** Opponent boost is excluded to prevent information leakage (you can't see opponent boost in-game).

### Key Types

| Type | Description |
|------|-------------|
| `SequenceModel<B>` | The LSTM neural network model, generic over Burn backend |
| `ModelConfig` | Architecture configuration (LSTM sizes, dropout) |
| `TrainingConfig` | Training hyperparameters (learning rate, epochs, batch size) |
| `PlayerCentricSequenceTrainingData` | Container for player-centric training samples |
| `TrainingOutput` | Results from training (final loss, epochs completed) |
| `SegmentStore` | Storage for cached player-centric features |

## Usage

### Creating a Model

```rust
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use ml_model::{create_model, ModelConfig};

type Backend = Wgpu;

let device = WgpuDevice::default();
let config = ModelConfig::new();  // Uses defaults: 128 → 64 LSTM hidden layers
let model = create_model::<Backend>(&device, &config);
```

### Inference

```rust
use ml_model::predict_player_centric;

// Predict MMR for all 6 players from raw game frames
let mmr_predictions = predict_player_centric(&model, &game_frames, &device, sequence_length);

// mmr_predictions is [f32; 6] with individual predictions per player
println!("Blue team: {:.0}, {:.0}, {:.0}", 
    mmr_predictions[0], mmr_predictions[1], mmr_predictions[2]);
println!("Orange team: {:.0}, {:.0}, {:.0}", 
    mmr_predictions[3], mmr_predictions[4], mmr_predictions[5]);
```

### Saving and Loading

```rust
use ml_model::{save_checkpoint, load_checkpoint};

// Save model
save_checkpoint(&model, "models/player_centric_v1", &training_config)?;

// Load model
let loaded_model = load_checkpoint::<Backend>("models/player_centric_v1", &device)?;
```

## Training Details

### Loss Function

**Mean Squared Error (MSE)** between predicted and target MMR values for all 6 players:

```
Loss = mean((predicted_mmr - target_mmr)²)
```

### Optimizer

**Adam optimizer** with configurable learning rate (default: 1e-4).

### Segment Caching

For efficient training, game replays are pre-processed into cached segments:

```
Format: [6_players × seq_len × 95_features] per segment
Size: 6 × 150 × 95 × 4 bytes = ~342KB per segment
```

This allows loading training data much faster than parsing replays each epoch.

## Configuration

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lstm_hidden_1` | 128 | Hidden size of first LSTM layer |
| `lstm_hidden_2` | 64 | Hidden size of second LSTM layer |
| `feedforward_hidden` | 32 | Size of feedforward layer |
| `dropout` | 0.5 | Dropout rate for regularization |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 2048 | Samples per training batch |
| `validation_split` | 0.1 | Fraction held for validation |
| `sequence_length` | 150 | Frames per segment (~5 seconds at 30fps) |

## File Format

Models are saved using Burn's **MessagePack** format:

```
models/player_centric_v1.mpk          # Model weights (binary)
models/player_centric_v1.config.json  # Training configuration (JSON)
```

Cached segments are stored as raw binary files:

```
segments/{rank}/{replay_id}/{start_frame}-{end_frame}.features
```

## Backend Support

| Backend | Use Case |
|---------|----------|
| `Wgpu` | GPU acceleration (cross-platform) |
| `NdArray` | CPU-only, good for testing |
| `Autodiff<B>` | Wraps any backend for training |

Training requires `Autodiff<Wgpu>`, while inference works with `Wgpu` directly.

## Dependencies

- **burn** - Core deep learning framework
- **feature_extractor** - Provides player-centric feature extraction
- **replay_structs** - Game frame data structures
- **anyhow** - Error handling
- **bytemuck** - Zero-copy byte casting

## Testing

Run the test suite:

```bash
cargo test -p ml_model
```

Tests cover:
- Model creation and forward pass
- Player-centric prediction
- Training data structures
- Segment frame padding
- Segment store operations
