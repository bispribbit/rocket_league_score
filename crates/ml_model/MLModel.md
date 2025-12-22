# ML Model Crate

This crate implements the neural network for predicting Rocket League player impact scores using the [Burn](https://burn.dev) deep learning framework.

## Overview

The `ml_model` crate provides:

1. **Neural Network Architecture** - A feedforward network for impact score prediction
2. **Training Pipeline** - Complete training loop with Adam optimizer and MSE loss
3. **Inference** - Single-frame and batch prediction functions
4. **Checkpointing** - Model saving and loading using Burn's record system

## Architecture

### Network Structure

```
Input: [batch_size, 147]  (frame features from feature_extractor)
         │
         ▼
┌─────────────────────────┐
│   Linear(147 → 256)     │  First hidden layer
│   ReLU activation       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Linear(256 → 128)     │  Second hidden layer
│   ReLU activation       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Linear(128 → 1)       │  Output layer
└───────────┬─────────────┘
            │
            ▼
Output: [batch_size, 1]   (predicted MMR/impact score)
```

### Key Types

| Type | Description |
|------|-------------|
| `ImpactModel<B>` | The neural network model, generic over Burn backend |
| `ModelConfig` | Architecture configuration (hidden layer sizes, dropout) |
| `TrainingConfig` | Training hyperparameters (learning rate, epochs, batch size) |
| `TrainingData` | Container for training samples |
| `TrainingOutput` | Results from training (final loss, epochs completed) |

## Usage

### Creating a Model

```rust
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;
use ml_model::{create_model, ModelConfig};

type Backend = Wgpu;

let device = WgpuDevice::default();
let config = ModelConfig::new();  // Uses defaults: 256 → 128 hidden layers
let model = create_model::<Backend>(&device, &config);
```

### Training

```rust
use ml_model::{train, TrainingConfig, TrainingData, ModelConfig};
use feature_extractor::TrainingSample;

// Prepare training data
let mut data = TrainingData::new();
data.add_samples(samples);  // Vec<TrainingSample> from feature_extractor

// Configure training
let config = TrainingConfig::new(ModelConfig::new())
    .with_epochs(100)
    .with_batch_size(64)
    .with_learning_rate(1e-4);

// Train (model is modified in-place)
let output = train(&mut model, &data, &config)?;
println!("Final loss: {}", output.final_train_loss);
```

### Inference

```rust
use ml_model::predict;
use feature_extractor::extract_frame_features;

// Single frame prediction
let features = extract_frame_features(&frame);
let score = predict(&model, &features, &device);
println!("Predicted MMR: {}", score);

// Batch prediction (more efficient for multiple frames)
use ml_model::predict_batch;
let features: Vec<_> = frames.iter().map(extract_frame_features).collect();
let scores = predict_batch(&model, &features, &device);
```

### Saving and Loading

```rust
use ml_model::{save_checkpoint, load_checkpoint};

// Save model
save_checkpoint(&model, "models/impact_v1", &training_config)?;

// Load model
let loaded_model = load_checkpoint::<Backend>("models/impact_v1", &device)?;
```

## Training Details

### Loss Function

The model uses **Mean Squared Error (MSE)** loss between predicted and target MMR values:

```
Loss = mean((predicted_mmr - target_mmr)²)
```

This is appropriate for regression tasks where we want to minimize the average squared difference between predictions and ground truth.

### Optimizer

**Adam optimizer** with configurable learning rate (default: 1e-4). Adam adapts learning rates per-parameter, which works well for this regression task.

### Data Handling

1. **Validation Split**: 10% of data held out for validation by default
2. **Shuffling**: Training data is shuffled each epoch using a deterministic seed
3. **Batching**: Samples are grouped into configurable batch sizes (default: 64)

### Early Stopping

Training automatically stops if validation loss doesn't improve for 10 consecutive epochs, preventing overfitting.

## Configuration

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size_1` | 256 | Neurons in first hidden layer |
| `hidden_size_2` | 128 | Neurons in second hidden layer |
| `dropout` | 0.1 | Dropout rate (not currently used) |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `epochs` | 100 | Maximum training epochs |
| `batch_size` | 64 | Samples per training batch |
| `validation_split` | 0.1 | Fraction held for validation |
| `model` | ModelConfig | Nested model architecture config |

## File Format

Models are saved using Burn's **MessagePack** format with full precision:

```
models/impact_v1.mpk          # Model weights (binary)
models/impact_v1.config.json  # Training configuration (JSON)
```

The config file allows the model architecture to be reconstructed when loading.

## Backend Support

The model is generic over Burn backends. Common choices:

| Backend | Use Case |
|---------|----------|
| `Wgpu` | GPU acceleration (cross-platform) |
| `NdArray` | CPU-only, good for testing |
| `Autodiff<B>` | Wraps any backend for training (automatic differentiation) |

Training requires an `AutodiffBackend` (e.g., `Autodiff<Wgpu>`), while inference works with any backend.

## Dependencies

- **burn** - Core deep learning framework
- **feature_extractor** - Provides `TrainingSample` and `FrameFeatures` types
- **anyhow** - Error handling

## Testing

Run the test suite:

```bash
cargo test -p ml_model
```

Tests cover:
- Model creation and forward pass
- Single and batch prediction
- Training loop execution
- Data splitting and shuffling

