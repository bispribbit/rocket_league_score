//! Checkpoint save/load for the sequence model (disk and binary formats).

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;
use serde_json::json;

use ml_model::{MMR_SCALE, ModelConfig, SequenceModel, TrainingConfig};

/// Validation metrics merged into the checkpoint `.config.json`.
///
/// Records Huber loss on normalized targets and an approximate RMSE in MMR units
/// (`sqrt(loss) * MMR_SCALE`; exact only in the quadratic Huber regime).
#[derive(Debug, Clone)]
pub struct CheckpointValidationMetrics {
    /// Huber loss on normalized `[0, 1]` targets at save time.
    pub validation_loss: f64,
    /// Approximate root mean squared error in MMR units for the same loss.
    pub validation_approx_rmse_mmr: f64,
}

impl CheckpointValidationMetrics {
    /// Builds metrics from the validation loss value used during training.
    #[must_use]
    pub fn from_validation_loss(validation_loss: f32) -> Self {
        Self {
            validation_loss: f64::from(validation_loss),
            validation_approx_rmse_mmr: f64::from(validation_loss.sqrt() * MMR_SCALE),
        }
    }
}

fn merge_validation_metrics_into_config_json(
    config_path: &str,
    metrics: &CheckpointValidationMetrics,
) -> anyhow::Result<()> {
    let json_string = std::fs::read_to_string(config_path)
        .map_err(|error| anyhow::anyhow!("Failed to read config JSON: {error}"))?;
    let mut value: serde_json::Value = serde_json::from_str(&json_string)
        .map_err(|error| anyhow::anyhow!("Failed to parse config JSON: {error}"))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Config JSON root must be an object"))?;
    object.insert(
        "validation_loss".to_string(),
        json!(metrics.validation_loss),
    );
    object.insert(
        "validation_approx_rmse_mmr".to_string(),
        json!(metrics.validation_approx_rmse_mmr),
    );
    let merged = serde_json::to_string_pretty(&value)
        .map_err(|error| anyhow::anyhow!("Failed to serialize merged config JSON: {error}"))?;
    std::fs::write(config_path, merged)
        .map_err(|error| anyhow::anyhow!("Failed to write config JSON: {error}"))?;
    Ok(())
}

/// Reference to a saved model checkpoint.
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// Path to the saved model.
    pub path: String,
    /// Model version.
    pub version: u32,
    /// Training configuration used.
    pub training_config: TrainingConfig,
}

/// Saves the model checkpoint to disk in NamedMpk format.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be saved.
pub fn save_checkpoint<B: Backend>(
    model: &SequenceModel<B>,
    path: &str,
    config: &TrainingConfig,
    validation_metrics: Option<CheckpointValidationMetrics>,
) -> anyhow::Result<ModelCheckpoint> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {e}"))?;

    let config_path = format!("{path}.config.json");
    config
        .save(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to save config: {e}"))?;
    if let Some(metrics) = validation_metrics.as_ref() {
        merge_validation_metrics_into_config_json(&config_path, metrics)?;
    }

    Ok(ModelCheckpoint {
        path: path.to_string(),
        version: 1,
        training_config: config.clone(),
    })
}

/// Saves the model checkpoint to disk in binary format (for web embedding).
///
/// The binary format can be loaded with `ml_model::load_checkpoint_from_bytes` using `include_bytes!`.
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be saved.
pub fn save_checkpoint_bin<B: Backend>(model: &SequenceModel<B>, path: &str) -> anyhow::Result<()> {
    use burn::record::BinFileRecorder;

    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model as bin: {e}"))?;

    Ok(())
}

/// Loads a model checkpoint from disk.
///
/// # Arguments
///
/// * `path` - Path to the model checkpoint file (without extension)
/// * `device` - The device to load the model onto
///
/// # Errors
///
/// Returns an error if the checkpoint cannot be loaded.
pub fn load_checkpoint<B: Backend>(
    path: &str,
    device: &B::Device,
) -> anyhow::Result<SequenceModel<B>> {
    let config_path = format!("{path}.config.json");
    let model_config = if Path::new(&config_path).exists() {
        let training_config = TrainingConfig::load(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to load config: {e}"))?;
        training_config.model
    } else {
        ModelConfig::new()
    };

    let model = SequenceModel::new(device, &model_config);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(path, &recorder, device)
        .map_err(|e| anyhow::anyhow!("Failed to load model weights: {e}"))?;

    Ok(model)
}
