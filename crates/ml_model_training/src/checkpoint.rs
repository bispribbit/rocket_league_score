//! Checkpoint save/load for the sequence model (disk and binary formats).

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use ml_model::{MMR_SCALE, ModelConfig, SequenceModel, TrainingConfig};
use serde::Serialize;
use serde_json::json;

/// One rank bucket from validation: RMSE in MMR units and sample count.
#[derive(Debug, Clone, Serialize)]
pub struct ValidationRankRmseEntry {
    /// Rank label as in the API (for example `bronze-1`, `supersonic-legend`).
    pub rank: String,
    /// Root mean squared error in MMR for this rank bucket.
    pub rmse_mmr: f64,
    /// Number of known-rank target slots contributing to this bucket.
    pub sample_count: u64,
}

/// Validation metrics merged into the checkpoint `.config.json`.
///
/// Records Huber loss on normalized targets and an approximate RMSE in MMR units
/// (`sqrt(loss) * MMR_SCALE`; exact only in the quadratic Huber regime).
#[expect(
    clippy::struct_field_names,
    reason = "Field names match the validation_* keys written into checkpoint .config.json."
)]
#[derive(Debug, Clone)]
pub struct CheckpointValidationMetrics {
    /// Huber loss on normalized `[0, 1]` targets at save time.
    pub validation_loss: f64,
    /// Approximate root mean squared error in MMR units for the same loss.
    pub validation_approx_rmse_mmr: f64,
    /// Per-rank RMSE in MMR units (same ordering as training logs).
    pub validation_rmse_by_rank: Vec<ValidationRankRmseEntry>,
}

impl CheckpointValidationMetrics {
    /// Builds metrics from the validation loss value used during training.
    ///
    /// Per-rank breakdown is empty; use [`Self::from_validation_loss_with_rank_breakdown`]
    /// when rank-level RMSE is available.
    #[must_use]
    pub fn from_validation_loss(validation_loss: f32) -> Self {
        Self::from_validation_loss_with_rank_breakdown(validation_loss, Vec::new())
    }

    /// Builds metrics including per-rank validation RMSE (MMR units).
    #[must_use]
    pub fn from_validation_loss_with_rank_breakdown(
        validation_loss: f32,
        validation_rmse_by_rank: Vec<ValidationRankRmseEntry>,
    ) -> Self {
        Self {
            validation_loss: f64::from(validation_loss),
            validation_approx_rmse_mmr: f64::from(validation_loss.sqrt() * MMR_SCALE),
            validation_rmse_by_rank,
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
    object.insert(
        "validation_rmse_by_rank".to_string(),
        serde_json::to_value(&metrics.validation_rmse_by_rank).map_err(|error| {
            anyhow::anyhow!("Failed to serialize validation_rmse_by_rank: {error}")
        })?,
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
pub fn save_checkpoint<B: ml_model::fused_lstm::FusedLstmBackend>(
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
pub fn save_checkpoint_bin<B: ml_model::fused_lstm::FusedLstmBackend>(
    model: &SequenceModel<B>,
    path: &str,
) -> anyhow::Result<()> {
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
pub fn load_checkpoint<B: ml_model::fused_lstm::FusedLstmBackend>(
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
