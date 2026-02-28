//! Checkpoint save/load for the sequence model (disk and binary formats).

use std::path::Path;

use burn::config::Config;
use burn::module::Module;
use burn::prelude::*;
use burn::record::FullPrecisionSettings;
use burn::record::NamedMpkFileRecorder;

use ml_model::{ModelConfig, SequenceModel, TrainingConfig};

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
