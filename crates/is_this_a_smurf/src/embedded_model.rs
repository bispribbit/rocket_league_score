//! Embedded checkpoint and training config for WASM inference.

/// Model weights in binary format (saved via `save_checkpoint_bin`).
pub(crate) static MODEL_BYTES: &[u8] = include_bytes!("../../../data/v6.mpk");

/// Model training config JSON (for architecture dimensions).
pub(crate) static MODEL_CONFIG: &str = include_str!("../../../data/v6.config.json");

/// Default sequence length for inference (subsampled frames per segment; must match checkpoint).
pub(crate) const DEFAULT_SEQUENCE_LENGTH: usize = 150;
