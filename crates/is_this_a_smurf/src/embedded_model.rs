//! Embedded checkpoint and training config for WASM inference.

/// Model weights in binary format (saved via `save_checkpoint_bin`).
pub(crate) static MODEL_BYTES: &[u8] = include_bytes!("../../../data/v6.mpk");

/// Model training config JSON (for architecture dimensions).
pub(crate) static MODEL_CONFIG: &str = include_str!("../../../data/v6.config.json");

/// Default sequence length for inference (subsampled frames per segment; must match checkpoint).
pub(crate) const DEFAULT_SEQUENCE_LENGTH: usize = 150;

/// Sequence length from embedded [`MODEL_CONFIG`], or [`DEFAULT_SEQUENCE_LENGTH`] if missing.
pub(crate) fn sequence_length_from_embedded_config() -> usize {
    if MODEL_CONFIG.is_empty() {
        return DEFAULT_SEQUENCE_LENGTH;
    }
    serde_json::from_str::<serde_json::Value>(MODEL_CONFIG)
        .ok()
        .and_then(|config_value| config_value.get("sequence_length")?.as_u64())
        .map_or(DEFAULT_SEQUENCE_LENGTH, |length_value| {
            length_value as usize
        })
}
