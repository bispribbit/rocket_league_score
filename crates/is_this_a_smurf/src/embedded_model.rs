//! Embedded checkpoint and training config for WASM inference.

/// Model weights in burnpack format, or empty when none is embedded.
///
/// **Currently empty, so the app reports a missing model instead of predicting.**
/// The last shipped weights, `data/v20.mpk`, are a pre-0.22 MessagePack
/// checkpoint; burn 0.22 dropped every reader for that format (and for the old
/// bin format) in favour of burnpack, and the weights were deliberately not
/// converted. There is no readable checkpoint to embed until a burn-0.22
/// training run writes one.
///
/// To ship a model again: train on 0.22, copy the resulting `.bpk` into `data/`,
/// and restore the include:
///
/// ```ignore
/// #[expect(clippy::large_include_file)]
/// pub(crate) static MODEL_BYTES: &[u8] = include_bytes!("../../../data/v20.bpk");
/// ```
///
/// This is a plain constant rather than a cargo feature on purpose: a feature
/// would be switched on by `cargo clippy --all-features`, the workspace's
/// standard compile check, and fail it until the file exists.
pub(crate) static MODEL_BYTES: &[u8] = &[];

/// Model training config JSON (for architecture dimensions).
pub(crate) static MODEL_CONFIG: &str = include_str!("../../../data/v20.config.json");

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
