//! The skill model bundle embedded in the app.
//!
//! Rebuilt by `cargo run --release -p skill_model_training --bin train`, which writes
//! `data/skill_model.bin` (see `docs/model.md`).

use skill_model::SkillModelBundle;

/// Encoded [`SkillModelBundle`].
#[expect(
    clippy::large_include_file,
    reason = "the model is meant to ship inside the app (about 1.4 MB)"
)]
static MODEL_BYTES: &[u8] = include_bytes!("../../../data/skill_model.bin");

/// Decodes the embedded bundle and checks it was trained on the stats this build computes.
///
/// # Errors
///
/// A human-readable message when the bundle is corrupt or out of date.
pub(crate) fn load_bundle() -> Result<SkillModelBundle, String> {
    let bundle = SkillModelBundle::from_bytes(MODEL_BYTES)
        .map_err(|error| format!("Model loading error: {error}"))?;
    if !bundle.match_model.matches_current_stats() || !bundle.window_model.matches_current_stats() {
        return Err(
            "The built-in model was trained on different stats than this version computes. \
             Retrain it with `skill_model_training`."
                .to_string(),
        );
    }
    Ok(bundle)
}
