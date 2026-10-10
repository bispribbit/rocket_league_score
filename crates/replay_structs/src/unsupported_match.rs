//! Rejection payload when a replay is not a supported soccar match.

/// Explains why a replay was rejected (only 1v1, 2v2 and 3v3 soccar, ranked or casual).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedReplayMatch {
    /// Short label for the detected mode, e.g. "ranked 2v2 (doubles)" or "Dropshot".
    pub detected_mode_label: String,
}

impl UnsupportedReplayMatch {
    /// Full sentence for UI copy (English, matches the rest of the app).
    #[must_use]
    pub fn user_message(&self) -> String {
        format!(
            "We currently do not support {} — only 1v1, 2v2 and 3v3 soccar matches (ranked or casual) are supported.",
            self.detected_mode_label
        )
    }
}
