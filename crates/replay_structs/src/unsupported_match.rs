//! Rejection payload when a replay is not ranked standard 3v3.

/// Explains why a replay was rejected for this product (only ranked 3v3 standard).
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
            "We currently do not support {} — only ranked 3v3 standard is supported at this time.",
            self.detected_mode_label
        )
    }
}
