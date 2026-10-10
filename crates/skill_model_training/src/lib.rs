//! Training for the shipped skill model ([`skill_model`]).
//!
//! Two binaries: `extract_stats` parses every downloaded replay into per-player stats CSVs
//! (whole match and 60-second windows), and `train` fits the gradient-boosted trees,
//! prints the evaluation, builds the coaching table and writes the bundle the app embeds.
//! See `docs/model.md`.

pub mod coaching_table;
pub mod dataset;
pub mod evaluation;
pub mod gradient_boosting;
