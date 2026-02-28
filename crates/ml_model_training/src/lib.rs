//! Training pipeline for the ML sequence model.
//!
//! This crate contains everything needed to train the model: dataset batching,
//! segment caching, training loop, and checkpoint save/load.

mod checkpoint;
mod dataset;
pub mod segment_cache;
mod training;

pub use checkpoint::{ModelCheckpoint, load_checkpoint, save_checkpoint, save_checkpoint_bin};
pub use dataset::{BatchPrefetcher, PreloadedBatchData, SequenceBatch, SequenceBatcher};
pub use training::{CheckpointConfig, TrainingOutput, TrainingState, train};
