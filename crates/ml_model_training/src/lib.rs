//! Training pipeline for the ML sequence model.
//!
//! This crate contains everything needed to train the model: dataset batching,
//! segment caching, training loop, and checkpoint save/load.

mod checkpoint;
mod dataset;
pub mod minibatch_loss;
pub mod segment_cache;
mod training;

pub use checkpoint::{
    CheckpointValidationMetrics, ModelCheckpoint, ValidationRankRmseEntry, load_checkpoint,
    save_checkpoint, save_checkpoint_bin,
};
pub use dataset::{BatchPrefetcher, PreloadedBatchData, SequenceBatch, SequenceBatcher};
pub use minibatch_loss::MseExtremeMmrRowBoost;
pub use training::{
    CheckpointConfig, TrainingOutput, TrainingState, compute_inverse_frequency_weights,
    lookup_rank_weights, pseudo_random_f32, train,
};
