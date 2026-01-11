//! Dataset and batching for sequence-based training.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use burn::prelude::*;
use feature_extractor::{FEATURE_COUNT, TOTAL_PLAYERS};
use rayon::prelude::*;
use tracing::info;

use crate::segment_cache::MmapSegmentStore;

// =============================================================================
// Batching
// =============================================================================

/// Batch of sequence data ready for model input.
#[derive(Debug, Clone)]
pub struct SequenceBatch<B: Backend> {
    /// Input tensor of shape [`batch_size`, `seq_len`, `FEATURE_COUNT`].
    pub inputs: Tensor<B, 3>,
    /// Target tensor of shape [`batch_size`, `TOTAL_PLAYERS`].
    pub targets: Tensor<B, 2>,
}

/// Batcher for creating batches from segment datasets.
pub struct SequenceBatcher<B: Backend> {
    device: B::Device,
    sequence_length: usize,
}

impl<B: Backend> SequenceBatcher<B> {
    /// Creates a new batcher.
    pub const fn new(device: B::Device, sequence_length: usize) -> Self {
        Self {
            device,
            sequence_length,
        }
    }

    /// Batches segments from a dataset by indices.
    ///
    /// This loads segments on-demand using the provided indices.
    pub fn batch_from_indices(
        &self,
        dataset: &Arc<MmapSegmentStore>,
        indices: &[usize],
    ) -> Option<SequenceBatch<B>> {
        if indices.is_empty() {
            return None;
        }

        let batch_size = indices.len();

        // Build input tensor [batch_size, seq_len, FEATURE_COUNT]
        let mut input_data = Vec::with_capacity(batch_size * self.sequence_length * FEATURE_COUNT);
        let mut target_data = Vec::with_capacity(batch_size * TOTAL_PLAYERS);

        for &idx in indices {
            let (features, target_mmr) = dataset.get(idx)?;
            input_data.extend_from_slice(&features);
            target_data.extend_from_slice(&target_mmr);
        }

        let inputs = Tensor::<B, 1>::from_floats(input_data.as_slice(), &self.device).reshape([
            batch_size,
            self.sequence_length,
            FEATURE_COUNT,
        ]);

        let targets = Tensor::<B, 1>::from_floats(target_data.as_slice(), &self.device)
            .reshape([batch_size, TOTAL_PLAYERS]);

        Some(SequenceBatch { inputs, targets })
    }
}

// =============================================================================
// Prefetching
// =============================================================================

/// Pre-loaded batch data on CPU ready for GPU transfer.
/// Contains raw f32 vectors that can be quickly converted to tensors.
#[derive(Debug)]
pub struct PreloadedBatchData {
    /// Flattened input data: [batch_size * seq_len * FEATURE_COUNT]
    pub input_data: Vec<f32>,
    /// Flattened target data: [batch_size * TOTAL_PLAYERS]
    pub target_data: Vec<f32>,
    /// Batch size for reshaping
    pub batch_size: usize,
    /// Sequence length for reshaping
    pub sequence_length: usize,
}

impl PreloadedBatchData {
    /// Converts the preloaded data to GPU tensors.
    pub fn to_batch<B: Backend>(&self, device: &B::Device) -> SequenceBatch<B> {
        let inputs = Tensor::<B, 1>::from_floats(self.input_data.as_slice(), device).reshape([
            self.batch_size,
            self.sequence_length,
            FEATURE_COUNT,
        ]);

        let targets = Tensor::<B, 1>::from_floats(self.target_data.as_slice(), device)
            .reshape([self.batch_size, TOTAL_PLAYERS]);

        SequenceBatch { inputs, targets }
    }
}

/// Prefetches batches in a background thread to keep the GPU fed.
///
/// This struct manages a background thread that prepares batches ahead of time,
/// loading data from memory-mapped files while the GPU processes the current batch.
pub struct BatchPrefetcher {
    /// Channel receiver for preloaded batches
    receiver: Receiver<PreloadedBatchData>,
    /// Handle to the background thread
    thread_handle: Option<JoinHandle<()>>,
    /// Total number of batches expected
    total_batches: usize,
    /// Batches received so far
    batches_received: usize,
}

impl BatchPrefetcher {
    /// Creates a new prefetcher that loads batches in the background.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to load from (must be Send + Sync)
    /// * `indices` - Pre-shuffled indices for the epoch
    /// * `batch_size` - Number of samples per batch
    /// * `sequence_length` - Frames per sequence segment
    /// * `prefetch_count` - Number of batches to keep buffered (recommended: 2-4)
    #[must_use]
    pub fn new(
        dataset: Arc<MmapSegmentStore>,
        indices: Vec<usize>,
        batch_size: usize,
        sequence_length: usize,
        prefetch_count: usize,
    ) -> Self {
        let num_samples = indices.len();
        let total_batches = num_samples.div_ceil(batch_size);

        // Use a synchronous channel with bounded capacity for backpressure
        let (sender, receiver) = mpsc::sync_channel(prefetch_count);

        let thread_handle = thread::spawn(move || {
            Self::prefetch_worker(dataset, indices, batch_size, sequence_length, sender);
        });

        Self {
            receiver,
            thread_handle: Some(thread_handle),
            total_batches,
            batches_received: 0,
        }
    }

    /// Background worker that loads batches and sends them through the channel.
    ///
    /// Loads segments in parallel using rayon to maximize I/O throughput.
    fn prefetch_worker(
        dataset: Arc<MmapSegmentStore>,
        indices: Vec<usize>,
        batch_size: usize,
        sequence_length: usize,
        sender: SyncSender<PreloadedBatchData>,
    ) {
        let num_samples = indices.len();

        for batch_idx in 0..num_samples.div_ceil(batch_size) {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(num_samples);

            let Some(batch_indices) = indices.get(batch_start..batch_end) else {
                continue;
            };

            let actual_batch_size = batch_indices.len();

            // Load segments in parallel using rayon
            let segment_data: Vec<Option<(Vec<f32>, [f32; TOTAL_PLAYERS])>> = batch_indices
                .par_iter()
                .map(|&idx| dataset.get(idx))
                .collect();

            // Check if all segments loaded successfully
            let mut valid = true;
            let mut input_data =
                Vec::with_capacity(actual_batch_size * sequence_length * FEATURE_COUNT);
            let mut target_data = Vec::with_capacity(actual_batch_size * TOTAL_PLAYERS);

            for data in segment_data {
                if let Some((features, target_mmr)) = data {
                    input_data.extend_from_slice(&features);
                    target_data.extend_from_slice(&target_mmr);
                } else {
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            let batch_data = PreloadedBatchData {
                input_data,
                target_data,
                batch_size: actual_batch_size,
                sequence_length,
            };

            // Send to the training thread (blocks if buffer is full - this is intentional backpressure)
            if sender.send(batch_data).is_err() {
                // Receiver dropped, training loop ended early
                break;
            }
        }
    }

    /// Gets the next preloaded batch, blocking until available.
    ///
    /// Returns `None` when all batches have been consumed.
    pub fn next_batch(&mut self) -> Option<PreloadedBatchData> {
        if self.batches_received >= self.total_batches {
            return None;
        }

        match self.receiver.recv() {
            Ok(batch) => {
                info!("Batch {}/{}", self.batches_received + 1, self.total_batches);
                self.batches_received += 1;
                Some(batch)
            }
            Err(_) => None, // Channel closed
        }
    }

    /// Returns the total number of batches for this epoch.
    #[must_use]
    pub const fn total_batches(&self) -> usize {
        self.total_batches
    }

    /// Returns the number of batches received so far.
    #[must_use]
    pub const fn batches_received(&self) -> usize {
        self.batches_received
    }
}

impl Drop for BatchPrefetcher {
    fn drop(&mut self) {
        // Drop the receiver first to signal the worker to stop
        // (we can't actually drop self.receiver, but when BatchPrefetcher drops, it drops)

        // Wait for the worker thread to finish
        if let Some(handle) = self.thread_handle.take() {
            // The thread will exit when it sees the channel is closed
            let _: Result<(), _> = handle.join();
        }
    }
}
