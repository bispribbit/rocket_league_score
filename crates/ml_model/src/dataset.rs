//! Dataset and batching for sequence-based training.

use burn::prelude::*;
use feature_extractor::{FEATURE_COUNT, TOTAL_PLAYERS};

use crate::segment_cache::MmapSegmentStore;

// =============================================================================
// MmapSegmentDataset - Zero-copy dataset backed by memory-mapped files
// =============================================================================

/// Dataset backed by memory-mapped segment files.
///
/// This provides zero-copy access to cached feature segments.
/// The underlying `MmapSegmentStore` owns the memory-mapped files.
pub struct MmapSegmentDataset {
    /// The backing segment store.
    store: MmapSegmentStore,
}

impl MmapSegmentDataset {
    /// Creates a new dataset from a segment store.
    #[must_use]
    pub const fn new(store: MmapSegmentStore) -> Self {
        Self { store }
    }

    /// Returns the segment store reference.
    #[must_use]
    pub const fn store(&self) -> &MmapSegmentStore {
        &self.store
    }

    /// Returns the segment length.
    #[must_use]
    pub const fn segment_length(&self) -> usize {
        self.store.segment_length()
    }

    /// Returns the number of segments.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns true if the dataset is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Gets features and target MMR for a segment by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<(&[f32], [f32; TOTAL_PLAYERS])> {
        self.store.get(index)
    }

    /// Consumes the dataset and returns the underlying store.
    #[must_use]
    pub fn into_store(self) -> MmapSegmentStore {
        self.store
    }
}

impl std::fmt::Debug for MmapSegmentDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapSegmentDataset")
            .field("segment_count", &self.store.len())
            .field("segment_length", &self.store.segment_length())
            .finish()
    }
}

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

    /// Batches segments from an MmapSegmentDataset by indices.
    ///
    /// This reads directly from memory-mapped storage using the provided indices.
    pub fn batch_from_indices(
        &self,
        dataset: &MmapSegmentDataset,
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
            let (features, target_mmr) = dataset.store.get(idx)?;
            input_data.extend_from_slice(features);
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
