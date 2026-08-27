//! Dataset and batching for sequence-based training.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

use burn::prelude::*;
use feature_extractor::{
    PLAYER_CENTRIC_FEATURE_COUNT, SELF_PLAYER_FEATURE_COUNT, TOTAL_PLAYERS,
};
use rayon::prelude::*;
use tracing::info;

use crate::segment_cache::SegmentStore;

type LoadedPlayerCentricSegment = Option<(Arc<Vec<f32>>, [f32; TOTAL_PLAYERS])>;

// =============================================================================
// Feature views
// =============================================================================

/// Which part of each frame's 106-feature vector the model is allowed to read.
///
/// Every player's feature vector carries the full state of the other five cars, which is
/// what makes the lobby shortcut available: read the lobby, emit the same answer six times,
/// and collect 98.4 % of the objective (see `docs/smurf-detection-handoff.md`). Removing
/// that context is the go/no-go ablation in step 3 of that document.
///
/// **Masked, not sliced.** [`Self::SelfOnly`] zeroes the context features rather than
/// narrowing the input tensor. Zero input contributes nothing through the LSTM's input
/// weights, so it removes the same information a narrower tensor would — but it leaves the
/// architecture, the parameter count and the tensor shapes byte-identical to the full-view
/// run. The two runs then differ only in what the model can see, which is the only
/// difference the ablation is trying to measure; a narrower tensor would confound it with a
/// capacity change.
///
/// Note that the retained slice still includes the 7 ball-state features, which all six
/// slots share. Those can identify *which lobby* this is, but they are identical across the
/// six slots, so they cannot order players within one — and within-lobby ordering is the
/// quantity step 3 scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeatureView {
    /// All 106 features, the other five cars included. What production trains on.
    #[default]
    Full,
    /// The focal player's own [`SELF_PLAYER_FEATURE_COUNT`] features; the rest zeroed.
    SelfOnly,
}

impl FeatureView {
    /// Zeroes the context features of every frame in a flat, frame-major buffer.
    ///
    /// `input_data` is laid out `[.., seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`, so each
    /// `PLAYER_CENTRIC_FEATURE_COUNT`-sized chunk is one frame for one player.
    pub fn mask_in_place(self, input_data: &mut [f32]) {
        if self == Self::Full {
            return;
        }
        debug_assert_eq!(
            input_data.len() % PLAYER_CENTRIC_FEATURE_COUNT,
            0,
            "feature buffer is not a whole number of frames; masking would slip out of \
             alignment and zero the wrong columns"
        );
        for frame in input_data.chunks_exact_mut(PLAYER_CENTRIC_FEATURE_COUNT) {
            if let Some(context) = frame.get_mut(SELF_PLAYER_FEATURE_COUNT..) {
                context.fill(0.0);
            }
        }
    }

    /// Short name for logs and experiment rows.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "full-106",
            Self::SelfOnly => "self-only-27",
        }
    }
}

impl From<bool> for FeatureView {
    /// Maps `TrainingConfig::self_only_features` onto the view it selects.
    fn from(self_only: bool) -> Self {
        if self_only { Self::SelfOnly } else { Self::Full }
    }
}

// =============================================================================
// Batching
// =============================================================================

/// Batch of sequence data ready for model input.
#[derive(Debug, Clone)]
pub struct SequenceBatch<B: Backend> {
    /// Input tensor for player-centric model.
    /// Shape: `[batch_size * 6_players, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]`
    /// Already reshaped so all 6 players are processed in parallel.
    pub inputs: Tensor<B, 3>,
    /// Target tensor of shape `[batch_size, TOTAL_PLAYERS]`.
    pub targets: Tensor<B, 2>,
}

/// Batcher for creating batches from segment datasets.
pub struct SequenceBatcher<B: Backend> {
    device: B::Device,
    sequence_length: usize,
    feature_view: FeatureView,
}

impl<B: Backend> SequenceBatcher<B> {
    /// Creates a new batcher over the full 106-feature view.
    pub const fn new(device: B::Device, sequence_length: usize) -> Self {
        Self {
            device,
            sequence_length,
            feature_view: FeatureView::Full,
        }
    }

    /// Restricts this batcher to `feature_view`.
    ///
    /// Must match the view the checkpoint was trained under. Scoring a self-only model on
    /// the full view feeds it context it has never seen and the metrics become noise.
    #[must_use]
    pub const fn with_feature_view(mut self, feature_view: FeatureView) -> Self {
        self.feature_view = feature_view;
        self
    }

    /// Batches segments from a dataset by indices using player-centric features.
    ///
    /// This loads segments on-demand using the provided indices.
    /// Each segment contains features for all 6 players.
    pub fn batch_from_indices(
        &self,
        dataset: &Arc<SegmentStore>,
        indices: &[usize],
    ) -> Option<SequenceBatch<B>> {
        if indices.is_empty() {
            return None;
        }

        let batch_size = indices.len();

        // Build input tensor [batch_size * 6, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]
        let mut input_data = Vec::with_capacity(
            batch_size * 6 * self.sequence_length * PLAYER_CENTRIC_FEATURE_COUNT,
        );
        let mut target_data = Vec::with_capacity(batch_size * TOTAL_PLAYERS);

        for &idx in indices {
            let (player_features, target_mmr) = dataset.get_player_centric(idx)?;
            // player_features is already flattened as [6, seq_len, features]
            input_data.extend_from_slice(player_features.as_slice());
            target_data.extend_from_slice(&target_mmr);
        }

        self.feature_view.mask_in_place(&mut input_data);

        // Create input tensor: [batch * 6, seq_len, features]
        let inputs = Tensor::<B, 1>::from_floats(input_data.as_slice(), &self.device).reshape([
            batch_size * 6,
            self.sequence_length,
            PLAYER_CENTRIC_FEATURE_COUNT,
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
    /// Flattened input data: [batch_size * 6 * seq_len * PLAYER_CENTRIC_FEATURE_COUNT]
    pub input_data: Vec<f32>,
    /// Flattened target data: [batch_size * TOTAL_PLAYERS]
    pub target_data: Vec<f32>,
    /// Batch size for reshaping
    pub batch_size: usize,
    /// Sequence length for reshaping
    pub sequence_length: usize,
    /// Original segment indices (into the SegmentStore) for this batch.
    /// Used by the smurf-masking EMA to track per-segment loss.
    pub segment_indices: Vec<usize>,
}

impl PreloadedBatchData {
    /// Converts the preloaded data to GPU tensors.
    pub fn to_batch<B: Backend>(&self, device: &B::Device) -> SequenceBatch<B> {
        let inputs = Tensor::<B, 1>::from_floats(self.input_data.as_slice(), device).reshape([
            self.batch_size * 6,
            self.sequence_length,
            PLAYER_CENTRIC_FEATURE_COUNT,
        ]);

        let targets = Tensor::<B, 1>::from_floats(self.target_data.as_slice(), device)
            .reshape([self.batch_size, TOTAL_PLAYERS]);

        SequenceBatch { inputs, targets }
    }
}

/// Prefetches batches in a background thread to keep the GPU fed.
///
/// This struct manages a background thread that prepares batches ahead of time,
/// loading data from disk while the GPU processes the current batch.
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
    /// * `feature_view` - Which features the model may read; masking happens here, on the
    ///   background thread, so the GPU path is unchanged and pays nothing for it
    #[must_use]
    pub fn new(
        dataset: Arc<SegmentStore>,
        indices: Vec<usize>,
        batch_size: usize,
        sequence_length: usize,
        prefetch_count: usize,
        feature_view: FeatureView,
    ) -> Self {
        let num_samples = indices.len();
        let total_batches = num_samples.div_ceil(batch_size);

        // Use a synchronous channel with bounded capacity for backpressure
        let (sender, receiver) = mpsc::sync_channel(prefetch_count);

        let thread_handle = thread::spawn(move || {
            Self::prefetch_worker(
                dataset,
                indices,
                batch_size,
                sequence_length,
                sender,
                feature_view,
            );
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
        dataset: Arc<SegmentStore>,
        indices: Vec<usize>,
        batch_size: usize,
        sequence_length: usize,
        sender: SyncSender<PreloadedBatchData>,
        feature_view: FeatureView,
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
            let segment_data: Vec<LoadedPlayerCentricSegment> = batch_indices
                .par_iter()
                .map(|&idx| dataset.get_player_centric(idx))
                .collect();

            // Check if all segments loaded successfully
            let mut valid = true;
            let mut input_data = Vec::with_capacity(
                actual_batch_size * 6 * sequence_length * PLAYER_CENTRIC_FEATURE_COUNT,
            );
            let mut target_data = Vec::with_capacity(actual_batch_size * TOTAL_PLAYERS);

            for data in segment_data {
                if let Some((features, target_mmr)) = data {
                    input_data.extend_from_slice(features.as_slice());
                    target_data.extend_from_slice(&target_mmr);
                } else {
                    valid = false;
                    break;
                }
            }

            if !valid {
                continue;
            }

            feature_view.mask_in_place(&mut input_data);

            let batch_data = PreloadedBatchData {
                input_data,
                target_data,
                batch_size: actual_batch_size,
                sequence_length,
                segment_indices: batch_indices.to_vec(),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame per player, values chosen so every column is distinguishable by index.
    fn frames(count: usize) -> Vec<f32> {
        (0..count * PLAYER_CENTRIC_FEATURE_COUNT)
            .map(|index| (index % PLAYER_CENTRIC_FEATURE_COUNT) as f32 + 1.0)
            .collect()
    }

    /// The full view is the identity — production must be untouched by this machinery.
    #[test]
    fn full_view_changes_nothing() {
        let mut data = frames(4);
        let original = data.clone();
        FeatureView::Full.mask_in_place(&mut data);
        assert_eq!(data, original);
    }

    /// Self-only keeps exactly the first 27 columns of every frame and zeroes the rest.
    #[test]
    fn self_only_zeroes_context_in_every_frame() {
        let mut data = frames(5);
        let original = data.clone();
        FeatureView::SelfOnly.mask_in_place(&mut data);

        assert_eq!(data.len(), original.len());
        for (index, (masked, before)) in data.iter().zip(original.iter()).enumerate() {
            let column = index % PLAYER_CENTRIC_FEATURE_COUNT;
            if column < SELF_PLAYER_FEATURE_COUNT {
                assert_eq!(masked, before, "self feature at column {column} was altered");
            } else {
                assert_eq!(*masked, 0.0, "context feature at column {column} survived");
            }
        }
    }

    /// The mask must actually remove information, not merely rearrange it: two players who
    /// differ only in their context features become indistinguishable, which is the
    /// property the step-3 ablation depends on.
    #[test]
    fn self_only_makes_context_only_differences_vanish() {
        let mut first = frames(1);
        let mut second = frames(1);
        // Perturb one context column and nothing else.
        second[SELF_PLAYER_FEATURE_COUNT + 3] += 999.0;
        assert_ne!(first, second);

        FeatureView::SelfOnly.mask_in_place(&mut first);
        FeatureView::SelfOnly.mask_in_place(&mut second);
        assert_eq!(first, second);
    }

    /// A difference in a *self* column must survive, or the ablation would be removing the
    /// signal it is meant to isolate rather than the shortcut.
    #[test]
    fn self_only_preserves_self_differences() {
        let mut first = frames(1);
        let mut second = frames(1);
        second[SELF_PLAYER_FEATURE_COUNT - 1] += 999.0;

        FeatureView::SelfOnly.mask_in_place(&mut first);
        FeatureView::SelfOnly.mask_in_place(&mut second);
        assert_ne!(first, second);
    }

    /// `TrainingConfig::self_only_features` maps onto the view, and the default is the
    /// production one.
    #[test]
    fn view_from_config_flag() {
        assert_eq!(FeatureView::from(false), FeatureView::Full);
        assert_eq!(FeatureView::from(true), FeatureView::SelfOnly);
        assert_eq!(FeatureView::default(), FeatureView::Full);
    }
}
