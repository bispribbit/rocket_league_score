//! Dataset and batching for sequence-based training.

use burn::data::dataset::Dataset;
use burn::prelude::*;
use feature_extractor::{FEATURE_COUNT, TOTAL_PLAYERS};

use crate::SequenceSample;

/// A single item in the sequence dataset.
#[derive(Debug, Clone)]
pub struct SequenceDatasetItem {
    /// Frame features as a flattened vector: [`seq_len` * `FEATURE_COUNT`].
    pub features: Vec<f32>,
    /// Target MMR for each player.
    pub target_mmr: [f32; TOTAL_PLAYERS],
    /// Actual sequence length (before padding).
    pub sequence_length: usize,
}

/// Dataset for sequence training samples.
pub struct SequenceDataset {
    items: Vec<SequenceDatasetItem>,
    target_sequence_length: usize,
}

impl SequenceDataset {
    /// Creates a dataset from sequence samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - The sequence samples to include.
    /// * `target_sequence_length` - The target sequence length for uniform batching.
    pub fn new(samples: &[SequenceSample], target_sequence_length: usize) -> Self {
        let items = samples
            .iter()
            .map(|sample| {
                let sampled_frames = sample_frames(&sample.frames, target_sequence_length);
                let mut features = Vec::with_capacity(target_sequence_length * FEATURE_COUNT);
                for frame in &sampled_frames {
                    features.extend_from_slice(&frame.features);
                }

                SequenceDatasetItem {
                    features,
                    target_mmr: sample.target_mmr,
                    sequence_length: sample.frames.len().min(target_sequence_length),
                }
            })
            .collect();

        Self {
            items,
            target_sequence_length,
        }
    }

    /// Returns the target sequence length.
    #[must_use]
    pub const fn target_sequence_length(&self) -> usize {
        self.target_sequence_length
    }
}

impl Dataset<SequenceDatasetItem> for SequenceDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Batch of sequence data ready for model input.
#[derive(Debug, Clone)]
pub struct SequenceBatch<B: Backend> {
    /// Input tensor of shape [`batch_size`, `seq_len`, `FEATURE_COUNT`].
    pub inputs: Tensor<B, 3>,
    /// Target tensor of shape [`batch_size`, `TOTAL_PLAYERS`].
    pub targets: Tensor<B, 2>,
}

/// Batcher for creating batches from sequence dataset items.
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

    /// Batches a collection of items into tensors.
    pub fn batch(&self, items: Vec<SequenceDatasetItem>) -> SequenceBatch<B> {
        let batch_size = items.len();

        // Build input tensor [batch_size, seq_len, FEATURE_COUNT]
        let mut input_data = Vec::with_capacity(batch_size * self.sequence_length * FEATURE_COUNT);
        for item in &items {
            input_data.extend_from_slice(&item.features);
        }

        let inputs = Tensor::<B, 1>::from_floats(input_data.as_slice(), &self.device).reshape([
            batch_size,
            self.sequence_length,
            FEATURE_COUNT,
        ]);

        // Build target tensor [batch_size, TOTAL_PLAYERS]
        let mut target_data = Vec::with_capacity(batch_size * TOTAL_PLAYERS);
        for item in &items {
            target_data.extend_from_slice(&item.target_mmr);
        }

        let targets = Tensor::<B, 1>::from_floats(target_data.as_slice(), &self.device)
            .reshape([batch_size, TOTAL_PLAYERS]);

        SequenceBatch { inputs, targets }
    }
}

/// Samples frames uniformly to match a target sequence length.
fn sample_frames(
    frames: &[feature_extractor::FrameFeatures],
    target_len: usize,
) -> Vec<feature_extractor::FrameFeatures> {
    if frames.is_empty() {
        // Return default frames if empty
        return vec![feature_extractor::FrameFeatures::default(); target_len];
    }

    if frames.len() <= target_len {
        // Pad with last frame if too short
        let mut result = frames.to_vec();
        if let Some(last) = frames.last() {
            while result.len() < target_len {
                result.push(last.clone());
            }
        }
        return result;
    }

    // Sample uniformly
    let step = frames.len() as f64 / target_len as f64;
    (0..target_len)
        .map(|i| {
            let idx = (i as f64 * step) as usize;
            frames[idx.min(frames.len() - 1)].clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use feature_extractor::FrameFeatures;

    use super::*;

    #[test]
    fn test_sequence_dataset_creation() {
        let samples = vec![
            SequenceSample {
                frames: (0..100).map(|_| FrameFeatures::default()).collect(),
                target_mmr: [1000.0; TOTAL_PLAYERS],
            },
            SequenceSample {
                frames: (0..200).map(|_| FrameFeatures::default()).collect(),
                target_mmr: [1500.0; TOTAL_PLAYERS],
            },
        ];

        let dataset = SequenceDataset::new(&samples, 50);
        assert_eq!(dataset.len(), 2);

        let item = dataset.get(0).expect("Should have item");
        assert_eq!(item.features.len(), 50 * FEATURE_COUNT);
    }

    #[test]
    fn test_sample_frames_exact() {
        let frames: Vec<FrameFeatures> = (0..100).map(|_| FrameFeatures::default()).collect();
        let sampled = sample_frames(&frames, 100);
        assert_eq!(sampled.len(), 100);
    }

    #[test]
    fn test_sample_frames_downsample() {
        let frames: Vec<FrameFeatures> = (0..1000).map(|_| FrameFeatures::default()).collect();
        let sampled = sample_frames(&frames, 100);
        assert_eq!(sampled.len(), 100);
    }

    #[test]
    fn test_sample_frames_pad() {
        let frames: Vec<FrameFeatures> = (0..30).map(|_| FrameFeatures::default()).collect();
        let sampled = sample_frames(&frames, 100);
        assert_eq!(sampled.len(), 100);
    }

    #[test]
    fn test_sample_frames_empty() {
        let frames: Vec<FrameFeatures> = vec![];
        let sampled = sample_frames(&frames, 100);
        assert_eq!(sampled.len(), 100);
    }
}
