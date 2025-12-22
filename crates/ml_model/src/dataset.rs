//! Dataset and batching for Burn training.

use burn::prelude::*;
use feature_extractor::{FEATURE_COUNT, TrainingSample};

/// A single item in the impact dataset.
#[derive(Debug, Clone)]
pub struct ImpactDatasetItem {
    /// Feature vector for this frame.
    pub features: [f32; FEATURE_COUNT],
    /// Target MMR value.
    pub target: f32,
}

impl From<&TrainingSample> for ImpactDatasetItem {
    fn from(sample: &TrainingSample) -> Self {
        Self {
            features: sample.features.features,
            target: sample.target_mmr,
        }
    }
}

impl From<TrainingSample> for ImpactDatasetItem {
    fn from(sample: TrainingSample) -> Self {
        Self {
            features: sample.features.features,
            target: sample.target_mmr,
        }
    }
}

/// Dataset for impact score training.
#[derive(Debug, Clone)]
pub struct ImpactDataset {
    items: Vec<ImpactDatasetItem>,
}

impl ImpactDataset {
    /// Creates a new dataset from training samples.
    #[must_use]
    pub fn new(samples: Vec<TrainingSample>) -> Self {
        let items = samples.into_iter().map(ImpactDatasetItem::from).collect();
        Self { items }
    }

    /// Creates a dataset from a slice of training samples.
    #[must_use]
    pub fn from_slice(samples: &[TrainingSample]) -> Self {
        let items = samples.iter().map(ImpactDatasetItem::from).collect();
        Self { items }
    }
}

impl burn::data::dataset::Dataset<ImpactDatasetItem> for ImpactDataset {
    fn get(&self, index: usize) -> Option<ImpactDatasetItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A batch of training data.
#[derive(Debug, Clone)]
pub struct ImpactBatch<B: Backend> {
    /// Input features tensor of shape `[batch_size, FEATURE_COUNT]`.
    pub inputs: Tensor<B, 2>,
    /// Target MMR values tensor of shape `[batch_size, 1]`.
    pub targets: Tensor<B, 2>,
}

/// Batcher for creating training batches.
#[derive(Debug, Clone)]
pub struct ImpactBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImpactBatcher<B> {
    /// Creates a new batcher for the given device.
    #[must_use]
    pub const fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Creates a batch from a vector of items.
    pub fn batch(&self, items: Vec<ImpactDatasetItem>) -> ImpactBatch<B> {
        let batch_size = items.len();

        // Collect all features into a flat vector
        let mut features_data = Vec::with_capacity(batch_size * FEATURE_COUNT);
        let mut targets_data = Vec::with_capacity(batch_size);

        for item in items {
            features_data.extend_from_slice(&item.features);
            targets_data.push(item.target);
        }

        // Create tensors
        let inputs = Tensor::<B, 1>::from_floats(features_data.as_slice(), &self.device)
            .reshape([batch_size, FEATURE_COUNT]);

        let targets = Tensor::<B, 1>::from_floats(targets_data.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        ImpactBatch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use feature_extractor::FrameFeatures;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn test_dataset_creation() {
        use burn::data::dataset::Dataset;

        let samples = vec![
            TrainingSample {
                features: FrameFeatures::default(),
                player_ratings: vec![],
                target_mmr: 1000.0,
            },
            TrainingSample {
                features: FrameFeatures::default(),
                player_ratings: vec![],
                target_mmr: 1500.0,
            },
        ];

        let dataset = ImpactDataset::new(samples);
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert!(dataset.get(0).is_some());
        assert!(dataset.get(2).is_none());
    }

    #[test]
    fn test_batcher() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let batcher = ImpactBatcher::<TestBackend>::new(device);

        let items = vec![
            ImpactDatasetItem {
                features: [0.0; FEATURE_COUNT],
                target: 1000.0,
            },
            ImpactDatasetItem {
                features: [1.0; FEATURE_COUNT],
                target: 1500.0,
            },
        ];

        let batch = batcher.batch(items);

        assert_eq!(batch.inputs.dims(), [2, FEATURE_COUNT]);
        assert_eq!(batch.targets.dims(), [2, 1]);
    }

    #[test]
    fn test_dataset_item_conversion() {
        let sample = TrainingSample {
            features: FrameFeatures::default(),
            player_ratings: vec![],
            target_mmr: 1234.5,
        };

        let item: ImpactDatasetItem = (&sample).into();
        assert!((item.target - 1234.5).abs() < f32::EPSILON);

        let item2: ImpactDatasetItem = sample.into();
        assert!((item2.target - 1234.5).abs() < f32::EPSILON);
    }
}
