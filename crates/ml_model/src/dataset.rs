//! Dataset and batching for sequence-based training.

use burn::data::dataset::Dataset;
use burn::prelude::*;
use feature_extractor::{FEATURE_COUNT, FrameFeatures, TOTAL_PLAYERS};

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

/// Metadata for a game, tracking how many segments it contains.
#[derive(Debug, Clone)]
struct GameMetadata {
    /// Index into the samples vector.
    sample_index: usize,
    /// Number of segments this game can produce.
    segment_count: usize,
    /// Starting segment index in the global index space.
    start_segment_index: usize,
}

/// Dataset that generates consecutive-frame segments on-the-fly.
///
/// Instead of pre-computing all segments (which would use ~65GB for 1.2M segments),
/// this dataset stores references to full games and extracts segments when accessed.
///
/// Each segment contains `segment_length` consecutive frames from a game.
/// A 5-minute game (~9000 frames) with segment_length=90 produces ~100 segments.
pub struct SegmentDataset {
    /// Reference to the original samples (full games).
    samples: Vec<SequenceSample>,
    /// Metadata for each game (segment counts and index ranges).
    game_metadata: Vec<GameMetadata>,
    /// Number of consecutive frames per segment.
    segment_length: usize,
    /// Total number of segments across all games.
    total_segments: usize,
}

impl SegmentDataset {
    /// Creates a segment dataset from game samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Full game samples (each with all frames).
    /// * `segment_length` - Number of consecutive frames per segment.
    pub fn new(samples: Vec<SequenceSample>, segment_length: usize) -> Self {
        let mut game_metadata = Vec::with_capacity(samples.len());
        let mut total_segments = 0;

        for (idx, sample) in samples.iter().enumerate() {
            // Calculate how many non-overlapping segments this game can produce
            let segment_count = if sample.frames.len() >= segment_length {
                sample.frames.len() / segment_length
            } else {
                // Games shorter than segment_length still produce 1 segment (padded)
                1
            };

            game_metadata.push(GameMetadata {
                sample_index: idx,
                segment_count,
                start_segment_index: total_segments,
            });

            total_segments += segment_count;
        }

        Self {
            samples,
            game_metadata,
            segment_length,
            total_segments,
        }
    }

    /// Returns the segment length.
    #[must_use]
    pub const fn segment_length(&self) -> usize {
        self.segment_length
    }

    /// Returns the number of games in the dataset.
    #[must_use]
    pub const fn game_count(&self) -> usize {
        self.samples.len()
    }

    /// Maps a global segment index to (game_index, local_segment_index).
    fn index_to_game_segment(&self, index: usize) -> Option<(usize, usize)> {
        // Binary search for the game containing this segment index
        let game_idx = self
            .game_metadata
            .binary_search_by(|meta| {
                if index < meta.start_segment_index {
                    std::cmp::Ordering::Greater
                } else if index >= meta.start_segment_index + meta.segment_count {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;

        let meta = self.game_metadata.get(game_idx)?;
        let local_segment_idx = index - meta.start_segment_index;

        Some((meta.sample_index, local_segment_idx))
    }

    /// Extracts a segment from a game.
    fn extract_segment(&self, sample: &SequenceSample, segment_idx: usize) -> SequenceDatasetItem {
        let start_frame = segment_idx * self.segment_length;

        // Extract consecutive frames
        let frames: Vec<&FrameFeatures> =
            if start_frame + self.segment_length <= sample.frames.len() {
                // Full segment available
                sample
                    .frames
                    .get(start_frame..start_frame + self.segment_length)
                    .unwrap_or(&[])
                    .iter()
                    .collect()
            } else {
                // Need to pad (short game or last partial segment)
                let available = sample
                    .frames
                    .get(start_frame.min(sample.frames.len())..)
                    .unwrap_or(&[]);
                let mut frames: Vec<&FrameFeatures> = available.iter().collect();

                // Pad with last frame if needed
                if let Some(last) = sample.frames.last() {
                    while frames.len() < self.segment_length {
                        frames.push(last);
                    }
                } else {
                    // Empty game - should not happen, but handle gracefully
                    return SequenceDatasetItem {
                        features: vec![0.0; self.segment_length * FEATURE_COUNT],
                        target_mmr: sample.target_mmr,
                        sequence_length: 0,
                    };
                }
                frames
            };

        // Flatten features
        let mut features = Vec::with_capacity(self.segment_length * FEATURE_COUNT);
        for frame in &frames {
            features.extend_from_slice(&frame.features);
        }

        SequenceDatasetItem {
            features,
            target_mmr: sample.target_mmr,
            sequence_length: frames.len().min(self.segment_length),
        }
    }
}

impl Dataset<SequenceDatasetItem> for SegmentDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        let (sample_idx, segment_idx) = self.index_to_game_segment(index)?;
        let sample = self.samples.get(sample_idx)?;
        Some(self.extract_segment(sample, segment_idx))
    }

    fn len(&self) -> usize {
        self.total_segments
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

#[cfg(test)]
#[expect(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_dataset_creation() {
        // Create a game with 300 frames
        let sample = SequenceSample {
            frames: (0..300)
                .map(|i| FrameFeatures {
                    features: [i as f32; FEATURE_COUNT],
                    time: i as f32 / 30.0,
                })
                .collect(),
            target_mmr: [1500.0; TOTAL_PLAYERS],
        };

        let dataset = SegmentDataset::new(vec![sample], 90);

        // 300 frames / 90 per segment = 3 segments
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.game_count(), 1);
    }

    #[test]
    fn test_segment_dataset_consecutive_frames() {
        // Create a game with frames numbered by their index
        let sample = SequenceSample {
            frames: (0..270)
                .map(|i| {
                    let mut features = [0.0; FEATURE_COUNT];
                    features[0] = i as f32; // Store frame index in first feature
                    FrameFeatures {
                        features,
                        time: i as f32 / 30.0,
                    }
                })
                .collect(),
            target_mmr: [1500.0; TOTAL_PLAYERS],
        };

        let dataset = SegmentDataset::new(vec![sample], 90);

        // Get segment 0 (frames 0-89)
        let seg0 = dataset.get(0).expect("Should have segment 0");
        assert_eq!(seg0.features[0], 0.0); // First frame of segment 0
        assert_eq!(seg0.features[FEATURE_COUNT], 1.0); // Second frame (index 1)

        // Get segment 1 (frames 90-179)
        let seg1 = dataset.get(1).expect("Should have segment 1");
        assert_eq!(seg1.features[0], 90.0); // First frame of segment 1

        // Get segment 2 (frames 180-269)
        let seg2 = dataset.get(2).expect("Should have segment 2");
        assert_eq!(seg2.features[0], 180.0); // First frame of segment 2
    }

    #[test]
    fn test_segment_dataset_multiple_games() {
        let samples = vec![
            SequenceSample {
                frames: (0..180).map(|_| FrameFeatures::default()).collect(), // 2 segments
                target_mmr: [1000.0; TOTAL_PLAYERS],
            },
            SequenceSample {
                frames: (0..270).map(|_| FrameFeatures::default()).collect(), // 3 segments
                target_mmr: [1500.0; TOTAL_PLAYERS],
            },
            SequenceSample {
                frames: (0..90).map(|_| FrameFeatures::default()).collect(), // 1 segment
                target_mmr: [2000.0; TOTAL_PLAYERS],
            },
        ];

        let dataset = SegmentDataset::new(samples, 90);

        // Total: 2 + 3 + 1 = 6 segments
        assert_eq!(dataset.len(), 6);

        // Check that segments map to correct games
        let seg0 = dataset.get(0).unwrap();
        assert_eq!(seg0.target_mmr[0], 1000.0); // Game 0

        let seg2 = dataset.get(2).unwrap();
        assert_eq!(seg2.target_mmr[0], 1500.0); // Game 1

        let seg5 = dataset.get(5).unwrap();
        assert_eq!(seg5.target_mmr[0], 2000.0); // Game 2
    }

    #[test]
    fn test_segment_dataset_short_game() {
        // Game shorter than segment length - should still work (padded)
        let sample = SequenceSample {
            frames: (0..50).map(|_| FrameFeatures::default()).collect(),
            target_mmr: [1500.0; TOTAL_PLAYERS],
        };

        let dataset = SegmentDataset::new(vec![sample], 90);

        // Short game still produces 1 segment
        assert_eq!(dataset.len(), 1);

        let seg = dataset.get(0).unwrap();
        assert_eq!(seg.features.len(), 90 * FEATURE_COUNT);
    }
}
