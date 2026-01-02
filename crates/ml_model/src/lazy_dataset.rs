//! Lazy-loading dataset for memory-efficient training.
//!
//! Instead of loading all games into memory upfront, this dataset stores only
//! lightweight metadata and loads games on-demand during training.

use std::sync::Arc;

use burn::data::dataset::Dataset;
use feature_extractor::{FEATURE_COUNT, FrameFeatures, TOTAL_PLAYERS};

use crate::dataset::SequenceDatasetItem;

/// Metadata for a single game (lightweight - no frame data).
#[derive(Debug, Clone)]
pub struct GameMetadata {
    /// Unique identifier for the game.
    pub game_id: String,
    /// Target MMR for each of the 6 players.
    pub target_mmr: [f32; TOTAL_PLAYERS],
    /// Estimated number of frames in the game (for segment counting).
    /// This is approximate - actual count determined when loaded.
    pub estimated_frame_count: usize,
}

/// Trait for loading game frame data on-demand.
///
/// Implement this trait to provide lazy loading from any data source
/// (object store, local files, database, etc.).
pub trait GameLoader: Send + Sync {
    /// Loads frame features for a game by its ID.
    ///
    /// Returns None if the game cannot be loaded.
    fn load_game(&self, game_id: &str) -> Option<Vec<FrameFeatures>>;
}

/// Index entry for a segment within the dataset.
#[derive(Debug, Clone)]
struct SegmentIndex {
    /// Index into the games vector.
    game_index: usize,
    /// Which segment within the game (0-based).
    segment_index: usize,
}

/// Lazy-loading dataset that loads games on-demand.
///
/// This dataset stores only lightweight metadata (~100 bytes per game instead of ~5MB).
/// Games are loaded, segmented, and discarded as needed during training.
///
/// Memory usage: O(num_games) for metadata vs O(num_games * frames_per_game) for eager loading.
pub struct LazySegmentDataset {
    /// Lightweight metadata for each game.
    games: Vec<GameMetadata>,
    /// Loader to fetch frame data on-demand.
    loader: Arc<dyn GameLoader>,
    /// Number of consecutive frames per segment.
    segment_length: usize,
    /// Pre-computed segment index for O(1) access.
    segment_index: Vec<SegmentIndex>,
}

impl LazySegmentDataset {
    /// Creates a new lazy dataset.
    ///
    /// # Arguments
    ///
    /// * `games` - Lightweight metadata for each game (no frame data).
    /// * `loader` - Implementation that can load frame data on-demand.
    /// * `segment_length` - Number of consecutive frames per segment.
    pub fn new(
        games: Vec<GameMetadata>,
        loader: Arc<dyn GameLoader>,
        segment_length: usize,
    ) -> Self {
        // Pre-compute segment index for fast lookup
        let mut segment_index = Vec::new();

        for (game_idx, game) in games.iter().enumerate() {
            let segment_count = if game.estimated_frame_count >= segment_length {
                game.estimated_frame_count / segment_length
            } else {
                1 // Short games still produce 1 segment
            };

            for seg_idx in 0..segment_count {
                segment_index.push(SegmentIndex {
                    game_index: game_idx,
                    segment_index: seg_idx,
                });
            }
        }

        Self {
            games,
            loader,
            segment_length,
            segment_index,
        }
    }

    /// Returns the number of games in the dataset.
    #[must_use]
    pub const fn game_count(&self) -> usize {
        self.games.len()
    }

    /// Returns the segment length.
    #[must_use]
    pub const fn segment_length(&self) -> usize {
        self.segment_length
    }

    /// Extracts a segment from loaded frames.
    fn extract_segment(
        &self,
        frames: &[FrameFeatures],
        segment_idx: usize,
        target_mmr: [f32; TOTAL_PLAYERS],
    ) -> SequenceDatasetItem {
        let start_frame = segment_idx * self.segment_length;

        // Get segment frames with padding if needed
        let mut features = Vec::with_capacity(self.segment_length * FEATURE_COUNT);

        // Get the last frame for padding (or default if empty)
        let default_frame = FrameFeatures::default();
        let padding_frame = frames.last().unwrap_or(&default_frame);

        for i in 0..self.segment_length {
            let frame_idx = start_frame + i;
            let frame = if frame_idx < frames.len() {
                frames.get(frame_idx).unwrap_or(&default_frame)
            } else {
                // Pad with last frame
                padding_frame
            };
            features.extend_from_slice(&frame.features);
        }

        SequenceDatasetItem {
            features,
            target_mmr,
            sequence_length: self
                .segment_length
                .min(frames.len().saturating_sub(start_frame)),
        }
    }
}

impl Dataset<SequenceDatasetItem> for LazySegmentDataset {
    fn get(&self, index: usize) -> Option<SequenceDatasetItem> {
        let seg_info = self.segment_index.get(index)?;
        let game = self.games.get(seg_info.game_index)?;

        // Load the game on-demand
        let frames = self.loader.load_game(&game.game_id)?;

        // Extract the requested segment
        Some(self.extract_segment(&frames, seg_info.segment_index, game.target_mmr))
    }

    fn len(&self) -> usize {
        self.segment_index.len()
    }
}

#[cfg(test)]
#[expect(clippy::float_cmp)]
mod tests {
    use super::*;

    /// Test loader that generates synthetic frames.
    struct TestLoader {
        frames_per_game: usize,
    }

    impl GameLoader for TestLoader {
        fn load_game(&self, _game_id: &str) -> Option<Vec<FrameFeatures>> {
            Some(
                (0..self.frames_per_game)
                    .map(|i| FrameFeatures {
                        features: [i as f32; FEATURE_COUNT],
                        time: i as f32 / 30.0,
                    })
                    .collect(),
            )
        }
    }

    #[test]
    fn test_lazy_dataset_creation() {
        let games = vec![
            GameMetadata {
                game_id: "game1".to_string(),
                target_mmr: [1000.0; TOTAL_PLAYERS],
                estimated_frame_count: 300,
            },
            GameMetadata {
                game_id: "game2".to_string(),
                target_mmr: [1500.0; TOTAL_PLAYERS],
                estimated_frame_count: 150,
            },
        ];

        let loader = Arc::new(TestLoader {
            frames_per_game: 300,
        });
        let dataset = LazySegmentDataset::new(games, loader, 30);

        // Game 1: 300/30 = 10 segments, Game 2: 150/30 = 5 segments
        assert_eq!(dataset.len(), 15);
        assert_eq!(dataset.game_count(), 2);
    }

    #[test]
    fn test_lazy_dataset_get() {
        let games = vec![GameMetadata {
            game_id: "game1".to_string(),
            target_mmr: [1234.0; TOTAL_PLAYERS],
            estimated_frame_count: 90,
        }];

        let loader = Arc::new(TestLoader {
            frames_per_game: 90,
        });
        let dataset = LazySegmentDataset::new(games, loader, 30);

        // 3 segments
        assert_eq!(dataset.len(), 3);

        // Get segment 0 (frames 0-29)
        let seg = dataset.get(0).unwrap();
        assert_eq!(seg.features.len(), 30 * FEATURE_COUNT);
        assert_eq!(seg.target_mmr[0], 1234.0);
        assert_eq!(seg.features[0], 0.0); // First frame, first feature

        // Get segment 1 (frames 30-59)
        let seg = dataset.get(1).unwrap();
        assert_eq!(seg.features[0], 30.0); // Frame 30, first feature

        // Get segment 2 (frames 60-89)
        let seg = dataset.get(2).unwrap();
        assert_eq!(seg.features[0], 60.0); // Frame 60, first feature
    }
}
