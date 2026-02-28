//! Segment caching for efficient player-centric feature storage and retrieval.
//!
//! This module provides functionality to:
//! - Write player-centric feature segments to disk in binary format
//! - Check if cached segments exist for a replay
//! - Read cached segments from disk

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{cast_slice, try_cast_vec};
use feature_extractor::{PLAYER_CENTRIC_FEATURE_COUNT, PlayerCentricFrameFeatures, TOTAL_PLAYERS};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Information about a cached segment file.
#[derive(Debug, Clone)]
pub struct SegmentFileInfo {
    /// Path to the segment file.
    pub path: PathBuf,
    /// Starting frame index (inclusive).
    pub start_frame: usize,
    /// Ending frame index (exclusive).
    pub end_frame: usize,
    /// Replay ID this segment belongs to.
    pub replay_id: Uuid,
}

/// Metadata for a replay's segments.
#[derive(Debug, Clone)]
pub struct ReplaySegmentMetadata {
    /// Replay ID.
    pub replay_id: Uuid,
    /// Target MMR for each of the 6 players.
    pub target_mmr: [f32; TOTAL_PLAYERS],
    /// Number of segments this replay produces.
    pub segment_count: usize,
    /// Total number of frames in the replay.
    pub total_frames: usize,
}

/// Computes how many segments a replay with the given frame count will produce.
///
/// Each segment contains `segment_length` consecutive frames.
/// Short replays (fewer frames than segment_length) still produce 1 segment.
#[must_use]
pub const fn compute_segment_count(total_frames: usize, segment_length: usize) -> usize {
    if total_frames >= segment_length {
        total_frames / segment_length
    } else {
        // Short replays still produce 1 segment (will be padded)
        1
    }
}

/// Generates the directory path for a replay's segments.
///
/// Format: `{base_path}/segments/{rank_category}/{rank}/{replay_id}/`
#[must_use]
pub fn segment_directory(base_path: &Path, file_path: &str, replay_id: Uuid) -> PathBuf {
    // file_path is like "replays/ranked-standard/bronze-1/uuid.replay"
    // We want to extract "ranked-standard/bronze-1"
    let path = Path::new(file_path);
    let mut components: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // Remove the filename (uuid.replay)
    components.pop();
    // Remove "replays" prefix if present
    if components.first() == Some(&"replays") {
        components.remove(0);
    }

    let rank_path = components.join("/");

    base_path
        .join("segments")
        .join(rank_path)
        .join(replay_id.to_string())
}

/// Generates the filename for a segment.
///
/// Format: `{start_frame}-{end_frame}.features`
#[must_use]
pub fn segment_filename(start_frame: usize, end_frame: usize) -> String {
    format!("{start_frame}-{end_frame}.features")
}

/// Checks if all segments for a replay already exist on disk.
///
/// Returns `true` if all expected segment files exist, `false` otherwise.
#[must_use]
pub fn check_segments_exist(
    base_path: &Path,
    file_path: &str,
    replay_id: Uuid,
    total_frames: usize,
    segment_length: usize,
) -> bool {
    let segment_dir = segment_directory(base_path, file_path, replay_id);

    if !segment_dir.exists() {
        return false;
    }

    let segment_count = compute_segment_count(total_frames, segment_length);

    for segment_idx in 0..segment_count {
        let start_frame = segment_idx * segment_length;
        let end_frame = ((segment_idx + 1) * segment_length).min(total_frames);
        let filename = segment_filename(start_frame, end_frame);
        let segment_path = segment_dir.join(&filename);

        if !segment_path.exists() {
            return false;
        }

        // Verify file size is correct for player-centric features
        let expected_size =
            6 * segment_length * PLAYER_CENTRIC_FEATURE_COUNT * std::mem::size_of::<f32>();
        if let Ok(metadata) = fs::metadata(&segment_path) {
            if metadata.len() != expected_size as u64 {
                debug!(
                    path = %segment_path.display(),
                    expected = expected_size,
                    actual = metadata.len(),
                    "Segment file has wrong size"
                );
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

/// Writes a single player-centric segment to disk in binary format.
///
/// The file contains raw f32 values for all 6 players:
/// `[player_0_frame_0_features, player_0_frame_1_features, ..., player_5_frame_N_features]`
///
/// # Arguments
///
/// * `segment_path` - Path where segment file will be written
/// * `player_frames` - Player-centric features for all frames (each frame has 6 player views)
/// * `segment_length` - Number of frames per segment (for padding)
///
/// # Errors
///
/// Returns an error if file writing fails.
pub fn write_player_centric_segment_file(
    segment_path: &Path,
    player_frames: &[[PlayerCentricFrameFeatures; 6]],
    segment_length: usize,
) -> anyhow::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = segment_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(segment_path)?;

    // Write features for each player for each frame
    // Format: [player_0_all_frames, player_1_all_frames, ..., player_5_all_frames]
    for player_idx in 0..6 {
        let mut frames_written = 0;
        for frame_set in player_frames.iter().take(segment_length) {
            if let Some(player_features) = frame_set.get(player_idx) {
                let bytes: &[u8] = cast_slice(&player_features.features);
                file.write_all(bytes)?;
                frames_written += 1;
            }
        }

        // Pad with last frame if needed
        if frames_written < segment_length {
            let padding_frame = player_frames
                .last()
                .and_then(|f| f.get(player_idx))
                .map_or([0.0f32; PLAYER_CENTRIC_FEATURE_COUNT], |p| p.features);
            let padding_bytes: &[u8] = cast_slice(&padding_frame);

            for _ in frames_written..segment_length {
                file.write_all(padding_bytes)?;
            }
        }
    }

    file.flush()?;
    Ok(())
}

/// Writes all player-centric segments for a replay to disk.
///
/// # Arguments
///
/// * `base_path` - Base directory for segment storage
/// * `file_path` - Original replay file path (for directory structure)
/// * `replay_id` - Unique replay identifier
/// * `player_frames` - All player-centric frame features (each frame has 6 player views)
/// * `segment_length` - Number of frames per segment
///
/// # Returns
///
/// Vector of `SegmentFileInfo` for all written segments.
///
/// # Errors
///
/// Returns an error if any segment cannot be written.
pub fn write_all_player_centric_segments(
    base_path: &Path,
    file_path: &str,
    replay_id: Uuid,
    player_frames: &[[PlayerCentricFrameFeatures; 6]],
    segment_length: usize,
) -> anyhow::Result<Vec<SegmentFileInfo>> {
    let segment_dir = segment_directory(base_path, file_path, replay_id);
    let segment_count = compute_segment_count(player_frames.len(), segment_length);

    let mut segment_infos = Vec::with_capacity(segment_count);

    for segment_idx in 0..segment_count {
        let start_frame = segment_idx * segment_length;
        let end_frame = ((segment_idx + 1) * segment_length).min(player_frames.len());

        let filename = segment_filename(start_frame, start_frame + segment_length);
        let segment_path = segment_dir.join(&filename);

        // Get the frames for this segment
        let segment_frames = player_frames
            .get(start_frame..end_frame.min(player_frames.len()))
            .unwrap_or(&[]);

        write_player_centric_segment_file(&segment_path, segment_frames, segment_length)?;

        segment_infos.push(SegmentFileInfo {
            path: segment_path,
            start_frame,
            end_frame: start_frame + segment_length,
            replay_id,
        });
    }

    debug!(
        replay_id = %replay_id,
        segment_count,
        dir = %segment_dir.display(),
        "Wrote player-centric segments to disk"
    );

    Ok(segment_infos)
}

/// Lists all existing segment files for a replay.
///
/// # Returns
///
/// Vector of `SegmentFileInfo` for existing segments, or empty if directory doesn't exist.
pub fn list_segment_files(
    base_path: &Path,
    file_path: &str,
    replay_id: Uuid,
    segment_length: usize,
) -> Vec<SegmentFileInfo> {
    let segment_dir = segment_directory(base_path, file_path, replay_id);

    if !segment_dir.exists() {
        return Vec::new();
    }

    let mut segments = Vec::new();

    // List all .features files in the directory
    if let Ok(entries) = fs::read_dir(&segment_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "features") {
                // Parse filename to get frame range
                if let Some(file_name) = path.file_stem().and_then(|s| s.to_str())
                    && let Some((start, end)) = parse_frame_range(file_name)
                {
                    segments.push(SegmentFileInfo {
                        path,
                        start_frame: start,
                        end_frame: end,
                        replay_id,
                    });
                }
            }
        }
    }

    // Sort by start frame
    segments.sort_by_key(|s| s.start_frame);

    // Filter to only include segments of the expected length
    segments.retain(|s| s.end_frame - s.start_frame == segment_length);

    segments
}

/// Parses a frame range from a segment filename.
///
/// Format: "start-end" (e.g., "0-150" -> Some((0, 150)))
fn parse_frame_range(filename: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = filename.split('-').collect();
    if parts.len() == 2 {
        let start = parts[0].parse().ok()?;
        let end = parts[1].parse().ok()?;
        Some((start, end))
    } else {
        None
    }
}

// =============================================================================
// Segment Store
// =============================================================================

/// Entry in the segment store.
struct SegmentEntry {
    /// Path to the segment file.
    path: PathBuf,
    /// Target MMR for this segment.
    target_mmr: [f32; TOTAL_PLAYERS],
}

/// Converts bytes to f32 Vec, using zero-copy when possible.
fn bytes_to_f32_vec(bytes: Vec<u8>) -> Vec<f32> {
    // Try zero-copy conversion first
    match try_cast_vec::<u8, f32>(bytes) {
        Ok(floats) => floats,
        Err((_, bytes)) => {
            // Fall back to copying if alignment doesn't match
            let f32_count = bytes.len() / 4;
            let mut result = Vec::with_capacity(f32_count);
            for chunk in bytes.chunks_exact(4) {
                result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            result
        }
    }
}

/// Storage for segment metadata and lazy loading.
///
/// Segments are stored on disk and loaded on-demand when accessed.
/// This allows handling datasets larger than memory.
pub struct SegmentStore {
    /// Name of the dataset.
    name: String,

    /// Maps segment index to segment metadata.
    entries: Vec<SegmentEntry>,

    /// Number of frames per segment.
    length: usize,
    /// Expected size of each segment in bytes.
    size_bytes: usize,
    /// Preloaded segments in memory (index -> Arc<Vec<f32>>).
    /// Used for validation datasets that should stay in memory.
    preloaded_segments: Option<HashMap<usize, Arc<Vec<f32>>>>,
}

impl SegmentStore {
    /// Creates a new empty segment store for player-centric features.
    ///
    /// # Arguments
    ///
    /// * `segment_length` - Number of frames per segment
    #[must_use]
    pub const fn new(name: String, segment_length: usize) -> Self {
        // Player-centric segments: 6 players × seq_len × features
        let segment_size_bytes =
            6 * segment_length * PLAYER_CENTRIC_FEATURE_COUNT * std::mem::size_of::<f32>();
        Self {
            entries: Vec::new(),
            name,
            length: segment_length,
            size_bytes: segment_size_bytes,
            preloaded_segments: None,
        }
    }

    pub const fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Adds segments to the store from a list of segment files.
    ///
    /// Returns the number of segments added.
    pub fn add_segments(
        &mut self,
        segment_files: &[SegmentFileInfo],
        target_mmr: [f32; TOTAL_PLAYERS],
    ) -> usize {
        let mut added = 0;
        for segment_file in segment_files {
            // Verify file exists and has correct size
            if let Ok(metadata) = fs::metadata(&segment_file.path) {
                if metadata.len() == self.size_bytes as u64 {
                    self.entries.push(SegmentEntry {
                        path: segment_file.path.clone(),
                        target_mmr,
                    });
                    added += 1;
                } else {
                    debug!(
                        path = %segment_file.path.display(),
                        expected = self.size_bytes,
                        actual = metadata.len(),
                        "Skipping segment with wrong size"
                    );
                }
            }
        }
        added
    }

    /// Loads segment data from disk.
    ///
    /// # Arguments
    ///
    /// * `index` - Segment index
    ///
    /// # Returns
    ///
    /// The loaded segment data as Arc<Vec<f32>>.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment cannot be loaded.
    fn load_segment_data(&self, index: usize) -> anyhow::Result<Arc<Vec<f32>>> {
        // Check if segment is preloaded
        if let Some(preloaded) = &self.preloaded_segments
            && let Some(data) = preloaded.get(&index)
        {
            return Ok(data.clone());
        }

        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("Segment index out of bounds: {index}"))?;

        // Load the segment file using std::fs::read
        let bytes = fs::read(&entry.path)?;

        // Verify file size
        if bytes.len() != self.size_bytes {
            anyhow::bail!(
                "Segment file has wrong size: expected {}, got {}",
                self.size_bytes,
                bytes.len()
            );
        }

        // Convert bytes to f32 Vec, using zero-copy when possible
        // Falls back to copying if alignment doesn't match
        let result = bytes_to_f32_vec(bytes);

        Ok(Arc::new(result))
    }

    /// Preloads all segments into memory.
    ///
    /// This is useful for validation datasets that are accessed frequently
    /// and are small enough to fit in memory.
    ///
    /// # Errors
    ///
    /// Returns an error if any segment file cannot be loaded.
    pub fn preload_all_segments(&mut self) -> anyhow::Result<()> {
        info!(
            segment_count = self.entries.len(),
            store_name = %self.name,
            "Preloading all segments into memory"
        );

        let mut preloaded = HashMap::new();

        for (index, entry) in self.entries.iter().enumerate() {
            // Load the segment file using std::fs::read
            let bytes = fs::read(&entry.path)?;

            // Verify file size
            if bytes.len() != self.size_bytes {
                warn!(
                    path = %entry.path.display(),
                    expected = self.size_bytes,
                    actual = bytes.len(),
                    "Segment file has wrong size, skipping"
                );
                continue;
            }

            // Convert bytes to f32 vector, using zero-copy when possible
            // Falls back to copying if alignment doesn't match
            let data = bytes_to_f32_vec(bytes);

            preloaded.insert(index, Arc::new(data));
        }

        self.preloaded_segments = Some(preloaded);
        info!(
            preloaded_count = self
                .preloaded_segments
                .as_ref()
                .map_or(0, std::collections::HashMap::len),
            store_name = %self.name,
            "All segments preloaded"
        );

        Ok(())
    }

    /// Returns the number of segments in the store.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the store is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the segment length (frames per segment).
    #[must_use]
    pub const fn segment_length(&self) -> usize {
        self.length
    }

    /// Gets a segment's target MMR.
    ///
    /// # Arguments
    ///
    /// * `index` - Segment index (0 to len()-1)
    ///
    /// # Returns
    ///
    /// The target MMR array for this segment, or `None` if index is out of bounds.
    #[must_use]
    pub fn get_target_mmr(&self, index: usize) -> Option<[f32; TOTAL_PLAYERS]> {
        self.entries.get(index).map(|e| e.target_mmr)
    }

    /// Gets player-centric features and target MMR for a segment.
    ///
    /// This method loads the cached segments containing player-centric features
    /// in the format: [6_players, seq_len, PLAYER_CENTRIC_FEATURE_COUNT]
    ///
    /// # Arguments
    ///
    /// * `index` - Segment index (0 to len()-1)
    ///
    /// # Returns
    ///
    /// Returns (features, target_mmr) where features is flattened player-centric data,
    /// or None if the segment cannot be loaded.
    pub fn get_player_centric(
        &self,
        index: usize,
    ) -> Option<(Arc<Vec<f32>>, [f32; TOTAL_PLAYERS])> {
        let target_mmr = self.get_target_mmr(index)?;
        let features = self.load_segment_data(index).ok()?;

        // Verify the loaded data has the correct size for player-centric features
        let expected_size = 6 * self.length * PLAYER_CENTRIC_FEATURE_COUNT;
        if features.len() != expected_size {
            warn!(
                index,
                expected = expected_size,
                actual = features.len(),
                "Segment has wrong size for player-centric features - cache may need regeneration"
            );
            return None;
        }

        Some((features, target_mmr))
    }
}

// =============================================================================
// Builder for loading segments from replays
// =============================================================================

/// Builder for creating a `SegmentStore` from replay data.
///
/// This handles the two-phase loading process:
/// 1. Ensure segments are cached (parse replay and write if missing)
/// 2. Add segment metadata (segments are loaded on-demand)
pub struct SegmentStoreBuilder {
    /// Base path for segment storage.
    base_path: PathBuf,
    /// Number of frames per segment.
    segment_length: usize,
    /// Accumulated segment store.
    store: SegmentStore,
    /// Statistics.
    replays_loaded: usize,
    replays_cached: usize,
    segments_loaded: usize,
}

impl SegmentStoreBuilder {
    /// Creates a new builder.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path for segment storage
    /// * `segment_length` - Number of frames per segment
    #[must_use]
    pub const fn new(base_path: PathBuf, name: String, segment_length: usize) -> Self {
        Self {
            base_path,
            segment_length,
            store: SegmentStore::new(name, segment_length),
            replays_loaded: 0,
            replays_cached: 0,
            segments_loaded: 0,
        }
    }

    /// Ensures player-centric segments are cached for a replay.
    ///
    /// If segments don't exist, they are created from the provided frames.
    /// If segments already exist and `total_frames` matches, they are assumed valid.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Original replay file path (for directory structure)
    /// * `replay_id` - Unique replay identifier
    /// * `player_frames` - Optional player-centric frame features (needed if not cached)
    /// * `total_frames` - Optional frame count (for validation)
    ///
    /// # Errors
    ///
    /// Returns error if segments need to be written but frames are not provided.
    pub fn ensure_player_centric_segments_cached(
        &mut self,
        file_path: &str,
        replay_id: Uuid,
        player_frames: Option<&[[PlayerCentricFrameFeatures; 6]]>,
        total_frames: Option<usize>,
    ) -> anyhow::Result<()> {
        // Check if segments already exist (only if total_frames is provided)
        let segments_exist = if let Some(total) = total_frames {
            check_segments_exist(
                &self.base_path,
                file_path,
                replay_id,
                total,
                self.segment_length,
            )
        } else {
            // If total_frames is None, assume segments exist (already verified externally)
            true
        };

        if !segments_exist {
            // Need to write segments
            let Some(frames) = player_frames else {
                anyhow::bail!(
                    "Segments not cached and no player frames provided for replay {replay_id}. Base path: {}. File path: {file_path}",
                    self.base_path.display()
                );
            };

            write_all_player_centric_segments(
                &self.base_path,
                file_path,
                replay_id,
                frames,
                self.segment_length,
            )?;
            self.replays_cached += 1;
        }

        Ok(())
    }

    /// Adds a replay's segments to the store.
    ///
    /// Adds segment metadata to the store (segments are loaded on-demand).
    /// Assumes segments are already cached (call `ensure_player_centric_segments_cached` first).
    ///
    /// # Arguments
    ///
    /// * `file_path` - Original replay file path (for directory structure)
    /// * `replay_id` - Unique replay identifier
    /// * `target_mmr` - Target MMR for this replay
    pub fn add_replay(
        &mut self,
        file_path: &str,
        replay_id: Uuid,
        target_mmr: [f32; TOTAL_PLAYERS],
    ) {
        let segment_files =
            list_segment_files(&self.base_path, file_path, replay_id, self.segment_length);

        let added = self.store.add_segments(&segment_files, target_mmr);
        self.segments_loaded += added;
        self.replays_loaded += 1;
    }

    /// Finishes building and returns the segment store.
    #[must_use]
    pub fn build(self) -> SegmentStore {
        info!(
            replays_loaded = self.replays_loaded,
            replays_cached = self.replays_cached,
            segments_loaded = self.segments_loaded,
            "Segment store built"
        );
        self.store
    }

    /// Returns loading statistics.
    #[must_use]
    pub const fn stats(&self) -> (usize, usize, usize) {
        (
            self.replays_loaded,
            self.replays_cached,
            self.segments_loaded,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_segment_count() {
        assert_eq!(compute_segment_count(1000, 300), 3);
        assert_eq!(compute_segment_count(299, 300), 1);
        assert_eq!(compute_segment_count(300, 300), 1);
        assert_eq!(compute_segment_count(301, 300), 1);
        assert_eq!(compute_segment_count(600, 300), 2);
    }

    #[test]
    fn test_segment_filename() {
        assert_eq!(segment_filename(0, 150), "0-150.features");
        assert_eq!(segment_filename(150, 300), "150-300.features");
    }

    #[test]
    fn test_segment_directory() {
        let base = Path::new("/data");
        let file_path = "replays/ranked-standard/bronze-1/abc123.replay";
        let replay_id = Uuid::parse_str("12345678-1234-1234-1234-123456789abc").unwrap();

        let dir = segment_directory(base, file_path, replay_id);
        assert!(dir.to_string_lossy().contains("segments"));
        assert!(dir.to_string_lossy().contains("ranked-standard"));
        assert!(dir.to_string_lossy().contains("bronze-1"));
    }

    #[test]
    fn test_parse_frame_range() {
        assert_eq!(parse_frame_range("0-150"), Some((0, 150)));
        assert_eq!(parse_frame_range("150-300"), Some((150, 300)));
        assert_eq!(parse_frame_range("invalid"), None);
        assert_eq!(parse_frame_range("abc-def"), None);
    }
}
