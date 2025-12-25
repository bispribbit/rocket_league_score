//! Utility functions for resolving file paths and working with `object_store`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use bytes::Bytes;
use config::OBJECT_STORE;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectStorePath;

/// Resolves a relative file path to a full path by joining it with a base directory.
///
/// The relative path should use forward slashes as separators (for cross-platform compatibility).
/// The function normalizes the path and handles both relative and absolute paths.
///
/// # Arguments
///
/// * `base_path` - The base directory path
/// * `relative_path` - The relative path (may use forward slashes)
///
/// # Returns
///
/// The full resolved path.
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
/// use database::resolve_file_path;
///
/// let base = PathBuf::from("/workspace/ballchasing");
/// let relative = "replays/3v3/bronze1/abc123.replay";
/// let full = resolve_file_path(&base, relative);
/// assert_eq!(full, PathBuf::from("/workspace/ballchasing/replays/3v3/bronze1/abc123.replay"));
/// ```
#[must_use]
pub fn resolve_file_path(base_path: &Path, relative_path: &str) -> PathBuf {
    // Convert forward slashes to the platform's path separator
    let sep = std::path::MAIN_SEPARATOR;
    let normalized = relative_path.replace('/', &sep.to_string());
    base_path.join(normalized)
}

/// Reads a file from `object_store` and returns the data as bytes.
///
/// # Arguments
///
/// * `relative_path` - The relative path in `object_store` (using forward slashes)
///
/// # Returns
///
/// The file data as a byte vector.
///
/// # Errors
///
/// Returns an error if reading from `object_store` fails.
pub async fn read_from_object_store(relative_path: &str) -> Result<Bytes> {
    let object_path = ObjectStorePath::from(relative_path);

    OBJECT_STORE
        .get(&object_path)
        .await
        .context("Failed to read from object_store")?
        .bytes()
        .await
        .context("Failed to read bytes from object_store")
}
