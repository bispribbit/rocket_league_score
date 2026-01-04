use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::Context;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

/// Returns the base path for the object store.
#[must_use]
pub fn get_base_path() -> PathBuf {
    dotenvy::dotenv().ok();

    #[cfg(target_os = "linux")]
    let base_path_unwrap = PathBuf::from("/workspace/ballchasing");

    #[cfg(target_os = "windows")]
    let base_path_unwrap = PathBuf::from(r"C:\GitHub\rocket_league_score\ballchasing");

    std::env::var("REPLAY_BASE_PATH").map_or_else(|_| base_path_unwrap, PathBuf::from)
}

/// Global object store instance, lazily initialized.
pub static OBJECT_STORE: LazyLock<Arc<dyn ObjectStore>> = LazyLock::new(|| {
    let base_path = get_base_path();

    std::fs::create_dir_all(&base_path).expect("Failed to create object store directory");

    Arc::new(LocalFileSystem::new_with_prefix(&base_path).expect("Failed to create object store"))
});

pub static CONFIG: LazyLock<Config> =
    LazyLock::new(|| Config::from_env().expect("Failed to create config"));

/// Application configuration loaded from environment variables.
#[derive(Clone)]
pub struct Config {
    /// Database connection URL
    pub database_url: String,

    /// Ballchasing.com API key
    pub ballchasing_api_key: String,
}

impl Config {
    /// Loads configuration from environment variables.
    ///
    /// Required environment variables:
    /// - `DATABASE_URL`: `PostgreSQL` connection string
    /// - `BALLCHASING_API_KEY`: API key for ballchasing.com
    ///
    /// Optional environment variables:
    /// - `REPLAY_BASE_PATH`: Base directory for storing replay files (default: `/workspace/ballchasing`)
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing.
    fn from_env() -> anyhow::Result<Self> {
        // Load .env file
        dotenvy::dotenv().ok();

        let database_url =
            std::env::var("DATABASE_URL").context("DATABASE_URL environment variable not set")?;

        let ballchasing_api_key = std::env::var("BALLCHASING_API_KEY")
            .context("BALLCHASING_API_KEY environment variable not set")?;

        Ok(Self {
            database_url,
            ballchasing_api_key,
        })
    }
}
