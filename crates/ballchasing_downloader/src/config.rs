//! Configuration loading from environment variables.

use anyhow::{Context, Result};

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
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
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing.
    pub fn from_env() -> Result<Self> {
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
