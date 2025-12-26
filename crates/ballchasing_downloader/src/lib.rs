//! Ballchasing.com replay downloader library.
//!
//! Downloads Rocket League replays from ballchasing.com organized by rank.
//! Runs metadata fetching and file downloads in parallel.

#![expect(
    clippy::std_instead_of_alloc,
    reason = "alloc crate not available in std environment"
)]

pub mod api;
mod downloader;
mod players;

pub use api::client::BallchasingClient;
pub use config::Config;
pub use downloader::run;
