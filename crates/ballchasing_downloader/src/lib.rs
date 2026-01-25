//! Ballchasing.com replay downloader library.
//!
//! Downloads Rocket League replays from ballchasing.com organized by rank.
//! Runs metadata fetching and file downloads in parallel.
//!
//! Also supports importing local replay bundles via the `bundle` module.

#![expect(
    clippy::std_instead_of_alloc,
    reason = "alloc crate not available in std environment"
)]

pub mod api;
pub mod bundle;
mod downloader;
mod players;

pub use api::client::BallchasingClient;
pub use bundle::import_bundle;
pub use config::Config;
pub use downloader::run;
