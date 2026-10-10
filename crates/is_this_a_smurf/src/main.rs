#![allow(clippy::same_name_method)]
#![allow(clippy::redundant_pub_crate)]
//! Dioxus web application: tongue-in-cheek smurf detection from Rocket League replays.
//!
//! Upload a `.replay` file and get per-player rank estimates for the whole match, a one-minute
//! form timeline, and a roast about what the next rank up does better.

mod app_state;
mod branding;
mod browser_async;
mod embedded_model;
mod prediction;
mod rank_icon;
mod ui;
mod verdict_copy;

use ui::App;

fn main() {
    dioxus::logger::init(tracing::Level::DEBUG).expect("failed to init dioxus logger");
    dioxus::launch(App);
}
