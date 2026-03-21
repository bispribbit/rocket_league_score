#![allow(clippy::same_name_method)]
#![allow(clippy::redundant_pub_crate)]
//! Dioxus web application for Rocket League replay MMR prediction.
//!
//! Upload a `.replay` file and get per-player MMR predictions displayed
//! in per-segment and summary tables.

mod app_state;
mod browser_async;
mod embedded_model;
mod prediction;
mod rank_icon;
mod ui;

use ui::App;

fn main() {
    dioxus::logger::init(tracing::Level::DEBUG).expect("failed to init dioxus logger");
    dioxus::launch(App);
}
