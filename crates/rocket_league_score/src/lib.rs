//! Rocket League Impact Score Calculator
//!
//! A machine learning-based tool for evaluating player performance
//! in Rocket League replays.

use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub mod commands;
