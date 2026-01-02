//! CLI command implementations.

mod device;
pub mod full_pipeline;
pub mod predict;
pub mod train;

pub use device::init_device;
