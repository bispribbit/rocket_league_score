//! CLI command implementations.

mod device;
pub mod pipeline;
pub mod predict;
pub mod train;

pub use device::init_wgpu_device;
