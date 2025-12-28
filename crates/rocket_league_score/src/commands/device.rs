//! WGPU device initialization with helpful error messages.

use anyhow::Result;
use burn::backend::wgpu::WgpuDevice;
use tracing::info;

/// Initializes a WGPU device for GPU acceleration.
///
/// WGPU is a cross-platform GPU API that works on Windows, Linux, and macOS.
/// It uses Vulkan on Linux/Windows and Metal on macOS.
pub fn init_device() -> WgpuDevice {
    info!("Initializing device...");
    WgpuDevice::default()
}
