use burn::backend::wgpu::WgpuDevice;
use tracing::info;

/// Initializes a WGPU device for GPU acceleration.
///
/// WGPU is a cross-platform GPU API that works on Windows, Linux, and macOS.
/// It uses Vulkan on Linux/Windows and Metal on macOS.
///
/// This function only exists to be able to change the device at a single location.
pub fn init_device() -> WgpuDevice {
    info!("Initializing device...");
    WgpuDevice::default()
}
