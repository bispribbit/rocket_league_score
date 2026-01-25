use burn::backend::wgpu::WgpuDevice;
use tracing::info;

/// Initializes a WGPU device for GPU acceleration.
///
/// WGPU is a cross-platform GPU API that provides high-performance computing.
///
/// This function only exists to be able to change the device at a single location.
pub fn init_device() -> WgpuDevice {
    info!("Initializing WGPU device...");
    WgpuDevice::default()
}
