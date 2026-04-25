use burn::backend::wgpu::WgpuDevice;
use tracing::info;

/// Initializes a Wgpu device for GPU acceleration.
///
/// Works on Windows (DX12/Vulkan), Linux (Vulkan), and macOS (Metal) without
/// any extra runtime installation. Swap `WgpuDevice::default()` for
/// `WgpuDevice::DiscreteGpu(0)` if you need to pin a specific adapter.
pub fn init_device() -> WgpuDevice {
    info!("Initializing Wgpu device...");
    WgpuDevice::default()
}
