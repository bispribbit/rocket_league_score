use burn::backend::cuda::CudaDevice;
use tracing::info;

/// Initializes a CUDA device for GPU acceleration.
///
/// CUDA is NVIDIA's GPU API that provides high-performance computing.
///
/// This function only exists to be able to change the device at a single location.
pub fn init_device() -> CudaDevice {
    info!("Initializing CUDA device...");
    CudaDevice::default()
}
